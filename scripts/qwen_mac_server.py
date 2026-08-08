"""OpenAI-compatible Qwen2.5-Omni server for Apple Silicon.

This deliberately uses the complete Hugging Face Qwen implementation. The
mlx-lm-omni package currently implements audio input but not Qwen's vision
tower, while FrameGuard requires video, audio, and text in the same request.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import av
from fastapi import FastAPI, HTTPException, Request
from PIL import Image

MODEL_ID = os.environ.get("FRAMEGUARD_MODEL", "Qwen/Qwen2.5-Omni-3B")
MAX_MEDIA_BYTES = int(os.environ.get("FRAMEGUARD_MAX_MEDIA_BYTES", str(200 * 1024 * 1024)))
LOGGER = logging.getLogger("frameguard.qwen_mac")


class _UnusedAudioOutputWarning(logging.Filter):
    """Hide Qwen's talker warning because FrameGuard requests text only."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(
            "System prompt modified, audio output may not work as expected"
        )


logging.getLogger().addFilter(_UnusedAudioOutputWarning())


class QwenRuntime:
    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None
        self.process_mm_info: Any = None
        self.device = "cpu"
        self.lock = threading.Lock()

    def load(self) -> None:
        import torch
        from qwen_omni_utils import process_mm_info
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )
        from transformers.utils import logging as transformers_logging

        self.torch = torch
        self.process_mm_info = process_mm_info
        if torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float16
        else:
            dtype = torch.float32

        print(f"[Qwen] Loading {MODEL_ID} on {self.device}...", flush=True)
        previous_verbosity = transformers_logging.get_verbosity()
        try:
            # Qwen 2.5's published metadata contains two harmless fields that
            # Transformers 4.52 warns about while still loading correctly.
            # Keep the shard progress bar and exceptions, but omit that noise.
            transformers_logging.set_verbosity_error()
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            self.model.disable_talker()
            self.model.to(self.device)
            self.model.eval()
            self.processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
        finally:
            transformers_logging.set_verbosity(previous_verbosity)
        print(f"[Qwen] {MODEL_ID} is ready.", flush=True)

    def generate(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> str:
        if self.model is None:
            raise RuntimeError("Qwen is not loaded")

        with tempfile.TemporaryDirectory(prefix="frameguard-qwen-media-") as temp_dir:
            conversation = _materialize_messages(messages, Path(temp_dir))
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            audios, images, videos = self.process_mm_info(
                conversation,
                use_audio_in_video=False,
            )
            inputs = self.processor(
                text=prompt,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            inputs = inputs.to(self.device)
            floating_dtype = next(self.model.parameters()).dtype
            for key, value in inputs.items():
                if hasattr(value, "is_floating_point") and value.is_floating_point():
                    inputs[key] = value.to(floating_dtype)

            generation = {
                "max_new_tokens": min(2048, int(options.get("max_tokens", 1400))),
                "do_sample": float(options.get("temperature", 0.0)) > 0,
                "return_audio": False,
                "use_audio_in_video": False,
            }
            if generation["do_sample"]:
                generation["temperature"] = max(0.01, float(options["temperature"]))
                generation["top_p"] = float(options.get("top_p", 1.0))

            with self.lock, self.torch.inference_mode():
                output_ids = self.model.generate(**inputs, **generation)
            # Hugging Face decoder-only generation returns the prompt followed
            # by the completion. Decoding the whole tensor leaks the prompt's
            # example JSON into the API response and can create false findings.
            prompt_length = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, prompt_length:]
            return self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]


RUNTIME = QwenRuntime()


def _decode_data_uri(uri: str, target_dir: Path, media_type: str) -> str:
    if not uri.startswith("data:") or ";base64," not in uri:
        return uri
    header, encoded = uri.split(",", 1)
    mime = header[5:].split(";", 1)[0]
    extensions = {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "audio/wav": ".wav",
        "audio/mpeg": ".mp3",
        "image/jpeg": ".jpg",
        "image/png": ".png",
    }
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(400, f"Invalid base64 {media_type} payload") from exc
    if len(payload) > MAX_MEDIA_BYTES:
        raise HTTPException(413, f"{media_type.title()} payload is too large")
    path = target_dir / f"{uuid.uuid4().hex}{extensions.get(mime, '.bin')}"
    path.write_bytes(payload)
    return str(path)


def _read_video_frames(
    video_path: str,
    *,
    sample_fps: float = 2.0,
    maximum_frames: int = 16,
) -> list[Image.Image]:
    """Decode evenly sampled frames without torchvision's removed video API."""

    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            source_fps = float(stream.average_rate or 0.0)
            decoded = [frame.to_image().convert("RGB") for frame in container.decode(stream)]
    except (av.FFmpegError, IndexError, OSError) as exc:
        raise HTTPException(400, "Qwen could not open the uploaded video clip") from exc
    frame_count = len(decoded)
    if source_fps <= 0 or frame_count < 2:
        raise HTTPException(400, "The uploaded video clip has invalid timing metadata")

    wanted = max(
        2,
        min(maximum_frames, round(frame_count / source_fps * sample_fps)),
    )
    indices = [
        min(frame_count - 1, round(index * (frame_count - 1) / max(1, wanted - 1)))
        for index in range(wanted)
    ]
    frames: list[Image.Image] = [decoded[frame_index] for frame_index in indices]
    if len(frames) < 2:
        raise HTTPException(400, "Qwen could not decode enough frames from the video clip")
    return frames


def _materialize_messages(
    messages: list[dict[str, Any]],
    target_dir: Path,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            converted.append({"role": message.get("role", "user"), "content": content})
            continue
        items: list[dict[str, Any]] = []
        for item in content:
            kind = item.get("type")
            if kind == "video_url":
                value = item.get("video_url", {})
                uri = value.get("url", "") if isinstance(value, dict) else str(value)
                video_path = _decode_data_uri(uri, target_dir, "video")
                items.append(
                    {
                        "type": "video",
                        "video": _read_video_frames(video_path),
                        "sample_fps": 2.0,
                        "raw_fps": 2.0,
                    }
                )
            elif kind == "audio_url":
                value = item.get("audio_url", {})
                uri = value.get("url", "") if isinstance(value, dict) else str(value)
                items.append({"type": "audio", "audio": _decode_data_uri(uri, target_dir, "audio")})
            elif kind == "image_url":
                value = item.get("image_url", {})
                uri = value.get("url", "") if isinstance(value, dict) else str(value)
                items.append({"type": "image", "image": _decode_data_uri(uri, target_dir, "image")})
            elif kind == "text":
                items.append({"type": "text", "text": str(item.get("text", ""))})
        converted.append({"role": message.get("role", "user"), "content": items})
    return converted


@asynccontextmanager
async def lifespan(_: FastAPI):
    RUNTIME.load()
    yield


app = FastAPI(title="FrameGuard local Qwen server", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID, "device": RUNTIME.device}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict[str, Any]:
    payload = await request.json()
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "messages must be a non-empty list")
    started = time.perf_counter()
    try:
        content = RUNTIME.generate(messages, payload)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Local Qwen inference failed")
        raise HTTPException(500, f"Qwen inference failed: {exc}") from exc
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "frameguard": {"elapsed_seconds": round(time.perf_counter() - started, 3)},
    }
