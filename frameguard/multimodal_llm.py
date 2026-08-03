from __future__ import annotations

import base64
import json
import mimetypes
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from .observability import RunEventRecorder
from .parse_findings import parse_model_findings
from .prompts import DETECTION_PROMPT, SYSTEM_PROMPT
from .schemas import ModelFinding


@dataclass(frozen=True, slots=True)
class LLMResponse:
    findings: list[ModelFinding]
    raw_text: str
    metadata: dict[str, object] = field(default_factory=dict)


class ClipAnalyzer(Protocol):
    def analyze_clip(self, clip_path: str | Path, clip_duration_seconds: float) -> LLMResponse:
        """Return semantic findings for one video clip."""


def _data_uri(path: Path) -> str:
    explicit_types = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
    }
    mime = explicit_types.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]
    mime = mime or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_audio(clip_path: Path, audio_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(clip_path),
        "-map",
        "0:a:0?",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (
        result.returncode == 0
        and audio_path.exists()
        and audio_path.stat().st_size > 44
    )


def _safe_server_error(body: object) -> tuple[str, str | None, int]:
    if not isinstance(body, dict):
        return "unknown", None, 0
    error = body.get("error")
    if not isinstance(error, dict):
        return "unknown", None, 0
    error_type = str(error.get("type") or "unknown")
    code = error.get("code")
    message = str(error.get("message") or "")
    return error_type, None if code is None else str(code), len(message)


class QwenOmniClient:
    """Client for plain vLLM V1 serving the Qwen2.5-Omni Thinker."""

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_seconds: float = 300.0,
        recorder: RunEventRecorder | None = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.recorder = recorder

    def healthcheck(self) -> None:
        base = self.api_base.removesuffix("/v1")
        response = httpx.get(f"{base}/health", timeout=10.0)
        response.raise_for_status()

    def analyze_clip(self, clip_path: str | Path, clip_duration_seconds: float) -> LLMResponse:
        clip_path = Path(clip_path).resolve()
        if not clip_path.exists():
            raise FileNotFoundError(f"Video clip does not exist: {clip_path}")
        if clip_path.stat().st_size == 0:
            raise ValueError(f"Video clip is empty: {clip_path}")

        request_id = f"fgreq_{uuid.uuid4().hex[:12]}"
        started = time.perf_counter()

        with tempfile.TemporaryDirectory(prefix="frameguard-audio-") as temp_dir:
            audio_path = Path(temp_dir) / f"{clip_path.stem}.wav"
            audio_available = _extract_audio(clip_path, audio_path)

            content: list[dict[str, Any]] = [
                {
                    "type": "video_url",
                    "video_url": {"url": _data_uri(clip_path)},
                }
            ]
            if audio_available:
                content.append(
                    {
                        "type": "audio_url",
                        "audio_url": {"url": _data_uri(audio_path)},
                    }
                )

            prompt = (
                f"{DETECTION_PROMPT}\n\n"
                f"Current chunk duration: {clip_duration_seconds:.3f} seconds."
            )
            content.append({"type": "text", "text": prompt})

            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                    },
                    {"role": "user", "content": content},
                ],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 1400,
                "seed": 42,
                "repetition_penalty": 1.05,
            }

            headers = {"Content-Type": "application/json", "X-FrameGuard-Request-ID": request_id}
            if self.api_key and self.api_key != "EMPTY":
                headers["Authorization"] = f"Bearer {self.api_key}"

            endpoint = f"{self.api_base}/chat/completions"
            if self.recorder:
                self.recorder.info(
                    "model.request.started",
                    request_id=request_id,
                    clip=clip_path.name,
                    clip_duration_seconds=round(clip_duration_seconds, 3),
                    video_bytes=clip_path.stat().st_size,
                    audio_included=audio_available,
                    audio_bytes=audio_path.stat().st_size if audio_available else 0,
                    model=self.model,
                )

            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(endpoint, headers=headers, json=payload)
            except httpx.ConnectError as exc:
                if self.recorder:
                    self.recorder.error(
                        "model.request.connection_failed",
                        request_id=request_id,
                        exception_type=type(exc).__name__,
                    )
                raise RuntimeError(
                    f"Could not connect to the Qwen server for request {request_id}."
                ) from exc
            except httpx.TimeoutException as exc:
                if self.recorder:
                    self.recorder.error(
                        "model.request.timed_out",
                        request_id=request_id,
                        timeout_seconds=self.timeout_seconds,
                    )
                raise RuntimeError(
                    f"Qwen request {request_id} timed out after {self.timeout_seconds} seconds."
                ) from exc

            elapsed = time.perf_counter() - started

            try:
                body: object = response.json()
            except ValueError as exc:
                if self.recorder:
                    self.recorder.error(
                        "model.response.invalid_json",
                        request_id=request_id,
                        http_status=response.status_code,
                        response_characters=len(response.text),
                        elapsed_seconds=round(elapsed, 4),
                    )
                raise RuntimeError(
                    f"Qwen request {request_id} returned non-JSON output."
                ) from exc

            error_type, error_code, error_message_characters = _safe_server_error(body)
            if response.is_error or (isinstance(body, dict) and "error" in body):
                if self.recorder:
                    self.recorder.error(
                        "model.response.error",
                        request_id=request_id,
                        http_status=response.status_code,
                        error_type=error_type,
                        error_code=error_code,
                        error_message_characters=error_message_characters,
                        elapsed_seconds=round(elapsed, 4),
                    )
                raise RuntimeError(
                    "Qwen server failed request "
                    f"{request_id}: HTTP {response.status_code}, type={error_type}, "
                    f"code={error_code or 'none'}. Check the vLLM terminal and run log."
                )

            if not isinstance(body, dict):
                raise RuntimeError(f"Qwen request {request_id} returned an unexpected JSON shape.")

            choices = body.get("choices")
            if not isinstance(choices, list) or not choices:
                if self.recorder:
                    self.recorder.error(
                        "model.response.no_choices",
                        request_id=request_id,
                        response_keys=sorted(body.keys()),
                        elapsed_seconds=round(elapsed, 4),
                    )
                raise RuntimeError(f"Qwen request {request_id} returned no choices.")

            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
            message_content = message.get("content", "") if isinstance(message, dict) else ""

            if isinstance(message_content, str):
                raw_text = message_content
            elif isinstance(message_content, list):
                parts: list[str] = []
                for part in message_content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(str(part["text"]))
                raw_text = "\n".join(parts)
            else:
                raw_text = str(message_content)

            raw_text = raw_text.strip()
            if not raw_text:
                raise RuntimeError(f"Qwen request {request_id} returned an empty message.")

            findings = parse_model_findings(raw_text, clip_duration_seconds)
            usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
            metadata: dict[str, object] = {
                "request_id": request_id,
                "http_status": response.status_code,
                "elapsed_seconds": round(elapsed, 4),
                "output_characters": len(raw_text),
                "audio_included": audio_available,
                "usage": usage,
            }

            if self.recorder:
                self.recorder.info(
                    "model.response.completed",
                    request_id=request_id,
                    http_status=response.status_code,
                    elapsed_seconds=round(elapsed, 4),
                    output_characters=len(raw_text),
                    finding_count=len(findings),
                    finding_types=sorted({item.type for item in findings}),
                    usage=usage,
                )

            return LLMResponse(findings=findings, raw_text=raw_text, metadata=metadata)


class DemoMockClient:
    """Deterministic sample-video client for local pipeline tests."""

    def __init__(self, recorder: RunEventRecorder | None = None) -> None:
        self._clip_index = 0
        self.recorder = recorder

    def analyze_clip(self, clip_path: str | Path, clip_duration_seconds: float) -> LLMResponse:
        del clip_path
        templates: list[list[dict[str, object]]] = [
            [
                {
                    "type": "api_key",
                    "value": "sk-demo-83hhd8282hd91jd82",
                    "modality": "visual",
                    "start_seconds": 0.0,
                    "end_seconds": 0.0,
                    "confidence": 0.99,
                    "reason": "Controlled smoke-test finding.",
                    "visual_location": "Terminal line after API_KEY=",
                }
            ],
            [
                {
                    "type": "email",
                    "value": "alice@example.com",
                    "modality": "both",
                    "start_seconds": 0.0,
                    "end_seconds": 0.0,
                    "confidence": 0.99,
                    "reason": "Controlled smoke-test finding.",
                    "visual_location": "Customer profile email field",
                }
            ],
            [
                {
                    "type": "account_id",
                    "value": "CUST-493821",
                    "modality": "both",
                    "start_seconds": 0.0,
                    "end_seconds": 14.8,
                    "confidence": 0.99,
                    "reason": "Controlled smoke-test finding.",
                    "visual_location": "Support dashboard account field",
                }
            ],
        ]
        payload = templates[self._clip_index] if self._clip_index < len(templates) else []
        self._clip_index += 1
        raw_text = json.dumps(payload)
        findings = parse_model_findings(raw_text, clip_duration_seconds)
        return LLMResponse(
            findings=findings,
            raw_text=raw_text,
            metadata={"request_id": f"mock_{self._clip_index:03d}", "http_status": 200},
        )
