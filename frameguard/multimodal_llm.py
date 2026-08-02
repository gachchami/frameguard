from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from .parse_findings import parse_model_findings
from .prompts import DETECTION_PROMPT, SYSTEM_PROMPT
from .schemas import ModelFinding


@dataclass(frozen=True, slots=True)
class LLMResponse:
    findings: list[ModelFinding]
    raw_text: str


class ClipAnalyzer(Protocol):
    def analyze_clip(self, clip_path: str | Path, clip_duration_seconds: float) -> LLMResponse:
        """Return semantic findings for one video clip."""


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class QwenOmniClient:
    """Client for a local or remote vLLM-Omni OpenAI-compatible server."""

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_seconds: float = 300.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def healthcheck(self) -> None:
        base = self.api_base.removesuffix("/v1")
        response = httpx.get(f"{base}/health", timeout=10.0)
        response.raise_for_status()

    def analyze_clip(
        self,
        clip_path: str | Path,
        clip_duration_seconds: float,
    ) -> LLMResponse:
        """Analyze one video chunk using Qwen2.5-Omni.

        The video and its extracted audio are sent as separate multimodal
        inputs. This avoids relying on the V0-only embedded-audio-in-video path.
        """

        clip_path = Path(clip_path).resolve()

        if not clip_path.exists():
            raise FileNotFoundError(f"Video clip does not exist: {clip_path}")

        if clip_path.stat().st_size == 0:
            raise ValueError(f"Video clip is empty: {clip_path}")

        with tempfile.TemporaryDirectory(
            prefix="frameguard_audio_",
        ) as temporary_directory:
            audio_path = Path(temporary_directory) / f"{clip_path.stem}.wav"

            # Extract the first audio stream as a mono, 16 kHz PCM WAV.
            #
            # The trailing ? in 0:a:0? makes the audio stream optional.
            # This lets FrameGuard continue with video-only clips.
            ffmpeg_command = [
                "ffmpeg",
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

            ffmpeg_result = subprocess.run(
                ffmpeg_command,
                capture_output=True,
                text=True,
                check=False,
            )

            audio_available = (
                ffmpeg_result.returncode == 0
                and audio_path.exists()
                and audio_path.stat().st_size > 44
            )

            # Start with the visual video input.
            content: list[dict[str, Any]] = [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": _data_uri(clip_path),
                    },
                }
            ]

            # Add the audio as a separate input when the video has audio.
            if audio_available:
                content.append(
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": _data_uri(audio_path),
                        },
                    }
                )

            content.append(
                {
                    "type": "text",
                    "text": DETECTION_PROMPT,
                }
            )

            # This is a plain-vLLM request.
            #
            # Do not include:
            # - modalities
            # - sampling_params_list
            # - mm_processor_kwargs/use_audio_in_video
            #
            # Those belong to other serving paths and were ignored by the
            # current plain-vLLM server.
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": SYSTEM_PROMPT,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 1400,
                "seed": 42,
                "repetition_penalty": 1.05,
            }

            headers = {
                "Content-Type": "application/json",
            }

            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            endpoint = f"{self.api_base.rstrip('/')}/chat/completions"

            try:
                with httpx.Client(
                    timeout=self.timeout_seconds,
                ) as client:
                    response = client.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                    )
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    "Could not connect to the Qwen model server at "
                    f"{self.api_base}. Confirm that vLLM is running."
                ) from exc
            except httpx.TimeoutException as exc:
                raise RuntimeError(
                    f"The Qwen model request timed out after {self.timeout_seconds} seconds."
                ) from exc

            if response.is_error:
                raise RuntimeError(
                    f"Qwen server returned HTTP {response.status_code}: {response.text[-4000:]}"
                )

            try:
                body = response.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"Qwen server returned a non-JSON response: {response.text[-4000:]}"
                ) from exc

            if "error" in body:
                raise RuntimeError(f"Qwen server returned an internal error: {body['error']}")

            choices = body.get("choices")

            if not choices:
                raise RuntimeError(f"Qwen response contained no choices: {body}")

            message = choices[0].get("message", {})
            message_content = message.get("content", "")

            # Most vLLM responses return content as a string. This also
            # supports OpenAI-style content-part lists.
            if isinstance(message_content, str):
                raw_text = message_content
            elif isinstance(message_content, list):
                text_parts: list[str] = []

                for part in message_content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict):
                        part_text = part.get("text")

                        if isinstance(part_text, str):
                            text_parts.append(part_text)

                raw_text = "\n".join(text_parts)
            else:
                raw_text = str(message_content)

            raw_text = raw_text.strip()

            if not raw_text:
                raise RuntimeError(f"Qwen returned an empty message. Full response: {body}")

            findings = parse_model_findings(
                raw_text,
                clip_duration_seconds,
            )

            return LLMResponse(
                findings=findings,
                raw_text=raw_text,
            )


class DemoMockClient:
    """Deterministic sample-video detector for laptop plumbing tests.

    This is deliberately not presented as AI inference. It returns the findings
    expected in ``samples/frameguard_demo.mp4`` so the video splitting, OCR
    localization, blurring, audio muting, reporting, and Gradio UI can be tested
    before a Linux GPU model server is available.
    """

    def __init__(self) -> None:
        self._clip_index = 0

    def analyze_clip(self, clip_path: str | Path, clip_duration_seconds: float) -> LLMResponse:
        del clip_path
        templates: list[list[dict[str, object]]] = [
            [
                {
                    "type": "api_key",
                    "value": "sk-demo-83hhd8282hd91jd82",
                    "modality": "visual",
                    "start_seconds": 0.2,
                    "end_seconds": min(4.9, clip_duration_seconds),
                    "confidence": 0.99,
                    "reason": "Controlled laptop smoke-test finding.",
                    "visual_location": "Terminal line after API_KEY=",
                }
            ],
            [
                {
                    "type": "email",
                    "value": "alice@example.com",
                    "modality": "visual",
                    "start_seconds": 0.1,
                    "end_seconds": min(4.9, clip_duration_seconds),
                    "confidence": 0.99,
                    "reason": "Controlled laptop smoke-test finding.",
                    "visual_location": "Customer profile email field",
                },
                {
                    "type": "ip_address",
                    "value": "192.168.1.24",
                    "modality": "visual",
                    "start_seconds": 0.1,
                    "end_seconds": min(4.9, clip_duration_seconds),
                    "confidence": 0.99,
                    "reason": "Controlled laptop smoke-test finding.",
                    "visual_location": "Internal server field",
                },
            ],
            [
                {
                    "type": "account_id",
                    "value": "CUST-493821",
                    "modality": "visual",
                    "start_seconds": 0.1,
                    "end_seconds": min(4.9, clip_duration_seconds),
                    "confidence": 0.99,
                    "reason": "Controlled laptop smoke-test finding.",
                    "visual_location": "Support dashboard account field",
                },
                {
                    "type": "phone_number",
                    "value": "+91 98765 43210",
                    "modality": "audio",
                    "start_seconds": 0.0,
                    "end_seconds": min(4.9, clip_duration_seconds),
                    "confidence": 0.99,
                    "reason": "Controlled laptop smoke-test finding.",
                    "visual_location": None,
                },
            ],
        ]
        findings_payload = templates[self._clip_index] if self._clip_index < len(templates) else []
        self._clip_index += 1
        raw_text = json.dumps({"findings": findings_payload})
        return LLMResponse(
            findings=parse_model_findings(raw_text, clip_duration_seconds),
            raw_text=raw_text,
        )
