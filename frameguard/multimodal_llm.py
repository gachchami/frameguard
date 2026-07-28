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

    def analyze_clip(self, clip_path: str | Path, clip_duration_seconds: float) -> LLMResponse:
        clip_path = Path(clip_path)
        content = [
            {
                "type": "video_url",
                "video_url": {"url": _data_uri(clip_path)},
            },
            {"type": "text", "text": DETECTION_PROMPT},
        ]
        thinker_sampling: dict[str, Any] = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 1400,
            "seed": 42,
            "detokenize": True,
            "repetition_penalty": 1.05,
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {"role": "user", "content": content},
            ],
            "modalities": ["text"],
            "mm_processor_kwargs": {"use_audio_in_video": True},
            "sampling_params_list": [thinker_sampling, thinker_sampling, thinker_sampling],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
            )
        if response.is_error:
            detail = response.text[-2000:]
            raise RuntimeError(f"Qwen Omni server returned {response.status_code}: {detail}")

        body = response.json()
        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError(f"Qwen Omni response contained no choices: {body}")
        raw_text = str(choices[0].get("message", {}).get("content", ""))
        findings = parse_model_findings(raw_text, clip_duration_seconds)
        return LLMResponse(findings=findings, raw_text=raw_text)


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
