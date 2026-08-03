from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Modality = Literal["visual", "audio", "both"]


@dataclass(slots=True)
class ModelFinding:
    """A sensitive item identified semantically by the multimodal model."""

    type: str
    value: str
    modality: Modality
    start_seconds: float
    end_seconds: float
    confidence: float
    reason: str = ""
    visual_location: str | None = None

    def shifted(self, offset_seconds: float) -> ModelFinding:
        return ModelFinding(
            type=self.type,
            value=self.value,
            modality=self.modality,
            start_seconds=max(0.0, self.start_seconds + offset_seconds),
            end_seconds=max(0.0, self.end_seconds + offset_seconds),
            confidence=self.confidence,
            reason=self.reason,
            visual_location=self.visual_location,
        )


@dataclass(slots=True)
class BoxObservation:
    """A pixel rectangle observed at a specific point in the video."""

    time_ms: int
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0


@dataclass(slots=True)
class Finding:
    """A global, redaction-ready finding on the original video timeline."""

    id: str
    type: str
    value: str
    modality: Modality
    start_ms: int
    end_ms: int
    confidence: float
    reason: str = ""
    visual_location: str | None = None
    observations: list[BoxObservation] = field(default_factory=list)
    action: str = "blur_or_mute"
    sources: list[str] = field(default_factory=list)

    def nearest_observation(self, time_ms: int) -> BoxObservation | None:
        if not self.observations:
            return None
        return min(self.observations, key=lambda item: abs(item.time_ms - time_ms))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
