"""Schemas and dataclasses for computer vision detection, tracking, and runtime metadata."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.schemas import CamelModel


class DetectionBox(CamelModel):
    """Detected object bounding box matching OnnxSpikeDtos.DetectionBox."""

    class_id: int
    class_name: str
    confidence: float
    track_id: int | None
    x1: float
    y1: float
    x2: float
    y2: float
    width_px: float
    height_px: float
    norm_x: float
    norm_y: float
    norm_width: float
    norm_height: float


class RuntimeOptionDto(CamelModel):
    """Selectable inference runtime descriptor exposed to the frontend lab."""

    id: str
    label: str
    device_type: str
    available: bool
    provider: str
    diagnostic: str


@dataclass(frozen=True, slots=True)
class FrameDetectionResult:
    """Single-frame detection outcome with explicit runtime provenance and stage latencies."""

    detections: list[DetectionBox]
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    runtime_id: str
    requested_execution_provider: str
    active_execution_provider: str
    device_type: str
    fallback_occurred: bool
