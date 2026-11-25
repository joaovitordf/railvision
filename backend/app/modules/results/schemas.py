"""Pydantic DTOs for wagon results, observations, and registered model versions."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from app.core.schemas import CamelModel


class WagonIdentityDto(CamelModel):
    """Track and sequence identification for a consolidated wagon."""

    track_id: str
    sequence_index: int
    physical_id: str | None = None


class MaterialEstimateDto(CamelModel):
    """Consolidated cargo material classification metrics."""

    material_class: str | None
    confidence: float | None
    vote_distribution: dict[str, int]
    rejected_as_unknown: bool


class LoadingEstimateDto(CamelModel):
    """Consolidated wagon loading level and measurement provenance."""

    level: str
    level_observation_spread: list[str]
    estimated_fill_percentage: float | None
    confidence: float | None
    measurement_type: str
    physical_volume_verified: bool


class ConsolidationDto(CamelModel):
    """Cross-frame observation consolidation summary."""

    observations_count: int
    disagreements: list[str]
    method: str | None


class WagonResultResponse(CamelModel):
    """Consolidated per-wagon inspection result."""

    id: UUID
    inspection_id: UUID
    job_id: UUID
    wagon_identity: WagonIdentityDto
    material: MaterialEstimateDto | None
    loading_estimate: LoadingEstimateDto
    model_versions: dict[str, str]
    consolidation: ConsolidationDto
    quality_flags: list[str]
    review_required: bool
    review_reasons: list[str]
    is_example: bool
    created_at: datetime


class BoundingBoxDto(CamelModel):
    """Bounding box rectangle matching Java API contract."""

    x: float
    y: float
    width: float
    height: float
    coordinate_system: str


class WagonObservationResponse(CamelModel):
    """Per-frame observation linked to a consolidated wagon result."""

    id: UUID
    wagon_result_id: UUID
    frame_index: int
    frame_timestamp_ms: int
    bounding_box: BoundingBoxDto
    detected_class: str
    detection_confidence: float
    material_class: str | None
    material_confidence: float | None
    loading_level: str | None
    loading_confidence: float | None
    is_example: bool


class ModelVersionResponse(CamelModel):
    """Registered inference/tracking model metadata row."""

    id: UUID
    task: str
    name: str
    version: str
    runtime: str
    weights_sha256: str | None
    artifact_ref: str | None
    is_demo_placeholder: bool
    created_at: datetime


class ModelCheckpointDto(CamelModel):
    """Registered open-top wagon detection checkpoint entry from registry.json."""

    id: str
    model_name: str
    source_experiment: str
    architecture: str
    status: str
    input_size: int
    class_names: dict[str, str]
    pytorch_weights_path: str | None
    pytorch_weights_sha256: str | None
    onnx_model_path: str | None
    onnx_meta_path: str | None
    onnx_model_sha256: str | None
    limitations_notice: str
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
