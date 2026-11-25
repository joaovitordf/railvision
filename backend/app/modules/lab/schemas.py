"""Pydantic camelCase schemas for /api/v1/onnx-spike lab endpoints."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field

from app.core.schemas import CamelModel
from app.modules.vision.schemas import DetectionBox, RuntimeOptionDto


class RuntimeStatusDto(CamelModel):
    """Runtime and model inspection payload matching OnnxSpikeDtos.RuntimeStatusDto plus Python options."""

    available: bool
    unavailable_reason: str | None
    model_name: str
    source_experiment: str
    model_display_path: str
    model_sha256: str
    input_size: int
    input_shape: list[int]
    output_shape: list[int]
    class_names: dict[int, str]
    requested_execution_provider: str
    active_execution_provider: str
    device_type: str
    cuda_available: bool
    fallback_occurred: bool
    available_providers: list[str]
    device_diagnostic: str
    tracking_integrated: bool
    tracking_status: str
    tracking_notice: str
    limitations_notice: str
    default_conf_threshold: float
    default_iou_threshold: float
    max_video_frames_per_ui_run: int
    supported_image_extensions: list[str]
    supported_video_extensions: list[str]
    backend_engine: str = "python-fastapi"
    default_runtime: str = "pytorch_cuda"
    available_runtimes: list[RuntimeOptionDto] = []
    default_output_resolution: str = "1080p"
    available_output_resolutions: list[str] = ["1080p", "native"]
    video_encoder: str = "libx264"
    video_encoder_crf: int = 23
    tracking_algorithms: list[str] = ["none", "bytetrack"]
    active_model_id: str | None = None


class InferenceTimingMetrics(CamelModel):
    """Frame latency and throughput telemetry."""

    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    decode_or_encode_ms: float
    total_ms: float
    avg_frame_ms: float
    p50_frame_ms: float
    p95_frame_ms: float
    processing_fps: float
    inference_only_fps: float


class ModelConfigUsed(CamelModel):
    """Model, runtime, threshold, and tracking metadata used by a run."""

    model_name: str
    source_experiment: str
    model_sha256: str
    input_size: int
    conf_threshold: float
    iou_threshold: float
    sample_fps: float | None
    device_type: str
    active_execution_provider: str
    tracking_integrated: bool
    tracking_status: str
    exploratory_notice: str
    runtime_id: str | None = None
    model_id: str | None = None
    fallback_occurred: bool = False
    output_resolution: str | None = None
    tracker_algorithm: str | None = None


class RunFileLinks(CamelModel):
    """Download and inline streaming URLs for original and annotated files."""

    original_url: str
    annotated_url: str | None
    original_download_url: str
    annotated_download_url: str | None
    original_filename: str
    annotated_filename: str | None
    annotated_content_type: str | None


class ImageRunResultDto(CamelModel):
    """Completed single-image inference response."""

    run_id: UUID
    media_type: str
    status: str
    created_at: datetime
    original_filename: str
    image_width: int
    image_height: int
    detection_count: int
    no_detections: bool
    empty_message: str | None
    detections: list[DetectionBox]
    metrics: InferenceTimingMetrics
    model_config_used: ModelConfigUsed = Field(alias="modelConfig")
    files: RunFileLinks


class VideoFrameSummaryDto(CamelModel):
    """Per-frame detection telemetry within a processed video run."""

    frame_index: int
    timestamp_ms: int
    detection_count: int
    frame_latency_ms: float
    detections: list[DetectionBox]


class VideoRunStatusDto(CamelModel):
    """Queued, running, completed, or failed video analysis run payload."""

    run_id: UUID
    media_type: str
    status: str
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    original_filename: str
    video_width: int
    video_height: int
    source_fps: float
    source_codec: str
    source_duration_ms: int
    output_codec: str | None
    output_container: str | None
    browser_playable: bool
    estimated_total_frames: int
    processed_frames: int
    progress_percent: int
    frames_with_detections: int
    total_detections: int
    no_detections: bool
    truncated_by_frame_limit: bool
    empty_message: str | None
    error_message: str | None
    metrics: InferenceTimingMetrics | None
    model_config_used: ModelConfigUsed = Field(alias="modelConfig")
    files: RunFileLinks
    frame_summaries: list[VideoFrameSummaryDto]
    frame_summaries_truncated: bool = False
    total_frame_summaries: int | None = None
    output_resolution: str | None = None
    output_width: int | None = None
    output_height: int | None = None
    output_encoder: str | None = None
    output_crf: int | None = None
    unique_track_count: int | None = None


class RunListItemDto(CamelModel):
    """Compact summary row for the historical runs table."""

    run_id: UUID
    media_type: str
    status: str
    original_filename: str
    created_at: datetime
    total_detections: int
    no_detections: bool
    processing_fps: float | None
    avg_frame_ms: float | None
    total_ms: float | None
    device_type: str
    active_execution_provider: str
    model_name: str
    model_id: str | None = None
    runtime_id: str | None = None
    tracking_integrated: bool = False
    unique_track_count: int | None = None


class RunComparisonItemDto(CamelModel):
    """Normalized summary of a single run within an A/B comparison."""

    run_id: UUID
    media_type: str
    status: str
    original_filename: str
    model_name: str
    model_id: str | None = None
    model_sha256: str | None = None
    source_experiment: str
    input_size: int | None = None
    runtime_id: str | None = None
    device_type: str
    conf_threshold: float | None = None
    iou_threshold: float | None = None
    sample_fps: float | None = None
    source_resolution: str | None = None
    output_resolution: str | None = None
    tracking_integrated: bool = False
    total_detections: int
    unique_track_count: int | None = None
    processing_fps: float | None = None
    inference_only_fps: float | None = None
    avg_frame_ms: float | None = None
    p95_frame_ms: float | None = None


class FrameComparisonDeltaDto(CamelModel):
    """Per-frame detection delta when comparing two video runs."""

    frame_index: int
    timestamp_ms: int
    detections_a: int
    detections_b: int
    delta: int


class RunComparisonDto(CamelModel):
    """Side-by-side A/B run comparison response."""

    run_a: RunComparisonItemDto
    run_b: RunComparisonItemDto
    comparable: bool = True
    comparability_warnings: list[str] = []
    delta_detections: int
    delta_processing_fps: float | None = None
    delta_avg_frame_ms: float | None = None
    frame_deltas: list[FrameComparisonDeltaDto] = []


class CurateFrameRequest(CamelModel):
    """Request to export a clean original frame + pre-annotations into HITL curation inbox."""

    frame_index: int | None = Field(default=None, ge=0)
    reason: str = Field(min_length=2, max_length=120)
    notes: str | None = Field(default=None, max_length=1000)
    target_campaign: str | None = Field(default=None, max_length=120)


class CurationExportDto(CamelModel):
    """Metadata of an exported clean frame waiting for 100% human review in CVAT."""

    run_id: UUID
    frame_index: int
    image_path: str
    label_path: str
    metadata_path: str
    source_sha256: str
    model_sha256: str | None = None
    reserved_benchmark_warning: bool = False
    detection_count: int
    reason: str
    human_reviewed: bool
    auto_train_eligible: bool
    created_at: datetime
