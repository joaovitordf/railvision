"""Pydantic DTOs for the Annotation Studio (/api/v1/annotations)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field

from app.core.schemas import CamelModel

AnnotationItemStatus = Literal[
    "pending",
    "pre_annotated",
    "in_review",
    "reviewed",
    "approved",
    "rejected",
]
BoxOrigin = Literal["manual", "ai_suggestion", "ai_accepted", "ai_edited"]


class AnnotationBoxDto(CamelModel):
    """Normalized [top-left x, top-left y, width, height] bounding box in [0, 1]."""

    id: str
    class_id: int = 0
    class_name: str = "open_top_freight_wagon"
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)
    origin: BoxOrigin = "manual"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    model_id: str | None = None
    model_sha256: str | None = None


class CreateAnnotationProjectRequest(CamelModel):
    """Payload to create an isolated Annotation Studio project."""

    name: str = Field(min_length=2, max_length=120)
    campaign_id: str = Field(min_length=2, max_length=80)
    source_origin: str = Field(default="proprio", min_length=2, max_length=80)
    description: str | None = Field(default=None, max_length=1000)
    classes: list[str] = Field(default_factory=lambda: ["open_top_freight_wagon"])


class AnnotationProjectSummaryDto(CamelModel):
    """Project overview card with per-status counters."""

    id: UUID
    name: str
    campaign_id: str
    source_origin: str
    description: str | None = None
    classes: list[str]
    item_count: int = 0
    video_count: int = 0
    status_counts: dict[str, int] = Field(default_factory=dict)
    reviewed_count: int = 0
    approved_count: int = 0
    created_at: datetime
    updated_at: datetime


class AnnotationVideoDto(CamelModel):
    """Registered video within an Annotation Studio project for frame extraction."""

    id: UUID
    filename: str
    source_origin: str
    campaign_id: str
    sha256: str
    width_px: int
    height_px: int
    fps: float
    duration_ms: int
    codec: str
    estimated_frames: int
    reserved_benchmark_warning: bool = False
    imported_at: datetime


class AnnotationItemSummaryDto(CamelModel):
    """Summary projection for an image item inside an annotation project."""

    id: UUID
    project_id: UUID
    filename: str
    source_origin: str
    campaign_id: str
    source_sha256: str
    width_px: int
    height_px: int
    status: AnnotationItemStatus = "pending"
    human_reviewed: bool = False
    no_objects_confirmed: bool = False
    box_count: int = 0
    ai_suggestion_count: int = 0
    reserved_benchmark_warning: bool = False
    source_video_id: UUID | None = None
    source_video_filename: str | None = None
    frame_index: int | None = None
    timestamp_ms: int | None = None
    image_url: str
    thumbnail_url: str
    created_at: datetime
    updated_at: datetime
    reviewed_at: datetime | None = None
    approved_at: datetime | None = None


class AnnotationItemDetailDto(AnnotationItemSummaryDto):
    """Full annotation item state including editable boxes and audit provenance."""

    boxes: list[AnnotationBoxDto] = Field(default_factory=list)
    review_notes: str | None = None
    false_positives_removed: int = 0
    false_negatives_added: int = 0
    preannotation_model_id: str | None = None
    preannotation_model_sha256: str | None = None
    preannotation_conf: float | None = None
    preannotation_iou: float | None = None


class AnnotationProjectDetailDto(AnnotationProjectSummaryDto):
    """Project details including filtered items and imported videos."""

    items: list[AnnotationItemSummaryDto] = Field(default_factory=list)
    videos: list[AnnotationVideoDto] = Field(default_factory=list)


class UpdateItemAnnotationsRequest(CamelModel):
    """Human review update persisting boxes, negative confirmation, status, and audit stats."""

    boxes: list[AnnotationBoxDto] = Field(default_factory=list)
    no_objects_confirmed: bool = False
    status: AnnotationItemStatus | None = None
    review_notes: str | None = Field(default=None, max_length=2000)
    false_positives_removed: int = Field(default=0, ge=0)
    false_negatives_added: int = Field(default=0, ge=0)


class PreviewFramesRequest(CamelModel):
    """Parameters to preview deduplicated candidate frames from a project video."""

    sample_fps: float = Field(default=2.0, gt=0.05, le=30.0)
    max_frames: int = Field(default=40, ge=1, le=120)
    dedup_threshold: float = Field(default=4.0, ge=0.0, le=50.0)


class PreviewFrameCandidateDto(CamelModel):
    """Candidate video frame thumbnail metadata before explicit import."""

    frame_index: int
    timestamp_ms: int
    diff_score: float
    preview_url: str


class PreviewFramesResponse(CamelModel):
    """Response summarizing preview generation and near-duplicate skips."""

    video_id: UUID
    candidates: list[PreviewFrameCandidateDto]
    scanned_frames: int
    skipped_similar: int


class ExtractFramesRequest(CamelModel):
    """Selected or auto-sampled frame indices to commit into the project workspace."""

    frame_indices: list[int] | None = None
    sample_fps: float = Field(default=2.0, gt=0.05, le=30.0)
    max_frames: int = Field(default=40, ge=1, le=120)
    dedup_threshold: float = Field(default=4.0, ge=0.0, le=50.0)


class PreannotateRequest(CamelModel):
    """Run local VisionRuntimeManager inference to populate AI suggestions."""

    model_id: str | None = None
    runtime: str | None = None
    conf: float = Field(default=0.25, ge=0.01, le=0.99)
    iou: float = Field(default=0.45, ge=0.05, le=0.95)
    overwrite: bool = False
    item_ids: list[UUID] | None = None


class PreannotateBatchResponse(CamelModel):
    """Summary of batch AI pre-annotation execution across project items."""

    project_id: UUID
    processed_items: int
    skipped_reviewed: int
    total_suggestions: int
    model_id: str
    runtime_id: str


class ExportYoloRequest(CamelModel):
    """Options for CVAT-compatible YOLO 1.1 zip export."""

    only_human_reviewed: bool = True
    include_unreviewed_drafts: bool = False


class ImportCurationResponse(CamelModel):
    """Summary returned when importing frames from Vision Lab curation-inbox."""

    imported_count: int
    skipped_duplicates: int
    items: list[AnnotationItemSummaryDto]
