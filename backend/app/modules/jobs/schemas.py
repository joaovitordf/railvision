"""Pydantic schemas for inspection processing jobs."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import AliasChoices, Field

from app.core.schemas import CamelModel


class ProcessInspectionRequest(CamelModel):
    """Optional override parameters when enqueueing inspection video processing."""

    model_id: str | None = None
    runtime: str | None = None
    allow_cpu_fallback: bool = False
    conf_threshold: float | None = Field(
        default=None,
        validation_alias=AliasChoices("confThreshold", "conf_threshold", "conf"),
    )
    iou_threshold: float | None = Field(
        default=None,
        validation_alias=AliasChoices("iouThreshold", "iou_threshold", "iou"),
    )
    sample_fps: float | None = Field(
        default=None,
        validation_alias=AliasChoices("sampleFps", "sample_fps"),
    )


class ProcessingJobResponse(CamelModel):
    """Processing job status response matching Java ApiDtos.ProcessingJobResponse."""

    id: UUID
    inspection_id: UUID
    status: str
    progress_percent: int | None
    engine_type: str
    is_demo_simulation: bool
    error_code: str | None
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
