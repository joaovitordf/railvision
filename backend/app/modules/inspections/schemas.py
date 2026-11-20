"""Pydantic response DTOs for inspections."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from app.core.schemas import CamelModel
from app.modules.jobs.schemas import ProcessingJobResponse


class InspectionSummaryResponse(CamelModel):
    """Summary card projection matching Java ApiDtos.InspectionSummaryResponse."""

    id: UUID
    title: str
    notes: str | None
    status: str
    original_filename: str
    content_type: str
    file_size_bytes: int
    sha256_checksum: str
    latest_job_id: UUID | None
    created_at: datetime
    updated_at: datetime


class InspectionDetailResponse(InspectionSummaryResponse):
    """Full inspection detail projection matching Java ApiDtos.InspectionDetailResponse."""

    duration_ms: int | None
    width_px: int | None
    height_px: int | None
    fps: float | None
    codec: str | None
    wagon_count: int
    review_required_count: int
    latest_job: ProcessingJobResponse | None
