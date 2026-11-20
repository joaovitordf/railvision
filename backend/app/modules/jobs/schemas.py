"""Pydantic schemas for inspection processing jobs."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from app.core.schemas import CamelModel


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
