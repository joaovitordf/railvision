"""Inspection upload, listing, detail lookup, and DTO mapping service."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import UploadFile
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.core.exceptions import ResourceNotFoundException
from app.db.models import Inspection, ProcessingJob
from app.modules.inspections.schemas import InspectionDetailResponse, InspectionSummaryResponse
from app.modules.jobs.schemas import ProcessingJobResponse
from app.modules.media.probe import try_probe_video_metadata
from app.modules.media.storage import LocalMediaStorageService
from app.modules.media.validator import VideoUploadValidator, strip_extension


class InspectionService:
    """Manages inspection uploads, persistence, and retrieval."""

    def __init__(
        self,
        *,
        upload_validator: VideoUploadValidator,
        storage_service: LocalMediaStorageService,
    ) -> None:
        self._upload_validator = upload_validator
        self._storage_service = storage_service

    def create_inspection(
        self,
        db: Session,
        *,
        file: UploadFile,
        title: str | None = None,
        notes: str | None = None,
    ) -> Inspection:
        """Validate upload, persist locally with SHA-256, probe video metadata, and save Inspection row."""
        validated = self._upload_validator.validate(file)
        stored = self._storage_service.store_upload(file, validated.extension)
        meta = try_probe_video_metadata(stored.full_path)

        resolved_title = title.strip() if (title and title.strip()) else strip_extension(validated.original_filename)
        if len(resolved_title) > 200:
            resolved_title = resolved_title[:200]
        resolved_notes = notes.strip() if (notes and notes.strip()) else None

        now = datetime.now(timezone.utc)
        entity = Inspection(
            id=uuid4(),
            title=resolved_title,
            notes=resolved_notes,
            status="UPLOADED",
            original_filename=validated.original_filename,
            stored_filename=stored.stored_filename,
            content_type=validated.content_type,
            file_size_bytes=stored.file_size_bytes,
            sha256_checksum=stored.sha256_checksum,
            duration_ms=meta.duration_ms if meta else None,
            width_px=meta.width_px if meta else None,
            height_px=meta.height_px if meta else None,
            fps=round(meta.fps, 3) if meta else None,
            codec=meta.codec if meta else None,
            latest_job_id=None,
            created_at=now,
            updated_at=now,
        )
        db.add(entity)
        db.commit()
        db.refresh(entity)
        return entity

    def list_inspections(self, db: Session) -> list[Inspection]:
        """Return inspections ordered newest-first."""
        return list(db.scalars(sa.select(Inspection).order_by(Inspection.created_at.desc())))

    def get_by_id(self, db: Session, inspection_id: UUID) -> Inspection:
        """Fetch inspection or raise ResourceNotFoundException."""
        entity = db.get(Inspection, inspection_id)
        if entity is None:
            raise ResourceNotFoundException(f"Inspeção não encontrada: {inspection_id}")
        return entity

    def to_summary(self, entity: Inspection) -> InspectionSummaryResponse:
        """Map ORM entity to summary response DTO."""
        return InspectionSummaryResponse.model_validate(entity)

    def to_detail(
        self,
        entity: Inspection,
        *,
        latest_job: ProcessingJob | None,
        wagon_count: int,
        review_required_count: int,
    ) -> InspectionDetailResponse:
        """Map ORM entity, latest job, and counts to detail response DTO."""
        return InspectionDetailResponse(
            id=entity.id,
            title=entity.title,
            notes=entity.notes,
            status=entity.status,
            original_filename=entity.original_filename,
            content_type=entity.content_type,
            file_size_bytes=entity.file_size_bytes,
            sha256_checksum=entity.sha256_checksum,
            latest_job_id=entity.latest_job_id,
            duration_ms=entity.duration_ms,
            width_px=entity.width_px,
            height_px=entity.height_px,
            fps=entity.fps,
            codec=entity.codec,
            wagon_count=wagon_count,
            review_required_count=review_required_count,
            latest_job=ProcessingJobResponse.model_validate(latest_job) if latest_job else None,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )
