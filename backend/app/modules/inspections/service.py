"""Inspection upload, listing, detail lookup, and DTO mapping service."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import UploadFile
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.core.exceptions import InvalidMediaException, ResourceNotFoundException
from app.db.models import Inspection, ProcessingJob, WagonResult
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
        resolved_notes = notes.strip() if (notes and notes.strip()) else None
        if resolved_notes is not None and len(resolved_notes) > 2000:
            raise InvalidMediaException(
                "NOTES_TOO_LONG",
                "O campo de observações excede o limite de 2000 caracteres.",
            )

        stored = self._storage_service.store_upload(file, validated.extension)
        try:
            meta = try_probe_video_metadata(stored.full_path)

            resolved_title = title.strip() if (title and title.strip()) else strip_extension(validated.original_filename)
            if len(resolved_title) > 200:
                resolved_title = resolved_title[:200]

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
        except Exception:
            stored.full_path.unlink(missing_ok=True)
            raise

    def list_inspections(self, db: Session) -> list[Inspection]:
        """Return inspections ordered newest-first."""
        return list(db.scalars(sa.select(Inspection).order_by(Inspection.created_at.desc())))

    def list_inspections_with_counts(self, db: Session) -> list[tuple[Inspection, int]]:
        """Return inspections newest-first alongside latest completed job wagon counts."""
        inspections = self.list_inspections(db)
        if not inspections:
            return []

        completed_rows = db.execute(
            sa.select(
                ProcessingJob.inspection_id,
                ProcessingJob.id,
                ProcessingJob.created_at,
            )
            .where(ProcessingJob.status == "COMPLETED")
            .order_by(ProcessingJob.created_at.desc())
        ).all()

        latest_completed_job_by_insp: dict[UUID, UUID] = {}
        for insp_id, job_id, _ in completed_rows:
            if insp_id not in latest_completed_job_by_insp:
                latest_completed_job_by_insp[insp_id] = job_id

        wagon_counts_by_job: dict[UUID, int] = {}
        target_job_ids = list(latest_completed_job_by_insp.values())
        if target_job_ids:
            count_rows = db.execute(
                sa.select(WagonResult.job_id, sa.func.count())
                .where(WagonResult.job_id.in_(target_job_ids))
                .group_by(WagonResult.job_id)
            ).all()
            wagon_counts_by_job = {job_id: int(cnt) for job_id, cnt in count_rows}

        return [
            (
                insp,
                wagon_counts_by_job.get(latest_completed_job_by_insp.get(insp.id), 0),  # type: ignore[arg-type]
            )
            for insp in inspections
        ]

    def get_by_id(self, db: Session, inspection_id: UUID) -> Inspection:
        """Fetch inspection or raise ResourceNotFoundException."""
        entity = db.get(Inspection, inspection_id)
        if entity is None:
            raise ResourceNotFoundException(f"Inspeção não encontrada: {inspection_id}")
        return entity

    def to_summary(self, entity: Inspection, *, wagon_count: int = 0) -> InspectionSummaryResponse:
        """Map ORM entity to summary response DTO."""
        return InspectionSummaryResponse(
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
            wagon_count=wagon_count,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

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
