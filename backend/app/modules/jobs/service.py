"""Processing job creation, query, and DTO mapping service."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import ConflictException, ResourceNotFoundException
from app.db.models import Inspection, ProcessingJob
from app.modules.jobs.engine import InspectionInferenceEngine
from app.modules.jobs.schemas import ProcessInspectionRequest, ProcessingJobResponse
from app.modules.jobs.worker import JobExecutionManager

ACTIVE_JOB_STATUSES = ("QUEUED", "RUNNING")


class ProcessingJobService:
    """Coordinates enqueueing and retrieval of inspection processing jobs."""

    def __init__(
        self,
        *,
        inference_engine: InspectionInferenceEngine,
        worker_manager: JobExecutionManager,
    ) -> None:
        self._inference_engine = inference_engine
        self._worker_manager = worker_manager

    def enqueue_processing(
        self,
        db: Session,
        inspection_id: UUID,
        *,
        options: ProcessInspectionRequest | None = None,
    ) -> ProcessingJob:
        """Create a QUEUED job under row lock, commit DB changes, then dispatch async execution."""
        inspection = db.scalar(
            sa.select(Inspection).where(Inspection.id == inspection_id).with_for_update()
        )
        if inspection is None:
            raise ResourceNotFoundException(f"Inspeção não encontrada: {inspection_id}")

        active_job_id = db.scalar(
            sa.select(ProcessingJob.id)
            .where(
                ProcessingJob.inspection_id == inspection_id,
                ProcessingJob.status.in_(ACTIVE_JOB_STATUSES),
            )
            .limit(1)
        )
        if active_job_id is not None:
            raise ConflictException("Já existe um job em fila ou execução para esta inspeção.")

        now = datetime.now(timezone.utc)
        job = ProcessingJob(
            id=uuid4(),
            inspection_id=inspection_id,
            status="QUEUED",
            progress_percent=0,
            engine_type=self._inference_engine.engine_name,
            is_demo_simulation=self._inference_engine.is_demo_simulation,
            created_at=now,
        )
        db.add(job)
        inspection.status = "QUEUED"
        inspection.latest_job_id = job.id
        inspection.updated_at = now
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise ConflictException("Já existe um job em fila ou execução para esta inspeção.") from exc
        db.refresh(job)

        try:
            self._worker_manager.submit(job.id, options=options)
        except Exception as exc:
            fail_now = datetime.now(timezone.utc)
            job.status = "FAILED"
            job.error_code = "SUBMIT_FAILED"
            job.error_message = "Falha ao submeter job para a fila de execução."
            job.finished_at = fail_now
            inspection.status = "FAILED"
            inspection.updated_at = fail_now
            db.commit()
            raise ConflictException("Não foi possível submeter o processamento neste momento.") from exc
        return job

    def get_job_by_id(self, db: Session, job_id: UUID) -> ProcessingJob:
        """Fetch processing job or raise ResourceNotFoundException."""
        job = db.get(ProcessingJob, job_id)
        if job is None:
            raise ResourceNotFoundException(f"Job de processamento não encontrado: {job_id}")
        return job

    def find_latest_for_inspection(self, db: Session, inspection_id: UUID) -> ProcessingJob | None:
        """Return latest processing job for an inspection ordered by created_at DESC."""
        return db.scalar(
            sa.select(ProcessingJob)
            .where(ProcessingJob.inspection_id == inspection_id)
            .order_by(ProcessingJob.created_at.desc())
            .limit(1)
        )

    def to_job_response(self, job: ProcessingJob) -> ProcessingJobResponse:
        """Map ORM ProcessingJob to camelCase API DTO."""
        return ProcessingJobResponse.model_validate(job)
