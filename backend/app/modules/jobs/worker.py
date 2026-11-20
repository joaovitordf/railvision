"""Asynchronous background worker for inspection processing jobs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Inspection, ProcessingJob
from app.modules.jobs.engine import InferenceOutput, InferenceRequest, InspectionInferenceEngine
from app.modules.media.storage import LocalMediaStorageService
from app.modules.results.service import ResultsService

logger = logging.getLogger(__name__)


class ProcessingFailureError(RuntimeError):
    """Domain processing error carrying an explicit machine-readable errorCode."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class _JobContext:
    inspection_id: UUID
    stored_filename: str
    content_type: str


class JobExecutionManager:
    """Executes inspection inference jobs asynchronously with transactional state transitions."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        storage_service: LocalMediaStorageService,
        inference_engine: InspectionInferenceEngine,
        results_service: ResultsService,
        max_concurrency: int = 1,
    ) -> None:
        self._session_factory = session_factory
        self._storage_service = storage_service
        self._inference_engine = inference_engine
        self._results_service = results_service
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_concurrency)),
            thread_name_prefix="railvision-job-worker",
        )

    def submit(self, job_id: UUID) -> None:
        """Schedule a queued job on the managed worker pool."""
        self._executor.submit(self.execute_job, job_id)

    def execute_job(self, job_id: UUID) -> None:
        """Transition QUEUED -> RUNNING -> COMPLETED/FAILED around inference execution."""
        context = self._mark_running(job_id)
        if context is None:
            return

        try:
            video_path = self._storage_service.resolve_safe_upload_path(context.stored_filename)
            if not video_path.is_file() or not os.access(video_path, os.R_OK):
                raise ProcessingFailureError(
                    "MEDIA_UNAVAILABLE",
                    "O arquivo de vídeo armazenado para esta inspeção não está disponível para leitura.",
                )

            logger.info(
                "Starting inference jobId=%s inspectionId=%s engine=%s demoSimulation=%s",
                job_id,
                context.inspection_id,
                self._inference_engine.engine_name,
                self._inference_engine.is_demo_simulation,
            )
            output = self._inference_engine.analyze(
                InferenceRequest(
                    inspection_id=context.inspection_id,
                    job_id=job_id,
                    video_path=video_path,
                    content_type=context.content_type,
                )
            )
            self._mark_completed(job_id, context.inspection_id, output)
            logger.info(
                "Completed inference jobId=%s inspectionId=%s wagons=%d",
                job_id,
                context.inspection_id,
                len(output.wagons),
            )
        except ProcessingFailureError as exc:
            logger.warning(
                "Processing failed jobId=%s inspectionId=%s code=%s",
                job_id,
                context.inspection_id,
                exc.error_code,
            )
            self._mark_failed(job_id, context.inspection_id, exc.error_code, str(exc))
        except Exception:
            logger.exception(
                "Unexpected processing failure jobId=%s inspectionId=%s",
                job_id,
                context.inspection_id,
            )
            self._mark_failed(
                job_id,
                context.inspection_id,
                "INFERENCE_EXECUTION_ERROR",
                "Falha interna durante o processamento da inspeção. Tente reprocessar.",
            )

    def _mark_running(self, job_id: UUID) -> _JobContext | None:
        with self._session_factory() as db:
            job = db.get(ProcessingJob, job_id)
            if job is None or job.status != "QUEUED":
                return None
            inspection = db.get(Inspection, job.inspection_id)
            if inspection is None:
                return None

            now = datetime.now(timezone.utc)
            job.status = "RUNNING"
            job.progress_percent = 15
            job.started_at = now
            inspection.status = "PROCESSING"
            inspection.updated_at = now
            db.commit()
            return _JobContext(
                inspection_id=inspection.id,
                stored_filename=inspection.stored_filename,
                content_type=inspection.content_type,
            )

    def _mark_completed(self, job_id: UUID, inspection_id: UUID, output: InferenceOutput) -> None:
        with self._session_factory() as db:
            self._results_service.persist_inference_output(db, inspection_id, job_id, output)
            now = datetime.now(timezone.utc)
            job = db.get(ProcessingJob, job_id)
            inspection = db.get(Inspection, inspection_id)
            if job is None or inspection is None:
                raise RuntimeError(f"Missing job or inspection when completing {job_id}")
            job.status = "COMPLETED"
            job.progress_percent = 100
            job.finished_at = now
            inspection.status = "COMPLETED"
            inspection.updated_at = now
            db.commit()

    def _mark_failed(self, job_id: UUID, inspection_id: UUID, error_code: str, message: str) -> None:
        with self._session_factory() as db:
            now = datetime.now(timezone.utc)
            job = db.get(ProcessingJob, job_id)
            if job is not None:
                job.status = "FAILED"
                job.error_code = error_code
                job.error_message = message
                job.finished_at = now
            inspection = db.get(Inspection, inspection_id)
            if inspection is not None:
                inspection.status = "FAILED"
                inspection.updated_at = now
            db.commit()

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop accepting new jobs and wait for active workers on shutdown."""
        self._executor.shutdown(wait=wait, cancel_futures=False)
