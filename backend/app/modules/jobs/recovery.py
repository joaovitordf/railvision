"""Startup recovery for processing jobs interrupted by server termination."""

from __future__ import annotations

from datetime import datetime, timezone
import logging

import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Inspection, ProcessingJob

logger = logging.getLogger(__name__)
INTERRUPTED_STATUSES = ("QUEUED", "RUNNING")
ACTIVE_INSPECTION_STATUSES = ("QUEUED", "PROCESSING")


def recover_interrupted_jobs(session_factory: sessionmaker[Session]) -> int:
    """Mark leftover QUEUED/RUNNING jobs as FAILED (PROCESS_INTERRUPTED)."""
    with session_factory() as db:
        inspector = sa.inspect(db.get_bind())
        if not inspector.has_table("processing_jobs") or not inspector.has_table("inspections"):
            return 0

        stale_jobs = list(
            db.scalars(
                sa.select(ProcessingJob).where(ProcessingJob.status.in_(INTERRUPTED_STATUSES))
            )
        )
        if not stale_jobs:
            return 0

        now = datetime.now(timezone.utc)
        for job in stale_jobs:
            job.status = "FAILED"
            job.error_code = "PROCESS_INTERRUPTED"
            job.error_message = (
                "O processamento foi interrompido pelo encerramento do servidor. Solicite o reprocessamento."
            )
            job.finished_at = now

            inspection = db.get(Inspection, job.inspection_id)
            if inspection is not None and inspection.status in ACTIVE_INSPECTION_STATUSES:
                inspection.status = "FAILED"
                inspection.updated_at = now

        db.commit()
        logger.warning(
            "Marked %d interrupted processing job(s) as FAILED (PROCESS_INTERRUPTED).",
            len(stale_jobs),
        )
        return len(stale_jobs)
