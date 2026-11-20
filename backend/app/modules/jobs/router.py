"""FastAPI router for processing job status polling."""

from uuid import UUID

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.modules.jobs.schemas import ProcessingJobResponse
from app.modules.jobs.service import ProcessingJobService

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=ProcessingJobResponse)
def get_job(
    job_id: UUID,
    request: Request,
    db: Session = Depends(get_db_session),
) -> ProcessingJobResponse:
    """Retrieve current processing job status and metadata."""
    job_service: ProcessingJobService = request.app.state.job_service
    job = job_service.get_job_by_id(db, job_id)
    return job_service.to_job_response(job)
