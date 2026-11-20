"""FastAPI router for inspection lifecycle, media streaming, jobs, wagons, and observations."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Header, Request, Response, UploadFile, status
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.modules.inspections.schemas import InspectionDetailResponse, InspectionSummaryResponse
from app.modules.inspections.service import InspectionService
from app.modules.jobs.schemas import ProcessingJobResponse
from app.modules.jobs.service import ProcessingJobService
from app.modules.media.storage import LocalMediaStorageService
from app.modules.results.schemas import WagonObservationResponse, WagonResultResponse
from app.modules.results.service import ResultsService

router = APIRouter(prefix="/api/v1/inspections", tags=["inspections"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=InspectionSummaryResponse)
def create_inspection(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    notes: str | None = Form(default=None),
    db: Session = Depends(get_db_session),
) -> InspectionSummaryResponse:
    """Upload a freight train inspection video and register its metadata."""
    inspection_service: InspectionService = request.app.state.inspection_service
    created = inspection_service.create_inspection(db, file=file, title=title, notes=notes)
    response.headers["Location"] = str(request.url_for("get_inspection", inspection_id=str(created.id)))
    return inspection_service.to_summary(created)


@router.get("", response_model=list[InspectionSummaryResponse])
def list_inspections(
    request: Request,
    db: Session = Depends(get_db_session),
) -> list[InspectionSummaryResponse]:
    """List inspections ordered by creation timestamp descending."""
    inspection_service: InspectionService = request.app.state.inspection_service
    return [inspection_service.to_summary(item) for item in inspection_service.list_inspections(db)]


@router.get("/{inspection_id}", response_model=InspectionDetailResponse, name="get_inspection")
def get_inspection(
    inspection_id: UUID,
    request: Request,
    db: Session = Depends(get_db_session),
) -> InspectionDetailResponse:
    """Get full inspection details including latest job and consolidated wagon counts."""
    inspection_service: InspectionService = request.app.state.inspection_service
    job_service: ProcessingJobService = request.app.state.job_service
    results_service: ResultsService = request.app.state.results_service

    inspection = inspection_service.get_by_id(db, inspection_id)
    latest_job = job_service.find_latest_for_inspection(db, inspection_id)
    completed_job_id = latest_job.id if (latest_job is not None and latest_job.status == "COMPLETED") else None
    wagon_count, review_required_count = results_service.summarize_for_job(db, inspection_id, completed_job_id)

    return inspection_service.to_detail(
        inspection,
        latest_job=latest_job,
        wagon_count=wagon_count,
        review_required_count=review_required_count,
    )


@router.post("/{inspection_id}/process", status_code=status.HTTP_202_ACCEPTED, response_model=ProcessingJobResponse)
def process_inspection(
    inspection_id: UUID,
    request: Request,
    db: Session = Depends(get_db_session),
) -> ProcessingJobResponse:
    """Enqueue asynchronous processing for an inspection."""
    job_service: ProcessingJobService = request.app.state.job_service
    job = job_service.enqueue_processing(db, inspection_id)
    return job_service.to_job_response(job)


@router.get("/{inspection_id}/media")
def stream_inspection_media(
    inspection_id: UUID,
    request: Request,
    range_header: str | None = Header(default=None, alias="Range"),
    db: Session = Depends(get_db_session),
) -> Response:
    """Stream uploaded inspection video with full (200) or HTTP byte-range (206) support."""
    inspection_service: InspectionService = request.app.state.inspection_service
    storage_service: LocalMediaStorageService = request.app.state.storage_service

    inspection = inspection_service.get_by_id(db, inspection_id)
    media_path = storage_service.load_existing_path(inspection.stored_filename)
    return storage_service.build_media_response(
        file_path=media_path,
        content_type=inspection.content_type,
        range_header=range_header,
    )


@router.get("/{inspection_id}/wagons", response_model=list[WagonResultResponse])
def list_wagons(
    inspection_id: UUID,
    request: Request,
    db: Session = Depends(get_db_session),
) -> list[WagonResultResponse]:
    """List consolidated wagon results for the latest completed job."""
    results_service: ResultsService = request.app.state.results_service
    return [
        results_service.to_wagon_response(item)
        for item in results_service.list_wagons_for_inspection(db, inspection_id)
    ]


@router.get("/{inspection_id}/wagons/{wagon_id}", response_model=WagonResultResponse)
def get_wagon(
    inspection_id: UUID,
    wagon_id: UUID,
    request: Request,
    db: Session = Depends(get_db_session),
) -> WagonResultResponse:
    """Fetch a single consolidated wagon result owned by the inspection."""
    results_service: ResultsService = request.app.state.results_service
    wagon = results_service.get_wagon_for_inspection(db, inspection_id, wagon_id)
    return results_service.to_wagon_response(wagon)


@router.get("/{inspection_id}/wagons/{wagon_id}/observations", response_model=list[WagonObservationResponse])
def list_observations(
    inspection_id: UUID,
    wagon_id: UUID,
    request: Request,
    db: Session = Depends(get_db_session),
) -> list[WagonObservationResponse]:
    """List per-frame observations for a consolidated wagon result."""
    results_service: ResultsService = request.app.state.results_service
    return [
        results_service.to_observation_response(item)
        for item in results_service.list_observations_for_wagon(db, inspection_id, wagon_id)
    ]
