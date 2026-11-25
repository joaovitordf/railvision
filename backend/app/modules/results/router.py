"""FastAPI router for registered model versions."""

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.modules.results.schemas import ModelCheckpointDto, ModelVersionResponse
from app.modules.results.service import ResultsService
from app.modules.vision.registry import ModelRegistryService

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("/checkpoints", response_model=list[ModelCheckpointDto])
def list_checkpoints(request: Request) -> list[dict[str, object]]:
    """List registered detection checkpoints from the declarative Model Registry."""
    registry: ModelRegistryService = request.app.state.model_registry
    return [cp.to_dict() for cp in registry.list_checkpoints()]


@router.get("", response_model=list[ModelVersionResponse])
def list_models(
    request: Request,
    include_placeholders: bool = Query(default=False, alias="includePlaceholders"),
    db: Session = Depends(get_db_session),
) -> list[ModelVersionResponse]:
    """List persisted inference and tracking model versions (excluding legacy placeholders by default)."""
    results_service: ResultsService = request.app.state.results_service
    return [
        results_service.to_model_version_response(item)
        for item in results_service.list_model_versions(db, include_placeholders=include_placeholders)
    ]
