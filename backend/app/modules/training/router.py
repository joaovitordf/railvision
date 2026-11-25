"""FastAPI router for /api/v1/training incremental detector training & live curves."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Body, Request, status

from app.modules.results.schemas import ModelCheckpointDto
from app.modules.training.schemas import (
    RegisterTrainingCheckpointRequest,
    StartTrainingRequest,
    TrainingOverviewDto,
    TrainingRunDetailDto,
    TrainingRunSummaryDto,
)
from app.modules.training.service import TrainingService

router = APIRouter(prefix="/api/v1/training", tags=["training"])


@router.get("/overview", response_model=TrainingOverviewDto)
def get_training_overview(request: Request) -> TrainingOverviewDto:
    """Return sequential model lineage, datasets, baseline reference, GPU readiness, and run history."""
    service: TrainingService = request.app.state.training_service
    return service.get_overview()


@router.get("/runs", response_model=list[TrainingRunSummaryDto])
def list_training_runs(request: Request) -> list[TrainingRunSummaryDto]:
    """List managed training runs newest first."""
    service: TrainingService = request.app.state.training_service
    return service.list_runs()


@router.get("/runs/{run_id}", response_model=TrainingRunDetailDto)
def get_training_run(run_id: UUID, request: Request) -> TrainingRunDetailDto:
    """Retrieve live epoch history, log tail, and validation calibration curve for a training run."""
    service: TrainingService = request.app.state.training_service
    return service.get_run(run_id)


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED, response_model=TrainingRunSummaryDto)
def start_training_run(payload: StartTrainingRequest, request: Request) -> TrainingRunSummaryDto:
    """Start a new incremental versioned training run on the local CUDA GPU."""
    service: TrainingService = request.app.state.training_service
    return service.start_run(payload)


@router.post("/runs/{run_id}/stop", response_model=TrainingRunSummaryDto)
def stop_training_run(run_id: UUID, request: Request) -> TrainingRunSummaryDto:
    """Stop an active training run safely and preserve partial checkpoints/metrics."""
    service: TrainingService = request.app.state.training_service
    return service.stop_run(run_id)


@router.post(
    "/runs/{run_id}/register",
    status_code=status.HTTP_201_CREATED,
    response_model=ModelCheckpointDto,
)
def register_training_checkpoint(
    run_id: UUID,
    request: Request,
    payload: RegisterTrainingCheckpointRequest | None = Body(default=None),
) -> dict[str, Any]:
    """Register the run's best.pt checkpoint into artifacts/models/registry.json."""
    service: TrainingService = request.app.state.training_service
    return service.register_run_checkpoint(run_id, payload)
