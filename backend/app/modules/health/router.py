"""Health and operational readiness endpoints including Spring Actuator compatibility."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import sqlalchemy as sa
import torch

from app.core.config import Settings
from app.modules.jobs.engine import InspectionInferenceEngine
from app.modules.media.storage import LocalMediaStorageService

router = APIRouter(tags=["health"])


def _is_db_reachable(request: Request) -> bool:
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        return False
    try:
        with factory() as db:
            db.execute(sa.text("SELECT 1"))
        return True
    except Exception:
        return False


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    """FastAPI backend liveness endpoint."""
    storage_service: LocalMediaStorageService = request.app.state.storage_service
    settings: Settings = request.app.state.settings
    return {
        "status": "ok",
        "stage": "WS1",
        "version": "0.4.0",
        "backend": "python-fastapi",
        "storageWritable": storage_service.is_writable(),
        "defaultRuntime": settings.default_runtime,
        "cudaAvailable": bool(torch.cuda.is_available()),
    }


@router.get("/actuator/health")
def actuator_health(request: Request) -> JSONResponse:
    """Actuator-compatible health probe checking writable storage and PostgreSQL readiness."""
    storage_service: LocalMediaStorageService = request.app.state.storage_service
    inference_engine: InspectionInferenceEngine = request.app.state.inference_engine
    settings: Settings = request.app.state.settings

    storage_writable = storage_service.is_writable()
    db_reachable = _is_db_reachable(request)
    healthy = storage_writable and db_reachable
    overall = "UP" if healthy else "DOWN"
    status_code = 200 if healthy else 503

    payload = {
        "status": overall,
        "components": {
            "railvision": {
                "status": overall,
                "details": {
                    "backend": "python-fastapi",
                    "version": "0.4.0",
                    "storageWritable": storage_writable,
                    "dbReachable": db_reachable,
                    "inferenceEngine": inference_engine.engine_name,
                    "demoSimulation": inference_engine.is_demo_simulation,
                    "defaultRuntime": settings.default_runtime,
                    "cudaAvailable": bool(torch.cuda.is_available()),
                },
            }
        },
    }
    return JSONResponse(status_code=status_code, content=payload)
