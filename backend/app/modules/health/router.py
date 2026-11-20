"""Health and operational readiness endpoints including Spring Actuator compatibility."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import torch

from app.core.config import Settings
from app.modules.jobs.engine import InspectionInferenceEngine
from app.modules.media.storage import LocalMediaStorageService

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    """FastAPI backend readiness endpoint."""
    storage_service: LocalMediaStorageService = request.app.state.storage_service
    settings: Settings = request.app.state.settings
    return {
        "status": "ok",
        "stage": "M4",
        "backend": "python-fastapi",
        "storageWritable": storage_service.is_writable(),
        "defaultRuntime": settings.default_runtime,
        "cudaAvailable": bool(torch.cuda.is_available()),
    }


@router.get("/actuator/health")
def actuator_health(request: Request) -> JSONResponse:
    """Actuator-compatible health probe preserved for proxy and startup scripts."""
    storage_service: LocalMediaStorageService = request.app.state.storage_service
    inference_engine: InspectionInferenceEngine = request.app.state.inference_engine
    settings: Settings = request.app.state.settings

    storage_writable = storage_service.is_writable()
    overall = "UP" if storage_writable else "DOWN"
    status_code = 200 if storage_writable else 503

    payload = {
        "status": overall,
        "components": {
            "railvision": {
                "status": overall,
                "details": {
                    "backend": "python-fastapi",
                    "storageWritable": storage_writable,
                    "inferenceEngine": inference_engine.engine_name,
                    "demoSimulation": inference_engine.is_demo_simulation,
                    "defaultRuntime": settings.default_runtime,
                    "cudaAvailable": bool(torch.cuda.is_available()),
                },
            }
        },
    }
    return JSONResponse(status_code=status_code, content=payload)
