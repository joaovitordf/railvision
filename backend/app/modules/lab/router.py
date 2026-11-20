"""FastAPI router preserving /api/v1/onnx-spike viewer and streaming endpoints."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, File, Header, Query, Request, Response, UploadFile, status

from app.modules.lab.schemas import RunListItemDto, RuntimeStatusDto, VideoRunStatusDto
from app.modules.lab.service import LabService
from app.modules.media.storage import LocalMediaStorageService

router = APIRouter(prefix="/api/v1/onnx-spike", tags=["vision-lab"])


@router.get("/status", response_model=RuntimeStatusDto)
def get_status(request: Request) -> RuntimeStatusDto:
    """Return active runtime diagnostics, available Python/ONNX runtimes, and tracking options."""
    lab_service: LabService = request.app.state.lab_service
    return lab_service.get_runtime_status()


@router.post("/analyze-image")
def analyze_image(
    request: Request,
    file: UploadFile = File(...),
    conf: float | None = Query(default=None),
    iou: float | None = Query(default=None),
    runtime: str | None = Query(default=None),
    allow_cpu_fallback: bool = Query(default=False, alias="allowCpuFallback"),
) -> dict[str, Any]:
    """Run local image inference using PyTorch CUDA (default), PyTorch CPU, or ONNX CPU."""
    lab_service: LabService = request.app.state.lab_service
    return lab_service.analyze_image(
        file=file,
        conf=conf,
        iou=iou,
        runtime=runtime,
        allow_cpu_fallback=allow_cpu_fallback,
    )


@router.post("/analyze-video", status_code=status.HTTP_202_ACCEPTED, response_model=VideoRunStatusDto)
def analyze_video(
    request: Request,
    file: UploadFile = File(...),
    conf: float | None = Query(default=None),
    iou: float | None = Query(default=None),
    sample_fps: float | None = Query(default=None, alias="sampleFps"),
    runtime: str | None = Query(default=None),
    allow_cpu_fallback: bool = Query(default=False, alias="allowCpuFallback"),
    output_resolution: str = Query(default="1080p", alias="outputResolution"),
    enable_tracking: bool = Query(default=False, alias="enableTracking"),
) -> VideoRunStatusDto:
    """Submit asynchronous local video inference/tracking returning a pollable run ID."""
    lab_service: LabService = request.app.state.lab_service
    return lab_service.submit_video_analysis(
        file=file,
        conf=conf,
        iou=iou,
        sample_fps=sample_fps,
        runtime=runtime,
        allow_cpu_fallback=allow_cpu_fallback,
        output_resolution=output_resolution,
        enable_tracking=enable_tracking,
    )


@router.get("/runs", response_model=list[RunListItemDto])
def list_runs(request: Request) -> list[RunListItemDto]:
    """List up to 30 recent lab runs across both Python and preserved Java history."""
    lab_service: LabService = request.app.state.lab_service
    return lab_service.list_runs()


@router.get("/runs/{run_id}")
def get_run(run_id: UUID, request: Request) -> dict[str, Any]:
    """Retrieve full status and frame-level detections for a lab run."""
    lab_service: LabService = request.app.state.lab_service
    return lab_service.get_run(run_id)


@router.get("/runs/{run_id}/files/{kind}")
def stream_run_file(
    run_id: UUID,
    kind: str,
    request: Request,
    download: bool = Query(default=False),
    range_header: str | None = Header(default=None, alias="Range"),
) -> Response:
    """Stream or download original/annotated run files with HTTP byte-range support."""
    lab_service: LabService = request.app.state.lab_service
    storage_service: LocalMediaStorageService = request.app.state.storage_service
    stored = lab_service.load_run_file(run_id, kind)
    return storage_service.build_media_response(
        file_path=stored.file_path,
        content_type=stored.content_type,
        range_header=range_header if not download else None,
        download_filename=stored.download_filename if download else None,
    )
