"""FastAPI routers for /api/v1/vision-lab and backwards-compatible /api/v1/onnx-spike endpoints."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, File, Header, Query, Request, Response, UploadFile, status

from app.modules.lab.schemas import (
    CurateFrameRequest,
    CurationExportDto,
    ImageRunResultDto,
    RunComparisonDto,
    RunListItemDto,
    RuntimeStatusDto,
    VideoRunStatusDto,
)
from app.modules.lab.service import LabService
from app.modules.media.storage import LocalMediaStorageService


def _build_lab_router(*, prefix: str, tags: list[str], include_in_schema: bool = True) -> APIRouter:
    lab_router = APIRouter(prefix=prefix, tags=tags, include_in_schema=include_in_schema)

    @lab_router.get("/status", response_model=RuntimeStatusDto)
    def get_status(
        request: Request,
        model_id: str | None = Query(default=None, alias="modelId"),
    ) -> RuntimeStatusDto:
        """Return active runtime diagnostics, available Python/ONNX runtimes, and tracking options."""
        lab_service: LabService = request.app.state.lab_service
        return lab_service.get_runtime_status(model_id=model_id)

    @lab_router.post("/analyze-image", response_model=ImageRunResultDto)
    def analyze_image(
        request: Request,
        file: UploadFile = File(...),
        conf: float | None = Query(default=None),
        iou: float | None = Query(default=None),
        runtime: str | None = Query(default=None),
        allow_cpu_fallback: bool = Query(default=False, alias="allowCpuFallback"),
        model_id: str | None = Query(default=None, alias="modelId"),
    ) -> dict[str, Any]:
        """Run local image inference using PyTorch CUDA (default), PyTorch CPU, or ONNX CPU."""
        lab_service: LabService = request.app.state.lab_service
        return lab_service.analyze_image(
            file=file,
            conf=conf,
            iou=iou,
            runtime=runtime,
            allow_cpu_fallback=allow_cpu_fallback,
            model_id=model_id,
        )

    @lab_router.post("/analyze-video", status_code=status.HTTP_202_ACCEPTED, response_model=VideoRunStatusDto)
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
        model_id: str | None = Query(default=None, alias="modelId"),
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
            model_id=model_id,
        )

    @lab_router.get("/runs", response_model=list[RunListItemDto])
    def list_runs(request: Request) -> list[RunListItemDto]:
        """List up to 30 recent lab runs across both Python and preserved Java history."""
        lab_service: LabService = request.app.state.lab_service
        return lab_service.list_runs()

    @lab_router.get("/compare", response_model=RunComparisonDto)
    def compare_runs(
        request: Request,
        run_a: UUID = Query(alias="runA"),
        run_b: UUID = Query(alias="runB"),
    ) -> RunComparisonDto:
        """Compare two completed lab runs side by side."""
        lab_service: LabService = request.app.state.lab_service
        return lab_service.compare_runs(run_a, run_b)

    @lab_router.get("/runs/{run_id}", response_model=ImageRunResultDto | VideoRunStatusDto)
    def get_run(run_id: UUID, request: Request) -> dict[str, Any]:
        """Retrieve full status and frame-level detections for a lab run."""
        lab_service: LabService = request.app.state.lab_service
        return lab_service.get_run(run_id)

    @lab_router.post(
        "/runs/{run_id}/curate-frame",
        status_code=status.HTTP_201_CREATED,
        response_model=CurationExportDto,
    )
    def curate_frame(
        run_id: UUID,
        payload: CurateFrameRequest,
        request: Request,
    ) -> CurationExportDto:
        """Extract clean original frame + YOLO pre-annotation into data/processed/curation-inbox."""
        lab_service: LabService = request.app.state.lab_service
        return lab_service.curate_frame(run_id, payload)

    @lab_router.get("/runs/{run_id}/files/{kind}")
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

    return lab_router


vision_lab_router = _build_lab_router(prefix="/api/v1/vision-lab", tags=["vision-lab"], include_in_schema=True)
router = _build_lab_router(prefix="/api/v1/onnx-spike", tags=["onnx-spike-compat"], include_in_schema=True)
