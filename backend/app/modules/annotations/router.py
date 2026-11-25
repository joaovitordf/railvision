"""FastAPI router for /api/v1/annotations Annotation Studio MVP endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse

from app.modules.annotations.schemas import (
    AnnotationItemDetailDto,
    AnnotationItemSummaryDto,
    AnnotationProjectDetailDto,
    AnnotationProjectSummaryDto,
    AnnotationVideoDto,
    CreateAnnotationProjectRequest,
    ExportYoloRequest,
    ExtractFramesRequest,
    ImportCurationResponse,
    PreannotateBatchResponse,
    PreannotateRequest,
    PreviewFramesRequest,
    PreviewFramesResponse,
    UpdateItemAnnotationsRequest,
)
from app.modules.annotations.service import AnnotationStudioService

router = APIRouter(prefix="/api/v1/annotations", tags=["annotations"])


@router.get("/projects", response_model=list[AnnotationProjectSummaryDto])
def list_projects(request: Request) -> list[AnnotationProjectSummaryDto]:
    """List isolated Annotation Studio projects."""
    service: AnnotationStudioService = request.app.state.annotations_service
    return service.list_projects()


@router.post("/projects", status_code=status.HTTP_201_CREATED, response_model=AnnotationProjectSummaryDto)
def create_project(
    payload: CreateAnnotationProjectRequest,
    request: Request,
) -> AnnotationProjectSummaryDto:
    """Create a new isolated Annotation Studio project."""
    service: AnnotationStudioService = request.app.state.annotations_service
    return service.create_project(payload)


@router.get("/projects/{project_id}", response_model=AnnotationProjectDetailDto)
def get_project(
    project_id: UUID,
    request: Request,
    status_filter: str | None = Query(default=None, alias="status"),
    search: str | None = Query(default=None),
) -> AnnotationProjectDetailDto:
    """Get project metadata, filtered image items, and registered videos."""
    service: AnnotationStudioService = request.app.state.annotations_service
    return service.get_project(project_id, status_filter=status_filter, search=search)


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: UUID, request: Request) -> Response:
    """Delete an isolated Annotation Studio project workspace."""
    service: AnnotationStudioService = request.app.state.annotations_service
    service.delete_project(project_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/projects/{project_id}/images",
    status_code=status.HTTP_201_CREATED,
    response_model=list[AnnotationItemSummaryDto],
)
def upload_images(
    project_id: UUID,
    request: Request,
    files: list[UploadFile] = File(...),
    source_origin: str | None = Form(default=None, alias="sourceOrigin"),
    campaign_id: str | None = Form(default=None, alias="campaignId"),
) -> list[AnnotationItemSummaryDto]:
    """Upload one or more clean images into the annotation project."""
    service: AnnotationStudioService = request.app.state.annotations_service
    return service.upload_images(
        project_id,
        files,
        source_origin=source_origin,
        campaign_id=campaign_id,
    )


@router.post(
    "/projects/{project_id}/videos",
    status_code=status.HTTP_201_CREATED,
    response_model=AnnotationVideoDto,
)
def upload_video(
    project_id: UUID,
    request: Request,
    file: UploadFile = File(...),
    source_origin: str | None = Form(default=None, alias="sourceOrigin"),
    campaign_id: str | None = Form(default=None, alias="campaignId"),
) -> AnnotationVideoDto:
    """Upload and probe a video for configurable frame extraction."""
    service: AnnotationStudioService = request.app.state.annotations_service
    return service.upload_video(
        project_id,
        file,
        source_origin=source_origin,
        campaign_id=campaign_id,
    )


@router.post("/projects/{project_id}/videos/{video_id}/preview-frames", response_model=PreviewFramesResponse)
def preview_video_frames(
    project_id: UUID,
    video_id: UUID,
    payload: PreviewFramesRequest,
    request: Request,
) -> PreviewFramesResponse:
    """Extract candidate deduplicated thumbnails from a project video for interactive selection."""
    service: AnnotationStudioService = request.app.state.annotations_service
    return service.preview_video_frames(project_id, video_id, payload)


@router.get("/projects/{project_id}/videos/{video_id}/preview/{frame_index}")
def get_preview_frame_thumbnail(
    project_id: UUID,
    video_id: UUID,
    frame_index: int,
    request: Request,
) -> Response:
    """Serve cached preview thumbnail for a video frame candidate."""
    service: AnnotationStudioService = request.app.state.annotations_service
    path = service.load_preview_thumbnail(project_id, video_id, frame_index)
    return FileResponse(path, media_type="image/jpeg")


@router.post(
    "/projects/{project_id}/videos/{video_id}/extract-frames",
    status_code=status.HTTP_201_CREATED,
    response_model=list[AnnotationItemSummaryDto],
)
def extract_video_frames(
    project_id: UUID,
    video_id: UUID,
    payload: ExtractFramesRequest,
    request: Request,
) -> list[AnnotationItemSummaryDto]:
    """Commit selected or sampled frames at original resolution into the project."""
    service: AnnotationStudioService = request.app.state.annotations_service
    return service.extract_video_frames(project_id, video_id, payload)


@router.post("/projects/{project_id}/import-curation-inbox", response_model=ImportCurationResponse)
def import_curation_inbox(project_id: UUID, request: Request) -> ImportCurationResponse:
    """Import non-duplicate frames queued from Vision Lab curation-inbox."""
    service: AnnotationStudioService = request.app.state.annotations_service
    return service.import_curation_inbox(project_id)


@router.get("/projects/{project_id}/items/{item_id}", response_model=AnnotationItemDetailDto)
def get_item(project_id: UUID, item_id: UUID, request: Request) -> AnnotationItemDetailDto:
    """Return full annotation item boxes and audit provenance."""
    service: AnnotationStudioService = request.app.state.annotations_service
    return service.get_item(project_id, item_id)


@router.get("/projects/{project_id}/items/{item_id}/image")
def get_item_image(
    project_id: UUID,
    item_id: UUID,
    request: Request,
    thumbnail: bool = Query(default=False),
) -> Response:
    """Serve clean full-resolution image or thumbnail for an annotation item."""
    service: AnnotationStudioService = request.app.state.annotations_service
    path = service.load_item_image_path(project_id, item_id, thumbnail=thumbnail)
    return FileResponse(path, media_type="image/jpeg")


@router.put("/projects/{project_id}/items/{item_id}/annotations", response_model=AnnotationItemDetailDto)
def update_item_annotations(
    project_id: UUID,
    item_id: UUID,
    payload: UpdateItemAnnotationsRequest,
    request: Request,
) -> AnnotationItemDetailDto:
    """Persist human-reviewed boxes, negative confirmation, and audit state."""
    service: AnnotationStudioService = request.app.state.annotations_service
    return service.update_item_annotations(project_id, item_id, payload)


@router.delete("/projects/{project_id}/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(project_id: UUID, item_id: UUID, request: Request) -> Response:
    """Remove an annotation item from the project workspace."""
    service: AnnotationStudioService = request.app.state.annotations_service
    service.delete_item(project_id, item_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/projects/{project_id}/items/{item_id}/preannotate", response_model=AnnotationItemDetailDto)
def preannotate_item(
    project_id: UUID,
    item_id: UUID,
    payload: PreannotateRequest,
    request: Request,
) -> AnnotationItemDetailDto:
    """Execute local AI detector pre-annotation on a single item."""
    service: AnnotationStudioService = request.app.state.annotations_service
    return service.preannotate_item(project_id, item_id, payload)


@router.post("/projects/{project_id}/preannotate-batch", response_model=PreannotateBatchResponse)
def preannotate_batch(
    project_id: UUID,
    payload: PreannotateRequest,
    request: Request,
) -> PreannotateBatchResponse:
    """Run batch AI pre-annotation on pending items in the project."""
    service: AnnotationStudioService = request.app.state.annotations_service
    return service.preannotate_batch(project_id, payload)


@router.post("/projects/{project_id}/export-yolo")
def export_yolo_zip(
    project_id: UUID,
    payload: ExportYoloRequest,
    request: Request,
) -> Response:
    """Export CVAT YOLO 1.1 .zip archive enforcing human review gate by default."""
    service: AnnotationStudioService = request.app.state.annotations_service
    zip_bytes, filename = service.export_yolo_zip(project_id, payload)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "/projects/{project_id}/import-yolo",
    status_code=status.HTTP_201_CREATED,
    response_model=list[AnnotationItemSummaryDto],
)
def import_yolo_zip(
    project_id: UUID,
    request: Request,
    file: UploadFile = File(...),
) -> list[AnnotationItemSummaryDto]:
    """Import images and YOLO 1.1 labels from a CVAT .zip archive into the project."""
    service: AnnotationStudioService = request.app.state.annotations_service
    return service.import_yolo_zip(project_id, file)
