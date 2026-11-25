"""Annotation Studio persistence, frame extraction, deduplication, AI pre-annotation, and YOLO 1.1 I/O."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import threading
from typing import Any
from uuid import UUID, uuid4
import zipfile

import av
import cv2
from fastapi import UploadFile
import numpy as np

from app.core.config import Settings
from app.core.exceptions import (
    ConflictException,
    InvalidMediaException,
    MediaTooLargeException,
    ResourceNotFoundException,
)
from app.core.paths import REPO_ROOT
from app.modules.annotations.schemas import (
    AnnotationBoxDto,
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
    PreviewFrameCandidateDto,
    PreviewFramesRequest,
    PreviewFramesResponse,
    UpdateItemAnnotationsRequest,
)
from app.modules.media.probe import probe_video_metadata
from app.modules.media.validator import (
    ALLOWED_VIDEO_EXTENSIONS,
    extract_extension,
    matches_container_signature,
    sanitize_original_filename,
)
from app.modules.vision.runtimes import VisionRuntimeManager
from app.modules.vision.video_pipeline import iter_decoded_video_frames

IMAGE_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "bmp", "webp"})
REVIEWED_STATUSES = frozenset({"reviewed", "approved"})
RESERVED_PREFIXES = ("vid_01", "vid_09", "vid_16", "camp_vfli_cosne")
DEFAULT_CLASSES = ["open_top_freight_wagon"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_filename_stem(name: str, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in Path(name).stem.strip())
    cleaned = cleaned.strip("._") or fallback
    return cleaned[:96]


def _encode_jpeg(bgr: np.ndarray, *, max_width: int | None = None, quality: int = 92) -> bytes:
    target = bgr
    if max_width is not None and bgr.shape[1] > max_width:
        scale = max_width / float(bgr.shape[1])
        target_h = max(2, int(round(bgr.shape[0] * scale)))
        target = cv2.resize(bgr, (max_width, target_h), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", target, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Falha ao codificar imagem JPEG.")
    return encoded.tobytes()


def _grayscale_signature(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)


class AnnotationStudioService:
    """Isolated workspace manager for native image annotation, frame extraction, and CVAT YOLO 1.1 I/O."""

    def __init__(
        self,
        *,
        settings: Settings,
        runtime_manager: VisionRuntimeManager,
    ) -> None:
        self._settings = settings
        self._runtime_manager = runtime_manager
        self._root = settings.annotation_studio_path.resolve()
        self._lock = threading.RLock()
        self._max_upload_bytes = settings.upload_max_size_mb * 1024 * 1024

    def _project_dir(self, project_id: UUID) -> Path:
        candidate = (self._root / str(project_id)).resolve()
        if candidate.parent != self._root:
            raise ValueError("Identificador de projeto inválido.")
        return candidate

    def _project_json_path(self, project_id: UUID) -> Path:
        return self._project_dir(project_id) / "project.json"

    def _load_project(self, project_id: UUID) -> dict[str, Any]:
        meta_path = self._project_json_path(project_id)
        if not meta_path.is_file():
            raise ResourceNotFoundException(f"Projeto de anotação não encontrado: {project_id}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _save_project(self, project_id: UUID, data: dict[str, Any]) -> None:
        proj_dir = self._project_dir(project_id)
        proj_dir.mkdir(parents=True, exist_ok=True)
        data["updatedAt"] = _utcnow().isoformat()
        target = proj_dir / "project.json"
        tmp = proj_dir / "project.tmp.json"
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)

    @staticmethod
    def _is_reserved_benchmark_media(sha256_hex: str, filename: str | None) -> bool:
        low = (filename or "").lower()
        if any(prefix in low for prefix in RESERVED_PREFIXES):
            return True
        pub_manifest = REPO_ROOT / "specs" / "data-manifests" / "public-videos-manifest.json"
        if pub_manifest.is_file():
            try:
                manifest = json.loads(pub_manifest.read_text(encoding="utf-8"))
                for rec in manifest.get("videos") or []:
                    vid_id = str(rec.get("id") or "").lower()
                    if any(vid_id.startswith(prefix) for prefix in RESERVED_PREFIXES):
                        if str(rec.get("sha256") or "") == sha256_hex:
                            return True
            except Exception:
                pass
        return False

    def _to_project_summary(self, data: dict[str, Any]) -> AnnotationProjectSummaryDto:
        items = list(data.get("items") or [])
        videos = list(data.get("videos") or [])
        status_counts: dict[str, int] = {
            "pending": 0,
            "pre_annotated": 0,
            "in_review": 0,
            "reviewed": 0,
            "approved": 0,
            "rejected": 0,
        }
        reviewed_count = 0
        approved_count = 0
        for item in items:
            st = str(item.get("status") or "pending")
            status_counts[st] = status_counts.get(st, 0) + 1
            if item.get("humanReviewed") or st in REVIEWED_STATUSES:
                reviewed_count += 1
            if st == "approved":
                approved_count += 1

        return AnnotationProjectSummaryDto(
            id=UUID(str(data["id"])),
            name=str(data["name"]),
            campaign_id=str(data["campaignId"]),
            source_origin=str(data.get("sourceOrigin") or "proprio"),
            description=data.get("description"),
            classes=list(data.get("classes") or DEFAULT_CLASSES),
            item_count=len(items),
            video_count=len(videos),
            status_counts=status_counts,
            reviewed_count=reviewed_count,
            approved_count=approved_count,
            created_at=datetime.fromisoformat(str(data["createdAt"])),
            updated_at=datetime.fromisoformat(str(data["updatedAt"])),
        )

    @staticmethod
    def _to_video_dto(rec: dict[str, Any]) -> AnnotationVideoDto:
        return AnnotationVideoDto(
            id=UUID(str(rec["id"])),
            filename=str(rec["filename"]),
            source_origin=str(rec.get("sourceOrigin") or "proprio"),
            campaign_id=str(rec.get("campaignId") or ""),
            sha256=str(rec["sha256"]),
            width_px=int(rec["widthPx"]),
            height_px=int(rec["heightPx"]),
            fps=float(rec["fps"]),
            duration_ms=int(rec["durationMs"]),
            codec=str(rec["codec"]),
            estimated_frames=int(rec["estimatedFrames"]),
            reserved_benchmark_warning=bool(rec.get("reservedBenchmarkWarning", False)),
            imported_at=datetime.fromisoformat(str(rec["importedAt"])),
        )

    @staticmethod
    def _to_item_summary(project_id: UUID, rec: dict[str, Any]) -> AnnotationItemSummaryDto:
        item_id = UUID(str(rec["id"]))
        boxes = list(rec.get("boxes") or [])
        ai_count = sum(1 for b in boxes if b.get("origin") == "ai_suggestion")
        return AnnotationItemSummaryDto(
            id=item_id,
            project_id=project_id,
            filename=str(rec["filename"]),
            source_origin=str(rec.get("sourceOrigin") or "proprio"),
            campaign_id=str(rec.get("campaignId") or ""),
            source_sha256=str(rec["sourceSha256"]),
            width_px=int(rec["widthPx"]),
            height_px=int(rec["heightPx"]),
            status=rec.get("status", "pending"),
            human_reviewed=bool(rec.get("humanReviewed", False)),
            no_objects_confirmed=bool(rec.get("noObjectsConfirmed", False)),
            box_count=len(boxes),
            ai_suggestion_count=ai_count,
            reserved_benchmark_warning=bool(rec.get("reservedBenchmarkWarning", False)),
            source_video_id=UUID(str(rec["sourceVideoId"])) if rec.get("sourceVideoId") else None,
            source_video_filename=rec.get("sourceVideoFilename"),
            frame_index=rec.get("frameIndex"),
            timestamp_ms=rec.get("timestampMs"),
            image_url=f"/api/v1/annotations/projects/{project_id}/items/{item_id}/image",
            thumbnail_url=f"/api/v1/annotations/projects/{project_id}/items/{item_id}/image?thumbnail=true",
            created_at=datetime.fromisoformat(str(rec["createdAt"])),
            updated_at=datetime.fromisoformat(str(rec["updatedAt"])),
            reviewed_at=datetime.fromisoformat(str(rec["reviewedAt"])) if rec.get("reviewedAt") else None,
            approved_at=datetime.fromisoformat(str(rec["approvedAt"])) if rec.get("approvedAt") else None,
        )

    @classmethod
    def _to_item_detail(cls, project_id: UUID, rec: dict[str, Any]) -> AnnotationItemDetailDto:
        summary = cls._to_item_summary(project_id, rec)
        boxes = [AnnotationBoxDto.model_validate(b) for b in (rec.get("boxes") or [])]
        return AnnotationItemDetailDto(
            **summary.model_dump(),
            boxes=boxes,
            review_notes=rec.get("reviewNotes"),
            false_positives_removed=int(rec.get("falsePositivesRemoved") or 0),
            false_negatives_added=int(rec.get("falseNegativesAdded") or 0),
            preannotation_model_id=rec.get("preannotationModelId"),
            preannotation_model_sha256=rec.get("preannotationModelSha256"),
            preannotation_conf=rec.get("preannotationConf"),
            preannotation_iou=rec.get("preannotationIou"),
        )

    def list_projects(self) -> list[AnnotationProjectSummaryDto]:
        """List all isolated Annotation Studio projects sorted newest-updated first."""
        with self._lock:
            if not self._root.is_dir():
                return []
            summaries: list[AnnotationProjectSummaryDto] = []
            for entry in self._root.iterdir():
                if not entry.is_dir():
                    continue
                meta = entry / "project.json"
                if not meta.is_file():
                    continue
                try:
                    data = json.loads(meta.read_text(encoding="utf-8"))
                    summaries.append(self._to_project_summary(data))
                except Exception:
                    continue
            summaries.sort(key=lambda item: item.updated_at, reverse=True)
            return summaries

    def create_project(self, payload: CreateAnnotationProjectRequest) -> AnnotationProjectSummaryDto:
        """Create a new isolated Annotation Studio project."""
        clean_classes = [c.strip() for c in payload.classes if c.strip()] or list(DEFAULT_CLASSES)
        project_id = uuid4()
        now = _utcnow().isoformat()
        data = {
            "id": str(project_id),
            "name": payload.name.strip(),
            "campaignId": payload.campaign_id.strip(),
            "sourceOrigin": payload.source_origin.strip(),
            "description": payload.description.strip() if payload.description else None,
            "classes": clean_classes,
            "videos": [],
            "items": [],
            "createdAt": now,
            "updatedAt": now,
        }
        with self._lock:
            proj_dir = self._project_dir(project_id)
            for sub in ("images", "thumbnails", "videos", "exports"):
                (proj_dir / sub).mkdir(parents=True, exist_ok=True)
            self._save_project(project_id, data)
            return self._to_project_summary(data)

    def get_project(
        self,
        project_id: UUID,
        *,
        status_filter: str | None = None,
        search: str | None = None,
    ) -> AnnotationProjectDetailDto:
        """Return project metadata, optional filtered item list, and imported videos."""
        with self._lock:
            data = self._load_project(project_id)
            summary = self._to_project_summary(data)
            query = (search or "").strip().lower()
            items: list[AnnotationItemSummaryDto] = []
            for rec in data.get("items") or []:
                if status_filter and status_filter != "ALL" and rec.get("status") != status_filter:
                    continue
                if query:
                    haystack = " ".join(
                        [
                            str(rec.get("filename") or ""),
                            str(rec.get("campaignId") or ""),
                            str(rec.get("sourceOrigin") or ""),
                            str(rec.get("sourceVideoFilename") or ""),
                        ]
                    ).lower()
                    if query not in haystack:
                        continue
                items.append(self._to_item_summary(project_id, rec))
            videos = [self._to_video_dto(v) for v in (data.get("videos") or [])]
            return AnnotationProjectDetailDto(
                **summary.model_dump(),
                items=items,
                videos=videos,
            )

    def delete_project(self, project_id: UUID) -> None:
        """Delete an entire Annotation Studio project directory (never touching official datasets)."""
        with self._lock:
            proj_dir = self._project_dir(project_id)
            if not (proj_dir / "project.json").is_file():
                raise ResourceNotFoundException(f"Projeto de anotação não encontrado: {project_id}")
            shutil.rmtree(proj_dir, ignore_errors=False)

    def _register_image_bgr(
        self,
        *,
        project_id: UUID,
        data: dict[str, Any],
        bgr: np.ndarray,
        raw_bytes: bytes,
        filename: str,
        source_origin: str,
        campaign_id: str,
        status: str = "pending",
        boxes: list[dict[str, Any]] | None = None,
        source_video_id: str | None = None,
        source_video_filename: str | None = None,
        frame_index: int | None = None,
        timestamp_ms: int | None = None,
        preannotation_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        item_id = uuid4()
        sha = hashlib.sha256(raw_bytes).hexdigest()
        reserved_warning = self._is_reserved_benchmark_media(sha, filename)
        proj_dir = self._project_dir(project_id)
        (proj_dir / "images").mkdir(parents=True, exist_ok=True)
        (proj_dir / "thumbnails").mkdir(parents=True, exist_ok=True)

        img_rel = f"images/{item_id}.jpg"
        thumb_rel = f"thumbnails/{item_id}.jpg"
        ext = extract_extension(filename)
        if ext in {"jpg", "jpeg"} and raw_bytes[:2] == b"\xff\xd8":
            (proj_dir / img_rel).write_bytes(raw_bytes)
        else:
            (proj_dir / img_rel).write_bytes(_encode_jpeg(bgr, quality=95))
        (proj_dir / thumb_rel).write_bytes(_encode_jpeg(bgr, max_width=360, quality=84))

        now = _utcnow().isoformat()
        rec: dict[str, Any] = {
            "id": str(item_id),
            "filename": filename,
            "imageRelPath": img_rel,
            "thumbRelPath": thumb_rel,
            "sourceOrigin": source_origin,
            "campaignId": campaign_id,
            "sourceSha256": sha,
            "widthPx": int(bgr.shape[1]),
            "heightPx": int(bgr.shape[0]),
            "status": status,
            "humanReviewed": False,
            "noObjectsConfirmed": False,
            "boxes": list(boxes or []),
            "reservedBenchmarkWarning": reserved_warning,
            "sourceVideoId": source_video_id,
            "sourceVideoFilename": source_video_filename,
            "frameIndex": frame_index,
            "timestampMs": timestamp_ms,
            "reviewNotes": None,
            "falsePositivesRemoved": 0,
            "falseNegativesAdded": 0,
            "createdAt": now,
            "updatedAt": now,
            "reviewedAt": None,
            "approvedAt": None,
        }
        if preannotation_meta:
            rec.update(preannotation_meta)
        data.setdefault("items", []).append(rec)
        return rec

    def upload_images(
        self,
        project_id: UUID,
        files: list[UploadFile],
        *,
        source_origin: str | None = None,
        campaign_id: str | None = None,
    ) -> list[AnnotationItemSummaryDto]:
        """Validate and import one or more clean images into the annotation project."""
        if not files:
            raise InvalidMediaException("MISSING_FILES", "Envie ao menos uma imagem.")
        with self._lock:
            data = self._load_project(project_id)
            origin = (source_origin or data.get("sourceOrigin") or "proprio").strip()
            campaign = (campaign_id or data.get("campaignId") or "default").strip()
            created: list[AnnotationItemSummaryDto] = []
            for upload in files:
                filename = sanitize_original_filename(upload.filename)
                ext = extract_extension(filename)
                if ext not in IMAGE_EXTENSIONS:
                    raise InvalidMediaException(
                        "UNSUPPORTED_IMAGE_FORMAT",
                        f"Formato de imagem não suportado em '{filename}'. Envie JPG, PNG, BMP ou WEBP.",
                    )
                raw = upload.file.read(self._max_upload_bytes + 1)
                if not raw:
                    raise InvalidMediaException("EMPTY_FILE", f"O arquivo '{filename}' está vazio.")
                if len(raw) > self._max_upload_bytes:
                    raise MediaTooLargeException(
                        f"A imagem '{filename}' excede o limite de {self._settings.upload_max_size_mb} MB."
                    )
                arr = np.frombuffer(raw, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None or bgr.size == 0:
                    raise InvalidMediaException(
                        "INVALID_IMAGE_CONTENT",
                        f"Não foi possível decodificar a imagem '{filename}'.",
                    )
                rec = self._register_image_bgr(
                    project_id=project_id,
                    data=data,
                    bgr=bgr,
                    raw_bytes=raw,
                    filename=filename,
                    source_origin=origin,
                    campaign_id=campaign,
                )
                created.append(self._to_item_summary(project_id, rec))
            self._save_project(project_id, data)
            return created

    def upload_video(
        self,
        project_id: UUID,
        file: UploadFile,
        *,
        source_origin: str | None = None,
        campaign_id: str | None = None,
    ) -> AnnotationVideoDto:
        """Validate, probe, and persist a video inside the isolated project for frame extraction."""
        filename = sanitize_original_filename(file.filename)
        ext = extract_extension(filename)
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise InvalidMediaException(
                "UNSUPPORTED_VIDEO_FORMAT",
                "Formato de vídeo não suportado. Envie MP4, MOV, MKV, WEBM, AVI ou OGV.",
            )

        with self._lock:
            data = self._load_project(project_id)
            proj_dir = self._project_dir(project_id)
            videos_dir = proj_dir / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)
            video_id = uuid4()
            rel_path = f"videos/{video_id}.{ext}"
            dest = proj_dir / rel_path

            hasher = hashlib.sha256()
            total_written = 0
            try:
                file.file.seek(0)
                with dest.open("wb") as out:
                    while True:
                        chunk = file.file.read(1024 * 1024)
                        if not chunk:
                            break
                        total_written += len(chunk)
                        if total_written > self._max_upload_bytes:
                            raise MediaTooLargeException(
                                f"O vídeo excede o limite de {self._settings.upload_max_size_mb} MB."
                            )
                        hasher.update(chunk)
                        out.write(chunk)
                if total_written <= 0:
                    raise InvalidMediaException("EMPTY_FILE", "Arquivo de vídeo vazio.")
                with dest.open("rb") as handle:
                    header = handle.read(16)
                if not matches_container_signature(ext, header):
                    raise InvalidMediaException(
                        "INVALID_VIDEO_SIGNATURE",
                        f"Assinatura binária inválida para container .{ext}.",
                    )
                meta = probe_video_metadata(dest)
            except (InvalidMediaException, MediaTooLargeException):
                dest.unlink(missing_ok=True)
                raise
            except Exception as exc:
                dest.unlink(missing_ok=True)
                raise InvalidMediaException("CORRUPT_VIDEO", f"Falha ao abrir vídeo: {exc}") from exc

            sha = hasher.hexdigest()
            rec = {
                "id": str(video_id),
                "filename": filename,
                "relPath": rel_path,
                "sourceOrigin": (source_origin or data.get("sourceOrigin") or "proprio").strip(),
                "campaignId": (campaign_id or data.get("campaignId") or "default").strip(),
                "sha256": sha,
                "widthPx": meta.width_px,
                "heightPx": meta.height_px,
                "fps": round(meta.fps, 2),
                "durationMs": meta.duration_ms,
                "codec": meta.codec,
                "estimatedFrames": meta.estimated_frames,
                "reservedBenchmarkWarning": self._is_reserved_benchmark_media(sha, filename),
                "importedAt": _utcnow().isoformat(),
            }
            data.setdefault("videos", []).append(rec)
            self._save_project(project_id, data)
            return self._to_video_dto(rec)

    def _find_video(self, data: dict[str, Any], video_id: UUID) -> dict[str, Any]:
        vid_str = str(video_id)
        for rec in data.get("videos") or []:
            if str(rec.get("id")) == vid_str:
                return rec
        raise ResourceNotFoundException(f"Vídeo não encontrado no projeto: {video_id}")

    def preview_video_frames(
        self,
        project_id: UUID,
        video_id: UUID,
        payload: PreviewFramesRequest,
    ) -> PreviewFramesResponse:
        """Scan a project video at sample_fps, deduplicate visually similar frames, and cache candidate thumbnails."""
        with self._lock:
            data = self._load_project(project_id)
            vid = self._find_video(data, video_id)
            video_path = self._project_dir(project_id) / str(vid["relPath"])
            if not video_path.is_file():
                raise ResourceNotFoundException("Arquivo de vídeo ausente no workspace.")
            preview_dir = self._project_dir(project_id) / "videos" / f"previews_{video_id}"
            shutil.rmtree(preview_dir, ignore_errors=True)
            preview_dir.mkdir(parents=True, exist_ok=True)

        meta = probe_video_metadata(video_path)
        sample_interval_ms = 1000.0 / float(payload.sample_fps)
        next_sample_ms: float | None = None
        last_sig: np.ndarray | None = None
        candidates: list[PreviewFrameCandidateDto] = []
        scanned = 0
        skipped_similar = 0
        raw_idx = 0

        with av.open(str(video_path)) as container:
            in_stream = container.streams.video[0]
            for frame in iter_decoded_video_frames(container):
                if frame.pts is not None and frame.time_base is not None:
                    ts_ms = max(0, int(round(float(frame.pts * frame.time_base) * 1000.0)))
                elif in_stream.time_base is not None and frame.pts is not None:
                    ts_ms = max(0, int(round(float(frame.pts * in_stream.time_base) * 1000.0)))
                else:
                    ts_ms = int(round((raw_idx * 1000.0) / max(1.0, meta.fps)))
                cur_idx = raw_idx
                raw_idx += 1

                if next_sample_ms is None:
                    next_sample_ms = float(ts_ms)
                if (ts_ms + 1.0) < next_sample_ms:
                    continue
                while next_sample_ms <= (ts_ms + 1.0):
                    next_sample_ms += sample_interval_ms

                scanned += 1
                bgr = frame.to_ndarray(format="bgr24")
                sig = _grayscale_signature(bgr)
                if last_sig is None:
                    diff_score = 100.0
                else:
                    diff_score = float(np.mean(np.abs(sig - last_sig)))

                if last_sig is not None and payload.dedup_threshold > 0.0 and diff_score < payload.dedup_threshold:
                    skipped_similar += 1
                    continue

                last_sig = sig
                (preview_dir / f"f{cur_idx:06d}.jpg").write_bytes(_encode_jpeg(bgr, max_width=320, quality=80))
                candidates.append(
                    PreviewFrameCandidateDto(
                        frame_index=cur_idx,
                        timestamp_ms=ts_ms,
                        diff_score=round(diff_score, 2),
                        preview_url=f"/api/v1/annotations/projects/{project_id}/videos/{video_id}/preview/{cur_idx}",
                    )
                )
                if len(candidates) >= payload.max_frames:
                    break

        return PreviewFramesResponse(
            video_id=video_id,
            candidates=candidates,
            scanned_frames=scanned,
            skipped_similar=skipped_similar,
        )

    def load_preview_thumbnail(self, project_id: UUID, video_id: UUID, frame_index: int) -> Path:
        """Resolve cached candidate frame preview JPEG path."""
        preview = self._project_dir(project_id) / "videos" / f"previews_{video_id}" / f"f{frame_index:06d}.jpg"
        if not preview.is_file():
            raise ResourceNotFoundException("Miniatura de prévia não encontrada.")
        return preview

    def extract_video_frames(
        self,
        project_id: UUID,
        video_id: UUID,
        payload: ExtractFramesRequest,
    ) -> list[AnnotationItemSummaryDto]:
        """Extract selected (or deduplicated sampled) frames at original resolution into the project."""
        with self._lock:
            data = self._load_project(project_id)
            vid = self._find_video(data, video_id)
            video_path = self._project_dir(project_id) / str(vid["relPath"])
            if not video_path.is_file():
                raise ResourceNotFoundException("Arquivo de vídeo ausente no workspace.")
            vid_filename = str(vid["filename"])
            vid_origin = str(vid.get("sourceOrigin") or data.get("sourceOrigin") or "proprio")
            vid_campaign = str(vid.get("campaignId") or data.get("campaignId") or "default")

        meta = probe_video_metadata(video_path)
        target_indices = set(payload.frame_indices) if payload.frame_indices else None
        max_wanted = max(target_indices) if target_indices else None

        sample_interval_ms = 1000.0 / float(payload.sample_fps)
        next_sample_ms: float | None = None
        last_sig: np.ndarray | None = None
        extracted_frames: list[tuple[int, int, np.ndarray]] = []
        raw_idx = 0

        with av.open(str(video_path)) as container:
            in_stream = container.streams.video[0]
            for frame in iter_decoded_video_frames(container):
                if frame.pts is not None and frame.time_base is not None:
                    ts_ms = max(0, int(round(float(frame.pts * frame.time_base) * 1000.0)))
                elif in_stream.time_base is not None and frame.pts is not None:
                    ts_ms = max(0, int(round(float(frame.pts * in_stream.time_base) * 1000.0)))
                else:
                    ts_ms = int(round((raw_idx * 1000.0) / max(1.0, meta.fps)))
                cur_idx = raw_idx
                raw_idx += 1

                if target_indices is not None:
                    if cur_idx in target_indices:
                        extracted_frames.append((cur_idx, ts_ms, frame.to_ndarray(format="bgr24")))
                    if max_wanted is not None and cur_idx >= max_wanted and len(extracted_frames) >= len(target_indices):
                        break
                else:
                    if next_sample_ms is None:
                        next_sample_ms = float(ts_ms)
                    if (ts_ms + 1.0) < next_sample_ms:
                        continue
                    while next_sample_ms <= (ts_ms + 1.0):
                        next_sample_ms += sample_interval_ms
                    bgr = frame.to_ndarray(format="bgr24")
                    sig = _grayscale_signature(bgr)
                    if last_sig is not None and payload.dedup_threshold > 0.0:
                        if float(np.mean(np.abs(sig - last_sig))) < payload.dedup_threshold:
                            continue
                    last_sig = sig
                    extracted_frames.append((cur_idx, ts_ms, bgr))
                    if len(extracted_frames) >= payload.max_frames:
                        break

        if not extracted_frames:
            raise InvalidMediaException("NO_FRAMES_EXTRACTED", "Nenhum frame correspondente foi extraído do vídeo.")

        stem = _safe_filename_stem(vid_filename, "video")
        created: list[AnnotationItemSummaryDto] = []
        with self._lock:
            data = self._load_project(project_id)
            for frame_idx, ts_ms, bgr in extracted_frames:
                jpg_bytes = _encode_jpeg(bgr, quality=95)
                frame_name = f"{stem}__f{frame_idx:06d}.jpg"
                rec = self._register_image_bgr(
                    project_id=project_id,
                    data=data,
                    bgr=bgr,
                    raw_bytes=jpg_bytes,
                    filename=frame_name,
                    source_origin=vid_origin,
                    campaign_id=vid_campaign,
                    source_video_id=str(video_id),
                    source_video_filename=vid_filename,
                    frame_index=frame_idx,
                    timestamp_ms=ts_ms,
                )
                if vid.get("reservedBenchmarkWarning"):
                    rec["reservedBenchmarkWarning"] = True
                created.append(self._to_item_summary(project_id, rec))
            self._save_project(project_id, data)
        return created

    def import_curation_inbox(self, project_id: UUID) -> ImportCurationResponse:
        """Import non-duplicate curated frames from Vision Lab's curation-inbox into this project."""
        inbox = self._settings.curation_inbox_path
        with self._lock:
            data = self._load_project(project_id)
            if not inbox.is_dir():
                return ImportCurationResponse(imported_count=0, skipped_duplicates=0, items=[])

            existing_shas = {str(it.get("sourceSha256") or "") for it in (data.get("items") or [])}
            imported: list[AnnotationItemSummaryDto] = []
            skipped = 0

            for meta_path in sorted(inbox.rglob("*.meta.json")):
                stem_name = meta_path.name[: -len(".meta.json")]
                img_path = meta_path.with_name(f"{stem_name}.jpg")
                txt_path = meta_path.with_name(f"{stem_name}.txt")
                if not img_path.is_file():
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}

                raw_bytes = img_path.read_bytes()
                sha = hashlib.sha256(raw_bytes).hexdigest()
                if sha in existing_shas:
                    skipped += 1
                    continue

                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue

                boxes: list[dict[str, Any]] = []
                if txt_path.is_file():
                    for line in txt_path.read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                        x1 = _clamp01(cx - bw / 2.0)
                        y1 = _clamp01(cy - bh / 2.0)
                        nw = max(0.002, min(1.0 - x1, bw))
                        nh = max(0.002, min(1.0 - y1, bh))
                        boxes.append(
                            {
                                "id": f"box_{uuid4().hex[:10]}",
                                "classId": int(parts[0]),
                                "className": "open_top_freight_wagon",
                                "x": round(x1, 6),
                                "y": round(y1, 6),
                                "width": round(nw, 6),
                                "height": round(nh, 6),
                                "origin": "ai_suggestion",
                                "confidence": None,
                                "modelId": meta.get("modelId"),
                                "modelSha256": meta.get("modelSha256"),
                            }
                        )

                rec = self._register_image_bgr(
                    project_id=project_id,
                    data=data,
                    bgr=bgr,
                    raw_bytes=raw_bytes,
                    filename=str(meta.get("sourceFilename") or img_path.name),
                    source_origin="vision_lab_curation",
                    campaign_id=str(meta.get("targetCampaign") or data.get("campaignId") or "default"),
                    status="pre_annotated" if boxes else "pending",
                    boxes=boxes,
                    frame_index=meta.get("frameIndex"),
                    timestamp_ms=meta.get("timestampMs"),
                    preannotation_meta={
                        "preannotationModelId": meta.get("modelId"),
                        "preannotationModelSha256": meta.get("modelSha256"),
                        "preannotationConf": meta.get("confThreshold"),
                        "preannotationIou": meta.get("iouThreshold"),
                        "reviewNotes": meta.get("notes"),
                    },
                )
                if meta.get("reservedBenchmarkWarning"):
                    rec["reservedBenchmarkWarning"] = True
                existing_shas.add(sha)
                imported.append(self._to_item_summary(project_id, rec))

            if imported:
                self._save_project(project_id, data)
            return ImportCurationResponse(imported_count=len(imported), skipped_duplicates=skipped, items=imported)

    def _find_item(self, data: dict[str, Any], item_id: UUID) -> dict[str, Any]:
        target = str(item_id)
        for rec in data.get("items") or []:
            if str(rec.get("id")) == target:
                return rec
        raise ResourceNotFoundException(f"Item de anotação não encontrado: {item_id}")

    def get_item(self, project_id: UUID, item_id: UUID) -> AnnotationItemDetailDto:
        """Fetch full annotation item detail including boxes and provenance."""
        with self._lock:
            data = self._load_project(project_id)
            rec = self._find_item(data, item_id)
            return self._to_item_detail(project_id, rec)

    def load_item_image_path(self, project_id: UUID, item_id: UUID, *, thumbnail: bool = False) -> Path:
        """Resolve item clean JPEG or thumbnail path."""
        with self._lock:
            data = self._load_project(project_id)
            rec = self._find_item(data, item_id)
            rel = rec["thumbRelPath"] if thumbnail else rec["imageRelPath"]
            path = self._project_dir(project_id) / str(rel)
            if not path.is_file():
                raise ResourceNotFoundException("Arquivo de imagem não encontrado.")
            return path

    def delete_item(self, project_id: UUID, item_id: UUID) -> None:
        """Remove a single image item from the project."""
        with self._lock:
            data = self._load_project(project_id)
            rec = self._find_item(data, item_id)
            proj_dir = self._project_dir(project_id)
            for key in ("imageRelPath", "thumbRelPath"):
                rel = rec.get(key)
                if rel:
                    (proj_dir / str(rel)).unlink(missing_ok=True)
            data["items"] = [it for it in (data.get("items") or []) if str(it.get("id")) != str(item_id)]
            self._save_project(project_id, data)

    def update_item_annotations(
        self,
        project_id: UUID,
        item_id: UUID,
        payload: UpdateItemAnnotationsRequest,
    ) -> AnnotationItemDetailDto:
        """Save human-edited bounding boxes, negative confirmation, status, and review timestamps."""
        with self._lock:
            data = self._load_project(project_id)
            rec = self._find_item(data, item_id)
            clean_boxes = [box.model_dump(mode="json", by_alias=True) for box in payload.boxes]
            now_iso = _utcnow().isoformat()

            next_status = payload.status or rec.get("status") or "in_review"
            if payload.status is None and (clean_boxes or payload.no_objects_confirmed):
                next_status = "reviewed"

            rec["boxes"] = clean_boxes
            rec["noObjectsConfirmed"] = bool(payload.no_objects_confirmed)
            rec["status"] = next_status
            rec["reviewNotes"] = payload.review_notes
            rec["falsePositivesRemoved"] = int(payload.false_positives_removed)
            rec["falseNegativesAdded"] = int(payload.false_negatives_added)
            rec["updatedAt"] = now_iso

            if next_status in REVIEWED_STATUSES or payload.no_objects_confirmed:
                rec["humanReviewed"] = True
                if not rec.get("reviewedAt"):
                    rec["reviewedAt"] = now_iso
                if next_status == "approved":
                    rec["approvedAt"] = now_iso
            elif next_status in {"pending", "pre_annotated"}:
                rec["humanReviewed"] = False
                rec["approvedAt"] = None
            elif next_status == "rejected":
                rec["humanReviewed"] = True
                if not rec.get("reviewedAt"):
                    rec["reviewedAt"] = now_iso
                rec["approvedAt"] = None

            self._save_project(project_id, data)
            return self._to_item_detail(project_id, rec)

    def _run_preannotation_on_rec(
        self,
        *,
        proj_dir: Path,
        rec: dict[str, Any],
        model_id: str | None,
        runtime: str | None,
        conf: float,
        iou: float,
    ) -> tuple[int, str, str]:
        img_path = proj_dir / str(rec["imageRelPath"])
        arr = np.fromfile(str(img_path), dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise InvalidMediaException("CORRUPT_IMAGE", "Falha ao ler imagem do item para pré-anotação.")

        det = self._runtime_manager.detect_bgr(
            bgr,
            conf_threshold=conf,
            iou_threshold=iou,
            runtime=runtime,
            allow_cpu_fallback=False,
            model_id=model_id,
        )
        meta = self._runtime_manager.resolve_metadata(model_id)
        _, model_sha = self._runtime_manager.model_info_for_runtime(det.runtime_id, model_id=model_id)

        boxes = [
            {
                "id": f"ai_{uuid4().hex[:10]}",
                "classId": int(box.class_id),
                "className": str(box.class_name),
                "x": round(_clamp01(box.norm_x), 6),
                "y": round(_clamp01(box.norm_y), 6),
                "width": round(_clamp01(box.norm_width), 6),
                "height": round(_clamp01(box.norm_height), 6),
                "origin": "ai_suggestion",
                "confidence": round(float(box.confidence), 4),
                "modelId": meta.model_id,
                "modelSha256": model_sha,
            }
            for box in det.detections
            if box.norm_width > 0.0 and box.norm_height > 0.0
        ]
        now_iso = _utcnow().isoformat()
        rec["boxes"] = boxes
        rec["status"] = "pre_annotated"
        rec["humanReviewed"] = False
        rec["noObjectsConfirmed"] = False
        rec["preannotationModelId"] = meta.model_id
        rec["preannotationModelSha256"] = model_sha
        rec["preannotationConf"] = round(conf, 2)
        rec["preannotationIou"] = round(iou, 2)
        rec["updatedAt"] = now_iso
        return len(boxes), meta.model_id, det.runtime_id

    def preannotate_item(self, project_id: UUID, item_id: UUID, payload: PreannotateRequest) -> AnnotationItemDetailDto:
        """Run AI pre-annotation on one item without silently overwriting already human-reviewed annotations."""
        with self._lock:
            data = self._load_project(project_id)
            rec = self._find_item(data, item_id)
            if rec.get("humanReviewed") and not payload.overwrite:
                raise ConflictException(
                    "Este item já passou por revisão humana. Marque 'sobrescrever' para gerar novas sugestões de IA."
                )
            self._run_preannotation_on_rec(
                proj_dir=self._project_dir(project_id),
                rec=rec,
                model_id=payload.model_id,
                runtime=payload.runtime,
                conf=payload.conf,
                iou=payload.iou,
            )
            self._save_project(project_id, data)
            return self._to_item_detail(project_id, rec)

    def preannotate_batch(self, project_id: UUID, payload: PreannotateRequest) -> PreannotateBatchResponse:
        """Pre-annotate pending items (or specified IDs) in batch, skipping human-reviewed items by default."""
        with self._lock:
            data = self._load_project(project_id)
            proj_dir = self._project_dir(project_id)
            selected_ids = {str(x) for x in payload.item_ids} if payload.item_ids else None
            processed = 0
            skipped_reviewed = 0
            total_suggestions = 0
            last_model_id = self._runtime_manager.resolve_metadata(payload.model_id).model_id
            last_runtime_id = self._runtime_manager.normalize_runtime_id(payload.runtime)

            for rec in data.get("items") or []:
                if selected_ids is not None:
                    if str(rec.get("id")) not in selected_ids:
                        continue
                elif rec.get("status") != "pending" and not payload.overwrite:
                    continue

                if rec.get("humanReviewed") and not payload.overwrite:
                    skipped_reviewed += 1
                    continue

                cnt, last_model_id, last_runtime_id = self._run_preannotation_on_rec(
                    proj_dir=proj_dir,
                    rec=rec,
                    model_id=payload.model_id,
                    runtime=payload.runtime,
                    conf=payload.conf,
                    iou=payload.iou,
                )
                processed += 1
                total_suggestions += cnt

            if processed > 0:
                self._save_project(project_id, data)
            return PreannotateBatchResponse(
                project_id=project_id,
                processed_items=processed,
                skipped_reviewed=skipped_reviewed,
                total_suggestions=total_suggestions,
                model_id=last_model_id,
                runtime_id=last_runtime_id,
            )

    def export_yolo_zip(self, project_id: UUID, payload: ExportYoloRequest) -> tuple[bytes, str]:
        """Build a CVAT YOLO 1.1 .zip archive enforcing the human-reviewed ingestion gate by default."""
        with self._lock:
            data = self._load_project(project_id)
            proj_dir = self._project_dir(project_id)
            classes = list(data.get("classes") or DEFAULT_CLASSES)
            items = list(data.get("items") or [])

            selected: list[dict[str, Any]] = []
            for rec in items:
                if rec.get("status") == "rejected":
                    continue
                is_reviewed = bool(rec.get("humanReviewed")) or rec.get("status") in REVIEWED_STATUSES
                if payload.only_human_reviewed and not payload.include_unreviewed_drafts:
                    if not is_reviewed:
                        continue
                selected.append(rec)

            if not selected:
                raise ConflictException(
                    "Nenhum item com revisão humana concluída disponível para exportação oficial YOLO."
                )

            buf = io.BytesIO()
            train_lines: list[str] = []
            provenance_items: list[dict[str, Any]] = []
            all_reviewed = True

            with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("obj.names", "\n".join(classes) + "\n")
                zf.writestr(
                    "obj.data",
                    f"classes = {len(classes)}\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n",
                )
                for idx, rec in enumerate(selected):
                    stem = f"{idx:04d}_{_safe_filename_stem(str(rec['filename']), str(rec['id'])[:8])}"
                    img_rel = f"data/obj_train_data/{stem}.jpg"
                    txt_rel = f"data/obj_train_data/{stem}.txt"
                    img_bytes = (proj_dir / str(rec["imageRelPath"])).read_bytes()
                    zf.writestr(img_rel, img_bytes)

                    yolo_lines: list[str] = []
                    for box in rec.get("boxes") or []:
                        cid = int(box.get("classId") or 0)
                        bx = float(box.get("x") or 0.0)
                        by = float(box.get("y") or 0.0)
                        bw = float(box.get("width") or 0.0)
                        bh = float(box.get("height") or 0.0)
                        cx = _clamp01(bx + bw / 2.0)
                        cy = _clamp01(by + bh / 2.0)
                        yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {_clamp01(bw):.6f} {_clamp01(bh):.6f}")
                    zf.writestr(txt_rel, "\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
                    train_lines.append(img_rel)

                    item_reviewed = bool(rec.get("humanReviewed")) or rec.get("status") in REVIEWED_STATUSES
                    all_reviewed = all_reviewed and item_reviewed
                    provenance_items.append(
                        {
                            "itemId": rec.get("id"),
                            "exportedImage": img_rel,
                            "sourceFilename": rec.get("filename"),
                            "sourceSha256": rec.get("sourceSha256"),
                            "campaignId": rec.get("campaignId"),
                            "sourceOrigin": rec.get("sourceOrigin"),
                            "status": rec.get("status"),
                            "humanReviewed": item_reviewed,
                            "noObjectsConfirmed": bool(rec.get("noObjectsConfirmed", False)),
                            "preannotationModelId": rec.get("preannotationModelId"),
                            "preannotationModelSha256": rec.get("preannotationModelSha256"),
                            "reservedBenchmarkWarning": bool(rec.get("reservedBenchmarkWarning", False)),
                            "reviewedAt": rec.get("reviewedAt"),
                            "approvedAt": rec.get("approvedAt"),
                        }
                    )

                zf.writestr("train.txt", "\n".join(train_lines) + "\n")
                zf.writestr(
                    "provenance.json",
                    json.dumps(
                        {
                            "projectId": str(project_id),
                            "projectName": data.get("name"),
                            "campaignId": data.get("campaignId"),
                            "humanReviewed": all_reviewed,
                            "exportedAt": _utcnow().isoformat(),
                            "classes": classes,
                            "items": provenance_items,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                )

            zip_name = f"railvision-annotations-{_safe_filename_stem(str(data['campaignId']), 'pack')}.zip"
            return buf.getvalue(), zip_name

    def import_yolo_zip(self, project_id: UUID, file: UploadFile) -> list[AnnotationItemSummaryDto]:
        """Import images and YOLO 1.1 labels from a CVAT-compatible .zip archive."""
        raw_zip = file.file.read(self._max_upload_bytes + 1)
        if not raw_zip:
            raise InvalidMediaException("EMPTY_FILE", "Arquivo .zip vazio.")
        if len(raw_zip) > self._max_upload_bytes:
            raise MediaTooLargeException(f"O pacote .zip excede o limite de {self._settings.upload_max_size_mb} MB.")

        try:
            zf_ctx = zipfile.ZipFile(io.BytesIO(raw_zip), mode="r")
        except zipfile.BadZipFile as exc:
            raise InvalidMediaException("INVALID_ZIP", "Arquivo .zip inválido para importação YOLO.") from exc

        with self._lock, zf_ctx as zf:
            data = self._load_project(project_id)
            classes = list(data.get("classes") or DEFAULT_CLASSES)
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if len(names) > 500:
                raise InvalidMediaException(
                    "ZIP_TOO_MANY_FILES",
                    "O pacote .zip excede o limite de 500 arquivos por importação.",
                )

            img_entries: dict[str, str] = {}
            txt_entries: dict[str, str] = {}
            for member in names:
                p = Path(member)
                if ".." in p.parts:
                    continue
                ext = p.suffix.lower().lstrip(".")
                if ext in IMAGE_EXTENSIONS:
                    img_entries[p.stem] = member
                elif ext == "txt" and p.name not in {"train.txt"}:
                    txt_entries[p.stem] = member

            if not img_entries:
                raise InvalidMediaException("NO_IMAGES_IN_ZIP", "Nenhuma imagem encontrada no arquivo .zip YOLO.")

            max_member_bytes = 50 * 1024 * 1024
            max_total_bytes = self._max_upload_bytes * 4
            total_uncompressed_bytes = 0
            for member in list(img_entries.values()) + list(txt_entries.values()):
                info = zf.getinfo(member)
                if info.file_size > max_member_bytes:
                    raise MediaTooLargeException(
                        f"Arquivo '{Path(member).name}' no .zip excede o limite de 50 MB descompactado."
                    )
                total_uncompressed_bytes += info.file_size
                if total_uncompressed_bytes > max_total_bytes:
                    raise MediaTooLargeException(
                        "Tamanho total descompactado do .zip excede o limite permitido."
                    )

            created: list[AnnotationItemSummaryDto] = []
            for stem, member in sorted(img_entries.items()):
                img_bytes = zf.read(member)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                boxes: list[dict[str, Any]] = []
                txt_member = txt_entries.get(stem)
                if txt_member:
                    txt_text = zf.read(txt_member).decode("utf-8", errors="ignore")
                    for line in txt_text.splitlines():
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        try:
                            cid = int(parts[0])
                            cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                        except ValueError:
                            continue
                        x1 = _clamp01(cx - bw / 2.0)
                        y1 = _clamp01(cy - bh / 2.0)
                        nw = max(0.002, min(1.0 - x1, bw))
                        nh = max(0.002, min(1.0 - y1, bh))
                        cname = classes[cid] if 0 <= cid < len(classes) else classes[0]
                        boxes.append(
                            {
                                "id": f"imp_{uuid4().hex[:10]}",
                                "classId": cid,
                                "className": cname,
                                "x": round(x1, 6),
                                "y": round(y1, 6),
                                "width": round(nw, 6),
                                "height": round(nh, 6),
                                "origin": "manual",
                                "confidence": None,
                            }
                        )
                rec = self._register_image_bgr(
                    project_id=project_id,
                    data=data,
                    bgr=bgr,
                    raw_bytes=img_bytes,
                    filename=Path(member).name,
                    source_origin="cvat_yolo_import",
                    campaign_id=str(data.get("campaignId") or "default"),
                    status="in_review" if boxes else "pending",
                    boxes=boxes,
                )
                created.append(self._to_item_summary(project_id, rec))

            self._save_project(project_id, data)
            return created
