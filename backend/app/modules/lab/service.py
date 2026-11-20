"""Local inference and ByteTrack inspection lab service preserving /api/v1/onnx-spike contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import shutil
import threading
import time
from typing import Any
from uuid import UUID, uuid4

import cv2
from fastapi import UploadFile
import numpy as np

from app.core.config import Settings
from app.core.exceptions import (
    InvalidMediaException,
    MediaNotFoundException,
    MediaTooLargeException,
    ResourceNotFoundException,
    RuntimeUnavailableException,
)
from app.modules.lab.schemas import (
    InferenceTimingMetrics,
    ModelConfigUsed,
    RunFileLinks,
    RunListItemDto,
    RuntimeStatusDto,
    VideoFrameSummaryDto,
    VideoRunStatusDto,
)
from app.modules.media.probe import probe_video_metadata
from app.modules.vision.renderer import render_annotated_frame
from app.modules.vision.runtimes import VisionRuntimeManager
from app.modules.vision.video_pipeline import (
    DEFAULT_X264_CRF,
    compute_output_dimensions,
    ensure_h264_encoder_available,
    process_video,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ("bmp", "jpeg", "jpg", "png", "webp")
VIDEO_EXTENSIONS = ("avi", "mkv", "mov", "mp4", "ogv", "webm")
MAX_IMAGE_BYTES = 30 * 1024 * 1024
EXPLORATORY_NOTICE = (
    "Execução exploratória local no Laboratório de Visão (não altera conjuntos de treino, validação ou teste)."
)
TRACKING_DISABLED_STATUS = "NOT_ENABLED_IN_RUN"
TRACKING_ENABLED_STATUS = "BYTETRACK_ENABLED"


@dataclass(frozen=True, slots=True)
class StoredRunFile:
    """Resolved file path, MIME type, and download filename for a lab run artifact."""

    file_path: Path
    content_type: str
    download_filename: str


def _clamp_threshold(raw: float | None, fallback: float, min_val: float, max_val: float) -> float:
    if raw is None or math.isnan(raw) or math.isinf(raw):
        return fallback
    return max(min_val, min(max_val, float(raw)))


def _sanitize_filename(raw: str | None, fallback: str) -> str:
    if not raw or not raw.strip():
        return fallback
    name = Path(raw.replace("\\", "/")).name.strip()
    return name or fallback


def _extract_extension(filename: str) -> str:
    dot = filename.rfind(".")
    if dot <= 0 or dot == len(filename) - 1:
        return ""
    return filename[dot + 1 :].lower()


def _resolve_content_type(filename: str) -> str:
    ext = _extract_extension(filename)
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "bmp": "image/bmp",
        "webp": "image/webp",
        "mp4": "video/mp4",
        "webm": "video/webm",
        "ogv": "video/ogg",
        "mov": "video/quicktime",
        "mkv": "video/x-matroska",
        "avi": "video/x-msvideo",
    }.get(ext, "application/octet-stream")


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def _save_upload_bounded(upload: UploadFile, dest: Path, max_bytes: int, too_large_msg: str, empty_msg: str) -> int:
    total = 0
    upload.file.seek(0)
    with dest.open("wb") as handle:
        while True:
            chunk = upload.file.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise MediaTooLargeException(too_large_msg)
            handle.write(chunk)
    if total == 0:
        raise InvalidMediaException("EMPTY_MEDIA", empty_msg)
    return total


class LabService:
    """Manages local image/video inspection runs across PyTorch CUDA, PyTorch CPU, and ONNX CPU."""

    def __init__(self, *, settings: Settings, runtime_manager: VisionRuntimeManager) -> None:
        self._settings = settings
        self._runtime_manager = runtime_manager
        self._runs_root = settings.onnx_runs_storage_path.resolve()
        self._runs_root.mkdir(parents=True, exist_ok=True)
        self._max_video_bytes = settings.upload_max_size_mb * 1024 * 1024
        self._runs_lock = threading.Lock()
        self._active_video_runs: dict[UUID, VideoRunStatusDto] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="railvision-lab-video")

    def get_runtime_status(self) -> RuntimeStatusDto:
        """Return runtime diagnostics, active model metadata, and available Python options."""
        overview = self._runtime_manager.get_runtime_overview()
        default_rt = self._runtime_manager.normalize_runtime_id(self._settings.default_runtime)
        return RuntimeStatusDto(
            available=overview["available"],
            unavailable_reason=overview["unavailable_reason"],
            model_name=overview["model_name"],
            source_experiment=overview["source_experiment"],
            model_display_path=overview["model_display_path"],
            model_sha256=overview["model_sha256"],
            input_size=overview["input_size"],
            input_shape=overview["input_shape"],
            output_shape=overview["output_shape"],
            class_names=overview["class_names"],
            requested_execution_provider=overview["requested_execution_provider"],
            active_execution_provider=overview["active_execution_provider"],
            device_type=overview["device_type"],
            cuda_available=overview["cuda_available"],
            fallback_occurred=overview["fallback_occurred"],
            available_providers=overview["available_providers"],
            device_diagnostic=overview["device_diagnostic"],
            tracking_integrated=True,
            tracking_status="OPTIONAL_BYTETRACK",
            tracking_notice=(
                "Tracking via ByteTrack disponível como opção habilitável na análise de vídeo "
                "(desativado por padrão para inspeção por frame)."
            ),
            limitations_notice=overview["limitations_notice"],
            default_conf_threshold=self._settings.default_confidence_threshold,
            default_iou_threshold=self._settings.default_iou_threshold,
            max_video_frames_per_ui_run=self._settings.max_video_frames_per_ui_run,
            supported_image_extensions=list(IMAGE_EXTENSIONS),
            supported_video_extensions=list(VIDEO_EXTENSIONS),
            backend_engine="python-fastapi",
            default_runtime=default_rt,
            available_runtimes=overview["runtime_options"],
            default_output_resolution="1080p",
            available_output_resolutions=["1080p", "native"],
            video_encoder="libx264",
            video_encoder_crf=DEFAULT_X264_CRF,
            tracking_algorithms=["none", "bytetrack"],
        )

    def analyze_image(
        self,
        *,
        file: UploadFile,
        conf: float | None,
        iou: float | None,
        runtime: str | None,
        allow_cpu_fallback: bool = False,
    ) -> dict[str, Any]:
        """Run synchronous single-image inference and save original + annotated PNG artifacts."""
        runtime_id = self._runtime_manager.normalize_runtime_id(runtime)
        if runtime_id == "pytorch_cuda" and not allow_cpu_fallback and not self._runtime_manager.get_runtime_overview()["cuda_available"]:
            raise RuntimeUnavailableException(
                "CUDA não está disponível para o runtime pytorch_cuda. "
                "Selecione pytorch_cpu/onnx_cpu ou habilite allowCpuFallback=true."
            )

        original_name = _sanitize_filename(file.filename, "image.png")
        ext = _extract_extension(original_name)
        if ext not in IMAGE_EXTENSIONS:
            raise InvalidMediaException(
                "UNSUPPORTED_IMAGE_FORMAT",
                "Formato de imagem não suportado. Envie JPG, PNG, BMP ou WEBP.",
            )

        conf_val = _clamp_threshold(conf, self._settings.default_confidence_threshold, 0.01, 0.99)
        iou_val = _clamp_threshold(iou, self._settings.default_iou_threshold, 0.05, 0.95)

        run_id = uuid4()
        run_dir = self._resolve_run_dir(run_id)
        original_path = run_dir / f"original.{ext}"
        annotated_path = run_dir / "annotated.png"

        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            _save_upload_bounded(
                file,
                original_path,
                MAX_IMAGE_BYTES,
                "A imagem excede o limite de 30 MB para o laboratório local.",
                "Selecione uma imagem válida para análise.",
            )
            raw_bytes = np.fromfile(str(original_path), dtype=np.uint8)
            bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            if bgr is None:
                raise InvalidMediaException("CORRUPT_IMAGE", "Não foi possível decodificar a imagem enviada.")
            img_h, img_w = bgr.shape[:2]

            wall_t0 = time.perf_counter()
            det_res = self._runtime_manager.detect_bgr(
                bgr,
                conf_threshold=conf_val,
                iou_threshold=iou_val,
                runtime=runtime_id,
                allow_cpu_fallback=allow_cpu_fallback,
            )
            render_t0 = time.perf_counter()
            annotated = render_annotated_frame(bgr, det_res.detections)
            ok, encoded = cv2.imencode(".png", annotated)
            if not ok:
                raise RuntimeError("Falha ao codificar imagem anotada PNG.")
            annotated_path.write_bytes(encoded.tobytes())
            render_ms = (time.perf_counter() - render_t0) * 1000.0
            total_wall_ms = (time.perf_counter() - wall_t0) * 1000.0

            no_detections = len(det_res.detections) == 0
            empty_message = (
                f"Nenhum objeto da classe monitorada (open_top_freight_wagon) foi detectado com confiança >= {conf_val:.2f}."
                if no_detections
                else None
            )

            _, model_sha = self._runtime_manager.model_info_for_runtime(det_res.runtime_id)
            meta = self._runtime_manager.metadata
            metrics = InferenceTimingMetrics(
                preprocess_ms=round(det_res.preprocess_ms, 2),
                inference_ms=round(det_res.inference_ms, 2),
                postprocess_ms=round(det_res.postprocess_ms, 2),
                decode_or_encode_ms=round(render_ms, 2),
                total_ms=round(total_wall_ms, 2),
                avg_frame_ms=round(det_res.total_ms, 2),
                p50_frame_ms=round(det_res.total_ms, 2),
                p95_frame_ms=round(det_res.total_ms, 2),
                processing_fps=round(1000.0 / total_wall_ms, 2) if total_wall_ms > 0 else 0.0,
                inference_only_fps=round(1000.0 / det_res.inference_ms, 2) if det_res.inference_ms > 0 else 0.0,
            )
            model_cfg = ModelConfigUsed(
                model_name=meta.model_name,
                source_experiment=meta.source_experiment,
                model_sha256=model_sha,
                input_size=meta.input_size,
                conf_threshold=round(conf_val, 2),
                iou_threshold=round(iou_val, 2),
                sample_fps=None,
                device_type=det_res.device_type,
                active_execution_provider=det_res.active_execution_provider,
                tracking_integrated=False,
                tracking_status=TRACKING_DISABLED_STATUS,
                exploratory_notice=EXPLORATORY_NOTICE,
                runtime_id=det_res.runtime_id,
                fallback_occurred=det_res.fallback_occurred,
                output_resolution=None,
                tracker_algorithm="none",
            )
            files = RunFileLinks(
                original_url=f"/api/v1/onnx-spike/runs/{run_id}/files/original",
                annotated_url=f"/api/v1/onnx-spike/runs/{run_id}/files/annotated",
                original_download_url=f"/api/v1/onnx-spike/runs/{run_id}/files/original?download=true",
                annotated_download_url=f"/api/v1/onnx-spike/runs/{run_id}/files/annotated?download=true",
                original_filename=f"original.{ext}",
                annotated_filename="annotated.png",
                annotated_content_type="image/png",
            )

            payload: dict[str, Any] = {
                "runId": str(run_id),
                "mediaType": "IMAGE",
                "status": "COMPLETED",
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "originalFilename": original_name,
                "imageWidth": int(img_w),
                "imageHeight": int(img_h),
                "detectionCount": len(det_res.detections),
                "noDetections": no_detections,
                "emptyMessage": empty_message,
                "detections": [d.model_dump(mode="json", by_alias=True) for d in det_res.detections],
                "metrics": metrics.model_dump(mode="json", by_alias=True),
                "modelConfig": model_cfg.model_dump(mode="json", by_alias=True),
                "files": files.model_dump(mode="json", by_alias=True),
            }
            self._write_run_json(run_dir, payload)
            return payload
        except InvalidMediaException:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        except MediaTooLargeException:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        except RuntimeUnavailableException:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise

    def submit_video_analysis(
        self,
        *,
        file: UploadFile,
        conf: float | None,
        iou: float | None,
        sample_fps: float | None,
        runtime: str | None,
        allow_cpu_fallback: bool = False,
        output_resolution: str = "1080p",
        enable_tracking: bool = False,
    ) -> VideoRunStatusDto:
        """Validate and enqueue async video inference/tracking returning initial QUEUED status."""
        runtime_id = self._runtime_manager.normalize_runtime_id(runtime)
        if runtime_id == "pytorch_cuda" and not allow_cpu_fallback and not self._runtime_manager.get_runtime_overview()["cuda_available"]:
            raise RuntimeUnavailableException(
                "CUDA não está disponível para o runtime pytorch_cuda. "
                "Selecione pytorch_cpu/onnx_cpu ou habilite allowCpuFallback=true."
            )
        ensure_h264_encoder_available("libx264")
        _, _, resolved_res = compute_output_dimensions(
            1920,
            1080,
            output_resolution,
            max_height=self._settings.output_video_max_height,
        )

        original_name = _sanitize_filename(file.filename, "video.mp4")
        ext = _extract_extension(original_name)
        if ext not in VIDEO_EXTENSIONS:
            raise InvalidMediaException(
                "UNSUPPORTED_VIDEO_FORMAT",
                "Formato de vídeo não suportado. Envie MP4, WEBM, OGV, MOV, MKV ou AVI.",
            )

        conf_val = _clamp_threshold(conf, self._settings.default_confidence_threshold, 0.01, 0.99)
        iou_val = _clamp_threshold(iou, self._settings.default_iou_threshold, 0.05, 0.95)
        fps_val = None if (sample_fps is None or sample_fps <= 0.0) else _clamp_threshold(sample_fps, 5.0, 0.5, 60.0)

        run_id = uuid4()
        run_dir = self._resolve_run_dir(run_id)
        original_path = run_dir / f"original.{ext}"

        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            _save_upload_bounded(
                file,
                original_path,
                self._max_video_bytes,
                f"O vídeo excede o limite de {self._settings.upload_max_size_mb} MB para o laboratório local.",
                "Selecione um vídeo válido para análise.",
            )
            vid_meta = probe_video_metadata(original_path)
        except (InvalidMediaException, MediaTooLargeException):
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise InvalidMediaException("CORRUPT_VIDEO", f"Não foi possível abrir o vídeo enviado: {exc}") from exc

        est_sampled = (
            max(1, math.ceil((vid_meta.duration_ms / 1000.0) * fps_val))
            if (fps_val is not None and vid_meta.duration_ms > 0)
            else max(1, vid_meta.estimated_frames)
        )
        est_sampled = min(est_sampled, self._settings.max_video_frames_per_ui_run)

        out_w, out_h, _ = compute_output_dimensions(
            vid_meta.width_px,
            vid_meta.height_px,
            resolved_res,
            max_height=self._settings.output_video_max_height,
        )

        rt_option = next(opt for opt in self._runtime_manager.list_runtime_options() if opt.id == runtime_id)
        _, model_sha = self._runtime_manager.model_info_for_runtime(runtime_id)
        meta = self._runtime_manager.metadata

        model_cfg = ModelConfigUsed(
            model_name=meta.model_name,
            source_experiment=meta.source_experiment,
            model_sha256=model_sha,
            input_size=meta.input_size,
            conf_threshold=round(conf_val, 2),
            iou_threshold=round(iou_val, 2),
            sample_fps=round(fps_val, 2) if fps_val is not None else None,
            device_type=rt_option.device_type if rt_option.available else "CPU",
            active_execution_provider=rt_option.provider,
            tracking_integrated=enable_tracking,
            tracking_status=TRACKING_ENABLED_STATUS if enable_tracking else TRACKING_DISABLED_STATUS,
            exploratory_notice=EXPLORATORY_NOTICE,
            runtime_id=runtime_id,
            fallback_occurred=False,
            output_resolution=resolved_res,
            tracker_algorithm="bytetrack" if enable_tracking else "none",
        )

        initial = VideoRunStatusDto(
            run_id=run_id,
            media_type="VIDEO",
            status="QUEUED",
            created_at=datetime.now(timezone.utc),
            started_at=None,
            completed_at=None,
            original_filename=original_name,
            video_width=vid_meta.width_px,
            video_height=vid_meta.height_px,
            source_fps=round(vid_meta.fps, 2),
            source_codec=vid_meta.codec,
            source_duration_ms=vid_meta.duration_ms,
            output_codec=None,
            output_container=None,
            browser_playable=False,
            estimated_total_frames=est_sampled,
            processed_frames=0,
            progress_percent=0,
            frames_with_detections=0,
            total_detections=0,
            no_detections=False,
            truncated_by_frame_limit=False,
            empty_message=None,
            error_message=None,
            metrics=None,
            model_config_used=model_cfg,
            files=RunFileLinks(
                original_url=f"/api/v1/onnx-spike/runs/{run_id}/files/original",
                annotated_url=None,
                original_download_url=f"/api/v1/onnx-spike/runs/{run_id}/files/original?download=true",
                annotated_download_url=None,
                original_filename=f"original.{ext}",
                annotated_filename=None,
                annotated_content_type=None,
            ),
            frame_summaries=[],
            output_resolution=resolved_res,
            output_width=out_w,
            output_height=out_h,
            output_encoder="libx264",
            output_crf=DEFAULT_X264_CRF,
            unique_track_count=0 if enable_tracking else None,
        )

        with self._runs_lock:
            self._active_video_runs[run_id] = initial
        self._write_run_json_quietly(run_dir, initial.model_dump(mode="json", by_alias=True))

        self._executor.submit(
            self._execute_video_job,
            run_id,
            run_dir,
            original_path,
            original_name,
            ext,
            conf_val,
            iou_val,
            fps_val,
            runtime_id,
            allow_cpu_fallback,
            resolved_res,
            enable_tracking,
        )
        return initial

    def _execute_video_job(
        self,
        run_id: UUID,
        run_dir: Path,
        original_path: Path,
        original_name: str,
        original_ext: str,
        conf: float,
        iou: float,
        sample_fps: float | None,
        runtime_id: str,
        allow_cpu_fallback: bool,
        output_resolution: str,
        enable_tracking: bool,
    ) -> None:
        started_at = datetime.now(timezone.utc)
        with self._runs_lock:
            current = self._active_video_runs.get(run_id)
            if current is not None:
                self._active_video_runs[run_id] = current.model_copy(
                    update={"status": "RUNNING", "started_at": started_at}
                )

        def _on_progress(processed: int, est_total: int) -> None:
            pct = min(99, int(round((processed * 100.0) / est_total))) if est_total > 0 else 0
            with self._runs_lock:
                cur = self._active_video_runs.get(run_id)
                if cur is not None:
                    self._active_video_runs[run_id] = cur.model_copy(
                        update={
                            "status": "RUNNING",
                            "started_at": started_at,
                            "processed_frames": processed,
                            "estimated_total_frames": est_total,
                            "progress_percent": pct,
                        }
                    )

        try:
            summary = process_video(
                input_path=original_path,
                run_dir=run_dir,
                runtime_manager=self._runtime_manager,
                conf_threshold=conf,
                iou_threshold=iou,
                sample_fps=sample_fps,
                max_frames=self._settings.max_video_frames_per_ui_run,
                runtime=runtime_id,
                allow_cpu_fallback=allow_cpu_fallback,
                output_resolution=output_resolution,
                max_output_height=self._settings.output_video_max_height,
                enable_tracking=enable_tracking,
                progress_cb=_on_progress,
            )

            avg_pre = _mean(summary.preprocess_ms)
            avg_inf = _mean(summary.inference_ms)
            avg_post = _mean(summary.postprocess_ms)
            avg_enc = _mean(summary.encode_ms)
            avg_frame = _mean(summary.frame_total_ms)
            processing_fps = (
                (summary.sampled_frames * 1000.0 / summary.wall_time_ms)
                if summary.wall_time_ms > 0
                else 0.0
            )
            infer_only_fps = (1000.0 / avg_inf) if avg_inf > 0 else 0.0

            metrics = InferenceTimingMetrics(
                preprocess_ms=round(avg_pre, 2),
                inference_ms=round(avg_inf, 2),
                postprocess_ms=round(avg_post, 2),
                decode_or_encode_ms=round(avg_enc, 2),
                total_ms=round(summary.wall_time_ms, 2),
                avg_frame_ms=round(avg_frame, 2),
                p50_frame_ms=round(_percentile(summary.frame_total_ms, 50.0), 2),
                p95_frame_ms=round(_percentile(summary.frame_total_ms, 95.0), 2),
                processing_fps=round(processing_fps, 2),
                inference_only_fps=round(infer_only_fps, 2),
            )

            no_detections = summary.total_detections == 0
            empty_message = (
                f"Nenhum objeto da classe monitorada (open_top_freight_wagon) foi detectado nos {summary.sampled_frames} frames processados."
                if no_detections
                else None
            )

            _, model_sha = self._runtime_manager.model_info_for_runtime(summary.runtime_id)
            meta = self._runtime_manager.metadata
            completed_cfg = ModelConfigUsed(
                model_name=meta.model_name,
                source_experiment=meta.source_experiment,
                model_sha256=model_sha,
                input_size=meta.input_size,
                conf_threshold=round(conf, 2),
                iou_threshold=round(iou, 2),
                sample_fps=round(sample_fps, 2) if sample_fps is not None else None,
                device_type=summary.device_type,
                active_execution_provider=summary.active_execution_provider,
                tracking_integrated=summary.tracking_integrated,
                tracking_status=TRACKING_ENABLED_STATUS if summary.tracking_integrated else TRACKING_DISABLED_STATUS,
                exploratory_notice=EXPLORATORY_NOTICE,
                runtime_id=summary.runtime_id,
                fallback_occurred=summary.fallback_occurred,
                output_resolution=summary.output_resolution,
                tracker_algorithm=summary.tracker_algorithm,
            )

            annotated_name = summary.output_video_path.name
            files = RunFileLinks(
                original_url=f"/api/v1/onnx-spike/runs/{run_id}/files/original",
                annotated_url=f"/api/v1/onnx-spike/runs/{run_id}/files/annotated",
                original_download_url=f"/api/v1/onnx-spike/runs/{run_id}/files/original?download=true",
                annotated_download_url=f"/api/v1/onnx-spike/runs/{run_id}/files/annotated?download=true",
                original_filename=f"original.{original_ext}",
                annotated_filename=annotated_name,
                annotated_content_type="video/mp4",
            )

            summaries_slice = summary.frame_summaries[:300]
            frame_dtos = [
                VideoFrameSummaryDto(
                    frame_index=item.frame_index,
                    timestamp_ms=item.timestamp_ms,
                    detection_count=item.detection_count,
                    frame_latency_ms=item.frame_latency_ms,
                    detections=item.detections,
                )
                for item in summaries_slice
            ]

            with self._runs_lock:
                cur = self._active_video_runs.get(run_id)
                created_at = cur.created_at if cur is not None else started_at

            completed = VideoRunStatusDto(
                run_id=run_id,
                media_type="VIDEO",
                status="COMPLETED",
                created_at=created_at,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                original_filename=original_name,
                video_width=summary.metadata.width_px,
                video_height=summary.metadata.height_px,
                source_fps=round(summary.metadata.fps, 2),
                source_codec=summary.metadata.codec,
                source_duration_ms=summary.metadata.duration_ms,
                output_codec=summary.output_codec,
                output_container=summary.output_container,
                browser_playable=summary.browser_playable,
                estimated_total_frames=summary.sampled_frames,
                processed_frames=summary.sampled_frames,
                progress_percent=100,
                frames_with_detections=summary.frames_with_detections,
                total_detections=summary.total_detections,
                no_detections=no_detections,
                truncated_by_frame_limit=summary.truncated_by_frame_limit,
                empty_message=empty_message,
                error_message=None,
                metrics=metrics,
                model_config_used=completed_cfg,
                files=files,
                frame_summaries=frame_dtos,
                output_resolution=summary.output_resolution,
                output_width=summary.output_width,
                output_height=summary.output_height,
                output_encoder=summary.output_encoder,
                output_crf=summary.output_crf,
                unique_track_count=summary.unique_track_count,
            )

            self._write_run_json(run_dir, completed.model_dump(mode="json", by_alias=True))
            with self._runs_lock:
                self._active_video_runs[run_id] = completed
        except Exception as exc:
            logger.exception("Falha ao processar vídeo %s no laboratório de visão", run_id)
            with self._runs_lock:
                cur = self._active_video_runs.get(run_id)
                if cur is None:
                    return
                failed = cur.model_copy(
                    update={
                        "status": "FAILED",
                        "completed_at": datetime.now(timezone.utc),
                        "error_message": f"Falha ao processar vídeo em PyAV/Visão: {exc}",
                        "no_detections": True,
                    }
                )
                self._active_video_runs[run_id] = failed
            self._write_run_json_quietly(run_dir, failed.model_dump(mode="json", by_alias=True))

    def list_runs(self) -> list[RunListItemDto]:
        """List live and persisted historical runs (including Java Phase 4 runs) sorted newest-first."""
        items: list[RunListItemDto] = []
        if not self._runs_root.is_dir():
            return items

        with self._runs_lock:
            live_snapshot = dict(self._active_video_runs)

        for entry in self._runs_root.iterdir():
            if not entry.is_dir():
                continue
            try:
                run_id = UUID(entry.name)
            except ValueError:
                continue

            live = live_snapshot.get(run_id)
            if live is not None:
                items.append(
                    RunListItemDto(
                        run_id=live.run_id,
                        media_type=live.media_type,
                        status=live.status,
                        original_filename=live.original_filename,
                        created_at=live.created_at,
                        total_detections=live.total_detections,
                        no_detections=live.no_detections,
                        processing_fps=live.metrics.processing_fps if live.metrics else None,
                        avg_frame_ms=live.metrics.avg_frame_ms if live.metrics else None,
                        total_ms=live.metrics.total_ms if live.metrics else None,
                        device_type=live.model_config_used.device_type,
                        active_execution_provider=live.model_config_used.active_execution_provider,
                        model_name=live.model_config_used.model_name,
                        runtime_id=live.model_config_used.runtime_id,
                        tracking_integrated=live.model_config_used.tracking_integrated,
                        unique_track_count=live.unique_track_count,
                    )
                )
                continue

            result_path = entry / "result.json"
            if not result_path.is_file():
                continue
            try:
                node = json.loads(result_path.read_text(encoding="utf-8"))
                media_type = str(node.get("mediaType") or "IMAGE")
                total_det = int(node.get("totalDetections" if media_type == "VIDEO" else "detectionCount") or 0)
                metrics_node = node.get("metrics") or {}
                cfg_node = node.get("modelConfig") or {}
                created_raw = str(node.get("createdAt") or datetime.fromtimestamp(0, tz=timezone.utc).isoformat())
                items.append(
                    RunListItemDto(
                        run_id=run_id,
                        media_type=media_type,
                        status=str(node.get("status") or "COMPLETED"),
                        original_filename=str(node.get("originalFilename") or "arquivo"),
                        created_at=datetime.fromisoformat(created_raw.replace("Z", "+00:00")),
                        total_detections=total_det,
                        no_detections=bool(node.get("noDetections", total_det == 0)),
                        processing_fps=metrics_node.get("processingFps"),
                        avg_frame_ms=metrics_node.get("avgFrameMs"),
                        total_ms=metrics_node.get("totalMs"),
                        device_type=str(cfg_node.get("deviceType") or "CPU"),
                        active_execution_provider=str(cfg_node.get("activeExecutionProvider") or "unknown"),
                        model_name=str(cfg_node.get("modelName") or self._runtime_manager.metadata.model_name),
                        runtime_id=cfg_node.get("runtimeId"),
                        tracking_integrated=bool(cfg_node.get("trackingIntegrated", False)),
                        unique_track_count=node.get("uniqueTrackCount"),
                    )
                )
            except Exception as exc:
                logger.warning("Ignorando result.json inválido em %s: %s", result_path, exc)

        items.sort(key=lambda item: item.created_at, reverse=True)
        return items[:30]

    def get_run(self, run_id: UUID) -> dict[str, Any]:
        """Fetch live or persisted result.json payload for a lab run."""
        with self._runs_lock:
            live = self._active_video_runs.get(run_id)
        if live is not None:
            return live.model_dump(mode="json", by_alias=True)

        result_path = self._resolve_run_dir(run_id) / "result.json"
        if not result_path.is_file():
            raise ResourceNotFoundException(f"Execução do laboratório ONNX não encontrada: {run_id}")
        return json.loads(result_path.read_text(encoding="utf-8"))

    def load_run_file(self, run_id: UUID, kind: str) -> StoredRunFile:
        """Resolve original or annotated artifact inside a run directory."""
        if kind not in {"original", "annotated"}:
            raise ResourceNotFoundException(f"Tipo de arquivo inválido: {kind}")
        run_dir = self._resolve_run_dir(run_id)
        if not run_dir.is_dir():
            raise ResourceNotFoundException(f"Execução não encontrada: {run_id}")

        for entry in sorted(run_dir.iterdir()):
            if (
                entry.is_file()
                and entry.name.startswith(f"{kind}.")
                and ".tmp." not in entry.name
            ):
                return StoredRunFile(
                    file_path=entry,
                    content_type=_resolve_content_type(entry.name),
                    download_filename=entry.name,
                )
        raise MediaNotFoundException(f"Arquivo '{kind}' não encontrado para a execução {run_id}")

    def _resolve_run_dir(self, run_id: UUID) -> Path:
        resolved = (self._runs_root / str(run_id)).resolve()
        if resolved.parent != self._runs_root:
            raise ValueError("Identificador de execução inválido.")
        return resolved

    def _write_run_json(self, run_dir: Path, payload: dict[str, Any]) -> None:
        target = run_dir / "result.json"
        tmp = run_dir / "result.tmp.json"
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)

    def _write_run_json_quietly(self, run_dir: Path, payload: dict[str, Any]) -> None:
        try:
            self._write_run_json(run_dir, payload)
        except Exception:
            pass

    def shutdown(self, *, wait: bool = True) -> None:
        """Gracefully shut down the lab video background executor."""
        self._executor.shutdown(wait=wait, cancel_futures=False)
