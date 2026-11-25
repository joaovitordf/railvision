"""Controlled PyAV video decoding, multi-runtime detection/tracking, and libx264 MP4 encoding."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
import logging
from pathlib import Path
import time

import av

logger = logging.getLogger(__name__)

from app.core.exceptions import InvalidMediaException, RuntimeUnavailableException
from app.modules.media.probe import VideoMetadata, probe_video_metadata
from app.modules.vision.renderer import render_annotated_frame
from app.modules.vision.runtimes import VisionRuntimeManager
from app.modules.vision.schemas import DetectionBox
from app.modules.vision.tracker import ByteTrackSession

OUTPUT_RESOLUTIONS = ("1080p", "native")
DEFAULT_X264_CRF = 23
DEFAULT_X264_PRESET = "fast"


def _even_dim(value: int) -> int:
    rounded = max(2, int(value))
    return rounded if (rounded % 2 == 0) else (rounded - 1)


def compute_output_dimensions(
    src_width: int,
    src_height: int,
    output_resolution: str | None,
    *,
    max_height: int = 1080,
) -> tuple[int, int, str]:
    """Compute even-aligned output dimensions for default 1080p downscale or native resolution."""
    resolved = (output_resolution or "1080p").strip().lower()
    if resolved not in OUTPUT_RESOLUTIONS:
        raise InvalidMediaException(
            "INVALID_OUTPUT_RESOLUTION",
            f"Resolução de saída inválida '{output_resolution}'. Use '1080p' ou 'native'.",
        )
    if resolved == "1080p" and src_height > max_height:
        scale = max_height / float(src_height)
        return _even_dim(int(round(src_width * scale))), _even_dim(max_height), resolved
    return _even_dim(src_width), _even_dim(src_height), resolved


def ensure_h264_encoder_available(encoder_name: str = "libx264") -> str:
    """Verify explicit controlled H.264 encoder availability without silent fallbacks."""
    if encoder_name not in av.codecs_available:
        raise RuntimeUnavailableException(
            f"Encoder de vídeo controlado '{encoder_name}' indisponível no PyAV."
        )
    try:
        codec = av.codec.Codec(encoder_name, "w")
    except Exception as exc:
        raise RuntimeUnavailableException(
            f"Falha ao inicializar encoder controlado '{encoder_name}': {exc}"
        ) from exc
    return codec.name


@dataclass(frozen=True, slots=True)
class ProcessedFrameSummary:
    """Per-frame summary matching VideoFrameSummaryDto contract."""

    frame_index: int
    timestamp_ms: int
    detection_count: int
    frame_latency_ms: float
    detections: list[DetectionBox]


@dataclass(frozen=True, slots=True)
class VideoPipelineSummary:
    """Complete execution telemetry and output media metadata from process_video."""

    metadata: VideoMetadata
    output_video_path: Path
    output_resolution: str
    output_width: int
    output_height: int
    output_codec: str
    output_encoder: str
    output_container: str
    output_crf: int
    browser_playable: bool
    sampled_frames: int
    frames_with_detections: int
    total_detections: int
    unique_track_count: int | None
    truncated_by_frame_limit: bool
    wall_time_ms: float
    preprocess_ms: list[float]
    inference_ms: list[float]
    postprocess_ms: list[float]
    encode_ms: list[float]
    frame_total_ms: list[float]
    runtime_id: str
    requested_execution_provider: str
    active_execution_provider: str
    device_type: str
    fallback_occurred: bool
    tracking_integrated: bool
    tracker_algorithm: str
    frame_summaries: list[ProcessedFrameSummary]


def iter_decoded_video_frames(container: av.container.InputContainer) -> Iterator[av.VideoFrame]:
    """Decode video stream frames while tolerating zero-size trailing container EOF packets (e.g. Ogg/Theora)."""
    decoded_count = 0
    for packet in container.demux(video=0):
        # Some Ogg/Theora files append a 0-byte timestamped packet at stream end, which makes avcodec_send_packet fail with EINVAL (22).
        if packet.size == 0 and packet.dts is not None:
            logger.debug("Ignoring empty timestamped video packet (dts=%s) after %d frames.", packet.dts, decoded_count)
            continue
        try:
            for frame in packet.decode():
                decoded_count += 1
                yield frame
        except (av.error.ArgumentError, av.error.InvalidDataError) as exc:
            if decoded_count > 0:
                logger.warning("Ignoring trailing/corrupt video packet after %d decoded frames: %s", decoded_count, exc)
                continue
            raise


def process_video(
    *,
    input_path: Path,
    run_dir: Path,
    runtime_manager: VisionRuntimeManager,
    conf_threshold: float,
    iou_threshold: float,
    sample_fps: float | None,
    max_frames: int,
    runtime: str | None = None,
    allow_cpu_fallback: bool = False,
    model_id: str | None = None,
    output_resolution: str = "1080p",
    max_output_height: int = 1080,
    enable_tracking: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
) -> VideoPipelineSummary:
    """Decode original video with PyAV, infer at native input size (640 letterbox), and encode MP4."""
    metadata = probe_video_metadata(input_path)
    out_w, out_h, resolved_res = compute_output_dimensions(
        metadata.width_px,
        metadata.height_px,
        output_resolution,
        max_height=max_output_height,
    )
    encoder_name = ensure_h264_encoder_available("libx264")

    est_sampled = (
        max(1, int(round((metadata.duration_ms / 1000.0) * sample_fps)))
        if (sample_fps is not None and sample_fps > 0.0 and metadata.duration_ms > 0)
        else max(1, metadata.estimated_frames)
    )
    est_sampled = min(est_sampled, max_frames)

    out_fps_float = sample_fps if (sample_fps is not None and sample_fps > 0.0) else metadata.fps
    out_fps_float = max(0.5, min(120.0, float(out_fps_float)))
    out_rate = Fraction(max(1, int(round(out_fps_float * 1000))), 1000)

    tmp_output_path = run_dir / "annotated.tmp.mp4"
    final_output_path = run_dir / "annotated.mp4"

    effective_track_fps = float(sample_fps if (sample_fps is not None and sample_fps > 0.0) else metadata.fps)
    tracker = ByteTrackSession(high_thresh=conf_threshold, sample_fps=effective_track_fps) if enable_tracking else None
    detector_conf = tracker.candidate_conf_threshold if tracker is not None else conf_threshold

    preprocess_times: list[float] = []
    inference_times: list[float] = []
    postprocess_times: list[float] = []
    encode_times: list[float] = []
    frame_total_times: list[float] = []
    frame_summaries: list[ProcessedFrameSummary] = []
    unique_tracks: set[int] = set()

    frames_with_detections = 0
    total_detections = 0
    truncated = False
    sampled_idx = 0
    raw_idx = 0
    next_sample_ms: float | None = None
    sample_interval_ms = (1000.0 / sample_fps) if (sample_fps is not None and sample_fps > 0.0) else None
    first_sampled_ts_ms: int | None = None
    last_out_pts_ms = -1

    last_runtime_id = runtime_manager.normalize_runtime_id(runtime)
    last_requested = last_runtime_id
    last_provider = ""
    last_device = ""
    any_fallback = False

    wall_t0 = time.perf_counter()

    try:
        with av.open(str(input_path)) as in_container:
            in_stream = in_container.streams.video[0]
            with av.open(str(tmp_output_path), mode="w", format="mp4", options={"movflags": "+faststart"}) as out_container:
                out_stream = out_container.add_stream(encoder_name, rate=out_rate)
                out_stream.time_base = Fraction(1, 1000)
                out_stream.width = out_w
                out_stream.height = out_h
                out_stream.pix_fmt = "yuv420p"
                out_stream.options = {"crf": str(DEFAULT_X264_CRF), "preset": DEFAULT_X264_PRESET}

                for frame in iter_decoded_video_frames(in_container):
                    if frame.pts is not None and frame.time_base is not None:
                        frame_ts_ms = max(0, int(round(float(frame.pts * frame.time_base) * 1000.0)))
                    elif in_stream.time_base is not None and frame.pts is not None:
                        frame_ts_ms = max(0, int(round(float(frame.pts * in_stream.time_base) * 1000.0)))
                    else:
                        frame_ts_ms = int(round((raw_idx * 1000.0) / max(1.0, metadata.fps)))
                    source_frame_idx = raw_idx
                    raw_idx += 1

                    if next_sample_ms is None:
                        next_sample_ms = float(frame_ts_ms)
                    if sample_interval_ms is not None and (frame_ts_ms + 1.0) < next_sample_ms:
                        continue
                    if sampled_idx >= max_frames:
                        truncated = True
                        break
                    if sample_interval_ms is not None:
                        while next_sample_ms <= (frame_ts_ms + 1.0):
                            next_sample_ms += sample_interval_ms

                    frame_wall0 = time.perf_counter()
                    bgr = frame.to_ndarray(format="bgr24")

                    det_result = runtime_manager.detect_bgr(
                        bgr,
                        conf_threshold=detector_conf,
                        iou_threshold=iou_threshold,
                        runtime=runtime,
                        allow_cpu_fallback=allow_cpu_fallback,
                        model_id=model_id,
                    )
                    last_runtime_id = det_result.runtime_id
                    last_requested = det_result.requested_execution_provider
                    last_provider = det_result.active_execution_provider
                    last_device = det_result.device_type
                    any_fallback = any_fallback or det_result.fallback_occurred

                    detections = det_result.detections
                    if tracker is not None:
                        detections = tracker.update(detections)
                        for box in detections:
                            if box.track_id is not None:
                                unique_tracks.add(box.track_id)

                    if first_sampled_ts_ms is None:
                        first_sampled_ts_ms = frame_ts_ms
                    pts_ms = max(last_out_pts_ms + 1, frame_ts_ms - first_sampled_ts_ms)
                    last_out_pts_ms = pts_ms

                    enc_t0 = time.perf_counter()
                    annotated_bgr = render_annotated_frame(
                        bgr,
                        detections,
                        target_width=out_w,
                        target_height=out_h,
                    )
                    out_frame = av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")
                    out_frame.pts = pts_ms
                    out_frame.time_base = Fraction(1, 1000)
                    for packet in out_stream.encode(out_frame):
                        out_container.mux(packet)
                    encode_ms = (time.perf_counter() - enc_t0) * 1000.0

                    frame_total_ms = (time.perf_counter() - frame_wall0) * 1000.0
                    preprocess_times.append(det_result.preprocess_ms)
                    inference_times.append(det_result.inference_ms)
                    postprocess_times.append(det_result.postprocess_ms)
                    encode_times.append(encode_ms)
                    frame_total_times.append(frame_total_ms)

                    if detections:
                        frames_with_detections += 1
                        total_detections += len(detections)

                    frame_summaries.append(
                        ProcessedFrameSummary(
                            frame_index=source_frame_idx,
                            timestamp_ms=frame_ts_ms,
                            detection_count=len(detections),
                            frame_latency_ms=round(frame_total_ms, 2),
                            detections=detections,
                        )
                    )

                    sampled_idx += 1
                    if progress_cb is not None:
                        progress_cb(sampled_idx, max(est_sampled, sampled_idx))

                for packet in out_stream.encode(None):
                    out_container.mux(packet)

        tmp_output_path.replace(final_output_path)
    except Exception:
        tmp_output_path.unlink(missing_ok=True)
        raise

    wall_time_ms = (time.perf_counter() - wall_t0) * 1000.0
    if not last_provider:
        overview = runtime_manager.get_runtime_overview()
        last_provider = str(overview["active_execution_provider"])
        last_device = str(overview["device_type"])

    return VideoPipelineSummary(
        metadata=metadata,
        output_video_path=final_output_path,
        output_resolution=resolved_res,
        output_width=out_w,
        output_height=out_h,
        output_codec="h264",
        output_encoder=encoder_name,
        output_container="mp4",
        output_crf=DEFAULT_X264_CRF,
        browser_playable=True,
        sampled_frames=sampled_idx,
        frames_with_detections=frames_with_detections,
        total_detections=total_detections,
        unique_track_count=len(unique_tracks) if enable_tracking else None,
        truncated_by_frame_limit=truncated,
        wall_time_ms=wall_time_ms,
        preprocess_ms=preprocess_times,
        inference_ms=inference_times,
        postprocess_ms=postprocess_times,
        encode_ms=encode_times,
        frame_total_ms=frame_total_times,
        runtime_id=last_runtime_id,
        requested_execution_provider=last_requested,
        active_execution_provider=last_provider,
        device_type=last_device,
        fallback_occurred=any_fallback,
        tracking_integrated=enable_tracking,
        tracker_algorithm="bytetrack" if enable_tracking else "none",
        frame_summaries=frame_summaries,
    )
