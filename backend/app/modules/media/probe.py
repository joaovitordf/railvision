"""Video stream metadata probing powered by PyAV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Intrinsic technical metadata extracted from a video file."""

    width_px: int
    height_px: int
    fps: float
    duration_ms: int
    codec: str
    estimated_frames: int


def probe_video_metadata(video_path: Path) -> VideoMetadata:
    """Extract dimensions, FPS, duration, codec, and estimated frame count using PyAV."""
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError("Arquivo de mídia não possui stream de vídeo.")
        stream = container.streams.video[0]
        width = int(stream.width or 0)
        height = int(stream.height or 0)
        if width <= 0 or height <= 0:
            raise ValueError("Dimensões de vídeo inválidas.")

        raw_rate = stream.average_rate or stream.guessed_rate
        fps = float(raw_rate) if raw_rate is not None and float(raw_rate) > 0.0 else 25.0

        duration_ms = 0
        if stream.duration is not None and stream.time_base is not None:
            duration_ms = max(0, int(round(float(stream.duration * stream.time_base) * 1000.0)))
        elif container.duration is not None and container.duration > 0:
            duration_ms = max(0, int(round(container.duration / 1000.0)))

        frame_count = int(stream.frames or 0)
        if frame_count <= 0 and duration_ms > 0:
            frame_count = max(1, int(round((duration_ms / 1000.0) * fps)))
        if duration_ms <= 0 and frame_count > 0 and fps > 0.0:
            duration_ms = int(round((frame_count / fps) * 1000.0))

        codec_name = str(stream.codec_context.name or "unknown")
        return VideoMetadata(
            width_px=width,
            height_px=height,
            fps=fps,
            duration_ms=duration_ms,
            codec=codec_name,
            estimated_frames=frame_count,
        )


def try_probe_video_metadata(video_path: Path) -> VideoMetadata | None:
    """Best-effort metadata probe returning None for minimal synthetic container headers."""
    try:
        return probe_video_metadata(video_path)
    except Exception:
        return None
