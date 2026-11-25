"""WS0 hardening tests: range suffix, upload limits/signatures, ByteTrack FPS buffer, sampleFps=0.5, lab recovery."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
from pathlib import Path
from uuid import uuid4

import av
from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.core.config import get_settings
from app.core.exceptions import InvalidMediaException
from app.core.paths import REPO_ROOT
from app.main import create_app
from app.modules.lab.service import LabService
from app.modules.media.storage import parse_byte_range
from app.modules.media.validator import VideoUploadValidator, matches_container_signature, sanitize_original_filename
from app.modules.vision.runtimes import VisionRuntimeManager
from app.modules.vision.tracker import ByteTrackSession
from app.modules.vision.video_pipeline import process_video


def _write_synthetic_h264_mp4(out_path: Path, *, num_frames: int = 8, fps: int = 2, start_pts_offset: int = 0) -> Path:
    with av.open(str(out_path), mode="w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.time_base = Fraction(1, fps)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = "yuv420p"
        for idx in range(num_frames):
            frame_bgr = np.zeros((240, 320, 3), dtype=np.uint8)
            frame_bgr[:, :, 1] = (idx * 25) % 255
            nf = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
            nf.pts = start_pts_offset + idx
            nf.time_base = Fraction(1, fps)
            for pkt in stream.encode(nf):
                out.mux(pkt)
        for pkt in stream.encode(None):
            out.mux(pkt)
    return out_path


@pytest.mark.unit
def test_parse_byte_range_suffix_always_reaches_eof() -> None:
    """Suffix range larger than max_chunk_bytes must return the trailing bytes up to content_length - 1."""
    total = 5_000_000
    start, end = parse_byte_range("bytes=-2000000", total, max_chunk_bytes=1_048_576)
    assert end == total - 1
    assert end - start + 1 == 1_048_576
    assert start == total - 1_048_576


@pytest.mark.unit
def test_validator_allows_ogv_and_rejects_long_filename() -> None:
    """Validator supports OGV container signature and blocks filenames > 255 chars cleanly."""
    assert matches_container_signature("ogv", b"OggS\x00\x02\x00\x00")
    assert not matches_container_signature("ogv", b"\x00\x00\x00\x18ftyp")

    too_long = ("a" * 252) + ".mp4"
    assert len(too_long) == 256
    with pytest.raises(InvalidMediaException) as exc_info:
        sanitize_original_filename(too_long)
    assert exc_info.value.error_code == "FILENAME_TOO_LONG"

    with pytest.raises(InvalidMediaException) as traversal:
        sanitize_original_filename("../secret.mp4")
    assert traversal.value.error_code == "INVALID_FILENAME"


@pytest.mark.unit
def test_bytetrack_scales_buffer_by_sample_fps() -> None:
    """ByteTrackSession adapts track_buffer proportionally to sample_fps."""
    session_5 = ByteTrackSession(high_thresh=0.25, sample_fps=5.0, buffer_seconds=1.5)
    session_25 = ByteTrackSession(high_thresh=0.25, sample_fps=25.0, buffer_seconds=1.5)
    assert session_5.effective_track_buffer == 8
    assert session_25.effective_track_buffer == 38


@pytest.mark.unit
def test_lab_rejects_bad_image_signature_and_recovers_stale_runs(tmp_path: Path) -> None:
    """Lab validates image magic bytes and startup recovery marks leftover QUEUED/RUNNING runs FAILED."""
    settings = get_settings().model_copy(
        update={
            "storage_root_raw": str(tmp_path / "storage"),
            "onnx_runs_storage_path_raw": str(tmp_path / "lab-runs"),
            "default_runtime": "onnx_cpu",
        }
    )
    stale_id = str(uuid4())
    stale_dir = tmp_path / "lab-runs" / stale_id
    stale_dir.mkdir(parents=True)
    (stale_dir / "result.json").write_text(
        json.dumps({"runId": stale_id, "mediaType": "VIDEO", "status": "RUNNING"}),
        encoding="utf-8",
    )

    app = create_app(settings, enable_startup_recovery=True)
    with TestClient(app) as client:
        recovered = json.loads((stale_dir / "result.json").read_text(encoding="utf-8"))
        assert recovered["status"] == "FAILED"
        assert "interrompida" in recovered["errorMessage"].lower()

        bad_img = client.post(
            "/api/v1/onnx-spike/analyze-image?runtime=onnx_cpu",
            files={"file": ("spoof.png", b"NOT_A_PNG_FILE_HEADER_DATA", "image/png")},
        )
        assert bad_img.status_code == 400
        assert bad_img.json()["errorCode"] == "INVALID_IMAGE_SIGNATURE"

        bad_vid = client.post(
            "/api/v1/onnx-spike/analyze-video?runtime=onnx_cpu",
            files={"file": ("spoof.mp4", b"NOT_AN_MP4_HEADER_BYTES", "video/mp4")},
        )
        assert bad_vid.status_code == 400
        assert bad_vid.json()["errorCode"] == "INVALID_VIDEO_SIGNATURE"


@pytest.mark.gpu
def test_concurrent_gpu_and_cpu_detect_bgr_thread_safety() -> None:
    """Concurrent inference calls on VisionRuntimeManager complete cleanly under _infer_lock."""
    manager = VisionRuntimeManager(get_settings())
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    def _call(rt: str):
        return manager.detect_bgr(frame, conf_threshold=0.25, iou_threshold=0.45, runtime=rt)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_call, rt) for rt in ("pytorch_cuda", "pytorch_cpu", "onnx_cpu", "pytorch_cuda")]
        results = [f.result() for f in futures]

    assert len(results) == 4
    assert all(r.inference_ms >= 0.0 for r in results)


@pytest.mark.unit
def test_video_pipeline_sub1fps_pts_sync_and_native_frame_index(tmp_path: Path) -> None:
    """sample_fps=0.5 encodes monotonic ms PTS at real duration and preserves native frame_index."""
    src_mp4 = _write_synthetic_h264_mp4(tmp_path / "src_2fps.mp4", num_frames=8, fps=2, start_pts_offset=100)
    run_dir = tmp_path / "run_subfps"
    run_dir.mkdir()
    settings = get_settings().model_copy(update={"default_runtime": "onnx_cpu"})
    manager = VisionRuntimeManager(settings)

    summary = process_video(
        input_path=src_mp4,
        run_dir=run_dir,
        runtime_manager=manager,
        conf_threshold=0.25,
        iou_threshold=0.45,
        sample_fps=0.5,
        max_frames=20,
        runtime="onnx_cpu",
        output_resolution="1080p",
        enable_tracking=True,
    )
    assert summary.sampled_frames == 2
    assert [fs.frame_index for fs in summary.frame_summaries] == [0, 4]
    with av.open(str(summary.output_video_path)) as out_c:
        frames = list(out_c.decode(video=0))
        assert len(frames) == 2
        assert frames[1].pts > frames[0].pts
