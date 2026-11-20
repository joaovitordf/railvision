"""Stage M3 computer vision verification: Phase 4 8-frame parity, no silent fallback, PyAV 4K/1080p, ByteTrack."""

from __future__ import annotations

import json
from pathlib import Path

import av
import cv2
import pytest
import torch

from app.core.config import get_settings
from app.core.exceptions import InvalidMediaException, RuntimeUnavailableException
from app.core.paths import REPO_ROOT
from app.modules.vision.preprocess import compute_iou_xyxy
from app.modules.vision.runtimes import VisionRuntimeManager
from app.modules.vision.video_pipeline import process_video


def _match_against_reference(ref_boxes: list[dict], cand_boxes) -> tuple[float, float, float]:
    assert len(cand_boxes) == len(ref_boxes)
    if not ref_boxes:
        return 1.0, 0.0, 0.0

    used: set[int] = set()
    min_iou = 1.0
    max_coord = 0.0
    max_conf = 0.0
    for ref in ref_boxes:
        ref_xyxy = (ref["x1"], ref["y1"], ref["x2"], ref["y2"])
        best_iou = -1.0
        best_idx = -1
        for idx, cand in enumerate(cand_boxes):
            if idx in used or cand.class_id != ref["classId"]:
                continue
            iou = compute_iou_xyxy(ref_xyxy, (cand.x1, cand.y1, cand.x2, cand.y2))
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        assert best_idx >= 0
        used.add(best_idx)
        cand = cand_boxes[best_idx]
        min_iou = min(min_iou, best_iou)
        max_coord = max(
            max_coord,
            abs(ref["x1"] - cand.x1),
            abs(ref["y1"] - cand.y1),
            abs(ref["x2"] - cand.x2),
            abs(ref["y2"] - cand.y2),
        )
        max_conf = max(max_conf, abs(ref["confidence"] - cand.confidence))
    return min_iou, max_coord, max_conf


@pytest.mark.parametrize("runtime_id,expected_device", [
    ("pytorch_cuda", "GPU"),
    ("pytorch_cpu", "CPU"),
    ("onnx_cpu", "CPU"),
])
def test_phase4_8_frame_parity_across_all_runtimes(runtime_id: str, expected_device: str) -> None:
    """All 3 runtimes match Phase 4 parity_reference.json across all 8 benchmark frames."""
    ref_path = REPO_ROOT / "artifacts/runs/phase4-onnx-spike/parity_reference.json"
    frames_dir = REPO_ROOT / "artifacts/runs/phase4-onnx-spike/parity_frames"
    ref_root = json.loads(ref_path.read_text(encoding="utf-8"))

    manager = VisionRuntimeManager(get_settings())
    conf = float(ref_root["confThreshold"])
    iou = float(ref_root["iouThreshold"])

    for frame_entry in ref_root["frames"]:
        frame_path = frames_dir / frame_entry["filename"]
        bgr = cv2.imread(str(frame_path))
        assert bgr is not None

        res = manager.detect_bgr(bgr, conf_threshold=conf, iou_threshold=iou, runtime=runtime_id)
        assert res.device_type == expected_device
        assert res.fallback_occurred is False
        min_iou, max_coord, max_conf = _match_against_reference(frame_entry["expectedDetections"], res.detections)
        assert min_iou >= 0.99, f"{runtime_id} IoU mismatch on {frame_entry['filename']}: {min_iou}"
        assert max_coord <= 1.0, f"{runtime_id} coord diff on {frame_entry['filename']}: {max_coord}"
        assert max_conf <= 0.005, f"{runtime_id} conf diff on {frame_entry['filename']}: {max_conf}"


def test_no_silent_fallback_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail explicitly when CUDA is requested without allow_cpu_fallback; mark fallback explicitly when allowed."""
    manager = VisionRuntimeManager(get_settings())
    ref_img = cv2.imread(
        str(REPO_ROOT / "artifacts/runs/phase4-onnx-spike/parity_frames/val__camp_sishen_saldanha__vid_15__f000075.jpg")
    )
    assert ref_img is not None

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeUnavailableException):
        manager.detect_bgr(ref_img, conf_threshold=0.25, iou_threshold=0.45, runtime="pytorch_cuda", allow_cpu_fallback=False)

    res = manager.detect_bgr(ref_img, conf_threshold=0.25, iou_threshold=0.45, runtime="pytorch_cuda", allow_cpu_fallback=True)
    assert res.fallback_occurred is True
    assert res.device_type == "CPU"
    assert res.runtime_id == "pytorch_cpu"
    assert "fallback" in res.active_execution_provider.lower()
    assert len(res.detections) == 4


def test_pyav_4k_vid01_negative_1080p_and_native_libx264(tmp_path: Path) -> None:
    """Verify 4K vid_01 negative reference emits 0 detections and encodes controlled 1080p & native libx264 MP4."""
    vid01_path = REPO_ROOT / "data/raw/public-videos/vid_01_langer_unbeladener_g_terzug_von_oben_k_l.webm"
    assert vid01_path.is_file()
    manager = VisionRuntimeManager(get_settings())

    out_1080 = tmp_path / "run_1080p"
    out_1080.mkdir()
    sum_1080 = process_video(
        input_path=vid01_path,
        run_dir=out_1080,
        runtime_manager=manager,
        conf_threshold=0.25,
        iou_threshold=0.45,
        sample_fps=2.0,
        max_frames=10,
        runtime="pytorch_cuda",
        output_resolution="1080p",
        enable_tracking=False,
    )
    assert sum_1080.output_width == 1920
    assert sum_1080.output_height == 1080
    assert sum_1080.output_encoder == "libx264"
    assert sum_1080.output_crf == 23
    assert sum_1080.total_detections == 0
    assert sum_1080.sampled_frames == 10
    with av.open(str(sum_1080.output_video_path)) as c:
        s = c.streams.video[0]
        assert (s.width, s.height) == (1920, 1080)
        assert s.codec_context.name == "h264"

    out_native = tmp_path / "run_native"
    out_native.mkdir()
    sum_native = process_video(
        input_path=vid01_path,
        run_dir=out_native,
        runtime_manager=manager,
        conf_threshold=0.25,
        iou_threshold=0.45,
        sample_fps=2.0,
        max_frames=6,
        runtime="pytorch_cuda",
        output_resolution="native",
        enable_tracking=False,
    )
    assert sum_native.output_width == 3840
    assert sum_native.output_height == 2160
    assert sum_native.output_encoder == "libx264"
    assert sum_native.total_detections == 0
    with av.open(str(sum_native.output_video_path)) as c:
        s = c.streams.video[0]
        assert (s.width, s.height) == (3840, 2160)
        assert s.codec_context.name == "h264"

    with pytest.raises(InvalidMediaException):
        process_video(
            input_path=vid01_path,
            run_dir=tmp_path,
            runtime_manager=manager,
            conf_threshold=0.25,
            iou_threshold=0.45,
            sample_fps=2.0,
            max_frames=2,
            output_resolution="invalid_res",
        )


def test_vid15_positive_detection_parity_and_opt_in_bytetrack(tmp_path: Path) -> None:
    """Verify positive wagon detection on vid_15 and persistent track IDs when ByteTrack is enabled."""
    vid15_path = REPO_ROOT / "data/raw/public-videos/vid_15_sishen_saldanha_railway_line.ogv"
    assert vid15_path.is_file()
    manager = VisionRuntimeManager(get_settings())

    no_track_dir = tmp_path / "no_track"
    no_track_dir.mkdir()
    no_track = process_video(
        input_path=vid15_path,
        run_dir=no_track_dir,
        runtime_manager=manager,
        conf_threshold=0.25,
        iou_threshold=0.45,
        sample_fps=5.0,
        max_frames=18,
        runtime="pytorch_cuda",
        output_resolution="1080p",
        enable_tracking=False,
    )
    assert no_track.total_detections > 0
    assert no_track.unique_track_count is None
    assert all(box.track_id is None for fs in no_track.frame_summaries for box in fs.detections)

    tracked_dir = tmp_path / "bytetrack"
    tracked_dir.mkdir()
    tracked = process_video(
        input_path=vid15_path,
        run_dir=tracked_dir,
        runtime_manager=manager,
        conf_threshold=0.25,
        iou_threshold=0.45,
        sample_fps=5.0,
        max_frames=18,
        runtime="pytorch_cuda",
        output_resolution="1080p",
        enable_tracking=True,
    )
    assert tracked.tracking_integrated is True
    assert tracked.tracker_algorithm == "bytetrack"
    assert tracked.total_detections > 0
    assert tracked.unique_track_count is not None and tracked.unique_track_count >= 1
    track_ids = {box.track_id for fs in tracked.frame_summaries for box in fs.detections if box.track_id is not None}
    assert len(track_ids) == tracked.unique_track_count
