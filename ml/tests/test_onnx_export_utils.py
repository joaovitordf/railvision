"""Unit tests for ONNX letterbox preprocessing and YOLO output decoding helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_and_verify_onnx import (  # noqa: E402
    compute_iou_xyxy,
    compute_letterbox_params,
    decode_yolo_output,
    preprocess_bgr_letterbox,
    unletterbox_xyxy,
)


def test_letterbox_16_by_9_geometry() -> None:
    meta = compute_letterbox_params(orig_width=1920, orig_height=1080, target_size=640)
    assert meta["resized_width"] == 640
    assert meta["resized_height"] == 360
    assert meta["pad_left"] == 0
    assert meta["pad_right"] == 0
    assert meta["pad_top"] == 140
    assert meta["pad_bottom"] == 140


def test_preprocess_bgr_to_nchw_padding_color() -> None:
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # Pure red in BGR -> RGB (255, 0, 0)
    nchw, meta = preprocess_bgr_letterbox(img, target_size=640, pad_value=114)
    assert nchw.shape == (1, 3, 640, 640)
    assert nchw.dtype == np.float32
    # Top padding row should be 114/255 across all channels
    expected_pad = 114.0 / 255.0
    assert np.isclose(nchw[0, 0, 10, 320], expected_pad, atol=1e-4)
    assert np.isclose(nchw[0, 1, 10, 320], expected_pad, atol=1e-4)
    # Center active region should be red (R=1.0, G=0.0, B=0.0)
    assert np.isclose(nchw[0, 0, 320, 320], 1.0, atol=1e-4)
    assert np.isclose(nchw[0, 1, 320, 320], 0.0, atol=1e-4)
    assert meta["scale"] == 640.0 / 1920.0


def test_unletterbox_and_nms_decode() -> None:
    meta = compute_letterbox_params(orig_width=1280, orig_height=720, target_size=640)
    # Raw YOLO [1, 5, 3]
    raw = np.zeros((1, 5, 3), dtype=np.float32)
    # Candidate 0: high score box centered at (320, 320), size (100, 80)
    raw[0, :, 0] = [320.0, 320.0, 100.0, 80.0, 0.90]
    # Candidate 1: heavily overlapping lower score -> should be suppressed by NMS
    raw[0, :, 1] = [322.0, 321.0, 100.0, 80.0, 0.65]
    # Candidate 2: below threshold
    raw[0, :, 2] = [100.0, 100.0, 50.0, 40.0, 0.10]

    dets = decode_yolo_output(
        raw,
        meta,
        class_names={0: "open_top_freight_wagon"},
        conf_threshold=0.25,
        iou_threshold=0.45,
    )
    assert len(dets) == 1
    assert dets[0]["classId"] == 0
    assert dets[0]["className"] == "open_top_freight_wagon"
    assert abs(dets[0]["confidence"] - 0.90) < 1e-5
    # Verify unletterbox inverse (scale = 0.5, pad_top = 140)
    ox1, oy1, ox2, oy2 = unletterbox_xyxy(270.0, 280.0, 370.0, 360.0, meta)
    assert abs(ox1 - 540.0) < 1e-3
    assert abs(ox2 - 740.0) < 1e-3
    assert abs(oy1 - 280.0) < 1e-3
    assert abs(oy2 - 440.0) < 1e-3
    assert compute_iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) > 0.3
