"""WS0 tests for training split hygiene, opt-in test eval, ONNX metadata, and MOT two-stage config."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_open_top_tracking_mot import build_tracker_config_text  # noqa: E402
from export_and_verify_onnx import build_model_metadata_payload, extract_raw_top_score  # noqa: E402
from train_detector import (  # noqa: E402
    compute_calibration_curve_from_predictions,
    verify_dataset_split_hygiene,
)


def test_verify_dataset_split_hygiene_blocks_reserved_campaigns(tmp_path: Path) -> None:
    """vid_01, vid_09, and vid_16 must be rejected if present in train or val."""
    train_imgs = tmp_path / "train" / "images"
    val_imgs = tmp_path / "val" / "images"
    train_imgs.mkdir(parents=True)
    val_imgs.mkdir(parents=True)

    (train_imgs / "train__camp_coal_tunnel__vid_14__f000010.jpg").write_bytes(b"jpg")
    (val_imgs / "val__camp_sishen_saldanha__vid_15__f000010.jpg").write_bytes(b"jpg")
    verify_dataset_split_hygiene(tmp_path)

    (train_imgs / "train__vid_01_langer_unbeladener__f000001.jpg").write_bytes(b"jpg")
    with pytest.raises(ValueError, match="vid_01"):
        verify_dataset_split_hygiene(tmp_path)


def test_compute_calibration_curve_picks_optimal_f1_threshold() -> None:
    """Confidence sweep should choose threshold maximizing validation F1."""
    frames_gt = [[(0, 10.0, 10.0, 110.0, 110.0)], [(0, 20.0, 20.0, 120.0, 120.0)]]
    frames_preds = [
        [(0, 0.16, 10.0, 10.0, 110.0, 110.0), (0, 0.08, 200.0, 200.0, 240.0, 240.0)],
        [(0, 0.22, 20.0, 20.0, 120.0, 120.0)],
    ]
    cal = compute_calibration_curve_from_predictions(frames_gt, frames_preds)
    assert cal["best_f1"] == 1.0
    assert 0.10 <= cal["best_f1_conf"] <= 0.15


def test_onnx_metadata_and_raw_top_score_support_generic_models(tmp_path: Path) -> None:
    """Metadata payload uses dynamic experiment/model names and correct python/torch versions."""
    weights = tmp_path / "artifacts" / "runs" / "detect" / "exp05_demo" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"pt")
    onnx_out = tmp_path / "artifacts" / "models" / "onnx" / "exp05_demo_960.onnx"
    onnx_out.parent.mkdir(parents=True)
    onnx_out.write_bytes(b"onnx")

    payload = build_model_metadata_payload(
        model_name="exp05_demo_960",
        source_experiment="exp05_demo",
        limitations_notice="Val-only prototype",
        weights_path=weights,
        output_onnx_path=onnx_out,
        source_sha256="a" * 64,
        export_info={
            "onnxSha256": "b" * 64,
            "onnxFileSizeBytes": 4,
            "opsetImports": [{"domain": "ai.onnx", "version": 17}],
            "end2end": False,
            "graphInputs": [{"dims": [1, 3, 960, 960]}],
            "graphOutputs": [{"dims": [1, 5, 18900]}],
            "classNames": {0: "open_top_freight_wagon"},
        },
        input_name="images",
        output_name="output0",
        args_opset=17,
        args_imgsz=960,
        args_conf=0.25,
        args_iou=0.45,
        repo_root=tmp_path,
    )
    assert payload["modelName"] == "exp05_demo_960"
    assert payload["sourceExperiment"] == "exp05_demo"
    assert payload["versions"]["python"] == platform.python_version()
    assert "torch" in payload["versions"]

    standard = np.zeros((1, 5, 10), dtype=np.float32)
    standard[0, 4, 3] = 0.77
    assert extract_raw_top_score(standard) == pytest.approx(0.77, abs=1e-5)

    end2end = np.zeros((1, 8, 6), dtype=np.float32)
    end2end[0, 2, 4] = 0.91
    assert extract_raw_top_score(end2end) == pytest.approx(0.91, abs=1e-5)


def test_tracker_yaml_scales_buffer_by_fps() -> None:
    """Two-stage tracker YAML scales track_buffer proportionally to effective FPS."""
    cfg_5 = build_tracker_config_text(tracker_type="bytetrack", high_thresh=0.25, low_thresh=0.10, effective_fps=5.0)
    cfg_25 = build_tracker_config_text(tracker_type="botsort", high_thresh=0.30, low_thresh=0.10, effective_fps=25.0)
    assert "track_buffer: 8" in cfg_5
    assert "track_low_thresh: 0.1000" in cfg_5
    assert "track_buffer: 38" in cfg_25
    assert "gmc_method: sparseOptFlow" in cfg_25
