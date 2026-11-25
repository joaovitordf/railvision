"""WS1 ML tests for user-video registration, pack extraction, pre-annotation, and leakage guards."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
import pytest

ML_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(ML_SCRIPTS))

from benchmark_open_vocab_preannotators import assert_no_reserved_benchmark_paths, compute_detection_metrics  # noqa: E402
from extract_dataset_packs import compute_dhash, extract_packs_from_manifests, hamming_distance  # noqa: E402
from preannotate_cvat import generate_preannotations  # noqa: E402
from register_user_video import register_video  # noqa: E402
from validate_cvat_exports import ingest_generalized_pack  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
VAL_SAMPLE = REPO_ROOT / "artifacts/runs/phase4-onnx-spike/parity_frames/val__camp_sishen_saldanha__vid_15__f000075.jpg"
VID15_SAMPLE = REPO_ROOT / "data/raw/public-videos/vid_15_sishen_saldanha_railway_line.ogv"


def test_register_user_video_enforces_provenance_sha_and_split_hygiene(tmp_path: Path) -> None:
    """User video manifest registration detects SHA duplicates, cross-split leakage, and reserved benchmarks."""
    assert VID15_SAMPLE.is_file()
    manifest_p = tmp_path / "user-videos-manifest.json"
    pub_manifest = REPO_ROOT / "specs/data-manifests/public-videos-manifest.json"

    user_vid = tmp_path / "user_yard_clip_01.ogv"
    user_vid.write_bytes(VID15_SAMPLE.read_bytes() + b"\x99")

    entry = register_video(
        video_path=user_vid,
        video_id="usr_vid_01",
        campaign_id="camp_user_demo_a",
        origin="user_owned",
        provenance="Gravação própria pátio A",
        target_split="train",
        expected_content="positive_open_top",
        manifest_path=manifest_p,
        public_manifest_path=pub_manifest,
    )
    assert entry["auto_train_eligible"] is False
    assert entry["human_reviewed"] is False
    assert entry["media"]["width"] > 0

    # Duplicate SHA-256 blocked even with different ID/campaign (both vs user & public manifests)
    with pytest.raises(ValueError, match="SHA-256"):
        register_video(
            video_path=user_vid,
            video_id="usr_vid_02",
            campaign_id="camp_user_demo_b",
            origin="user_owned",
            provenance="Cópia",
            target_split="train",
            expected_content="positive_open_top",
            manifest_path=manifest_p,
            public_manifest_path=pub_manifest,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        register_video(
            video_path=VID15_SAMPLE,
            video_id="usr_vid_pubdup",
            campaign_id="camp_user_pubdup",
            origin="public_curated",
            provenance="Duplicata de vid_15",
            target_split="train",
            expected_content="positive_open_top",
            manifest_path=manifest_p,
            public_manifest_path=pub_manifest,
        )

    # Reserved vid_01/vid_09/vid_16 blocked from train/val
    fake_vid01 = tmp_path / "vid_01_reserved_copy.ogv"
    fake_vid01.write_bytes(VID15_SAMPLE.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="Isolamento violado"):
        register_video(
            video_path=fake_vid01,
            video_id="vid_01_copy",
            campaign_id="camp_res",
            origin="user_owned",
            provenance="Teste",
            target_split="train",
            expected_content="hard_negative",
            manifest_path=manifest_p,
        )

    # Cross-split campaign leakage blocked
    other_vid = tmp_path / "custom_second.ogv"
    other_vid.write_bytes(VID15_SAMPLE.read_bytes() + b"\x01")
    with pytest.raises(ValueError, match="Vazamento de campanha"):
        register_video(
            video_path=other_vid,
            video_id="usr_vid_03",
            campaign_id="camp_user_demo_a",
            origin="user_owned",
            provenance="Mesma campanha em val",
            target_split="val",
            expected_content="positive_open_top",
            manifest_path=manifest_p,
        )


def test_preannotate_cvat_builds_yolo11_zip_provenance_and_uncertainty_ranking(tmp_path: Path) -> None:
    """preannotate_cvat generates single-class YOLO 1.1 zip, humanReviewed=false provenance, and uncertainty sorting."""
    assert VAL_SAMPLE.is_file()
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    img1 = img_dir / "frame_a.jpg"
    img2 = img_dir / "frame_b.jpg"
    img1.write_bytes(VAL_SAMPLE.read_bytes())
    img2.write_bytes(VAL_SAMPLE.read_bytes())

    def _dummy_pred(bgr: np.ndarray, _conf: float, _iou: float):
        # frame_a vs frame_b alternating confidence
        if not hasattr(_dummy_pred, "calls"):
            _dummy_pred.calls = 0
        _dummy_pred.calls += 1
        if _dummy_pred.calls == 1:
            return [(100.0, 100.0, 300.0, 250.0, 0.92)]
        return [(100.0, 100.0, 300.0, 250.0, 0.30)]

    out_dir = tmp_path / "preann"
    res = generate_preannotations(
        image_paths=[img1, img2],
        output_dir=out_dir,
        predict_fn=_dummy_pred,
        engine_name="exp04c",
        weights_ref="dummy.pt",
        weights_sha256="deadbeef",
        conf=0.25,
        iou=0.45,
        device="cpu",
        mot_tracking=False,
    )
    assert res["frameCount"] == 2
    assert res["totalPreannotatedBoxes"] == 2

    with zipfile.ZipFile(out_dir / "cvat_yolo11_preannotations.zip") as zf:
        names = set(zf.namelist())
        assert "obj.names" in names
        assert zf.read("obj.names").decode("utf-8").strip() == "open_top_freight_wagon"
        assert "obj_train_data/frame_a.txt" in names
        assert zf.read("obj_train_data/frame_a.txt").decode("utf-8").startswith("0 ")

    prov = json.loads((out_dir / "preannotation_provenance.json").read_text(encoding="utf-8"))
    assert prov["humanReviewed"] is False
    assert prov["requiresFullHumanReview"] is True

    ranking = json.loads((out_dir / "uncertainty_ranking.json").read_text(encoding="utf-8"))
    assert ranking["items"][0]["filename"] == "frame_b.jpg"


def test_extract_packs_dedup_and_validate_cvat_exports_hitl_gate(tmp_path: Path) -> None:
    """Pack extractor deduplicates near-identical frames and validate_cvat_exports blocks unreviewed/leaky exports."""
    a = np.full((64, 64, 3), 120, dtype=np.uint8)
    cv2.rectangle(a, (10, 10), (40, 40), (255, 255, 255), -1)
    assert hamming_distance(compute_dhash(a), compute_dhash(a.copy())) == 0

    # Manifest containing vid_09 must fail if train extraction is requested on it
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(
        json.dumps({"videos": [{"video_id": "vid_09_test", "target_split": "train", "local_path": str(VID15_SAMPLE)}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Extração bloqueada"):
        extract_packs_from_manifests(
            manifest_paths=[bad_manifest],
            output_dir=tmp_path / "packs",
            allowed_splits={"train"},
            sample_fps=1.0,
        )

    # Valid pack extraction from vid_15 labeled as train in a temp isolated manifest
    good_manifest = tmp_path / "good_manifest.json"
    good_manifest.write_text(
        json.dumps(
            {
                "videos": [
                    {
                        "video_id": "custom_train_clip",
                        "campaign_id": "camp_custom_1",
                        "target_split": "train",
                        "local_path": str(VID15_SAMPLE),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    packs_dir = tmp_path / "packs_good"
    pack_summary = extract_packs_from_manifests(
        manifest_paths=[good_manifest],
        output_dir=packs_dir,
        allowed_splits={"train"},
        sample_fps=2.0,
        max_frames_per_video=3,
        dedup_hamming_threshold=2,
    )
    assert pack_summary["frame_count"] >= 1

    export_dir = tmp_path / "cvat_export"
    (export_dir / "obj_train_data").mkdir(parents=True)
    (export_dir / "obj.names").write_text("open_top_freight_wagon\n", encoding="utf-8")
    train_names = [f["filename"] for f in pack_summary["frames"]]
    (export_dir / "train.txt").write_text("\n".join(f"data/obj_train_data/{fn}" for fn in train_names) + "\n", encoding="utf-8")
    for fn in train_names:
        (export_dir / "obj_train_data" / f"{Path(fn).stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    # Refuses without --confirm-human-reviewed
    with pytest.raises(PermissionError, match="confirm-human-reviewed"):
        ingest_generalized_pack(
            manifest=json.loads(good_manifest.read_text(encoding="utf-8")),
            packs_dir=packs_dir,
            det_export_dir=export_dir,
            output_dir=tmp_path / "out_ds",
            confirm_human_reviewed=False,
        )

    # Refuses if preannotation_provenance.json still has humanReviewed: false
    (export_dir / "preannotation_provenance.json").write_text(json.dumps({"humanReviewed": False}), encoding="utf-8")
    with pytest.raises(PermissionError, match="humanReviewed=false"):
        ingest_generalized_pack(
            manifest=json.loads(good_manifest.read_text(encoding="utf-8")),
            packs_dir=packs_dir,
            det_export_dir=export_dir,
            output_dir=tmp_path / "out_ds",
            confirm_human_reviewed=True,
        )

    # Succeeds when humanReviewed is cleared/true and confirmation flag passed
    (export_dir / "preannotation_provenance.json").unlink()
    out_res = ingest_generalized_pack(
        manifest=json.loads(good_manifest.read_text(encoding="utf-8")),
        packs_dir=packs_dir,
        det_export_dir=export_dir,
        output_dir=tmp_path / "out_ds",
        confirm_human_reviewed=True,
    )
    assert out_res["ingested_frames"] == len(train_names)
    assert (tmp_path / "out_ds" / "data.yaml").is_file()


def test_benchmark_guard_blocks_reserved_videos_and_metrics_math() -> None:
    """Benchmark guard forbids vid_01/vid_09/vid_16 and computes exact TP/FP/AP50."""
    for reserved in ("vid_01_test.jpg", "vid_09_frame.jpg", "vid_16_frame.jpg"):
        with pytest.raises(ValueError, match="Isolamento de benchmark violado"):
            assert_no_reserved_benchmark_paths([Path(f"data/{reserved}")])

    metrics = compute_detection_metrics(
        preds_by_stem={"a": [(10.0, 10.0, 50.0, 50.0, 0.9), (100.0, 100.0, 120.0, 120.0, 0.6)]},
        gts_by_stem={"a": np.asarray([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)},
    )
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert metrics["ap50"] == 1.0
