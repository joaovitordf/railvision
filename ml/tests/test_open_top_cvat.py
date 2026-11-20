"""Unit tests for Phase 3.1 open-top campaign split integrity and CVAT export validators."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_cvat_exports import validate_mot_tracking_export, validate_yolo_detection_export  # noqa: E402


def test_open_top_manifest_has_zero_campaign_leakage_and_5fps_tracking_order() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "specs/data-manifests/open-top-split-v1.json").read_text(encoding="utf-8"))

    assert manifest["class_map"] == {"0": "open_top_freight_wagon"}
    assert manifest["counts"] == {
        "detection_total": 141,
        "detection_by_split": {"train": 82, "val": 24, "test": 35},
        "tracking_frames": 61,
    }

    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for rec in manifest["detection_frames"]:
        by_split[rec["split"]].add(rec["campaign"])
        if rec["split"] in ("train", "val"):
            assert rec["video_short"] not in ("vid_09", "vid_16"), "camp_vfli_cosne leaked outside test"

    assert by_split["train"] == {"camp_coal_tunnel", "camp_perm_overpass"}
    assert by_split["val"] == {"camp_sishen_saldanha"}
    assert by_split["test"] == {"camp_vfli_cosne"}
    assert by_split["train"].isdisjoint(by_split["val"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["val"].isdisjoint(by_split["test"])

    policy = manifest["tracking_policy"]
    assert policy["excluded_from_training"] is True
    assert policy["sampled_fps"] == 5.0
    assert policy["num_frames"] == 61

    trk = manifest["tracking_frames"]
    assert len(trk) == 61
    assert [r["mot_frame_id"] for r in trk] == list(range(1, 62))
    assert [r["native_frame"] for r in trk] == list(range(300, 601, 5))


def test_validate_yolo_detection_export_valid_and_invalid_cases(tmp_path: Path) -> None:
    good = tmp_path / "good"
    (good / "labels/train").mkdir(parents=True)
    (good / "data.yaml").write_text("names:\n  0: open_top_freight_wagon\n", encoding="utf-8")
    (good / "train.txt").write_text("images/train/a.jpg\n", encoding="utf-8")
    (good / "labels/train/a.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
    (good / "labels/train/b.txt").write_text("", encoding="utf-8")

    res = validate_yolo_detection_export(good, {"a.jpg", "b.jpg"}, smoke_mode=True)
    assert res["total_boxes"] == 1
    assert res["frames_with_boxes"] == 1
    assert res["empty_label_files"] == 1
    assert res["missing_label_files"] == []

    # Invalid class id != 0
    bad_cls = tmp_path / "bad_cls"
    bad_cls.mkdir()
    (bad_cls / "a.txt").write_text("1 0.5 0.5 0.2 0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="class_id=1"):
        validate_yolo_detection_export(bad_cls, {"a.jpg"})

    # Out-of-range coordinate
    bad_coord = tmp_path / "bad_coord"
    bad_coord.mkdir()
    (bad_coord / "a.txt").write_text("0 1.2 0.5 0.2 0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="out-of-range"):
        validate_yolo_detection_export(bad_coord, {"a.jpg"})

    # Unknown filename stem
    bad_stem = tmp_path / "bad_stem"
    bad_stem.mkdir()
    (bad_stem / "unexpected.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected label stem"):
        validate_yolo_detection_export(bad_stem, {"a.jpg"})


def test_validate_mot_tracking_export_valid_and_invalid_cases(tmp_path: Path) -> None:
    good_mot = tmp_path / "gt.txt"
    good_mot.write_text(
        "1,1,100,200,300,150,1,1,1\n"
        "2,1,110,200,300,150,1,1,1\n"
        "2,2,500,180,250,140,1,1,1\n",
        encoding="utf-8",
    )
    res = validate_mot_tracking_export(good_mot, expected_num_frames=5, img_w=1920, img_h=1080)
    assert res["rows"] == 3
    assert res["unique_track_ids"] == [1, 2]
    assert res["tracks"][1]["first_frame"] == 1
    assert res["tracks"][1]["last_frame"] == 2

    # Out-of-range frame_id
    bad_frame = tmp_path / "bad_frame.txt"
    bad_frame.write_text("6,1,100,200,300,150,1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="out of expected 1..5"):
        validate_mot_tracking_export(bad_frame, expected_num_frames=5)

    # Duplicate (frame_id, track_id)
    dup_pair = tmp_path / "dup.txt"
    dup_pair.write_text("1,1,100,200,300,150,1,1,1\n1,1,120,200,300,150,1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        validate_mot_tracking_export(dup_pair, expected_num_frames=5)

    # Zero width
    bad_box = tmp_path / "bad_box.txt"
    bad_box.write_text("1,1,100,200,0,150,1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid box geometry"):
        validate_mot_tracking_export(bad_box, expected_num_frames=5)


def test_validators_accept_zip_archives_directly(tmp_path: Path) -> None:
    yolo_zip = tmp_path / "smoke_det.zip"
    with zipfile.ZipFile(yolo_zip, "w") as zf:
        zf.writestr("data.yaml", "names:\n  0: open_top_freight_wagon\npath: .\ntrain: train.txt\n")
        zf.writestr("train.txt", "data/images/train/a.jpg\ndata/images/train/b.jpg\n")
        zf.writestr("labels/train/a.txt", "0 0.5 0.5 0.3 0.25\n")

    det_res = validate_yolo_detection_export(yolo_zip, {"a.jpg", "b.jpg"}, smoke_mode=True)
    assert det_res["train_txt_images"] == 2
    assert det_res["found_label_files"] == 1
    assert det_res["missing_label_files"] == ["b"]
    assert det_res["total_boxes"] == 1

    mot_zip = tmp_path / "smoke_trk.zip"
    with zipfile.ZipFile(mot_zip, "w") as zf:
        zf.writestr("gt/gt.txt", "1,1,100,200,300,150,1,1,1\n2,1,105,200,300,150,1,1,1\n")

    mot_res = validate_mot_tracking_export(mot_zip, expected_num_frames=5)
    assert mot_res["rows"] == 2
    assert mot_res["unique_track_ids"] == [1]
