"""Unit tests for RailVision ML audit policies and manifest schemas."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_policies import (
    REQUIRED_VIDEO_MANIFEST_FIELDS,
    is_license_acceptable,
    validate_video_manifest_entry,
    validate_yolo_label_line,
)


def test_persisted_public_videos_manifest_is_valid_and_open_licensed() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / "specs" / "data-manifests" / "public-videos-manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(data["videos"]) >= 1
    for entry in data["videos"]:
        assert validate_video_manifest_entry(entry) == []
        ok, _ = is_license_acceptable(entry["license"], entry["license_url"])
        assert ok is True
        assert entry["angle_category"] != "pending_visual_inspection"
        assert isinstance(entry["cargo_visible"], bool)


def test_license_policy_accepts_explicit_open_licenses() -> None:
    assert is_license_acceptable("CC0", "https://creativecommons.org/publicdomain/zero/1.0/")[0] is True
    assert is_license_acceptable("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0")[0] is True
    assert is_license_acceptable("CC BY 3.0", "https://creativecommons.org/licenses/by/3.0")[0] is True
    assert is_license_acceptable("CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0")[0] is True


def test_license_policy_rejects_nc_nd_copyright_and_missing() -> None:
    assert is_license_acceptable("CC-BY-NC-ND-4.0", "")[0] is False
    assert is_license_acceptable("CC BY-NC 4.0", "https://creativecommons.org/licenses/by-nc/4.0")[0] is False
    assert is_license_acceptable("CC BY-ND 4.0", "")[0] is False
    assert is_license_acceptable("All rights reserved", "")[0] is False
    assert is_license_acceptable(None, None)[0] is False
    assert is_license_acceptable("", "")[0] is False


def test_yolo_label_line_validation() -> None:
    assert validate_yolo_label_line("2 0.50 0.50 0.25 0.40") == (True, 2, None)
    assert validate_yolo_label_line("0 1.20 0.50 0.25 0.40")[0] is False
    assert validate_yolo_label_line("0 0.50 0.50 0.0 0.40")[0] is False
    assert validate_yolo_label_line("-1 0.50 0.50 0.20 0.20")[0] is False
    assert validate_yolo_label_line("bad line")[0] is False


def test_video_manifest_required_fields_validation() -> None:
    complete = {field: "placeholder" for field in REQUIRED_VIDEO_MANIFEST_FIELDS}
    assert validate_video_manifest_entry(complete) == []

    incomplete = dict(complete)
    del incomplete["sha256"]
    del incomplete["angle_category"]
    assert validate_video_manifest_entry(incomplete) == ["sha256", "angle_category"]
