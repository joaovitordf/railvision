"""Tests guaranteeing zero data leakage, class sanity, and dedup integrity for TWIST event splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_twist_event_splits_have_zero_leakage_and_valid_classes() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "specs" / "data-manifests" / "twist-split-v1.json"
    assert manifest_path.is_file(), "twist-split-v1.json deve existir apos prepare_twist_dataset.py"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. Verify no passage event appears in multiple splits
    events_by_split: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    for ev in manifest["events"]:
        events_by_split[ev["split"]].add(ev["event_id"])
    assert events_by_split["train"].isdisjoint(events_by_split["val"])
    assert events_by_split["train"].isdisjoint(events_by_split["test"])
    assert events_by_split["val"].isdisjoint(events_by_split["test"])

    # 2. Verify zero SHA-256 overlap across splits on disk and valid remapped classes {0,1,2,3}
    ds_root = root / "data" / "processed" / "twist-yolo-v1"
    hashes: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        split_hashes: set[str] = set()
        imgs = list((ds_root / split / "images").glob("*"))
        assert len(imgs) > 0
        for im in imgs:
            split_hashes.add(hashlib.sha256(im.read_bytes()).hexdigest())
            lbl = ds_root / split / "labels" / f"{im.stem}.txt"
            assert lbl.is_file()
            for line in lbl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                assert len(parts) == 5
                cid = int(parts[0])
                assert cid in {0, 1, 2, 3}, f"Unexpected class id {cid} in {lbl}"
                cx, cy, w, h = (float(x) for x in parts[1:])
                assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0
        hashes[split] = split_hashes

    assert hashes["train"].isdisjoint(hashes["val"])
    assert hashes["train"].isdisjoint(hashes["test"])
    assert hashes["val"].isdisjoint(hashes["test"])

    # 3. Verify freight_car (class 2) exists in all three splits
    for split in ("train", "val", "test"):
        assert manifest["twist_yolo_v1_stats"][split]["class_counts"]["freight_car"] > 0

    # 4. Verify near-dedup variant keeps identical val and test sets
    near_root = root / "data" / "processed" / "twist-yolo-v1-dedup-near"
    for split in ("val", "test"):
        exact_names = sorted(p.name for p in (ds_root / split / "images").glob("*"))
        near_names = sorted(p.name for p in (near_root / split / "images").glob("*"))
        assert exact_names == near_names
