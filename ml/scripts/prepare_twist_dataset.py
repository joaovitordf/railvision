"""Build zero-leakage event-grouped TWIST YOLO datasets (exact-dedup baseline + train-only near-dedup ablation)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_policies import validate_yolo_label_line  # noqa: E402

# Remap ONLY TWIST classes compatible with their real annotations.
# Excluded intentionally: TWIST 0 (train - whole consist) and TWIST 6 (object - pedestrian crossing sign/fence).
# NEVER equate TWIST class 2 (wagon = passenger coach) or class 3 (freight car) with open-top bulk wagons.
TWIST_CLASS_REMAP: dict[int, int] = {
    1: 0,  # locomotive
    2: 1,  # passenger_wagon (TWIST "wagon")
    3: 2,  # freight_car (TWIST "freight car")
    4: 3,  # high_speed_train (TWIST "ICE")
}

TARGET_CLASS_NAMES: dict[int, str] = {
    0: "locomotive",
    1: "passenger_wagon",
    2: "freight_car",
    3: "high_speed_train",
}

TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})\.(\d{2})\.(\d{2})")


def compute_dhash64(image_path: Path) -> int:
    with Image.open(image_path) as im:
        gray = im.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        pixels = list(gray.getdata())
    bits = 0
    for row in range(8):
        off = row * 9
        for col in range(8):
            bits = (bits << 1) | (1 if pixels[off + col] > pixels[off + col + 1] else 0)
    return bits


def remap_label_file(label_path: Path) -> tuple[list[str], Counter[int], int]:
    """Read TWIST detailed label file, drop incompatible/invalid lines, and remap class IDs."""
    remapped_lines: list[str] = []
    class_counts: Counter[int] = Counter()
    invalid_count = 0

    for raw in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        ok, src_cls, _ = validate_yolo_label_line(raw)
        if not ok or src_cls is None:
            invalid_count += 1
            continue
        if src_cls not in TWIST_CLASS_REMAP:
            continue
        dst_cls = TWIST_CLASS_REMAP[src_cls]
        parts = raw.strip().split()
        remapped_lines.append(f"{dst_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
        class_counts[dst_cls] += 1

    return remapped_lines, class_counts, invalid_count


def assign_events_to_splits(events: list[dict[str, Any]]) -> dict[int, str]:
    """Assign each distinct train passage event to train / val / test without frame leakage."""
    assignment: dict[int, str] = {}

    # Freight-car events (5 distinct passage events in TWIST detailed_labels):
    # Ensure freight_car appears in train, val, and independent held-out test.
    freight_events = [ev for ev in events if ev["class_counts"].get(2, 0) > 0]
    if len(freight_events) >= 3:
        # Sort by freight_car count descending, then deterministic event_id:
        # Largest -> train; mid -> val; distinct held-out -> test; remaining -> train.
        by_fc = sorted(freight_events, key=lambda e: (-e["class_counts"][2], e["event_id"]))
        assignment[by_fc[0]["event_id"]] = "train"
        assignment[by_fc[1]["event_id"]] = "test"
        assignment[by_fc[2]["event_id"]] = "val"
        for extra in by_fc[3:]:
            assignment[extra["event_id"]] = "train"

    # Remaining non-freight events stratified deterministically across val (~15%), test (~15%), train (~70%)
    # Prioritize placing one ICE (class 3) event and one passenger_wagon (class 1) event in val and test.
    remaining = [ev for ev in events if ev["event_id"] not in assignment]

    ice_events = [ev for ev in remaining if ev["class_counts"].get(3, 0) > 0]
    if len(ice_events) >= 2:
        assignment[ice_events[0]["event_id"]] = "val"
        assignment[ice_events[1]["event_id"]] = "test"

    remaining = [ev for ev in events if ev["event_id"] not in assignment]
    for pos, ev in enumerate(remaining):
        mod = pos % 7
        if mod == 1:
            assignment[ev["event_id"]] = "val"
        elif mod == 4:
            assignment[ev["event_id"]] = "test"
        else:
            assignment[ev["event_id"]] = "train"

    return assignment


def write_dataset_variant(
    out_dir: Path,
    split_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if out_dir.exists():
        shutil.rmtree(out_dir)

    stats: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        img_dir = out_dir / split_name / "images"
        lbl_dir = out_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        cls_totals: Counter[int] = Counter()
        empty_files = 0
        for rec in split_records[split_name]:
            src_img: Path = rec["image_path"]
            shutil.copy2(src_img, img_dir / src_img.name)
            lines: list[str] = rec["remapped_lines"]
            (lbl_dir / f"{src_img.stem}.txt").write_text(
                ("\n".join(lines) + "\n") if lines else "",
                encoding="utf-8",
            )
            if not lines:
                empty_files += 1
            cls_totals.update(rec["remapped_counts"])

        stats[split_name] = {
            "images": len(split_records[split_name]),
            "empty_label_files": empty_files,
            "class_counts": {TARGET_CLASS_NAMES[k]: cls_totals.get(k, 0) for k in sorted(TARGET_CLASS_NAMES)},
        }

    yaml_lines = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "",
        "names:",
    ]
    for cid in sorted(TARGET_CLASS_NAMES):
        yaml_lines.append(f"  {cid}: {TARGET_CLASS_NAMES[cid]}")
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    src_base = root / "data" / "processed" / "twist-labelled" / "labelled_data" / "detailed_labels"
    out_exact = root / "data" / "processed" / "twist-yolo-v1"
    out_near = root / "data" / "processed" / "twist-yolo-v1-dedup-near"
    manifest_out = root / "specs" / "data-manifests" / "twist-split-v1.json"

    imgs = sorted((src_base / "images").glob("*"))
    lbls = {p.stem: p for p in (src_base / "labels").glob("*.txt")}

    seen_sha: dict[str, str] = {}
    exact_dups_removed: list[dict[str, str]] = []
    invalid_lines_total = 0
    items: list[dict[str, Any]] = []

    for im in imgs:
        sha = hashlib.sha256(im.read_bytes()).hexdigest()
        if sha in seen_sha:
            exact_dups_removed.append({"removed": im.name, "kept": seen_sha[sha], "sha256": sha})
            continue
        seen_sha[sha] = im.name

        m = TS_RE.search(im.name)
        if not m:
            raise RuntimeError(f"Nao foi possivel extrair timestamp de {im.name}")
        dt = datetime.strptime(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}", "%Y-%m-%d %H:%M:%S")
        lp = lbls[im.stem]
        lines, cls_counts, inv = remap_label_file(lp)
        invalid_lines_total += inv
        dhash = compute_dhash64(im)

        items.append(
            {
                "dt": dt,
                "image_path": im,
                "filename": im.name,
                "sha256": sha,
                "dhash": dhash,
                "remapped_lines": lines,
                "remapped_counts": cls_counts,
            }
        )

    items.sort(key=lambda x: (x["dt"], x["filename"]))

    # Group into passage events (gap > 120s)
    events: list[dict[str, Any]] = []
    for item in items:
        if not events or (item["dt"] - events[-1]["end_dt"]).total_seconds() > 120:
            events.append(
                {
                    "event_id": len(events),
                    "start_dt": item["dt"],
                    "end_dt": item["dt"],
                    "items": [item],
                    "class_counts": Counter(item["remapped_counts"]),
                }
            )
        else:
            ev = events[-1]
            ev["end_dt"] = item["dt"]
            ev["items"].append(item)
            ev["class_counts"].update(item["remapped_counts"])

    event_to_split = assign_events_to_splits(events)

    exact_split_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    near_split_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    near_dups_filtered_in_train = 0

    event_summaries: list[dict[str, Any]] = []
    for ev in events:
        sp = event_to_split[ev["event_id"]]
        exact_split_records[sp].extend(ev["items"])

        if sp != "train":
            near_split_records[sp].extend(ev["items"])
        else:
            kept_in_ev: list[dict[str, Any]] = []
            for item in ev["items"]:
                if not kept_in_ev:
                    kept_in_ev.append(item)
                else:
                    dist = (item["dhash"] ^ kept_in_ev[-1]["dhash"]).bit_count()
                    if dist > 2:
                        kept_in_ev.append(item)
                    else:
                        near_dups_filtered_in_train += 1
            near_split_records[sp].extend(kept_in_ev)

        event_summaries.append(
            {
                "event_id": ev["event_id"],
                "split": sp,
                "start": ev["start_dt"].isoformat(),
                "end": ev["end_dt"].isoformat(),
                "images": len(ev["items"]),
                "class_counts": {TARGET_CLASS_NAMES[k]: ev["class_counts"].get(k, 0) for k in sorted(TARGET_CLASS_NAMES)},
            }
        )

    exact_stats = write_dataset_variant(out_exact, exact_split_records)
    near_stats = write_dataset_variant(out_near, near_split_records)

    manifest = {
        "dataset": "TWIST detailed_labels (event-grouped, exact-dedup baseline + train-only near-dedup ablation)",
        "source_dir": str(src_base.relative_to(root)).replace("\\", "/"),
        "class_remap_policy": {
            "TWIST_0_train": "dropped_whole_consist",
            "TWIST_1_locomotive": "0: locomotive",
            "TWIST_2_wagon": "1: passenger_wagon",
            "TWIST_3_freight_car": "2: freight_car",
            "TWIST_4_ICE": "3: high_speed_train",
            "TWIST_6_object": "dropped_foreground_sign",
        },
        "source_total_images": len(imgs),
        "exact_duplicates_removed_count": len(exact_dups_removed),
        "exact_duplicates_removed": exact_dups_removed,
        "invalid_label_lines_dropped": invalid_lines_total,
        "near_duplicates_filtered_train_only_dhash_le_2": near_dups_filtered_in_train,
        "distinct_events_total": len(events),
        "twist_yolo_v1_stats": exact_stats,
        "twist_yolo_v1_dedup_near_stats": near_stats,
        "events": event_summaries,
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v for k, v in manifest.items() if k != "events"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
