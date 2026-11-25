"""Validate CVAT YOLO Detection and MOT 1.1 exports and ingest verified data without leakage."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_utils import save_labeled_grid  # noqa: E402

IGNORED_TXT_NAMES = {
    "train.txt",
    "val.txt",
    "validation.txt",
    "test.txt",
    "trainval.txt",
    "default.txt",
    "classes.txt",
    "obj.names",
    "names.txt",
}


@contextmanager
def resolved_export_dir(export_path: Path) -> Iterator[Path]:
    """Yield a directory from either an extracted folder or a CVAT YOLO .zip archive."""
    if export_path.is_file() and export_path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(export_path) as zf:
                zf.extractall(tmp)
            yield Path(tmp)
    elif export_path.is_dir():
        yield export_path
    else:
        raise FileNotFoundError(f"Export directory or .zip not found: {export_path}")


@contextmanager
def resolved_mot_file(mot_path: Path) -> Iterator[Path]:
    """Yield gt.txt directly or locate gt.txt inside a CVAT MOT 1.1 .zip archive."""
    if mot_path.is_file() and mot_path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(mot_path) as zf:
                zf.extractall(tmp)
            tmp_root = Path(tmp)
            gt_candidates = sorted(tmp_root.rglob("gt.txt"))
            if not gt_candidates:
                gt_candidates = sorted(p for p in tmp_root.rglob("*.txt") if p.name.lower() != "labels.txt")
            if not gt_candidates:
                raise FileNotFoundError(f"No gt.txt found inside MOT archive: {mot_path}")
            yield gt_candidates[0]
    elif mot_path.is_file():
        yield mot_path
    else:
        raise FileNotFoundError(f"MOT file or .zip not found: {mot_path}")


def _validate_yolo_detection_folder(
    export_dir: Path,
    expected_filenames: set[str],
    smoke_mode: bool = False,
) -> dict[str, Any]:
    for meta_name in ("classes.txt", "obj.names"):
        for mp in export_dir.rglob(meta_name):
            lines = [ln.strip() for ln in mp.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines != ["open_top_freight_wagon"]:
                raise ValueError(f"Unexpected class names in {mp}: {lines} (expected ['open_top_freight_wagon'])")

    for yp in export_dir.rglob("data.yaml"):
        if "open_top_freight_wagon" not in yp.read_text(encoding="utf-8"):
            raise ValueError(f"Expected class 'open_top_freight_wagon' in {yp}")

    train_txt_images: list[str] = []
    for tp in export_dir.rglob("train.txt"):
        listed = [Path(ln.strip()).name for ln in tp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        unknown = sorted(set(listed) - expected_filenames)
        if unknown:
            raise ValueError(f"Unexpected filenames in {tp}: {unknown[:5]}")
        train_txt_images.extend(listed)

    expected_stems = {Path(fn).stem: fn for fn in expected_filenames}
    txt_files = [p for p in export_dir.rglob("*.txt") if p.name.lower() not in IGNORED_TXT_NAMES]

    label_by_stem: dict[str, Path] = {}
    boxes_per_stem: dict[str, list[tuple[int, float, float, float, float]]] = {}
    total_boxes = 0

    for tp in sorted(txt_files):
        stem = tp.stem
        if stem not in expected_stems:
            raise ValueError(f"Unexpected label stem not in manifest: {tp}")
        if stem in label_by_stem:
            raise ValueError(f"Duplicate label file for stem {stem}: {tp} and {label_by_stem[stem]}")
        label_by_stem[stem] = tp

        parsed: list[tuple[int, float, float, float, float]] = []
        for lno, line in enumerate(tp.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 5:
                raise ValueError(f"{tp}:{lno} expected 5 tokens, got {len(parts)}")
            cid = int(parts[0])
            if cid != 0:
                raise ValueError(f"{tp}:{lno} invalid class_id={cid}; only class 0 is allowed")
            cx, cy, w, h = (float(v) for v in parts[1:])
            if not all(math.isfinite(v) for v in (cx, cy, w, h)):
                raise ValueError(f"{tp}:{lno} contains non-finite coordinates")
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                raise ValueError(f"{tp}:{lno} out-of-range normalized box: {cx}, {cy}, {w}, {h}")
            parsed.append((cid, cx, cy, w, h))
        boxes_per_stem[stem] = parsed
        total_boxes += len(parsed)

    missing_stems = sorted(set(expected_stems.keys()) - set(label_by_stem.keys()))
    if smoke_mode and total_boxes == 0:
        raise ValueError("Smoke detection export contains 0 boxes; draw at least 1 box to verify round-trip.")

    return {
        "expected_images": len(expected_filenames),
        "train_txt_images": len(train_txt_images),
        "found_label_files": len(label_by_stem),
        "missing_label_files": missing_stems,
        "frames_with_boxes": sum(1 for v in boxes_per_stem.values() if len(v) > 0),
        "empty_label_files": sum(1 for v in boxes_per_stem.values() if len(v) == 0),
        "total_boxes": total_boxes,
        "label_paths": label_by_stem,
        "boxes_by_stem": boxes_per_stem,
    }


def validate_yolo_detection_export(
    export_dir: Path,
    expected_filenames: set[str],
    smoke_mode: bool = False,
) -> dict[str, Any]:
    """Verify Ultralytics YOLO export (folder or .zip) contains only class 0, normalized boxes, and valid stems."""
    with resolved_export_dir(export_dir) as resolved:
        return _validate_yolo_detection_folder(resolved, expected_filenames, smoke_mode=smoke_mode)


def _validate_mot_tracking_txt(
    mot_file: Path,
    expected_num_frames: int,
    img_w: int = 1920,
    img_h: int = 1080,
) -> dict[str, Any]:
    seen_pairs: set[tuple[int, int]] = set()
    frames_by_track: dict[int, list[int]] = defaultdict(list)
    rows_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    warnings: list[str] = []
    row_count = 0

    for lno, raw in enumerate(mot_file.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        parts = [p.strip() for p in stripped.split(",")]
        if len(parts) < 6:
            raise ValueError(f"{mot_file}:{lno} expected >= 6 CSV columns, got {len(parts)}")
        try:
            fid = int(parts[0])
            tid = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"{mot_file}:{lno} frame_id and track_id must be integers") from exc

        if not (1 <= fid <= expected_num_frames):
            raise ValueError(f"{mot_file}:{lno} frame_id={fid} out of expected 1..{expected_num_frames}")
        if tid < 1:
            raise ValueError(f"{mot_file}:{lno} invalid track_id={tid}; expected >= 1")
        if (fid, tid) in seen_pairs:
            raise ValueError(f"{mot_file}:{lno} duplicate (frame_id={fid}, track_id={tid})")
        seen_pairs.add((fid, tid))

        x, y, w, h = (float(v) for v in parts[2:6])
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0.0 or h <= 0.0:
            raise ValueError(f"{mot_file}:{lno} invalid box geometry x={x}, y={y}, w={w}, h={h}")
        if x + w <= 0.0 or y + h <= 0.0 or x >= img_w or y >= img_h:
            raise ValueError(f"{mot_file}:{lno} box completely outside {img_w}x{img_h}: {x},{y},{w},{h}")

        if len(parts) >= 8:
            cls_val = int(float(parts[7]))
            if cls_val not in (1, 0):
                raise ValueError(f"{mot_file}:{lno} unexpected MOT class_id={cls_val}")
            if cls_val == 0:
                warnings.append(f"line {lno}: MOT class_id=0 (CVAT MOT 1.1 usually exports 1-based class_id=1)")

        frames_by_track[tid].append(fid)
        rows_by_frame[fid].append((tid, x, y, w, h))
        row_count += 1

    if row_count == 0:
        raise ValueError(f"MOT file {mot_file} has 0 annotations")

    track_summary: dict[int, dict[str, Any]] = {}
    for tid in sorted(frames_by_track.keys()):
        flist = sorted(frames_by_track[tid])
        gaps = sum(1 for a, b in zip(flist, flist[1:]) if (b - a) > 1)
        track_summary[tid] = {
            "first_frame": flist[0],
            "last_frame": flist[-1],
            "annotations": len(flist),
            "internal_frame_gaps": gaps,
        }

    return {
        "mot_file": str(mot_file),
        "rows": row_count,
        "expected_num_frames": expected_num_frames,
        "frames_with_annotations": len(rows_by_frame),
        "unique_track_ids": sorted(track_summary.keys()),
        "tracks": track_summary,
        "warnings": warnings[:20],
        "rows_by_frame": dict(rows_by_frame),
    }


def validate_mot_tracking_export(
    mot_file: Path,
    expected_num_frames: int,
    img_w: int = 1920,
    img_h: int = 1080,
) -> dict[str, Any]:
    """Validate CVAT MOT 1.1 tracking export from gt.txt or .zip archive."""
    with resolved_mot_file(mot_file) as resolved:
        return _validate_mot_tracking_txt(resolved, expected_num_frames=expected_num_frames, img_w=img_w, img_h=img_h)


def render_qa_overlays(
    root: Path,
    yolo_root: Path,
    mot_summary: dict[str, Any],
) -> list[str]:
    """Render QA contact sheets of ingested detection labels and tracking GT for manual Condition #3 review."""
    triage_dir = root / "data/processed/triage-frames"
    trk_dir = root / "data/processed/open-top-v1/cvat_tasks/task2_tracking_vid09_5fps_61frames"
    out_paths: list[str] = []

    for sp in ("train", "val", "test"):
        img_paths = sorted((yolo_root / sp / "images").glob("*.jpg"))
        step = max(1, len(img_paths) // 12)
        picks = img_paths[::step][:12]
        tiles: list[tuple[str, Any]] = []
        for ip in picks:
            bgr = cv2.imread(str(ip))
            ih, iw = bgr.shape[:2]
            lbl = yolo_root / sp / "labels" / f"{ip.stem}.txt"
            nbox = 0
            if lbl.is_file():
                for ln in lbl.read_text(encoding="utf-8").splitlines():
                    if not ln.strip():
                        continue
                    _, cx, cy, w, h = (float(x) for x in ln.split())
                    x1, y1 = int((cx - w / 2) * iw), int((cy - h / 2) * ih)
                    x2, y2 = int((cx + w / 2) * iw), int((cy + h / 2) * ih)
                    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 215, 255), 3)
                    nbox += 1
            tiles.append((f"{ip.stem[:36]} | boxes={nbox}", bgr))
        out_p = triage_dir / f"qa_detection_{sp}.jpg"
        save_labeled_grid(tiles, out_p, cols=4)
        out_paths.append(str(out_p.relative_to(root)).replace("\\", "/"))

    rows_by_frame = mot_summary["rows_by_frame"]
    trk_tiles: list[tuple[str, Any]] = []
    for fid in (1, 13, 25, 37, 49, 61):
        img_p = trk_dir / f"{fid:06d}.jpg"
        bgr = cv2.imread(str(img_p))
        items = rows_by_frame.get(fid, [])
        for tid, x, y, w, h in items:
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (50, 230, 110), 3)
            cv2.putText(bgr, f"ID:{tid}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 230, 110), 2)
        trk_tiles.append((f"mot={fid:02d}/61 | tracks={len(items)}", bgr))
    out_trk = triage_dir / "qa_tracking_vid09_5fps.jpg"
    save_labeled_grid(trk_tiles, out_trk, cols=3)
    out_paths.append(str(out_trk.relative_to(root)).replace("\\", "/"))
    return out_paths


def ingest_full(root: Path, manifest: dict[str, Any], det_export_dir: Path, mot_file: Path) -> None:
    expected_det = {r["filename"] for r in manifest["detection_frames"]}
    num_trk = int(manifest["tracking_policy"]["num_frames"])

    campaign_by_split: dict[str, set[str]] = defaultdict(set)
    for rec in manifest["detection_frames"]:
        campaign_by_split[rec["split"]].add(rec["campaign"])
    assert campaign_by_split["train"].isdisjoint(campaign_by_split["val"])
    assert campaign_by_split["train"].isdisjoint(campaign_by_split["test"])
    assert campaign_by_split["val"].isdisjoint(campaign_by_split["test"])

    with resolved_export_dir(det_export_dir) as resolved_det, resolved_mot_file(mot_file) as resolved_mot:
        det_res = _validate_yolo_detection_folder(resolved_det, expected_det, smoke_mode=False)
        mot_res = _validate_mot_tracking_txt(resolved_mot, expected_num_frames=num_trk, img_w=1920, img_h=1080)

        src_det_dir = root / "data/processed/open-top-v1/cvat_tasks/task1_detection_141imgs"
        yolo_root = root / "data/processed/open-top-yolo-v1"
        if yolo_root.exists():
            shutil.rmtree(yolo_root)
        for sp in ("train", "val", "test"):
            (yolo_root / sp / "images").mkdir(parents=True, exist_ok=True)
            (yolo_root / sp / "labels").mkdir(parents=True, exist_ok=True)

        missing_treated_empty = 0
        for rec in manifest["detection_frames"]:
            sp = rec["split"]
            fname = rec["filename"]
            if sp in ("train", "val"):
                assert not rec["video_short"].startswith("vid_09") and not rec["video_short"].startswith("vid_16"), "Leakage blocked"
            src_img = src_det_dir / fname
            if not src_img.is_file():
                raise FileNotFoundError(f"Source extracted frame missing: {src_img}")
            shutil.copy2(src_img, yolo_root / sp / "images" / fname)
            stem = Path(fname).stem
            dst_lbl = yolo_root / sp / "labels" / f"{stem}.txt"
            if stem in det_res["label_paths"]:
                shutil.copy2(det_res["label_paths"][stem], dst_lbl)
            else:
                missing_treated_empty += 1
                dst_lbl.write_text("", encoding="utf-8")

        gt_dst = yolo_root / "tracking_gt/vid09_5fps_gt.txt"
        gt_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved_mot, gt_dst)

    data_yaml = "\n".join([
        f"path: {yolo_root.resolve()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "names:",
        "  0: open_top_freight_wagon",
        "",
    ])
    (yolo_root / "data.yaml").write_text(data_yaml, encoding="utf-8")

    qa_grids = render_qa_overlays(root, yolo_root, mot_res)
    summary = {
        "detection_validation": {k: v for k, v in det_res.items() if k not in ("label_paths", "boxes_by_stem")},
        "missing_txt_written_as_empty": missing_treated_empty,
        "tracking_validation": {k: v for k, v in mot_res.items() if k != "rows_by_frame"},
        "yolo_root": str(yolo_root.relative_to(root)).replace("\\", "/"),
        "tracking_gt": str(gt_dst.relative_to(root)).replace("\\", "/"),
        "qa_contact_sheets": qa_grids,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def ingest_generalized_pack(
    *,
    manifest: dict[str, Any],
    packs_dir: Path | str,
    det_export_dir: Path | str,
    output_dir: Path | str,
    confirm_human_reviewed: bool,
) -> dict[str, Any]:
    """Validate and ingest a generalized HITL-reviewed CVAT YOLO detection export into a target dataset root."""
    packs_dir = Path(packs_dir)
    det_export_dir = Path(det_export_dir)
    output_dir = Path(output_dir)
    if not confirm_human_reviewed:
        raise PermissionError(
            "Ingestão bloqueada: passe --confirm-human-reviewed após revisão humana de 100% das anotações no CVAT."
        )

    prov_path = det_export_dir / "preannotation_provenance.json" if det_export_dir.is_dir() else None
    if prov_path and prov_path.is_file():
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        if prov.get("humanReviewed") is False:
            raise PermissionError(
                f"Arquivo {prov_path} marca humanReviewed=false. Confirme a revisão humana antes da ingestão."
            )

    pack_manifest_path = packs_dir / "pack_manifest.json"
    if not pack_manifest_path.is_file():
        raise FileNotFoundError(f"pack_manifest.json não encontrado em {packs_dir}")
    pack = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
    frames = list(pack.get("frames") or [])
    if not frames:
        raise ValueError("pack_manifest.json não possui frames para ingestão.")

    expected_filenames: set[str] = set()
    campaign_by_split: dict[str, set[str]] = defaultdict(set)
    reserved = ("vid_01", "vid_09", "vid_16")
    for rec in frames:
        split = str(rec.get("split") or "")
        vid_id = str(rec.get("video_id") or "")
        campaign = str(rec.get("campaign_id") or "")
        fname = str(rec.get("filename") or "")
        if split in ("train", "val"):
            if any(vid_id.lower().startswith(r) or r in fname.lower() for r in reserved):
                raise ValueError(f"Isolamento violado: vídeo reservado '{vid_id}' proibido em split '{split}'.")
        if campaign and split:
            campaign_by_split[split].add(campaign)
        if fname:
            expected_filenames.add(fname)

    # Include manifest cross-split campaign checks if supplied
    for rec in manifest.get("videos") or []:
        sp = str(rec.get("target_split") or rec.get("split") or "")
        cp = str(rec.get("campaign_id") or "")
        if cp and sp:
            campaign_by_split[sp].add(cp)

    for a, b in (("train", "val"), ("train", "test"), ("val", "test"), ("train", "test_negative_only"), ("val", "test_negative_only")):
        if not campaign_by_split[a].isdisjoint(campaign_by_split[b]):
            overlap = sorted(campaign_by_split[a] & campaign_by_split[b])
            raise ValueError(f"Vazamento de campanha entre splits {a}/{b}: {overlap}")

    with resolved_export_dir(det_export_dir) as resolved_det:
        det_res = _validate_yolo_detection_folder(resolved_det, expected_filenames, smoke_mode=False)
        ingested = 0
        empty_labels = 0
        for rec in frames:
            sp = str(rec.get("split") or "train")
            fname = str(rec["filename"])
            src_img = packs_dir / fname
            if not src_img.is_file():
                raise FileNotFoundError(f"Frame ausente no pack: {src_img}")
            dst_img = output_dir / sp / "images" / fname
            dst_lbl = output_dir / sp / "labels" / f"{Path(fname).stem}.txt"
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dst_img)
            stem = Path(fname).stem
            if stem in det_res["label_paths"]:
                shutil.copy2(det_res["label_paths"][stem], dst_lbl)
            else:
                empty_labels += 1
                dst_lbl.write_text("", encoding="utf-8")
            ingested += 1

    data_yaml = "\n".join(
        [
            f"path: {output_dir.resolve()}",
            "train: train/images",
            "val: val/images",
            "test: test/images",
            "names:",
            "  0: open_top_freight_wagon",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    return {
        "ingested_frames": ingested,
        "empty_labels_written": empty_labels,
        "output_dir": str(output_dir),
        "detection_validation": {k: v for k, v in det_res.items() if k not in ("label_paths", "boxes_by_stem")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["smoke-detection", "smoke-tracking", "ingest-full", "ingest-pack"])
    parser.add_argument("--export-dir", help="CVAT Ultralytics YOLO export directory or .zip (smoke-detection)")
    parser.add_argument("--det-export-dir", help="CVAT Ultralytics YOLO export directory or .zip (ingest-full / ingest-pack)")
    parser.add_argument("--mot-file", help="Path to CVAT MOT 1.1 gt.txt or .zip archive")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional video manifest JSON for generalized pack ingestion")
    parser.add_argument("--packs-dir", type=Path, default=None, help="Directory with pack_manifest.json and clean frames")
    parser.add_argument("--output-dir", type=Path, default=None, help="Target YOLO dataset directory for ingest-pack")
    parser.add_argument("--confirm-human-reviewed", action="store_true", help="Explicit confirmation that 100% of labels were human-reviewed in CVAT")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "specs/data-manifests/open-top-split-v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if args.mode == "smoke-detection":
        if not args.export_dir:
            raise ValueError("--export-dir is required for --mode smoke-detection")
        expected = set(manifest["smoke_test"]["detection_filenames"])
        res = validate_yolo_detection_export(Path(args.export_dir), expected, smoke_mode=True)
        print(json.dumps({k: v for k, v in res.items() if k not in ("label_paths", "boxes_by_stem")}, indent=2, ensure_ascii=False))
    elif args.mode == "smoke-tracking":
        if not args.mot_file:
            raise ValueError("--mot-file is required for --mode smoke-tracking")
        expected_n = int(manifest["smoke_test"]["tracking_num_frames"])
        res = validate_mot_tracking_export(Path(args.mot_file), expected_num_frames=expected_n, img_w=1920, img_h=1080)
        print(json.dumps({k: v for k, v in res.items() if k != "rows_by_frame"}, indent=2, ensure_ascii=False))
    elif args.mode == "ingest-pack":
        if not args.det_export_dir or not args.packs_dir or not args.output_dir:
            raise ValueError("--det-export-dir, --packs-dir, and --output-dir are required for --mode ingest-pack")
        custom_manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else manifest
        summary = ingest_generalized_pack(
            manifest=custom_manifest,
            packs_dir=Path(args.packs_dir),
            det_export_dir=Path(args.det_export_dir),
            output_dir=Path(args.output_dir),
            confirm_human_reviewed=args.confirm_human_reviewed,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        if not args.det_export_dir or not args.mot_file:
            raise ValueError("--det-export-dir and --mot-file are required for --mode ingest-full")
        if not args.confirm_human_reviewed:
            raise PermissionError("Ingestão bloqueada: --confirm-human-reviewed é obrigatório após revisão humana 100% no CVAT.")
        ingest_full(root, manifest, Path(args.det_export_dir), Path(args.mot_file))


if __name__ == "__main__":
    main()
