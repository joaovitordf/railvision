"""Deterministic frame extraction and CVAT import package builder for Phase 3.1 (open_top_freight_wagon)."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_utils import save_labeled_grid  # noqa: E402


@dataclass(frozen=True)
class DetectionSpec:
    split: str
    campaign: str
    video_short: str
    video_id: str
    video_path: Path
    timestamp_s: float
    expected_role: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_frames_sequential(video_path: Path, target_frames: dict[int, Any]) -> dict[int, tuple[Any, float]]:
    """Decode exact frame indices sequentially (eliminates VP9/Theora seek drift)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    max_target = max(target_frames.keys())
    captured: dict[int, tuple[Any, float]] = {}
    idx = 0
    while idx <= max_target:
        if idx in target_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                raise RuntimeError(f"Failed reading frame {idx} from {video_path.name}")
            if float(frame.mean()) < 5.0:
                cap.release()
                raise RuntimeError(f"Extracted near-black frame {idx} from {video_path.name}")
            captured[idx] = (frame, round(idx / fps, 3))
        else:
            ok = cap.grab()
            if not ok:
                break
        idx += 1
    cap.release()
    missing = set(target_frames.keys()) - set(captured.keys())
    if missing:
        raise RuntimeError(f"Missing target frames {sorted(missing)} in {video_path.name}")
    return captured


def extract_frames_by_seek(video_path: Path, target_frames: list[int]) -> dict[int, tuple[Any, float]]:
    """Extract distant sparse frames from long videos via frame seek + decode."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 59.94)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    captured: dict[int, tuple[Any, float]] = {}
    for fidx in sorted(set(target_frames)):
        safe_idx = min(max(0, fidx), max(0, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Failed seeking frame {safe_idx} in {video_path.name}")
        if float(frame.mean()) < 5.0:
            cap.release()
            raise RuntimeError(f"Extracted near-black frame {safe_idx} from {video_path.name}")
        captured[safe_idx] = (frame, round(safe_idx / fps, 3))
    cap.release()
    return captured


def write_zip_from_files(files: list[Path], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in sorted(files, key=lambda x: x.name):
            zf.write(p, arcname=p.name)


def write_mp4_from_frames(frames_bgr: list[Any], out_mp4: Path, fps: float) -> None:
    if not frames_bgr:
        raise ValueError("Cannot build empty MP4")
    h, w = frames_bgr[0].shape[:2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed creating VideoWriter for {out_mp4}")
    for fr in frames_bgr:
        writer.write(fr)
    writer.release()

    # Verify exact decoded frame count
    cap = cv2.VideoCapture(str(out_mp4))
    decoded = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    if decoded != len(frames_bgr):
        raise RuntimeError(f"MP4 frame count mismatch for {out_mp4.name}: expected {len(frames_bgr)}, decoded {decoded}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data/raw/public-videos"
    out_base = root / "data/processed/open-top-v1/cvat_tasks"
    if out_base.exists():
        shutil.rmtree(out_base)
    det_dir = out_base / "task1_detection_141imgs"
    trk_dir = out_base / "task2_tracking_vid09_5fps_61frames"
    det_dir.mkdir(parents=True, exist_ok=True)
    trk_dir.mkdir(parents=True, exist_ok=True)

    videos = {
        "vid_14": ("vid_14_coal_train_locomotive_exit_the_tunnel_cr", raw_dir / "vid_14_coal_train_locomotive_exit_the_tunnel_cr.webm"),
        "vid_04": ("vid_04_2es10_003_with_freight_train_from_above", raw_dir / "vid_04_2es10_003_with_freight_train_from_above.webm"),
        "vid_15": ("vid_15_sishen_saldanha_railway_line", raw_dir / "vid_15_sishen_saldanha_railway_line.ogv"),
        "vid_09": ("vid_09_vossloh_g_1206_bb_of_vfli_video", raw_dir / "vid_09_vossloh_g_1206_bb_of_vfli_video.webm"),
        "vid_16": ("vid_16_vossloh_g_1206_bb_61735_and_bb_61744_vfl", raw_dir / "vid_16_vossloh_g_1206_bb_61735_and_bb_61744_vfl.webm"),
    }

    specs: list[DetectionSpec] = []
    # Train: camp_coal_tunnel (vid_14: 72) + camp_perm_overpass (vid_04: 10)
    for i in range(40):
        specs.append(DetectionSpec("train", "camp_coal_tunnel", "vid_14", videos["vid_14"][0], videos["vid_14"][1], 520.0 + 8.0 * i, "positive_elevated_coal"))
    for i in range(14):
        specs.append(DetectionSpec("train", "camp_coal_tunnel", "vid_14", videos["vid_14"][0], videos["vid_14"][1], 40.0 + 16.0 * i, "positive_roadside_coal"))
    for i in range(10):
        specs.append(DetectionSpec("train", "camp_coal_tunnel", "vid_14", videos["vid_14"][0], videos["vid_14"][1], 935.0 + 18.0 * i, "positive_occluded_coal"))
    for ts in [472.0, 478.0, 483.0, 488.0, 1143.0, 1148.0, 1153.0, 1158.0]:
        specs.append(DetectionSpec("train", "camp_coal_tunnel", "vid_14", videos["vid_14"][0], videos["vid_14"][1], ts, "hard_negative_or_loco_heavy"))
    for ts in [2.0, 14.0, 24.0, 34.0, 42.0, 46.0, 50.0, 53.0, 55.0, 56.4]:
        specs.append(DetectionSpec("train", "camp_perm_overpass", "vid_04", videos["vid_04"][0], videos["vid_04"][1], ts, "positive_parked_snow_with_closed_moving_train"))

    # Val: camp_sishen_saldanha (vid_15: 24)
    for i in range(19):
        specs.append(DetectionSpec("val", "camp_sishen_saldanha", "vid_15", videos["vid_15"][0], videos["vid_15"][1], 0.5 + 1.0 * i, "positive_iron_ore_gondolas"))
    for ts in [19.5, 20.5, 21.5, 22.5, 23.5]:
        specs.append(DetectionSpec("val", "camp_sishen_saldanha", "vid_15", videos["vid_15"][0], videos["vid_15"][1], ts, "tail_and_locomotive_transition"))

    # Test sparse detection: camp_vfli_cosne (vid_09: 25 + vid_16: 10)
    for i in range(25):
        specs.append(DetectionSpec("test", "camp_vfli_cosne", "vid_09", videos["vid_09"][0], videos["vid_09"][1], 1.0 + 2.0 * i, "positive_ballast_hoppers_and_tail_loco"))
    for ts in [2.0, 5.0, 8.0, 10.5]:
        specs.append(DetectionSpec("test", "camp_vfli_cosne", "vid_16", videos["vid_16"][0], videos["vid_16"][1], ts, "hard_negative_coupled_locomotives"))
    for ts in [13.5, 15.0, 16.5, 18.0, 19.2, 20.2]:
        specs.append(DetectionSpec("test", "camp_vfli_cosne", "vid_16", videos["vid_16"][0], videos["vid_16"][1], ts, "positive_lead_ballast_hopper"))

    # Map specs to exact native frames per video
    by_vid: dict[str, list[tuple[int, DetectionSpec]]] = {k: [] for k in videos}
    fps_map: dict[str, float] = {}
    for k, (_, vp) in videos.items():
        cap = cv2.VideoCapture(str(vp))
        fps_map[k] = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        cap.release()
    for sp in specs:
        fidx = int(round(sp.timestamp_s * fps_map[sp.video_short]))
        by_vid[sp.video_short].append((fidx, sp))

    # Extract detection frames
    decoded_det: dict[tuple[str, int], tuple[Any, float]] = {}
    for vk, items in by_vid.items():
        if not items:
            continue
        vp = videos[vk][1]
        if vk == "vid_14":
            got = extract_frames_by_seek(vp, [f for f, _ in items])
        else:
            got = extract_frames_sequential(vp, {f: True for f, _ in items})
        for f, tup in got.items():
            decoded_det[(vk, f)] = tup

    detection_records: list[dict[str, Any]] = []
    det_files: list[Path] = []
    sample_buckets: dict[str, list[tuple[str, Any]]] = {"train": [], "val": [], "test": []}
    seen_filenames: set[str] = set()

    for sp in specs:
        fidx = int(round(sp.timestamp_s * fps_map[sp.video_short]))
        frame_bgr, actual_ts = decoded_det[(sp.video_short, fidx)]
        fname = f"{sp.split}__{sp.campaign}__{sp.video_short}__f{fidx:06d}.jpg"
        assert fname not in seen_filenames, f"Duplicate filename generated: {fname}"
        seen_filenames.add(fname)
        out_img = det_dir / fname
        cv2.imwrite(str(out_img), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = frame_bgr.shape[:2]
        det_files.append(out_img)
        detection_records.append(
            {
                "filename": fname,
                "split": sp.split,
                "campaign": sp.campaign,
                "video_short": sp.video_short,
                "video_id": sp.video_id,
                "native_frame": fidx,
                "timestamp_s": actual_ts,
                "width": w,
                "height": h,
                "sha256": sha256_file(out_img),
                "expected_role": sp.expected_role,
            }
        )
        sample_buckets[sp.split].append((f"{sp.video_short} f={fidx} t={actual_ts:.1f}s ({sp.expected_role[:18]})", frame_bgr))

    # Extract dense tracking clip from vid_09: native frames 300..600 step 5 (61 frames @ 5 fps)
    trk_native = list(range(300, 601, 5))
    decoded_trk = extract_frames_sequential(videos["vid_09"][1], {f: True for f in trk_native})
    tracking_records: list[dict[str, Any]] = []
    trk_files: list[Path] = []
    trk_bgrs: list[Any] = []
    trk_grid_samples: list[tuple[str, Any]] = []

    for mot_idx, nf in enumerate(trk_native, start=1):
        frame_bgr, actual_ts = decoded_trk[nf]
        fname = f"{mot_idx:06d}.jpg"
        out_img = trk_dir / fname
        cv2.imwrite(str(out_img), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = frame_bgr.shape[:2]
        trk_files.append(out_img)
        trk_bgrs.append(frame_bgr)
        tracking_records.append(
            {
                "mot_frame_id": mot_idx,
                "filename": fname,
                "native_frame": nf,
                "timestamp_s": actual_ts,
                "width": w,
                "height": h,
                "sha256": sha256_file(out_img),
            }
        )
        if mot_idx in {1, 13, 25, 37, 49, 61}:
            trk_grid_samples.append((f"mot={mot_idx:02d}/61 native={nf} t={actual_ts:.2f}s", frame_bgr))

    # Build ZIPs, MP4s, CVAT label spec, and smoke-test packages (Condition #1)
    write_zip_from_files(det_files, out_base / "task1_detection_141imgs.zip")
    write_zip_from_files(trk_files, out_base / "task2_tracking_vid09_5fps_61frames.zip")
    write_mp4_from_frames(trk_bgrs, out_base / "vid09_tracking_5fps_61frames.mp4", fps=5.0)

    # Condition #1 smoke detection (5 representative images across the 4 campaigns)
    smoke_det_picks = [
        next(p for p in det_files if p.name.startswith("train__camp_coal_tunnel__vid_14__")),
        next(p for p in det_files if p.name.startswith("train__camp_perm_overpass__vid_04__")),
        next(p for p in det_files if p.name.startswith("val__camp_sishen_saldanha__vid_15__")),
        next(p for p in det_files if p.name.startswith("test__camp_vfli_cosne__vid_09__")),
        next(p for p in det_files if p.name.startswith("test__camp_vfli_cosne__vid_16__f000050")),
    ]
    write_zip_from_files(smoke_det_picks, out_base / "smoke_test_detection_5imgs.zip")

    # Condition #1 smoke tracking (first 5 frames of 5 fps clip)
    smoke_trk_files = trk_files[:5]
    write_zip_from_files(smoke_trk_files, out_base / "smoke_test_tracking_5frames.zip")
    write_mp4_from_frames(trk_bgrs[:5], out_base / "smoke_test_tracking_5frames.mp4", fps=5.0)

    cvat_labels = [{"name": "open_top_freight_wagon", "color": "#ff6a00", "type": "rectangle", "attributes": []}]
    (out_base / "cvat_labels_open_top.json").write_text(json.dumps(cvat_labels, indent=2), encoding="utf-8")

    # Render contact sheets for immediate visual verification
    triage_dir = root / "data/processed/triage-frames"
    # Pick 12 evenly distributed from train (covering vid_14 segments, locos, and vid_04)
    train_indices = [0, 8, 16, 24, 32, 42, 50, 58, 65, 70, 73, 79]
    save_labeled_grid([sample_buckets["train"][i] for i in train_indices], triage_dir / "phase3_1_extract_train_sample.jpg", cols=4)
    save_labeled_grid([sample_buckets["val"][i] for i in [0, 5, 10, 15, 20, 22]], triage_dir / "phase3_1_extract_val_sample.jpg", cols=3)
    save_labeled_grid([sample_buckets["test"][i] for i in [0, 8, 16, 22, 26, 33]], triage_dir / "phase3_1_extract_test_sample.jpg", cols=3)
    save_labeled_grid(trk_grid_samples, triage_dir / "phase3_1_extract_tracking_sample.jpg", cols=3)

    # Save tracked manifest in specs/data-manifests/open-top-split-v1.json
    manifest = {
        "dataset": "RailVision Phase 3.1 Specialized Open-Top Freight Wagon Split (Pre-Annotation)",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "class_map": {"0": "open_top_freight_wagon"},
        "campaign_split": {
            "train": ["camp_coal_tunnel", "camp_perm_overpass"],
            "val": ["camp_sishen_saldanha"],
            "test": ["camp_vfli_cosne"],
        },
        "counts": {
            "detection_total": len(detection_records),
            "detection_by_split": {
                sp: sum(1 for r in detection_records if r["split"] == sp) for sp in ("train", "val", "test")
            },
            "tracking_frames": len(tracking_records),
        },
        "tracking_policy": {
            "video_id": "vid_09_vossloh_g_1206_bb_of_vfli_video",
            "campaign": "camp_vfli_cosne",
            "split": "test_tracking_only",
            "excluded_from_training": True,
            "native_fps": 25.0,
            "sampled_fps": 5.0,
            "frame_stride": 5,
            "start_native_frame": 300,
            "end_native_frame": 600,
            "num_frames": len(tracking_records),
            "mot_metric_note": (
                "MOT metrics are computed strictly on the 61 frames sampled at 5 fps "
                "(t=12.0s..24.0s), not all native 25 fps frames."
            ),
        },
        "smoke_test": {
            "detection_filenames": [p.name for p in smoke_det_picks],
            "tracking_num_frames": 5,
        },
        "detection_frames": detection_records,
        "tracking_frames": tracking_records,
    }
    manifest_path = root / "specs/data-manifests/open-top-split-v1.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "manifest": str(manifest_path.relative_to(root)),
        "counts": manifest["counts"],
        "cvat_tasks_dir": str(out_base.relative_to(root)),
        "smoke_detection_files": manifest["smoke_test"]["detection_filenames"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
