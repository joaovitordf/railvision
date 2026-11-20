"""Cross-domain detection & tracking PoC evaluation on vid_09 (ByteTrack vs BoT-SORT), vid_05 (point), and vid_04."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import torch
import ultralytics
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


def get_execution_environment() -> dict[str, Any]:
    """Capture runtime hardware/device and library metadata for experiment traceability."""
    cuda_ok = bool(torch.cuda.is_available())
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": cuda_ok,
        "device": "cuda:0" if cuda_ok else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if cuda_ok else None,
    }


def save_annotated_grid(frames_bgr: list[tuple[str, Any]], out_path: Path, cols: int = 3) -> None:
    if not frames_bgr:
        return
    cell_w, cell_h = 640, 360
    rows = (len(frames_bgr) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (18, 22, 28))
    font = ImageFont.load_default()
    for idx, (caption, bgr) in enumerate(frames_bgr):
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 0, cell_w, 20], fill=(0, 0, 0))
        draw.text((8, 4), caption[:90], fill="#ffffff", font=font)
        r, c = divmod(idx, cols)
        sheet.paste(im, (c * cell_w, r * cell_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=88)


def run_tracker_on_video(
    weights_path: str,
    video_path: Path,
    tracker_cfg: str,
    out_grid_path: Path,
    classes_filter: list[int] | None,
    conf: float = 0.25,
    imgsz: int = 640,
) -> dict[str, Any]:
    """Run frame-by-frame YOLO + tracker on a video and summarize track continuity, fragmentation, and speed."""
    model = YOLO(weights_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps_video = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_targets = {int(i * max(1, total_frames - 1) / 5) for i in range(6)}

    track_frames: dict[int, list[int]] = defaultdict(list)
    track_classes: dict[int, Counter[str]] = defaultdict(Counter)
    track_first_seen: dict[int, int] = {}
    frames_with_detections = 0
    total_detections = 0
    sampled_annotated: list[tuple[str, Any]] = []

    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        results = model.track(
            source=frame_bgr,
            persist=True,
            tracker=tracker_cfg,
            conf=conf,
            imgsz=imgsz,
            classes=classes_filter,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False,
        )
        res = results[0]
        boxes = res.boxes
        frame_track_ids = []
        if boxes is not None and len(boxes) > 0:
            frames_with_detections += 1
            total_detections += len(boxes)
            ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
            clss = boxes.cls.int().tolist()
            for tid, cid in zip(ids, clss):
                cls_name = model.names.get(int(cid), str(cid))
                if tid is not None:
                    track_frames[tid].append(frame_idx)
                    track_classes[tid][cls_name] += 1
                    track_first_seen.setdefault(tid, frame_idx)
                    frame_track_ids.append(tid)

        if frame_idx in sample_targets:
            annotated = res.plot()
            ts_s = frame_idx / fps_video if fps_video > 0 else 0.0
            cap_txt = f"{tracker_cfg} | f={frame_idx} t={ts_s:.1f}s | dets={len(boxes) if boxes is not None else 0} tids={frame_track_ids[:6]}"
            sampled_annotated.append((cap_txt, annotated))

        frame_idx += 1

    cap.release()
    elapsed_s = time.perf_counter() - t0
    peak_vram = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 1) if torch.cuda.is_available() else None

    save_annotated_grid(sampled_annotated, out_grid_path, cols=3)

    # Analyze track durations and potential fragmentation (gap > 3 frames inside same track_id)
    ordered_tids = sorted(track_first_seen.keys(), key=lambda tid: track_first_seen[tid])
    durations = [len(track_frames[tid]) for tid in ordered_tids]
    short_tracks_lt10 = sum(1 for d in durations if d < 10)
    sustained_tracks_ge25 = sum(1 for d in durations if d >= 25)
    tracks_with_gaps = 0
    for tid in ordered_tids:
        f_list = track_frames[tid]
        if any((b - a) > 3 for a, b in zip(f_list, f_list[1:])):
            tracks_with_gaps += 1

    sequence_table = [
        {
            "sequence_index": seq_idx,
            "track_id": tid,
            "first_frame": track_frames[tid][0],
            "last_frame": track_frames[tid][-1],
            "observed_frames": len(track_frames[tid]),
            "dominant_class": track_classes[tid].most_common(1)[0][0],
        }
        for seq_idx, tid in enumerate(ordered_tids[:40], start=1)
    ]

    exec_env = get_execution_environment()
    return {
        "weights": weights_path,
        "tracker": tracker_cfg,
        "device": exec_env["device"],
        "gpu_name": exec_env["gpu_name"],
        "frames_processed": frame_idx,
        "frames_with_detections": frames_with_detections,
        "total_detections": total_detections,
        "unique_track_ids": len(ordered_tids),
        "short_tracks_lt10_frames": short_tracks_lt10,
        "sustained_tracks_ge25_frames": sustained_tracks_ge25,
        "tracks_with_frame_gaps_gt3": tracks_with_gaps,
        "median_track_frames": sorted(durations)[len(durations) // 2] if durations else 0,
        "max_track_frames": max(durations) if durations else 0,
        "wall_time_s": round(elapsed_s, 2),
        "effective_fps": round(frame_idx / elapsed_s, 2) if elapsed_s > 0 else 0.0,
        "peak_vram_mib": peak_vram,
        "sample_grid": str(out_grid_path).replace("\\", "/"),
        "sequence_preview": sequence_table,
    }


def eval_point_frames(
    weights_path: str,
    video_path: Path,
    frame_indices: list[int],
    out_grid_path: Path,
    classes_filter: list[int] | None,
    conf: float = 0.25,
) -> dict[str, Any]:
    """Point-in-time frame detection (explicitly NO tracking) for timelapse or parked evaluation."""
    model = YOLO(weights_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps_video = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    target_set = set(frame_indices)
    max_idx = max(frame_indices)

    annotated_samples: list[tuple[str, Any]] = []
    per_frame_counts: list[dict[str, Any]] = []
    idx = 0
    while idx <= max_idx:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx in target_set:
            res = model.predict(
                source=frame_bgr,
                conf=conf,
                imgsz=640,
                classes=classes_filter,
                device=0 if torch.cuda.is_available() else "cpu",
                verbose=False,
            )[0]
            cls_counts: Counter[str] = Counter()
            if res.boxes is not None:
                for cid in res.boxes.cls.int().tolist():
                    cls_counts[model.names.get(int(cid), str(cid))] += 1
            ts_s = idx / fps_video if fps_video > 0 else 0.0
            per_frame_counts.append({"frame": idx, "timestamp_s": round(ts_s, 2), "detections": dict(cls_counts)})
            annotated_samples.append((f"f={idx} t={ts_s:.1f}s | {dict(cls_counts)}", res.plot()))
        idx += 1
    cap.release()

    save_annotated_grid(annotated_samples, out_grid_path, cols=3)
    exec_env = get_execution_environment()
    return {
        "weights": weights_path,
        "video": str(video_path.name),
        "device": exec_env["device"],
        "gpu_name": exec_env["gpu_name"],
        "per_frame": per_frame_counts,
        "sample_grid": str(out_grid_path).replace("\\", "/"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--twist-weights", required=True, help="Path to TWIST-trained best.pt")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    vid09 = root / "data/raw/public-videos/vid_09_vossloh_g_1206_bb_of_vfli_video.webm"
    vid05 = root / "data/raw/public-videos/vid_05_timelapse_freight_bypass_railway_eichenb.webm"
    vid04 = root / "data/raw/public-videos/vid_04_2es10_003_with_freight_train_from_above.webm"
    coco_weights = str(root / "artifacts/models/pretrained/yolo26n.pt")

    out_dir = root / "artifacts/runs/tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = root / "data/processed/triage-frames"

    # 1. vid_09 primary moving open-top PoC: TWIST-finetuned with ByteTrack & BoT-SORT
    #    PLUS COCO pretrained reference (class 6 = train) for honest cross-domain comparison
    vid09_results = {
        "twist_finetuned_bytetrack": run_tracker_on_video(
            args.twist_weights, vid09, "bytetrack.yaml",
            frames_dir / "vid09_twist_bytetrack_samples.jpg", classes_filter=None,
        ),
        "twist_finetuned_botsort": run_tracker_on_video(
            args.twist_weights, vid09, "botsort.yaml",
            frames_dir / "vid09_twist_botsort_samples.jpg", classes_filter=None,
        ),
        "pretrained_coco_ref_bytetrack": run_tracker_on_video(
            coco_weights, vid09, "bytetrack.yaml",
            frames_dir / "vid09_coco_ref_bytetrack_samples.jpg", classes_filter=[6],
        ),
    }

    # 2. vid_05 point-in-time detection ONLY (NO tracking, timelapse video)
    vid05_results = {
        "twist_finetuned": eval_point_frames(
            args.twist_weights, vid05, [10, 66, 140, 142, 260, 269],
            frames_dir / "vid05_twist_point_detection.jpg", classes_filter=None,
        ),
        "pretrained_coco_ref": eval_point_frames(
            coco_weights, vid05, [10, 66, 140, 142, 260, 269],
            frames_dir / "vid05_coco_ref_point_detection.jpg", classes_filter=[6],
        ),
    }

    # 3. vid_04 separate parked / overpass evaluation
    vid04_results = {
        "twist_finetuned": eval_point_frames(
            args.twist_weights, vid04, [0, 340, 645, 725, 806, 847],
            frames_dir / "vid04_twist_overpass_detection.jpg", classes_filter=None,
        ),
    }

    full_report = {
        "execution_environment": get_execution_environment(),
        "vid_09_moving_open_top_poc": vid09_results,
        "vid_05_point_in_time_only": vid05_results,
        "vid_04_overpass_and_parked": vid04_results,
    }
    report_path = out_dir / "phase3_video_tracking_eval.json"
    report_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(full_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
