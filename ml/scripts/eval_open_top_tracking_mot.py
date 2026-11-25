"""Generalizable MOT 1.1 evaluation for ByteTrack vs. BoT-SORT with two-stage confidence and FPS-scaled buffer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any, Iterator

import cv2
import lap
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_utils import save_labeled_grid  # noqa: E402

SAMPLE_MOT_FRAMES = (1, 13, 25, 37, 49, 61)
HISTORICAL_PHASE31_REPORT = Path("artifacts/runs/tracking/phase3_1_open_top_mot_eval.json")


def get_execution_environment() -> dict[str, Any]:
    """Capture Python/PyTorch/Ultralytics/CUDA environment details."""
    cuda_ok = bool(torch.cuda.is_available())
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": cuda_ok,
        "device": "cuda:0" if cuda_ok else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if cuda_ok else None,
    }


def compute_xywh_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0.0 else 0.0


def load_mot_gt(mot_path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """Load MOT 1.1 ground truth file grouped by 1-based frame_id."""
    by_frame: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for raw in mot_path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        parts = [p.strip() for p in s.split(",")]
        fid = int(parts[0])
        tid = int(parts[1])
        x, y, w, h = (float(v) for v in parts[2:6])
        by_frame[fid].append((tid, x, y, w, h))
    return dict(by_frame)


def build_tracker_config_text(
    *,
    tracker_type: str,
    high_thresh: float,
    low_thresh: float,
    effective_fps: float,
    buffer_seconds: float = 1.5,
    track_buffer: int | None = None,
    match_thresh: float = 0.8,
) -> str:
    """Generate Ultralytics ByteTrack/BoT-SORT YAML with FPS-scaled buffer and explicit two-stage thresholds."""
    resolved_buffer = (
        int(track_buffer)
        if track_buffer is not None
        else max(3, int(round(buffer_seconds * max(1.0, effective_fps))))
    )
    resolved_low = min(float(low_thresh), float(high_thresh))
    lines = [
        f"tracker_type: {tracker_type}",
        f"track_high_thresh: {float(high_thresh):.4f}",
        f"track_low_thresh: {resolved_low:.4f}",
        f"new_track_thresh: {float(high_thresh):.4f}",
        f"track_buffer: {resolved_buffer}",
        f"match_thresh: {float(match_thresh):.4f}",
        "fuse_score: True",
    ]
    if tracker_type == "botsort":
        lines.extend([
            "gmc_method: sparseOptFlow",
            "proximity_thresh: 0.5",
            "appearance_thresh: 0.8",
            "with_reid: False",
            "model: auto",
        ])
    return "\n".join(lines) + "\n"


@contextmanager
def temporary_tracker_yaml(
    *,
    tracker_type: str,
    high_thresh: float,
    low_thresh: float,
    effective_fps: float,
    track_buffer: int | None = None,
) -> Iterator[Path]:
    """Create a temporary tracker YAML file preserving Ultralytics installed configs untouched."""
    cfg_text = build_tracker_config_text(
        tracker_type=tracker_type,
        high_thresh=high_thresh,
        low_thresh=low_thresh,
        effective_fps=effective_fps,
        track_buffer=track_buffer,
    )
    with tempfile.TemporaryDirectory(prefix="railvision_tracker_cfg_") as tmpdir:
        cfg_path = Path(tmpdir) / f"{tracker_type}.yaml"
        cfg_path.write_text(cfg_text, encoding="utf-8")
        yield cfg_path


def evaluate_mot_sequence(
    gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]],
    pred_by_frame: dict[int, list[tuple[int, float, float, float, float]]],
    frames_order: list[int],
    iou_thresh: float = 0.50,
) -> dict[str, Any]:
    """Compute exact CLEAR MOT (MOTA, IDSW, FM) and global IDF1/IDP/IDR at IoU >= iou_thresh."""
    last_matched_pred: dict[int, int] = {}
    ever_tracked: set[int] = set()
    prev_frame_tracked: dict[int, bool] = {}

    tp = 0
    fp = 0
    fn = 0
    idsw = 0
    fm = 0
    matched_ious: list[float] = []
    pair_counts: Counter[tuple[int, int]] = Counter()
    gt_id_set: set[int] = set()
    pred_id_set: set[int] = set()
    total_gt = 0
    total_pred = 0

    for fid in frames_order:
        gts = gt_by_frame.get(fid, [])
        preds = pred_by_frame.get(fid, [])
        total_gt += len(gts)
        total_pred += len(preds)
        for gid, *_ in gts:
            gt_id_set.add(gid)
        for pid, *_ in preds:
            pred_id_set.add(pid)

        pred_idx_by_id = {pid: j for j, (pid, *_) in enumerate(preds)}
        matched_gt_indices: set[int] = set()
        matched_pred_indices: set[int] = set()
        frame_matches: list[tuple[int, int, float]] = []

        for gi, (gid, gx, gy, gw, gh) in enumerate(gts):
            prev_pid = last_matched_pred.get(gid)
            if prev_pid is not None and prev_pid in pred_idx_by_id:
                pj = pred_idx_by_id[prev_pid]
                if pj not in matched_pred_indices:
                    _, px, py, pw, ph = preds[pj]
                    iou = compute_xywh_iou((gx, gy, gw, gh), (px, py, pw, ph))
                    if iou >= iou_thresh:
                        matched_gt_indices.add(gi)
                        matched_pred_indices.add(pj)
                        frame_matches.append((gid, prev_pid, iou))

        rem_g = [i for i in range(len(gts)) if i not in matched_gt_indices]
        rem_p = [j for j in range(len(preds)) if j not in matched_pred_indices]
        if rem_g and rem_p:
            cost = np.full((len(rem_g), len(rem_p)), 1e6, dtype=np.float64)
            ious_cache = np.zeros((len(rem_g), len(rem_p)), dtype=np.float64)
            for ri, gi in enumerate(rem_g):
                _, gx, gy, gw, gh = gts[gi]
                for cj, pj in enumerate(rem_p):
                    _, px, py, pw, ph = preds[pj]
                    iou = compute_xywh_iou((gx, gy, gw, gh), (px, py, pw, ph))
                    if iou >= iou_thresh:
                        cost[ri, cj] = 1.0 - iou
                        ious_cache[ri, cj] = iou
            _, x_assign, _ = lap.lapjv(cost, extend_cost=True, cost_limit=(1.0 - iou_thresh + 1e-9))
            for ri, cj in enumerate(x_assign):
                if cj >= 0 and ious_cache[ri, cj] >= iou_thresh:
                    gi = rem_g[ri]
                    pj = rem_p[int(cj)]
                    matched_gt_indices.add(gi)
                    matched_pred_indices.add(pj)
                    frame_matches.append((gts[gi][0], preds[pj][0], float(ious_cache[ri, cj])))

        for gid, pid, iou in frame_matches:
            if gid in last_matched_pred and last_matched_pred[gid] != pid:
                idsw += 1
            last_matched_pred[gid] = pid
            if gid in ever_tracked and not prev_frame_tracked.get(gid, False):
                fm += 1
            ever_tracked.add(gid)
            prev_frame_tracked[gid] = True
            tp += 1
            pair_counts[(gid, pid)] += 1
            matched_ious.append(iou)

        for gi, (gid, *_) in enumerate(gts):
            if gi not in matched_gt_indices:
                fn += 1
                prev_frame_tracked[gid] = False

        fp += len(preds) - len(matched_pred_indices)

    g_ids = sorted(gt_id_set)
    p_ids = sorted(pred_id_set)
    idtp = 0
    if g_ids and p_ids and pair_counts:
        n_g, n_p = len(g_ids), len(p_ids)
        sq = np.zeros((n_g + n_p, n_g + n_p), dtype=np.float64)
        for i, gid in enumerate(g_ids):
            for j, pid in enumerate(p_ids):
                cnt = pair_counts.get((gid, pid), 0)
                if cnt > 0:
                    sq[i, j] = -float(cnt)
        _, x_glob, _ = lap.lapjv(sq)
        for i in range(n_g):
            j = int(x_glob[i])
            if j < n_p:
                idtp += pair_counts.get((g_ids[i], p_ids[j]), 0)

    idfp = total_pred - idtp
    idfn = total_gt - idtp
    idf1 = (2.0 * idtp / (total_gt + total_pred)) if (total_gt + total_pred) > 0 else 0.0
    idp = (idtp / total_pred) if total_pred > 0 else 0.0
    idr = (idtp / total_gt) if total_gt > 0 else 0.0

    det_precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    det_recall = (tp / total_gt) if total_gt > 0 else 0.0
    det_f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    mota = (1.0 - (fn + fp + idsw) / total_gt) if total_gt > 0 else 0.0

    return {
        "iou_thresh": iou_thresh,
        "frames_evaluated": len(frames_order),
        "total_gt": total_gt,
        "total_pred": total_pred,
        "unique_gt_ids": len(g_ids),
        "unique_pred_ids": len(p_ids),
        "count_abs_error": abs(len(p_ids) - len(g_ids)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "idsw": idsw,
        "fm": fm,
        "mota": round(mota, 4),
        "idf1": round(idf1, 4),
        "idp": round(idp, 4),
        "idr": round(idr, 4),
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "det_precision": round(det_precision, 4),
        "det_recall": round(det_recall, 4),
        "det_f1": round(det_f1, 4),
        "mean_matched_iou": round(float(sum(matched_ious) / len(matched_ious)), 4) if matched_ious else 0.0,
    }


def _extract_frame_preds(
    res: Any,
) -> tuple[list[tuple[int, float, float, float, float]], list[tuple[int, float, float, float, float, float]]]:
    """Return (eval_boxes [(pid, x, y, w, h)], draw_boxes [(pid, conf, x, y, w, h)])."""
    eval_boxes: list[tuple[int, float, float, float, float]] = []
    draw_boxes: list[tuple[int, float, float, float, float, float]] = []
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return eval_boxes, draw_boxes
    xyxys = boxes.xyxy.tolist()
    confs = boxes.conf.tolist()
    if boxes.id is not None:
        pids = boxes.id.int().tolist()
    else:
        pids = [900000 + i for i in range(len(xyxys))]
    for idx, (xyxy, cf, pid) in enumerate(zip(xyxys, confs, pids)):
        x1, y1, x2, y2 = xyxy
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        use_pid = int(pid) if pid >= 1 else (900000 + idx)
        eval_boxes.append((use_pid, x1, y1, w, h))
        draw_boxes.append((use_pid, float(cf), x1, y1, w, h))
    return eval_boxes, draw_boxes


def run_benchmark(
    name: str,
    weights_path: Path,
    tracker_type: str,
    mode: str,
    video_path: Path,
    frames_dir: Path,
    gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]],
    *,
    conf: float = 0.25,
    low_conf: float = 0.10,
    imgsz: int = 640,
    start_frame: int = 300,
    end_frame: int = 600,
    frame_stride: int = 5,
    track_buffer: int | None = None,
    overlay_out: Path | None = None,
) -> dict[str, Any]:
    """Run tracker over native video segment or direct sampled frames and compute CLEAR MOT + IDF1."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; aborting tracking benchmark.")

    model = YOLO(str(weights_path))
    pred_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
    sample_tiles: list[tuple[str, Any]] = []
    frames_processed = 0

    max_mot_frame = max(gt_by_frame.keys())
    frames_order = list(range(1, max_mot_frame + 1))

    cap_probe = cv2.VideoCapture(str(video_path))
    native_fps = float(cap_probe.get(cv2.CAP_PROP_FPS) or 25.0) if cap_probe.isOpened() else 25.0
    if cap_probe.isOpened():
        cap_probe.release()

    effective_fps = native_fps if mode == "native25_eval5" else max(1.0, native_fps / max(1, frame_stride))
    detector_conf = min(float(low_conf), float(conf))

    torch.cuda.init()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    with temporary_tracker_yaml(
        tracker_type=tracker_type,
        high_thresh=conf,
        low_thresh=low_conf,
        effective_fps=effective_fps,
        track_buffer=track_buffer,
    ) as cfg_path:
        if mode == "native25_eval5":
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed opening video: {video_path}")
            nf = 0
            while nf < start_frame:
                if not cap.grab():
                    cap.release()
                    raise RuntimeError(f"Video ended early before frame {start_frame}: {video_path}")
                nf += 1
            while nf <= end_frame:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    cap.release()
                    raise RuntimeError(f"Failed reading native frame {nf}")
                res = model.track(
                    source=bgr,
                    persist=True,
                    tracker=str(cfg_path),
                    conf=detector_conf,
                    imgsz=imgsz,
                    device=0,
                    verbose=False,
                )[0]
                frames_processed += 1
                if (nf - start_frame) % frame_stride == 0:
                    mot_fid = (nf - start_frame) // frame_stride + 1
                    eb, db = _extract_frame_preds(res)
                    pred_by_frame[mot_fid] = eb
                    if overlay_out is not None and mot_fid in SAMPLE_MOT_FRAMES:
                        vis = bgr.copy()
                        for gid, gx, gy, gw, gh in gt_by_frame.get(mot_fid, []):
                            cv2.rectangle(vis, (int(gx), int(gy)), (int(gx + gw), int(gy + gh)), (60, 220, 100), 3)
                            cv2.putText(vis, f"GT:{gid}", (int(gx), max(22, int(gy) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 220, 100), 2)
                        for pid, cf, px, py, pw, ph in db:
                            cv2.rectangle(vis, (int(px), int(py)), (int(px + pw), int(py + ph)), (255, 220, 0), 2)
                            cv2.putText(vis, f"P:{pid} {cf:.2f}", (int(px), min(1070, int(py + ph) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)
                        sample_tiles.append((f"{name[:22]} | mot={mot_fid:02d}/{max_mot_frame} gt={len(gt_by_frame.get(mot_fid, []))} pr={len(eb)}", vis))
                nf += 1
            cap.release()
        elif mode == "direct5":
            for mot_fid in frames_order:
                img_p = frames_dir / f"{mot_fid:06d}.jpg"
                bgr = cv2.imread(str(img_p))
                if bgr is None:
                    raise RuntimeError(f"Failed reading frame image: {img_p}")
                res = model.track(
                    source=bgr,
                    persist=True,
                    tracker=str(cfg_path),
                    conf=detector_conf,
                    imgsz=imgsz,
                    device=0,
                    verbose=False,
                )[0]
                frames_processed += 1
                eb, db = _extract_frame_preds(res)
                pred_by_frame[mot_fid] = eb
                if overlay_out is not None and mot_fid in SAMPLE_MOT_FRAMES:
                    vis = bgr.copy()
                    for gid, gx, gy, gw, gh in gt_by_frame.get(mot_fid, []):
                        cv2.rectangle(vis, (int(gx), int(gy)), (int(gx + gw), int(gy + gh)), (60, 220, 100), 3)
                        cv2.putText(vis, f"GT:{gid}", (int(gx), max(22, int(gy) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 220, 100), 2)
                    for pid, cf, px, py, pw, ph in db:
                        cv2.rectangle(vis, (int(px), int(py)), (int(px + pw), int(py + ph)), (255, 220, 0), 2)
                        cv2.putText(vis, f"P:{pid} {cf:.2f}", (int(px), min(1070, int(py + ph) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)
                    sample_tiles.append((f"{name[:22]} | mot={mot_fid:02d}/{max_mot_frame} gt={len(gt_by_frame.get(mot_fid, []))} pr={len(eb)}", vis))
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    elapsed_s = time.perf_counter() - t0
    peak_vram = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 1)
    inner = getattr(model.predictor.model, "model", model.predictor.model)
    runtime_device = str(next(inner.parameters()).device)

    if overlay_out is not None and sample_tiles:
        save_labeled_grid(sample_tiles, overlay_out, cols=3)

    metrics_50 = evaluate_mot_sequence(gt_by_frame, pred_by_frame, frames_order, iou_thresh=0.50)
    metrics_30 = evaluate_mot_sequence(gt_by_frame, pred_by_frame, frames_order, iou_thresh=0.30)

    return {
        "run_name": name,
        "weights": str(weights_path),
        "tracker": tracker_type,
        "input_mode": mode,
        "effective_fps_for_buffer": round(effective_fps, 2),
        "device": runtime_device,
        "gpu_name": torch.cuda.get_device_name(0),
        "high_conf_thresh": conf,
        "detector_low_conf_thresh": detector_conf,
        "frames_tracked": frames_processed,
        "frames_scored": len(frames_order),
        "wall_time_s": round(elapsed_s, 2),
        "effective_fps": round(frames_processed / elapsed_s, 2) if elapsed_s > 0 else 0.0,
        "peak_vram_mib": peak_vram,
        "mot_metrics_iou50": metrics_50,
        "diagnostic_iou30": metrics_30,
        "sample_grid": str(overlay_out).replace("\\", "/") if overlay_out else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalizable MOT evaluation with two-stage tracking thresholds.")
    parser.add_argument("--phase31-weights", "--candidate-weights", dest="candidate_weights", required=True, help="Candidate detector best.pt")
    parser.add_argument("--candidate-name", default="candidate", help="Short label for the evaluated model")
    parser.add_argument(
        "--phase3-weights",
        default="artifacts/runs/detect/exp02_yolo26n_twist_neardedup_ablation/weights/best.pt",
        help="Preserved Phase 3 baseline best.pt",
    )
    parser.add_argument("--skip-phase3-baseline", action="store_true")
    parser.add_argument("--video", default="data/raw/public-videos/vid_09_vossloh_g_1206_bb_of_vfli_video.webm")
    parser.add_argument("--frames-dir", default="data/processed/open-top-v1/cvat_tasks/task2_tracking_vid09_5fps_61frames")
    parser.add_argument("--gt-file", default="data/processed/open-top-yolo-v1/tracking_gt/vid09_5fps_gt.txt")
    parser.add_argument("--start-frame", type=int, default=300)
    parser.add_argument("--end-frame", type=int, default=600)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--low-conf", type=float, default=0.10)
    parser.add_argument("--track-buffer", type=int, default=None)
    parser.add_argument("--out-json", default=None, help="Output report JSON path (protects historical Phase 3.1 report unless --overwrite)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    video_path = (root / args.video) if not Path(args.video).is_absolute() else Path(args.video)
    frames_dir = (root / args.frames_dir) if not Path(args.frames_dir).is_absolute() else Path(args.frames_dir)
    gt_path = (root / args.gt_file) if not Path(args.gt_file).is_absolute() else Path(args.gt_file)
    triage_dir = root / "data/processed/triage-frames"

    if args.out_json:
        out_json = (root / args.out_json) if not Path(args.out_json).is_absolute() else Path(args.out_json)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_json = root / f"artifacts/runs/tracking/mot_eval_{args.candidate_name}_{stamp}.json"

    if out_json.resolve() == (root / HISTORICAL_PHASE31_REPORT).resolve() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite historical Phase 3.1 MOT report {out_json}. Pass a new --out-json or explicit --overwrite."
        )

    gt_by_frame = load_mot_gt(gt_path)
    p3_weights = Path(args.phase3_weights)
    cand_weights = Path(args.candidate_weights)

    runs: list[dict[str, Any]] = []
    if not args.skip_phase3_baseline and p3_weights.is_file():
        runs.extend([
            run_benchmark(
                "phase3_exp02_bytetrack_native_eval_sampled",
                p3_weights, "bytetrack", "native25_eval5",
                video_path, frames_dir, gt_by_frame,
                conf=args.conf, low_conf=args.low_conf, imgsz=args.imgsz,
                start_frame=args.start_frame, end_frame=args.end_frame, frame_stride=args.frame_stride,
                track_buffer=args.track_buffer,
            ),
            run_benchmark(
                "phase3_exp02_botsort_native_eval_sampled",
                p3_weights, "botsort", "native25_eval5",
                video_path, frames_dir, gt_by_frame,
                conf=args.conf, low_conf=args.low_conf, imgsz=args.imgsz,
                start_frame=args.start_frame, end_frame=args.end_frame, frame_stride=args.frame_stride,
                track_buffer=args.track_buffer,
            ),
        ])

    runs.extend([
        run_benchmark(
            f"{args.candidate_name}_bytetrack_native",
            cand_weights, "bytetrack", "native25_eval5",
            video_path, frames_dir, gt_by_frame,
            conf=args.conf, low_conf=args.low_conf, imgsz=args.imgsz,
            start_frame=args.start_frame, end_frame=args.end_frame, frame_stride=args.frame_stride,
            track_buffer=args.track_buffer,
            overlay_out=triage_dir / f"mot_{args.candidate_name}_bytetrack_native.jpg",
        ),
        run_benchmark(
            f"{args.candidate_name}_botsort_native",
            cand_weights, "botsort", "native25_eval5",
            video_path, frames_dir, gt_by_frame,
            conf=args.conf, low_conf=args.low_conf, imgsz=args.imgsz,
            start_frame=args.start_frame, end_frame=args.end_frame, frame_stride=args.frame_stride,
            track_buffer=args.track_buffer,
            overlay_out=triage_dir / f"mot_{args.candidate_name}_botsort_native.jpg",
        ),
        run_benchmark(
            f"{args.candidate_name}_bytetrack_direct_sampled",
            cand_weights, "bytetrack", "direct5",
            video_path, frames_dir, gt_by_frame,
            conf=args.conf, low_conf=args.low_conf, imgsz=args.imgsz,
            start_frame=args.start_frame, end_frame=args.end_frame, frame_stride=args.frame_stride,
            track_buffer=args.track_buffer,
            overlay_out=triage_dir / f"mot_{args.candidate_name}_bytetrack_direct.jpg",
        ),
        run_benchmark(
            f"{args.candidate_name}_botsort_direct_sampled",
            cand_weights, "botsort", "direct5",
            video_path, frames_dir, gt_by_frame,
            conf=args.conf, low_conf=args.low_conf, imgsz=args.imgsz,
            start_frame=args.start_frame, end_frame=args.end_frame, frame_stride=args.frame_stride,
            track_buffer=args.track_buffer,
            overlay_out=triage_dir / f"mot_{args.candidate_name}_botsort_direct.jpg",
        ),
    ])

    report = {
        "execution_environment": get_execution_environment(),
        "evaluation_scope_note": (
            f"MOT metrics computed on {len(gt_by_frame)} GT frames from {video_path.name} "
            f"(native {args.start_frame}..{args.end_frame} stride {args.frame_stride})."
        ),
        "gt_summary": {
            "gt_file": gt_path.as_posix(),
            "frames": len(gt_by_frame),
            "total_boxes": sum(len(v) for v in gt_by_frame.values()),
            "unique_ids": len({gid for v in gt_by_frame.values() for gid, *_ in v}),
        },
        "benchmarks": {r["run_name"]: r for r in runs},
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
