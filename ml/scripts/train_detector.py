"""Reproducible YOLO detector training, val confidence calibration, and opt-in held-out test evaluation."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
import ultralytics
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

RESERVED_TEST_ONLY_TOKENS = (
    "vid_01",
    "vid_09",
    "vid_16",
    "camp_vfli_cosne",
    "langer_unbeladener_g_terzug",
)


def extract_box_metrics(metrics: Any, names: dict[int, str]) -> dict[str, Any]:
    """Extract overall and per-class precision, recall, and mAP metrics from Ultralytics Val."""
    box = metrics.box
    per_class: dict[str, dict[str, float]] = {}
    for idx, cls_id in enumerate(getattr(box, "ap_class_index", [])):
        cls_name = names.get(int(cls_id), str(cls_id))
        per_class[cls_name] = {
            "precision": round(float(box.p[idx]), 4),
            "recall": round(float(box.r[idx]), 4),
            "mAP50": round(float(box.ap50[idx]), 4),
            "mAP50_95": round(float(box.ap[idx]), 4),
        }

    return {
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "per_class": per_class,
        "speed_ms_per_image": {k: round(float(v), 3) for k, v in (metrics.speed or {}).items()},
    }


def compute_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Compute intersection-over-union for two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def load_yolo_gt_boxes(label_path: Path, iw: int, ih: int) -> list[tuple[int, float, float, float, float]]:
    """Parse normalized YOLO label txt lines into pixel (cls_id, x1, y1, x2, y2) boxes."""
    gt_boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.is_file():
        return gt_boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cid, cx, cy, w, h = line.split()
        cx_f, cy_f, w_f, h_f = float(cx) * iw, float(cy) * ih, float(w) * iw, float(h) * ih
        gt_boxes.append((int(cid), cx_f - w_f / 2, cy_f - h_f / 2, cx_f + w_f / 2, cy_f + h_f / 2))
    return gt_boxes


def match_frame_predictions(
    gt_boxes: list[tuple[int, float, float, float, float]],
    pred_boxes: list[tuple[int, float, float, float, float, float]],
    iou_thresh: float = 0.50,
) -> tuple[int, int, int]:
    """Greedy confidence-ordered TP/FP/FN matching on a single frame."""
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    for pcid, _, x1, y1, x2, y2 in sorted(pred_boxes, key=lambda x: -x[1]):
        best_iou = 0.0
        best_idx = -1
        for gi, (gcid, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
            if gi in matched_gt or gcid != pcid:
                continue
            iou = compute_iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
            if iou > best_iou:
                best_iou = iou
                best_idx = gi
        if best_iou >= iou_thresh and best_idx >= 0:
            matched_gt.add(best_idx)
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def verify_dataset_split_hygiene(data_root: Path) -> None:
    """Prevent accidental inclusion of frozen test/negative campaigns (vid_01/vid_09/vid_16) in train/val."""
    violations: list[str] = []
    for split in ("train", "val"):
        img_dir = data_root / split / "images"
        if not img_dir.is_dir():
            continue
        for img_path in img_dir.iterdir():
            lower_name = img_path.name.lower()
            for token in RESERVED_TEST_ONLY_TOKENS:
                if token in lower_name:
                    violations.append(f"{split}/images/{img_path.name} matches reserved token '{token}'")
    if violations:
        raise ValueError(
            "Dataset hygiene violation: reserved held-out campaigns found in train/val: "
            + "; ".join(violations[:10])
        )


def compute_calibration_curve_from_predictions(
    frames_gt: list[list[tuple[int, float, float, float, float]]],
    frames_preds: list[list[tuple[int, float, float, float, float, float]]],
    thresholds: list[float] | None = None,
    iou_thresh: float = 0.50,
) -> dict[str, Any]:
    """Sweep confidence thresholds on validation predictions and return optimal F1 operating point."""
    scan = thresholds or [round(0.05 * i, 2) for i in range(1, 18)]
    curve: list[dict[str, Any]] = []
    best_entry: dict[str, Any] | None = None

    for conf_t in scan:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for gts, raw_preds in zip(frames_gt, frames_preds):
            filtered = [p for p in raw_preds if p[1] >= conf_t]
            tp, fp, fn = match_frame_predictions(gts, filtered, iou_thresh=iou_thresh)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        recall = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        entry = {
            "conf": conf_t,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        curve.append(entry)
        if best_entry is None or (entry["f1"], entry["recall"], -abs(entry["conf"] - 0.25)) > (
            best_entry["f1"],
            best_entry["recall"],
            -abs(best_entry["conf"] - 0.25),
        ):
            best_entry = entry

    return {
        "best_f1_conf": best_entry["conf"] if best_entry else 0.25,
        "best_f1": best_entry["f1"] if best_entry else 0.0,
        "best_precision": best_entry["precision"] if best_entry else 0.0,
        "best_recall": best_entry["recall"] if best_entry else 0.0,
        "curve": curve,
    }


def calibrate_confidence_threshold(
    model: YOLO,
    val_images_dir: Path,
    val_labels_dir: Path,
    imgsz: int,
) -> dict[str, Any]:
    """Run low-threshold inference once on val images and calibrate F1-optimal confidence threshold."""
    frames_gt: list[list[tuple[int, float, float, float, float]]] = []
    frames_preds: list[list[tuple[int, float, float, float, float, float]]] = []

    for img_path in sorted(val_images_dir.glob("*")):
        if not img_path.is_file():
            continue
        lbl_path = val_labels_dir / f"{img_path.stem}.txt"
        with Image.open(img_path) as im:
            iw, ih = im.size
        frames_gt.append(load_yolo_gt_boxes(lbl_path, iw, ih))

        preds = model.predict(source=str(img_path), imgsz=imgsz, conf=0.05, device=0, verbose=False)
        pred_boxes: list[tuple[int, float, float, float, float, float]] = []
        if preds and preds[0].boxes is not None:
            for box in preds[0].boxes:
                xyxy = box.xyxy[0].tolist()
                pred_boxes.append((int(box.cls[0].item()), float(box.conf[0].item()), xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
        frames_preds.append(pred_boxes)

    return compute_calibration_curve_from_predictions(frames_gt, frames_preds)


def build_test_error_sheet(
    model: YOLO,
    test_images_dir: Path,
    test_labels_dir: Path,
    names: dict[int, str],
    out_path: Path,
    imgsz: int,
    conf: float = 0.25,
) -> dict[str, int]:
    """Run predictions on held-out test split, count FP/FN (IoU>=0.5), and save error contact sheet."""
    scored_frames: list[tuple[int, Path, list[tuple[int, float, float, float, float]], list[tuple[int, float, float, float, float, float]]]] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_path in sorted(test_images_dir.glob("*")):
        if not img_path.is_file():
            continue
        lbl_path = test_labels_dir / f"{img_path.stem}.txt"
        with Image.open(img_path) as im:
            iw, ih = im.size
        gt_boxes = load_yolo_gt_boxes(lbl_path, iw, ih)

        preds = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, device=0, verbose=False)
        pred_boxes: list[tuple[int, float, float, float, float, float]] = []
        if preds and preds[0].boxes is not None:
            for box in preds[0].boxes:
                xyxy = box.xyxy[0].tolist()
                pred_boxes.append((int(box.cls[0].item()), float(box.conf[0].item()), xyxy[0], xyxy[1], xyxy[2], xyxy[3]))

        tp, fp, fn = match_frame_predictions(gt_boxes, pred_boxes, iou_thresh=0.50)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        scored_frames.append((fp + fn, img_path, gt_boxes, pred_boxes))

    scored_frames.sort(key=lambda x: (-x[0], -len(x[2]), x[1].name))
    chosen = scored_frames[:8]

    cell_w, cell_h = 480, 360
    cols = 4
    rows = max(1, (len(chosen) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (18, 22, 28))
    font = ImageFont.load_default()

    for idx, (err_score, img_path, gt_boxes, pred_boxes) in enumerate(chosen):
        with Image.open(img_path) as im:
            iw, ih = im.size
            rgb = im.convert("RGB").resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        sx, sy = cell_w / iw, cell_h / ih
        draw = ImageDraw.Draw(rgb)
        for cid, x1, y1, x2, y2 in gt_boxes:
            draw.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline="#22c55e", width=2)
            draw.text((x1 * sx + 2, max(2, y1 * sy + 2)), f"GT:{names.get(cid, cid)}", fill="#22c55e", font=font)
        for cid, box_conf, x1, y1, x2, y2 in pred_boxes:
            draw.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline="#22d3ee", width=2)
            draw.text((x1 * sx + 2, max(14, y2 * sy - 14)), f"P:{names.get(cid, cid)} {box_conf:.2f}", fill="#22d3ee", font=font)
        draw.rectangle([0, 0, cell_w, 18], fill=(0, 0, 0))
        draw.text((6, 3), f"{img_path.name[:40]} | errs(FP+FN)={err_score}", fill="#ffffff", font=font)
        r, c = divmod(idx, cols)
        sheet.paste(rgb, (c * cell_w, r * cell_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)
    return {
        "conf_threshold": conf,
        "test_tp": total_tp,
        "test_fp": total_fp,
        "test_fn": total_fn,
    }


def _repo_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO detector with strict val/test separation.")
    parser.add_argument("--model", required=True, help="e.g. yolo26n.pt or path to .pt")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--name", required=True, help="Unique experiment run name")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--hsv-h", type=float, default=None)
    parser.add_argument("--hsv-s", type=float, default=None)
    parser.add_argument("--hsv-v", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--translate", type=float, default=None)
    parser.add_argument("--degrees", type=float, default=None)
    parser.add_argument("--perspective", type=float, default=None)
    parser.add_argument("--mosaic", type=float, default=None)
    parser.add_argument("--close-mosaic", type=int, default=None)
    parser.add_argument("--mixup", type=float, default=None)
    parser.add_argument("--erasing", type=float, default=None)
    parser.add_argument("--fliplr", type=float, default=None)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--eval-test", action="store_true", help="Opt-in held-out test evaluation (Gate G3 final candidate only)")
    parser.add_argument("--overwrite", action="store_true", help="Allow reusing an existing experiment directory")
    parser.add_argument("--project-dir", type=Path, default=None, help="Optional custom directory for experiment outputs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU indisponivel no PyTorch; abortando treino para nao mascarar ambiente.")

    root = Path(__file__).resolve().parents[2]
    data_yaml_path = Path(args.data).resolve()
    data_root = data_yaml_path.parent
    verify_dataset_split_hygiene(data_root)

    project_dir = args.project_dir.resolve() if args.project_dir else (root / "artifacts" / "runs" / "detect")
    project_dir.mkdir(parents=True, exist_ok=True)
    run_dir = project_dir / args.name
    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Experiment directory already exists: {run_dir}. Pass a new --name or explicit --overwrite."
        )

    torch.cuda.init()
    torch.cuda.set_device(0)

    pretrained_dir = root / "artifacts" / "models" / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    weights_arg = str(pretrained_dir / args.model) if not Path(args.model).is_file() else args.model

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    model = YOLO(weights_arg)
    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml_path),
        "project": str(project_dir),
        "name": args.name,
        "exist_ok": bool(args.overwrite),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "deterministic": True,
        "workers": 2,
        "device": 0,
        "verbose": True,
        "flipud": float(args.flipud),
    }
    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze
    if args.lr0 is not None:
        train_kwargs["optimizer"] = "AdamW"
        train_kwargs["lr0"] = args.lr0
        train_kwargs["cos_lr"] = bool(args.cos_lr)

    optional_aug_map = {
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "scale": args.scale,
        "translate": args.translate,
        "degrees": args.degrees,
        "perspective": args.perspective,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "mixup": args.mixup,
        "erasing": args.erasing,
        "fliplr": args.fliplr,
    }
    for k, v in optional_aug_map.items():
        if v is not None:
            train_kwargs[k] = v

    train_results = model.train(**train_kwargs)
    train_elapsed_s = round(time.perf_counter() - t0, 2)
    peak_vram_mib = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 1)

    save_dir = Path(train_results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    best_model = YOLO(str(best_pt))

    val_metrics = best_model.val(
        data=str(data_yaml_path),
        split="val",
        imgsz=args.imgsz,
        device=0,
        project=str(project_dir),
        name=f"{args.name}_val",
        exist_ok=True,
        verbose=False,
    )
    val_calibration = calibrate_confidence_threshold(
        best_model,
        data_root / "val" / "images",
        data_root / "val" / "labels",
        args.imgsz,
    )

    test_metrics_dict: dict[str, Any] | None = None
    err_counts: dict[str, int] | None = None
    err_sheet_rel: str | None = None
    if args.eval_test:
        test_metrics = best_model.val(
            data=str(data_yaml_path),
            split="test",
            imgsz=args.imgsz,
            device=0,
            project=str(project_dir),
            name=f"{args.name}_test",
            exist_ok=True,
            verbose=False,
        )
        test_metrics_dict = extract_box_metrics(test_metrics, best_model.names)
        err_sheet_path = root / "data" / "processed" / "triage-frames" / f"{args.name}_test_errors.jpg"
        err_counts = build_test_error_sheet(
            best_model,
            data_root / "test" / "images",
            data_root / "test" / "labels",
            best_model.names,
            err_sheet_path,
            args.imgsz,
            conf=float(val_calibration["best_f1_conf"]),
        )
        err_sheet_rel = _repo_rel(err_sheet_path, root)

    epochs_completed = None
    results_csv = save_dir / "results.csv"
    if results_csv.is_file():
        lines = [ln for ln in results_csv.read_text(encoding="utf-8").splitlines() if ln.strip()]
        epochs_completed = max(0, len(lines) - 1)

    summary = {
        "experiment_name": args.name,
        "base_model": args.model,
        "data_yaml": _repo_rel(data_yaml_path, root),
        "best_weights": _repo_rel(best_pt, root),
        "eval_test_executed": bool(args.eval_test),
        "hyperparameters": {
            "epochs_requested": args.epochs,
            "epochs_completed": epochs_completed,
            "patience": args.patience,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "seed": args.seed,
            "freeze": args.freeze,
            "lr0": args.lr0,
            "cos_lr": bool(args.cos_lr),
            "deterministic": True,
            **{k: v for k, v in optional_aug_map.items() if v is not None},
            "flipud": float(args.flipud),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": True,
            "device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(0),
            "peak_allocated_vram_mib": peak_vram_mib,
            "train_wall_time_s": train_elapsed_s,
        },
        "val_metrics": extract_box_metrics(val_metrics, best_model.names),
        "val_conf_calibration": val_calibration,
        "test_metrics": test_metrics_dict,
        "test_fp_fn_summary": err_counts,
        "error_contact_sheet": err_sheet_rel,
    }

    summary_path = save_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
