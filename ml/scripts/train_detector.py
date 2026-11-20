"""Reproducible YOLO26 detector training, held-out evaluation, and test error visualization."""

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


def extract_box_metrics(metrics: Any, names: dict[int, str]) -> dict[str, Any]:
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


def build_test_error_sheet(
    model: YOLO,
    test_images_dir: Path,
    test_labels_dir: Path,
    names: dict[int, str],
    out_path: Path,
    imgsz: int,
) -> dict[str, int]:
    """Run predictions on held-out test split, count FP/FN (IoU>=0.5, conf>=0.25), and save error contact sheet."""
    scored_frames: list[tuple[int, Path, list[tuple[int, float, float, float, float]], list[tuple[int, float, float, float, float, float]]]] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_path in sorted(test_images_dir.glob("*")):
        lbl_path = test_labels_dir / f"{img_path.stem}.txt"
        with Image.open(img_path) as im:
            iw, ih = im.size
        gt_boxes: list[tuple[int, float, float, float, float]] = []
        if lbl_path.is_file():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                cid, cx, cy, w, h = line.split()
                cx_f, cy_f, w_f, h_f = float(cx) * iw, float(cy) * ih, float(w) * iw, float(h) * ih
                gt_boxes.append((int(cid), cx_f - w_f / 2, cy_f - h_f / 2, cx_f + w_f / 2, cy_f + h_f / 2))

        preds = model.predict(source=str(img_path), imgsz=imgsz, conf=0.25, device=0, verbose=False)
        pred_boxes: list[tuple[int, float, float, float, float, float]] = []
        if preds and preds[0].boxes is not None:
            for box in preds[0].boxes:
                xyxy = box.xyxy[0].tolist()
                pred_boxes.append((int(box.cls[0].item()), float(box.conf[0].item()), xyxy[0], xyxy[1], xyxy[2], xyxy[3]))

        matched_gt = set()
        fp = 0
        tp = 0
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
            if best_iou >= 0.5 and best_idx >= 0:
                matched_gt.add(best_idx)
                tp += 1
            else:
                fp += 1
        fn = len(gt_boxes) - len(matched_gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        err_score = fp + fn
        scored_frames.append((err_score, img_path, gt_boxes, pred_boxes))

    # Pick top error frames + a couple freight_car frames for qualitative analysis
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
        for cid, conf, x1, y1, x2, y2 in pred_boxes:
            draw.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline="#22d3ee", width=2)
            draw.text((x1 * sx + 2, max(14, y2 * sy - 14)), f"P:{names.get(cid, cid)} {conf:.2f}", fill="#22d3ee", font=font)
        draw.rectangle([0, 0, cell_w, 18], fill=(0, 0, 0))
        draw.text((6, 3), f"{img_path.name[:40]} | errs(FP+FN)={err_score}", fill="#ffffff", font=font)
        r, c = divmod(idx, cols)
        sheet.paste(rgb, (c * cell_w, r * cell_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)
    return {"test_tp_conf25_iou50": total_tp, "test_fp_conf25_iou50": total_fp, "test_fn_conf25_iou50": total_fn}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="e.g. yolo26n.pt or yolo26s.pt")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--name", required=True, help="Experiment run name")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--cos-lr", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU indisponivel no PyTorch; abortando treino para nao mascarar ambiente.")

    torch.cuda.init()
    torch.cuda.set_device(0)

    root = Path(__file__).resolve().parents[2]
    pretrained_dir = root / "artifacts" / "models" / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    weights_arg = str(pretrained_dir / args.model) if not Path(args.model).is_file() else args.model

    project_dir = root / "artifacts" / "runs" / "detect"
    project_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    model = YOLO(weights_arg)
    train_kwargs: dict[str, Any] = {
        "data": args.data,
        "project": str(project_dir),
        "name": args.name,
        "exist_ok": True,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "deterministic": True,
        "workers": 2,
        "device": 0,
        "verbose": True,
    }
    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze
    if args.lr0 is not None:
        train_kwargs["optimizer"] = "AdamW"
        train_kwargs["lr0"] = args.lr0
        train_kwargs["cos_lr"] = bool(args.cos_lr)
    train_results = model.train(**train_kwargs)
    train_elapsed_s = round(time.perf_counter() - t0, 2)
    peak_vram_mib = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 1)

    save_dir = Path(train_results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    best_model = YOLO(str(best_pt))

    val_metrics = best_model.val(
        data=args.data,
        split="val",
        imgsz=args.imgsz,
        device=0,
        project=str(project_dir),
        name=f"{args.name}_val",
        exist_ok=True,
        verbose=False,
    )
    test_metrics = best_model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        device=0,
        project=str(project_dir),
        name=f"{args.name}_test",
        exist_ok=True,
        verbose=False,
    )

    data_root = Path(args.data).parent
    err_sheet_path = root / "data" / "processed" / "triage-frames" / f"{args.name}_test_errors.jpg"
    err_counts = build_test_error_sheet(
        best_model,
        data_root / "test" / "images",
        data_root / "test" / "labels",
        best_model.names,
        err_sheet_path,
        args.imgsz,
    )

    epochs_completed = None
    results_csv = save_dir / "results.csv"
    if results_csv.is_file():
        lines = [ln for ln in results_csv.read_text(encoding="utf-8").splitlines() if ln.strip()]
        epochs_completed = max(0, len(lines) - 1)

    summary = {
        "experiment_name": args.name,
        "base_model": args.model,
        "data_yaml": str(Path(args.data).resolve()),
        "best_weights": str(best_pt),
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
        "test_metrics": extract_box_metrics(test_metrics, best_model.names),
        "test_fp_fn_summary": err_counts,
        "error_contact_sheet": str(err_sheet_path.relative_to(root)).replace("\\", "/"),
    }

    summary_path = save_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
