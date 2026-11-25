"""Benchmark open-vocabulary local pre-annotation teachers vs exp04c on val (vid_15) and train negatives (vid_04)."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preannotate_cvat import (  # noqa: E402
    DEFAULT_EXP04C,
    DEFAULT_PROMPTS,
    PRETRAINED_DIR,
    box_iou_matrix,
    load_yolo_boxes_xyxy,
    nms_merge_xyxy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESERVED_IDS = ("vid_01", "vid_09", "vid_16", "camp_vfli_cosne")


@contextmanager
def _in_pretrained() -> Iterator[None]:
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    prev = os.getcwd()
    try:
        os.chdir(str(PRETRAINED_DIR))
        yield
    finally:
        os.chdir(prev)


def assert_no_reserved_benchmark_paths(paths: list[Path]) -> None:
    """Enforce strict benchmark isolation: vid_01, vid_09, and vid_16 are forbidden here."""
    for p in paths:
        low = str(p).lower().replace("\\", "/")
        for reserved in RESERVED_IDS:
            if reserved in low:
                raise ValueError(
                    f"Isolamento de benchmark violado! Arquivo reservado '{reserved}' detectado em: {p}"
                )


def compute_detection_metrics(
    preds_by_stem: dict[str, list[tuple[float, float, float, float, float]]],
    gts_by_stem: dict[str, np.ndarray],
    iou_thresh: float = 0.50,
) -> dict[str, float | int]:
    """Compute TP, FP, FN, Precision, Recall, F1@IoU0.50, and All-Point AP@0.50."""
    total_gt = sum(int(len(arr)) for arr in gts_by_stem.values())
    all_preds: list[tuple[float, str, np.ndarray]] = []
    for stem, boxes in preds_by_stem.items():
        for x1, y1, x2, y2, conf in boxes:
            all_preds.append((float(conf), stem, np.asarray([x1, y1, x2, y2], dtype=np.float32)))
    all_preds.sort(key=lambda item: item[0], reverse=True)

    matched_flags: dict[str, np.ndarray] = {
        stem: np.zeros((len(arr),), dtype=bool) for stem, arr in gts_by_stem.items()
    }
    tp_vec = np.zeros((len(all_preds),), dtype=np.float32)
    fp_vec = np.zeros((len(all_preds),), dtype=np.float32)

    for idx, (_conf, stem, pred_box) in enumerate(all_preds):
        gt_boxes = gts_by_stem.get(stem)
        if gt_boxes is None or len(gt_boxes) == 0:
            fp_vec[idx] = 1.0
            continue
        ious = box_iou_matrix(pred_box[None, :], gt_boxes)[0]
        best_gt = int(np.argmax(ious))
        if ious[best_gt] >= iou_thresh and not matched_flags[stem][best_gt]:
            matched_flags[stem][best_gt] = True
            tp_vec[idx] = 1.0
        else:
            fp_vec[idx] = 1.0

    tp = int(tp_vec.sum())
    fp = int(fp_vec.sum())
    fn = max(0, total_gt - tp)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / total_gt) if total_gt > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if total_gt > 0 and len(all_preds) > 0:
        cum_tp = np.cumsum(tp_vec)
        cum_fp = np.cumsum(fp_vec)
        recalls = cum_tp / float(total_gt)
        precisions = cum_tp / np.maximum(1e-9, cum_tp + cum_fp)
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([1.0], precisions, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        ap50 = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    else:
        ap50 = 0.0

    return {
        "gt_boxes": total_gt,
        "predictions": len(all_preds),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_50": round(f1, 4),
        "ap50": round(ap50, 4),
    }


def _evaluate_stream(
    *,
    label: str,
    val_imgs: list[Path],
    val_gts: dict[str, np.ndarray],
    neg_imgs: list[Path],
    infer_fn: Any,
    conf: float,
    iou: float,
    device: str,
) -> dict[str, Any]:
    assert_no_reserved_benchmark_paths(val_imgs + neg_imgs)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warmup on first validation image
    first_bgr = cv2.imread(str(val_imgs[0]))
    _ = infer_fn(first_bgr, conf, iou)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    preds_val: dict[str, list[tuple[float, float, float, float, float]]] = {}
    latencies_ms: list[float] = []
    for p in val_imgs:
        bgr = cv2.imread(str(p))
        t0 = time.perf_counter()
        boxes = infer_fn(bgr, conf, iou)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        preds_val[p.stem] = boxes

    neg_fp = 0
    neg_preds: dict[str, int] = {}
    for p in neg_imgs:
        bgr = cv2.imread(str(p))
        t0 = time.perf_counter()
        boxes = infer_fn(bgr, conf, iou)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        neg_fp += len(boxes)
        neg_preds[p.stem] = len(boxes)

    peak_vram_mb = (
        round(float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0), 2)
        if (device.startswith("cuda") and torch.cuda.is_available())
        else 0.0
    )

    val_metrics = compute_detection_metrics(preds_val, val_gts, iou_thresh=0.50)
    return {
        "model_key": label,
        "val_vid15": val_metrics,
        "train_neg_vid04": {
            "frames": len(neg_imgs),
            "false_positives": neg_fp,
            "fp_per_frame": round(float(neg_fp / max(1, len(neg_imgs))), 4),
            "per_frame": neg_preds,
        },
        "mean_ms_per_frame": round(float(sum(latencies_ms) / max(1, len(latencies_ms))), 2),
        "peak_vram_mb": peak_vram_mb,
        "preds_val": preds_val,
    }


def _extract_boxes(result: Any) -> list[tuple[float, float, float, float, float]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    return [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)) for b, c in zip(xyxy, confs)]


def run_benchmark(
    *,
    dataset_root: Path,
    out_json: Path,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cuda:0",
    prompts: list[str] | None = None,
) -> dict[str, Any]:
    """Run controlled open-vocab + exp04c teacher benchmark strictly on val (vid_15) and train negatives (vid_04)."""
    from ultralytics import YOLO, YOLOE, YOLOWorld
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    resolved_prompts = prompts or DEFAULT_PROMPTS
    val_imgs = sorted((dataset_root / "val" / "images").glob("*__vid_15__*.jpg"))
    neg_imgs = sorted((dataset_root / "train" / "images").glob("*__vid_04__*.jpg"))
    exemplar_imgs = sorted((dataset_root / "train" / "images").glob("*__vid_14__*.jpg"))

    if not val_imgs:
        raise FileNotFoundError(f"Imagens val vid_15 não encontradas em {dataset_root / 'val' / 'images'}")
    if not neg_imgs:
        raise FileNotFoundError(f"Imagens train negativas vid_04 não encontradas em {dataset_root / 'train' / 'images'}")
    if not exemplar_imgs:
        raise FileNotFoundError(f"Exemplar train vid_14 não encontrado em {dataset_root / 'train' / 'images'}")

    assert_no_reserved_benchmark_paths(val_imgs + neg_imgs + exemplar_imgs)

    val_gts: dict[str, np.ndarray] = {}
    for p in val_imgs:
        bgr = cv2.imread(str(p))
        ih, iw = bgr.shape[:2]
        val_gts[p.stem] = load_yolo_boxes_xyxy(dataset_root / "val" / "labels" / f"{p.stem}.txt", iw, ih)

    # Pick first train vid_14 exemplar that has >= 2 GT boxes for YOLOE visual prompting
    ref_img: Path | None = None
    ref_boxes: np.ndarray | None = None
    for cand in exemplar_imgs:
        lbl = dataset_root / "train" / "labels" / f"{cand.stem}.txt"
        bgr = cv2.imread(str(cand))
        ih, iw = bgr.shape[:2]
        gt = load_yolo_boxes_xyxy(lbl, iw, ih)
        if len(gt) >= 2:
            ref_img = cand
            ref_boxes = gt[:4]
            break
    if ref_img is None or ref_boxes is None:
        raise RuntimeError("Nenhum frame exemplar em train vid_14 possui >= 2 caixas para prompt visual.")

    runs: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    cached_val_preds: dict[str, dict[str, list[tuple[float, float, float, float, float]]]] = {}
    cached_neg_preds: dict[str, dict[str, list[tuple[float, float, float, float, float]]]] = {}

    # 1. Baseline exp04c
    exp04c_model = YOLO(str(DEFAULT_EXP04C))
    res_exp04c = _evaluate_stream(
        label="exp04c_yolo26n",
        val_imgs=val_imgs,
        val_gts=val_gts,
        neg_imgs=neg_imgs,
        infer_fn=lambda bgr, c, iv: _extract_boxes(exp04c_model.predict(bgr, conf=c, iou=iv, device=device, verbose=False)[0]),
        conf=conf,
        iou=iou,
        device=device,
    )
    cached_val_preds["exp04c"] = res_exp04c.pop("preds_val")
    # Collect neg raw boxes for hybrid combination
    cached_neg_preds["exp04c"] = {
        p.stem: _extract_boxes(exp04c_model.predict(cv2.imread(str(p)), conf=conf, iou=iou, device=device, verbose=False)[0])
        for p in neg_imgs
    }
    runs.append(res_exp04c)
    del exp04c_model

    # 2. YOLOE text + visual prompts
    yoloe_name = "yoloe-26s-seg.pt"
    try:
        with _in_pretrained():
            try:
                yoloe_text = YOLOE(yoloe_name)
            except Exception:
                yoloe_name = "yoloe-11s-seg.pt"
                yoloe_text = YOLOE(yoloe_name)
            yoloe_text.set_classes(resolved_prompts, yoloe_text.get_text_pe(resolved_prompts))
        res_ye_text = _evaluate_stream(
            label=f"yoloe_text ({yoloe_name})",
            val_imgs=val_imgs,
            val_gts=val_gts,
            neg_imgs=neg_imgs,
            infer_fn=lambda bgr, c, iv: _extract_boxes(yoloe_text.predict(bgr, conf=c, iou=iv, device=device, verbose=False)[0]),
            conf=conf,
            iou=iou,
            device=device,
        )
        cached_val_preds["yoloe_text"] = res_ye_text.pop("preds_val")
        cached_neg_preds["yoloe_text"] = {
            p.stem: _extract_boxes(yoloe_text.predict(cv2.imread(str(p)), conf=c, iou=iou, device=device, verbose=False)[0])
            for p in neg_imgs
            for c in [conf]
        }
        runs.append(res_ye_text)
        del yoloe_text
    except Exception as exc:
        errors["yoloe_text"] = repr(exc)

    try:
        with _in_pretrained():
            yoloe_vp = YOLOE(yoloe_name)
        vp_payload = {"bboxes": ref_boxes, "cls": np.zeros((len(ref_boxes),), dtype=np.int32)}
        res_ye_vp = _evaluate_stream(
            label=f"yoloe_visual_prompt ({yoloe_name})",
            val_imgs=val_imgs,
            val_gts=val_gts,
            neg_imgs=neg_imgs,
            infer_fn=lambda bgr, c, iv: _extract_boxes(
                yoloe_vp.predict(
                    bgr,
                    refer_image=str(ref_img),
                    visual_prompts=vp_payload,
                    predictor=YOLOEVPSegPredictor,
                    conf=c,
                    iou=iv,
                    device=device,
                    verbose=False,
                )[0]
            ),
            conf=conf,
            iou=iou,
            device=device,
        )
        cached_val_preds["yoloe_vp"] = res_ye_vp.pop("preds_val")
        cached_neg_preds["yoloe_vp"] = {
            p.stem: _extract_boxes(
                yoloe_vp.predict(
                    cv2.imread(str(p)),
                    refer_image=str(ref_img),
                    visual_prompts=vp_payload,
                    predictor=YOLOEVPSegPredictor,
                    conf=conf,
                    iou=iou,
                    device=device,
                    verbose=False,
                )[0]
            )
            for p in neg_imgs
        }
        runs.append(res_ye_vp)
        del yoloe_vp
    except Exception as exc:
        errors["yoloe_visual_prompt"] = repr(exc)

    # 3. YOLO-Worldv2 text prompt
    try:
        with _in_pretrained():
            yw_model = YOLOWorld("yolov8s-worldv2.pt")
            yw_model.set_classes(resolved_prompts)
        res_yw = _evaluate_stream(
            label="yolov8s-worldv2_text",
            val_imgs=val_imgs,
            val_gts=val_gts,
            neg_imgs=neg_imgs,
            infer_fn=lambda bgr, c, iv: _extract_boxes(yw_model.predict(bgr, conf=c, iou=iv, device=device, verbose=False)[0]),
            conf=conf,
            iou=iou,
            device=device,
        )
        cached_val_preds["yoloworld"] = res_yw.pop("preds_val")
        cached_neg_preds["yoloworld"] = {
            p.stem: _extract_boxes(yw_model.predict(cv2.imread(str(p)), conf=conf, iou=iou, device=device, verbose=False)[0])
            for p in neg_imgs
        }
        runs.append(res_yw)
        del yw_model
    except Exception as exc:
        errors["yoloworld"] = repr(exc)

    # 4. Hybrid merge: exp04c + best F1 open-vocab teacher (NMS IoU 0.50)
    teacher_candidates = [
        (k, r["val_vid15"]["f1_50"])
        for k, r in zip(
            [key for key in ("yoloe_text", "yoloe_vp", "yoloworld") if key in cached_val_preds],
            [r for r in runs if any(tag in r["model_key"] for tag in ("yoloe_text", "yoloe_visual_prompt", "yolov8s-worldv2"))],
        )
    ]
    if teacher_candidates:
        best_teacher_key = max(teacher_candidates, key=lambda pair: pair[1])[0]
        hybrid_val = {
            p.stem: nms_merge_xyxy(cached_val_preds["exp04c"][p.stem], cached_val_preds[best_teacher_key][p.stem], iou_thresh=0.50)
            for p in val_imgs
        }
        hybrid_neg_fp = sum(
            len(nms_merge_xyxy(cached_neg_preds["exp04c"][p.stem], cached_neg_preds[best_teacher_key][p.stem], iou_thresh=0.50))
            for p in neg_imgs
        )
        runs.append(
            {
                "model_key": f"hybrid (exp04c + {best_teacher_key})",
                "val_vid15": compute_detection_metrics(hybrid_val, val_gts, iou_thresh=0.50),
                "train_neg_vid04": {
                    "frames": len(neg_imgs),
                    "false_positives": hybrid_neg_fp,
                    "fp_per_frame": round(float(hybrid_neg_fp / max(1, len(neg_imgs))), 4),
                },
                "mean_ms_per_frame": None,
                "peak_vram_mb": None,
            }
        )

    report = {
        "schemaVersion": "1.0",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "gpuName": torch.cuda.get_device_name(0) if (device.startswith("cuda") and torch.cuda.is_available()) else "CPU",
        "confThreshold": conf,
        "iouThreshold": iou,
        "prompts": resolved_prompts,
        "visualPromptRefImage": str(ref_img.relative_to(REPO_ROOT)).replace("\\", "/"),
        "valFrameCount": len(val_imgs),
        "trainNegativeFrameCount": len(neg_imgs),
        "isolatedExcludedIds": list(RESERVED_IDS),
        "runs": runs,
        "errors": errors,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark open-vocab teachers vs exp04c cleanly without test leakage.")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "data/processed/open-top-yolo-v1")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "artifacts/runs/preannotation/ws1_teacher_benchmark.json",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    report = run_benchmark(
        dataset_root=args.dataset_root,
        out_json=args.out_json,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
