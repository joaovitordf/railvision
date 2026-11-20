"""Export exp04c YOLOv26n weights to ONNX and verify PyTorch vs. ONNX Runtime parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import ultralytics
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "artifacts/runs/detect/exp04c_yolo26n_opentop_freeze10/weights/best.pt"
DEFAULT_ONNX_OUTPUT = REPO_ROOT / "artifacts/models/onnx/exp04c_yolo26n_opentop_640.onnx"
DEFAULT_REPORT_DIR = REPO_ROOT / "artifacts/runs/phase4-onnx-spike"

PARITY_SAMPLE_SPECS = [
    ("val", "data/processed/open-top-yolo-v1/val/images/val__camp_sishen_saldanha__vid_15__f000075.jpg"),
    ("val", "data/processed/open-top-yolo-v1/val/images/val__camp_sishen_saldanha__vid_15__f000255.jpg"),
    ("val", "data/processed/open-top-yolo-v1/val/images/val__camp_sishen_saldanha__vid_15__f000524.jpg"),
    ("test", "data/processed/open-top-yolo-v1/test/images/test__camp_vfli_cosne__vid_09__f000275.jpg"),
    ("test", "data/processed/open-top-yolo-v1/test/images/test__camp_vfli_cosne__vid_09__f000075.jpg"),
    ("test", "data/processed/open-top-yolo-v1/test/images/test__camp_vfli_cosne__vid_16__f000125.jpg"),
    ("train", "data/processed/open-top-yolo-v1/train/images/train__camp_coal_tunnel__vid_14__f003357.jpg"),
    ("train", "data/processed/open-top-yolo-v1/train/images/train__camp_perm_overpass__vid_04__f000030.jpg"),
]


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_letterbox_params(orig_width: int, orig_height: int, target_size: int = 640) -> dict[str, Any]:
    """Compute deterministic square letterbox geometry matching Ultralytics auto=False."""
    if orig_width <= 0 or orig_height <= 0 or target_size <= 0:
        raise ValueError("Image dimensions and target_size must be positive.")

    scale = min(target_size / orig_width, target_size / orig_height)
    new_w = int(round(orig_width * scale))
    new_h = int(round(orig_height * scale))
    dw = (target_size - new_w) / 2.0
    dh = (target_size - new_h) / 2.0
    pad_left = int(round(dw - 0.1))
    pad_right = int(round(dw + 0.1))
    pad_top = int(round(dh - 0.1))
    pad_bottom = int(round(dh + 0.1))

    return {
        "orig_width": orig_width,
        "orig_height": orig_height,
        "target_size": target_size,
        "scale": float(scale),
        "resized_width": new_w,
        "resized_height": new_h,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
    }


def preprocess_bgr_letterbox(
    bgr_image: np.ndarray,
    target_size: int = 640,
    pad_value: int = 114,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resize with aspect ratio preservation, pad, and convert BGR HWC uint8 to RGB NCHW float32 [0, 1]."""
    orig_h, orig_w = bgr_image.shape[:2]
    meta = compute_letterbox_params(orig_w, orig_h, target_size=target_size)
    new_w = meta["resized_width"]
    new_h = meta["resized_height"]

    resized = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized,
        meta["pad_top"],
        meta["pad_bottom"],
        meta["pad_left"],
        meta["pad_right"],
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    if padded.shape[0] != target_size or padded.shape[1] != target_size:
        padded = padded[:target_size, :target_size]

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)), dtype=np.float32) / 255.0
    nchw = np.expand_dims(chw, axis=0)
    return nchw, meta


def compute_iou_xyxy(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Compute intersection-over-union for two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def unletterbox_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    letterbox_meta: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Map letterboxed target-space [x1, y1, x2, y2] coordinates back to clipped original image pixels."""
    scale = letterbox_meta["scale"]
    pad_left = letterbox_meta["pad_left"]
    pad_top = letterbox_meta["pad_top"]
    orig_w = float(letterbox_meta["orig_width"])
    orig_h = float(letterbox_meta["orig_height"])

    ox1 = max(0.0, min(orig_w, (x1 - pad_left) / scale))
    oy1 = max(0.0, min(orig_h, (y1 - pad_top) / scale))
    ox2 = max(0.0, min(orig_w, (x2 - pad_left) / scale))
    oy2 = max(0.0, min(orig_h, (y2 - pad_top) / scale))
    return ox1, oy1, ox2, oy2


def decode_yolo_output(
    raw_output: np.ndarray,
    letterbox_meta: dict[str, Any],
    class_names: dict[int, str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> list[dict[str, Any]]:
    """Decode raw YOLO ONNX output ([1, 4+nc, N] or [1, N, 6]) into clipped original-pixel detections."""
    arr = np.asarray(raw_output, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"Unsupported YOLO ONNX output shape: {arr.shape}")

    candidates: list[dict[str, Any]] = []
    if arr.shape[2] == 6 and arr.shape[1] != 6:
        # End-to-end [1, N, 6]: x1, y1, x2, y2, score, class_id
        for row in arr[0]:
            score = float(row[4])
            if score < conf_threshold:
                continue
            cls_id = int(round(float(row[5])))
            ox1, oy1, ox2, oy2 = unletterbox_xyxy(
                float(row[0]), float(row[1]), float(row[2]), float(row[3]), letterbox_meta
            )
            if ox2 > ox1 and oy2 > oy1:
                candidates.append(
                    {
                        "classId": cls_id,
                        "className": class_names.get(cls_id, f"class_{cls_id}"),
                        "confidence": score,
                        "x1": ox1,
                        "y1": oy1,
                        "x2": ox2,
                        "y2": oy2,
                    }
                )
        candidates.sort(key=lambda d: d["confidence"], reverse=True)
        return candidates[:max_detections]

    # Standard Detect head output [1, 4 + nc, num_anchors] -> transpose to [num_anchors, 4 + nc]
    channels = arr.shape[1]
    if channels < 5:
        raise ValueError(f"Expected >= 5 channels in YOLO output, got shape {arr.shape}")
    num_classes = channels - 4
    preds = arr[0].T

    class_scores = preds[:, 4 : 4 + num_classes]
    best_classes = np.argmax(class_scores, axis=1)
    best_scores = class_scores[np.arange(preds.shape[0]), best_classes]
    mask = best_scores >= conf_threshold

    filtered_preds = preds[mask]
    filtered_classes = best_classes[mask]
    filtered_scores = best_scores[mask]

    for row, cls_id_np, score_np in zip(filtered_preds, filtered_classes, filtered_scores):
        cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        ox1, oy1, ox2, oy2 = unletterbox_xyxy(x1, y1, x2, y2, letterbox_meta)
        if ox2 <= ox1 or oy2 <= oy1:
            continue
        cls_id = int(cls_id_np)
        candidates.append(
            {
                "classId": cls_id,
                "className": class_names.get(cls_id, f"class_{cls_id}"),
                "confidence": float(score_np),
                "x1": ox1,
                "y1": oy1,
                "x2": ox2,
                "y2": oy2,
            }
        )

    candidates.sort(key=lambda d: d["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []
    for cand in candidates:
        cand_box = (cand["x1"], cand["y1"], cand["x2"], cand["y2"])
        suppressed = False
        for prev in kept:
            if prev["classId"] != cand["classId"]:
                continue
            prev_box = (prev["x1"], prev["y1"], prev["x2"], prev["y2"])
            if compute_iou_xyxy(cand_box, prev_box) >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(cand)
            if len(kept) >= max_detections:
                break
    return kept


def match_detections(
    ref_boxes: list[dict[str, Any]],
    cand_boxes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare two detection lists by class and greedy best-IoU matching."""
    used = set()
    ious: list[float] = []
    max_coord_diffs: list[float] = []
    conf_diffs: list[float] = []

    for ref in ref_boxes:
        best_iou = -1.0
        best_idx = -1
        ref_xyxy = (ref["x1"], ref["y1"], ref["x2"], ref["y2"])
        for idx, cand in enumerate(cand_boxes):
            if idx in used or cand["classId"] != ref["classId"]:
                continue
            cand_xyxy = (cand["x1"], cand["y1"], cand["x2"], cand["y2"])
            iou = compute_iou_xyxy(ref_xyxy, cand_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0:
            used.add(best_idx)
            matched = cand_boxes[best_idx]
            ious.append(float(best_iou))
            max_coord_diffs.append(
                max(
                    abs(ref["x1"] - matched["x1"]),
                    abs(ref["y1"] - matched["y1"]),
                    abs(ref["x2"] - matched["x2"]),
                    abs(ref["y2"] - matched["y2"]),
                )
            )
            conf_diffs.append(abs(ref["confidence"] - matched["confidence"]))

    return {
        "refCount": len(ref_boxes),
        "candCount": len(cand_boxes),
        "countMatch": len(ref_boxes) == len(cand_boxes),
        "matchedCount": len(ious),
        "minMatchedIou": float(min(ious)) if ious else None,
        "maxCoordDiffPx": float(max(max_coord_diffs)) if max_coord_diffs else 0.0,
        "maxConfDiff": float(max(conf_diffs)) if conf_diffs else 0.0,
    }


def export_isolated_onnx(
    weights_path: Path,
    output_onnx_path: Path,
    imgsz: int,
    opset: int,
) -> dict[str, Any]:
    """Export PyTorch checkpoint in a temp directory so original checkpoint folder is untouched."""
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="railvision_onnx_export_") as tmpdir:
        tmp_dir = Path(tmpdir)
        tmp_weights = tmp_dir / weights_path.name
        shutil.copy2(weights_path, tmp_weights)

        yolo = YOLO(str(tmp_weights))
        exported = Path(
            yolo.export(
                format="onnx",
                imgsz=imgsz,
                opset=opset,
                dynamic=False,
                simplify=False,
            )
        )
        if not exported.exists():
            raise FileNotFoundError(f"Expected exported ONNX artifact at {exported}")

        shutil.copy2(exported, output_onnx_path)
        class_names = {int(k): str(v) for k, v in yolo.names.items()}
        end2end = bool(getattr(yolo.model, "end2end", False))

    model_proto = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(model_proto)
    opset_imports = [{"domain": imp.domain or "ai.onnx", "version": int(imp.version)} for imp in model_proto.opset_import]
    graph_inputs = [
        {
            "name": inp.name,
            "dims": [
                int(d.dim_value) if d.HasField("dim_value") else str(d.dim_param)
                for d in inp.type.tensor_type.shape.dim
            ],
        }
        for inp in model_proto.graph.input
    ]
    graph_outputs = [
        {
            "name": out.name,
            "dims": [
                int(d.dim_value) if d.HasField("dim_value") else str(d.dim_param)
                for d in out.type.tensor_type.shape.dim
            ],
        }
        for out in model_proto.graph.output
    ]
    return {
        "classNames": class_names,
        "end2end": end2end,
        "opsetImports": opset_imports,
        "graphInputs": graph_inputs,
        "graphOutputs": graph_outputs,
        "nodeCount": len(model_proto.graph.node),
        "onnxSha256": sha256_file(output_onnx_path),
        "onnxFileSizeBytes": output_onnx_path.stat().st_size,
    }


def benchmark_python_runtimes(
    pt_model_cpu: torch.nn.Module,
    pt_model_cuda: torch.nn.Module | None,
    ort_session: ort.InferenceSession,
    input_name: str,
    sample_nchw: np.ndarray,
    warmup: int = 5,
    runs: int = 25,
) -> dict[str, Any]:
    """Measure isolated single-tensor forward latency across Python runtimes."""
    results: dict[str, Any] = {}

    x_cpu = torch.from_numpy(sample_nchw).cpu()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = pt_model_cpu(x_cpu)
        times_cpu_ms = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = pt_model_cpu(x_cpu)
            times_cpu_ms.append((time.perf_counter() - t0) * 1000.0)
    results["pytorch_cpu"] = summarize_ms(times_cpu_ms)

    if pt_model_cuda is not None and torch.cuda.is_available():
        x_cuda = torch.from_numpy(sample_nchw).to("cuda:0")
        with torch.inference_mode():
            for _ in range(warmup):
                _ = pt_model_cuda(x_cuda)
            torch.cuda.synchronize()
            times_cuda_ms = []
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = pt_model_cuda(x_cuda)
                torch.cuda.synchronize()
                times_cuda_ms.append((time.perf_counter() - t0) * 1000.0)
        results["pytorch_cuda"] = summarize_ms(times_cuda_ms)

    for _ in range(warmup):
        _ = ort_session.run(None, {input_name: sample_nchw})
    times_ort_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = ort_session.run(None, {input_name: sample_nchw})
        times_ort_ms.append((time.perf_counter() - t0) * 1000.0)
    results["onnxruntime_python_cpu"] = summarize_ms(times_ort_ms)
    return results


def summarize_ms(values_ms: list[float]) -> dict[str, float]:
    """Compute avg, p50, p95, min, max, and theoretical FPS from millisecond timings."""
    arr = np.asarray(values_ms, dtype=np.float64)
    avg_ms = float(np.mean(arr))
    return {
        "runs": len(values_ms),
        "avgMs": round(avg_ms, 3),
        "p50Ms": round(float(np.percentile(arr, 50)), 3),
        "p95Ms": round(float(np.percentile(arr, 95)), 3),
        "minMs": round(float(np.min(arr)), 3),
        "maxMs": round(float(np.max(arr)), 3),
        "theoreticalFps": round(1000.0 / avg_ms, 2) if avg_ms > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export exp04c to ONNX and verify PyTorch vs ONNX parity.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_ONNX_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    weights_path = args.weights.resolve()
    output_onnx_path = args.output.resolve()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    parity_frames_dir = report_dir / "parity_frames"
    parity_frames_dir.mkdir(parents=True, exist_ok=True)

    source_sha256 = sha256_file(weights_path)
    export_info = export_isolated_onnx(weights_path, output_onnx_path, imgsz=args.imgsz, opset=args.opset)
    class_names: dict[int, str] = export_info["classNames"]

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(
        str(output_onnx_path),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # Write model sidecar metadata
    meta_path = output_onnx_path.with_suffix(".meta.json")
    model_meta_payload = {
        "modelName": "exp04c_yolo26n_opentop_640",
        "sourceExperiment": "exp04c_yolo26n_opentop_freeze10",
        "sourceWeightsRelativePath": str(weights_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "sourceWeightsSha256": source_sha256,
        "onnxRelativePath": str(output_onnx_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "onnxSha256": export_info["onnxSha256"],
        "onnxFileSizeBytes": export_info["onnxFileSizeBytes"],
        "opset": args.opset,
        "opsetImports": export_info["opsetImports"],
        "imgsz": args.imgsz,
        "end2end": export_info["end2end"],
        "inputName": input_name,
        "inputShape": export_info["graphInputs"][0]["dims"],
        "outputName": output_name,
        "outputShape": export_info["graphOutputs"][0]["dims"],
        "classNames": {str(k): v for k, v in class_names.items()},
        "defaultConf": args.conf,
        "defaultIou": args.iou,
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "limitationsNotice": (
            "PoC model (exp04c): good validation score on vid_15 (mAP50=0.9218), "
            "limited generalization on held-out test vid_09/vid_16 (mAP50=0.2558). "
            "Not approved for production."
        ),
    }
    meta_path.write_text(json.dumps(model_meta_payload, indent=2), encoding="utf-8")

    # Prepare PyTorch CPU, PyTorch CUDA, and Ultralytics predictors
    yolo_ultralytics = YOLO(str(weights_path))
    pt_model_cpu = YOLO(str(weights_path)).model.eval().cpu()
    pt_model_cuda = YOLO(str(weights_path)).model.eval().to("cuda:0") if torch.cuda.is_available() else None

    frame_reports: list[dict[str, Any]] = []
    parity_ref_frames: list[dict[str, Any]] = []
    first_nchw: np.ndarray | None = None

    for split_name, rel_path in PARITY_SAMPLE_SPECS:
        src_path = REPO_ROOT / rel_path
        if not src_path.exists():
            raise FileNotFoundError(f"Parity sample missing: {src_path}")

        copied_path = parity_frames_dir / src_path.name
        shutil.copy2(src_path, copied_path)

        bgr = cv2.imread(str(src_path))
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {src_path}")
        nchw, lb_meta = preprocess_bgr_letterbox(bgr, target_size=args.imgsz)
        if first_nchw is None:
            first_nchw = nchw

        with torch.inference_mode():
            pt_cpu_raw = pt_model_cpu(torch.from_numpy(nchw).cpu())[0].numpy()
            pt_cuda_raw = (
                pt_model_cuda(torch.from_numpy(nchw).to("cuda:0"))[0].detach().cpu().numpy()
                if pt_model_cuda is not None
                else None
            )

        ort_raw = np.asarray(ort_session.run([output_name], {input_name: nchw})[0])

        diff_pt_cpu_vs_ort = np.abs(pt_cpu_raw - ort_raw)
        diff_pt_cuda_vs_ort = np.abs(pt_cuda_raw - ort_raw) if pt_cuda_raw is not None else None

        pt_cpu_dets = decode_yolo_output(pt_cpu_raw, lb_meta, class_names, conf_threshold=args.conf, iou_threshold=args.iou)
        ort_dets = decode_yolo_output(ort_raw, lb_meta, class_names, conf_threshold=args.conf, iou_threshold=args.iou)
        pt_cuda_dets = (
            decode_yolo_output(pt_cuda_raw, lb_meta, class_names, conf_threshold=args.conf, iou_threshold=args.iou)
            if pt_cuda_raw is not None
            else []
        )

        # Also compare directly against Ultralytics high-level predict(rect=False)
        ultra_res = yolo_ultralytics.predict(
            str(src_path),
            imgsz=args.imgsz,
            rect=False,
            conf=args.conf,
            iou=args.iou,
            device="cpu",
            verbose=False,
        )[0]
        ultra_dets = []
        if ultra_res.boxes is not None and len(ultra_res.boxes) > 0:
            xyxys = ultra_res.boxes.xyxy.cpu().numpy()
            confs = ultra_res.boxes.conf.cpu().numpy()
            clss = ultra_res.boxes.cls.cpu().numpy()
            for xyxy, cf, cl in zip(xyxys, confs, clss):
                cls_id = int(cl)
                ultra_dets.append(
                    {
                        "classId": cls_id,
                        "className": class_names.get(cls_id, f"class_{cls_id}"),
                        "confidence": float(cf),
                        "x1": float(xyxy[0]),
                        "y1": float(xyxy[1]),
                        "x2": float(xyxy[2]),
                        "y2": float(xyxy[3]),
                    }
                )
            ultra_dets.sort(key=lambda d: d["confidence"], reverse=True)

        cmp_cpu_ort = match_detections(pt_cpu_dets, ort_dets)
        cmp_cuda_ort = match_detections(pt_cuda_dets, ort_dets) if pt_cuda_raw is not None else None
        cmp_ultra_ort = match_detections(ultra_dets, ort_dets)

        frame_entry = {
            "filename": src_path.name,
            "split": split_name,
            "width": lb_meta["orig_width"],
            "height": lb_meta["orig_height"],
            "rawTensorDiff": {
                "pytorchCpuVsOrtCpuMaxAbs": float(np.max(diff_pt_cpu_vs_ort)),
                "pytorchCpuVsOrtCpuMeanAbs": float(np.mean(diff_pt_cpu_vs_ort)),
                "pytorchCudaVsOrtCpuMaxAbs": float(np.max(diff_pt_cuda_vs_ort))
                if diff_pt_cuda_vs_ort is not None
                else None,
                "pytorchCudaVsOrtCpuMeanAbs": float(np.mean(diff_pt_cuda_vs_ort))
                if diff_pt_cuda_vs_ort is not None
                else None,
            },
            "boxComparison": {
                "pytorchCpuVsOrtCpu": cmp_cpu_ort,
                "pytorchCudaVsOrtCpu": cmp_cuda_ort,
                "ultralyticsCpuVsOrtCpu": cmp_ultra_ort,
            },
            "ortDetections": ort_dets,
            "pytorchCpuDetections": pt_cpu_dets,
            "ultralyticsDetections": ultra_dets,
        }
        frame_reports.append(frame_entry)

        parity_ref_frames.append(
            {
                "filename": src_path.name,
                "split": split_name,
                "width": lb_meta["orig_width"],
                "height": lb_meta["orig_height"],
                "expectedDetections": ort_dets,
                "pytorchCpuCount": len(pt_cpu_dets),
                "rawTopScore": float(np.max(ort_raw[0, 4:, :])),
            }
        )

    assert first_nchw is not None
    python_timings = benchmark_python_runtimes(
        pt_model_cpu=pt_model_cpu,
        pt_model_cuda=pt_model_cuda,
        ort_session=ort_session,
        input_name=input_name,
        sample_nchw=first_nchw,
    )

    parity_summary = {
        "totalFrames": len(frame_reports),
        "pytorchCpuVsOrtCountMatches": sum(
            1 for f in frame_reports if f["boxComparison"]["pytorchCpuVsOrtCpu"]["countMatch"]
        ),
        "pytorchCudaVsOrtCountMatches": sum(
            1
            for f in frame_reports
            if f["boxComparison"]["pytorchCudaVsOrtCpu"] is not None
            and f["boxComparison"]["pytorchCudaVsOrtCpu"]["countMatch"]
        ),
        "ultralyticsVsOrtCountMatches": sum(
            1 for f in frame_reports if f["boxComparison"]["ultralyticsCpuVsOrtCpu"]["countMatch"]
        ),
        "maxRawAbsDiffPytorchCpuVsOrt": max(
            f["rawTensorDiff"]["pytorchCpuVsOrtCpuMaxAbs"] for f in frame_reports
        ),
        "maxRawAbsDiffPytorchCudaVsOrt": max(
            f["rawTensorDiff"]["pytorchCudaVsOrtCpuMaxAbs"]
            for f in frame_reports
            if f["rawTensorDiff"]["pytorchCudaVsOrtCpuMaxAbs"] is not None
        )
        if pt_model_cuda is not None
        else None,
        "maxBoxCoordDiffPxCpuVsOrt": max(
            f["boxComparison"]["pytorchCpuVsOrtCpu"]["maxCoordDiffPx"] for f in frame_reports
        ),
        "maxConfDiffCpuVsOrt": max(
            f["boxComparison"]["pytorchCpuVsOrtCpu"]["maxConfDiff"] for f in frame_reports
        ),
        "maxBoxCoordDiffPxUltralyticsVsOrt": max(
            f["boxComparison"]["ultralyticsCpuVsOrtCpu"]["maxCoordDiffPx"] for f in frame_reports
        ),
    }

    full_report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "model": model_meta_payload,
        "ortAvailableProviders": ort.get_available_providers(),
        "ortActiveProviders": ort_session.get_providers(),
        "confThreshold": args.conf,
        "iouThreshold": args.iou,
        "paritySummary": parity_summary,
        "pythonRuntimeBenchmarks": python_timings,
        "frames": frame_reports,
    }
    report_path = report_dir / "pytorch_vs_onnx_parity.json"
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    ref_payload = {
        "onnxSha256": export_info["onnxSha256"],
        "imgsz": args.imgsz,
        "confThreshold": args.conf,
        "iouThreshold": args.iou,
        "classNames": {str(k): v for k, v in class_names.items()},
        "frames": parity_ref_frames,
    }
    ref_path = report_dir / "parity_reference.json"
    ref_path.write_text(json.dumps(ref_payload, indent=2), encoding="utf-8")

    print(f"Exported ONNX: {output_onnx_path} ({export_info['onnxFileSizeBytes']} bytes, sha256={export_info['onnxSha256'][:12]})")
    print(f"ORT providers: active={ort_session.get_providers()} available={ort.get_available_providers()}")
    print(f"Parity summary: {json.dumps(parity_summary, indent=2)}")
    print(f"Python timings: {json.dumps(python_timings, indent=2)}")


if __name__ == "__main__":
    main()
