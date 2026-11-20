"""Deterministic YOLO letterbox preprocessing and output tensor postprocessing."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.modules.vision.schemas import DetectionBox


def compute_letterbox_params(orig_width: int, orig_height: int, target_size: int = 640) -> dict[str, Any]:
    """Compute deterministic square letterbox geometry matching Ultralytics auto=False."""
    if orig_width <= 0 or orig_height <= 0 or target_size <= 0:
        raise ValueError("Image dimensions and target_size must be positive.")

    scale = min(target_size / orig_width, target_size / orig_height)
    new_w = max(1, int(round(orig_width * scale)))
    new_h = max(1, int(round(orig_height * scale)))
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


def compute_iou_xyxy(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
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


def _build_detection_box(
    *,
    class_id: int,
    class_names: dict[int, str],
    confidence: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    letterbox_meta: dict[str, Any],
) -> DetectionBox | None:
    ox1, oy1, ox2, oy2 = unletterbox_xyxy(x1, y1, x2, y2, letterbox_meta)
    bw = ox2 - ox1
    bh = oy2 - oy1
    if bw <= 0.0 or bh <= 0.0:
        return None
    orig_w = float(letterbox_meta["orig_width"])
    orig_h = float(letterbox_meta["orig_height"])
    return DetectionBox(
        class_id=class_id,
        class_name=class_names.get(class_id, f"class_{class_id}"),
        confidence=float(confidence),
        track_id=None,
        x1=ox1,
        y1=oy1,
        x2=ox2,
        y2=oy2,
        width_px=bw,
        height_px=bh,
        norm_x=ox1 / orig_w,
        norm_y=oy1 / orig_h,
        norm_width=bw / orig_w,
        norm_height=bh / orig_h,
    )


def decode_yolo_output(
    raw_output: np.ndarray,
    letterbox_meta: dict[str, Any],
    class_names: dict[int, str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> list[DetectionBox]:
    """Decode raw YOLO output ([1, 4+nc, N] or [1, N, 6]) into clipped original-pixel detections."""
    arr = np.asarray(raw_output, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"Unsupported YOLO output shape: {arr.shape}")

    candidates: list[DetectionBox] = []
    if arr.shape[2] == 6 and arr.shape[1] != 6:
        for row in arr[0]:
            conf = float(row[4])
            if conf < conf_threshold:
                continue
            cls_id = int(round(float(row[5])))
            box = _build_detection_box(
                class_id=cls_id,
                class_names=class_names,
                confidence=conf,
                x1=float(row[0]),
                y1=float(row[1]),
                x2=float(row[2]),
                y2=float(row[3]),
                letterbox_meta=letterbox_meta,
            )
            if box is not None:
                candidates.append(box)
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates[:max_detections]

    channels = arr.shape[1]
    if channels < 5:
        raise ValueError(f"Expected >= 5 channels in [1, 4+nc, N], got shape {arr.shape}")

    preds = arr[0].T
    class_scores = preds[:, 4:]
    best_classes = np.argmax(class_scores, axis=1)
    best_scores = class_scores[np.arange(len(preds)), best_classes]
    keep_mask = best_scores >= conf_threshold

    for row, cls_id_np, score_np in zip(preds[keep_mask], best_classes[keep_mask], best_scores[keep_mask]):
        cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        box = _build_detection_box(
            class_id=int(cls_id_np),
            class_names=class_names,
            confidence=float(score_np),
            x1=cx - w / 2.0,
            y1=cy - h / 2.0,
            x2=cx + w / 2.0,
            y2=cy + h / 2.0,
            letterbox_meta=letterbox_meta,
        )
        if box is not None:
            candidates.append(box)

    candidates.sort(key=lambda item: item.confidence, reverse=True)
    kept: list[DetectionBox] = []
    for cand in candidates:
        cand_box = (cand.x1, cand.y1, cand.x2, cand.y2)
        if any(
            prev.class_id == cand.class_id and compute_iou_xyxy(cand_box, (prev.x1, prev.y1, prev.x2, prev.y2)) >= iou_threshold
            for prev in kept
        ):
            continue
        kept.append(cand)
        if len(kept) >= max_detections:
            break
    return kept
