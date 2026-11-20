"""Bounding-box overlay rendering onto defensive frame copies."""

from __future__ import annotations

import cv2
import numpy as np

from app.modules.vision.schemas import DetectionBox

BOX_COLOR_BGR = (0, 106, 255)
LABEL_BG_BGR = (40, 24, 16)
TEXT_COLOR_BGR = (255, 255, 255)


def render_annotated_frame(
    bgr_frame: np.ndarray,
    detections: list[DetectionBox],
    *,
    target_width: int | None = None,
    target_height: int | None = None,
) -> np.ndarray:
    """Return an annotated BGR uint8 copy resized cleanly without mutating the source array."""
    src_h, src_w = bgr_frame.shape[:2]
    out_w = int(target_width) if target_width is not None else src_w
    out_h = int(target_height) if target_height is not None else src_h

    if out_w != src_w or out_h != src_h:
        interp = cv2.INTER_AREA if (out_w < src_w or out_h < src_h) else cv2.INTER_LINEAR
        canvas = cv2.resize(bgr_frame, (out_w, out_h), interpolation=interp)
    else:
        canvas = bgr_frame.copy()

    if not detections:
        return canvas

    scale_x = out_w / max(1, src_w)
    scale_y = out_h / max(1, src_h)
    thickness = max(2, min(out_w, out_h) // 280)
    font_scale = max(0.5, min(out_w, out_h) / 1400.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = max(1, thickness - 1)

    for det in detections:
        x1 = max(0, min(out_w - 1, int(round(det.x1 * scale_x))))
        y1 = max(0, min(out_h - 1, int(round(det.y1 * scale_y))))
        x2 = max(x1 + 1, min(out_w - 1, int(round(det.x2 * scale_x))))
        y2 = max(y1 + 1, min(out_h - 1, int(round(det.y2 * scale_y))))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR_BGR, thickness=thickness, lineType=cv2.LINE_AA)

        prefix = f"#{det.track_id} " if det.track_id is not None else ""
        label = f"{prefix}{det.class_name} {det.confidence * 100.0:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        pad_x = 6
        pad_y = 4
        pill_w = text_w + (pad_x * 2)
        pill_h = text_h + baseline + (pad_y * 2)

        pill_x1 = max(0, min(out_w - pill_w, x1))
        pill_y1 = (y1 - pill_h - 2) if y1 >= pill_h + 2 else min(out_h - pill_h, y1 + 2)
        pill_x2 = min(out_w - 1, pill_x1 + pill_w)
        pill_y2 = min(out_h - 1, pill_y1 + pill_h)

        cv2.rectangle(canvas, (pill_x1, pill_y1), (pill_x2, pill_y2), LABEL_BG_BGR, thickness=-1)
        cv2.rectangle(canvas, (pill_x1, pill_y1), (pill_x2, pill_y2), BOX_COLOR_BGR, thickness=1)
        text_org = (pill_x1 + pad_x, pill_y1 + pad_y + text_h)
        cv2.putText(canvas, label, text_org, font, font_scale, TEXT_COLOR_BGR, text_thickness, cv2.LINE_AA)

    return canvas
