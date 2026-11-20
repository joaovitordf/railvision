"""Shared contact-sheet rendering helper for audit, extraction, and QA inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont


def save_labeled_grid(
    frames_bgr: list[tuple[str, Any]],
    out_path: Path,
    cols: int = 3,
    cell_w: int = 480,
    cell_h: int = 270,
    quality: int = 88,
) -> None:
    """Save a JPEG contact sheet of BGR frames with black caption banners."""
    if not frames_bgr:
        return
    rows = (len(frames_bgr) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (18, 22, 28))
    font = ImageFont.load_default()
    for idx, (caption, bgr) in enumerate(frames_bgr):
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 0, cell_w, 18], fill=(0, 0, 0))
        draw.text((6, 3), caption[:95], fill="#ffffff", font=font)
        r, c = divmod(idx, cols)
        sheet.paste(im, (c * cell_w, r * cell_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=quality)
