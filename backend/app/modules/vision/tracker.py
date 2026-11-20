"""Opt-in ByteTrack multi-object tracking wrapper isolated per video execution."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from app.modules.vision.schemas import DetectionBox


@dataclass(slots=True)
class _TrackerBoxes:
    """Lightweight NumPy adapter satisfying Ultralytics BYTETracker.update."""

    xywh: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return int(len(self.conf))

    def __getitem__(self, mask: np.ndarray) -> _TrackerBoxes:
        return _TrackerBoxes(
            xywh=self.xywh[mask],
            conf=self.conf[mask],
            cls=self.cls[mask],
        )


class ByteTrackSession:
    """Per-video stateful ByteTrack tracker associating per-frame DetectionBox predictions."""

    def __init__(
        self,
        *,
        high_thresh: float = 0.25,
        low_thresh: float = 0.1,
        new_track_thresh: float | None = None,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ) -> None:
        args = SimpleNamespace(
            track_high_thresh=max(0.05, float(high_thresh)),
            track_low_thresh=float(low_thresh),
            new_track_thresh=float(new_track_thresh if new_track_thresh is not None else max(high_thresh, 0.25)),
            track_buffer=int(track_buffer),
            match_thresh=float(match_thresh),
            fuse_score=True,
        )
        self._tracker = BYTETracker(args)

    @property
    def candidate_conf_threshold(self) -> float:
        """Lower detection threshold forwarded to detector so ByteTrack can run two-stage association."""
        return min(float(self._tracker.args.track_high_thresh), float(self._tracker.args.track_low_thresh))

    def update(self, detections: list[DetectionBox]) -> list[DetectionBox]:
        """Update tracker state for the next frame and return detections with persistent track_id."""
        if not detections:
            empty = _TrackerBoxes(
                xywh=np.empty((0, 4), dtype=np.float32),
                conf=np.empty((0,), dtype=np.float32),
                cls=np.empty((0,), dtype=np.float32),
            )
            self._tracker.update(empty)
            return []

        xywh = np.array(
            [
                [
                    det.x1 + (det.width_px / 2.0),
                    det.y1 + (det.height_px / 2.0),
                    det.width_px,
                    det.height_px,
                ]
                for det in detections
            ],
            dtype=np.float32,
        )
        conf = np.array([det.confidence for det in detections], dtype=np.float32)
        cls = np.array([det.class_id for det in detections], dtype=np.float32)

        tracked_rows = self._tracker.update(_TrackerBoxes(xywh=xywh, conf=conf, cls=cls))
        if tracked_rows is None or len(tracked_rows) == 0:
            return []

        tracked_boxes: list[DetectionBox] = []
        for row in tracked_rows:
            track_id = int(round(float(row[4])))
            det_idx = int(round(float(row[7])))
            if det_idx < 0 or det_idx >= len(detections):
                continue
            src = detections[det_idx]
            tracked_boxes.append(src.model_copy(update={"track_id": track_id}))

        tracked_boxes.sort(key=lambda item: item.confidence, reverse=True)
        return tracked_boxes
