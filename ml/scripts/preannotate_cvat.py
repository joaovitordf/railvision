"""Local CVAT-compatible pre-annotation generator with strict HITL provenance and uncertainty ranking."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterator
import zipfile

import av
import cv2
import numpy as np
import torch
import torchvision

REPO_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_DIR = REPO_ROOT / "artifacts" / "models" / "pretrained"
DEFAULT_EXP04C = REPO_ROOT / "artifacts" / "runs" / "detect" / "exp04c_yolo26n_opentop_freeze10" / "weights" / "best.pt"
DEFAULT_PROMPTS = ["open-top freight railcar", "gondola rail car", "hopper railcar", "open top freight wagon"]
CLASS_NAME = "open_top_freight_wagon"


@contextmanager
def _in_pretrained_dir() -> Iterator[None]:
    import os

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    prev = os.getcwd()
    try:
        os.chdir(str(PRETRAINED_DIR))
        yield
    finally:
        os.chdir(prev)


def _sha256_or_none(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_weights_path(raw: str | None, fallback_name: str) -> Path:
    if not raw:
        candidate = PRETRAINED_DIR / fallback_name
        return candidate if candidate.is_file() else Path(fallback_name)
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    rel_repo = (REPO_ROOT / p).resolve()
    if rel_repo.is_file():
        return rel_repo
    rel_pre = (PRETRAINED_DIR / p).resolve()
    return rel_pre


def load_yolo_boxes_xyxy(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Read normalized YOLO .txt boxes into pixel xyxy array."""
    rows: list[list[float]] = []
    if label_path.is_file():
        for ln in label_path.read_text(encoding="utf-8").splitlines():
            parts = ln.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = (float(x) for x in parts)
            x1 = max(0.0, (cx - bw / 2.0) * img_w)
            y1 = max(0.0, (cy - bh / 2.0) * img_h)
            x2 = min(float(img_w), (cx + bw / 2.0) * img_w)
            y2 = min(float(img_h), (cy + bh / 2.0) * img_h)
            if x2 > x1 and y2 > y1:
                rows.append([x1, y1, x2, y2])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)


def nms_merge_xyxy(
    boxes_a: list[tuple[float, float, float, float, float]],
    boxes_b: list[tuple[float, float, float, float, float]],
    iou_thresh: float = 0.5,
) -> list[tuple[float, float, float, float, float]]:
    """Class-agnostic NMS merge across two detector streams [(x1,y1,x2,y2,conf), ...]."""
    merged = list(boxes_a) + list(boxes_b)
    if not merged:
        return []
    tensor_boxes = torch.tensor([b[:4] for b in merged], dtype=torch.float32)
    tensor_scores = torch.tensor([b[4] for b in merged], dtype=torch.float32)
    keep = torchvision.ops.nms(tensor_boxes, tensor_scores, float(iou_thresh)).tolist()
    return [merged[idx] for idx in keep]


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between two [N, 4] and [M, 4] xyxy arrays."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    ix1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    iy1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    ix2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    iy2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    union = np.maximum(1e-6, area_a[:, None] + area_b[None, :] - inter)
    return inter / union


def _compute_uncertainty_score(
    boxes_primary: list[tuple[float, float, float, float, float]],
    boxes_secondary: list[tuple[float, float, float, float, float]] | None = None,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if not boxes_primary:
        score = 0.75
        reasons.append("zero_detections")
    else:
        confs = [b[4] for b in boxes_primary]
        mean_conf = sum(confs) / len(confs)
        min_conf = min(confs)
        score = max(0.0, min(1.0, (1.0 - min_conf) * 0.6 + (1.0 - mean_conf) * 0.4))
        if min_conf < 0.45:
            reasons.append("low_min_confidence")
        if len(boxes_primary) >= 6:
            score = min(1.0, score + 0.1)
            reasons.append("crowded_frame")

    if boxes_secondary is not None:
        arr_a = np.asarray([b[:4] for b in boxes_primary], dtype=np.float32) if boxes_primary else np.zeros((0, 4), dtype=np.float32)
        arr_b = np.asarray([b[:4] for b in boxes_secondary], dtype=np.float32) if boxes_secondary else np.zeros((0, 4), dtype=np.float32)
        if len(arr_a) != len(arr_b):
            score = min(1.0, score + 0.25)
            reasons.append("teacher_student_count_mismatch")
        elif len(arr_a) > 0:
            iou_mat = box_iou_matrix(arr_a, arr_b)
            matched = int((iou_mat.max(axis=1) >= 0.5).sum())
            if matched < len(arr_a):
                score = min(1.0, score + 0.2)
                reasons.append("teacher_student_iou_disagreement")
    return round(float(score), 4), reasons


def build_predictor(
    *,
    engine: str,
    device: str,
    weights: str | None = None,
    prompts: list[str] | None = None,
    visual_ref_image: Path | None = None,
    visual_ref_label: Path | None = None,
):
    """Build local single-frame predictor function(bgr) -> list[(x1,y1,x2,y2,conf)]."""
    from ultralytics import YOLO, YOLOE, YOLOWorld
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    resolved_prompts = prompts or DEFAULT_PROMPTS

    if engine == "exp04c":
        w_path = _resolve_weights_path(weights, str(DEFAULT_EXP04C))
        if not w_path.is_file():
            raise FileNotFoundError(f"Checkpoint exp04c não encontrado: {w_path}")
        model = YOLO(str(w_path))

        def _pred(bgr: np.ndarray, conf: float, iou: float) -> list[tuple[float, float, float, float, float]]:
            res = model.predict(bgr, conf=conf, iou=iou, device=device, verbose=False)[0]
            return _extract_ultralytics_boxes(res)

        return _pred, str(w_path), _sha256_or_none(w_path)

    if engine == "yoloe":
        with _in_pretrained_dir():
            w_path = _resolve_weights_path(weights, "yoloe-26s-seg.pt")
            model = YOLOE(str(w_path))
            if visual_ref_image is not None and visual_ref_label is not None:
                ref_bgr = cv2.imread(str(visual_ref_image))
                if ref_bgr is None:
                    raise ValueError(f"Não foi possível ler imagem de referência: {visual_ref_image}")
                rh, rw = ref_bgr.shape[:2]
                gt_xyxy = load_yolo_boxes_xyxy(visual_ref_label, rw, rh)
                if len(gt_xyxy) == 0:
                    raise ValueError(f"Label de referência sem caixas válidas: {visual_ref_label}")
                vp = {"bboxes": gt_xyxy[:4], "cls": np.zeros((min(4, len(gt_xyxy)),), dtype=np.int32)}

                def _pred_vp(bgr: np.ndarray, conf: float, iou: float) -> list[tuple[float, float, float, float, float]]:
                    res = model.predict(
                        bgr,
                        refer_image=str(visual_ref_image),
                        visual_prompts=vp,
                        predictor=YOLOEVPSegPredictor,
                        conf=conf,
                        iou=iou,
                        device=device,
                        verbose=False,
                    )[0]
                    return _extract_ultralytics_boxes(res)

                return _pred_vp, str(w_path), _sha256_or_none(w_path)

            model.set_classes(resolved_prompts, model.get_text_pe(resolved_prompts))

            def _pred_text(bgr: np.ndarray, conf: float, iou: float) -> list[tuple[float, float, float, float, float]]:
                res = model.predict(bgr, conf=conf, iou=iou, device=device, verbose=False)[0]
                return _extract_ultralytics_boxes(res)

            return _pred_text, str(w_path), _sha256_or_none(w_path)

    if engine == "yoloworld":
        with _in_pretrained_dir():
            w_path = _resolve_weights_path(weights, "yolov8s-worldv2.pt")
            model = YOLOWorld(str(w_path))
            model.set_classes(resolved_prompts)

            def _pred_world(bgr: np.ndarray, conf: float, iou: float) -> list[tuple[float, float, float, float, float]]:
                res = model.predict(bgr, conf=conf, iou=iou, device=device, verbose=False)[0]
                return _extract_ultralytics_boxes(res)

            return _pred_world, str(w_path), _sha256_or_none(w_path)

    raise ValueError(f"Engine desconhecido: {engine}")


def _extract_ultralytics_boxes(result: Any) -> list[tuple[float, float, float, float, float]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    return [
        (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
        for b, c in zip(xyxy, confs)
    ]


def generate_preannotations(
    *,
    image_paths: list[Path],
    output_dir: Path,
    predict_fn: Any,
    engine_name: str,
    weights_ref: str,
    weights_sha256: str | None = None,
    secondary_predict_fn: Any | None = None,
    prompts: list[str] | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
    mot_tracking: bool = False,
) -> dict[str, Any]:
    """Run local pre-annotation over images and emit CVAT YOLO 1.1 zip, provenance, and uncertainty ranking."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_imgs = sorted(image_paths)
    if not sorted_imgs:
        raise ValueError("Nenhuma imagem encontrada para pré-anotação.")

    tracker_session = None
    if mot_tracking:
        backend_root = REPO_ROOT / "backend"
        if str(backend_root) not in sys.path:
            sys.path.insert(0, str(backend_root))
        from app.modules.vision.schemas import DetectionBox
        from app.modules.vision.tracker import ByteTrackSession

        tracker_session = ByteTrackSession(high_thresh=conf, sample_fps=5.0)

    yolo_zip_path = output_dir / "cvat_yolo11_preannotations.zip"
    mot_zip_path = output_dir / "cvat_mot11_preannotations.zip"
    prov_path = output_dir / "preannotation_provenance.json"
    ranking_path = output_dir / "uncertainty_ranking.json"

    uncertainty_items: list[dict[str, Any]] = []
    mot_rows: list[str] = []
    total_boxes = 0
    first_shape: tuple[int, int] | None = None

    with zipfile.ZipFile(yolo_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obj.names", f"{CLASS_NAME}\n")
        zf.writestr(
            "obj.data",
            "classes = 1\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n",
        )
        train_lines: list[str] = []

        for frame_seq, img_path in enumerate(sorted_imgs, start=1):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise ValueError(f"Imagem ilegível: {img_path}")
            ih, iw = bgr.shape[:2]
            if first_shape is None:
                first_shape = (iw, ih)

            boxes_main = predict_fn(bgr, conf, iou)
            boxes_sec = secondary_predict_fn(bgr, conf, iou) if secondary_predict_fn is not None else None
            boxes_final = nms_merge_xyxy(boxes_main, boxes_sec, iou_thresh=0.5) if boxes_sec is not None else boxes_main

            yolo_lines: list[str] = []
            for x1, y1, x2, y2, _ in boxes_final:
                cx = max(0.0, min(1.0, ((x1 + x2) / 2.0) / iw))
                cy = max(0.0, min(1.0, ((y1 + y2) / 2.0) / ih))
                bw = max(0.0, min(1.0, (x2 - x1) / iw))
                bh = max(0.0, min(1.0, (y2 - y1) / ih))
                if bw > 0.0 and bh > 0.0:
                    yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            total_boxes += len(yolo_lines)
            rel_img = f"data/obj_train_data/{img_path.name}"
            train_lines.append(rel_img)
            zf.writestr(f"obj_train_data/{img_path.stem}.txt", "\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

            u_score, reasons = _compute_uncertainty_score(boxes_main, boxes_sec)
            uncertainty_items.append(
                {
                    "filename": img_path.name,
                    "detectionCount": len(yolo_lines),
                    "meanConfidence": round(float(sum(b[4] for b in boxes_final) / len(boxes_final)), 4)
                    if boxes_final
                    else 0.0,
                    "uncertaintyScore": u_score,
                    "reasons": reasons,
                }
            )

            if tracker_session is not None:
                from app.modules.vision.schemas import DetectionBox

                det_dtos = [
                    DetectionBox(
                        class_id=0,
                        class_name=CLASS_NAME,
                        confidence=float(c),
                        track_id=None,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        width_px=max(0.0, x2 - x1),
                        height_px=max(0.0, y2 - y1),
                        norm_x=max(0.0, x1 / iw),
                        norm_y=max(0.0, y1 / ih),
                        norm_width=max(0.0, (x2 - x1) / iw),
                        norm_height=max(0.0, (y2 - y1) / ih),
                    )
                    for x1, y1, x2, y2, c in boxes_final
                ]
                tracked = tracker_session.update(det_dtos)
                for tb in tracked:
                    if tb.track_id is not None:
                        mot_rows.append(
                            f"{frame_seq},{tb.track_id},{tb.x1:.2f},{tb.y1:.2f},{tb.width_px:.2f},{tb.height_px:.2f},1,1,1.0"
                        )

        zf.writestr("train.txt", "\n".join(train_lines) + "\n")

    if mot_tracking and first_shape is not None:
        iw, ih = first_shape
        seqinfo = "\n".join(
            [
                "[Sequence]",
                "name=preannotation_seq",
                "imDir=img1",
                "frameRate=5",
                f"seqLength={len(sorted_imgs)}",
                f"imWidth={iw}",
                f"imHeight={ih}",
                "imExt=.jpg",
                "",
            ]
        )
        with zipfile.ZipFile(mot_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as mzf:
            mzf.writestr("gt/gt.txt", "\n".join(mot_rows) + ("\n" if mot_rows else ""))
            mzf.writestr("seqinfo.ini", seqinfo)

    uncertainty_items.sort(key=lambda item: item["uncertaintyScore"], reverse=True)
    ranking_path.write_text(
        json.dumps({"items": uncertainty_items}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    provenance = {
        "schemaVersion": "1.0",
        "engine": engine_name,
        "weights": weights_ref,
        "weightsSha256": weights_sha256,
        "prompts": prompts or [],
        "confThreshold": float(conf),
        "iouThreshold": float(iou),
        "device": device,
        "frameCount": len(sorted_imgs),
        "totalPreannotatedBoxes": total_boxes,
        "classNames": {"0": CLASS_NAME},
        "humanReviewed": False,
        "requiresFullHumanReview": True,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }
    prov_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "yoloZip": str(yolo_zip_path),
        "motZip": str(mot_zip_path) if mot_tracking else None,
        "provenance": str(prov_path),
        "uncertaintyRanking": str(ranking_path),
        "frameCount": len(sorted_imgs),
        "totalPreannotatedBoxes": total_boxes,
    }


def _extract_frames_from_video(video_path: Path, dst_dir: Path, sample_fps: float) -> list[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    interval_ms = 1000.0 / max(0.1, float(sample_fps))
    next_ms: float | None = None
    paths: list[Path] = []
    with av.open(str(video_path)) as container:
        in_stream = container.streams.video[0]
        fps_est = float(in_stream.average_rate or in_stream.guessed_rate or 25.0)
        for idx, frame in enumerate(container.decode(video=0)):
            if frame.pts is not None and frame.time_base is not None:
                ts_ms = int(round(float(frame.pts * frame.time_base) * 1000.0))
            else:
                ts_ms = int(round((idx * 1000.0) / max(1.0, fps_est)))
            if next_ms is None:
                next_ms = float(ts_ms)
            if (ts_ms + 1.0) < next_ms:
                continue
            while next_ms <= (ts_ms + 1.0):
                next_ms += interval_ms
            bgr = frame.to_ndarray(format="bgr24")
            out_img = dst_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_img), bgr)
            paths.append(out_img)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Local CVAT pre-annotation pipeline (100% HITL mandatory).")
    parser.add_argument("--engine", required=True, choices=["exp04c", "yoloe", "yoloworld", "hybrid"])
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--teacher-weights", default=None, help="Open-vocab weights when --engine=hybrid (default: yoloe-26s-seg.pt)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--visual-ref-image", type=Path, default=None)
    parser.add_argument("--visual-ref-label", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mot-tracking", action="store_true")
    args = parser.parse_args()

    if not args.images_dir and not args.video:
        raise ValueError("Informe --images-dir ou --video.")

    secondary_fn = None
    if args.engine == "hybrid":
        primary_fn, w_ref, w_sha = build_predictor(engine="exp04c", device=args.device, weights=args.weights)
        secondary_fn, _, _ = build_predictor(
            engine="yoloe",
            device=args.device,
            weights=args.teacher_weights,
            prompts=args.prompts,
            visual_ref_image=args.visual_ref_image,
            visual_ref_label=args.visual_ref_label,
        )
    else:
        primary_fn, w_ref, w_sha = build_predictor(
            engine=args.engine,
            device=args.device,
            weights=args.weights,
            prompts=args.prompts,
            visual_ref_image=args.visual_ref_image,
            visual_ref_label=args.visual_ref_label,
        )

    if args.images_dir:
        imgs = sorted(
            p for p in args.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        res = generate_preannotations(
            image_paths=imgs,
            output_dir=args.output_dir,
            predict_fn=primary_fn,
            engine_name=args.engine,
            weights_ref=w_ref,
            weights_sha256=w_sha,
            secondary_predict_fn=secondary_fn,
            prompts=args.prompts,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            mot_tracking=args.mot_tracking,
        )
    else:
        with tempfile.TemporaryDirectory() as tmp:
            imgs = _extract_frames_from_video(args.video, Path(tmp), args.sample_fps)
            res = generate_preannotations(
                image_paths=imgs,
                output_dir=args.output_dir,
                predict_fn=primary_fn,
                engine_name=args.engine,
                weights_ref=w_ref,
                weights_sha256=w_sha,
                secondary_predict_fn=secondary_fn,
                prompts=args.prompts,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                mot_tracking=args.mot_tracking,
            )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
