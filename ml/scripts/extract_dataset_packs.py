"""Manifest-driven frame pack extraction with PyAV sampling, dHash deduplication, and split hygiene."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESERVED_BENCHMARK_IDS = {"vid_01", "vid_09", "vid_16"}


def compute_dhash(bgr: np.ndarray, hash_size: int = 8) -> int:
    """Compute 64-bit difference hash (dHash) for near-duplicate frame filtering."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for val in diff.flatten():
        bits = (bits << 1) | int(bool(val))
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Return Hamming distance between two 64-bit integer hashes."""
    return (a ^ b).bit_count()


def extract_packs_from_manifests(
    *,
    manifest_paths: list[Path],
    output_dir: Path,
    allowed_splits: set[str],
    sample_fps: float = 2.0,
    max_frames_per_video: int = 60,
    dedup_hamming_threshold: int = 6,
) -> dict[str, Any]:
    """Extract clean frames from registered videos matching allowed splits without benchmark leakage."""
    if not manifest_paths:
        raise ValueError("Informe ao menos um manifest JSON.")

    videos: list[dict[str, Any]] = []
    for mp in manifest_paths:
        data = json.loads(mp.read_text(encoding="utf-8"))
        for rec in data.get("videos") or []:
            videos.append(rec)

    output_dir.mkdir(parents=True, exist_ok=True)
    interval_ms = 1000.0 / max(0.1, float(sample_fps))
    extracted_records: list[dict[str, Any]] = []
    skipped_duplicates = 0

    for rec in videos:
        vid_id = str(rec.get("video_id") or rec.get("short_id") or "")
        split = str(rec.get("target_split") or rec.get("split") or "")
        if not split or split not in allowed_splits:
            continue

        if split in {"train", "val"}:
            lowered = vid_id.lower()
            if any(lowered.startswith(res) for res in RESERVED_BENCHMARK_IDS):
                raise ValueError(
                    f"Extração bloqueada: vídeo reservado '{vid_id}' nunca pode ser extraído para split '{split}'."
                )

        raw_path = rec.get("local_path") or rec.get("path")
        if not raw_path:
            continue
        vid_path = Path(str(raw_path))
        if not vid_path.is_absolute():
            vid_path = (REPO_ROOT / vid_path).resolve()
        if not vid_path.is_file():
            raise FileNotFoundError(f"Arquivo de vídeo ausente para '{vid_id}': {vid_path}")

        campaign_id = str(rec.get("campaign_id") or "default_campaign")
        last_hash: int | None = None
        next_sample_ms: float | None = None
        saved_count = 0

        with av.open(str(vid_path)) as container:
            in_stream = container.streams.video[0]
            fps_est = float(in_stream.average_rate or in_stream.guessed_rate or 25.0)
            for raw_idx, frame in enumerate(container.decode(video=0)):
                if frame.pts is not None and frame.time_base is not None:
                    ts_ms = int(round(float(frame.pts * frame.time_base) * 1000.0))
                elif frame.pts is not None and in_stream.time_base is not None:
                    ts_ms = int(round(float(frame.pts * in_stream.time_base) * 1000.0))
                else:
                    ts_ms = int(round((raw_idx * 1000.0) / max(1.0, fps_est)))

                if next_sample_ms is None:
                    next_sample_ms = float(ts_ms)
                if (ts_ms + 1.0) < next_sample_ms:
                    continue
                while next_sample_ms <= (ts_ms + 1.0):
                    next_sample_ms += interval_ms

                bgr = frame.to_ndarray(format="bgr24")
                cur_hash = compute_dhash(bgr)
                if dedup_hamming_threshold > 0 and last_hash is not None:
                    if hamming_distance(cur_hash, last_hash) <= dedup_hamming_threshold:
                        skipped_duplicates += 1
                        continue
                last_hash = cur_hash

                filename = f"{split}__{campaign_id}__{vid_id}__f{raw_idx:06d}.jpg"
                out_file = output_dir / filename
                ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if not ok:
                    raise RuntimeError(f"Falha ao salvar frame JPEG: {out_file}")
                out_file.write_bytes(enc.tobytes())

                extracted_records.append(
                    {
                        "filename": filename,
                        "video_id": vid_id,
                        "campaign_id": campaign_id,
                        "split": split,
                        "frame_index": raw_idx,
                        "timestamp_ms": ts_ms,
                        "dhash_hex": f"{cur_hash:016x}",
                    }
                )
                saved_count += 1
                if saved_count >= max_frames_per_video:
                    break

    pack_manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "allowed_splits": sorted(allowed_splits),
        "sample_fps": float(sample_fps),
        "dedup_hamming_threshold": int(dedup_hamming_threshold),
        "skipped_near_duplicates": skipped_duplicates,
        "frame_count": len(extracted_records),
        "frames": extracted_records,
    }
    manifest_out = output_dir / "pack_manifest.json"
    manifest_out.write_text(json.dumps(pack_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return pack_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CVAT-ready frame packs from registered video manifests.")
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--splits", default="train,val", help="Comma-separated allowed splits (e.g. train,val)")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-frames-per-video", type=int, default=60)
    parser.add_argument("--dedup-hamming-threshold", type=int, default=6)
    args = parser.parse_args()

    allowed = {s.strip() for s in args.splits.split(",") if s.strip()}
    result = extract_packs_from_manifests(
        manifest_paths=args.manifest,
        output_dir=args.output_dir,
        allowed_splits=allowed,
        sample_fps=args.sample_fps,
        max_frames_per_video=args.max_frames_per_video,
        dedup_hamming_threshold=args.dedup_hamming_threshold,
    )
    print(
        json.dumps(
            {
                "frame_count": result["frame_count"],
                "skipped_near_duplicates": result["skipped_near_duplicates"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
