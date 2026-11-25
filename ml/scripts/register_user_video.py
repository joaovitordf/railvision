"""Register user-owned or partner-curated videos explicitly without automatic dataset ingestion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import av

REPO_ROOT = Path(__file__).resolve().parents[2]
RESERVED_BENCHMARK_IDS = {"vid_01", "vid_09", "vid_16"}
VALID_ORIGINS = {"user_owned", "partner_provided", "public_curated"}
VALID_SPLITS = {"train", "val", "test", "test_negative_only"}
VALID_CONTENT = {"positive_open_top", "hard_negative", "mixed"}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a media file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video_pyav(path: Path) -> dict[str, Any]:
    """Probe resolution, FPS, duration, and codec with PyAV."""
    with av.open(str(path)) as container:
        video_streams = [s for s in container.streams if s.type == "video"]
        if not video_streams:
            raise ValueError(f"Arquivo sem stream de vídeo: {path}")
        stream = video_streams[0]
        fps = float(stream.average_rate or stream.guessed_rate or 0.0)
        if container.duration is not None:
            duration_s = float(container.duration) / float(av.time_base)
        elif stream.duration is not None and stream.time_base is not None:
            duration_s = float(stream.duration * stream.time_base)
        else:
            duration_s = 0.0
        return {
            "width": int(stream.codec_context.width or 0),
            "height": int(stream.codec_context.height or 0),
            "fps": round(fps, 3),
            "duration_s": round(duration_s, 3),
            "codec": str(stream.codec_context.name or "unknown"),
            "size_bytes": int(path.stat().st_size),
        }


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def register_video(
    *,
    video_path: Path,
    video_id: str,
    campaign_id: str,
    origin: str,
    provenance: str,
    target_split: str,
    expected_content: str,
    manifest_path: Path,
    public_manifest_path: Path | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Validate provenance and split/campaign isolation rules, then append entry to user-videos manifest."""
    resolved_video = video_path.resolve()
    if not resolved_video.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {resolved_video}")

    clean_id = video_id.strip()
    clean_campaign = campaign_id.strip()
    if not clean_id or not clean_campaign:
        raise ValueError("video_id e campaign_id são obrigatórios.")
    if origin not in VALID_ORIGINS:
        raise ValueError(f"origin inválido '{origin}'. Use: {sorted(VALID_ORIGINS)}")
    if target_split not in VALID_SPLITS:
        raise ValueError(f"target_split inválido '{target_split}'. Use: {sorted(VALID_SPLITS)}")
    if expected_content not in VALID_CONTENT:
        raise ValueError(f"expected_content inválido '{expected_content}'. Use: {sorted(VALID_CONTENT)}")
    if not provenance.strip():
        raise ValueError("provenance não pode estar vazia.")

    lowered_id = clean_id.lower()
    lowered_name = resolved_video.name.lower()
    for reserved in RESERVED_BENCHMARK_IDS:
        if (lowered_id.startswith(reserved) or reserved in lowered_name) and target_split in {"train", "val"}:
            raise ValueError(
                f"Isolamento violado: '{reserved}' é benchmark congelado e nunca pode ser registrado em '{target_split}'."
            )
    if "camp_vfli_cosne" in clean_campaign.lower() and target_split in {"train", "val"}:
        raise ValueError("Isolamento violado: campanha congelada 'camp_vfli_cosne' proibida em train/val.")

    user_manifest: dict[str, Any]
    if manifest_path.is_file():
        user_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        user_manifest = {
            "schema_version": "1.0",
            "policy_notes": [
                "Explicit registration required for user-owned or curated videos.",
                "Exploratory Lab/Inspection runs are NEVER auto-added here.",
                "100% human review in CVAT is mandatory before dataset ingestion.",
            ],
            "videos": [],
        }

    existing_videos = list(user_manifest.get("videos") or [])
    existing_public: list[dict[str, Any]] = []
    if public_manifest_path and public_manifest_path.is_file():
        pub_json = json.loads(public_manifest_path.read_text(encoding="utf-8"))
        existing_public = list(pub_json.get("videos") or [])

    file_sha = sha256_file(resolved_video)

    for rec in existing_videos + existing_public:
        rec_id = str(rec.get("video_id") or rec.get("id") or "")
        rec_short = str(rec.get("short_id") or "")
        if clean_id in {rec_id, rec_short}:
            raise ValueError(f"video_id duplicado no manifest: '{clean_id}'")
        rec_sha = str(rec.get("sha256") or "")
        if rec_sha and rec_sha == file_sha:
            raise ValueError(f"Vídeo duplicado por SHA-256 ({file_sha[:12]} já registrado como '{rec_id or rec_short}').")
        rec_campaign = str(rec.get("campaign_id") or "")
        rec_split = str(rec.get("target_split") or rec.get("split") or "")
        if rec_campaign and rec_campaign == clean_campaign and rec_split and rec_split != target_split:
            raise ValueError(
                f"Vazamento de campanha bloqueado: campaign_id='{clean_campaign}' já existe no split '{rec_split}' "
                f"e não pode entrar em '{target_split}'."
            )

    media_info = probe_video_pyav(resolved_video)
    entry = {
        "video_id": clean_id,
        "campaign_id": clean_campaign,
        "origin": origin,
        "provenance": provenance.strip(),
        "target_split": target_split,
        "expected_content": expected_content,
        "local_path": _display_path(resolved_video),
        "filename": resolved_video.name,
        "sha256": file_sha,
        "media": media_info,
        "auto_train_eligible": False,
        "human_reviewed": False,
        "notes": notes.strip(),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }

    existing_videos.append(entry)
    user_manifest["videos"] = existing_videos
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(user_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(manifest_path)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Register user-owned or partner video explicitly into dataset manifest.")
    parser.add_argument("--video-path", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--origin", required=True, choices=sorted(VALID_ORIGINS))
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--target-split", required=True, choices=sorted(VALID_SPLITS))
    parser.add_argument("--expected-content", required=True, choices=sorted(VALID_CONTENT))
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=REPO_ROOT / "specs/data-manifests/user-videos-manifest.json",
    )
    parser.add_argument(
        "--public-manifest-path",
        type=Path,
        default=REPO_ROOT / "specs/data-manifests/public-videos-manifest.json",
    )
    args = parser.parse_args()

    entry = register_video(
        video_path=args.video_path,
        video_id=args.video_id,
        campaign_id=args.campaign_id,
        origin=args.origin,
        provenance=args.provenance,
        target_split=args.target_split,
        expected_content=args.expected_content,
        manifest_path=args.manifest_path,
        public_manifest_path=args.public_manifest_path,
        notes=args.notes,
    )
    print(json.dumps(entry, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
