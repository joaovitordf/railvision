"""Parallel discovery of additional open-licensed bulk cargo / open-top wagon videos for future Phase 5."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_policies import is_license_acceptable, validate_video_manifest_entry  # noqa: E402
from audit_public_videos import (  # noqa: E402
    COMMONS_API,
    HEADERS,
    download_video,
    get_with_backoff,
    inspect_video_and_build_grid,
    slugify,
    strip_html,
)

CARGO_QUERIES = [
    "open wagon coal OR ore OR ballast filetype:video",
    "Güterwagen Kohle OR Erz OR Schotter filetype:video",
    "rail hopper OR gondola OR ore cars filetype:video",
    "mineral train OR iron ore train filetype:video",
]

MAX_NEW_VIDEOS = 4
MAX_BYTES = 140 * 1024 * 1024


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "specs/data-manifests/public-videos-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_urls = {v["source_url"] for v in manifest["videos"]}

    session = requests.Session()
    session.headers.update(HEADERS)

    candidates: dict[str, dict[str, Any]] = {}

    # 1. Paginate Category:Videos_of_freight_trains up to 80 items + run cargo queries
    cm_resp = get_with_backoff(
        session,
        COMMONS_API,
        params={
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": "Category:Videos_of_freight_trains",
            "gcmtype": "file",
            "gcmlimit": 50,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "format": "json",
        },
    )
    for page in cm_resp.json().get("query", {}).get("pages", {}).values():
        t = page.get("title", "")
        if t:
            candidates[t] = page
    time.sleep(1.5)

    for q in CARGO_QUERIES:
        r = get_with_backoff(
            session,
            COMMONS_API,
            params={
                "action": "query",
                "generator": "search",
                "gsrnamespace": 6,
                "gsrlimit": 15,
                "gsrsearch": q,
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "format": "json",
            },
        )
        for page in r.json().get("query", {}).get("pages", {}).values():
            t = page.get("title", "")
            if t and t not in candidates:
                candidates[t] = page
        time.sleep(1.5)

    scored: list[dict[str, Any]] = []
    for title, page in candidates.items():
        ii = (page.get("imageinfo") or [{}])[0]
        if not ii:
            continue
        src = ii.get("descriptionurl") or ""
        if src in existing_urls:
            continue
        mime = ii.get("mime", "")
        size = int(ii.get("size") or 0)
        if not mime.startswith("video/") or size <= 0 or size > MAX_BYTES:
            continue
        meta = ii.get("extmetadata") or {}
        lic = strip_html((meta.get("LicenseShortName") or {}).get("value"))
        lurl = strip_html((meta.get("LicenseUrl") or {}).get("value"))
        ok, lic_reason = is_license_acceptable(lic, lurl)
        if not ok:
            continue
        desc = strip_html((meta.get("ImageDescription") or {}).get("value"))
        blob = f"{title} {desc}".lower()

        pts = 0
        for kw, w in (
            ("ballast", 45), ("trémie", 45), ("hopper", 45), ("gondola", 45),
            ("coal", 40), ("ore", 40), ("limestone", 40), ("sand", 40),
            ("schotter", 40), ("kohle", 40), ("erz", 40), ("mineral", 35),
            ("open", 25), ("bulk", 25), ("above", 35), ("bridge", 20), ("freight", 10),
        ):
            if kw in blob:
                pts += w
        for neg in ("protest", "mural", "museum", "container", "tank", "passenger", "steam", "dampflok"):
            if neg in blob:
                pts -= 40
        if pts >= 25:
            scored.append(
                {
                    "title": title,
                    "source_url": src,
                    "download_url": ii.get("url"),
                    "size": size,
                    "license": lic,
                    "license_url": lurl,
                    "license_check": lic_reason,
                    "author": strip_html((meta.get("Artist") or {}).get("value")) or "Unknown",
                    "description": desc[:300],
                    "score": (pts, -size),
                }
            )

    scored.sort(key=lambda x: x["score"], reverse=True)
    print(f"[CARGO-SEARCH] Found {len(scored)} new high-relevance open-licensed bulk candidates.")

    selected = scored[:MAX_NEW_VIDEOS]
    next_idx = len(manifest["videos"]) + 1
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    raw_dir = root / "data/raw/public-videos"
    grid_dir = root / "data/processed/triage-frames"

    added_ids: list[str] = []
    for cand in selected:
        ext = "." + cand["title"].rsplit(".", 1)[-1].lower()
        vid_id = slugify(cand["title"], next_idx)
        next_idx += 1
        dest = raw_dir / f"{vid_id}{ext}"
        grid = grid_dir / f"{vid_id}_grid.jpg"
        sha = download_video(session, cand["download_url"], dest)
        metrics = inspect_video_and_build_grid(dest, grid)
        entry = {
            "id": vid_id,
            "source_url": cand["source_url"],
            "download_url": cand["download_url"],
            "retrieved_at": now_iso,
            "author_or_owner": cand["author"],
            "license": cand["license"],
            "license_url": cand["license_url"],
            "permitted_uses": f"local_storage_analysis_and_study_training ({cand['license_check']})",
            "sha256": sha,
            "file_size_bytes": dest.stat().st_size,
            "duration_s": metrics["duration_s"],
            "resolution": metrics["resolution"],
            "fps": metrics["fps"],
            "codec": metrics["fourcc"],
            "angle_category": "pending_visual_inspection",
            "cargo_visible": None,
            "wagon_type": "pending_visual_inspection",
            "role_in_phase3": "parallel_cargo_discovery_candidate",
            "triage_grid": str(grid.relative_to(root)).replace("\\", "/"),
            "notes": f"Wikimedia title={cand['title']}; desc={cand['description']}",
        }
        assert validate_video_manifest_entry(entry) == []
        manifest["videos"].append(entry)
        added_ids.append(vid_id)

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[CARGO-SEARCH] Added {len(added_ids)} candidates: {added_ids}")


if __name__ == "__main__":
    main()
