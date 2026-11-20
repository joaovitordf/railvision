"""Discover, license-filter, download (gitignored), measure, and generate triage grids for public freight videos."""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import requests
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_policies import is_license_acceptable, validate_video_manifest_entry  # noqa: E402

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "RailVisionAudit/0.1 (https://github.com/joaovitordf/railvision; study-audit)"}

SEARCH_QUERIES = [
    "freight train above filetype:video",
    "Güterzug von oben filetype:video",
    "freight train bridge filetype:video",
    "coal train filetype:video",
    "iron ore train filetype:video",
    "freight train filetype:video",
]

MAX_FILE_BYTES = 85 * 1024 * 1024
TARGET_VIDEO_COUNT = 7


def strip_html(raw: str | None) -> str:
    if not raw:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def slugify(title: str, index: int) -> str:
    clean = title.removeprefix("File:")
    stem = clean.rsplit(".", 1)[0]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()[:40]
    return f"vid_{index:02d}_{slug}"


def priority_score(title: str, desc: str, width: int, height: int, size: int) -> tuple[int, int]:
    text = f"{title} {desc}".lower()
    score = 0
    for kw, pts in (
        ("above", 50),
        ("von oben", 50),
        ("overpass", 35),
        ("bridge", 30),
        ("coal", 30),
        ("ore", 30),
        ("gondola", 30),
        ("hopper", 30),
        ("unbeladen", 25),
        ("freight", 15),
        ("güterzug", 15),
        ("cargo", 15),
    ):
        if kw in text:
            score += pts
    for neg in ("protest", "mural", "polarbear", "partition", "begrafenis", "museum", "ride", "cab view", "hopping"):
        if neg in text:
            score -= 60
    if width >= 1280 and height >= 720:
        score += 15
    elif width >= 640:
        score += 5
    return (score, -size)


def get_with_backoff(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    stream: bool = False,
) -> requests.Response:
    delay = 3.0
    for attempt in range(5):
        resp = session.get(url, params=params, stream=stream, timeout=60)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        retry_after = resp.headers.get("Retry-After")
        wait_s = float(retry_after) if retry_after and retry_after.isdigit() else delay
        print(f"[VIDEO] HTTP 429 rate-limit; aguardando {wait_s:.1f}s (tentativa {attempt + 1}/5)...")
        time.sleep(wait_s)
        delay *= 2
    resp.raise_for_status()
    return resp


def query_wikimedia_candidates() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    session = requests.Session()
    session.headers.update(HEADERS)

    pages_by_title: dict[str, dict[str, Any]] = {}
    for q in SEARCH_QUERIES:
        resp = get_with_backoff(
            session,
            COMMONS_API,
            params={
                "action": "query",
                "generator": "search",
                "gsrnamespace": 6,
                "gsrlimit": 12,
                "gsrsearch": q,
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "format": "json",
            },
        )
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "")
            if title and title not in pages_by_title:
                pages_by_title[title] = page
        time.sleep(1.2)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for title, page in pages_by_title.items():
        ii_list = page.get("imageinfo") or []
        if not ii_list:
            continue
        ii = ii_list[0]
        mime = ii.get("mime", "")
        size = int(ii.get("size") or 0)
        width = int(ii.get("width") or 0)
        height = int(ii.get("height") or 0)
        meta = ii.get("extmetadata") or {}
        license_short = strip_html((meta.get("LicenseShortName") or {}).get("value"))
        license_url = strip_html((meta.get("LicenseUrl") or {}).get("value"))
        artist = strip_html((meta.get("Artist") or {}).get("value")) or "Unknown on Wikimedia Commons"
        description = strip_html((meta.get("ImageDescription") or {}).get("value"))
        ok_license, license_reason = is_license_acceptable(license_short, license_url)

        record = {
            "title": title,
            "source_url": ii.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
            "download_url": ii.get("url"),
            "mime": mime,
            "size_bytes": size,
            "width": width,
            "height": height,
            "author_or_owner": artist,
            "license": license_short or "unknown",
            "license_url": license_url or "",
            "license_check": license_reason,
            "description": description[:300],
        }

        if not mime.startswith("video/"):
            record["rejection_reason"] = f"not_video_mime:{mime}"
            rejected.append(record)
            continue
        if not ok_license:
            record["rejection_reason"] = license_reason
            rejected.append(record)
            continue
        if size <= 0 or size > MAX_FILE_BYTES:
            record["rejection_reason"] = f"size_out_of_budget:{size}"
            rejected.append(record)
            continue

        record["score"] = priority_score(title, description, width, height, size)
        accepted.append(record)

    accepted.sort(key=lambda r: r["score"], reverse=True)
    return accepted, rejected


def download_video(session: requests.Session, url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        print(f"[VIDEO] Baixando {dest.name} ...")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with get_with_backoff(session, url, stream=True) as resp:
            with tmp.open("wb") as out:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)
        tmp.replace(dest)
        time.sleep(1.5)
    else:
        print(f"[VIDEO] Arquivo ja existe localmente: {dest.name}")

    sha = hashlib.sha256()
    with dest.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def inspect_video_and_build_grid(video_path: Path, grid_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV nao conseguiu abrir: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
    duration_s = round(frame_count / fps, 2) if fps > 0 and frame_count > 0 else 0.0

    indices = (
        [int(i * (frame_count - 1) / 5) for i in range(6)]
        if frame_count >= 6
        else list(range(max(1, frame_count)))
    )

    cell_w, cell_h = 480, 270
    cols, rows = 3, 2
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (18, 22, 28))
    font = ImageFont.load_default()

    decoded_frames = 0
    for slot, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        decoded_frames += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(pil_img)
        ts_s = round(idx / fps, 1) if fps > 0 else 0.0
        tag = f"frame {idx} | t={ts_s}s"
        draw.rectangle([6, 6, 150, 24], fill=(0, 0, 0))
        draw.text((10, 10), tag, fill="#ffffff", font=font)
        r, c = divmod(slot, cols)
        sheet.paste(pil_img, (c * cell_w, r * cell_h))

    cap.release()
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(grid_path, quality=88)

    return {
        "duration_s": duration_s,
        "resolution": f"{width}x{height}",
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "fourcc": fourcc or None,
        "decoded_sample_frames": decoded_frames,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw" / "public-videos"
    grid_dir = root / "data" / "processed" / "triage-frames"
    manifest_path = root / "specs" / "data-manifests" / "public-videos-manifest.json"
    candidates_log = root / "artifacts" / "runs" / "audit" / "public_videos_candidates.json"

    accepted, rejected = query_wikimedia_candidates()
    candidates_log.parent.mkdir(parents=True, exist_ok=True)
    candidates_log.write_text(
        json.dumps({"accepted_candidates": accepted, "rejected_candidates": rejected}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    selected = [c for c in accepted if c["score"][0] > -20][:TARGET_VIDEO_COUNT]
    manifest_entries: list[dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    dl_session = requests.Session()
    dl_session.headers.update(HEADERS)

    for idx, cand in enumerate(selected, start=1):
        ext = "." + cand["title"].rsplit(".", 1)[-1].lower()
        vid_id = slugify(cand["title"], idx)
        dest = raw_dir / f"{vid_id}{ext}"
        grid_path = grid_dir / f"{vid_id}_grid.jpg"

        sha256_hex = download_video(dl_session, cand["download_url"], dest)
        metrics = inspect_video_and_build_grid(dest, grid_path)

        entry = {
            "id": vid_id,
            "source_url": cand["source_url"],
            "download_url": cand["download_url"],
            "retrieved_at": now_iso,
            "author_or_owner": cand["author_or_owner"],
            "license": cand["license"],
            "license_url": cand["license_url"],
            "permitted_uses": f"local_storage_analysis_and_study_training ({cand['license_check']})",
            "sha256": sha256_hex,
            "file_size_bytes": dest.stat().st_size,
            "duration_s": metrics["duration_s"],
            "resolution": metrics["resolution"],
            "fps": metrics["fps"],
            "codec": metrics["fourcc"],
            "angle_category": "pending_visual_inspection",
            "cargo_visible": None,
            "wagon_type": "pending_visual_inspection",
            "triage_grid": str(grid_path.relative_to(root)).replace("\\", "/"),
            "notes": f"Wikimedia title={cand['title']}; desc={cand['description']}",
        }
        missing = validate_video_manifest_entry(entry)
        if missing:
            raise RuntimeError(f"Manifest entry missing required fields: {missing}")
        manifest_entries.append(entry)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": now_iso,
                "source_platform": "Wikimedia Commons",
                "total_candidates_accepted_by_license": len(accepted),
                "total_candidates_rejected": len(rejected),
                "videos": manifest_entries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[VIDEO] Manifesto salvo em {manifest_path} com {len(manifest_entries)} videos.")


if __name__ == "__main__":
    main()
