"""Curated open-license video and photo downloader + triage contact-sheet builder for WS2.

Fetches candidate videos and high-resolution images from Wikimedia Commons under open
licenses (CC0 / CC-BY / CC-BY-SA / Public Domain), deduplicates against existing
manifests by title/URL/SHA-256, validates video decoding via PyAV, and renders
contact sheets for mandatory human/visual triage before dataset registration.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import av
from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "ws2-public-intake"
PUBLIC_MANIFEST_PATH = REPO_ROOT / "specs" / "data-manifests" / "public-videos-manifest.json"
USER_MANIFEST_PATH = REPO_ROOT / "specs" / "data-manifests" / "user-videos-manifest.json"

USER_AGENT = "RailVisionDatasetCurator/1.0 (https://github.com/local/railvision; academic-ml)"
ALLOWED_LICENSE_PREFIXES = ("cc0", "cc-by", "cc by", "public domain", "pd-")
FORBIDDEN_LICENSE_TOKENS = ("nc", "nd", "noncommercial", "noderivatives", "fair use", "copyrighted")

EXCLUDED_TITLE_TOKENS = (
    "191", "192", "193", "194", "195", "196",
    "museum", "model", "miniature", "lego", "toy", "drawing", "diagram",
    "sketch", "painting", "engraving", "map", "poster", "protest",
    "mural", "crash", "accident", "war", "gun camera", "undergorund", "underground",
    "subterranean", "tram ", "streetcar", "bus ", "trailer", "bicycle",
)


def _strip_html(raw: str) -> str:
    clean = re.sub(r"<[^>]+>", " ", raw or "")
    return re.sub(r"\s+", " ", html.unescape(clean)).strip()


def _slugify(text: str, max_len: int = 44) -> str:
    stem = re.sub(r"^File:", "", text)
    stem = re.sub(r"\.[a-zA-Z0-9]+$", "", stem)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return (slug[:max_len].strip("_") or "media")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_open_license(license_short: str) -> bool:
    low = (license_short or "").strip().lower()
    if not low:
        return False
    if any(bad in low for bad in FORBIDDEN_LICENSE_TOKENS):
        return False
    return any(low.startswith(pref) or pref in low for pref in ALLOWED_LICENSE_PREFIXES)


def _has_excluded_terms(*texts: str) -> bool:
    joined = " ".join(texts).lower()
    return any(tok in joined for tok in EXCLUDED_TITLE_TOKENS)


def _request_with_backoff(url: str, *, timeout: int = 30, sleep_before: float = 1.0) -> bytes:
    time.sleep(sleep_before)
    delays = [2.0, 5.0, 12.0, 25.0]
    last_err: Exception | None = None
    for attempt in range(len(delays) + 1):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in (429, 500, 502, 503, 504) and attempt < len(delays):
                wait_s = delays[attempt]
                retry_after = exc.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait_s = max(wait_s, float(retry_after))
                print(f"  [backoff] HTTP {exc.code} -> aguardando {wait_s:.1f}s...")
                time.sleep(wait_s)
                continue
            raise
        except Exception as exc:
            last_err = exc
            if attempt < len(delays):
                time.sleep(delays[attempt])
                continue
            raise
    assert last_err is not None
    raise last_err


def _download_file(url: str, dest: Path, *, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    delays = [2.0, 6.0, 15.0]
    for attempt in range(len(delays) + 1):
        time.sleep(0.75)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as out:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                tmp.replace(dest)
                return
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < len(delays):
                wait_s = delays[attempt]
                print(f"  [download backoff] HTTP {exc.code} -> {wait_s:.1f}s")
                time.sleep(wait_s)
                continue
            raise


def _api_query(params: dict[str, Any]) -> dict[str, Any]:
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    raw = _request_with_backoff(url, timeout=30, sleep_before=1.15)
    return json.loads(raw.decode("utf-8"))


def _load_existing_keys() -> tuple[set[str], set[str]]:
    titles: set[str] = set()
    shas: set[str] = set()
    for path in (PUBLIC_MANIFEST_PATH, USER_MANIFEST_PATH):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for rec in data.get("videos", []):
            src = str(rec.get("source_url") or "")
            if "/wiki/" in src:
                titles.add(urllib.parse.unquote(src.split("/wiki/")[-1]).replace("_", " "))
            sha = str(rec.get("sha256") or "").strip().lower()
            if sha:
                shas.add(sha)
    return titles, shas


def _parse_pages(data: dict[str, Any], existing_titles: set[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for page in (data.get("query", {}).get("pages") or {}).values():
        title = str(page.get("title") or "")
        if not title or title in existing_titles:
            continue
        ii = (page.get("imageinfo") or [{}])[0]
        ext = ii.get("extmetadata") or {}
        lic = _strip_html((ext.get("LicenseShortName") or {}).get("value") or "")
        if not _is_open_license(lic):
            continue
        lic_url = str((ext.get("LicenseUrl") or {}).get("value") or "").strip()
        author = _strip_html((ext.get("Artist") or {}).get("value") or "Unknown")[:140]
        desc = _strip_html((ext.get("ImageDescription") or {}).get("value") or "")[:360]
        cats = _strip_html((ext.get("Categories") or {}).get("value") or "")[:320]
        url = str(ii.get("url") or "")
        if not url:
            continue
        results.append(
            {
                "title": title,
                "source_url": str(ii.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"),
                "download_url": url,
                "mediatype": str(ii.get("mediatype") or ""),
                "mime": str(ii.get("mime") or ""),
                "width": int(ii.get("width") or 0),
                "height": int(ii.get("height") or 0),
                "size_bytes": int(ii.get("size") or 0),
                "duration_s": round(float(ii.get("duration") or 0.0), 2),
                "license": lic,
                "license_url": lic_url,
                "author": author,
                "description": desc,
                "categories": cats,
            }
        )
    return results


def _fetch_category(category: str, existing_titles: set[str], limit: int = 50) -> list[dict[str, Any]]:
    data = _api_query(
        {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata|mediatype",
        }
    )
    return _parse_pages(data, existing_titles)


def _fetch_search(query: str, filetype: str, existing_titles: set[str], limit: int = 20) -> list[dict[str, Any]]:
    data = _api_query(
        {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:{filetype} {query}",
            "gsrnamespace": 6,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata|mediatype",
        }
    )
    return _parse_pages(data, existing_titles)


def _iter_decoded_video_frames(container: av.container.InputContainer):
    decoded_count = 0
    for packet in container.demux(video=0):
        if packet.size == 0 and packet.dts is not None:
            continue
        try:
            for frame in packet.decode():
                decoded_count += 1
                yield frame
        except (av.error.ArgumentError, av.error.InvalidDataError):
            if decoded_count > 0:
                continue
            raise


def _probe_and_build_video_grid(video_path: Path, grid_path: Path) -> dict[str, Any]:
    with av.open(str(video_path)) as container:
        vstreams = [s for s in container.streams if s.type == "video"]
        if not vstreams:
            raise ValueError(f"Sem stream de vídeo: {video_path}")
        vs = vstreams[0]
        fps = float(vs.average_rate or vs.guessed_rate or 0.0)
        duration_s = (
            float(container.duration) / float(av.time_base)
            if container.duration is not None
            else (float(vs.duration * vs.time_base) if vs.duration and vs.time_base else 0.0)
        )
        width = int(vs.codec_context.width or 0)
        height = int(vs.codec_context.height or 0)
        codec = str(vs.codec_context.name or "unknown")

        total_est = max(1, int(round(duration_s * (fps if fps > 0 else 25.0))))
        sample_targets = {int(round(i * (total_est - 1) / 8.0)) for i in range(9)}
        sampled: list[tuple[int, float, Image.Image]] = []
        last_img: Image.Image | None = None
        last_t = 0.0

        for idx, frame in enumerate(_iter_decoded_video_frames(container)):
            t_s = float(frame.time) if frame.time is not None else (idx / max(fps, 1.0))
            if idx in sample_targets or len(sampled) == 0:
                pil_img = frame.to_image().convert("RGB")
                sampled.append((idx, t_s, pil_img))
            if idx % 20 == 0:
                last_img = frame.to_image().convert("RGB")
                last_t = t_s
            if len(sampled) >= 9 and idx >= max(sample_targets):
                break

        while len(sampled) < 9 and last_img is not None:
            sampled.append((total_est, last_t, last_img))

    tile_w, tile_h = 420, 236
    canvas = Image.new("RGB", (tile_w * 3 + 16, tile_h * 3 + 48), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"{video_path.name}  ({width}x{height} · {fps:.1f}fps · {duration_s:.1f}s)", fill=(241, 245, 249))

    for pos, (fidx, t_s, img) in enumerate(sampled[:9]):
        row, col = divmod(pos, 3)
        thumb = ImageOps.contain(img, (tile_w, tile_h), method=Image.Resampling.BILINEAR)
        tile = Image.new("RGB", (tile_w, tile_h), (6, 10, 18))
        tile.paste(thumb, ((tile_w - thumb.width) // 2, (tile_h - thumb.height) // 2))
        tdraw = ImageDraw.Draw(tile)
        tdraw.rectangle([6, 6, 126, 26], fill=(11, 21, 38))
        tdraw.text((12, 10), f"#{fidx}  t={t_s:.1f}s", fill=(250, 204, 21))
        canvas.paste(tile, (col * tile_w + 8, row * tile_h + 38))

    grid_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(grid_path, quality=88)
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "duration_s": round(duration_s, 2),
        "codec": codec,
    }


def _build_photo_grid(items: list[dict[str, Any]], grid_path: Path, title: str) -> None:
    cols = 4
    rows = max(1, (len(items) + cols - 1) // cols)
    tile_w, tile_h = 360, 240
    canvas = Image.new("RGB", (tile_w * cols + 16, tile_h * rows + 44), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill=(241, 245, 249))

    for idx, item in enumerate(items):
        row, col = divmod(idx, cols)
        try:
            with Image.open(item["local_path"]) as im:
                im_rgb = ImageOps.exif_transpose(im).convert("RGB")
                thumb = ImageOps.contain(im_rgb, (tile_w - 8, tile_h - 32), method=Image.Resampling.BILINEAR)
        except Exception:
            thumb = Image.new("RGB", (tile_w - 8, tile_h - 32), (80, 20, 20))
        tile = Image.new("RGB", (tile_w - 4, tile_h - 4), (19, 34, 56))
        tile.paste(thumb, ((tile_w - 4 - thumb.width) // 2, 26))
        tdraw = ImageDraw.Draw(tile)
        label = f"{item['tag']} · {item['width']}x{item['height']} · {item['title'].replace('File:', '')[:34]}"
        tdraw.rectangle([0, 0, tile_w - 4, 24], fill=(11, 21, 38))
        tdraw.text((8, 6), label, fill=(248, 250, 252))
        canvas.paste(tile, (col * tile_w + 10, row * tile_h + 38))

    grid_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(grid_path, quality=88)


def _score_positive_video(item: dict[str, Any]) -> int:
    text = f"{item['title']} {item['description']} {item['categories']}".lower()
    if _has_excluded_terms(text):
        return -100
    score = 0
    for kw, pts in (
        ("coal", 8), ("kohle", 8), ("ore", 8), ("erz", 7), ("ballast", 8), ("schotter", 8),
        ("hopper", 7), ("gondola", 7), ("open wagon", 7), ("offener", 7), ("trémie", 7),
        ("bridge", 5), ("überführung", 5), ("brücke", 5), ("above", 6), ("von oben", 6),
        ("viaduct", 5), ("freight", 3), ("güterzug", 4), ("cargo", 3),
    ):
        if kw in text:
            score += pts
    if item["width"] >= 1920:
        score += 3
    elif item["width"] >= 1280:
        score += 2
    if 12 <= item["duration_s"] <= 130:
        score += 2
    return score


def _score_negative_video(item: dict[str, Any]) -> int:
    text = f"{item['title']} {item['description']} {item['categories']}".lower()
    if _has_excluded_terms(text):
        return -100
    if any(pos in text for pos in ("coal", "kohle", "ore", "erz", "ballast", "schotter", "hopper", "gondola", "mineral")):
        return -50
    score = 0
    for kw, pts in (
        ("container", 8), ("intermodal", 8), ("tank", 8), ("boxcar", 6),
        ("covered", 6), ("passenger", 5), ("locomotive", 4), ("flatcar", 5),
    ):
        if kw in text:
            score += pts
    if item["width"] >= 1280:
        score += 2
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch curated open-license train videos & photos for WS2.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-pos-videos", type=int, default=12)
    parser.add_argument("--max-neg-videos", type=int, default=5)
    parser.add_argument("--max-pos-photos", type=int, default=44)
    parser.add_argument("--max-neg-photos", type=int, default=16)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    staging_vid_dir = out_dir / "_staging_videos"
    staging_img_pos_dir = out_dir / "_staging_images" / "positive"
    staging_img_neg_dir = out_dir / "_staging_images" / "negative"
    grid_dir = out_dir / "triage-grids"

    existing_titles, existing_shas = _load_existing_keys()
    print(f"[init] Manifestos carregados: {len(existing_titles)} títulos e {len(existing_shas)} SHA-256 existentes ignorados.")

    # 1. Discover video candidates via searches + categories
    raw_videos: dict[str, dict[str, Any]] = {}
    vid_searches = [
        "coal train freight",
        "iron ore train",
        "ballast hopper train",
        "gondola freight train",
        "Güterzug Kohle OR Schotter OR Erz",
        "freight train bridge",
        "tank freight train",
        "container freight train",
    ]
    for q in vid_searches:
        try:
            for item in _fetch_search(q, "video", existing_titles, limit=18):
                raw_videos[item["title"]] = item
        except Exception as exc:
            print(f"[warn] Busca de vídeo '{q}' falhou: {exc}")

    for cat in ("Videos of freight trains", "Videos of rail transport"):
        try:
            for item in _fetch_category(cat, existing_titles, limit=50):
                if item["mediatype"] == "VIDEO":
                    raw_videos[item["title"]] = item
        except Exception as exc:
            print(f"[warn] Categoria de vídeo '{cat}' falhou: {exc}")

    valid_vids = [
        v for v in raw_videos.values()
        if v["mediatype"] == "VIDEO"
        and v["width"] >= 640
        and 8.0 <= v["duration_s"] <= 210.0
        and 100_000 <= v["size_bytes"] <= 115 * 1024 * 1024
        and not _has_excluded_terms(v["title"], v["description"])
    ]

    pos_ranked = sorted(valid_vids, key=_score_positive_video, reverse=True)
    neg_ranked = sorted(valid_vids, key=_score_negative_video, reverse=True)

    selected_vid_pool: list[tuple[str, dict[str, Any]]] = []
    seen_vid_titles: set[str] = set()
    author_counts_v: Counter[str] = Counter()

    for v in pos_ranked:
        if len([k for k, _ in selected_vid_pool if k == "pos"]) >= args.max_pos_videos:
            break
        if _score_positive_video(v) < 3:
            continue
        if v["title"] in seen_vid_titles or author_counts_v[v["author"]] >= 2:
            continue
        seen_vid_titles.add(v["title"])
        author_counts_v[v["author"]] += 1
        selected_vid_pool.append(("pos", v))

    for v in neg_ranked:
        if len([k for k, _ in selected_vid_pool if k == "neg"]) >= args.max_neg_videos:
            break
        if _score_negative_video(v) < 5:
            continue
        if v["title"] in seen_vid_titles or author_counts_v[v["author"]] >= 2:
            continue
        seen_vid_titles.add(v["title"])
        author_counts_v[v["author"]] += 1
        selected_vid_pool.append(("neg", v))

    print(f"[videos] Baixando {len(selected_vid_pool)} vídeos candidatos para triagem visual...")
    downloaded_videos: list[dict[str, Any]] = []
    for idx, (kind, v) in enumerate(selected_vid_pool, start=1):
        ext = Path(urllib.parse.urlparse(v["download_url"]).path).suffix.lower() or ".webm"
        if ext not in {".webm", ".ogv", ".mp4", ".mov", ".mkv", ".avi"}:
            ext = ".webm"
        tag = f"V{idx:02d}_{kind}"
        slug = _slugify(v["title"])
        local_path = staging_vid_dir / f"{tag}_{slug}{ext}"
        grid_path = grid_dir / f"{tag}_{slug}_grid.jpg"
        print(f"  [{idx}/{len(selected_vid_pool)}] {v['title']} ({round(v['size_bytes']/1048576, 1)} MB, {v['license']})")
        try:
            if not local_path.is_file():
                _download_file(v["download_url"], local_path)
            sha = _sha256_file(local_path)
            if sha in existing_shas:
                local_path.unlink(missing_ok=True)
                print("    -> ignorado (SHA duplicado no repositório)")
                continue
            existing_shas.add(sha)
            media_meta = _probe_and_build_video_grid(local_path, grid_path)
            downloaded_videos.append(
                {
                    **v,
                    "tag": tag,
                    "candidate_role": kind,
                    "local_path": str(local_path),
                    "grid_path": str(grid_path),
                    "sha256": sha,
                    "media": media_meta,
                }
            )
        except Exception as exc:
            local_path.unlink(missing_ok=True)
            print(f"    [warn] Falha em {v['title']}: {exc}")

    # 2. Discover high-res photos from Wikimedia Categories
    pos_cats = [
        "Coal trains",
        "Ore trains",
        "Gondola cars",
        "Open freight wagons",
        "Ballast wagons",
        "Hopper cars",
        "Eanos",
        "Falns",
    ]
    neg_cats = [
        "Container trains",
        "Tank cars",
        "Boxcars",
        "Freight trains of Germany",
    ]

    raw_pos_imgs: dict[str, dict[str, Any]] = {}
    for cat in pos_cats:
        try:
            for item in _fetch_category(cat, existing_titles, limit=30):
                raw_pos_imgs[item["title"]] = item
        except Exception as exc:
            print(f"[warn] Categoria foto positiva '{cat}' falhou: {exc}")

    for q in ("open top freight wagon", "coal hopper train"):
        try:
            for item in _fetch_search(q, "bitmap", existing_titles, limit=25):
                raw_pos_imgs[item["title"]] = item
        except Exception as exc:
            print(f"[warn] Busca foto '{q}' falhou: {exc}")

    raw_neg_imgs: dict[str, dict[str, Any]] = {}
    for cat in neg_cats:
        try:
            for item in _fetch_category(cat, existing_titles, limit=25):
                if item["title"] not in raw_pos_imgs:
                    raw_neg_imgs[item["title"]] = item
        except Exception as exc:
            print(f"[warn] Categoria foto negativa '{cat}' falhou: {exc}")

    def filter_photos(pool: dict[str, dict[str, Any]], cap: int) -> list[dict[str, Any]]:
        authors: Counter[str] = Counter()
        kept: list[dict[str, Any]] = []
        for it in pool.values():
            if len(kept) >= cap:
                break
            if it["mediatype"] != "BITMAP" or it["mime"] not in {"image/jpeg", "image/png"}:
                continue
            if it["width"] < 1024 or it["height"] < 576 or not (40_000 <= it["size_bytes"] <= 8 * 1024 * 1024):
                continue
            if _has_excluded_terms(it["title"], it["description"], it["categories"]):
                continue
            if authors[it["author"]] >= 2:
                continue
            authors[it["author"]] += 1
            kept.append(it)
        return kept

    pos_photo_pool = filter_photos(raw_pos_imgs, args.max_pos_photos)
    neg_photo_pool = filter_photos(raw_neg_imgs, args.max_neg_photos)

    downloaded_pos_photos: list[dict[str, Any]] = []
    print(f"[photos-pos] Baixando {len(pos_photo_pool)} fotos candidatas positivas...")
    for idx, p in enumerate(pos_photo_pool, start=1):
        ext = ".jpg" if p["mime"] == "image/jpeg" else ".png"
        tag = f"P{idx:02d}"
        local_path = staging_img_pos_dir / f"{tag}_{_slugify(p['title'], 36)}{ext}"
        try:
            if not local_path.is_file():
                _download_file(p["download_url"], local_path, timeout=45)
            sha = _sha256_file(local_path)
            if sha in existing_shas:
                local_path.unlink(missing_ok=True)
                continue
            existing_shas.add(sha)
            downloaded_pos_photos.append({**p, "tag": tag, "candidate_role": "pos", "local_path": str(local_path), "sha256": sha})
        except Exception as exc:
            local_path.unlink(missing_ok=True)
            print(f"  [warn] Foto {p['title']} falhou: {exc}")

    downloaded_neg_photos: list[dict[str, Any]] = []
    print(f"[photos-neg] Baixando {len(neg_photo_pool)} fotos candidatas negativas difíceis...")
    for idx, p in enumerate(neg_photo_pool, start=1):
        ext = ".jpg" if p["mime"] == "image/jpeg" else ".png"
        tag = f"N{idx:02d}"
        local_path = staging_img_neg_dir / f"{tag}_{_slugify(p['title'], 36)}{ext}"
        try:
            if not local_path.is_file():
                _download_file(p["download_url"], local_path, timeout=45)
            sha = _sha256_file(local_path)
            if sha in existing_shas:
                local_path.unlink(missing_ok=True)
                continue
            existing_shas.add(sha)
            downloaded_neg_photos.append({**p, "tag": tag, "candidate_role": "neg", "local_path": str(local_path), "sha256": sha})
        except Exception as exc:
            local_path.unlink(missing_ok=True)
            print(f"  [warn] Foto neg {p['title']} falhou: {exc}")

    mid = (len(downloaded_pos_photos) + 1) // 2
    if downloaded_pos_photos[:mid]:
        _build_photo_grid(downloaded_pos_photos[:mid], grid_dir / "photos_pos_grid_1.jpg", "WS2 Triagem Fotos Positivas (Parte 1)")
    if downloaded_pos_photos[mid:]:
        _build_photo_grid(downloaded_pos_photos[mid:], grid_dir / "photos_pos_grid_2.jpg", "WS2 Triagem Fotos Positivas (Parte 2)")
    if downloaded_neg_photos:
        _build_photo_grid(downloaded_neg_photos, grid_dir / "photos_neg_grid.jpg", "WS2 Triagem Fotos Negativos Difíceis")

    catalog = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "videos": downloaded_videos,
        "photos_positive": downloaded_pos_photos,
        "photos_negative": downloaded_neg_photos,
    }
    catalog_path = out_dir / "_staging_catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[OK] Staging concluído: {len(downloaded_videos)} vídeos, "
        f"{len(downloaded_pos_photos)} fotos positivas, {len(downloaded_neg_photos)} fotos negativas. "
        f"Catálogo: {catalog_path}"
    )


if __name__ == "__main__":
    main()
