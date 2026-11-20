"""Download (into gitignored data/raw) and audit TWIST labelled_dataset.tar.gz."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_policies import validate_yolo_label_line  # noqa: E402

ZENODO_URL = "https://zenodo.org/api/records/19472084/files/labelled_dataset.tar.gz/content"
EXPECTED_MD5 = "64d50175ce51c52f292c688a37c7a453"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def download_if_needed(dest: Path) -> tuple[str, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        md5_hex, sha_hex = file_hashes(dest)
        if md5_hex == EXPECTED_MD5:
            print(f"[TWIST] Arquivo ja existe com MD5 valido: {md5_hex}")
            return md5_hex, sha_hex
        print(f"[TWIST] MD5 divergente ({md5_hex}), baixando novamente...")

    print(f"[TWIST] Baixando {ZENODO_URL} -> {dest} ...")
    tmp = dest.with_suffix(".part")
    with requests.get(ZENODO_URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as out:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)
    tmp.replace(dest)
    md5_hex, sha_hex = file_hashes(dest)
    if md5_hex != EXPECTED_MD5:
        raise RuntimeError(f"MD5 divergente apos download: esperado={EXPECTED_MD5}, obtido={md5_hex}")
    print(f"[TWIST] Download concluido. MD5={md5_hex} SHA256={sha_hex}")
    return md5_hex, sha_hex


def safe_extract_tar(archive: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    marker = target_dir / ".extracted_ok"
    if marker.is_file():
        print(f"[TWIST] Ja extraido em {target_dir}")
        return
    print(f"[TWIST] Extraindo para {target_dir} ...")
    resolved_target = target_dir.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            dest_path = (resolved_target / member.name).resolve()
            if not str(dest_path).startswith(str(resolved_target)):
                raise RuntimeError(f"Entrada tar invalida (path traversal): {member.name}")
        tf.extractall(resolved_target)
    marker.write_text("ok\n", encoding="utf-8")


def dhash64(img: Image.Image) -> int:
    gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    pixels = list(gray.getdata())
    bits = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            bits = (bits << 1) | (1 if pixels[offset + col] > pixels[offset + col + 1] else 0)
    return bits


def build_contact_sheet(samples: list[tuple[Path, list[tuple[int, float, float, float, float]]]], class_names: dict[int, str], out_path: Path) -> None:
    if not samples:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h = 480, 360
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (20, 24, 30))
    colors = {0: "#22b8cf", 1: "#f59e0b", 2: "#10b981", 3: "#ef4444"}
    font = ImageFont.load_default()

    for idx, (img_path, boxes) in enumerate(samples):
        r, c = divmod(idx, cols)
        with Image.open(img_path) as im:
            rgb = im.convert("RGB").resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(rgb)
        for cls_id, cx, cy, w, h in boxes:
            x0 = int((cx - w / 2) * cell_w)
            y0 = int((cy - h / 2) * cell_h)
            x1 = int((cx + w / 2) * cell_w)
            y1 = int((cy + h / 2) * cell_h)
            color = colors.get(cls_id, "#ffffff")
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            label = class_names.get(cls_id, f"cls_{cls_id}")
            draw.rectangle([x0, max(0, y0 - 14), x0 + 6 + len(label) * 6, max(14, y0)], fill=color)
            draw.text((x0 + 3, max(1, y0 - 13)), label, fill="#000000", font=font)
        draw.rectangle([0, cell_h - 18, cell_w, cell_h], fill=(0, 0, 0))
        draw.text((6, cell_h - 14), img_path.name[:55], fill="#ffffff", font=font)
        sheet.paste(rgb, (c * cell_w, r * cell_h))

    sheet.save(out_path, quality=90)
    print(f"[TWIST] Contact sheet salvo em {out_path}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    archive = root / "data" / "raw" / "twist" / "labelled_dataset.tar.gz"
    extracted = root / "data" / "processed" / "twist-labelled"
    out_json = root / "artifacts" / "runs" / "audit" / "twist_audit.json"
    contact_sheet = root / "data" / "processed" / "triage-frames" / "twist_detailed_samples.jpg"

    md5_hex, sha_hex = download_if_needed(archive)
    safe_extract_tar(archive, extracted)

    all_files = [p for p in extracted.rglob("*") if p.is_file() and p.name != ".extracted_ok"]
    top_tree = sorted(str(p.relative_to(extracted)) for p in extracted.iterdir() if p.name != ".extracted_ok")
    meta_files = [p for p in all_files if p.suffix.lower() in {".yaml", ".yml", ".md", ".txt", ".json", ".csv"} and p.parent.name not in {"labels", "train", "val", "test"}]

    meta_previews: dict[str, str] = {}
    for mp in meta_files[:20]:
        try:
            meta_previews[str(mp.relative_to(extracted))] = mp.read_text(encoding="utf-8", errors="replace")[:1500]
        except OSError:
            pass

    # Group files by top-level directory under extracted
    images = [p for p in all_files if p.suffix.lower() in IMAGE_EXTENSIONS]
    txt_files = [p for p in all_files if p.suffix.lower() == ".txt"]

    folder_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "images": 0,
            "corrupt_images": 0,
            "resolutions": Counter(),
            "label_files": 0,
            "empty_label_files": 0,
            "boxes_total": 0,
            "invalid_box_lines": 0,
            "class_counts": Counter(),
            "images_with_label_file": 0,
            "images_without_label_file": 0,
        }
    )

    label_by_parent_and_stem: dict[tuple[str, str], Path] = {}
    for lp in txt_files:
        rel = lp.relative_to(extracted)
        top = rel.parts[0] if len(rel.parts) > 1 else "."
        # Check if this looks like a YOLO label file (not classes.txt / readme)
        if lp.stem.lower() in {"classes", "readme", "notes"}:
            continue
        label_by_parent_and_stem[(top, lp.stem)] = lp
        st = folder_stats[top]
        st["label_files"] += 1
        content = lp.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            st["empty_label_files"] += 1
            continue
        for line in content.splitlines():
            if not line.strip():
                continue
            ok, cls_id, _ = validate_yolo_label_line(line)
            if ok and cls_id is not None:
                st["boxes_total"] += 1
                st["class_counts"][str(cls_id)] += 1
            else:
                st["invalid_box_lines"] += 1

    sha_seen: dict[str, str] = {}
    exact_duplicates = 0
    detailed_candidates: list[tuple[Path, list[tuple[int, float, float, float, float]]]] = []
    dhashes: list[tuple[str, int]] = []

    for img_path in sorted(images):
        rel = img_path.relative_to(extracted)
        top = rel.parts[0] if len(rel.parts) > 1 else "."
        st = folder_stats[top]
        st["images"] += 1

        digest = hashlib.sha256(img_path.read_bytes()).hexdigest()
        if digest in sha_seen:
            exact_duplicates += 1
        else:
            sha_seen[digest] = str(rel)

        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im2:
                w, h = im2.size
                st["resolutions"][f"{w}x{h}"] += 1
                if len(st["class_counts"]) > 2 or "detail" in str(rel).lower() or "multi" in str(rel).lower():
                    dhashes.append((str(rel), dhash64(im2)))
        except Exception:
            st["corrupt_images"] += 1

        # Match label by stem within same top folder
        matched_label = label_by_parent_and_stem.get((top, img_path.stem))
        if matched_label is not None:
            st["images_with_label_file"] += 1
            parsed_boxes: list[tuple[int, float, float, float, float]] = []
            for line in matched_label.read_text(encoding="utf-8", errors="replace").splitlines():
                ok, cls_id, _ = validate_yolo_label_line(line)
                if ok and cls_id is not None:
                    parts = line.split()
                    parsed_boxes.append((cls_id, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
            if parsed_boxes:
                detailed_candidates.append((img_path, parsed_boxes))
        else:
            st["images_without_label_file"] += 1

    # Prioritize images with multiple boxes or higher class IDs (detailed multi-class subset) for the contact sheet
    multi_class_candidates = [item for item in detailed_candidates if any(b[0] >= 1 for b in item[1]) or len(item[1]) >= 2]
    pool = multi_class_candidates if len(multi_class_candidates) >= 12 else detailed_candidates
    step = max(1, len(pool) // 12)
    sampled = [pool[i] for i in range(0, len(pool), step)][:12]
    build_contact_sheet(sampled, {}, contact_sheet)

    # Near-duplicate count on collected dhashes (Hamming <= 2)
    near_dups = 0
    for i in range(len(dhashes)):
        h1 = dhashes[i][1]
        for j in range(i + 1, min(i + 40, len(dhashes))):
            if (h1 ^ dhashes[j][1]).bit_count() <= 2:
                near_dups += 1
                break

    serializable_folders = {}
    for k, v in sorted(folder_stats.items()):
        serializable_folders[k] = {
            "images": v["images"],
            "corrupt_images": v["corrupt_images"],
            "resolutions": dict(v["resolutions"]),
            "label_files": v["label_files"],
            "empty_label_files": v["empty_label_files"],
            "boxes_total": v["boxes_total"],
            "invalid_box_lines": v["invalid_box_lines"],
            "class_counts": dict(sorted(v["class_counts"].items(), key=lambda kv: int(kv[0]))),
            "images_with_label_file": v["images_with_label_file"],
            "images_without_label_file": v["images_without_label_file"],
        }

    report = {
        "source_url": ZENODO_URL,
        "archive_size_bytes": archive.stat().st_size,
        "archive_md5": md5_hex,
        "archive_sha256": sha_hex,
        "top_level_entries": top_tree,
        "metadata_previews": meta_previews,
        "total_images": len(images),
        "total_txt_files": len(txt_files),
        "exact_image_duplicates": exact_duplicates,
        "near_duplicate_window_matches": near_dups,
        "folders": serializable_folders,
        "contact_sheet": str(contact_sheet.relative_to(root)),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "metadata_previews"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
