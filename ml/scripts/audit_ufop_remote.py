"""Inspect UFOP Iron Ore Conveyor ZIP64 central directory via HTTP Range without downloading 5.8 GiB."""

from __future__ import annotations

import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import requests

MENDELEY_FILES_API = "https://data.mendeley.com/public-api/datasets/s25x2bnshz/files?folder_id=root&version=1"


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only stream backed by HTTP Range requests with simple tail/chunk caching."""

    def __init__(self, url: str, total_size: int, chunk_size: int = 512 * 1024) -> None:
        self.url = url
        self.total_size = total_size
        self.chunk_size = chunk_size
        self._pos = 0
        self._session = requests.Session()
        self._cache_start = -1
        self._cache_data = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.total_size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        self._pos = max(0, min(self._pos, self.total_size))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if self._pos >= self.total_size:
            return b""
        if size is None or size < 0:
            size = self.total_size - self._pos
        size = min(size, self.total_size - self._pos)
        if size == 0:
            return b""

        if not (self._cache_start <= self._pos < self._cache_start + len(self._cache_data)):
            fetch_start = self._pos
            fetch_end = min(self.total_size - 1, fetch_start + max(size, self.chunk_size) - 1)
            resp = self._session.get(
                self.url,
                headers={"Range": f"bytes={fetch_start}-{fetch_end}"},
                timeout=30,
            )
            if resp.status_code not in (200, 206):
                raise RuntimeError(f"HTTP Range request failed with status {resp.status_code}")
            self._cache_start = fetch_start
            self._cache_data = resp.content

        rel = self._pos - self._cache_start
        chunk = self._cache_data[rel : rel + size]
        self._pos += len(chunk)
        return chunk


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_json = root / "artifacts" / "runs" / "audit" / "ufop_audit.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    meta_resp = requests.get(MENDELEY_FILES_API, timeout=30)
    meta_resp.raise_for_status()
    files = meta_resp.json()
    target = files[0]
    details = target["content_details"]
    download_url = details["download_url"]
    total_size = int(target["size"])
    sha256_declared = details["sha256_hash"]

    # Probe HTTP Range support explicitly
    probe = requests.get(download_url, headers={"Range": "bytes=0-31"}, timeout=30)
    range_supported = probe.status_code == 206

    report: dict = {
        "api_url": MENDELEY_FILES_API,
        "filename": target["filename"],
        "size_bytes": total_size,
        "sha256_declared_by_mendeley": sha256_declared,
        "range_supported": range_supported,
    }

    if not range_supported:
        report["note"] = f"CDN retornou HTTP {probe.status_code}; central directory não inspecionado sem download completo."
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    stream = HttpRangeFile(download_url, total_size)
    with zipfile.ZipFile(stream, "r") as zf:
        infos = zf.infolist()
        yaml_preview = None
        for entry in infos:
            if entry.filename.endswith("data.yaml"):
                yaml_preview = zf.read(entry.filename).decode("utf-8", errors="replace")[:1500]
                break

    file_infos = [i for i in infos if not i.is_dir()]
    dir_infos = [i for i in infos if i.is_dir()]
    ext_counts = Counter(Path(i.filename).suffix.lower() or "<no_ext>" for i in file_infos)
    top_folders = Counter(i.filename.split("/")[0] for i in file_infos if "/" in i.filename)
    second_folders = Counter(
        "/".join(i.filename.split("/")[:2]) for i in file_infos if len(i.filename.split("/")) >= 2
    )
    sample_annotations = [
        i.filename
        for i in file_infos
        if Path(i.filename).suffix.lower() in {".txt", ".json", ".xml", ".csv", ".yaml", ".yml"}
    ][:25]
    sample_images = [
        i.filename
        for i in file_infos
        if Path(i.filename).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ][:15]

    report.update(
        {
            "total_zip_entries": len(infos),
            "directory_entries": len(dir_infos),
            "file_entries": len(file_infos),
            "uncompressed_total_bytes": sum(i.file_size for i in file_infos),
            "extensions": dict(ext_counts.most_common()),
            "top_level_folders": dict(top_folders.most_common()),
            "second_level_folders": dict(second_folders.most_common(30)),
            "sample_annotation_or_metadata_files": sample_annotations,
            "sample_image_files": sample_images,
            "data_yaml_preview": yaml_preview,
        }
    )

    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
