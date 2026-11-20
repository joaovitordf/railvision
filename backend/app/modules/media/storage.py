"""Local filesystem storage and RFC 9110 Range streaming for uploaded inspection media."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from urllib.parse import quote
from uuid import uuid4

from fastapi import UploadFile
from fastapi.responses import Response, StreamingResponse

from app.core.exceptions import (
    InvalidMediaException,
    InvalidMediaRangeException,
    MediaNotFoundException,
    MediaTooLargeException,
)

DEFAULT_CHUNK_SIZE_BYTES = 1024 * 1024
_SINGLE_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


@dataclass(frozen=True, slots=True)
class StoredMedia:
    """Metadata of a locally persisted media upload."""

    stored_filename: str
    sha256_checksum: str
    file_size_bytes: int
    full_path: Path


def parse_byte_range(range_header: str, content_length: int, max_chunk_bytes: int = DEFAULT_CHUNK_SIZE_BYTES) -> tuple[int, int]:
    """Parse a single HTTP byte range header matching LocalMediaStorageService.buildRegion."""
    if content_length <= 0:
        raise InvalidMediaRangeException(0)

    match = _SINGLE_RANGE_RE.match(range_header.strip())
    if not match:
        raise InvalidMediaRangeException(content_length)

    start_raw, end_raw = match.groups()
    if not start_raw and not end_raw:
        raise InvalidMediaRangeException(content_length)

    if not start_raw:
        suffix_len = int(end_raw)
        if suffix_len <= 0:
            raise InvalidMediaRangeException(content_length)
        start = max(0, content_length - suffix_len)
        end = content_length - 1
    else:
        start = int(start_raw)
        end = int(end_raw) if end_raw else (content_length - 1)

    if start < 0 or start >= content_length or start > end:
        raise InvalidMediaRangeException(content_length)

    bounded_end = min(end, content_length - 1)
    capped_len = min(max_chunk_bytes, bounded_end - start + 1)
    return start, start + capped_len - 1


def _iter_file_slice(path: Path, start: int, length: int, read_size: int = 64 * 1024) -> Iterator[bytes]:
    remaining = length
    with path.open("rb") as handle:
        if start > 0:
            handle.seek(start)
        while remaining > 0:
            chunk = handle.read(min(read_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


class LocalMediaStorageService:
    """Safe local upload repository and byte-range streaming service."""

    def __init__(self, storage_root: Path, max_upload_bytes: int) -> None:
        self.storage_root = storage_root.resolve()
        self.uploads_root = (self.storage_root / "uploads").resolve()
        self.max_upload_bytes = max_upload_bytes
        self.uploads_root.mkdir(parents=True, exist_ok=True)

    def resolve_safe_upload_path(self, stored_filename: str) -> Path:
        """Resolve an upload filename under uploads_root, blocking traversal."""
        if (
            not stored_filename
            or not stored_filename.strip()
            or ".." in stored_filename
            or "/" in stored_filename
            or "\\" in stored_filename
        ):
            raise ValueError("Nome de arquivo armazenado inválido.")
        resolved = (self.uploads_root / stored_filename).resolve()
        if resolved.parent != self.uploads_root:
            raise ValueError("Tentativa de acesso fora do diretório de armazenamento.")
        return resolved

    def store_upload(self, upload: UploadFile, extension: str) -> StoredMedia:
        """Persist an uploaded stream with incremental SHA-256 and strict size enforcement."""
        stored_filename = f"{uuid4()}.{extension}"
        destination = self.resolve_safe_upload_path(stored_filename)
        digest = hashlib.sha256()
        total_bytes = 0

        try:
            upload.file.seek(0)
            with destination.open("wb") as out:
                while True:
                    chunk = upload.file.read(64 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > self.max_upload_bytes:
                        raise MediaTooLargeException("O vídeo enviado excede o tamanho máximo permitido.")
                    digest.update(chunk)
                    out.write(chunk)
            if total_bytes == 0:
                raise InvalidMediaException(
                    "EMPTY_FILE",
                    "Nenhum arquivo de vídeo válido foi enviado ou o arquivo está vazio.",
                )
        except Exception:
            if destination.exists():
                destination.unlink(missing_ok=True)
            raise

        return StoredMedia(
            stored_filename=stored_filename,
            sha256_checksum=digest.hexdigest(),
            file_size_bytes=total_bytes,
            full_path=destination,
        )

    def load_existing_path(self, stored_filename: str) -> Path:
        """Return validated readable path or raise MediaNotFoundException."""
        path = self.resolve_safe_upload_path(stored_filename)
        if not path.is_file() or not os.access(path, os.R_OK):
            raise MediaNotFoundException("Arquivo de mídia da inspeção não encontrado.")
        return path

    def build_media_response(
        self,
        *,
        file_path: Path,
        content_type: str,
        range_header: str | None = None,
        download_filename: str | None = None,
    ) -> Response:
        """Build full (200) or bounded partial-content (206) streaming response."""
        if not file_path.is_file() or not os.access(file_path, os.R_OK):
            raise MediaNotFoundException("Arquivo de mídia não encontrado.")

        total_size = file_path.stat().st_size
        headers: dict[str, str] = {"Accept-Ranges": "bytes"}
        if download_filename:
            ascii_fallback = download_filename.encode("ascii", "ignore").decode("ascii") or "download"
            headers["Content-Disposition"] = (
                f'attachment; filename="{ascii_fallback}"; '
                f"filename*=UTF-8''{quote(download_filename)}"
            )

        if range_header:
            start, end = parse_byte_range(range_header, total_size)
            length = end - start + 1
            headers["Content-Range"] = f"bytes {start}-{end}/{total_size}"
            headers["Content-Length"] = str(length)
            return StreamingResponse(
                _iter_file_slice(file_path, start, length),
                status_code=206,
                media_type=content_type,
                headers=headers,
            )

        headers["Content-Length"] = str(total_size)
        return StreamingResponse(
            _iter_file_slice(file_path, 0, total_size),
            status_code=200,
            media_type=content_type,
            headers=headers,
        )

    def is_writable(self) -> bool:
        """Verify uploads directory exists and is writable."""
        return self.uploads_root.is_dir() and os.access(self.uploads_root, os.W_OK)
