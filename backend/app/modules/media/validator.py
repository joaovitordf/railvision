"""Video upload validation matching VideoUploadValidator.java."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from fastapi import UploadFile

from app.core.exceptions import InvalidMediaException, MediaTooLargeException

ALLOWED_VIDEO_EXTENSIONS = frozenset({"mp4", "mov", "mkv", "webm", "avi"})
EXTENSION_TO_MIME = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
    "webm": "video/webm",
    "avi": "video/x-msvideo",
}
QUICKTIME_ATOMS = frozenset({b"ftyp", b"moov", b"wide", b"free", b"mdat"})


@dataclass(frozen=True, slots=True)
class ValidatedVideoMetadata:
    """Validated upload metadata prior to persisting on disk."""

    original_filename: str
    extension: str
    content_type: str
    file_size_bytes: int


def sanitize_original_filename(raw_filename: str | None) -> str:
    """Validate and clean an uploaded video filename against path traversal."""
    if raw_filename is None or not raw_filename.strip():
        raise InvalidMediaException("MISSING_FILENAME", "Nome do arquivo de vídeo ausente.")
    trimmed = raw_filename.strip()
    if ".." in trimmed or "/" in trimmed or "\\" in trimmed:
        raise InvalidMediaException("INVALID_FILENAME", "Nome de arquivo inválido.")
    return trimmed


def extract_extension(filename: str) -> str:
    """Extract lowercase extension without leading dot, or empty string."""
    dot_idx = filename.rfind(".")
    if dot_idx <= 0 or dot_idx == len(filename) - 1:
        return ""
    return filename[dot_idx + 1 :].lower()


def matches_container_signature(extension: str, header: bytes) -> bool:
    """Check container magic bytes for supported video extensions."""
    if extension == "mp4":
        return len(header) >= 12 and header[4:8] == b"ftyp"
    if extension == "mov":
        return len(header) >= 12 and header[4:8] in QUICKTIME_ATOMS
    if extension in {"webm", "mkv"}:
        return len(header) >= 4 and header[:4] == b"\x1a\x45\xdf\xa3"
    if extension == "avi":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI "
    return False


def _measure_stream_size(stream: BinaryIO) -> int:
    if hasattr(stream, "seek") and hasattr(stream, "tell"):
        pos = stream.tell()
        stream.seek(0, 2)
        size = int(stream.tell())
        stream.seek(pos, 0)
        return size
    return 0


class VideoUploadValidator:
    """Validates uploaded video extension, container signature, and size limit."""

    def __init__(self, max_file_size_bytes: int) -> None:
        self.max_file_size_bytes = max_file_size_bytes

    def validate(self, upload: UploadFile) -> ValidatedVideoMetadata:
        """Validate uploaded video stream without exhausting memory and rewind to byte 0."""
        stream = upload.file
        size = upload.size if upload.size is not None else _measure_stream_size(stream)
        if size <= 0:
            raise InvalidMediaException(
                "EMPTY_FILE",
                "Nenhum arquivo de vídeo válido foi enviado ou o arquivo está vazio.",
            )
        if size > self.max_file_size_bytes:
            raise MediaTooLargeException("O vídeo enviado excede o tamanho máximo permitido.")

        sanitized_name = sanitize_original_filename(upload.filename)
        extension = extract_extension(sanitized_name)
        if extension not in ALLOWED_VIDEO_EXTENSIONS:
            raise InvalidMediaException(
                "UNSUPPORTED_EXTENSION",
                "Formato de arquivo não suportado. Utilize MP4, MOV, MKV, WEBM ou AVI.",
            )

        try:
            stream.seek(0)
            header = stream.read(16)
            stream.seek(0)
        except OSError as exc:
            raise InvalidMediaException(
                "UNREADABLE_MEDIA",
                "Não foi possível ler o arquivo de vídeo enviado.",
            ) from exc

        if not matches_container_signature(extension, header):
            raise InvalidMediaException(
                "INVALID_VIDEO_SIGNATURE",
                f"O conteúdo do arquivo não corresponde a um container de vídeo válido para a extensão .{extension}.",
            )

        content_type = EXTENSION_TO_MIME.get(extension, "application/octet-stream")
        return ValidatedVideoMetadata(
            original_filename=sanitized_name,
            extension=extension,
            content_type=content_type,
            file_size_bytes=int(size),
        )


def strip_extension(filename: str) -> str:
    """Return filename stem matching Java InspectionService.stripExtension."""
    dot_idx = Path(filename).name.rfind(".")
    return filename[:dot_idx] if dot_idx > 0 else filename
