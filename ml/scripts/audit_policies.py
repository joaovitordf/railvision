"""Shared dataset audit, license filtering, and manifest validation policies."""

from __future__ import annotations

import re
from typing import Any

REQUIRED_VIDEO_MANIFEST_FIELDS: tuple[str, ...] = (
    "id",
    "source_url",
    "retrieved_at",
    "author_or_owner",
    "license",
    "license_url",
    "permitted_uses",
    "sha256",
    "duration_s",
    "resolution",
    "fps",
    "angle_category",
    "cargo_visible",
    "wagon_type",
    "notes",
)

_RESTRICTIVE_RE = re.compile(
    r"(non[- ]?commercial|\bnc\b|no[- ]?deriv|\bnd\b|all rights reserved|copyrighted|fair use|proprietary)",
    re.IGNORECASE,
)

_PERMISSIVE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bcc0\b|public\s+domain|\bpd\b", re.IGNORECASE), "public_domain_or_cc0"),
    (
        re.compile(r"\bcc[- ]?by[- ]?sa(\s*[0-9.]+)?\b", re.IGNORECASE),
        "cc_by_sa_requires_attribution_and_share_alike",
    ),
    (
        re.compile(r"\bcc[- ]?by(\s*[0-9.]+)?\b", re.IGNORECASE),
        "cc_by_requires_attribution",
    ),
)


def is_license_acceptable(license_short_name: str | None, license_url: str | None = None) -> tuple[bool, str]:
    """Validate whether an explicit license permits local storage, analysis, and study/portfolio training."""
    combined = " ".join(part.strip() for part in (license_short_name or "", license_url or "") if part and part.strip())
    if not combined:
        return False, "rejected_missing_license"

    if _RESTRICTIVE_RE.search(combined):
        return False, "rejected_restrictive_nc_nd_or_copyright"

    for pattern, reason in _PERMISSIVE_PATTERNS:
        if pattern.search(license_short_name or "") or pattern.search(combined):
            return True, reason

    return False, "rejected_unrecognized_or_ambiguous_license"


def validate_yolo_label_line(line: str) -> tuple[bool, int | None, str | None]:
    """Validate one YOLO normalized detection annotation line: <class_id> <cx> <cy> <w> <h>."""
    parts = line.strip().split()
    if len(parts) != 5:
        return False, None, "expected_5_columns"
    try:
        class_id = int(parts[0])
        cx, cy, w, h = (float(x) for x in parts[1:])
    except ValueError:
        return False, None, "non_numeric_fields"

    if class_id < 0:
        return False, class_id, "negative_class_id"
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return False, class_id, "center_out_of_range"
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return False, class_id, "dimensions_out_of_range"
    return True, class_id, None


def validate_video_manifest_entry(entry: dict[str, Any]) -> list[str]:
    """Return list of missing required fields in a public video manifest entry."""
    return [field for field in REQUIRED_VIDEO_MANIFEST_FIELDS if field not in entry]
