from __future__ import annotations

import re
from pathlib import PurePath
from typing import Any


_UPLOAD_HASH_PREFIX = re.compile(r"^(?P<prefix>[0-9a-f]{8,32})_(?P<rest>.+)$", re.IGNORECASE)
_KNOWN_DATASET_PREFIX = re.compile(
    r"^(continuous|finite|actual|actual_drive|finite_recommended|second|2nd)_",
    re.IGNORECASE,
)


def canonicalize_upload_filename(
    filename: object,
    *,
    original_filename: object | None = None,
) -> dict[str, Any]:
    storage_filename = PurePath(str(filename or "")).name
    original = PurePath(str(original_filename or "")).name if original_filename else ""
    if original and original != storage_filename:
        return {
            "upload_storage_filename": storage_filename,
            "upload_original_filename": original,
            "upload_canonical_filename": original,
            "upload_filename_prefix_stripped": False,
            "upload_filename_prefix_strip_status": "metadata_original_filename",
        }
    match = _UPLOAD_HASH_PREFIX.match(storage_filename)
    if match and _KNOWN_DATASET_PREFIX.match(match.group("rest")):
        return {
            "upload_storage_filename": storage_filename,
            "upload_original_filename": match.group("rest"),
            "upload_canonical_filename": match.group("rest"),
            "upload_id": match.group("prefix"),
            "upload_filename_prefix_stripped": True,
            "upload_filename_prefix_strip_status": "stripped_known_upload_prefix",
        }
    return {
        "upload_storage_filename": storage_filename,
        "upload_original_filename": original or storage_filename,
        "upload_canonical_filename": original or storage_filename,
        "upload_filename_prefix_stripped": False,
        "upload_filename_prefix_strip_status": "unchanged",
    }


def canonical_upload_filename(filename: object, *, original_filename: object | None = None) -> str:
    return str(canonicalize_upload_filename(filename, original_filename=original_filename)["upload_canonical_filename"])
