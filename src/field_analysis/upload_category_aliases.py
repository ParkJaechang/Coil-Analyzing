from __future__ import annotations

from typing import Any


UPLOAD_CATEGORIES = ("continuous", "transient", "validation", "lcr")

CATEGORY_LABELS = {
    "continuous": "연속 cycle",
    "transient": "finite-cycle",
    "validation": "2차 보정 검증 run",
    "lcr": "LCR",
}

UPLOADER_SESSION_KEYS = {
    "continuous": "continuous_uploads",
    "transient": "transient_uploads",
    "validation": "validation_uploads",
    "lcr": "lcr_uploads",
}

UPLOAD_MEMORY_LABEL_BY_CATEGORY = {
    "continuous": "continuous_cycle",
    "transient": "finite_cycle",
    "validation": "actual_drive_validation_run",
    "lcr": "lcr",
}

UPLOAD_MEMORY_CATEGORY_BY_LABEL = {value: key for key, value in UPLOAD_MEMORY_LABEL_BY_CATEGORY.items()}

UPLOAD_CATEGORY_ALIASES = {
    "continuous": "continuous",
    "continuous-cycle": "continuous",
    "continuous_cycle": "continuous",
    "continuous steady state": "continuous",
    "continuous_steady_state": "continuous",
    "continuous-steady-state": "continuous",
    "continuous cycle": "continuous",
    "continuous_cycle_input": "continuous",
    "연속 cycle": "continuous",
    "연속-cycle": "continuous",
    "연속_cycle": "continuous",
    "transient": "transient",
    "finite": "transient",
    "finite-cycle": "transient",
    "finite_cycle": "transient",
    "finite cycle": "transient",
    "validation": "validation",
    "2nd": "validation",
    "second": "validation",
    "second_validation": "validation",
    "actual_drive_validation": "validation",
    "lcr": "lcr",
}


def normalize_upload_category(category: Any) -> str:
    text = str(category or "").strip()
    if not text:
        return ""
    lowered = text.lower().replace("/", "-").replace("\\", "-")
    return UPLOAD_CATEGORY_ALIASES.get(text, UPLOAD_CATEGORY_ALIASES.get(lowered, lowered))


def upload_category_alias_metadata(category: Any) -> dict[str, object]:
    original = str(category or "").strip()
    canonical = normalize_upload_category(original)
    return {
        "upload_category_original": original,
        "upload_category_canonical": canonical,
        "upload_category_alias_applied": bool(original and canonical and original != canonical),
    }
