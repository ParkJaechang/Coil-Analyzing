from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd

from .continuous_candidate_frequency import (
    attach_continuous_frequency_attrs,
    build_continuous_candidate_details,
    infer_continuous_source_frequency,
    matching_candidate_names,
)
from .continuous_steady_state_schema import adapt_continuous_source_frame
from .dataset_library import list_manifest_entries, load_dataset_library_settings, read_dataset_entry_bytes
from .ui_upload_state import category_payloads


def discover_continuous_candidate_frames(
    analysis_lookup: dict,
    *,
    upload_payloads: list[tuple[str, bytes]] | None = None,
    dataset_library_payloads: list[tuple[str, bytes]] | None = None,
    target_freq_hz: float | None = None,
) -> tuple[list[str], dict[str, pd.DataFrame], dict[str, Any]]:
    candidates: dict[str, pd.DataFrame] = {}
    rejected: list[str] = []
    counts = {"analysis_lookup": 0, "upload_memory_continuous": 0, "dataset_library": 0}
    for key, analysis in (analysis_lookup or {}).items():
        frame = getattr(getattr(analysis, "parsed", None), "normalized_frame", None)
        _try_add_candidate(candidates, rejected, f"analysis_lookup:{key}", frame, source_key="analysis_lookup", counts=counts)
    for name, payload in (upload_payloads if upload_payloads is not None else load_upload_memory_continuous_payloads()):
        frame, parse_error = _read_csv_payload(name, payload)
        if parse_error is not None:
            rejected.append(f"upload_memory:{name}: {parse_error}")
            continue
        _try_add_candidate(
            candidates,
            rejected,
            f"upload_memory:{name}",
            frame,
            source_key="upload_memory_continuous",
            counts=counts,
        )
    for name, payload in (
        dataset_library_payloads if dataset_library_payloads is not None else load_dataset_library_continuous_payloads()
    ):
        frame, parse_error = _read_csv_payload(name, payload)
        if parse_error is not None:
            rejected.append(f"dataset_library:{name}: {parse_error}")
            continue
        _try_add_candidate(candidates, rejected, f"dataset_library:{name}", frame, source_key="dataset_library", counts=counts)
    details = build_continuous_candidate_details(candidates, target_freq_hz=target_freq_hz)
    return [str(detail["name"]) for detail in details], candidates, {
        "continuous_candidate_source_counts": counts,
        "continuous_candidate_rejected_count": len(rejected),
        "continuous_candidate_reject_reasons": rejected,
        "continuous_candidate_rejection_reasons": rejected,
        "continuous_candidate_details": details,
        "continuous_candidate_matching_count": len(matching_candidate_names(details)),
    }


def is_continuous_candidate(frame: pd.DataFrame) -> bool:
    try:
        adapt_continuous_source_frame(frame)
    except ValueError:
        return False
    return True


def load_upload_memory_continuous_payloads() -> list[tuple[str, bytes]]:
    try:
        return category_payloads("continuous", None, include_cached_uploads=True)
    except Exception:
        return []


def load_dataset_library_continuous_payloads() -> list[tuple[str, bytes]]:
    try:
        settings = load_dataset_library_settings()
        dataset_root = str(settings.get("dataset_root") or "").strip()
        if not dataset_root:
            return []
        return [
            (str(entry.get("path") or ""), read_dataset_entry_bytes(dataset_root, str(entry.get("path") or "")))
            for entry in list_manifest_entries(dataset_root, dataset_mode="continuous")
        ]
    except Exception:
        return []


def _try_add_candidate(
    candidates: dict[str, pd.DataFrame],
    rejected: list[str],
    name: str,
    frame: pd.DataFrame | None,
    *,
    source_key: str,
    counts: dict[str, int],
) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return
    try:
        source = attach_continuous_frequency_attrs(frame, name=name)
        adapted, _metadata = adapt_continuous_source_frame(source)
    except ValueError as exc:
        rejected.append(f"{name}: {exc}")
        return
    candidates[name] = adapted
    counts[source_key] = int(counts.get(source_key, 0)) + 1


def _read_csv_payload(name: str, payload: bytes) -> tuple[pd.DataFrame | None, str | None]:
    if not str(name).lower().endswith(".csv"):
        return None, None
    try:
        text = payload.decode("utf-8-sig", errors="replace")
        attrs: dict[str, Any] = {"continuous_source_file": str(name)}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            key, sep, value = stripped[1:].partition(",")
            if sep and key.strip() == "Frequency(Hz)":
                try:
                    attrs.update(
                        {
                            "continuous_source_freq_hz": float(value.strip()),
                            "continuous_source_freq_source": "preamble",
                            "continuous_source_freq_inferred_from_preamble": True,
                        }
                    )
                except ValueError:
                    pass
        data_lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if not data_lines:
            return None, "csv_parse_error:no_data_rows_after_metadata_preamble"
        frame = pd.read_csv(StringIO("\n".join(data_lines)))
        if "continuous_source_freq_hz" not in attrs:
            filename_freq, filename_source = infer_continuous_source_frequency(name)
            if filename_freq is not None:
                attrs.update(
                    {
                        "continuous_source_freq_hz": filename_freq,
                        "continuous_source_freq_source": filename_source,
                        "continuous_source_freq_inferred_from_filename": True,
                    }
                )
        frame.attrs.update(attrs)
        return frame, None
    except Exception as exc:  # noqa: BLE001 - candidate scan should surface reject reasons in UI.
        return None, f"csv_parse_error:{type(exc).__name__}:{exc}"
