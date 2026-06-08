from __future__ import annotations

from io import StringIO
from pathlib import Path
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
from .upload_filename import canonicalize_upload_filename
from .ui_upload_state import list_persisted_uploads


def discover_continuous_candidate_frames(
    analysis_lookup: dict,
    *,
    upload_payloads: list[tuple[str, bytes]] | None = None,
    dataset_library_payloads: list[tuple[str, bytes]] | None = None,
    target_freq_hz: float | None = None,
    source_waveform_filter: str | None = None,
) -> tuple[list[str], dict[str, pd.DataFrame], dict[str, Any]]:
    candidates: dict[str, pd.DataFrame] = {}
    rejected: list[str] = []
    counts = {"analysis_lookup": 0, "upload_memory_continuous": 0, "dataset_library": 0}
    for key, analysis in (analysis_lookup or {}).items():
        frame = getattr(getattr(analysis, "parsed", None), "normalized_frame", None)
        _try_add_candidate(candidates, rejected, f"analysis_lookup:{key}", frame, source_key="analysis_lookup", counts=counts)
    for payload_item in (upload_payloads if upload_payloads is not None else load_upload_memory_continuous_payloads()):
        name, payload, storage_name = _payload_parts(payload_item)
        frame, parse_error = _read_csv_payload(name, payload, storage_name=storage_name)
        if parse_error is not None:
            rejected.append(f"upload_memory:{name}: {parse_error}")
            continue
        _try_add_candidate(
            candidates,
            rejected,
            f"upload_memory:{_canonical_name(name)}",
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
    requested_waveform = str(source_waveform_filter or "all")
    if requested_waveform != "all":
        candidates = {
            name: frame
            for name, frame in candidates.items()
            if str(frame.attrs.get("continuous_source_waveform_family") or "unknown") == requested_waveform
        }
    details = build_continuous_candidate_details(
        candidates,
        target_freq_hz=target_freq_hz,
        source_waveform_filter=requested_waveform,
    )
    matching_names = matching_candidate_names(details)
    return [str(detail["name"]) for detail in details], candidates, {
        "continuous_candidate_source_counts": counts,
        "continuous_candidate_rejected_count": len(rejected),
        "continuous_candidate_reject_reasons": rejected,
        "continuous_candidate_rejection_reasons": rejected,
        "continuous_candidate_details": details,
        "continuous_candidate_matching_count": len(matching_names),
        "matching_candidate_count": len(matching_names),
        "matching_candidate_names": matching_names,
        "continuous_source_waveform_filter": requested_waveform,
    }


def is_continuous_candidate(frame: pd.DataFrame) -> bool:
    try:
        adapt_continuous_source_frame(frame)
    except ValueError:
        return False
    return True


def load_upload_memory_continuous_payloads() -> list[Any]:
    try:
        payloads = []
        for record in list_persisted_uploads("continuous"):
            path = str(record.get("path") or "")
            if not path:
                continue
            canonical_name = str(record.get("canonical_filename") or record.get("original_filename") or record.get("cache_name") or "")
            storage_name = str(record.get("cache_name") or record.get("stored_filename") or canonical_name)
            payloads.append({"name": canonical_name, "storage_name": storage_name, "payload": Path(path).read_bytes()})
        return payloads
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
        columns = ",".join(str(column) for column in frame.columns)
        rejected.append(f"{name}: {exc}; columns={columns}")
        return
    candidates[name] = adapted
    counts[source_key] = int(counts.get(source_key, 0)) + 1


def _payload_parts(payload_item: Any) -> tuple[str, bytes, str | None]:
    if isinstance(payload_item, dict):
        name = str(payload_item.get("name") or payload_item.get("canonical_filename") or payload_item.get("storage_name") or "")
        payload = bytes(payload_item.get("payload") or b"")
        storage_name = str(payload_item.get("storage_name") or payload_item.get("stored_filename") or name)
        return name, payload, storage_name
    name, payload = payload_item
    return str(name), bytes(payload), str(name)


def _read_csv_payload(name: str, payload: bytes, *, storage_name: str | None = None) -> tuple[pd.DataFrame | None, str | None]:
    if not str(name).lower().endswith(".csv"):
        return None, None
    try:
        text = payload.decode("utf-8-sig", errors="replace")
        filename_meta = canonicalize_upload_filename(storage_name or name, original_filename=name)
        canonical_name = str(filename_meta["upload_canonical_filename"])
        attrs: dict[str, Any] = {
            "continuous_source_file": canonical_name,
            **filename_meta,
        }
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
            filename_freq, filename_source = infer_continuous_source_frequency(canonical_name)
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


def _canonical_name(name: str) -> str:
    return str(canonicalize_upload_filename(name)["upload_canonical_filename"])
