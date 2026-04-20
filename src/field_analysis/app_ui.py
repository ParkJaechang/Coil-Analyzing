from __future__ import annotations

import importlib
import importlib.util
from importlib.machinery import SourcelessFileLoader
from pathlib import Path
import re
import sys

import pandas as pd
import streamlit as st


def _package_name() -> str:
    return (__package__ or __name__.rpartition(".")[0] or __name__).rstrip(".")


def _base_dir() -> Path:
    base_dir = Path(__file__).resolve().parent
    if base_dir.name == "__pycache__":
        base_dir = base_dir.parent
    return base_dir


def _load_source_module():
    source_path = _base_dir() / "app_ui_snapshot.py"
    if not source_path.exists():
        raise FileNotFoundError(f"UI source snapshot not found: {source_path}")
    package_name = _package_name()
    module_name = f"{package_name}.app_ui_snapshot" if package_name else "app_ui_snapshot"
    return importlib.import_module(module_name)


def _load_compiled_module():
    base_dir = _base_dir()
    pyc_path = base_dir / "__pycache__" / f"app_ui.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
    if not pyc_path.exists():
        raise FileNotFoundError(f"Compiled UI module not found: {pyc_path}")

    current_path = Path(__file__).resolve()
    try:
        if current_path.samefile(pyc_path):
            raise ImportError(f"Refusing recursive self-import from compiled UI module: {pyc_path}")
    except FileNotFoundError:
        pass

    package_name = _package_name()
    loader = SourcelessFileLoader(f"{package_name}._app_ui_compiled", str(pyc_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise ImportError(f"Unable to build spec for {pyc_path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _load_ui_module():
    try:
        return _load_source_module()
    except FileNotFoundError:
        return _load_compiled_module()


_compiled = _load_ui_module()

for _name, _value in vars(_compiled).items():
    if _name in {"__name__", "__loader__", "__package__", "__spec__", "__file__", "__cached__"}:
        continue
    globals()[_name] = _value

__doc__ = getattr(_compiled, "__doc__", None)

_UPLOAD_HASHED_NAME_PATTERN = re.compile(r"^(?:[0-9a-f]{16})_(.+)$", re.IGNORECASE)


def _display_source_file_name(file_name: object) -> str:
    raw_name = str(file_name or "").strip()
    if not raw_name:
        return ""
    normalized = raw_name.replace("\\", "/")
    parent_parts = [part for part in normalized.split("/")[:-1] if part]
    leaf_name = normalized.rsplit("/", 1)[-1]
    matched = _UPLOAD_HASHED_NAME_PATTERN.match(leaf_name)
    cleaned_leaf = matched.group(1) if matched else leaf_name
    if parent_parts:
        return f"{parent_parts[-1]}/{cleaned_leaf}"
    return cleaned_leaf


def _build_metadata_editor_rows(previews: list) -> pd.DataFrame:
    rows = []
    lookup: dict[str, str] = {}
    for preview in previews:
        display_name = _display_source_file_name(preview.file_name)
        lookup.setdefault(display_name, preview.file_name)
        for sheet_preview in preview.sheet_previews:
            waveform_value = sheet_preview.metadata.get("waveform")
            if waveform_value in (None, "", "0", "0.0"):
                waveform_value = infer_waveform_from_text(preview.file_name, sheet_preview.sheet_name)
            freq_value = sheet_preview.metadata.get("frequency(Hz)")
            if freq_value in (None, "", "0", "0.0", "0.000"):
                freq_value = infer_frequency_from_text(preview.file_name, sheet_preview.sheet_name)
            target_current = sheet_preview.metadata.get("Target Current(A)")
            target_current_value = first_number(target_current)
            if target_current_value is None or target_current_value <= 0:
                target_current_value = infer_current_from_text(preview.file_name, sheet_preview.sheet_name)
            rows.append(
                {
                    "source_file": display_name,
                    "sheet_name": sheet_preview.sheet_name,
                    "waveform_type": waveform_value,
                    "freq_hz": first_number(freq_value),
                    "target_current_a": target_current_value,
                    "notes": sheet_preview.metadata.get("notes", ""),
                }
            )
    st.session_state["_source_file_display_lookup"] = lookup
    return pd.DataFrame(rows)


def _group_metadata_overrides(edited_metadata: pd.DataFrame) -> dict[str, dict[str, dict[str, object]]]:
    overrides: dict[str, dict[str, dict[str, object]]] = {}
    if edited_metadata.empty:
        return overrides
    lookup = st.session_state.get("_source_file_display_lookup", {})
    for row in edited_metadata.to_dict(orient="records"):
        display_name = str(row.get("source_file") or "")
        source_file = str(lookup.get(display_name) or display_name)
        file_overrides = overrides.setdefault(source_file, {})
        file_overrides[str(row["sheet_name"])] = {
            "waveform": row.get("waveform_type"),
            "frequency(Hz)": row.get("freq_hz"),
            "Target Current(A)": row.get("target_current_a"),
            "notes": row.get("notes"),
        }
    return overrides


def _render_mapping_editor(schema, previews: list) -> dict[str, str | None]:
    column_pool: list[str] = []
    for preview in previews:
        for sheet_preview in preview.sheet_previews:
            column_pool.extend(sheet_preview.columns)
    options = [""] + sorted(dict.fromkeys(column_pool))
    default_mapping = previews[0].sheet_previews[0].recommended_mapping if previews and previews[0].sheet_previews else {}

    with st.expander("컬럼 매핑 조정", expanded=False):
        overrides: dict[str, str | None] = {}
        columns = st.columns(2)
        for index, (field_key, spec) in enumerate(schema.field_specs.items()):
            target_column = columns[index % 2]
            with target_column:
                default_value = default_mapping.get(field_key) or ""
                selected = st.selectbox(
                    f"{spec.label_ko} ({field_key})",
                    options=options,
                    index=options.index(default_value) if default_value in options else 0,
                    key=f"mapping_{field_key}",
                )
                overrides[field_key] = selected or None
        return overrides


def _build_validation_candidate_summaries(
    base_profile: pd.DataFrame,
    validation_measurements: list | None,
    validation_preprocess_results: list | None,
    target_output_type: str,
    current_channel: str,
    field_channel: str,
) -> list[dict[str, object]]:
    validation_measurements = validation_measurements or []
    validation_preprocess_results = validation_preprocess_results or []
    if not validation_measurements or not validation_preprocess_results:
        return []

    base_waveform = None
    if not base_profile.empty and "waveform_type" in base_profile.columns:
        base_waveform = canonicalize_waveform_type(base_profile["waveform_type"].iloc[0])
    base_freq_hz = _first_numeric_frame_value(base_profile, "freq_hz")
    base_target_output_pp = _first_numeric_frame_value(base_profile, "target_output_pp")
    if base_target_output_pp is None:
        target_column = _resolve_target_output_column(base_profile)
        if target_column:
            base_target_output_pp = _signal_peak_to_peak(base_profile, target_column, mask_column="is_active_target")

    output_column = field_channel if target_output_type == "field" else current_channel
    summaries: list[dict[str, object]] = []
    for parsed, preprocess in zip(validation_measurements, validation_preprocess_results):
        normalized = parsed.normalized_frame if hasattr(parsed, "normalized_frame") else pd.DataFrame()
        corrected = preprocess.corrected_frame if hasattr(preprocess, "corrected_frame") else pd.DataFrame()
        display_source_file = _display_source_file_name(parsed.source_file)
        test_id = (
            str(normalized["test_id"].iloc[0])
            if not normalized.empty and "test_id" in normalized.columns
            else f"{display_source_file}/{parsed.sheet_name}"
        )

        waveform_value = canonicalize_waveform_type(parsed.metadata.get("waveform"))
        if waveform_value is None and not normalized.empty and "waveform_type" in normalized.columns:
            non_null_waveforms = normalized["waveform_type"].dropna()
            if not non_null_waveforms.empty:
                waveform_value = canonicalize_waveform_type(non_null_waveforms.iloc[0])

        freq_hz = _first_numeric_frame_value(normalized, "freq_hz")
        if freq_hz is None:
            for key in ("frequency(Hz)", "freq_hz", "frequency", "freq"):
                candidate_freq = first_number(str(parsed.metadata.get(key, "")))
                if candidate_freq is not None:
                    freq_hz = float(candidate_freq)
                    break

        output_pp = _signal_peak_to_peak(corrected, output_column)
        waveform_match = not base_waveform or not waveform_value or waveform_value == base_waveform
        freq_relative_error = (
            abs(float(freq_hz) - float(base_freq_hz)) / max(abs(float(base_freq_hz)), 1e-9)
            if freq_hz is not None and base_freq_hz is not None and _compiled.np.isfinite(freq_hz) and _compiled.np.isfinite(base_freq_hz)
            else float("nan")
        )
        output_relative_error = (
            abs(float(output_pp) - float(base_target_output_pp)) / max(abs(float(base_target_output_pp)), 1e-9)
            if _compiled.np.isfinite(output_pp) and base_target_output_pp is not None and _compiled.np.isfinite(base_target_output_pp)
            else float("nan")
        )

        eligibility_reasons: list[str] = []
        if not waveform_match:
            eligibility_reasons.append("waveform mismatch")
        if _compiled.np.isfinite(freq_relative_error) and freq_relative_error > 0.25:
            eligibility_reasons.append("frequency too far")
        if not _compiled.np.isfinite(output_pp):
            eligibility_reasons.append(f"{output_column} pp unavailable")
        eligible = not eligibility_reasons
        score = (
            (0.0 if waveform_match else 5.0)
            + (float(freq_relative_error) if _compiled.np.isfinite(freq_relative_error) else 1.0)
            + (float(output_relative_error) if _compiled.np.isfinite(output_relative_error) else 1.0)
        )
        if not eligible:
            score = float("inf")

        summaries.append(
            {
                "label": f"{test_id} | {display_source_file}",
                "test_id": test_id,
                "source_file": display_source_file,
                "waveform_type": waveform_value,
                "freq_hz": float(freq_hz) if freq_hz is not None and _compiled.np.isfinite(freq_hz) else float("nan"),
                "output_pp": float(output_pp) if _compiled.np.isfinite(output_pp) else float("nan"),
                "freq_relative_error": float(freq_relative_error) if _compiled.np.isfinite(freq_relative_error) else float("nan"),
                "output_relative_error": float(output_relative_error) if _compiled.np.isfinite(output_relative_error) else float("nan"),
                "score": float(score),
                "eligible": bool(eligible),
                "eligibility_reason": ", ".join(eligibility_reasons) if eligibility_reasons else "ok",
                "parsed": parsed,
                "preprocess": preprocess,
            }
        )

    summaries.sort(
        key=lambda item: (
            0 if item["eligible"] else 1,
            float(item["score"]) if _compiled.np.isfinite(item["score"]) else float("inf"),
            float(item["freq_relative_error"]) if _compiled.np.isfinite(item["freq_relative_error"]) else float("inf"),
            str(item["test_id"]),
        )
    )
    return summaries


for _patched_name in (
    "_build_metadata_editor_rows",
    "_group_metadata_overrides",
    "_render_mapping_editor",
    "_build_validation_candidate_summaries",
    "_display_source_file_name",
):
    _compiled.__dict__[_patched_name] = globals()[_patched_name]
