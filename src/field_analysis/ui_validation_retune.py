from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from .recommendation_constants import BZ_FIRST_ARTIFACT_DIR, PRODUCT_ROOT
from .ui_upload_state import build_upload_state_paths
from .validation_retune import execute_validation_retune, save_retune_artifacts
from .validation_retune_catalog import (
    build_corrected_lut_catalog_payload,
    build_retune_picker_payload,
    build_validation_catalog_payload,
    write_json,
)


APP_STATE_PATHS = build_upload_state_paths()
RETUNE_OUTPUT_DIR = APP_STATE_PATHS.app_state_dir / "validation_retune"
VALIDATION_REPORT_DIRS = [
    PRODUCT_ROOT / "artifacts" / "validation_retune_mvp_example",
    PRODUCT_ROOT / "artifacts" / "validation_retune_real_example",
    RETUNE_OUTPUT_DIR,
]
LUT_CATALOG_PATH = BZ_FIRST_ARTIFACT_DIR / "lut_catalog.json"
VALIDATION_CATALOG_PATH = BZ_FIRST_ARTIFACT_DIR / "validation_catalog.json"
CORRECTED_CATALOG_PATH = BZ_FIRST_ARTIFACT_DIR / "corrected_lut_catalog.json"
RETUNE_PICKER_PATH = BZ_FIRST_ARTIFACT_DIR / "retune_picker_catalog.json"
DIAGNOSTIC_ARTIFACTS = {
    "shape_engine_audit": BZ_FIRST_ARTIFACT_DIR / "shape_engine_audit.json",
    "amplitude_lut_audit": BZ_FIRST_ARTIFACT_DIR / "amplitude_lut_audit.json",
    "same_freq_level_sensitivity": BZ_FIRST_ARTIFACT_DIR / "same_freq_level_sensitivity.json",
    "support_route_level_influence": BZ_FIRST_ARTIFACT_DIR / "support_route_level_influence.json",
    "real_validation_suite_result": BZ_FIRST_ARTIFACT_DIR / "real_validation_suite_result.json",
    "field_prediction_debug_report": BZ_FIRST_ARTIFACT_DIR / "field_prediction_debug_report.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(resolved, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(resolved)


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _signal_peak_to_peak(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.nanmax(finite) - np.nanmin(finite))


def _frame_text(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame.columns:
        return None
    series = frame[column].dropna()
    if series.empty:
        return None
    return str(series.iloc[0])


def _frame_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[0])


def _target_output_pp(base_profile: pd.DataFrame, target_output_type: str) -> float | None:
    target_field = _frame_numeric(base_profile, "target_field_mT")
    target_current = _frame_numeric(base_profile, "target_current_a")
    target_output = _frame_numeric(base_profile, "target_output_pp")
    if target_output is not None:
        return target_output
    if target_output_type == "field":
        return target_field
    return target_current


def build_validation_candidate_summaries(
    *,
    base_profile: pd.DataFrame,
    validation_measurements: list,
    validation_preprocess_results: list,
    target_output_type: str,
    current_channel: str,
    field_channel: str,
) -> list[dict[str, Any]]:
    source_waveform = _frame_text(base_profile, "waveform_type")
    source_freq = _frame_numeric(base_profile, "freq_hz")
    source_target_pp = _target_output_pp(base_profile, target_output_type)
    output_column = field_channel if target_output_type == "field" else current_channel
    rows: list[dict[str, Any]] = []

    for parsed, preprocess in zip(validation_measurements or [], validation_preprocess_results or [], strict=False):
        corrected_frame = getattr(preprocess, "corrected_frame", pd.DataFrame())
        normalized_frame = getattr(parsed, "normalized_frame", pd.DataFrame())
        summary_frame = corrected_frame if not corrected_frame.empty else normalized_frame
        test_id = (
            str(normalized_frame["test_id"].iloc[0])
            if not normalized_frame.empty and "test_id" in normalized_frame.columns
            else f"{Path(str(parsed.source_file)).stem}/{parsed.sheet_name}"
        )
        waveform_type = _frame_text(normalized_frame, "waveform_type") or str(parsed.metadata.get("waveform") or "")
        freq_hz = _frame_numeric(normalized_frame, "freq_hz")
        output_pp = _signal_peak_to_peak(summary_frame, output_column)
        freq_rel = (
            abs(float(freq_hz) - float(source_freq)) / max(abs(float(source_freq)), 1e-9)
            if freq_hz is not None and source_freq is not None
            else None
        )
        output_rel = (
            abs(float(output_pp) - float(source_target_pp)) / max(abs(float(source_target_pp)), 1e-9)
            if output_pp is not None and source_target_pp is not None
            else None
        )
        waveform_match = not source_waveform or not waveform_type or waveform_type == source_waveform
        eligible = bool(waveform_match and (freq_rel is None or freq_rel <= 0.25))
        score = float((freq_rel or 0.0) + (output_rel or 0.0) + (0.0 if waveform_match else 5.0))
        rows.append(
            {
                "label": f"{test_id} | {Path(str(parsed.source_file)).name}",
                "test_id": test_id,
                "source_file": str(parsed.source_file),
                "waveform_type": waveform_type,
                "freq_hz": freq_hz,
                "output_pp": output_pp,
                "freq_relative_error": freq_rel,
                "output_relative_error": output_rel,
                "score": score,
                "eligible": eligible,
                "eligibility_reason": "ok" if eligible else "waveform/frequency mismatch",
                "parsed": parsed,
                "preprocess": preprocess,
                "frame": summary_frame,
            }
        )

    return sorted(rows, key=lambda item: (0 if item["eligible"] else 1, float(item["score"]), str(item["test_id"])))


def _build_history_entry(retune_result: Any, artifact_paths: dict[str, str]) -> dict[str, Any]:
    return {
        "retune_id": retune_result.validation_run.export_file_prefix,
        "created_at": retune_result.validation_run.created_at,
        "original_recommendation_id": retune_result.validation_run.original_recommendation_id,
        "validation_run_id": retune_result.validation_run.validation_run_id,
        "validation_test_id": retune_result.validation_run.selected_validation_test_id,
        "lut_id": retune_result.validation_run.lut_id,
        "source_kind": retune_result.validation_run.source_kind,
        "source_lut_filename": retune_result.validation_run.source_lut_filename,
        "corrected_lut_id": retune_result.validation_run.corrected_lut_id,
        "iteration_index": retune_result.validation_run.iteration_index,
        "exact_path": retune_result.validation_run.exact_path,
        "correction_rule": retune_result.validation_run.correction_rule,
        "target_output_type": retune_result.validation_run.target_output_type,
        "waveform_type": retune_result.validation_run.waveform_type,
        "freq_hz": retune_result.validation_run.freq_hz,
        "commanded_cycles": retune_result.validation_run.commanded_cycles,
        "target_level_value": retune_result.validation_run.target_level_value,
        "target_level_kind": retune_result.validation_run.target_level_kind,
        "quality_label": retune_result.quality_label,
        "quality_tone": retune_result.quality_tone,
        "quality_reasons": retune_result.quality_reasons,
        "acceptance_decision": retune_result.acceptance_decision,
        "preferred_output_id": retune_result.preferred_output_id,
        "candidate_status": retune_result.acceptance_decision.get("decision"),
        "candidate_status_label": retune_result.acceptance_decision.get("label"),
        "rejection_reason": retune_result.acceptance_decision.get("rejection_reason"),
        "artifact_paths": artifact_paths,
        "before_nrmse": retune_result.baseline_comparison.nrmse,
        "after_nrmse": retune_result.corrected_comparison.nrmse,
        "before_shape_corr": retune_result.baseline_comparison.shape_corr,
        "after_shape_corr": retune_result.corrected_comparison.shape_corr,
        "before_bz_nrmse": retune_result.baseline_bz_comparison.nrmse,
        "after_bz_nrmse": retune_result.corrected_bz_comparison.nrmse,
        "before_bz_shape_corr": retune_result.baseline_bz_comparison.shape_corr,
        "after_bz_shape_corr": retune_result.corrected_bz_comparison.shape_corr,
    }


def _record_retune_history(entry: dict[str, Any]) -> None:
    payload = _load_json(APP_STATE_PATHS.validation_retune_history_path)
    rows = payload.get("retunes", [])
    if not isinstance(rows, list):
        rows = []
    corrected_id = str(entry.get("corrected_lut_id") or entry.get("retune_id") or "")
    rows = [item for item in rows if str(item.get("corrected_lut_id") or item.get("retune_id") or "") != corrected_id]
    rows.append(entry)
    rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    write_json(APP_STATE_PATHS.validation_retune_history_path, {"retunes": rows})


def refresh_catalog_artifacts() -> dict[str, Any]:
    validation_catalog = build_validation_catalog_payload(
        report_dirs=VALIDATION_REPORT_DIRS,
        history_path=APP_STATE_PATHS.validation_retune_history_path,
    )
    corrected_catalog = build_corrected_lut_catalog_payload(validation_catalog.get("entries", []))
    lut_catalog = _load_json(LUT_CATALOG_PATH)
    picker_catalog = build_retune_picker_payload(
        lut_entries=list(lut_catalog.get("entries", [])),
        validation_entries=list(validation_catalog.get("entries", [])),
        corrected_entries=list(corrected_catalog.get("entries", [])),
    )
    write_json(VALIDATION_CATALOG_PATH, validation_catalog)
    write_json(CORRECTED_CATALOG_PATH, corrected_catalog)
    write_json(RETUNE_PICKER_PATH, picker_catalog)
    return {
        "validation_catalog": validation_catalog,
        "corrected_catalog": corrected_catalog,
        "picker_catalog": picker_catalog,
    }


def _load_catalog_bundle() -> dict[str, Any]:
    return {
        "lut_catalog": _load_json(LUT_CATALOG_PATH),
        "validation_catalog": _load_json(VALIDATION_CATALOG_PATH),
        "corrected_catalog": _load_json(CORRECTED_CATALOG_PATH),
        "picker_catalog": _load_json(RETUNE_PICKER_PATH),
    }


def _metrics_table(payload: dict[str, Any], key: str) -> pd.DataFrame:
    metrics = payload.get(key, {})
    rows = []
    for domain, values in metrics.items():
        rows.append(
            {
                "domain": domain,
                "nrmse": values.get("nrmse"),
                "shape_corr": values.get("shape_corr"),
                "phase_lag_s": values.get("phase_lag_s"),
                "clipping_detected": values.get("clipping_detected"),
                "metrics_available": values.get("metrics_available"),
                "unavailable_reason": values.get("unavailable_reason"),
            }
        )
    return pd.DataFrame(rows)


def render_validation_retune_section(
    *,
    current_channel: str,
    field_channel: str,
    max_daq_voltage_pp: float,
    amp_gain_at_100_pct: float,
    amp_gain_limit_pct: float,
    amp_max_output_pk_v: float,
    default_support_amp_gain_pct: float,
    validation_measurements: list,
    validation_preprocess_results: list,
) -> None:
    if st.button("Catalog 새로고침", key="retune_catalog_refresh"):
        refresh_catalog_artifacts()
        st.rerun()

    catalogs = _load_catalog_bundle()
    picker_entries = list(catalogs["picker_catalog"].get("entries", []))
    validation_entries = list(catalogs["validation_catalog"].get("entries", []))
    corrected_entries = list(catalogs["corrected_catalog"].get("entries", []))

    a, b, c = st.columns(3)
    a.metric("Picker Sources", len(picker_entries))
    b.metric("Validation Runs", len(validation_entries))
    c.metric("Corrected LUTs", len(corrected_entries))

    show_ineligible = st.checkbox("non-exact source 포함", value=False, key="retune_show_ineligible")
    source_entries = [item for item in picker_entries if show_ineligible or bool(item.get("retune_eligible"))]
    if not source_entries:
        if picker_entries:
            hidden_ineligible_count = sum(1 for item in picker_entries if not bool(item.get("retune_eligible")))
            st.info("Retune picker catalog is loaded, but there is no retune-eligible exact source yet.")
            if not show_ineligible and hidden_ineligible_count:
                st.caption(
                    f"{hidden_ineligible_count} source(s) are present but hidden by the current filter. "
                    "Enable `non-exact source 포함` to inspect them. Retune itself still requires an exact source."
                )
            else:
                st.caption(
                    "Catalog entries exist, but none of them currently qualify as an exact retune source. "
                    "Generate or refresh exact LUT artifacts, then reload the catalog."
                )
        else:
            st.info("No retune picker sources are available yet. This is normal in a clean repo.")
            st.caption(
                "Generate exact LUT/export artifacts first, then click `Catalog 새로고침`. "
                "Validation uploads alone do not create picker sources."
            )
        return

    source_labels = {item["display_label"]: item for item in source_entries}
    selected_source = source_labels[st.selectbox("Retune Source", options=list(source_labels.keys()), key="retune_source_select")]
    profile_path = selected_source.get("profile_csv_path") or selected_source.get("source_profile_path")
    if not profile_path or not Path(str(profile_path)).exists():
        missing_profile = str(profile_path or "(missing profile path)")
        st.error(f"Selected source is missing its waveform profile CSV: `{missing_profile}`")
        st.caption(
            "This is an artifact-path problem rather than a normal empty state. "
            "Refresh the catalog or regenerate the source artifact."
        )
        return

    base_profile = _read_csv(profile_path)
    target_output_type = str(selected_source.get("target_output_type") or "current")
    validation_candidates = build_validation_candidate_summaries(
        base_profile=base_profile,
        validation_measurements=validation_measurements,
        validation_preprocess_results=validation_preprocess_results,
        target_output_type=target_output_type,
        current_channel=current_channel,
        field_channel=field_channel,
    )
    if not validation_candidates:
        if validation_measurements:
            st.warning("Validation files are loaded, but no candidate summary could be built from them.")
            st.caption(
                "Check `Data Import` and confirm waveform/frequency metadata plus target columns were parsed correctly."
            )
        else:
            st.info("No validation run files are loaded yet, so there is nothing to retune against.")
            st.caption(
                "Upload validation run files in the sidebar to choose a measured run for comparison."
            )
    else:
        candidate_labels = {item["label"]: item for item in validation_candidates}
        selected_candidate = candidate_labels[st.selectbox("Validation Run", options=list(candidate_labels.keys()), key="retune_validation_select")]
        settings_left, settings_mid, settings_right = st.columns(3)
        with settings_left:
            correction_gain = float(st.number_input("correction_gain", min_value=0.05, max_value=1.0, value=0.7, step=0.05, key="retune_gain"))
        with settings_mid:
            max_iterations = int(st.number_input("max_iterations", min_value=1, max_value=5, value=2, step=1, key="retune_iterations"))
        with settings_right:
            improvement_threshold = float(st.number_input("improvement_threshold", min_value=-1.0, max_value=1.0, value=0.0, step=0.01, key="retune_threshold"))

        if st.button("Validation / Retune 실행", use_container_width=True, key="retune_execute"):
            prefix = f"{selected_source['source_id']}_ui_validation"
            retune_result = execute_validation_retune(
                base_profile=base_profile,
                validation_candidate=selected_candidate,
                validation_frame=selected_candidate["frame"],
                export_file_prefix=prefix,
                target_output_type=target_output_type,
                current_channel=current_channel,
                field_channel=field_channel,
                max_daq_voltage_pp=max_daq_voltage_pp,
                amp_gain_at_100_pct=amp_gain_at_100_pct,
                amp_gain_limit_pct=amp_gain_limit_pct,
                amp_max_output_pk_v=amp_max_output_pk_v,
                support_amp_gain_pct=float(_frame_numeric(base_profile, "support_amp_gain_pct") or default_support_amp_gain_pct),
                correction_gain=correction_gain,
                max_iterations=max_iterations,
                improvement_threshold=improvement_threshold,
                original_recommendation_id=str(selected_source.get("original_recommendation_id") or selected_source.get("source_id")),
                source_selection=selected_source,
            )
            if retune_result is None:
                st.error("validation retune execution returned None")
            else:
                artifact_paths = save_retune_artifacts(retune_result=retune_result, output_dir=RETUNE_OUTPUT_DIR)
                _record_retune_history(_build_history_entry(retune_result, artifact_paths))
                refresh_catalog_artifacts()
                st.session_state["retune_last_payload"] = retune_result.artifact_payload
                st.session_state["retune_last_artifact_paths"] = artifact_paths
                st.success(f"retune 완료: {retune_result.validation_run.corrected_lut_id}")
                st.rerun()

    st.markdown("#### 선택 source")
    st.json(
        {
            key: selected_source.get(key)
            for key in [
                "selection_id",
                "source_kind",
                "source_id",
                "retune_eligible",
                "status",
                "exact_path",
                "latest_validation_run_id",
                "latest_validation_quality_label",
                "latest_corrected_candidate_id",
                "profile_csv_path",
            ]
        }
    )

    last_payload = st.session_state.get("retune_last_payload")
    last_artifacts = st.session_state.get("retune_last_artifact_paths", {})
    if isinstance(last_payload, dict):
        st.markdown("#### 최신 retune 결과")
        top_left, top_mid, top_right = st.columns(3)
        top_left.metric("candidate_status", str(last_payload.get("candidate_status") or "-"))
        top_mid.metric("preferred_output_kind", str(last_payload.get("preferred_output_kind") or "-"))
        top_right.metric("rejection_reason", str(last_payload.get("rejection_reason") or "-"))
        st.dataframe(_metrics_table(last_payload, "baseline_metrics"), use_container_width=True)
        st.dataframe(_metrics_table(last_payload, "corrected_metrics"), use_container_width=True)
        st.json(last_payload.get("acceptance_decision", {}))
        st.json(last_payload.get("quality_badge", {}))
        if last_artifacts:
            st.dataframe(pd.DataFrame([last_artifacts]), use_container_width=True)


def render_catalogs_and_diagnostics_section() -> None:
    catalogs = _load_catalog_bundle()
    picker_entries = list(catalogs["picker_catalog"].get("entries", []))
    validation_entries = list(catalogs["validation_catalog"].get("entries", []))
    corrected_entries = list(catalogs["corrected_catalog"].get("entries", []))

    tab_catalogs, tab_diagnostics = st.tabs(["Catalogs", "Diagnostics"])
    with tab_catalogs:
        if not picker_entries and not validation_entries and not corrected_entries:
            st.info("No catalog artifacts are available yet. This is normal in a clean repo.")
            st.caption(
                "Catalog tables appear after LUT/export/validation-retune artifacts are generated and the catalog is refreshed."
            )
        if picker_entries:
            picker_frame = pd.DataFrame(picker_entries)[[
                "display_label",
                "source_kind",
                "retune_eligible",
                "status",
                "exact_path",
                "latest_validation_quality_label",
                "latest_corrected_candidate_id",
            ]]
            st.markdown("#### Retune Picker")
            st.dataframe(picker_frame, use_container_width=True)
        else:
            st.caption("Retune Picker catalog is empty.")
        if validation_entries:
            validation_frame = pd.DataFrame(validation_entries)[[
                "created_at",
                "corrected_lut_id",
                "quality_label",
                "candidate_status",
                "preferred_output_id",
                "rejection_reason",
                "report_path",
            ]]
            st.markdown("#### Validation Catalog")
            st.dataframe(validation_frame, use_container_width=True)
        else:
            st.caption("Validation Catalog is empty.")
        if corrected_entries:
            corrected_frame = pd.DataFrame(corrected_entries)[[
                "created_at",
                "corrected_lut_id",
                "quality_label",
                "candidate_status",
                "preferred_output_id",
                "rejection_reason",
            ]]
            st.markdown("#### Corrected LUT Catalog")
            st.dataframe(corrected_frame, use_container_width=True)
        else:
            st.caption("Corrected LUT Catalog is empty.")

    with tab_diagnostics:
        rows = []
        for label, path in DIAGNOSTIC_ARTIFACTS.items():
            payload = _load_json(path)
            if not payload:
                continue
            rows.append({"artifact": label, "path": path.as_posix(), "summary": payload.get("summary")})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No diagnostic artifacts are available yet.")
            st.caption(
                "This is expected until audit or validation-retune outputs have been generated."
            )
        for label, path in DIAGNOSTIC_ARTIFACTS.items():
            payload = _load_json(path)
            if not payload:
                continue
            with st.expander(label, expanded=False):
                st.json(payload.get("summary", payload))


__all__ = [name for name in globals() if not name.startswith("_")]
