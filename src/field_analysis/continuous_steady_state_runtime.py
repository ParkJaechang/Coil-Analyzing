from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from .continuous_steady_state_extraction import (
    build_continuous_phase_aligned_command_profile,
    build_continuous_steady_state_modeling_case,
)

ModelingCaseBuilder = Callable[..., dict[str, Any]]


def run_continuous_steady_state_extraction(
    *,
    selected_candidate_name: str | None,
    selected_frame: pd.DataFrame | None,
    waveform_type: str | None,
    freq_hz: float | None,
    modeling_case_builder: ModelingCaseBuilder = build_continuous_steady_state_modeling_case,
) -> dict[str, Any]:
    if not selected_candidate_name or not isinstance(selected_frame, pd.DataFrame) or selected_frame.empty:
        return {"status": "error", "error_reason": "no_continuous_source_dataset"}
    try:
        case = modeling_case_builder(
            selected_frame,
            waveform_type=str(waveform_type or "sine"),
            freq_hz=float(freq_hz or 1.0),
        )
    except Exception as exc:  # noqa: BLE001 - UI orchestration returns user-facing error reasons.
        return {"status": "error", "error_reason": f"extraction_failed:{exc}"}
    window = case.get("steady_state_one_cycle_frame")
    metadata = dict(case.get("metadata") or {})
    if metadata.get("steady_state_extraction_status") != "ok":
        return {"status": "error", "error_reason": str(metadata.get("steady_state_extraction_status")), "extraction_result": case}
    if not isinstance(window, pd.DataFrame) or window.empty:
        metadata["steady_state_extraction_status"] = "extraction_result_empty"
        case["metadata"] = metadata
        return {"status": "error", "error_reason": "extraction_result_empty", "extraction_result": case}
    if metadata.get("selected_cycle_duration_status") not in (None, "ok"):
        return {"status": "error", "error_reason": str(metadata.get("selected_cycle_duration_status")), "extraction_result": case}
    return {
        "status": "ok",
        "extraction_result": case,
        "steady_window_frame": window,
        "extraction_metadata": metadata,
        "selected_candidate": selected_candidate_name,
    }


def run_continuous_first_modeling(
    *,
    extraction_result: dict[str, Any] | None,
    waveform_type: str | None,
    freq_hz: float | None,
    base_voltage_peak_v: float | None = None,
    target_peak_field_mT: float | None = None,
) -> dict[str, Any]:
    if not isinstance(extraction_result, dict):
        return {"status": "error", "error_reason": "missing_extraction_result"}
    metadata = dict(extraction_result.get("metadata") or {})
    if metadata.get("steady_state_extraction_status") != "ok":
        return {"status": "error", "error_reason": str(metadata.get("steady_state_extraction_status"))}
    if metadata.get("selected_cycle_duration_status") not in (None, "ok"):
        return {"status": "error", "error_reason": str(metadata.get("selected_cycle_duration_status"))}
    window = extraction_result.get("steady_state_one_cycle_frame")
    support = extraction_result.get("steady_state_support_frame")
    if not isinstance(window, pd.DataFrame) or window.empty:
        return {"status": "error", "error_reason": "extraction_result_empty"}
    if target_peak_field_mT is None:
        target_peak_field_mT = metadata.get("user_target_peak_field_mT") or metadata.get("target_peak_mT") or 50.0
    window = _scale_continuous_target_fields(window, target_peak_mT=float(target_peak_field_mT))
    if isinstance(support, pd.DataFrame) and not support.empty:
        support = _scale_continuous_target_fields(support, target_peak_mT=float(target_peak_field_mT))
    try:
        command_profile, model_metadata = build_continuous_phase_aligned_command_profile(
            window,
            support_frame=support if isinstance(support, pd.DataFrame) else None,
            freq_hz=float(freq_hz or 1.0),
            waveform_type=str(waveform_type) if waveform_type is not None else None,
            base_voltage_peak_v=base_voltage_peak_v,
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_reason": f"modeling_kernel_failed:{exc}"}
    if not isinstance(command_profile, pd.DataFrame) or command_profile.empty:
        return {"status": "error", "error_reason": "command_profile_empty"}
    model_metadata = dict(model_metadata)
    model_metadata.update(
        {
            "continuous_first_modeling_status": "ok",
            "continuous_first_modeling_input_valid": True,
            "continuous_first_modeling_tail_disabled": True,
            "continuous_first_modeling_cycle_count": 1.0,
            "continuous_loop_output": True,
        }
    )
    return {
        "status": "ok",
        "first_model_result": {
            "command_profile": command_profile.copy(deep=True),
            "metadata": model_metadata,
        },
        "command_profile": command_profile,
        "first_model_metadata": model_metadata,
    }


def _scale_continuous_target_fields(frame: pd.DataFrame, *, target_peak_mT: float) -> pd.DataFrame:
    out = frame.copy(deep=True)
    peak = abs(float(target_peak_mT)) if pd.notna(target_peak_mT) else 50.0
    if peak <= 1e-12:
        peak = 50.0
    for column in ("normalized_physical_target_output_mT", "physical_target_output_mT", "target_field_mT"):
        if column in out.columns:
            values = pd.to_numeric(out[column], errors="coerce")
            source_peak = float(values.abs().max()) if values.notna().any() else 0.0
            if source_peak > 1e-12:
                out[column] = values * (peak / source_peak)
    out["user_target_peak_field_mT"] = peak
    out["field_modeling_normalization_reference_mT"] = peak
    return out
