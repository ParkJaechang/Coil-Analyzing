from __future__ import annotations

from .validation_retune_shared import *
from .validation_retune_provenance import *
from .validation_retune_alignment import *
from .validation_retune_metric_utils import *

def build_validation_comparison(
    *,
    command_profile: pd.DataFrame,
    validation_frame: pd.DataFrame,
    label: str,
    comparison_source: str,
    target_output_type: str,
    current_channel: str,
    field_channel: str,
    metric_domain: str = TARGET_OUTPUT_DOMAIN,
) -> ValidationComparison:
    if metric_domain == BZ_EFFECTIVE_DOMAIN:
        output_column = field_channel
        target_column, target_basis = _resolve_bz_target_column(command_profile)
        expected_column = _resolve_bz_expected_column(command_profile)
    else:
        output_column = field_channel if str(target_output_type) == "field" else current_channel
        target_column, target_basis = _resolve_target_output_column(command_profile)
        expected_column = _resolve_expected_output_column(command_profile)

    fit_end_s = _infer_fit_end_s(command_profile)
    validation_end_s = _infer_fit_end_s(validation_frame)
    end_s = min(fit_end_s, validation_end_s) if np.isfinite(validation_end_s) and validation_end_s > 0 else fit_end_s
    if not np.isfinite(end_s) or end_s <= 0:
        end_s = max(_safe_float(command_profile.get("time_s", pd.Series(dtype=float)).max()), 1.0)
    sample_count = max(256, min(len(command_profile), len(validation_frame)) * 4 if len(validation_frame) else len(command_profile) * 4)
    time_grid = np.linspace(0.0, float(end_s), max(int(sample_count), 256))

    target_output = _interpolate_column(command_profile, target_column, time_grid)
    comparison_output = (
        _interpolate_column(validation_frame, output_column, time_grid)
        if comparison_source == "actual"
        else _interpolate_column(command_profile, expected_column, time_grid)
    )
    error = comparison_output - target_output
    finite_error = error[np.isfinite(error)]
    target_pp = _peak_to_peak(target_output)
    rmse = float(np.sqrt(np.nanmean(np.square(finite_error)))) if finite_error.size else float("nan")
    denom = max(target_pp / 2.0, 1e-12) if np.isfinite(target_pp) and target_pp > 0 else float("nan")
    nrmse = rmse / denom if np.isfinite(denom) else float("nan")
    pp_error = _peak_to_peak(comparison_output) - target_pp if np.isfinite(target_pp) else float("nan")
    peak_error = float(np.nanmax(np.abs(finite_error))) if finite_error.size else float("nan")
    shape_corr = _correlation(target_output, comparison_output)
    phase_lag_s = _estimate_phase_lag_seconds(target_output, comparison_output, time_grid)
    clipping_detected = _detect_clipping(validation_frame, output_column) or _detect_hardware_gate_violation(command_profile)
    saturation_detected = _detect_clipping(validation_frame, "daq_input_v")
    metrics_available, unavailable_reason, reason_codes, valid_sample_count = _resolve_metric_status(
        metric_domain=metric_domain,
        target_basis=target_basis,
        comparison_source=comparison_source,
        output_column=output_column,
        validation_frame=validation_frame,
        target_output=target_output,
        comparison_output=comparison_output,
        nrmse=nrmse,
        shape_corr=shape_corr,
        phase_lag_s=phase_lag_s,
        clipping_detected=bool(clipping_detected),
        saturation_detected=bool(saturation_detected),
    )



    window_info = validation_frame.attrs.get("validation_window", {})

    return ValidationComparison(
        label=label,
        output_column=output_column,
        rmse=rmse,
        nrmse=nrmse,
        shape_corr=shape_corr,
        phase_lag_s=phase_lag_s,
        pp_error=pp_error,
        peak_error=peak_error,
        clipping_detected=bool(clipping_detected),
        saturation_detected=bool(saturation_detected),
        metric_domain=metric_domain,
        target_basis=target_basis,
        comparison_source=comparison_source,
        sample_count=int(len(time_grid)),
        fit_end_s=float(end_s),
        metrics_available=bool(metrics_available),
        unavailable_reason=unavailable_reason,
        reason_codes=reason_codes,
        valid_sample_count=valid_sample_count,
        active_window_start_s=_safe_float(window_info.get("start_s")),
        active_window_end_s=_safe_float(window_info.get("end_s")),
    )


def build_validation_overlay_frame(
    *,
    base_profile: pd.DataFrame,
    validation_frame: pd.DataFrame,
    corrected_profile: pd.DataFrame,
    target_output_type: str,
    current_channel: str,
    field_channel: str,
) -> pd.DataFrame:
    output_column = field_channel if str(target_output_type) == "field" else current_channel
    target_column, _ = _resolve_target_output_column(base_profile)
    predicted_column = _resolve_expected_output_column(base_profile)
    corrected_column = _resolve_expected_output_column(corrected_profile)
    fit_end_s = _infer_fit_end_s(base_profile)
    validation_end_s = _infer_fit_end_s(validation_frame)
    corrected_end_s = _infer_fit_end_s(corrected_profile)
    finite_candidates = [value for value in (fit_end_s, validation_end_s, corrected_end_s) if np.isfinite(value) and value > 0]
    end_s = min(finite_candidates) if finite_candidates else 1.0
    sample_count = max(256, min(len(base_profile), len(validation_frame), len(corrected_profile)) * 4 if len(validation_frame) and len(corrected_profile) else 256)
    time_grid = np.linspace(0.0, float(end_s), max(int(sample_count), 256))
    target_bz_column, _ = _resolve_bz_target_column(base_profile)
    predicted_bz_column = _resolve_bz_expected_column(base_profile)
    corrected_bz_column = _resolve_bz_expected_column(corrected_profile)
    return pd.DataFrame(
        {
            "time_s": time_grid,
            "target_output": _interpolate_column(base_profile, target_column, time_grid),
            "predicted_output": _interpolate_column(base_profile, predicted_column, time_grid),
            "actual_measured": _interpolate_column(validation_frame, output_column, time_grid),
            "corrected_prediction": _interpolate_column(corrected_profile, corrected_column, time_grid),
            "target_bz_effective": _interpolate_column(base_profile, target_bz_column, time_grid),
            "predicted_bz_effective": _interpolate_column(base_profile, predicted_bz_column, time_grid),
            "actual_bz_effective": _interpolate_column(validation_frame, field_channel, time_grid),
            "corrected_bz_effective": _interpolate_column(corrected_profile, corrected_bz_column, time_grid),
        }
    )



__all__ = [name for name in globals() if not name.startswith('__')]

