from __future__ import annotations

from .validation_retune_shared import *

def _copy_frame_with_attrs(frame: pd.DataFrame) -> pd.DataFrame:
    copied = frame.copy()
    copied.attrs = dict(getattr(frame, "attrs", {}))
    return copied


def _prediction_objective_audit(
    *,
    target_output_type: str,
    harmonic_weights_used: dict[str, Any],
) -> dict[str, Any]:
    if str(target_output_type) == "field":
        return {
            "loss_target_type": "field",
            "w_bz_nrmse": 1.0,
            "w_bz_shape": 1.0,
            "w_bz_phase": 1.0,
            "w_bz_pp": 1.0,
            "w_current_limit": 0.15,
            "w_voltage_limit": 0.15,
            "harmonic_weights_used": harmonic_weights_used,
            "objective_weight_source": "route_inferred",
        }
    return {
        "loss_target_type": "current",
        "w_bz_nrmse": 0.0,
        "w_bz_shape": 0.0,
        "w_bz_phase": 0.0,
        "w_bz_pp": 0.0,
        "w_current_limit": 1.0,
        "w_voltage_limit": 0.25,
        "harmonic_weights_used": harmonic_weights_used,
        "objective_weight_source": "route_inferred",
    }


def _detect_target_leak_suspect(
    *,
    command_profile: pd.DataFrame,
    target_output_type: str,
    request_route: str,
    solver_route: str,
) -> tuple[bool, str | None, float | None]:
    if str(request_route) != "exact" or str(target_output_type) != "current" or str(solver_route) != "finite_exact_direct":
        return False, None, None
    try:
        target_column, _ = _resolve_target_output_column(command_profile)
        expected_column = _resolve_expected_output_column(command_profile)
    except KeyError:
        return False, None, None
    target_output = pd.to_numeric(command_profile[target_column], errors="coerce").to_numpy(dtype=float)
    expected_output = pd.to_numeric(command_profile[expected_column], errors="coerce").to_numpy(dtype=float)
    if target_output.size == 0 or expected_output.size == 0:
        return False, None, None
    leak_corr = _correlation(target_output, expected_output)
    target_pp = _peak_to_peak(target_output)
    expected_pp = _peak_to_peak(expected_output)
    pp_ratio = (
        float(expected_pp / target_pp)
        if np.isfinite(expected_pp) and np.isfinite(target_pp) and float(target_pp) > 1e-9
        else float("nan")
    )
    if np.isfinite(leak_corr) and leak_corr >= 0.999 and np.isfinite(pp_ratio) and 0.95 <= pp_ratio <= 1.05:
        return True, "expected_output_matches_target_template", float(leak_corr)
    return False, None, float(leak_corr) if np.isfinite(leak_corr) else None


def build_prediction_debug_snapshot(
    *,
    command_profile: pd.DataFrame,
    validation_frame: pd.DataFrame | None,
    target_output_type: str,
    current_channel: str,
    field_channel: str,
) -> dict[str, Any]:
    return build_prediction_debug_from_profile(
        command_profile=command_profile,
        validation_frame=validation_frame,
        target_output_type=target_output_type,
        current_channel=current_channel,
        field_channel=field_channel,
    )



__all__ = [name for name in globals() if not name.startswith('__')]

