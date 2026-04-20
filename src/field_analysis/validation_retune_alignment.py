from __future__ import annotations

from .validation_retune_shared import *
from .validation_retune_provenance import *

def _valid_time_signal(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    if frame.empty or "time_s" not in frame.columns or column not in frame.columns:
        return np.array([], dtype=float), np.array([], dtype=float)
    time_values = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    signal_values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time_values) & np.isfinite(signal_values)
    if valid.sum() < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    ordered = np.argsort(time_values[valid])
    return time_values[valid][ordered], signal_values[valid][ordered]


def _normalized_peak_ratio(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_pp = _peak_to_peak(reference)
    candidate_pp = _peak_to_peak(candidate)
    if not np.isfinite(reference_pp) or not np.isfinite(candidate_pp) or reference_pp <= 1e-9 or candidate_pp <= 1e-9:
        return float("nan")
    return float(min(candidate_pp / reference_pp, reference_pp / candidate_pp))


def _max_aligned_shape_score(reference: np.ndarray, candidate: np.ndarray) -> float:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if valid.sum() < 8:
        return float("nan")
    ref = reference[valid] - float(np.nanmean(reference[valid]))
    comp = candidate[valid] - float(np.nanmean(candidate[valid]))
    ref_std = float(np.nanstd(ref))
    comp_std = float(np.nanstd(comp))
    if ref_std <= 1e-12 or comp_std <= 1e-12:
        return float("nan")
    correlation = np.correlate(comp, ref, mode="full")
    denom = float(len(ref) * ref_std * comp_std)
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return float(np.clip(np.nanmax(correlation) / denom, -1.0, 1.0))


def _shift_signal_by_seconds(signal: np.ndarray, lag_s: float, time_grid: np.ndarray) -> np.ndarray:
    if len(signal) != len(time_grid):
        return signal
    if not np.isfinite(lag_s) or abs(float(lag_s)) <= 1e-12:
        return signal
    valid = np.isfinite(signal) & np.isfinite(time_grid)
    if valid.sum() < 2:
        return signal
    shifted = np.full_like(signal, np.nan, dtype=float)
    shifted_source_time = time_grid[valid] - float(lag_s)
    shifted[valid] = np.interp(
        shifted_source_time,
        time_grid[valid],
        signal[valid],
        left=0.0,
        right=0.0,
    )
    return shifted


def _is_signal_stable(values: np.ndarray, *, min_valid: int = 16, min_pp: float = 1e-6) -> bool:
    finite_values = values[np.isfinite(values)]
    if finite_values.size < min_valid:
        return False
    signal_pp = _peak_to_peak(finite_values)
    signal_std = float(np.nanstd(finite_values)) if finite_values.size else float("nan")
    return bool(np.isfinite(signal_pp) and signal_pp > min_pp and np.isfinite(signal_std) and signal_std > min_pp / 4.0)


def _canonicalize_validation_frame(
    *,
    base_profile: pd.DataFrame,
    validation_frame: pd.DataFrame,
    target_output_type: str,
    current_channel: str,
    field_channel: str,
) -> pd.DataFrame:
    canonical = _copy_frame_with_attrs(validation_frame)
    canonical.attrs["validation_window"] = {
        "applied": False,
        "start_s": 0.0,
        "end_s": _infer_fit_end_s(validation_frame),
        "score": float("nan"),
        "output_column": field_channel if str(target_output_type) == "field" else current_channel,
    }
    output_column = field_channel if str(target_output_type) == "field" else current_channel
    if canonical.empty or "time_s" not in canonical.columns or output_column not in canonical.columns:
        return canonical

    fit_end_s = _infer_fit_end_s(base_profile)
    if not np.isfinite(fit_end_s) or fit_end_s <= 0:
        return canonical
    target_column, _ = _resolve_target_output_column(base_profile)
    validation_time, validation_output = _valid_time_signal(canonical, output_column)
    if len(validation_time) < 16:
        return canonical
    if validation_time[-1] <= float(fit_end_s) + 1e-9:
        in_window = canonical["time_s"].between(0.0, float(fit_end_s), inclusive="both")
        cropped = _copy_frame_with_attrs(canonical.loc[in_window].copy())
        cropped.attrs["validation_window"] = {
            "applied": False,
            "start_s": 0.0,
            "end_s": float(fit_end_s),
            "score": float("nan"),
            "output_column": output_column,
        }
        return cropped if not cropped.empty else canonical

    sample_count = max(256, min(len(base_profile), len(validation_time)) * 2)
    time_grid = np.linspace(0.0, float(fit_end_s), sample_count)
    target_output = _interpolate_column(base_profile, target_column, time_grid)
    if not _is_signal_stable(target_output, min_pp=1e-3):
        return canonical

    candidate_starts = validation_time[validation_time <= validation_time[-1] - float(fit_end_s) + 1e-9]
    if candidate_starts.size == 0:
        return canonical
    stride = max(int(candidate_starts.size // 400), 1)
    best_start = float(candidate_starts[0])
    best_score = float("-inf")
    for start_s in candidate_starts[::stride]:
        measured = np.interp(time_grid + float(start_s), validation_time, validation_output)
        aligned_corr = _max_aligned_shape_score(target_output, measured)
        amplitude_ratio = _normalized_peak_ratio(target_output, measured)
        score = float((aligned_corr if np.isfinite(aligned_corr) else -1.0) + 0.35 * (amplitude_ratio if np.isfinite(amplitude_ratio) else 0.0))
        if score > best_score:
            best_score = score
            best_start = float(start_s)

    window_end_s = best_start + float(fit_end_s)
    mask = pd.to_numeric(canonical["time_s"], errors="coerce").between(best_start - 1e-9, window_end_s + 1e-9, inclusive="both")
    if not mask.any():
        return canonical
    cropped = _copy_frame_with_attrs(canonical.loc[mask].copy())
    cropped["time_s"] = pd.to_numeric(cropped["time_s"], errors="coerce") - float(best_start)
    cropped.attrs["validation_window"] = {
        "applied": bool(best_start > 1e-9),
        "start_s": float(best_start),
        "end_s": float(window_end_s),
        "score": float(best_score),
        "output_column": output_column,
    }
    return cropped


def _estimate_signal_scale(source_signal: np.ndarray, target_signal: np.ndarray) -> float:
    valid = np.isfinite(source_signal) & np.isfinite(target_signal)
    if valid.sum() < 8:
        return float("nan")
    centered_source = source_signal[valid] - float(np.nanmean(source_signal[valid]))
    centered_target = target_signal[valid] - float(np.nanmean(target_signal[valid]))
    denom = float(np.dot(centered_source, centered_source))
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return float(np.dot(centered_source, centered_target) / denom)


def _project_signal_from_reference_transfer(
    *,
    reference_profile: pd.DataFrame,
    corrected_profile: pd.DataFrame,
    reference_signal_column: str,
) -> np.ndarray | None:
    reference_voltage_column = "limited_voltage_v" if "limited_voltage_v" in reference_profile.columns else "recommended_voltage_v"
    corrected_voltage_column = "limited_voltage_v" if "limited_voltage_v" in corrected_profile.columns else "recommended_voltage_v"
    if (
        reference_voltage_column not in reference_profile.columns
        or corrected_voltage_column not in corrected_profile.columns
        or "time_s" not in reference_profile.columns
        or "time_s" not in corrected_profile.columns
        or reference_signal_column not in reference_profile.columns
    ):
        return None

    profile_time = pd.to_numeric(corrected_profile["time_s"], errors="coerce").to_numpy(dtype=float)
    corrected_voltage = pd.to_numeric(corrected_profile[corrected_voltage_column], errors="coerce").to_numpy(dtype=float)
    reference_voltage = _interpolate_column(reference_profile, reference_voltage_column, profile_time)
    reference_signal = _interpolate_column(reference_profile, reference_signal_column, profile_time)
    if not _is_signal_stable(reference_voltage, min_pp=1e-3) or not _is_signal_stable(reference_signal, min_pp=1e-3):
        return None

    corrected_voltage_centered = corrected_voltage - float(np.nanmean(corrected_voltage))
    reference_voltage_centered = reference_voltage - float(np.nanmean(reference_voltage))
    reference_signal_centered = reference_signal - float(np.nanmean(reference_signal))
    reference_voltage_fft = np.fft.rfft(reference_voltage_centered)
    reference_signal_fft = np.fft.rfft(reference_signal_centered)
    corrected_voltage_fft = np.fft.rfft(corrected_voltage_centered)
    valid_transfer_mask = np.abs(reference_voltage_fft) > 1e-9
    if int(np.count_nonzero(valid_transfer_mask)) < 2:
        return None
    transfer = np.zeros_like(reference_signal_fft, dtype=np.complex128)
    transfer[valid_transfer_mask] = reference_signal_fft[valid_transfer_mask] / reference_voltage_fft[valid_transfer_mask]
    projected_signal = np.fft.irfft(corrected_voltage_fft * transfer, n=len(profile_time))
    return projected_signal if _is_signal_stable(projected_signal, min_pp=1e-3) else None





def _ensure_predicted_output_from_reference_transfer(
    *,
    reference_profile: pd.DataFrame,
    corrected_profile: pd.DataFrame,
    target_output_type: str,
) -> pd.DataFrame:
    enriched = _copy_frame_with_attrs(corrected_profile)
    if str(target_output_type) == "field":
        existing = pd.to_numeric(enriched.get("expected_field_mT", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        if _is_signal_stable(existing, min_pp=1e-3):
            return enriched
        reference_signal_column = "expected_field_mT" if "expected_field_mT" in reference_profile.columns else "expected_output"
        projected_signal = _project_signal_from_reference_transfer(
            reference_profile=reference_profile,
            corrected_profile=enriched,
            reference_signal_column=reference_signal_column,
        )
        if projected_signal is None:
            return enriched
        enriched["expected_field_mT"] = projected_signal
        enriched["expected_output"] = projected_signal
        enriched["modeled_field_mT"] = projected_signal
        enriched["modeled_output"] = projected_signal
        return enriched

    existing = pd.to_numeric(enriched.get("expected_current_a", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    if _is_signal_stable(existing, min_pp=1e-3):
        return enriched
    reference_signal_column = "expected_current_a" if "expected_current_a" in reference_profile.columns else "expected_output"
    projected_signal = _project_signal_from_reference_transfer(
        reference_profile=reference_profile,
        corrected_profile=enriched,
        reference_signal_column=reference_signal_column,
    )
    if projected_signal is None:
        return enriched
    enriched["expected_current_a"] = projected_signal
    enriched["expected_output"] = projected_signal
    enriched["modeled_current_a"] = projected_signal
    enriched["modeled_output"] = projected_signal
    return enriched


def _ensure_bz_target_mapping(
    *,
    reference_profile: pd.DataFrame,
    validation_frame: pd.DataFrame,
    target_output_type: str,
    current_channel: str,
    field_channel: str,
) -> pd.DataFrame:
    enriched = _copy_frame_with_attrs(reference_profile)
    try:
        target_field = pd.to_numeric(enriched["target_field_mT"], errors="coerce").to_numpy(dtype=float)
        if _is_signal_stable(target_field, min_pp=1e-3):
            enriched.attrs["bz_target_mapping"] = {
                "available": True,
                "reason_code": None,
                "basis": "target_field_mT",
            }
            return enriched
    except KeyError:
        pass

    target_column, _ = _resolve_target_output_column(enriched)
    if field_channel not in validation_frame.columns:
        enriched.attrs["bz_target_mapping"] = {
            "available": False,
            "reason_code": "missing_bz_channel",
            "basis": "target_output",
        }
        return enriched
    drive_column = field_channel if str(target_output_type) == "field" else current_channel
    if drive_column not in validation_frame.columns:
        enriched.attrs["bz_target_mapping"] = {
            "available": False,
            "reason_code": "invalid_target_mapping",
            "basis": "target_output",
        }
        return enriched

    profile_time = pd.to_numeric(enriched["time_s"], errors="coerce").to_numpy(dtype=float)
    target_output = pd.to_numeric(enriched[target_column], errors="coerce").to_numpy(dtype=float)
    actual_drive = _interpolate_column(validation_frame, drive_column, profile_time)
    actual_bz = _interpolate_column(validation_frame, field_channel, profile_time)
    if not _is_signal_stable(actual_drive, min_pp=1e-3) or not _is_signal_stable(actual_bz, min_pp=1e-3):
        enriched.attrs["bz_target_mapping"] = {
            "available": False,
            "reason_code": "insufficient_active_window",
            "basis": "target_output",
        }
        return enriched

    scale = _estimate_signal_scale(actual_drive, actual_bz)
    phase_lag_s = _estimate_phase_lag_seconds(actual_drive, actual_bz, profile_time)
    mapped_target = _shift_signal_by_seconds(target_output * scale, phase_lag_s, profile_time)
    if not np.isfinite(scale) or not _is_signal_stable(mapped_target, min_pp=1e-3):
        enriched.attrs["bz_target_mapping"] = {
            "available": False,
            "reason_code": "surrogate_unstable",
            "basis": "target_output",
        }
        return enriched

    enriched["mapped_target_bz_effective_mT"] = mapped_target
    enriched["mapped_target_bz_scale"] = float(scale)
    enriched["mapped_target_bz_phase_lag_s"] = float(phase_lag_s) if np.isfinite(phase_lag_s) else float("nan")
    enriched.attrs["bz_target_mapping"] = {
        "available": True,
        "reason_code": None,
        "basis": "mapped_target_bz_effective_mT",
        "scale": float(scale),
        "phase_lag_s": float(phase_lag_s) if np.isfinite(phase_lag_s) else None,
    }
    return enriched


def _project_bz_from_validation_transfer(
    *,
    reference_profile: pd.DataFrame,
    corrected_profile: pd.DataFrame,
    validation_frame: pd.DataFrame,
    field_channel: str,
) -> pd.DataFrame:
    enriched = _copy_frame_with_attrs(corrected_profile)
    for column in ("expected_field_mT", "modeled_field_mT", "support_scaled_field_mT"):
        if column in enriched.columns:
            candidate = pd.to_numeric(enriched[column], errors="coerce").to_numpy(dtype=float)
            if _is_signal_stable(candidate, min_pp=1e-3):
                enriched.attrs["bz_projection"] = {
                    "available": True,
                    "reason_code": None,
                    "source": column,
                }
                return enriched

    if field_channel not in validation_frame.columns or "daq_input_v" not in validation_frame.columns:
        projected_bz = _project_signal_from_reference_transfer(
            reference_profile=reference_profile,
            corrected_profile=enriched,
            reference_signal_column=_resolve_bz_expected_column(reference_profile),
        )
        if projected_bz is not None:
            enriched["expected_field_mT"] = projected_bz
            enriched["modeled_field_mT"] = projected_bz
            enriched.attrs["bz_projection"] = {
                "available": True,
                "reason_code": None,
                "source": "reference_voltage_to_bz_transfer",
            }
            return enriched
        enriched.attrs["bz_projection"] = {
            "available": False,
            "reason_code": "missing_bz_channel",
            "source": "validation_transfer",
        }
        return enriched

    voltage_column = "limited_voltage_v" if "limited_voltage_v" in enriched.columns else "recommended_voltage_v"
    if voltage_column not in enriched.columns or "time_s" not in enriched.columns:
        enriched.attrs["bz_projection"] = {
            "available": False,
            "reason_code": "other",
            "source": "validation_transfer",
        }
        return enriched

    profile_time = pd.to_numeric(enriched["time_s"], errors="coerce").to_numpy(dtype=float)
    corrected_voltage = pd.to_numeric(enriched[voltage_column], errors="coerce").to_numpy(dtype=float)
    validation_voltage = _interpolate_column(validation_frame, "daq_input_v", profile_time)
    validation_bz = _interpolate_column(validation_frame, field_channel, profile_time)
    if not _is_signal_stable(validation_voltage, min_pp=1e-3) or not _is_signal_stable(validation_bz, min_pp=1e-3):
        projected_bz = _project_signal_from_reference_transfer(
            reference_profile=reference_profile,
            corrected_profile=enriched,
            reference_signal_column=_resolve_bz_expected_column(reference_profile),
        )
        if projected_bz is not None:
            enriched["expected_field_mT"] = projected_bz
            enriched["modeled_field_mT"] = projected_bz
            enriched.attrs["bz_projection"] = {
                "available": True,
                "reason_code": None,
                "source": "reference_voltage_to_bz_transfer",
            }
            return enriched
        enriched.attrs["bz_projection"] = {
            "available": False,
            "reason_code": "insufficient_active_window",
            "source": "validation_transfer",
        }
        return enriched

    corrected_voltage_centered = corrected_voltage - float(np.nanmean(corrected_voltage))
    validation_voltage_centered = validation_voltage - float(np.nanmean(validation_voltage))
    validation_bz_centered = validation_bz - float(np.nanmean(validation_bz))
    validation_voltage_fft = np.fft.rfft(validation_voltage_centered)
    validation_bz_fft = np.fft.rfft(validation_bz_centered)
    corrected_voltage_fft = np.fft.rfft(corrected_voltage_centered)
    transfer = np.zeros_like(validation_bz_fft, dtype=np.complex128)
    valid_transfer_mask = np.abs(validation_voltage_fft) > 1e-9
    if int(np.count_nonzero(valid_transfer_mask)) < 2:
        projected_bz = _project_signal_from_reference_transfer(
            reference_profile=reference_profile,
            corrected_profile=enriched,
            reference_signal_column=_resolve_bz_expected_column(reference_profile),
        )
        if projected_bz is not None:
            enriched["expected_field_mT"] = projected_bz
            enriched["modeled_field_mT"] = projected_bz
            enriched.attrs["bz_projection"] = {
                "available": True,
                "reason_code": None,
                "source": "reference_voltage_to_bz_transfer",
            }
            return enriched
        enriched.attrs["bz_projection"] = {
            "available": False,
            "reason_code": "surrogate_unstable",
            "source": "validation_transfer",
        }
        return enriched
    transfer[valid_transfer_mask] = validation_bz_fft[valid_transfer_mask] / validation_voltage_fft[valid_transfer_mask]
    projected_bz = np.fft.irfft(corrected_voltage_fft * transfer, n=len(profile_time))
    if not _is_signal_stable(projected_bz, min_pp=1e-3):
        projected_bz = _project_signal_from_reference_transfer(
            reference_profile=reference_profile,
            corrected_profile=enriched,
            reference_signal_column=_resolve_bz_expected_column(reference_profile),
        )
        if projected_bz is not None:
            enriched["expected_field_mT"] = projected_bz
            enriched["modeled_field_mT"] = projected_bz
            enriched.attrs["bz_projection"] = {
                "available": True,
                "reason_code": None,
                "source": "reference_voltage_to_bz_transfer",
            }
            return enriched
        enriched.attrs["bz_projection"] = {
            "available": False,
            "reason_code": "surrogate_unstable",
            "source": "validation_transfer",
        }
        return enriched

    enriched["expected_field_mT"] = projected_bz
    enriched["modeled_field_mT"] = projected_bz
    enriched.attrs["bz_projection"] = {
        "available": True,
        "reason_code": None,
        "source": "validation_voltage_to_bz_transfer",
    }
    return enriched




__all__ = [name for name in globals() if not name.startswith('__')]

