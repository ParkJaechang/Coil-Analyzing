from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..utils import canonicalize_waveform_type
from .base import ModelContext, ModelPrediction, PlantModel


@dataclass(slots=True)
class HarmonicObservation:
    """Single steady-state harmonic observation extracted from one support run."""

    run_id: str
    waveform_type: str
    freq_hz: float
    target_level_value: float | None
    sample_rate_hz: float
    reference_axis: str
    output_type: str
    harmonic_n: int
    input_coeff: complex
    output_coeff: complex
    input_snr: float | None = None
    output_snr: float | None = None
    cycle_index: int | None = None
    is_warm_tail: bool = True


def harmonic_cap(
    sample_rate_hz: float,
    fundamental_freq_hz: float,
    user_cap: int,
    nyquist_margin: float = 0.4,
) -> int:
    """Compute a conservative harmonic cap from sampling conditions."""

    if not np.isfinite(sample_rate_hz) or not np.isfinite(fundamental_freq_hz):
        return 1
    if sample_rate_hz <= 0 or fundamental_freq_hz <= 0:
        return 1
    if user_cap <= 0:
        return 1
    safe_cap = int(math.floor(float(nyquist_margin) * float(sample_rate_hz) / float(fundamental_freq_hz)))
    return max(1, min(int(user_cap), safe_cap))


def extract_harmonic_components(
    time_s: np.ndarray,
    values: np.ndarray,
    fundamental_freq_hz: float,
    max_harmonics: int,
) -> pd.DataFrame:
    """Extract complex harmonic coefficients from a uniformly sampled signal."""

    time = np.asarray(time_s, dtype=float)
    signal = np.asarray(values, dtype=float)
    valid = np.isfinite(time) & np.isfinite(signal)
    if valid.sum() < 8 or fundamental_freq_hz <= 0 or max_harmonics <= 0:
        return pd.DataFrame(columns=["harmonic", "real", "imag", "magnitude", "phase_rad"])

    time = time[valid]
    signal = signal[valid]
    rows: list[dict[str, float]] = []
    for harmonic in range(1, int(max_harmonics) + 1):
        basis = np.exp(-1j * 2.0 * np.pi * harmonic * float(fundamental_freq_hz) * time)
        coefficient = 2.0 * np.mean(signal * basis)
        rows.append(
            {
                "harmonic": harmonic,
                "real": float(np.real(coefficient)),
                "imag": float(np.imag(coefficient)),
                "magnitude": float(np.abs(coefficient)),
                "phase_rad": float(np.angle(coefficient)),
            }
        )
    return pd.DataFrame(rows)


def build_harmonic_transfer_frame(
    time_s: np.ndarray,
    input_v: np.ndarray,
    output_signal: np.ndarray,
    fundamental_freq_hz: float,
    max_harmonics: int,
) -> pd.DataFrame:
    """Build a transfer table `V_n -> Y_n` for one representative cycle."""

    input_components = extract_harmonic_components(time_s, input_v, fundamental_freq_hz, max_harmonics)
    output_components = extract_harmonic_components(time_s, output_signal, fundamental_freq_hz, max_harmonics)
    if input_components.empty or output_components.empty:
        return pd.DataFrame()

    merged = input_components.merge(
        output_components,
        on="harmonic",
        suffixes=("_input", "_output"),
        how="inner",
    )
    input_complex = merged["real_input"].to_numpy(dtype=float) + 1j * merged["imag_input"].to_numpy(dtype=float)
    output_complex = merged["real_output"].to_numpy(dtype=float) + 1j * merged["imag_output"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(np.abs(input_complex) > 0, output_complex / input_complex, np.nan + 1j * np.nan)
    merged["gain_real"] = np.real(gain)
    merged["gain_imag"] = np.imag(gain)
    merged["gain_magnitude"] = np.abs(gain)
    merged["gain_phase_rad"] = np.angle(gain)
    return merged


def build_harmonic_observation_frame(
    *,
    run_id: str,
    waveform_type: str,
    freq_hz: float,
    target_level_value: float | None,
    sample_rate_hz: float,
    reference_axis: str,
    output_type: str,
    time_s: np.ndarray,
    input_v: np.ndarray,
    output_signal: np.ndarray,
    max_harmonics: int,
    cycle_index: int | None = None,
    is_warm_tail: bool = True,
) -> pd.DataFrame:
    """Build harmonic observations from one representative steady-state cycle."""

    transfer_frame = build_harmonic_transfer_frame(
        time_s=time_s,
        input_v=input_v,
        output_signal=output_signal,
        fundamental_freq_hz=float(freq_hz),
        max_harmonics=max_harmonics,
    )
    if transfer_frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    canonical_waveform = canonicalize_waveform_type(waveform_type) or waveform_type
    for row in transfer_frame.to_dict(orient="records"):
        rows.append(
            asdict(
                HarmonicObservation(
                    run_id=str(run_id),
                    waveform_type=str(canonical_waveform),
                    freq_hz=float(freq_hz),
                    target_level_value=float(target_level_value) if target_level_value is not None and np.isfinite(target_level_value) else None,
                    sample_rate_hz=float(sample_rate_hz),
                    reference_axis=str(reference_axis),
                    output_type=str(output_type),
                    harmonic_n=int(row["harmonic"]),
                    input_coeff=complex(float(row["real_input"]), float(row["imag_input"])),
                    output_coeff=complex(float(row["real_output"]), float(row["imag_output"])),
                    input_snr=float(row.get("magnitude_input", np.nan)),
                    output_snr=float(row.get("magnitude_output", np.nan)),
                    cycle_index=cycle_index,
                    is_warm_tail=bool(is_warm_tail),
                )
            )
        )
    return pd.DataFrame(rows)


def _coerce_observation_frame(observations: Iterable[Any] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(observations, pd.DataFrame):
        frame = observations.copy()
    else:
        rows: list[dict[str, Any]] = []
        for item in observations:
            if isinstance(item, HarmonicObservation):
                rows.append(asdict(item))
            elif isinstance(item, dict):
                rows.append(dict(item))
        frame = pd.DataFrame(rows)

    if frame.empty:
        return pd.DataFrame()

    for column in (
        "freq_hz",
        "target_level_value",
        "sample_rate_hz",
        "harmonic_n",
        "input_snr",
        "output_snr",
        "cycle_index",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "waveform_type" in frame.columns:
        frame["waveform_type"] = frame["waveform_type"].map(
            lambda value: canonicalize_waveform_type(value) or value
        )
    if "input_coeff" in frame.columns and "output_coeff" in frame.columns:
        frame["transfer_complex"] = [
            output_coeff / input_coeff if abs(input_coeff) >= 1e-12 else complex(np.nan, np.nan)
            for input_coeff, output_coeff in zip(frame["input_coeff"], frame["output_coeff"], strict=False)
        ]
    elif {"transfer_real", "transfer_imag"}.issubset(frame.columns):
        frame["transfer_complex"] = [
            complex(real, imag)
            for real, imag in zip(
                pd.to_numeric(frame["transfer_real"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(frame["transfer_imag"], errors="coerce").to_numpy(dtype=float),
                strict=False,
            )
        ]
    else:
        return pd.DataFrame()

    frame["transfer_real"] = [float(np.real(value)) for value in frame["transfer_complex"]]
    frame["transfer_imag"] = [float(np.imag(value)) for value in frame["transfer_complex"]]
    frame["transfer_magnitude"] = [float(np.abs(value)) for value in frame["transfer_complex"]]
    frame["transfer_phase_rad"] = [float(np.angle(value)) for value in frame["transfer_complex"]]
    frame["transfer_log_magnitude"] = np.log(np.clip(frame["transfer_magnitude"].to_numpy(dtype=float), 1e-12, None))
    frame = frame.dropna(
        subset=[
            "freq_hz",
            "sample_rate_hz",
            "harmonic_n",
            "transfer_real",
            "transfer_imag",
        ]
    ).copy()
    if frame.empty:
        return pd.DataFrame()

    frame["harmonic_n"] = frame["harmonic_n"].astype(int)
    frame["target_level_value"] = pd.to_numeric(frame.get("target_level_value"), errors="coerce")
    frame["transfer_phase_unwrapped_rad"] = frame["transfer_phase_rad"].to_numpy(dtype=float)
    group_columns = ["output_type", "waveform_type", "reference_axis", "harmonic_n"]
    frame = frame.sort_values(group_columns + ["freq_hz", "target_level_value", "run_id"]).reset_index(drop=True)
    for _, group in frame.groupby(group_columns, dropna=False, sort=False):
        phases = pd.to_numeric(group["transfer_phase_rad"], errors="coerce").to_numpy(dtype=float)
        frame.loc[group.index, "transfer_phase_unwrapped_rad"] = np.unwrap(phases)
    return frame.reset_index(drop=True)


def _weighted_linear_predict(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    weights: np.ndarray,
) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if int(valid.sum()) == 0:
        return float("nan")
    if int(valid.sum()) == 1:
        return float(y[valid][0])

    x_valid = np.asarray(x[valid], dtype=float)
    y_valid = np.asarray(y[valid], dtype=float)
    w_valid = np.asarray(weights[valid], dtype=float)
    design = np.column_stack([np.ones_like(x_valid), x_valid - float(x0)])
    weighted_design = design * np.sqrt(w_valid)[:, None]
    weighted_target = y_valid * np.sqrt(w_valid)
    try:
        coeffs, *_ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
    except np.linalg.LinAlgError:
        return float(np.sum(w_valid * y_valid) / np.sum(w_valid))
    return float(coeffs[0])


def _wrap_phase_delta_rad(phase_a: float, phase_b: float) -> float:
    return float(np.angle(np.exp(1j * (float(phase_a) - float(phase_b)))))


def _interpolate_lcr_current_prior(
    *,
    context: ModelContext,
    harmonic_n: int,
    base_freq_hz: float,
) -> dict[str, Any] | None:
    impedance_table = context.metadata.get("lcr_impedance_table")
    if not isinstance(impedance_table, pd.DataFrame) or impedance_table.empty:
        return None
    harmonic_freq_hz = float(base_freq_hz) * int(harmonic_n)
    freq_values = pd.to_numeric(impedance_table.get("freq_hz"), errors="coerce").to_numpy(dtype=float)
    real_values = pd.to_numeric(impedance_table.get("current_per_v_real"), errors="coerce").to_numpy(dtype=float)
    imag_values = pd.to_numeric(impedance_table.get("current_per_v_imag"), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(freq_values) & np.isfinite(real_values) & np.isfinite(imag_values)
    if int(valid.sum()) == 0:
        return None
    freq_values = freq_values[valid]
    real_values = real_values[valid]
    imag_values = imag_values[valid]
    order = np.argsort(freq_values)
    freq_values = freq_values[order]
    real_values = real_values[order]
    imag_values = imag_values[order]
    clamped_freq = float(np.clip(harmonic_freq_hz, float(freq_values.min()), float(freq_values.max())))
    transfer_real = float(np.interp(clamped_freq, freq_values, real_values))
    transfer_imag = float(np.interp(clamped_freq, freq_values, imag_values))
    transfer = complex(transfer_real, transfer_imag)
    return {
        "harmonic": int(harmonic_n),
        "harmonic_freq_hz": harmonic_freq_hz,
        "transfer_real": transfer_real,
        "transfer_imag": transfer_imag,
        "transfer_mag": float(abs(transfer)),
        "transfer_phase_deg": float(np.degrees(np.angle(transfer))),
        "frequency_clamped": bool(abs(clamped_freq - harmonic_freq_hz) > 1e-9),
    }


def _blend_transfer_with_lcr_prior(
    *,
    empirical_transfer: complex,
    lcr_prior: complex,
    lcr_weight: float,
) -> tuple[complex, dict[str, float]]:
    if (
        not np.isfinite(np.real(empirical_transfer))
        or not np.isfinite(np.imag(empirical_transfer))
        or abs(empirical_transfer) < 1e-12
        or not np.isfinite(np.real(lcr_prior))
        or not np.isfinite(np.imag(lcr_prior))
        or abs(lcr_prior) < 1e-12
        or lcr_weight <= 0.0
    ):
        return empirical_transfer, {
            "lcr_gain_mismatch_log_abs": float("nan"),
            "lcr_phase_mismatch_rad": float("nan"),
            "lcr_weight_used": 0.0,
        }

    weight = float(np.clip(lcr_weight, 0.0, 1.0))
    empirical_log_mag = float(np.log(max(abs(empirical_transfer), 1e-12)))
    prior_log_mag = float(np.log(max(abs(lcr_prior), 1e-12)))
    empirical_phase = float(np.angle(empirical_transfer))
    prior_phase = float(np.angle(lcr_prior))
    phase_delta = _wrap_phase_delta_rad(empirical_phase, prior_phase)
    blended_log_mag = float((1.0 - weight) * empirical_log_mag + weight * prior_log_mag)
    blended_phase = float(prior_phase + (1.0 - weight) * phase_delta)
    blended_transfer = complex(np.exp(blended_log_mag + 1j * blended_phase))
    return blended_transfer, {
        "lcr_gain_mismatch_log_abs": float(abs(empirical_log_mag - prior_log_mag)),
        "lcr_phase_mismatch_rad": float(abs(phase_delta)),
        "lcr_weight_used": weight,
    }


def _interpolate_scalar_with_bracket(
    points_x: np.ndarray,
    points_y: np.ndarray,
    x0: float,
) -> tuple[float, str]:
    valid = np.isfinite(points_x) & np.isfinite(points_y)
    if int(valid.sum()) == 0:
        return float("nan"), "missing"
    x = np.asarray(points_x[valid], dtype=float)
    y = np.asarray(points_y[valid], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    tolerance = max(abs(float(x0)) * 1e-9, 1e-12)
    exact_mask = np.isclose(x, float(x0), atol=tolerance, rtol=0.0)
    if exact_mask.any():
        return float(np.mean(y[exact_mask])), "exact"
    if len(x) == 1:
        return float(y[0]), "single"

    lower_mask = x < float(x0)
    upper_mask = x > float(x0)
    if lower_mask.any() and upper_mask.any():
        lower_idx = np.flatnonzero(lower_mask)[-1]
        upper_idx = np.flatnonzero(upper_mask)[0]
        x_lower = float(x[lower_idx])
        x_upper = float(x[upper_idx])
        y_lower = float(y[lower_idx])
        y_upper = float(y[upper_idx])
        if abs(x_upper - x_lower) <= 1e-12:
            return float((y_lower + y_upper) / 2.0), "bracket_degenerate"
        alpha = float((float(x0) - x_lower) / (x_upper - x_lower))
        return float((1.0 - alpha) * y_lower + alpha * y_upper), "bracket_linear"

    distances = np.abs(x - float(x0))
    nearest_count = min(3, len(x))
    nearest_order = np.argsort(distances)[:nearest_count]
    nearest_x = x[nearest_order]
    nearest_y = y[nearest_order]
    nearest_dist = distances[nearest_order]
    weights = 1.0 / (np.square(nearest_dist) + 1e-6)
    predicted = _weighted_linear_predict(nearest_x, nearest_y, float(x0), weights)
    return predicted, "local_weighted"


def _interpolate_transfer_for_level(
    frame: pd.DataFrame,
    *,
    used_level: float,
) -> tuple[complex, dict[str, Any]] | None:
    if frame.empty:
        return None

    level_frame = frame.copy()
    finite_levels = pd.to_numeric(level_frame["target_level_value"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(finite_levels).any():
        log_mag = float(np.nanmean(level_frame["transfer_log_magnitude"].to_numpy(dtype=float)))
        phase = float(np.nanmean(level_frame["transfer_phase_unwrapped_rad"].to_numpy(dtype=float)))
        return complex(np.exp(log_mag + 1j * phase)), {
            "support_observation_count": int(len(level_frame)),
            "level_interpolation_mode": "no_level_axis",
            "interpolated_log_magnitude": log_mag,
            "interpolated_phase_unwrapped_rad": phase,
        }

    grouped_rows: list[dict[str, Any]] = []
    for level_value, group in level_frame.groupby("target_level_value", dropna=True, sort=True):
        grouped_rows.append(
            {
                "target_level_value": float(level_value),
                "transfer_log_magnitude": float(np.nanmean(pd.to_numeric(group["transfer_log_magnitude"], errors="coerce"))),
                "transfer_phase_unwrapped_rad": float(np.nanmean(pd.to_numeric(group["transfer_phase_unwrapped_rad"], errors="coerce"))),
                "support_observation_count": int(len(group)),
            }
        )
    grouped = pd.DataFrame(grouped_rows).sort_values("target_level_value").reset_index(drop=True)
    levels = grouped["target_level_value"].to_numpy(dtype=float)
    log_magnitudes = grouped["transfer_log_magnitude"].to_numpy(dtype=float)
    phases = grouped["transfer_phase_unwrapped_rad"].to_numpy(dtype=float)

    interpolated_log_mag, level_mode_mag = _interpolate_scalar_with_bracket(levels, log_magnitudes, float(used_level))
    interpolated_phase, level_mode_phase = _interpolate_scalar_with_bracket(levels, phases, float(used_level))
    interpolation_mode = level_mode_mag if level_mode_mag == level_mode_phase else f"{level_mode_mag}+{level_mode_phase}"
    transfer = complex(np.exp(interpolated_log_mag + 1j * interpolated_phase))
    return transfer, {
        "support_observation_count": int(grouped["support_observation_count"].sum()),
        "level_interpolation_mode": interpolation_mode,
        "available_level_values": levels.tolist(),
        "interpolated_log_magnitude": interpolated_log_mag,
        "interpolated_phase_unwrapped_rad": interpolated_phase,
    }


def _interpolate_transfer_for_frequency(
    frequency_points: pd.DataFrame,
    *,
    used_frequency_hz: float,
) -> tuple[complex, dict[str, Any]] | None:
    if frequency_points.empty:
        return None

    points = frequency_points.copy().sort_values("freq_hz").reset_index(drop=True)
    freqs = pd.to_numeric(points["freq_hz"], errors="coerce").to_numpy(dtype=float)
    log_magnitudes = pd.to_numeric(points["transfer_log_magnitude"], errors="coerce").to_numpy(dtype=float)
    phases = pd.to_numeric(points["transfer_phase_unwrapped_rad"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(freqs) & np.isfinite(log_magnitudes) & np.isfinite(phases) & (freqs > 0)
    if int(valid.sum()) == 0:
        return None
    freqs = freqs[valid]
    log_magnitudes = log_magnitudes[valid]
    phases = np.unwrap(phases[valid])
    points = points.loc[valid].copy().reset_index(drop=True)
    points["transfer_phase_unwrapped_rad"] = phases

    requested = float(used_frequency_hz)
    log_freqs = np.log(freqs)
    requested_log = float(np.log(requested)) if requested > 0 else float("nan")
    tolerance = max(abs(requested) * 1e-9, 1e-12)
    exact_mask = np.isclose(freqs, requested, atol=tolerance, rtol=0.0)
    if exact_mask.any():
        exact_points = points.loc[exact_mask].copy()
        transfer = complex(
            np.exp(
                float(np.nanmean(pd.to_numeric(exact_points["transfer_log_magnitude"], errors="coerce")))
                + 1j * float(np.nanmean(pd.to_numeric(exact_points["transfer_phase_unwrapped_rad"], errors="coerce")))
            )
        )
        return transfer, {
            "frequency_interpolation_mode": "exact",
            "lower_support_hz": requested,
            "upper_support_hz": requested,
            "local_frequency_point_count": int(len(exact_points)),
        }

    lower_mask = freqs < requested
    upper_mask = freqs > requested
    lower_support = float(freqs[lower_mask][-1]) if lower_mask.any() else np.nan
    upper_support = float(freqs[upper_mask][0]) if upper_mask.any() else np.nan

    bracket_log_mag, bracket_mag_mode = _interpolate_scalar_with_bracket(log_freqs, log_magnitudes, requested_log)
    bracket_phase, bracket_phase_mode = _interpolate_scalar_with_bracket(log_freqs, phases, requested_log)
    nearest_count = min(4, len(freqs))
    nearest_order = np.argsort(np.abs(log_freqs - requested_log))[:nearest_count]
    local_logs = log_freqs[nearest_order]
    local_log_mag = log_magnitudes[nearest_order]
    local_phase = phases[nearest_order]
    local_dist = np.abs(local_logs - requested_log)
    local_weights = 1.0 / (np.square(local_dist) + 1e-6)
    fitted_log_mag = _weighted_linear_predict(local_logs, local_log_mag, requested_log, local_weights)
    anchor_phase = bracket_phase if lower_mask.any() and upper_mask.any() else float(local_phase[np.argmin(local_dist)])
    phase_residuals = local_phase - float(anchor_phase)
    fitted_phase_residual = _weighted_linear_predict(local_logs, phase_residuals, requested_log, local_weights)
    fitted_phase = float(anchor_phase + fitted_phase_residual)

    if lower_mask.any() and upper_mask.any():
        # In-hull interpolation keeps bracket interpolation as the anchor and applies
        # a small residual correction from the local weighted neighborhood.
        bracket_span_log = max(abs(np.log(upper_support / lower_support)), 1e-9)
        neighborhood_scale = float(np.clip(np.mean(local_dist) / bracket_span_log, 0.0, 1.0))
        residual_blend = 0.20 + 0.15 * neighborhood_scale
        blended_log_mag = float((1.0 - residual_blend) * bracket_log_mag + residual_blend * fitted_log_mag)
        blended_phase = float((1.0 - residual_blend) * bracket_phase + residual_blend * fitted_phase)
        transfer = complex(np.exp(blended_log_mag + 1j * blended_phase))
        frequency_mode = f"bracket_anchor+local_residual({bracket_mag_mode},{bracket_phase_mode})"
    else:
        transfer = complex(np.exp(fitted_log_mag + 1j * fitted_phase))
        frequency_mode = f"local_weighted({bracket_mag_mode},{bracket_phase_mode})"

    return transfer, {
        "frequency_interpolation_mode": frequency_mode,
        "lower_support_hz": lower_support,
        "upper_support_hz": upper_support,
        "local_frequency_point_count": int(nearest_count),
        "phase_anchor_rad": float(anchor_phase),
    }


@dataclass(slots=True)
class HarmonicSurfaceModel(PlantModel):
    """Steady-state harmonic transfer surface built from representative support cycles."""

    observation_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    transfer_frame: pd.DataFrame = field(default_factory=pd.DataFrame)

    def fit(self, runs: Iterable[Any] | pd.DataFrame) -> pd.DataFrame:
        frame = _coerce_observation_frame(runs)
        self.observation_frame = frame
        if frame.empty:
            self.transfer_frame = pd.DataFrame()
            return self.transfer_frame
        self.transfer_frame = frame[
            [
                column
                for column in (
                    "run_id",
                    "waveform_type",
                    "freq_hz",
                    "target_level_value",
                    "sample_rate_hz",
                    "reference_axis",
                    "output_type",
                    "harmonic_n",
                    "transfer_real",
                    "transfer_imag",
                    "transfer_magnitude",
                    "transfer_phase_rad",
                    "transfer_phase_unwrapped_rad",
                )
                if column in frame.columns
            ]
        ].copy()
        self.transfer_frame = self.transfer_frame.rename(columns={"harmonic_n": "harmonic"})
        return self.transfer_frame

    def support_summary(
        self,
        *,
        context: ModelContext,
        output_type: str,
        reference_axis: str | None = None,
    ) -> dict[str, Any]:
        frame = self._filter_frame(
            output_type=output_type,
            waveform_type=context.waveform_type,
            reference_axis=reference_axis,
        )
        if frame.empty:
            return {
                "observation_count": 0,
                "frequency_count": 0,
                "level_count": 0,
                "harmonic_count": 0,
                "exact_freq_match": False,
            }
        exact_freq = np.isclose(
            frame["freq_hz"].to_numpy(dtype=float),
            float(context.freq_hz),
            atol=1e-9,
            equal_nan=False,
        )
        return {
            "observation_count": int(len(frame)),
            "frequency_count": int(frame["freq_hz"].nunique(dropna=True)),
            "level_count": int(frame["target_level_value"].dropna().nunique()),
            "harmonic_count": int(frame["harmonic_n"].nunique()),
            "exact_freq_match": bool(exact_freq.any()),
            "available_freq_min": float(frame["freq_hz"].min()),
            "available_freq_max": float(frame["freq_hz"].max()),
        }

    def invert_target(
        self,
        *,
        target_output: np.ndarray,
        output_type: str,
        context: ModelContext,
        max_harmonics: int,
        reference_axis: str | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]] | None:
        output = np.asarray(target_output, dtype=float)
        if len(output) < 4 or not np.isfinite(output).any():
            return None

        centered_output = output - float(np.nanmean(output))
        target_fft = np.fft.rfft(centered_output)
        recommended_fft = np.zeros_like(target_fft, dtype=np.complex128)
        harmonic_limit = min(int(max_harmonics), len(target_fft) - 1)
        rows: list[dict[str, Any]] = []

        for harmonic in range(1, harmonic_limit + 1):
            desired_component = target_fft[harmonic]
            if abs(desired_component) < 1e-12:
                continue
            estimate = self._estimate_transfer_details(
                output_type=output_type,
                harmonic_n=harmonic,
                context=context,
                reference_axis=reference_axis,
            )
            if estimate is None:
                continue
            transfer = estimate["transfer"]
            if not np.isfinite(transfer.real) or not np.isfinite(transfer.imag) or abs(transfer) < 1e-12:
                continue
            recommended_component = desired_component / transfer
            recommended_fft[harmonic] = recommended_component
            rows.append(
                {
                    "harmonic": harmonic,
                    "desired_real": float(np.real(desired_component)),
                    "desired_imag": float(np.imag(desired_component)),
                    "recommended_real": float(np.real(recommended_component)),
                    "recommended_imag": float(np.imag(recommended_component)),
                    "transfer_real": float(np.real(transfer)),
                    "transfer_imag": float(np.imag(transfer)),
                    "transfer_magnitude": float(np.abs(transfer)),
                    "transfer_phase_rad": float(np.angle(transfer)),
                    "support_observation_count": int(estimate["support_observation_count"]),
                    "frequency_clamped": bool(estimate["frequency_clamped"]),
                    "level_clamped": bool(estimate["level_clamped"]),
                    "lcr_prior_available": bool(estimate.get("lcr_prior_available", False)),
                    "lcr_frequency_clamped": bool(estimate.get("lcr_frequency_clamped", False)),
                    "lcr_gain_mismatch_log_abs": float(estimate.get("lcr_gain_mismatch_log_abs", np.nan)),
                    "lcr_phase_mismatch_rad": float(estimate.get("lcr_phase_mismatch_rad", np.nan)),
                    "lcr_weight_used": float(estimate.get("lcr_weight_used", 0.0)),
                    "phase_residual_std_rad": float(estimate.get("phase_residual_std_rad", np.nan)),
                    "phase_residual_pred_rad": float(estimate.get("phase_residual_pred_rad", np.nan)),
                    "phase_model_mode": str(estimate.get("phase_model_mode", "unknown")),
                    "used_frequency_hz": float(estimate["used_frequency_hz"]),
                    "used_level_value": float(estimate["used_level_value"]) if np.isfinite(estimate["used_level_value"]) else np.nan,
                }
            )

        if not rows:
            return None

        recommended_voltage = np.fft.irfft(recommended_fft, n=len(output))
        detail_frame = pd.DataFrame(rows)
        clamp_mask = detail_frame["frequency_clamped"] | detail_frame["level_clamped"]
        return recommended_voltage, detail_frame, {
            "usable_harmonic_count": int(len(detail_frame)),
            "phase_clamp_fraction": float(clamp_mask.mean()) if len(detail_frame) else 0.0,
            "max_harmonics_used": int(detail_frame["harmonic"].max()) if len(detail_frame) else 0,
            "lcr_prior_fraction": float(pd.to_numeric(detail_frame.get("lcr_prior_available", False), errors="coerce").fillna(0.0).mean()) if len(detail_frame) else 0.0,
            "lcr_weight_mean": float(pd.to_numeric(detail_frame.get("lcr_weight_used", np.nan), errors="coerce").dropna().mean()) if len(detail_frame) and "lcr_weight_used" in detail_frame.columns else 0.0,
            "phase_residual_std_mean_rad": float(pd.to_numeric(detail_frame.get("phase_residual_std_rad", np.nan), errors="coerce").dropna().mean()) if len(detail_frame) and "phase_residual_std_rad" in detail_frame.columns else np.nan,
        }

    def predict(self, input_v: np.ndarray, context: ModelContext) -> ModelPrediction:
        signal = np.asarray(input_v, dtype=float)
        if len(signal) == 0:
            return ModelPrediction(
                time_s=np.array([], dtype=float),
                input_v=signal,
                predicted_current_a=None,
                predicted_field_mT=None,
                debug_frame=None,
                debug_info={"status": "empty_input"},
            )

        centered = signal - float(np.nanmean(signal))
        voltage_fft = np.fft.rfft(centered)
        requested_cap = int(context.metadata.get("max_harmonics", len(voltage_fft) - 1))
        harmonic_limit = min(max(requested_cap, 1), len(voltage_fft) - 1)
        predicted_current_fft = np.zeros_like(voltage_fft, dtype=np.complex128)
        predicted_field_fft = np.zeros_like(voltage_fft, dtype=np.complex128)
        debug_rows: list[dict[str, Any]] = []
        field_axis = str(context.metadata.get("field_channel", "bz_mT"))

        for harmonic in range(1, harmonic_limit + 1):
            component = voltage_fft[harmonic]
            if abs(component) < 1e-12:
                continue

            current_estimate = self._estimate_transfer_details(
                output_type="current",
                harmonic_n=harmonic,
                context=context,
                reference_axis="current",
            )
            if current_estimate is not None:
                predicted_current_fft[harmonic] = component * current_estimate["transfer"]

            field_estimate = self._estimate_transfer_details(
                output_type="field",
                harmonic_n=harmonic,
                context=context,
                reference_axis=field_axis,
            )
            if field_estimate is not None:
                predicted_field_fft[harmonic] = component * field_estimate["transfer"]

            debug_rows.append(
                {
                    "harmonic": harmonic,
                    "input_real": float(np.real(component)),
                    "input_imag": float(np.imag(component)),
                    "current_transfer_real": float(np.real(current_estimate["transfer"])) if current_estimate is not None else np.nan,
                    "current_transfer_imag": float(np.imag(current_estimate["transfer"])) if current_estimate is not None else np.nan,
                    "current_lcr_weight_used": float(current_estimate.get("lcr_weight_used", 0.0)) if current_estimate is not None else np.nan,
                    "field_transfer_real": float(np.real(field_estimate["transfer"])) if field_estimate is not None else np.nan,
                    "field_transfer_imag": float(np.imag(field_estimate["transfer"])) if field_estimate is not None else np.nan,
                }
            )

        predicted_current = np.fft.irfft(predicted_current_fft, n=len(signal)) if np.any(predicted_current_fft) else None
        predicted_field = np.fft.irfft(predicted_field_fft, n=len(signal)) if np.any(predicted_field_fft) else None
        period_s = 1.0 / float(context.freq_hz) if np.isfinite(context.freq_hz) and context.freq_hz > 0 else 1.0
        time_s = np.linspace(0.0, period_s, len(signal))
        return ModelPrediction(
            time_s=time_s,
            input_v=signal,
            predicted_current_a=predicted_current,
            predicted_field_mT=predicted_field,
            debug_frame=pd.DataFrame(debug_rows) if debug_rows else None,
            debug_info={
                "status": "ok",
                "harmonic_limit": harmonic_limit,
                "field_axis": field_axis,
            },
        )

    def _filter_frame(
        self,
        *,
        output_type: str,
        waveform_type: str,
        reference_axis: str | None,
        harmonic_n: int | None = None,
    ) -> pd.DataFrame:
        if self.observation_frame.empty:
            return pd.DataFrame()

        canonical_waveform = canonicalize_waveform_type(waveform_type) or waveform_type
        frame = self.observation_frame[
            (self.observation_frame["output_type"] == str(output_type))
            & (self.observation_frame["waveform_type"] == canonical_waveform)
        ].copy()
        if reference_axis is not None and "reference_axis" in frame.columns:
            frame = frame[frame["reference_axis"] == str(reference_axis)].copy()
        if harmonic_n is not None:
            frame = frame[frame["harmonic_n"] == int(harmonic_n)].copy()
        return frame.reset_index(drop=True)

    def _estimate_transfer_details(
        self,
        *,
        output_type: str,
        harmonic_n: int,
        context: ModelContext,
        reference_axis: str | None,
    ) -> dict[str, Any] | None:
        frame = self._filter_frame(
            output_type=output_type,
            waveform_type=context.waveform_type,
            reference_axis=reference_axis,
            harmonic_n=harmonic_n,
        )
        if frame.empty:
            return None

        valid = (
            np.isfinite(pd.to_numeric(frame["freq_hz"], errors="coerce"))
            & np.isfinite(pd.to_numeric(frame["transfer_log_magnitude"], errors="coerce"))
            & np.isfinite(pd.to_numeric(frame["transfer_phase_unwrapped_rad"], errors="coerce"))
        )
        frame = frame.loc[valid].copy()
        if frame.empty:
            return None

        used_frequency = float(context.freq_hz)
        frequency_clamped = False
        frequencies = pd.to_numeric(frame["freq_hz"], errors="coerce").to_numpy(dtype=float)
        if len(frequencies) > 0:
            min_frequency = float(np.nanmin(frequencies))
            max_frequency = float(np.nanmax(frequencies))
            if np.isfinite(used_frequency):
                clamped_frequency = float(np.clip(used_frequency, min_frequency, max_frequency))
                frequency_clamped = abs(clamped_frequency - used_frequency) > 1e-9
                used_frequency = clamped_frequency

        levels = pd.to_numeric(frame["target_level_value"], errors="coerce").to_numpy(dtype=float)
        finite_levels = levels[np.isfinite(levels)]
        requested_level = float(context.target_level_value) if context.target_level_value is not None and np.isfinite(context.target_level_value) else float("nan")
        if len(finite_levels) == 0:
            used_level = requested_level
            level_clamped = False
        else:
            min_level = float(np.nanmin(finite_levels))
            max_level = float(np.nanmax(finite_levels))
            allow_level_extrapolation = bool(context.metadata.get("allow_output_extrapolation", True))
            if not np.isfinite(requested_level):
                used_level = float(np.nanmedian(finite_levels))
            elif allow_level_extrapolation:
                used_level = requested_level
            else:
                used_level = float(np.clip(requested_level, min_level, max_level))
            level_clamped = np.isfinite(requested_level) and abs(used_level - requested_level) > 1e-9

        frequency_points: list[dict[str, Any]] = []
        for frequency_value, group in frame.groupby("freq_hz", dropna=True, sort=True):
            level_transfer = _interpolate_transfer_for_level(group, used_level=used_level)
            if level_transfer is None:
                continue
            transfer_value, level_meta = level_transfer
            frequency_points.append(
                {
                    "freq_hz": float(frequency_value),
                    "transfer_log_magnitude": float(level_meta.get("interpolated_log_magnitude", np.log(max(abs(transfer_value), 1e-12)))),
                    "transfer_phase_unwrapped_rad": float(level_meta.get("interpolated_phase_unwrapped_rad", np.angle(transfer_value))),
                    "support_observation_count": int(level_meta.get("support_observation_count", len(group))),
                    "level_interpolation_mode": str(level_meta.get("level_interpolation_mode", "unknown")),
                }
            )
        frequency_frame = pd.DataFrame(frequency_points)
        if frequency_frame.empty:
            return None

        # Re-unwrap across the local frequency points to stabilize phase interpolation.
        frequency_frame = frequency_frame.sort_values("freq_hz").reset_index(drop=True)
        frequency_frame["transfer_phase_unwrapped_rad"] = np.unwrap(
            pd.to_numeric(frequency_frame["transfer_phase_unwrapped_rad"], errors="coerce").to_numpy(dtype=float)
        )

        frequency_transfer = _interpolate_transfer_for_frequency(
            frequency_frame,
            used_frequency_hz=used_frequency,
        )
        if frequency_transfer is None:
            return None
        transfer, frequency_meta = frequency_transfer
        lcr_prior_meta = _interpolate_lcr_current_prior(
            context=context,
            harmonic_n=harmonic_n,
            base_freq_hz=used_frequency,
        )
        lcr_prior_available = False
        lcr_frequency_clamped = False
        lcr_gain_mismatch_log_abs = float("nan")
        lcr_phase_mismatch_rad = float("nan")
        lcr_weight_used = 0.0
        phase_residual_std_rad = float("nan")
        phase_residual_pred_rad = float("nan")
        phase_model_mode = "empirical_frequency_phase"
        if (
            output_type == "current"
            and lcr_prior_meta is not None
            and not bool(context.metadata.get("exact_frequency_match", False))
        ):
            requested_prior = complex(
                float(lcr_prior_meta.get("transfer_real", np.nan)),
                float(lcr_prior_meta.get("transfer_imag", np.nan)),
            )
            if np.isfinite(np.real(requested_prior)) and np.isfinite(np.imag(requested_prior)) and abs(requested_prior) >= 1e-12:
                requested_log = float(np.log(max(used_frequency, 1e-12)))
                support_phase_rows: list[dict[str, float]] = []
                for point in frequency_frame.to_dict(orient="records"):
                    support_freq = float(point.get("freq_hz", np.nan))
                    support_prior_meta = _interpolate_lcr_current_prior(
                        context=context,
                        harmonic_n=harmonic_n,
                        base_freq_hz=support_freq,
                    )
                    if support_prior_meta is None:
                        continue
                    support_prior = complex(
                        float(support_prior_meta.get("transfer_real", np.nan)),
                        float(support_prior_meta.get("transfer_imag", np.nan)),
                    )
                    support_phase = float(point.get("transfer_phase_unwrapped_rad", np.nan))
                    if not (
                        np.isfinite(support_freq)
                        and support_freq > 0
                        and np.isfinite(support_phase)
                        and np.isfinite(np.real(support_prior))
                        and np.isfinite(np.imag(support_prior))
                        and abs(support_prior) >= 1e-12
                    ):
                        continue
                    support_phase_rows.append(
                        {
                            "log_freq": float(np.log(support_freq)),
                            "phase_residual_rad": _wrap_phase_delta_rad(support_phase, float(np.angle(support_prior))),
                        }
                    )
                if support_phase_rows:
                    support_phase_frame = pd.DataFrame(support_phase_rows)
                    support_logs = pd.to_numeric(support_phase_frame["log_freq"], errors="coerce").to_numpy(dtype=float)
                    support_residuals = pd.to_numeric(support_phase_frame["phase_residual_rad"], errors="coerce").to_numpy(dtype=float)
                    local_count = min(4, len(support_logs))
                    local_order = np.argsort(np.abs(support_logs - requested_log))[:local_count]
                    local_logs = support_logs[local_order]
                    local_residuals = support_residuals[local_order]
                    local_dist = np.abs(local_logs - requested_log)
                    local_weights = 1.0 / (np.square(local_dist) + 1e-6)
                    residual_pred = _weighted_linear_predict(local_logs, local_residuals, requested_log, local_weights)
                    fitted_local = np.array(
                        [_weighted_linear_predict(local_logs, local_residuals, float(log_x), local_weights) for log_x in local_logs],
                        dtype=float,
                    )
                    phase_residual_std_rad = float(np.sqrt(np.nanmean(np.square(local_residuals - fitted_local))))
                    phase_residual_pred_rad = float(residual_pred)
                    phase_model_mode = "lcr_anchor_phase_residual_local"
                    transfer_mag = float(abs(transfer))
                    transfer_phase = float(np.angle(requested_prior) + residual_pred)
                    transfer = complex(transfer_mag * np.exp(1j * transfer_phase))
                    lcr_prior_available = True
                    lcr_frequency_clamped = bool(lcr_prior_meta.get("frequency_clamped", False))
                    lcr_gain_mismatch_log_abs = float("nan")
                    lcr_phase_mismatch_rad = float(abs(residual_pred))
                    lcr_weight_used = 1.0
        return {
            "transfer": transfer,
            "support_observation_count": int(len(frame)),
            "used_frequency_hz": used_frequency,
            "used_level_value": used_level,
            "frequency_clamped": frequency_clamped,
            "level_clamped": level_clamped,
            "frequency_interpolation_mode": str(frequency_meta.get("frequency_interpolation_mode", "unknown")),
            "lower_support_hz": float(frequency_meta.get("lower_support_hz", np.nan)),
            "upper_support_hz": float(frequency_meta.get("upper_support_hz", np.nan)),
            "local_frequency_point_count": int(frequency_meta.get("local_frequency_point_count", len(frequency_frame))),
            "phase_anchor_rad": float(frequency_meta.get("phase_anchor_rad", np.nan)),
            "lcr_prior_available": bool(lcr_prior_available),
            "lcr_frequency_clamped": bool(lcr_frequency_clamped),
            "lcr_gain_mismatch_log_abs": lcr_gain_mismatch_log_abs,
            "lcr_phase_mismatch_rad": lcr_phase_mismatch_rad,
            "lcr_weight_used": lcr_weight_used,
            "phase_residual_std_rad": phase_residual_std_rad,
            "phase_residual_pred_rad": phase_residual_pred_rad,
            "phase_model_mode": phase_model_mode,
        }
