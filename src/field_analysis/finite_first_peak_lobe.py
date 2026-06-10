from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .voltage_policy import COMMAND_VOLTAGE_LIMIT_V


PeakPolarity = Literal["positive", "negative"]


@dataclass(frozen=True)
class PeakLobe:
    index: int
    polarity: PeakPolarity
    start_s: float
    end_s: float
    target_peak_mT: float
    measured_peak_mT: float
    base_voltage_peak_v: float
    gain: float
    command_peak_v: float


@dataclass(frozen=True)
class PeakLobeModel:
    enabled: bool
    status: str
    cycle_policy: str
    lobes: tuple[PeakLobe, ...]
    expected_lobe_polarities: tuple[PeakPolarity, ...]
    detected_lobe_polarities: tuple[PeakPolarity, ...]
    gain_envelope: np.ndarray
    peak_lobe_command_voltage_v: np.ndarray
    peak_lobe_predicted_field_mT: np.ndarray
    voltage_limit_v: float
    peak_lobe_command_peak_abs_v: float
    voltage_limit_exceeded: bool
    voltage_exceeded_fraction: float
    peak_lobe_gain_envelope_smoothed: bool
    peak_lobe_boundary_taper_applied: bool

    @property
    def peak_lobe_base_voltage_v(self) -> np.ndarray:
        return self.peak_lobe_command_voltage_v


def build_peak_lobe_model(
    *,
    time_s: np.ndarray,
    target_field_mT: np.ndarray,
    aligned_measured_field_mT: np.ndarray,
    base_voltage_v: np.ndarray,
    active_mask: np.ndarray,
    cycle_count: float,
    voltage_limit_v: float = COMMAND_VOLTAGE_LIMIT_V,
) -> PeakLobeModel:
    time = np.asarray(time_s, dtype=float)
    target = np.asarray(target_field_mT, dtype=float)
    measured = np.asarray(aligned_measured_field_mT, dtype=float)
    voltage = np.asarray(base_voltage_v, dtype=float)
    active = np.asarray(active_mask, dtype=bool)
    if not (time.size == target.size == measured.size == voltage.size == active.size):
        raise ValueError("peak-lobe inputs must have the same length")

    expected_count, cycle_policy = _cycle_contract(float(cycle_count))
    if expected_count == 0:
        return _disabled("unsupported_cycle", cycle_policy, voltage, measured, voltage_limit_v=voltage_limit_v)

    finite = active & np.isfinite(time) & np.isfinite(target) & np.isfinite(measured) & np.isfinite(voltage)
    if int(finite.sum()) < 3:
        return _disabled("insufficient_active_samples", cycle_policy, voltage, measured, voltage_limit_v=voltage_limit_v)

    regions = _target_lobe_regions(target, finite)
    expected_polarities = _expected_polarities(float(cycle_count))
    detected_polarities = tuple(region[2] for region in regions[:expected_count])
    if len(regions) < expected_count:
        return _disabled(
            "missing_lobes",
            cycle_policy,
            voltage,
            measured,
            voltage_limit_v=voltage_limit_v,
            expected_lobe_polarities=expected_polarities,
            detected_lobe_polarities=detected_polarities,
        )
    regions = regions[:expected_count]
    detected_polarities = tuple(region[2] for region in regions)
    if detected_polarities != expected_polarities:
        return _disabled(
            "unexpected_lobe_polarity_sequence",
            cycle_policy,
            voltage,
            measured,
            voltage_limit_v=voltage_limit_v,
            expected_lobe_polarities=expected_polarities,
            detected_lobe_polarities=detected_polarities,
        )

    gain_envelope = np.ones(time.size, dtype=float)
    peak_lobe_command_voltage = voltage.copy()
    peak_lobe_field = measured.copy()
    lobes: list[PeakLobe] = []
    for lobe_index, (start_idx, end_idx, polarity) in enumerate(regions, start=1):
        region = np.zeros(time.size, dtype=bool)
        region[start_idx : end_idx + 1] = True
        region &= finite
        if int(region.sum()) < 2:
            return _disabled(
                "insufficient_lobe_samples",
                cycle_policy,
                voltage,
                measured,
                voltage_limit_v=voltage_limit_v,
                expected_lobe_polarities=expected_polarities,
                detected_lobe_polarities=detected_polarities,
            )

        target_peak = _signed_peak(target, region, polarity)
        measured_peak = _signed_peak(measured, region, polarity)
        voltage_peak = _signed_peak(voltage, region, polarity)
        if not np.isfinite(measured_peak) or abs(measured_peak) <= 1e-12:
            return _disabled(
                "zero_measured_lobe_peak",
                cycle_policy,
                voltage,
                measured,
                voltage_limit_v=voltage_limit_v,
                expected_lobe_polarities=expected_polarities,
                detected_lobe_polarities=detected_polarities,
            )
        if not _same_signed(target_peak, measured_peak, polarity):
            return _disabled(
                "measured_lobe_polarity_mismatch",
                cycle_policy,
                voltage,
                measured,
                voltage_limit_v=voltage_limit_v,
                expected_lobe_polarities=expected_polarities,
                detected_lobe_polarities=detected_polarities,
            )

        gain = float(target_peak / measured_peak)
        gain_envelope[region] = gain
        peak_lobe_command_voltage[region] = voltage[region] * gain
        peak_lobe_field[region] = measured[region] * gain
        lobes.append(
            PeakLobe(
                index=lobe_index,
                polarity=polarity,
                start_s=float(np.nanmin(time[region])),
                end_s=float(np.nanmax(time[region])),
                target_peak_mT=float(target_peak),
                measured_peak_mT=float(measured_peak),
                base_voltage_peak_v=float(voltage_peak),
                gain=gain,
                command_peak_v=float(voltage_peak * gain),
            )
        )
    limit_meta = _voltage_limit_metadata(peak_lobe_command_voltage, voltage_limit_v)

    return PeakLobeModel(
        enabled=True,
        status="ok",
        cycle_policy=cycle_policy,
        lobes=tuple(lobes),
        expected_lobe_polarities=expected_polarities,
        detected_lobe_polarities=detected_polarities,
        gain_envelope=gain_envelope,
        peak_lobe_command_voltage_v=peak_lobe_command_voltage,
        peak_lobe_predicted_field_mT=peak_lobe_field,
        peak_lobe_gain_envelope_smoothed=False,
        peak_lobe_boundary_taper_applied=False,
        **limit_meta,
    )


def _cycle_contract(cycle_count: float) -> tuple[int, str]:
    if abs(float(cycle_count) - 1.0) <= 1e-6:
        return 2, "1.0cycle_two_peak"
    if abs(float(cycle_count) - 1.5) <= 1e-6:
        return 3, "1.5cycle_three_peak"
    return 0, "unsupported_cycle"


def _expected_polarities(cycle_count: float) -> tuple[PeakPolarity, ...]:
    if abs(float(cycle_count) - 1.0) <= 1e-6:
        return ("positive", "negative")
    if abs(float(cycle_count) - 1.5) <= 1e-6:
        return ("positive", "negative", "positive")
    return ()


def _target_lobe_regions(target: np.ndarray, finite: np.ndarray) -> list[tuple[int, int, PeakPolarity]]:
    active_target = np.asarray(target, dtype=float)
    threshold = max(float(np.nanmax(np.abs(active_target[finite]))) * 0.02, 1e-9)
    signed = np.zeros(active_target.size, dtype=int)
    signed[finite & (active_target > threshold)] = 1
    signed[finite & (active_target < -threshold)] = -1

    regions: list[tuple[int, int, PeakPolarity]] = []
    start_idx: int | None = None
    current_sign = 0
    for index, sign in enumerate(signed):
        if sign == 0:
            if start_idx is not None:
                regions.append((start_idx, index - 1, _polarity(current_sign)))
                start_idx = None
                current_sign = 0
            continue
        if start_idx is None:
            start_idx = index
            current_sign = int(sign)
            continue
        if sign != current_sign:
            regions.append((start_idx, index - 1, _polarity(current_sign)))
            start_idx = index
            current_sign = int(sign)
    if start_idx is not None:
        regions.append((start_idx, signed.size - 1, _polarity(current_sign)))
    return regions


def _polarity(sign: int) -> PeakPolarity:
    return "positive" if int(sign) > 0 else "negative"


def _signed_peak(values: np.ndarray, mask: np.ndarray, polarity: PeakPolarity) -> float:
    selected = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if not selected.size:
        return float("nan")
    return float(np.nanmax(selected) if polarity == "positive" else np.nanmin(selected))


def _same_signed(target_peak: float, measured_peak: float, polarity: PeakPolarity) -> bool:
    if not np.isfinite(target_peak) or not np.isfinite(measured_peak):
        return False
    if polarity == "positive":
        return target_peak > 0.0 and measured_peak > 0.0
    return target_peak < 0.0 and measured_peak < 0.0


def _disabled(
    status: str,
    cycle_policy: str,
    base_voltage_v: np.ndarray,
    measured_field_mT: np.ndarray,
    *,
    voltage_limit_v: float,
    expected_lobe_polarities: tuple[PeakPolarity, ...] = (),
    detected_lobe_polarities: tuple[PeakPolarity, ...] = (),
) -> PeakLobeModel:
    voltage = np.asarray(base_voltage_v, dtype=float).copy()
    measured = np.asarray(measured_field_mT, dtype=float).copy()
    limit_meta = _voltage_limit_metadata(voltage, voltage_limit_v)
    return PeakLobeModel(
        enabled=False,
        status=status,
        cycle_policy=cycle_policy,
        lobes=(),
        expected_lobe_polarities=expected_lobe_polarities,
        detected_lobe_polarities=detected_lobe_polarities,
        gain_envelope=np.ones(voltage.size, dtype=float),
        peak_lobe_command_voltage_v=voltage,
        peak_lobe_predicted_field_mT=measured,
        peak_lobe_gain_envelope_smoothed=False,
        peak_lobe_boundary_taper_applied=False,
        **limit_meta,
    )


def _voltage_limit_metadata(values: np.ndarray, voltage_limit_v: float) -> dict[str, float | bool]:
    voltage = np.asarray(values, dtype=float)
    finite = voltage[np.isfinite(voltage)]
    limit = abs(float(voltage_limit_v)) if np.isfinite(float(voltage_limit_v)) else float(COMMAND_VOLTAGE_LIMIT_V)
    if limit <= 1e-12:
        limit = float(COMMAND_VOLTAGE_LIMIT_V)
    peak = float(np.nanmax(np.abs(finite))) if finite.size else 0.0
    exceeded = np.abs(finite) > limit + 1e-12 if finite.size else np.array([], dtype=bool)
    return {
        "voltage_limit_v": limit,
        "peak_lobe_command_peak_abs_v": peak,
        "voltage_limit_exceeded": bool(exceeded.any()) if finite.size else False,
        "voltage_exceeded_fraction": float(np.mean(exceeded)) if finite.size else 0.0,
    }
