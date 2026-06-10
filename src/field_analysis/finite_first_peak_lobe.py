from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


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
    gain_envelope: np.ndarray
    peak_lobe_base_voltage_v: np.ndarray
    peak_lobe_predicted_field_mT: np.ndarray


def build_peak_lobe_model(
    *,
    time_s: np.ndarray,
    target_field_mT: np.ndarray,
    aligned_measured_field_mT: np.ndarray,
    base_voltage_v: np.ndarray,
    active_mask: np.ndarray,
    cycle_count: float,
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
        return _disabled("unsupported_cycle", cycle_policy, voltage, measured)

    finite = active & np.isfinite(time) & np.isfinite(target) & np.isfinite(measured) & np.isfinite(voltage)
    if int(finite.sum()) < 3:
        return _disabled("insufficient_active_samples", cycle_policy, voltage, measured)

    regions = _target_lobe_regions(target, finite)
    if len(regions) < expected_count:
        return _disabled("missing_lobes", cycle_policy, voltage, measured)
    regions = regions[:expected_count]

    gain_envelope = np.ones(time.size, dtype=float)
    peak_lobe_voltage = voltage.copy()
    peak_lobe_field = measured.copy()
    lobes: list[PeakLobe] = []
    for lobe_index, (start_idx, end_idx, polarity) in enumerate(regions, start=1):
        region = np.zeros(time.size, dtype=bool)
        region[start_idx : end_idx + 1] = True
        region &= finite
        if int(region.sum()) < 2:
            return _disabled("insufficient_lobe_samples", cycle_policy, voltage, measured)

        target_peak = _signed_peak(target, region, polarity)
        measured_peak = _signed_peak(measured, region, polarity)
        voltage_peak = _signed_peak(voltage, region, polarity)
        if not np.isfinite(measured_peak) or abs(measured_peak) <= 1e-12:
            return _disabled("zero_measured_lobe_peak", cycle_policy, voltage, measured)
        if not _same_signed(target_peak, measured_peak, polarity):
            return _disabled("measured_lobe_polarity_mismatch", cycle_policy, voltage, measured)

        gain = float(target_peak / measured_peak)
        gain_envelope[region] = gain
        peak_lobe_voltage[region] = voltage[region] * gain
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

    return PeakLobeModel(
        enabled=True,
        status="ok",
        cycle_policy=cycle_policy,
        lobes=tuple(lobes),
        gain_envelope=gain_envelope,
        peak_lobe_base_voltage_v=peak_lobe_voltage,
        peak_lobe_predicted_field_mT=peak_lobe_field,
    )


def _cycle_contract(cycle_count: float) -> tuple[int, str]:
    if abs(float(cycle_count) - 1.0) <= 1e-6:
        return 2, "1.0cycle_two_peak"
    if abs(float(cycle_count) - 1.5) <= 1e-6:
        return 3, "1.5cycle_three_peak"
    return 0, "unsupported_cycle"


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
) -> PeakLobeModel:
    voltage = np.asarray(base_voltage_v, dtype=float).copy()
    measured = np.asarray(measured_field_mT, dtype=float).copy()
    return PeakLobeModel(
        enabled=False,
        status=status,
        cycle_policy=cycle_policy,
        lobes=(),
        gain_envelope=np.ones(voltage.size, dtype=float),
        peak_lobe_base_voltage_v=voltage,
        peak_lobe_predicted_field_mT=measured,
    )
