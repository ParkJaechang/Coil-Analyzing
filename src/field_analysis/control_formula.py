from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_control_formula(
    command_profile: pd.DataFrame,
    value_column: str = "limited_voltage_v",
    max_harmonics: int = 9,
    significant_digits: int = 6,
) -> dict[str, Any] | None:
    """Convert a recommended command waveform into a control-friendly Fourier formula."""

    if command_profile.empty or value_column not in command_profile.columns or "time_s" not in command_profile.columns:
        return None

    working = command_profile[["time_s", value_column]].copy()
    working = working.dropna(subset=["time_s", value_column]).sort_values("time_s")
    working = working.loc[working["time_s"].diff().fillna(1.0).ne(0.0)].reset_index(drop=True)
    if len(working) < 4:
        return None

    finite_cycle_mode = bool(
        command_profile["finite_cycle_mode"].iloc[0]
        if "finite_cycle_mode" in command_profile.columns
        else False
    )
    active_end_s = _infer_active_end_s(command_profile)
    total_end_s = float(working["time_s"].max())
    formula_period_s = active_end_s if active_end_s > 0 else total_end_s
    if not np.isfinite(formula_period_s) or formula_period_s <= 0:
        return None

    if finite_cycle_mode:
        fit_end_s = total_end_s
    else:
        fit_end_s = formula_period_s

    time_grid = np.linspace(0.0, fit_end_s, max(1024, len(working) * 4))
    values = np.interp(
        time_grid,
        working["time_s"].to_numpy(dtype=float),
        working[value_column].to_numpy(dtype=float),
    )
    coeff_table = _fourier_coefficients(
        time_grid=time_grid,
        values=values,
        period_s=fit_end_s,
        max_harmonics=max_harmonics,
    )
    reconstructed = _evaluate_fourier_series(
        coeff_table=coeff_table,
        time_grid=time_grid,
        period_s=fit_end_s,
    )
    reconstruction_frame = pd.DataFrame(
        {
            "time_s": time_grid,
            "original_voltage_v": values,
            "formula_voltage_v": reconstructed,
            "error_voltage_v": reconstructed - values,
        }
    )
    error = reconstruction_frame["error_voltage_v"].to_numpy(dtype=float)
    signal_pp = float(np.nanmax(values) - np.nanmin(values)) if len(values) else float("nan")
    rmse = float(np.sqrt(np.nanmean(np.square(error)))) if len(error) else float("nan")
    mae = float(np.nanmean(np.abs(error))) if len(error) else float("nan")
    max_abs_error = float(np.nanmax(np.abs(error))) if len(error) else float("nan")
    nrmse = rmse / max(signal_pp / 2.0, 1e-12) if np.isfinite(rmse) and np.isfinite(signal_pp) and signal_pp > 0 else float("nan")
    expression = _build_formula_expression(
        coeff_table=coeff_table,
        finite_cycle_mode=finite_cycle_mode,
        fit_end_s=fit_end_s,
        repeat_period_s=formula_period_s,
        significant_digits=significant_digits,
    )
    code_snippet = _build_python_snippet(
        coeff_table=coeff_table,
        finite_cycle_mode=finite_cycle_mode,
        fit_end_s=fit_end_s,
        repeat_period_s=formula_period_s,
        significant_digits=significant_digits,
    )

    return {
        "finite_cycle_mode": finite_cycle_mode,
        "value_column": value_column,
        "fit_end_s": float(fit_end_s),
        "repeat_period_s": float(formula_period_s),
        "coefficient_table": coeff_table,
        "reconstruction_frame": reconstruction_frame,
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs_error,
        "nrmse": nrmse,
        "formula_text": expression,
        "python_snippet": code_snippet,
    }


def build_control_lut(
    command_profile: pd.DataFrame,
    value_column: str = "limited_voltage_v",
    sample_count: int = 128,
) -> pd.DataFrame | None:
    """Build a control-friendly sampled LUT from the recommended command waveform."""

    if command_profile.empty or value_column not in command_profile.columns or "time_s" not in command_profile.columns:
        return None

    working = command_profile[["time_s", value_column]].copy()
    working = working.dropna(subset=["time_s", value_column]).sort_values("time_s")
    working = working.loc[working["time_s"].diff().fillna(1.0).ne(0.0)].reset_index(drop=True)
    if len(working) < 2:
        return None

    finite_cycle_mode = bool(
        command_profile["finite_cycle_mode"].iloc[0]
        if "finite_cycle_mode" in command_profile.columns
        else False
    )
    fit_end_s = _infer_active_end_s(command_profile) if finite_cycle_mode else float(working["time_s"].max())
    if not np.isfinite(fit_end_s) or fit_end_s <= 0:
        return None

    lut_time = np.linspace(0.0, fit_end_s, max(int(sample_count), 2))
    lut_values = np.interp(
        lut_time,
        working["time_s"].to_numpy(dtype=float),
        working[value_column].to_numpy(dtype=float),
    )
    lut = pd.DataFrame(
        {
            "lut_index": np.arange(len(lut_time), dtype=int),
            "time_s": lut_time,
            "command_voltage_v": lut_values,
            "finite_cycle_mode": finite_cycle_mode,
        }
    )
    if finite_cycle_mode:
        lut["time_fraction"] = lut["time_s"] / fit_end_s
    else:
        lut["cycle_progress"] = lut["time_s"] / fit_end_s
    return lut


def _infer_active_end_s(command_profile: pd.DataFrame) -> float:
    if "is_active_target" in command_profile.columns and command_profile["is_active_target"].any():
        return float(command_profile.loc[command_profile["is_active_target"], "time_s"].max())
    return float(command_profile["time_s"].max())


def _fourier_coefficients(
    time_grid: np.ndarray,
    values: np.ndarray,
    period_s: float,
    max_harmonics: int,
) -> pd.DataFrame:
    theta = 2.0 * np.pi * time_grid / float(period_s)
    rows: list[dict[str, float]] = []

    a0 = 2.0 * float(np.trapezoid(values, time_grid)) / float(period_s)
    rows.append(
        {
            "harmonic": 0,
            "a_n": a0 / 2.0,
            "b_n": 0.0,
            "amplitude": abs(a0 / 2.0),
            "phase_rad": 0.0,
            "phase_deg": 0.0,
        }
    )

    for harmonic in range(1, max_harmonics + 1):
        cos_term = np.cos(harmonic * theta)
        sin_term = np.sin(harmonic * theta)
        a_n = 2.0 * float(np.trapezoid(values * cos_term, time_grid)) / float(period_s)
        b_n = 2.0 * float(np.trapezoid(values * sin_term, time_grid)) / float(period_s)
        amplitude = float(np.hypot(a_n, b_n))
        phase_rad = float(np.arctan2(a_n, b_n))
        rows.append(
            {
                "harmonic": harmonic,
                "a_n": a_n,
                "b_n": b_n,
                "amplitude": amplitude,
                "phase_rad": phase_rad,
                "phase_deg": float(np.degrees(phase_rad)),
            }
        )
    return pd.DataFrame(rows)


def _build_formula_expression(
    coeff_table: pd.DataFrame,
    finite_cycle_mode: bool,
    fit_end_s: float,
    repeat_period_s: float,
    significant_digits: int,
) -> str:
    rows = coeff_table.to_dict(orient="records")
    constant = rows[0]["a_n"] if rows else 0.0
    terms: list[str] = [f"{constant:.{significant_digits}g}"]

    for row in rows[1:]:
        if abs(row["amplitude"]) < 1e-9:
            continue
        harmonic = int(row["harmonic"])
        terms.append(
            f"{row['a_n']:.{significant_digits}g}*cos(2*pi*{harmonic}*t/{fit_end_s:.{significant_digits}g})"
        )
        terms.append(
            f"{row['b_n']:.{significant_digits}g}*sin(2*pi*{harmonic}*t/{fit_end_s:.{significant_digits}g})"
        )

    body = " + ".join(terms).replace("+ -", "- ")
    if finite_cycle_mode:
        return (
            "V_cmd(t) = 0, t < 0\n"
            f"V_cmd(t) = {body}, 0 <= t <= {fit_end_s:.{significant_digits}g}\n"
            f"V_cmd(t) = 0, t > {fit_end_s:.{significant_digits}g}"
        )
    return (
        f"V_cmd(t) = {body}\n"
        f"repeat with period T = {repeat_period_s:.{significant_digits}g} s"
    )


def _build_python_snippet(
    coeff_table: pd.DataFrame,
    finite_cycle_mode: bool,
    fit_end_s: float,
    repeat_period_s: float,
    significant_digits: int,
) -> str:
    rows = coeff_table.to_dict(orient="records")
    coeff_lines = []
    for row in rows:
        coeff_lines.append(
            "    "
            + "{"
            + f"'harmonic': {int(row['harmonic'])}, 'a_n': {row['a_n']:.{significant_digits}g}, "
            + f"'b_n': {row['b_n']:.{significant_digits}g}"
            + "},"
        )
    if finite_cycle_mode:
        time_logic = (
            f"    if t < 0.0 or t > {fit_end_s:.{significant_digits}g}:\n"
            "        return 0.0\n"
            f"    tau = t\n"
        )
        period_value = fit_end_s
    else:
        time_logic = (
            f"    T = {repeat_period_s:.{significant_digits}g}\n"
            "    tau = t % T\n"
        )
        period_value = repeat_period_s

    return (
        "import math\n\n"
        "COEFFS = [\n"
        + "\n".join(coeff_lines)
        + "\n]\n\n"
        "def command_voltage(t: float) -> float:\n"
        + time_logic
        + "    value = 0.0\n"
        + "    for coeff in COEFFS:\n"
        + "        n = coeff['harmonic']\n"
        + "        if n == 0:\n"
        + "            value += coeff['a_n']\n"
        + "            continue\n"
        + f"        omega = 2.0 * math.pi * n / {period_value:.{significant_digits}g}\n"
        + "        value += coeff['a_n'] * math.cos(omega * tau)\n"
        + "        value += coeff['b_n'] * math.sin(omega * tau)\n"
        + "    return value\n"
    )


def _evaluate_fourier_series(
    coeff_table: pd.DataFrame,
    time_grid: np.ndarray,
    period_s: float,
) -> np.ndarray:
    values = np.zeros_like(time_grid, dtype=float)
    for row in coeff_table.to_dict(orient="records"):
        harmonic = int(row["harmonic"])
        if harmonic == 0:
            values += float(row["a_n"])
            continue
        omega = 2.0 * np.pi * harmonic / float(period_s)
        values += float(row["a_n"]) * np.cos(omega * time_grid)
        values += float(row["b_n"]) * np.sin(omega * time_grid)
    return values
