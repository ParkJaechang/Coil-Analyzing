from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def preferred_marker_channel(frame: pd.DataFrame, selected_channels: list[str]) -> str | None:
    for candidate in ("daq_input_v", "limited_voltage_v", "recommended_voltage_v", "bz_mT", "bmag_mT"):
        if candidate in frame.columns:
            return candidate
    return selected_channels[0] if selected_channels else None


def build_finite_marker_times(frame: pd.DataFrame, marker_channel: str | None) -> dict[str, float]:
    markers = {
        "detected_nonzero_start_s": _first_scalar_column_value(
            frame,
            ("detected_nonzero_start_s", "nonzero_start_s", "command_nonzero_start_s"),
        ),
        "detected_nonzero_end_s": _first_scalar_column_value(
            frame,
            ("detected_nonzero_end_s", "nonzero_end_s", "command_nonzero_end_s"),
        ),
        "target_active_end_s": _first_scalar_column_value(
            frame,
            ("target_active_end_s", "target_end_s", "active_target_end_s"),
        ),
        "zero_tail_start_s": _first_scalar_column_value(
            frame,
            ("zero_tail_start_s", "tail_start_s", "predicted_settle_end_s"),
        ),
    }
    if marker_channel:
        inferred_start, inferred_end = _infer_nonzero_bounds(frame, marker_channel)
        if not np.isfinite(markers["detected_nonzero_start_s"]):
            markers["detected_nonzero_start_s"] = inferred_start
        if not np.isfinite(markers["detected_nonzero_end_s"]):
            markers["detected_nonzero_end_s"] = inferred_end
    if not np.isfinite(markers["zero_tail_start_s"]):
        markers["zero_tail_start_s"] = markers["detected_nonzero_end_s"]
    if not np.isfinite(markers["target_active_end_s"]) and "is_active_target" in frame.columns:
        markers["target_active_end_s"] = _last_true_time(frame, "is_active_target")
    return markers


def add_finite_visual_markers(figure, marker_times: dict[str, float]) -> None:
    marker_styles = {
        "detected_nonzero_start_s": ("detected nonzero start", "green"),
        "detected_nonzero_end_s": ("detected nonzero end", "red"),
        "target_active_end_s": ("target active end", "blue"),
        "zero_tail_start_s": ("zero/tail section", "gray"),
    }
    for key, (label, color) in marker_styles.items():
        value = marker_times.get(key, float("nan"))
        if np.isfinite(value):
            figure.add_vline(
                x=float(value),
                line_dash="dash",
                line_color=color,
                annotation_text=label,
                annotation_position="top",
            )


def render_finite_marker_summary(marker_times: dict[str, float]) -> None:
    st.markdown("#### Finite-cycle visual markers")
    marker_frame = pd.DataFrame(
        [
            {"marker": key, "time_s": value if np.isfinite(value) else np.nan}
            for key, value in marker_times.items()
        ]
    )
    st.dataframe(marker_frame, hide_index=True, use_container_width=True)


def render_anomaly_helper(
    frame: pd.DataFrame,
    selected_channels: list[str],
    *,
    source_type: str,
    duration_s: float,
    freq_hz: float,
    cycle_count: float,
) -> None:
    rows = [
        _build_signal_quality_row(
            frame,
            channel,
            source_type=source_type,
            duration_s=duration_s,
            freq_hz=freq_hz,
            cycle_count=cycle_count,
        )
        for channel in selected_channels
    ]
    rows = [row for row in rows if row]
    if not rows:
        return
    st.markdown("#### Data quality quick checks")
    st.caption("Heuristic checks for manual audit only; they do not change modeling or solver behavior.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_channel_timebase_summary(display_frame: pd.DataFrame, selected_channels: list[str]) -> None:
    if "time_s" not in display_frame.columns:
        return
    time_values = pd.to_numeric(display_frame["time_s"], errors="coerce")
    rows: list[dict[str, object]] = []
    for channel in selected_channels:
        if channel not in display_frame.columns:
            continue
        values = pd.to_numeric(display_frame[channel], errors="coerce")
        valid = values.notna() & time_values.notna()
        if not valid.any():
            continue
        channel_time = time_values[valid]
        rows.append(
            {
                "channel": channel,
                "samples": int(valid.sum()),
                "time_min_s": float(channel_time.min()),
                "time_max_s": float(channel_time.max()),
                "duration_s": float(channel_time.max() - channel_time.min()),
            }
        )
    if rows:
        st.markdown("#### Channel timebase summary")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _build_signal_quality_row(
    frame: pd.DataFrame,
    channel: str,
    *,
    source_type: str,
    duration_s: float,
    freq_hz: float,
    cycle_count: float,
) -> dict[str, object] | None:
    if channel not in frame.columns:
        return None
    values = pd.to_numeric(frame[channel], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    pp = float(np.nanmax(finite) - np.nanmin(finite))
    rms = float(np.sqrt(np.nanmean(np.square(finite))))
    diffs = np.abs(np.diff(finite))
    max_jump = float(np.nanmax(diffs)) if len(diffs) else 0.0
    median_jump = float(np.nanmedian(diffs)) if len(diffs) else 0.0
    spike_threshold = max(0.6 * pp, 8.0 * median_jump, 1e-9)
    possible_spike = bool(max_jump > spike_threshold and pp > 0.0)
    flatline = bool(pp <= max(abs(float(np.nanmean(finite))) * 1e-6, 1e-9) or len(np.unique(np.round(finite, 9))) <= 2)
    return {
        "signal": channel,
        "pp": pp,
        "rms": rms,
        "max_adjacent_jump": max_jump,
        "possible_spike": possible_spike,
        "possible_clipping": _possible_clipping(finite),
        "flatline_suspicion": flatline,
        "duration_mismatch_suspicion": _duration_mismatch_suspicion(
            source_type=source_type,
            duration_s=duration_s,
            freq_hz=freq_hz,
            cycle_count=cycle_count,
        ),
    }


def _first_scalar_column_value(frame: pd.DataFrame, columns: tuple[str, ...]) -> float:
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    return float("nan")


def _infer_nonzero_bounds(frame: pd.DataFrame, channel: str) -> tuple[float, float]:
    if "time_s" not in frame.columns or channel not in frame.columns:
        return float("nan"), float("nan")
    values = pd.to_numeric(frame[channel], errors="coerce").to_numpy(dtype=float)
    times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(values) & np.isfinite(times)
    if not finite_mask.any():
        return float("nan"), float("nan")
    finite_values = values[finite_mask]
    finite_times = times[finite_mask]
    amplitude = float(np.nanmax(np.abs(finite_values))) if len(finite_values) else 0.0
    threshold = max(amplitude * 1e-3, 1e-9)
    active = np.abs(finite_values) > threshold
    if not active.any():
        return float("nan"), float("nan")
    active_times = finite_times[active]
    return float(active_times[0]), float(active_times[-1])


def _last_true_time(frame: pd.DataFrame, column: str) -> float:
    if "time_s" not in frame.columns or column not in frame.columns:
        return float("nan")
    flags = frame[column].astype(bool).to_numpy()
    times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    valid = flags & np.isfinite(times)
    if not valid.any():
        return float("nan")
    return float(times[valid][-1])


def _possible_clipping(values: np.ndarray) -> bool:
    if len(values) < 8:
        return False
    value_pp = float(np.nanmax(values) - np.nanmin(values))
    if value_pp <= 0.0:
        return False
    tolerance = max(value_pp * 1e-4, 1e-9)
    high_count = int(np.count_nonzero(np.abs(values - np.nanmax(values)) <= tolerance))
    low_count = int(np.count_nonzero(np.abs(values - np.nanmin(values)) <= tolerance))
    return high_count >= 4 or low_count >= 4


def _duration_mismatch_suspicion(
    *,
    source_type: str,
    duration_s: float,
    freq_hz: float,
    cycle_count: float,
) -> bool:
    if source_type != "finite-cycle":
        return False
    if not (np.isfinite(duration_s) and np.isfinite(freq_hz) and np.isfinite(cycle_count)):
        return False
    if freq_hz <= 0.0 or cycle_count <= 0.0:
        return False
    expected_duration = cycle_count / freq_hz
    return bool(duration_s < 0.5 * expected_duration or duration_s > 2.0 * expected_duration)
