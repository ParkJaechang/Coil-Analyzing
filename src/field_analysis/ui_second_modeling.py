"""UI helpers for user-triggered finite second modeled LUT generation."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import streamlit as st

from .finite_second_modeling import generate_second_modeled_voltage_lut
from .plotting import plot_waveforms
from .ui_voltage_lut_review import render_final_voltage_lut_export_panel


def render_second_modeling_controls(
    *,
    command_profile: pd.DataFrame,
    feedback_selection: dict[str, object] | None,
    freq_hz: float,
    cycle_count: float,
) -> None:
    st.markdown("#### 2nd Modeled Voltage LUT")
    st.caption("User-triggered only. No automatic second correction is generated on upload or option changes.")
    st.caption("Raw peak values are informational only. User decides final suitability from graphs. No automatic pass/fail judgement.")
    supported = np.isfinite(cycle_count) and abs(float(cycle_count) - 1.0) <= 1e-9
    if not supported:
        st.info("2nd modeling unavailable: production finite correction is 1.0 cycle only.")
        return
    if not feedback_selection or not feedback_selection.get("csv_bytes"):
        st.info("Upload/select an actual-drive result before generating the 2nd modeled LUT.")
        return
    gain = float(
        st.number_input(
            "2nd correction gain",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            key="second_modeling_correction_gain",
        )
    )
    if not st.button("Generate 2nd Modeled Voltage LUT", key="generate_second_modeled_voltage_lut"):
        cached = st.session_state.get("quick_lut_second_model_result")
        if isinstance(cached, dict) and isinstance(cached.get("command_profile"), pd.DataFrame):
            _render_second_modeling_result(cached["command_profile"], dict(cached.get("metadata") or {}))
        return
    suffix = "_" + Path(str(feedback_selection.get("filename") or "actual_drive_result.csv")).name
    with NamedTemporaryFile(prefix="quick_lut_second_model_", suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(bytes(feedback_selection["csv_bytes"]))
    try:
        second_profile, metadata = generate_second_modeled_voltage_lut(
            command_profile,
            temp_path,
            freq_hz=freq_hz,
            cycle_count=cycle_count,
            correction_gain=gain,
        )
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
    second_profile["second_modeling_available"] = bool(metadata.get("second_modeling_available", False))
    second_profile["second_modeling_status"] = str(metadata.get("second_modeling_status", "unavailable"))
    st.session_state["quick_lut_second_model_result"] = {
        "command_profile": second_profile,
        "metadata": metadata,
    }
    st.session_state["quick_lut_final_export_source"] = "second_model" if metadata.get("second_modeling_status") == "ok" else "first_model"
    _render_second_modeling_result(second_profile, metadata)


def _render_second_modeling_result(command_profile: pd.DataFrame, metadata: dict[str, object]) -> None:
    st.dataframe(pd.DataFrame([metadata]), use_container_width=True)
    if metadata.get("second_modeling_status") != "ok":
        st.info(f"2nd modeling unavailable: {metadata.get('second_modeling_status', 'unknown')}")
        return
    st.markdown("##### Intended vs Actual Comparison Plot")
    st.plotly_chart(
        plot_waveforms(
            command_profile,
            ["physical_target_output_mT", "measured_field_normalized_mT", "first_model_residual_mT"],
            title="Target vs actual field / residual",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        plot_waveforms(
            command_profile,
            ["first_modeled_voltage_v", "actual_drive_voltage_normalized_v", "second_correction_delta_v", "second_limited_voltage_v"],
            title="Command vs actual drive / second correction",
        ),
        use_container_width=True,
    )
    st.markdown("##### Raw Data Visualization Plot")
    st.plotly_chart(
        plot_waveforms(
            command_profile,
            [
                "measured_field_raw_mT",
                "measured_field_effective_mT",
                "measured_field_normalized_mT",
                "actual_drive_voltage_v",
                "actual_drive_voltage_normalized_v",
            ],
            title="Raw/effective/normalized actual-drive data",
        ),
        use_container_width=True,
    )
    render_final_voltage_lut_export_panel(
        command_profile=command_profile,
        finite_cycle_mode=True,
        waveform_type=None,
        freq_hz=None,
        cycle_count=1.0,
    )
