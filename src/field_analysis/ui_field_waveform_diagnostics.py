from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from .field_waveform_diagnostics import build_field_waveform_diagnostics


def _continuous_frames_by_test_id(analysis_lookup: dict[str, Any]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for test_id, analysis in analysis_lookup.items():
        corrected_frame = getattr(getattr(analysis, "preprocess", None), "corrected_frame", pd.DataFrame())
        if isinstance(corrected_frame, pd.DataFrame) and not corrected_frame.empty:
            frames[str(test_id)] = corrected_frame
            continue
        parsed_frame = getattr(getattr(analysis, "parsed", None), "normalized_frame", pd.DataFrame())
        if isinstance(parsed_frame, pd.DataFrame):
            frames[str(test_id)] = parsed_frame
    return frames


def _transient_frames(transient_measurements: list[Any] | None) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for measurement in transient_measurements or []:
        frame = getattr(measurement, "normalized_frame", pd.DataFrame())
        if isinstance(frame, pd.DataFrame):
            frames.append(frame)
    return frames


def render_field_waveform_diagnostics_section(
    *,
    per_test_summary: pd.DataFrame,
    analysis_lookup: dict[str, Any],
    transient_measurements: list[Any] | None,
    main_field_axis: str,
    voltage_input_column: str = "daq_input_v",
) -> None:
    st.markdown("#### Field Model Diagnostics")
    st.caption(
        "Use this view to inspect field-first coverage and support risk before changing the voltage-to-field model."
    )

    diagnostics = build_field_waveform_diagnostics(
        per_test_summary=per_test_summary,
        main_field_axis=main_field_axis,
        continuous_frames_by_test_id=_continuous_frames_by_test_id(analysis_lookup),
        transient_frames=_transient_frames(transient_measurements),
        voltage_input_column=voltage_input_column,
    )

    summary = diagnostics["summary"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Continuous Tests", summary["continuous_test_count"])
    c2.metric("Finite Tests", summary["finite_test_count"])
    c3.metric("OK Combos", summary["continuous_ok_combo_count"])
    c4.metric("Weak Combos", summary["continuous_weak_combo_count"])
    c5.metric("Shape-Comparable Combos", summary["shape_comparison_combo_count"])

    st.caption(
        f"Main field axis ready: {summary['main_field_axis_available']} | "
        f"Voltage input ready: {summary['voltage_input_available']} | "
        f"Target field metrics available: {summary['target_metric_candidate_count']}"
    )

    for note in diagnostics["notes"]:
        st.write(f"- {note}")

    st.markdown("#### Target Field Metric Candidates")
    st.dataframe(diagnostics["target_metric_candidates"], use_container_width=True, hide_index=True)

    st.markdown("#### Waveform Coverage")
    st.dataframe(diagnostics["waveform_counts"], use_container_width=True, hide_index=True)

    st.markdown("#### Frequency Coverage")
    st.dataframe(diagnostics["frequency_counts"], use_container_width=True, hide_index=True)

    st.markdown("#### Continuous Support by Waveform / Frequency")
    st.dataframe(diagnostics["continuous_support"], use_container_width=True, hide_index=True)

    st.markdown("#### Finite-Cycle Support by Waveform / Frequency")
    st.dataframe(diagnostics["finite_support"], use_container_width=True, hide_index=True)

    with st.expander("Continuous Test Capabilities", expanded=False):
        st.dataframe(diagnostics["continuous_test_details"], use_container_width=True, hide_index=True)

    with st.expander("Finite-Cycle Test Capabilities", expanded=False):
        st.dataframe(diagnostics["transient_test_details"], use_container_width=True, hide_index=True)


__all__ = ["render_field_waveform_diagnostics_section"]
