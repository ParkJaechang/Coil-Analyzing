from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


RISK_LEVEL_ORDER = ["Field Missing", "Voltage Missing", "Missing", "Weak"]


def build_diagnostics_focus_summary(
    *,
    continuous_support: pd.DataFrame,
    finite_support: pd.DataFrame,
) -> dict[str, Any]:
    continuous_problem_rows = extract_continuous_problem_combos(continuous_support)
    finite_missing_rows = extract_finite_missing_support_combos(finite_support)
    risk_counts = {
        risk_level: int((continuous_problem_rows.get("risk_level", pd.Series(dtype=object)) == risk_level).sum())
        for risk_level in RISK_LEVEL_ORDER
    }
    blocking_combo_count = risk_counts["Field Missing"] + risk_counts["Voltage Missing"] + risk_counts["Missing"]
    weak_combo_count = risk_counts["Weak"]
    return {
        "blocking_combo_count": blocking_combo_count,
        "weak_combo_count": weak_combo_count,
        "finite_missing_combo_count": int(len(finite_missing_rows)),
        "risk_counts": risk_counts,
        "continuous_problem_rows": continuous_problem_rows,
        "finite_missing_rows": finite_missing_rows,
        "compact_summary": build_diagnostics_focus_compact_summary(risk_counts, len(finite_missing_rows)),
    }


def extract_continuous_problem_combos(continuous_support: pd.DataFrame) -> pd.DataFrame:
    if continuous_support.empty:
        return _empty_continuous_problem_frame()

    problem_rows = continuous_support[continuous_support["risk_level"].isin(RISK_LEVEL_ORDER)].copy()
    if problem_rows.empty:
        return _empty_continuous_problem_frame()

    preferred_columns = [
        "risk_level",
        "waveform_type",
        "freq_hz",
        "continuous_test_count",
        "field_ready_test_count",
        "voltage_ready_test_count",
        "shape_comparison_possible",
    ]
    available_columns = [column for column in preferred_columns if column in problem_rows.columns]
    problem_rows = problem_rows[available_columns]
    problem_rows["risk_level"] = pd.Categorical(problem_rows["risk_level"], categories=RISK_LEVEL_ORDER, ordered=True)
    return problem_rows.sort_values(["risk_level", "waveform_type", "freq_hz"]).reset_index(drop=True)


def extract_finite_missing_support_combos(finite_support: pd.DataFrame) -> pd.DataFrame:
    if finite_support.empty:
        return _empty_finite_missing_frame()

    if "has_support" not in finite_support.columns:
        return _empty_finite_missing_frame()

    missing_rows = finite_support[~finite_support["has_support"]].copy()
    if missing_rows.empty:
        return _empty_finite_missing_frame()

    preferred_columns = [
        "waveform_type",
        "freq_hz",
        "finite_test_count",
        "field_ready_test_count",
        "voltage_ready_test_count",
        "risk_level",
    ]
    available_columns = [column for column in preferred_columns if column in missing_rows.columns]
    return missing_rows[available_columns].sort_values(["waveform_type", "freq_hz"]).reset_index(drop=True)


def build_diagnostics_focus_compact_summary(
    risk_counts: dict[str, int],
    finite_missing_combo_count: int,
) -> pd.DataFrame:
    rows = [
        {"focus_area": "Continuous", "risk_level": risk_level, "combo_count": int(risk_counts.get(risk_level, 0))}
        for risk_level in RISK_LEVEL_ORDER
    ]
    rows.append(
        {
            "focus_area": "Finite",
            "risk_level": "Missing",
            "combo_count": int(finite_missing_combo_count),
        }
    )
    return pd.DataFrame(rows)


def render_field_waveform_diagnostics_focus_block(
    *,
    continuous_support: pd.DataFrame,
    finite_support: pd.DataFrame,
) -> None:
    focus = build_diagnostics_focus_summary(
        continuous_support=continuous_support,
        finite_support=finite_support,
    )

    st.markdown("#### Problem-Focused View")
    st.caption("Use this block to isolate blocker and weak waveform/frequency combos before reading the full diagnostics tables.")

    metric_left, metric_mid, metric_right = st.columns(3)
    metric_left.metric("Blocking Combos", focus["blocking_combo_count"])
    metric_mid.metric("Weak Combos", focus["weak_combo_count"])
    metric_right.metric("Finite Missing Support", focus["finite_missing_combo_count"])

    if focus["blocking_combo_count"] == 0 and focus["weak_combo_count"] == 0 and focus["finite_missing_combo_count"] == 0:
        st.success("No blocker or weak support combos were detected in the current diagnostics snapshot.")
        return

    st.dataframe(focus["compact_summary"], use_container_width=True, hide_index=True)

    st.markdown("#### Continuous Problem Combos")
    if focus["continuous_problem_rows"].empty:
        st.caption("No continuous blocker or weak combos were detected.")
    else:
        st.dataframe(focus["continuous_problem_rows"], use_container_width=True, hide_index=True)

    st.markdown("#### Finite Missing Support Combos")
    if focus["finite_missing_rows"].empty:
        st.caption("No finite missing-support combos were detected.")
    else:
        st.dataframe(focus["finite_missing_rows"], use_container_width=True, hide_index=True)


def _empty_continuous_problem_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "risk_level",
            "waveform_type",
            "freq_hz",
            "continuous_test_count",
            "field_ready_test_count",
            "voltage_ready_test_count",
            "shape_comparison_possible",
        ]
    )


def _empty_finite_missing_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "waveform_type",
            "freq_hz",
            "finite_test_count",
            "field_ready_test_count",
            "voltage_ready_test_count",
            "risk_level",
        ]
    )


__all__ = [
    "RISK_LEVEL_ORDER",
    "build_diagnostics_focus_compact_summary",
    "build_diagnostics_focus_summary",
    "extract_continuous_problem_combos",
    "extract_finite_missing_support_combos",
    "render_field_waveform_diagnostics_focus_block",
]
