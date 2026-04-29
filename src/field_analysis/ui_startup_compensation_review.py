from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .utils import first_number


def render_startup_compensation_review(
    compensation: dict[str, object],
    command_profile: pd.DataFrame,
) -> None:
    st.markdown("#### Startup Compensation Review")
    st.caption(
        "이 섹션은 startup-aware compensation의 실사용자 검토용입니다. "
        "모델링 품질은 사용자 그래프 검수로 판단합니다. Physical Target은 변경되지 않습니다."
    )

    if not _has_startup_review_payload(compensation, command_profile):
        st.caption("startup compensation data unavailable")
        return

    applied = _coerce_boolish(
        _startup_payload_value(compensation, command_profile, "startup_transient_applied", "startup_compensation_applied")
    )
    status = _normalize_optional_text(
        _startup_payload_value(compensation, command_profile, "startup_status", "startup_transient_status")
    )
    rejected_reason = _normalize_optional_text(
        _startup_payload_value(
            compensation,
            command_profile,
            "startup_rejected_reason",
            "startup_compensation_reject_reason",
        )
    )
    data_quality_ok = _coerce_boolish(_startup_payload_value(compensation, command_profile, "startup_data_quality_ok"))

    if applied is True:
        st.info("startup compensation candidate is applied in this payload; inspect plots and before/after metrics.")
    elif rejected_reason:
        st.warning(f"startup compensation rejected or retained baseline: {rejected_reason}")
    elif applied is False:
        st.info("startup compensation was not applied; baseline command may be retained.")
    else:
        st.caption("startup compensation status is unavailable; this is not a quality failure by itself.")

    status_cols = st.columns(4)
    status_cols[0].metric("Applied", "yes" if applied is True else ("no" if applied is False else "n/a"))
    status_cols[1].metric("Status", status or "n/a")
    status_cols[2].metric("Source quality", "ok" if data_quality_ok is True else ("not ok" if data_quality_ok is False else "n/a"))
    status_cols[3].metric("Rejected reason", rejected_reason or "n/a")
    st.caption(
        "startup source: "
        f"type={_startup_payload_value(compensation, command_profile, 'startup_source_type') or 'n/a'} | "
        f"file={_startup_payload_value(compensation, command_profile, 'startup_source_file') or 'n/a'} | "
        f"support_id={_startup_payload_value(compensation, command_profile, 'startup_source_support_id') or 'n/a'}"
    )

    plot_left, plot_right = st.columns(2)
    field_figure = _startup_plot(
        command_profile,
        (
            ("physical_target_output_mT", "Physical Target", "solid"),
            ("open_loop_predicted_field_mT", "Open-loop Predicted Field", "dash"),
            ("compensated_predicted_field_mT", "Compensated Predicted Field", "solid"),
            ("startup_transient_component_mT", "Startup Transient Component", "dot"),
        ),
        title="Startup Field Comparison",
        yaxis_title="Field (mT)",
        secondary_trace_names=("Startup Transient Component",),
    )
    with plot_left:
        if field_figure is None:
            st.caption("startup field comparison data unavailable")
        else:
            st.plotly_chart(field_figure, use_container_width=True)

    voltage_figure = _startup_plot(
        command_profile,
        (
            ("baseline_recommended_voltage_v", "Baseline Recommended Voltage", "dash"),
            ("compensated_recommended_voltage_v", "Compensated Recommended Voltage", "solid"),
            ("startup_compensation_command_delta_v", "Startup Compensation Command Delta", "dot"),
        ),
        title="Startup Command Comparison",
        yaxis_title="Voltage (V)",
        secondary_trace_names=("Startup Compensation Command Delta",),
    )
    with plot_right:
        if voltage_figure is None:
            st.caption("startup command comparison data unavailable")
        else:
            st.plotly_chart(voltage_figure, use_container_width=True)

    metric_left, metric_right = st.columns(2)
    with metric_left:
        st.markdown("**Before / After Metrics**")
        _render_startup_metric_pair(
            "Startup residual",
            _startup_payload_value(compensation, command_profile, "startup_residual_before_mT", "early_cycle_residual_before"),
            _startup_payload_value(compensation, command_profile, "startup_residual_after_mT", "early_cycle_residual_after"),
            unit="mT",
        )
        _render_startup_metric_pair(
            "Active NRMSE",
            _startup_payload_value(compensation, command_profile, "active_nrmse_before"),
            _startup_payload_value(compensation, command_profile, "active_nrmse_after"),
        )
        _render_startup_metric_pair(
            "Active shape corr",
            _startup_payload_value(compensation, command_profile, "active_shape_corr_before"),
            _startup_payload_value(compensation, command_profile, "active_shape_corr_after"),
        )
    with metric_right:
        st.markdown("**Terminal / Tail Metrics**")
        _render_startup_metric_pair(
            "Terminal peak error",
            _startup_payload_value(compensation, command_profile, "terminal_peak_error_before_mT"),
            _startup_payload_value(compensation, command_profile, "terminal_peak_error_after_mT"),
            unit="mT",
        )
        _render_startup_metric_pair(
            "Tail residual",
            _startup_payload_value(compensation, command_profile, "tail_residual_before"),
            _startup_payload_value(compensation, command_profile, "tail_residual_after"),
        )
        st.write(
            "- Startup residual RMS: "
            f"`{_format_metric_transition(_startup_payload_value(compensation, command_profile, 'startup_residual_rms_before_mT'), _startup_payload_value(compensation, command_profile, 'startup_residual_rms_after_mT'), unit='mT', digits=4)}`"
        )


def _startup_payload_value(
    compensation: dict[str, object],
    command_profile: pd.DataFrame,
    *keys: str,
) -> object | None:
    for key in keys:
        if key in compensation and compensation.get(key) is not None:
            return compensation.get(key)
        if key in command_profile.columns:
            return command_profile[key].iloc[0] if not command_profile.empty else None
    return None


def _has_startup_review_payload(compensation: dict[str, object], command_profile: pd.DataFrame) -> bool:
    startup_keys = (
        "startup_transient_applied",
        "startup_status",
        "startup_transient_status",
        "startup_source_type",
        "startup_rejected_reason",
        "startup_residual_before_mT",
        "active_nrmse_before",
    )
    startup_columns = (
        "open_loop_predicted_field_mT",
        "startup_transient_component_mT",
        "compensated_predicted_field_mT",
        "baseline_recommended_voltage_v",
        "compensated_recommended_voltage_v",
        "startup_compensation_command_delta_v",
    )
    return any(key in compensation for key in startup_keys) or any(column in command_profile.columns for column in startup_columns)


def _startup_plot(
    command_profile: pd.DataFrame,
    traces: tuple[tuple[str, str, str], ...],
    *,
    title: str,
    yaxis_title: str,
    secondary_trace_names: tuple[str, ...] = (),
) -> go.Figure | None:
    if "time_s" not in command_profile.columns:
        return None
    figure = go.Figure()
    for column, label, dash in traces:
        if column not in command_profile.columns:
            continue
        visible = "legendonly" if label in secondary_trace_names else True
        figure.add_trace(
            go.Scatter(
                x=command_profile["time_s"],
                y=command_profile[column],
                mode="lines",
                name=label,
                visible=visible,
                line=dict(dash=dash),
            )
        )
    if not figure.data:
        return None
    figure.update_layout(
        template="plotly_white",
        height=420,
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=yaxis_title,
        legend_title="Signal",
    )
    return figure


def _render_startup_metric_pair(
    label: str,
    before: object,
    after: object,
    *,
    unit: str = "",
    digits: int = 4,
) -> None:
    st.write(f"- {label}: `{_format_metric_transition(before, after, unit=unit, digits=digits)}`")


def _format_metric_transition(before: object, after: object, unit: str = "", digits: int = 3) -> str:
    return f"{_format_optional_metric(before, unit, digits)} -> {_format_optional_metric(after, unit, digits)}"


def _format_optional_metric(value: object, unit: str = "", digits: int = 3) -> str:
    numeric = first_number(value)
    if numeric is None or not np.isfinite(float(numeric)):
        return "n/a"
    suffix = f" {unit}" if unit else ""
    return f"{float(numeric):.{digits}f}{suffix}"


def _coerce_boolish(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    numeric = first_number(value)
    if numeric is None or not np.isfinite(float(numeric)):
        return None
    return bool(float(numeric))


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
