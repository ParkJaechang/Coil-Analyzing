from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def format_ratio_pct(value: object, *, digits: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    if not np.isfinite(numeric):
        return "unavailable"
    return f"{numeric * 100.0:.{digits}f}%"


def format_gain(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    if not np.isfinite(numeric):
        return "unavailable"
    return f"{numeric:.3f}x"


def format_voltage(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    if not np.isfinite(numeric):
        return "unavailable"
    return f"{numeric:.3f} V"


def render_error_ratio_metrics(st: Any, metadata: dict[str, Any], *, title: str = "오차율 요약") -> None:
    st.markdown(f"##### {title}")
    cols = st.columns(5)
    cols[0].metric("평균 오차율", format_ratio_pct(metadata.get("mean_error_ratio")))
    cols[1].metric("RMS 오차율", format_ratio_pct(metadata.get("rms_error_ratio")))
    cols[2].metric("최대 순간 오차율", format_ratio_pct(metadata.get("max_error_ratio")))
    cols[3].metric("피크 진폭 오차율", format_ratio_pct(metadata.get("peak_to_peak_error_ratio")))
    cols[4].metric("목표", format_ratio_pct(metadata.get("target_error_ratio_goal", 0.01)))

    peak_cols = st.columns(3)
    peak_cols[0].metric("양 피크 진폭 오차", format_ratio_pct(metadata.get("positive_peak_error_ratio")))
    peak_cols[1].metric("음 피크 진폭 오차", format_ratio_pct(metadata.get("negative_peak_error_ratio")))
    peak_cols[2].metric("평가 구간", _evaluation_window_label(metadata))

    gain_cols = st.columns(4)
    gain_cols[0].metric("적용 gain", format_gain(metadata.get("correction_gain_used")))
    gain_cols[1].metric("계산상 필요 gain", format_gain(metadata.get("auto_gain_nominal_residual_fraction")))
    gain_cols[2].metric("field 응답비", _format_field_per_volt(metadata))
    gain_cols[3].metric("headroom clamp", "ON" if metadata.get("auto_gain_headroom_limited") else "OFF")

    detail_cols = st.columns(3)
    detail_cols[0].metric("보정 delta 95% peak", format_voltage(metadata.get("auto_gain_unit_delta_peak_v")))
    detail_cols[1].metric("목표 보정 peak", format_voltage(metadata.get("auto_gain_target_delta_peak_v")))
    detail_cols[2].metric("safe headroom", format_voltage(metadata.get("auto_gain_headroom_safe_v")))

    if "peak_lobe_status" in metadata:
        lobe_cols = st.columns(4)
        lobe_cols[0].metric("gain mode", str(metadata.get("finite_first_gain_mode") or "unavailable"))
        lobe_cols[1].metric("lobe count", str(metadata.get("peak_lobe_lobe_count") or "0"))
        lobe_cols[2].metric("command peak", format_voltage(metadata.get("peak_lobe_command_peak_abs_v")))
        lobe_cols[3].metric(
            "peak-lobe limit",
            "EXCEEDED" if metadata.get("peak_lobe_voltage_limit_exceeded") else "ok",
        )

    ratio_basis = metadata.get("auto_gain_error_ratio_basis") or "unavailable"
    eval_cycle = metadata.get("error_evaluation_cycle_count")
    eval_end = metadata.get("error_evaluation_end_s")
    if eval_cycle is not None or eval_end is not None:
        st.caption(
            "오차율과 최대 오차 위치는 초기 응답과 방전 tail을 제외한 평가 구간에서 계산합니다. "
            f"평가 구간: {_evaluation_window_label(metadata)}, "
            f"start={_fmt(metadata.get('error_evaluation_start_s'))} s, end={_fmt(eval_end)} s."
        )
    st.caption(
        "자동 gain 기준: 평가 구간의 field residual 오차율을 1% 목표로 줄이는 데 필요한 반영 비율을 추정합니다. "
        "`gain = max(0, 1 - 목표오차율 / 현재오차율)`이며, 남은 전압 headroom이 부족하면 clamp됩니다. "
        f"현재 gain 오차율 기준: {ratio_basis}."
    )
    st.caption(
        "최대 순간 오차율은 같은 시간점의 |target-measured| 최대값입니다. "
        "피크 진폭 오차율은 양/음 피크 진폭 차이로 계산하므로, 피크 타이밍이나 양·음 피크가 모두 틀어진 경우 최대 순간 오차율보다 크게 보일 수 있습니다."
    )
    if bool(metadata.get("goal_reached", False)):
        st.success("평가 구간 RMS 오차율이 목표 이하입니다.")
    else:
        st.info("평가 구간 RMS 오차율이 목표보다 큽니다. 보정 전압 변화량에 계속 반영합니다.")


def build_error_ratio_summary_rows(metadata: dict[str, Any]) -> list[dict[str, object]]:
    goal = metadata.get("target_error_ratio_goal", 0.01)
    return [
        {"항목": "목표 전체 오차율", "계산 기준": "RMS(평가 구간)", "값": format_ratio_pct(goal)},
        {
            "항목": "평균 오차율",
            "계산 기준": "mean(|목표-실측| / 목표 피크)",
            "값": format_ratio_pct(metadata.get("mean_error_ratio")),
        },
        {
            "항목": "RMS 오차율",
            "계산 기준": "sqrt(mean(error_ratio^2))",
            "값": format_ratio_pct(metadata.get("rms_error_ratio")),
        },
        {
            "항목": "최대 오차율",
            "계산 기준": "max(|목표-실측| / 목표 피크)",
            "값": format_ratio_pct(metadata.get("max_error_ratio")),
        },
        {
            "항목": "피크-피크 오차율",
            "계산 기준": "|실측 pp - 목표 pp| / 목표 pp",
            "값": format_ratio_pct(metadata.get("peak_to_peak_error_ratio")),
        },
        {
            "항목": "적용 gain",
            "계산 기준": str(metadata.get("auto_gain_formula", "unavailable")),
            "값": format_gain(metadata.get("correction_gain_used")),
        },
        {
            "항목": "headroom clamp",
            "계산 기준": "남은 전압 headroom이 목표 보정 peak보다 작으면 ON",
            "값": "ON" if metadata.get("auto_gain_headroom_limited") else "OFF",
        },
    ]


def build_error_ratio_summary_frame(metadata: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(build_error_ratio_summary_rows(metadata))


def _evaluation_window_label(metadata: dict[str, Any]) -> str:
    start = metadata.get("error_evaluation_start_cycle")
    end = metadata.get("error_evaluation_end_cycle", metadata.get("error_evaluation_cycle_count"))
    if end is None:
        return "active"
    if start is None:
        start = 0.0
    return f"{_fmt(start)}~{_fmt(end)} cycle"


def _format_field_per_volt(metadata: dict[str, Any]) -> str:
    value = metadata.get("field_per_volt_mT_per_v")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    if not np.isfinite(numeric):
        return "unavailable"
    return f"{numeric:.3f} mT/V"


def _fmt(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    if not np.isfinite(numeric):
        return "unavailable"
    return f"{numeric:g}"
