from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import streamlit as st


def render_quick_lut_target_summary(config: Mapping[str, Any], *, title: str = "Quick LUT target config") -> None:
    st.markdown(f"##### {title}")
    st.caption(
        f"현재 모델링 대상: {float(config.get('target_freq_hz') or 0):g} Hz / "
        f"{float(config.get('target_cycle_count') or 0):g} cycle"
    )
    st.caption("이 값은 사용자가 적용한 Quick LUT target config입니다.")
    rows = [
        ("mode", config.get("modeling_input_mode")),
        ("target frequency", f"{config.get('target_freq_hz')} Hz"),
        ("target cycle", f"{config.get('target_cycle_count')} cycle"),
        ("target field shape", config.get("target_field_shape")),
        ("target config source", config.get("quick_lut_target_config_source")),
        ("frequency source", config.get("target_freq_hz_source")),
        ("cycle source", config.get("target_cycle_count_source")),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["항목", "값"]), use_container_width=True, hide_index=True)


def render_quick_lut_target_debug(
    *,
    current_config: Mapping[str, Any],
    applied_config: Mapping[str, Any] | None,
    dirty: bool,
    last_modeling_config: Mapping[str, Any] | None = None,
    first_model_config: Mapping[str, Any] | None = None,
    second_model_config: Mapping[str, Any] | None = None,
) -> None:
    with st.expander("Quick LUT target/debug", expanded=False):
        rows = [
            ("UI current config", dict(current_config)),
            ("applied config", dict(applied_config or {})),
            ("config dirty", bool(dirty)),
            ("last modeling config", dict(last_modeling_config or {})),
            ("first model result config", dict(first_model_config or {})),
            ("second model result config", dict(second_model_config or {})),
            ("target freq/cycle", f"{current_config.get('target_freq_hz')} Hz / {current_config.get('target_cycle_count')} cycle"),
            ("effective freq/cycle", f"{(applied_config or current_config).get('target_freq_hz')} Hz / {(applied_config or current_config).get('target_cycle_count')} cycle"),
            ("mode policy reason", "continuous locks cycle to 1.0 only" if current_config.get("modeling_input_mode") == "continuous_steady_state" else "finite uses user-selected cycle"),
            ("target auto-overwrite detected", False),
            ("target auto-overwrite blocked", True),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["항목", "값"]), use_container_width=True, hide_index=True)
