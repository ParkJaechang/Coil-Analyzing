from __future__ import annotations

import numpy as np
import streamlit as st

from .finite_second_modeling_tail import resolve_finite_tail_policy


def render_finite_tail_policy_controls(*, freq_hz: float) -> dict[str, object]:
    st.markdown("##### Finite tail / 자기장 0복귀 전압")
    mode_label = st.selectbox(
        "Finite tail / 자기장 0복귀 전압",
        options=["자동: 주파수별 tail 정책", "사용", "사용 안 함"],
        index=0,
        key="finite_tail_mode",
    )
    mode = {"사용": "on", "사용 안 함": "off"}.get(mode_label, "auto")
    threshold = float(
        st.number_input(
            "tail 자동 OFF 기준 주파수",
            min_value=0.1,
            max_value=20.0,
            value=2.0,
            step=0.1,
            key="finite_tail_auto_threshold_hz",
        )
    )
    policy = resolve_finite_tail_policy(freq_hz=freq_hz, mode=mode, threshold_hz=threshold)
    st.caption("2Hz 이상에서는 phase delay 영향으로 마지막 피크 형성 전에 역전압이 걸릴 수 있어 tail을 기본 OFF로 둡니다.")
    st.caption("저주파에서는 자기장 0복귀가 필요한 경우 tail을 사용할 수 있습니다.")
    st.caption("자동 정책은 기본값이며, 필요하면 수동으로 tail 사용/미사용을 선택할 수 있습니다.")
    if policy.get("high_frequency_tail_auto_disabled"):
        st.warning("현재 주파수는 2Hz 이상입니다. 자동 정책에 따라 finite tail을 사용하지 않습니다.")
        st.caption("tail을 강제로 사용하려면 tail mode를 ‘사용’으로 변경하십시오.")
    status = "사용" if policy.get("finite_tail_effective_enabled") else "사용 안 함"
    st.info(f"현재 finite tail 상태: {status}")
    signature = {
        "finite_tail_mode": policy["finite_tail_mode"],
        "finite_tail_auto_threshold_hz": policy["finite_tail_auto_threshold_hz"],
        "finite_tail_policy_freq_hz": float(freq_hz) if np.isfinite(freq_hz) else None,
    }
    previous = st.session_state.get("finite_tail_policy_signature")
    if previous not in (None, signature) and st.session_state.get("quick_lut_second_model_result") is not None:
        st.session_state["quick_lut_second_model_dirty"] = True
        st.session_state["quick_lut_second_model_result_stale_reason"] = "finite_tail_policy_changed"
        st.warning("tail 설정이 변경되었습니다. 2차 보정 command를 다시 생성하십시오.")
    st.session_state["finite_tail_policy_signature"] = signature
    st.session_state["finite_tail_effective_enabled"] = bool(policy["finite_tail_effective_enabled"])
    return policy
