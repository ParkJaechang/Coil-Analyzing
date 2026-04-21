from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import streamlit as st

from .recommendation_output_contract import (
    build_continuous_recommendation_payload,
    build_finite_cycle_recommendation_payload,
    build_recommendation_debug_payload,
)


def build_recommendation_export_payloads(recommendation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if bool(recommendation.get("finite_cycle_mode", False)):
        primary_payload = build_finite_cycle_recommendation_payload(recommendation)
    else:
        primary_payload = build_continuous_recommendation_payload(recommendation)

    debug_payload = build_recommendation_debug_payload(recommendation)
    return {
        "primary": primary_payload,
        "debug": debug_payload,
    }


def payload_to_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def payload_to_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return payload_to_json_text(payload).encode("utf-8")


def render_recommendation_export_panel(
    recommendation: Mapping[str, Any],
    *,
    file_stem: str,
    key_prefix: str,
) -> None:
    payloads = build_recommendation_export_payloads(recommendation)
    primary_payload = payloads["primary"]
    debug_payload = payloads["debug"]
    primary_json = payload_to_json_text(primary_payload)
    debug_json = payload_to_json_text(debug_payload)

    with st.expander("Recommendation JSON Exports", expanded=False):
        st.caption("Primary payload stays field-first. Current, gain, and hardware values remain in debug/reference only.")

        primary_left, primary_right = st.columns([3, 1])
        with primary_left:
            st.markdown("#### Primary Payload")
            st.code(primary_json, language="json")
        with primary_right:
            st.download_button(
                label="Primary JSON 다운로드",
                data=payload_to_json_bytes(primary_payload),
                file_name=f"{file_stem}_primary.json",
                mime="application/json",
                key=f"{key_prefix}_primary_json",
            )

        debug_left, debug_right = st.columns([3, 1])
        with debug_left:
            st.markdown("#### Debug Payload")
            st.code(debug_json, language="json")
        with debug_right:
            st.download_button(
                label="Debug JSON 다운로드",
                data=payload_to_json_bytes(debug_payload),
                file_name=f"{file_stem}_debug.json",
                mime="application/json",
                key=f"{key_prefix}_debug_json",
            )
