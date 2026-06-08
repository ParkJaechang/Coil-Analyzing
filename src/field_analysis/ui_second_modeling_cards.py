from __future__ import annotations

import pandas as pd
import streamlit as st

from .voltage_policy import COMMAND_VOLTAGE_NORMALIZATION_OR_LIMIT_MODE


def render_actual_drive_data_card(metadata: dict[str, object], *, cycle_count: float) -> None:
    source_file = metadata.get("quick_lut_actual_drive_selected_file") or metadata.get("actual_drive_source_file", "unknown")
    file_freq = metadata.get("file_freq_hz", metadata.get("target_freq_hz"))
    file_cycle = metadata.get("file_cycle_count", cycle_count)
    rows = [
        ("파일명", source_file),
        ("데이터 source", metadata.get("feedback_source_label", metadata.get("metadata_source", "uploads/2nd 폴더 또는 업로드 파일"))),
        ("schema", "TimeMs / Voltage1_V / HallBz"),
        ("파일 metadata", f"freq={file_freq}, cycle={file_cycle}"),
        ("현재 Quick LUT 설정", f"freq={metadata.get('target_freq_hz', metadata.get('requested_freq_hz', 'unknown'))}, cycle={cycle_count}"),
        ("match 상태", metadata.get("second_modeling_status", "unknown")),
        ("timebase", metadata.get("timebase_status", metadata.get("native_timebase_status", "unknown"))),
        ("HallBz sign convention", "effective field = -HallBz raw"),
        ("field normalization", metadata.get("field_normalization_mode", "peak_to_50mT")),
        ("voltage normalization", metadata.get("voltage_normalization_mode", COMMAND_VOLTAGE_NORMALIZATION_OR_LIMIT_MODE)),
    ]
    st.markdown("##### 사용 중인 1차 실구동 데이터")
    st.caption("현재 이 파일을 2차 모델링 입력으로 사용합니다.")
    st.dataframe(pd.DataFrame(rows, columns=["항목", "값"]), use_container_width=True, hide_index=True)
