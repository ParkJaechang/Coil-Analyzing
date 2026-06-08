from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.ui_raw_waveforms_actual_drive import (
    classify_raw_waveform_actual_drive_csv,
    parse_raw_waveform_actual_drive_upload,
)


RAW_ACTUAL_UI = SRC_ROOT / "field_analysis" / "ui_raw_waveforms_actual_drive.py"
RAW_UI = SRC_ROOT / "field_analysis" / "ui_raw_waveforms.py"


def _actual_drive_csv_bytes(*, with_metadata: bool = True) -> bytes:
    time_ms = np.linspace(0.0, 1200.0, 61)
    active = (time_ms >= 100.0) & (time_ms <= 1100.0)
    voltage = np.where(active, 2.0 * np.sin(np.pi * (time_ms - 100.0) / 1000.0), 0.0)
    hallbz = 1.0 + 35.0 * np.sin(np.pi * np.clip((time_ms - 100.0) / 1000.0, 0.0, 1.0))
    preamble = []
    if with_metadata:
        preamble = [
            "# Frequency(Hz),1.000",
            "# Cycles,1.000",
            "# Waveform,sine",
        ]
    rows = ["TimeMs,Voltage1_V,HallBz,Current1_A"]
    rows.extend(f"{t:.6f},{v:.6f},{h:.6f},0.100000" for t, v, h in zip(time_ms, voltage, hallbz, strict=False))
    return ("\n".join([*preamble, *rows]) + "\n").encode("utf-8")


def test_raw_waveforms_actual_drive_section_markers_are_in_runtime_path() -> None:
    raw_source = RAW_UI.read_text(encoding="utf-8")
    actual_source = RAW_ACTUAL_UI.read_text(encoding="utf-8")

    assert "render_raw_waveforms_actual_drive_upload_section()" in raw_source
    for marker in [
        "1차 실구동 결과 업로드 확인",
        "Quick LUT 2차 보정에 사용하는 1차 실구동 결과 CSV를 Raw Waveforms에서도 같은 방식으로 확인합니다.",
        "raw_waveform_actual_drive_upload",
        "실구동 데이터 plot 생성",
        "사용 중인 실구동 데이터",
        "1차 실구동 데이터 확인",
        "Raw plot은 실제 측정 데이터의 native time_s 기준으로 표시됩니다.",
    ]:
        assert marker in actual_source


def test_raw_waveforms_actual_drive_processing_reuses_quick_lut_helpers() -> None:
    source = RAW_ACTUAL_UI.read_text(encoding="utf-8")

    assert "read_actual_drive_result(" in source
    assert "build_actual_drive_review_case(" in source
    assert "build_native_actual_drive_raw_plot_frame(" in source
    assert "command/target grid" not in source


def test_raw_waveforms_actual_drive_schema_classification() -> None:
    actual = classify_raw_waveform_actual_drive_csv("actual.csv", b"# x,y\nTimeMs,Voltage1_V,HallBz\n0,0,1\n")
    final_lut = classify_raw_waveform_actual_drive_csv("lut.csv", b"sample_index,time_s,voltage_v\n0,0,0\n")
    unsupported = classify_raw_waveform_actual_drive_csv("bad.csv", b"a,b,c\n1,2,3\n")

    assert actual["file_type"] == "actual_drive_result"
    assert final_lut["file_type"] == "final_voltage_lut"
    assert final_lut["schema_status"] == "final_voltage_lut_not_actual_drive_result"
    assert unsupported["file_type"] == "unsupported_schema"


def test_raw_waveforms_actual_drive_parse_uses_same_column_convention() -> None:
    payload = parse_raw_waveform_actual_drive_upload(
        filename="bench_measurement.csv",
        csv_bytes=_actual_drive_csv_bytes(with_metadata=False),
        waveform_type="sine",
        freq_hz=1.0,
        cycle_count=1.0,
    )
    frame = payload.review_frame

    expected_columns = {
        "time_s",
        "raw_hallbz_mT",
        "measured_field_effective_mT",
        "baseline_removed_effective_field_mT",
        "normalized_measured_field_mT",
        "raw_actual_drive_voltage_v",
        "normalized_actual_drive_voltage_v",
        "current_a",
    }
    assert expected_columns.issubset(frame.columns)
    assert np.allclose(frame["measured_field_effective_mT"], -frame["raw_hallbz_mT"])
    assert np.nanmax(np.abs(frame["normalized_measured_field_mT"])) <= 50.0 + 1e-9
    assert np.nanmax(np.abs(frame["normalized_actual_drive_voltage_v"])) <= 10.0 + 1e-9
    assert np.all(np.diff(pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)) > 0.0)
    assert payload.metadata["metadata_source"] == "current_quick_lut_selection"
    assert payload.metadata["source"] == "Raw Waveforms upload"


def test_raw_waveforms_actual_drive_button_gated_state_and_rejection_text() -> None:
    source = RAW_ACTUAL_UI.read_text(encoding="utf-8")

    assert "st.button(\"실구동 데이터 plot 생성\"" in source
    assert "업로드 또는 옵션 변경 즉시 parse" not in source
    assert "raw_waveform_actual_drive_review_result" in source
    assert "raw_waveform_actual_drive_review_metadata" in source
    assert "raw_waveform_actual_drive_render_key" in source
    assert "이 파일은 최종 전압 LUT CSV입니다." in source
    assert "필수 컬럼: TimeMs / Voltage1_V / HallBz" in source
