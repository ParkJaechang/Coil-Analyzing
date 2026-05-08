from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


REQUIRED_LUT_COLUMNS = ("sample_index", "time_s", "voltage_v")
DEBUG_VOLTAGE_COLUMNS = (
    "recommended_voltage_v",
    "baseline_recommended_voltage_v",
    "compensated_recommended_voltage_v",
    "startup_compensation_command_delta_v",
)


@dataclass(frozen=True)
class ParsedVoltageLut:
    source_name: str
    frame: pd.DataFrame
    ok: bool
    error: str | None = None


def build_final_voltage_lut_frame(command_profile: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ("time_s", "limited_voltage_v") if column not in command_profile.columns]
    if missing:
        raise ValueError(f"Missing final voltage LUT source columns: {missing}")

    lut_frame = pd.DataFrame(
        {
            "sample_index": np.arange(len(command_profile), dtype=int),
            "time_s": pd.to_numeric(command_profile["time_s"], errors="coerce"),
            "voltage_v": pd.to_numeric(command_profile["limited_voltage_v"], errors="coerce"),
        }
    )
    for column in DEBUG_VOLTAGE_COLUMNS:
        if column in command_profile.columns:
            lut_frame[column] = pd.to_numeric(command_profile[column], errors="coerce")
    return lut_frame


def build_final_voltage_lut_filename(
    *,
    waveform_type: object | None,
    freq_hz: object | None,
    cycle_count: object | None,
) -> str:
    waveform = _safe_name_part(waveform_type)
    freq = _format_number_for_filename(freq_hz)
    cycle = _format_number_for_filename(cycle_count)
    if waveform and freq and cycle:
        return f"finite_recommended_voltage_lut_{waveform}_{freq}Hz_{cycle}cycle.csv"
    return "finite_recommended_voltage_lut.csv"


def build_final_voltage_lut_csv_bytes(command_profile: pd.DataFrame) -> bytes:
    return build_final_voltage_lut_frame(command_profile).to_csv(index=False).encode("utf-8-sig")


def parse_voltage_lut_upload(source_name: str, data: bytes) -> ParsedVoltageLut:
    try:
        frame = pd.read_csv(_bytes_to_buffer(data))
    except Exception as exc:  # pragma: no cover - pandas error details vary
        return ParsedVoltageLut(source_name=source_name, frame=pd.DataFrame(), ok=False, error=str(exc))

    missing = [column for column in REQUIRED_LUT_COLUMNS if column not in frame.columns]
    if missing:
        return ParsedVoltageLut(
            source_name=source_name,
            frame=frame,
            ok=False,
            error=f"Missing required LUT columns: {missing}",
        )
    normalized = _normalize_lut_frame(frame)
    return ParsedVoltageLut(source_name=source_name, frame=normalized, ok=True)


def build_lut_diagnostics(frame: pd.DataFrame) -> dict[str, object]:
    time_s = pd.to_numeric(frame.get("time_s"), errors="coerce").to_numpy(dtype=float)
    voltage = pd.to_numeric(frame.get("voltage_v"), errors="coerce").to_numpy(dtype=float)
    finite_time = time_s[np.isfinite(time_s)]
    finite_voltage = voltage[np.isfinite(voltage)]
    diffs = np.diff(finite_time) if finite_time.size >= 2 else np.array([], dtype=float)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    duplicated_time_count = int(pd.Series(finite_time).duplicated().sum()) if finite_time.size else 0
    time_monotonic = bool(np.all(diffs > 0)) if diffs.size else True
    dt_min = _float_or_nan(np.min(positive_diffs)) if positive_diffs.size else float("nan")
    dt_median = _float_or_nan(np.median(positive_diffs)) if positive_diffs.size else float("nan")
    dt_max = _float_or_nan(np.max(positive_diffs)) if positive_diffs.size else float("nan")
    irregularity = float("nan")
    if np.isfinite(dt_median) and dt_median > 0:
        irregularity = float((dt_max - dt_min) / dt_median)

    duration = float("nan")
    if finite_time.size:
        duration = float(np.nanmax(finite_time) - np.nanmin(finite_time))
    suspected_time_unit = _infer_time_unit(duration, dt_median)
    time_axis_status = _time_axis_status(time_monotonic, duplicated_time_count, irregularity, suspected_time_unit)

    return {
        "sample_count": int(len(frame)),
        "time_start_s": _float_or_nan(np.nanmin(finite_time)) if finite_time.size else float("nan"),
        "time_end_s": _float_or_nan(np.nanmax(finite_time)) if finite_time.size else float("nan"),
        "duration_s": duration,
        "dt_min_s": dt_min,
        "dt_median_s": dt_median,
        "dt_max_s": dt_max,
        "dt_irregularity_ratio": irregularity,
        "time_monotonic": time_monotonic,
        "duplicated_time_count": duplicated_time_count,
        "voltage_min_v": _float_or_nan(np.nanmin(finite_voltage)) if finite_voltage.size else float("nan"),
        "voltage_max_v": _float_or_nan(np.nanmax(finite_voltage)) if finite_voltage.size else float("nan"),
        "suspected_time_unit": suspected_time_unit,
        "time_axis_status": time_axis_status,
    }


def build_normalized_lut_csv_bytes(frame: pd.DataFrame) -> bytes:
    return _normalize_lut_frame(frame).to_csv(index=False).encode("utf-8-sig")


def build_diagnostics_csv_bytes(source_name: str, diagnostics: dict[str, object]) -> bytes:
    row = {"source_name": source_name, **diagnostics}
    return pd.DataFrame([row]).to_csv(index=False).encode("utf-8-sig")


def render_final_voltage_lut_export_panel(
    *,
    command_profile: pd.DataFrame | None,
    finite_cycle_mode: bool,
    waveform_type: object | None,
    freq_hz: object | None,
    cycle_count: object | None,
) -> None:
    st.markdown("#### 최종 모델링 전압 LUT CSV 다운로드")
    st.caption(
        "화면 Command Waveform에 표시되는 최종 전압 배열을 Fourier 재합성 없이 그대로 저장합니다. "
        "voltage_v는 limited_voltage_v와 sample-by-sample 동일합니다."
    )
    st.caption("Fourier formula / harmonic coefficient export와 다른 time-voltage LUT입니다. no Fourier/resynthesis.")
    if not finite_cycle_mode:
        st.info("finite compensation LUT unavailable: finite compensation 결과에서만 다운로드할 수 있습니다.")
        return
    if command_profile is None or command_profile.empty:
        st.info("finite compensation LUT unavailable: command_profile이 없습니다.")
        return
    missing = [column for column in ("time_s", "limited_voltage_v") if column not in command_profile.columns]
    if missing:
        st.warning(f"finite compensation LUT unavailable: missing columns {missing}")
        return

    file_name = build_final_voltage_lut_filename(
        waveform_type=waveform_type,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
    )
    st.download_button(
        label="최종 모델링 전압 LUT CSV 다운로드",
        data=build_final_voltage_lut_csv_bytes(command_profile),
        file_name=file_name,
        mime="text/csv",
        key="download_final_modeled_voltage_lut_csv",
        help="Fourier 재합성 파형이 아니라 limited_voltage_v 기반 최종 time-voltage LUT입니다.",
    )


def render_voltage_lut_review_section(default_cache_root: Path | None = None) -> None:
    st.markdown("### LUT 검수 / LUT Review")
    st.caption("사용자 시간축/전압 파형 검수용 화면입니다. 장비 구동 적합성이나 보정 품질을 자동 판정하지 않음.")
    st.caption("필수 schema: sample_index, time_s, voltage_v")

    uploaded_files = st.file_uploader(
        "Exported voltage LUT CSV 업로드",
        type=["csv"],
        accept_multiple_files=True,
        key="voltage_lut_review_upload",
    )
    parsed_items: list[ParsedVoltageLut] = []
    for uploaded_file in uploaded_files or []:
        parsed_items.append(parse_voltage_lut_upload(uploaded_file.name, uploaded_file.getvalue()))

    cached_files = _discover_cached_lut_files(default_cache_root)
    if cached_files:
        with st.expander("cached/exported LUT file 불러오기", expanded=False):
            selected_cache = st.selectbox(
                "Cached LUT file",
                options=cached_files,
                format_func=lambda path: path.name,
                key="voltage_lut_cached_file",
            )
            if st.button("선택한 cached LUT 불러오기", key="load_cached_voltage_lut"):
                parsed_items.append(parse_voltage_lut_upload(selected_cache.name, selected_cache.read_bytes()))

    if not parsed_items:
        st.info("업로드된 LUT CSV가 없습니다. Quick LUT에서 다운로드한 최종 모델링 전압 LUT CSV를 업로드하세요.")
        return

    successes = [item for item in parsed_items if item.ok]
    failures = [item for item in parsed_items if not item.ok]
    st.write(f"- parsed LUT files: `{len(successes)}`")
    st.write(f"- failed LUT files: `{len(failures)}`")
    for failure in failures:
        st.warning(f"{failure.source_name}: {failure.error or 'parse failed'}")
    if not successes:
        return

    selected = st.selectbox(
        "LUT case",
        options=successes,
        format_func=lambda item: item.source_name,
        key="voltage_lut_review_case",
    )
    diagnostics = build_lut_diagnostics(selected.frame)
    _render_lut_plots(selected.frame)
    _render_lut_diagnostics(diagnostics)
    _render_lut_warnings(diagnostics)
    st.download_button(
        label="normalized LUT CSV 다운로드",
        data=build_normalized_lut_csv_bytes(selected.frame),
        file_name=f"normalized_{Path(selected.source_name).name}",
        mime="text/csv",
        key="download_normalized_voltage_lut_csv",
    )
    st.download_button(
        label="diagnostics summary CSV 다운로드",
        data=build_diagnostics_csv_bytes(selected.source_name, diagnostics),
        file_name=f"diagnostics_{Path(selected.source_name).stem}.csv",
        mime="text/csv",
        key="download_voltage_lut_diagnostics_csv",
    )


def _normalize_lut_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(
        {
            "sample_index": np.arange(len(frame), dtype=int),
            "time_s": pd.to_numeric(frame["time_s"], errors="coerce"),
            "voltage_v": pd.to_numeric(frame["voltage_v"], errors="coerce"),
        }
    )
    for column in DEBUG_VOLTAGE_COLUMNS:
        if column in frame.columns:
            normalized[column] = pd.to_numeric(frame[column], errors="coerce")
    return normalized


def _render_lut_plots(frame: pd.DataFrame) -> None:
    time_figure = go.Figure()
    time_figure.add_trace(go.Scatter(x=frame["time_s"], y=frame["voltage_v"], mode="lines", name="voltage_v"))
    time_figure.update_layout(
        template="plotly_white",
        height=360,
        title="LUT Voltage vs time_s",
        xaxis_title="time_s",
        yaxis_title="voltage_v",
    )
    st.plotly_chart(time_figure, use_container_width=True)

    sample_figure = go.Figure()
    sample_figure.add_trace(
        go.Scatter(x=frame["sample_index"], y=frame["voltage_v"], mode="lines", name="voltage_v")
    )
    sample_figure.update_layout(
        template="plotly_white",
        height=320,
        title="LUT Voltage vs sample_index",
        xaxis_title="sample_index",
        yaxis_title="voltage_v",
    )
    st.plotly_chart(sample_figure, use_container_width=True)


def _render_lut_diagnostics(diagnostics: dict[str, object]) -> None:
    st.markdown("#### LUT timebase diagnostics")
    labels = (
        "sample_count",
        "time_start_s",
        "time_end_s",
        "duration_s",
        "dt_min_s",
        "dt_median_s",
        "dt_max_s",
        "dt_irregularity_ratio",
        "time_monotonic",
        "duplicated_time_count",
        "voltage_min_v",
        "voltage_max_v",
        "suspected_time_unit",
        "time_axis_status",
    )
    st.dataframe(pd.DataFrame([diagnostics], columns=list(labels)), use_container_width=True)


def _render_lut_warnings(diagnostics: dict[str, object]) -> None:
    if diagnostics.get("suspected_time_unit") == "ms_like_seconds_column":
        st.warning("time_s가 ms처럼 보입니다. exported LUT의 time unit을 확인하세요.")
    if diagnostics.get("time_monotonic") is False or int(diagnostics.get("duplicated_time_count") or 0) > 0:
        st.warning("duplicate/non-monotonic time warning: time_s 축을 확인하세요.")
    irregularity = diagnostics.get("dt_irregularity_ratio")
    if isinstance(irregularity, (int, float)) and np.isfinite(irregularity) and irregularity > 0.05:
        st.warning("irregular dt warning: sample interval이 일정하지 않을 수 있습니다.")


def _discover_cached_lut_files(default_cache_root: Path | None) -> list[Path]:
    if default_cache_root is None:
        default_cache_root = Path("outputs") / "field_analysis_app_state"
    root = Path(default_cache_root)
    if not root.exists():
        return []
    patterns = ("finite_recommended_voltage_lut*.csv", "*recommended_voltage_lut*.csv")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(files), key=lambda path: str(path).lower())


def _bytes_to_buffer(data: bytes) -> object:
    from io import BytesIO

    return BytesIO(data)


def _safe_name_part(value: object | None) -> str:
    text = "" if value is None else str(value).strip().lower()
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in text).strip("_")


def _format_number_for_filename(value: object | None) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:g}"


def _float_or_nan(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _infer_time_unit(duration_s: float, dt_median_s: float) -> str:
    if np.isfinite(dt_median_s) and dt_median_s >= 1.0:
        return "ms_like_seconds_column"
    if np.isfinite(duration_s) and duration_s > 60.0:
        return "ms_like_seconds_column"
    return "seconds"


def _time_axis_status(
    time_monotonic: bool,
    duplicated_time_count: int,
    irregularity: float,
    suspected_time_unit: str,
) -> str:
    if not time_monotonic or duplicated_time_count > 0:
        return "error_duplicate_or_non_monotonic_time"
    if suspected_time_unit == "ms_like_seconds_column":
        return "warning_time_s_may_be_ms"
    if np.isfinite(irregularity) and irregularity > 0.05:
        return "warning_irregular_dt"
    return "ok"
