from __future__ import annotations

import json
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coil_analyzer.analysis.metrics import analyze_dataset, compute_gain_requirement, compute_lambda_metrics  # noqa: E402
from coil_analyzer.constants import APP_NAME, DEFAULT_INPUT_VIN_PK, DEFAULT_REQUEST_FREQUENCIES_HZ, DEFAULT_TARGET_IPP_A, GAIN_MODES  # noqa: E402
from coil_analyzer.export.excel_export import build_export_bundle  # noqa: E402
from coil_analyzer.io.data_loader import build_dataset_meta, infer_column_roles, list_excel_sheets, load_dataframe, normalize_headers  # noqa: E402
from coil_analyzer.io.reference_loader import discover_reference_files, infer_reference_columns, load_reference_workbook, summarize_reference_sheet  # noqa: E402
from coil_analyzer.io.workspace_store import WorkspaceStore  # noqa: E402
from coil_analyzer.models import AnalysisWindow, ChannelConfig, ChannelMapping, DatasetMeta, RequestPoint  # noqa: E402
from coil_analyzer.plotting.figures import frequency_summary_figure, loop_figure, phasor_summary_figure, reference_comparison_figure, status_heatmap, waveform_figure  # noqa: E402
from coil_analyzer.preprocessing.alignment import estimate_delay_cross_correlation  # noqa: E402
from coil_analyzer.preprocessing.channels import infer_time_unit, sample_rate_from_time, standardize_dataset, summarize_channels  # noqa: E402
from coil_analyzer.utils.example_data import build_example_waveform  # noqa: E402
from coil_analyzer.utils.logging_utils import get_logger  # noqa: E402

LOGGER = get_logger(__name__)
STORE = WorkspaceStore(ROOT)
PAGE_LABELS = [
    "1. 홈 / 전체 현황",
    "2. 데이터 불러오기",
    "3. 테스트 요청 관리",
    "4. 채널 매핑 / 보정",
    "5. 신호 분석",
    "6. 전기 분석",
    "7. 자기장 분석",
    "8. 고급 분석",
    "9. Gain / 구동 요구 분석",
    "10. LCR 비교",
    "11. 내보내기",
]


@st.cache_data(show_spinner=False)
def load_frame_cached(path: str, sheet_name: str | None) -> pd.DataFrame:
    return normalize_headers(load_dataframe(Path(path), sheet_name))


def default_request_points() -> list[RequestPoint]:
    return [RequestPoint(frequency_hz=freq, target_ipp_a=DEFAULT_TARGET_IPP_A) for freq in DEFAULT_REQUEST_FREQUENCIES_HZ]


def page_index(label: str) -> int:
    return PAGE_LABELS.index(label) + 1


def render_quick_guide(title: str, body: str) -> None:
    with st.expander("빠른 가이드", expanded=False):
        st.markdown(f"**{title}**")
        st.write(body)


def localized_status(status: str) -> str:
    mapping = {
        "not tested": "미시험",
        "data loaded": "데이터 로드됨",
        "analyzed": "분석 완료",
        "flagged": "확인 필요",
    }
    return mapping.get(status, status)


def ensure_state() -> None:
    if "app_state_initialized" in st.session_state:
        return
    manifest = STORE.load_manifest()
    st.session_state.datasets = [DatasetMeta.from_dict(item) for item in manifest.get("datasets", [])]
    st.session_state.request_points = [RequestPoint(**item) for item in manifest.get("request_points", [])] or default_request_points()
    st.session_state.analysis_window = AnalysisWindow.from_dict(manifest.get("analysis_window", {}))
    st.session_state.settings = manifest.get(
        "settings",
        {"vin_pk": DEFAULT_INPUT_VIN_PK, "gain_mode": GAIN_MODES[0], "representative_b": {}, "reference_workbook_path": None},
    )
    st.session_state.analysis_results = {}
    st.session_state.app_state_initialized = True


def persist_state() -> None:
    STORE.save_manifest(
        datasets=st.session_state.datasets,
        request_points=st.session_state.request_points,
        analysis_window=st.session_state.analysis_window.to_dict(),
        settings=st.session_state.settings,
    )


def dataset_by_id(dataset_id: str) -> DatasetMeta:
    for dataset in st.session_state.datasets:
        if dataset.dataset_id == dataset_id:
            return dataset
    raise KeyError(dataset_id)


def dataframe_for_dataset(dataset: DatasetMeta) -> pd.DataFrame:
    return load_frame_cached(dataset.stored_path, dataset.selected_sheet)


def signature_from_columns(columns: list[str]) -> str:
    return "|".join([column.strip().lower() for column in columns])


def auto_apply_mapping(dataset: DatasetMeta, df: pd.DataFrame) -> None:
    library = STORE.load_mapping_library()
    signature = signature_from_columns(list(df.columns))
    if signature in library and dataset.mapping.time.column is None:
        dataset.mapping = ChannelMapping.from_dict(library[signature])
    guesses = infer_column_roles(list(df.columns))
    if guesses["time"] and (dataset.mapping.time.column is None or dataset.mapping.time.column not in df.columns):
        dataset.mapping.time.column = guesses["time"][0]
        dataset.mapping.time.unit = infer_time_unit(df[guesses["time"][0]])
    if dataset.mapping.time.column and dataset.mapping.time.column in df.columns:
        dataset.mapping.time.unit = infer_time_unit(df[dataset.mapping.time.column])

    dataset.mapping.voltage.column = repair_role_mapping(
        current_column=dataset.mapping.voltage.column,
        candidates=guesses["voltage"],
        disallowed={dataset.mapping.time.column},
        df=df,
    )
    dataset.mapping.current.column = repair_role_mapping(
        current_column=dataset.mapping.current.column,
        candidates=guesses["current"],
        disallowed={dataset.mapping.time.column, dataset.mapping.voltage.column},
        df=df,
    )
    if guesses["magnetic"] and not dataset.mapping.magnetic:
        for idx, column in enumerate(
            [col for col in guesses["magnetic"] if col not in {dataset.mapping.time.column, dataset.mapping.voltage.column, dataset.mapping.current.column}][:3],
            start=1,
        ):
            dataset.mapping.magnetic[f"b{idx}"] = ChannelConfig(column=column, unit="mT")


def repair_role_mapping(
    current_column: str | None,
    candidates: list[str],
    disallowed: set[str | None],
    df: pd.DataFrame,
) -> str | None:
    if current_column and current_column in df.columns and current_column not in disallowed and is_numeric_like(df[current_column]):
        return current_column
    for candidate in candidates:
        if candidate in disallowed:
            continue
        if candidate in df.columns and is_numeric_like(df[candidate]):
            return candidate
    return current_column if current_column not in disallowed else None


def is_numeric_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    return bool(len(series)) and float(numeric.notna().mean()) > 0.8


def save_mapping_pattern(columns: list[str], mapping: ChannelMapping) -> None:
    library = STORE.load_mapping_library()
    library[signature_from_columns(columns)] = mapping.to_dict()
    STORE.save_mapping_library(library)


def dataset_frequency(dataset: DatasetMeta) -> float | None:
    return dataset.request_frequency_hz or dataset.metadata.get("frequency_hz") or dataset.detected_frequency_hz


def match_requests() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for request in st.session_state.request_points:
        matched_dataset = None
        for dataset in st.session_state.datasets:
            freq = dataset_frequency(dataset)
            if freq is None:
                continue
            tolerance = max(0.05, request.frequency_hz * 0.1)
            if abs(freq - request.frequency_hz) <= tolerance:
                matched_dataset = dataset
                break
        status = matched_dataset.status if matched_dataset else request.status
        rows.append(
            {
                "frequency_hz": request.frequency_hz,
                "target_ipp_a": request.target_ipp_a,
                "status": status,
                "dataset": matched_dataset.file_name if matched_dataset else "",
                "linked_dataset_id": matched_dataset.dataset_id if matched_dataset else request.linked_dataset_id,
                "notes": request.notes,
            }
        )
    return pd.DataFrame(rows)


def update_request_points_from_dataframe(df: pd.DataFrame) -> None:
    st.session_state.request_points = [
        RequestPoint(
            frequency_hz=float(row["frequency_hz"]),
            target_ipp_a=float(row["target_ipp_a"]),
            status=str(row["status"]),
            linked_dataset_id=str(row["linked_dataset_id"]) if row.get("linked_dataset_id") else None,
            notes=str(row["notes"]) if row.get("notes") else "",
        )
        for _, row in df.iterrows()
    ]
    persist_state()


def build_measured_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id, result in st.session_state.analysis_results.items():
        dataset = dataset_by_id(dataset_id)
        electrical = result.get("electrical", {})
        magnetic = result.get("magnetic", {})
        rows.append(
            {
                "dataset_id": dataset_id,
                "file_name": dataset.file_name,
                "frequency_hz": result.get("frequency_hz"),
                "|Z1|": electrical.get("|Z1|"),
                "Req": electrical.get("Req"),
                "Leq_H": electrical.get("Leq_H"),
                "B1_pk": magnetic.get("B1_pk"),
                "K_BI": magnetic.get("K_BI"),
                "status": dataset.status,
            }
        )
    return pd.DataFrame(rows)


def standardize_for_dataset(dataset: DatasetMeta) -> pd.DataFrame:
    df = dataframe_for_dataset(dataset)
    auto_apply_mapping(dataset, df)
    return standardize_dataset(df, dataset.mapping)


def run_analysis(dataset: DatasetMeta, representative_channel: str | None = None) -> dict[str, Any] | None:
    try:
        standardized = standardize_for_dataset(dataset)
        analysis = analyze_dataset(
            standardized_df=standardized,
            analysis_window=st.session_state.analysis_window,
            frequency_override_hz=dataset_frequency(dataset),
            metadata=dataset.metadata,
            representative_b_field=representative_channel,
        )
        dataset.detected_frequency_hz = analysis["frequency_hz"]
        dataset.status = "analyzed"
        st.session_state.analysis_results[dataset.dataset_id] = analysis
        persist_state()
        return analysis
    except Exception as exc:
        dataset.last_error = str(exc)
        dataset.status = "flagged"
        persist_state()
        st.error(f"{dataset.file_name}: {exc}")
        LOGGER.exception("Analysis failed for %s", dataset.file_name)
        return None


def add_uploaded_files(uploaded_files: list[Any]) -> None:
    existing_names = {(item.file_name, item.selected_sheet) for item in st.session_state.datasets}
    for uploaded in uploaded_files:
        stored_path = STORE.save_upload_bytes(uploaded.name, uploaded.getvalue())
        sheets = list_excel_sheets(stored_path)
        meta = build_dataset_meta(uuid.uuid4().hex, uploaded.name, stored_path, sheets)
        key = (meta.file_name, meta.selected_sheet)
        if key in existing_names:
            continue
        st.session_state.datasets.append(meta)
        existing_names.add(key)
    persist_state()


def add_example_dataset() -> None:
    df = build_example_waveform()
    stored_path = STORE.save_upload_bytes("example_1Hz_pp20A.csv", df.to_csv(index=False).encode("utf-8"))
    meta = build_dataset_meta(uuid.uuid4().hex, "example_1Hz_pp20A.csv", stored_path, [])
    meta.metadata["frequency_hz"] = 1.0
    meta.metadata["target_ipp_a"] = 20.0
    st.session_state.datasets.append(meta)
    persist_state()


def render_home(reference_statuses: list[Any]) -> None:
    st.header("홈 / 전체 현황")
    render_quick_guide(
        "가장 빠른 사용 순서",
        "1) 데이터 불러오기에서 CSV/XLSX 업로드  2) 채널 매핑/보정에서 시간/전압/전류/B 채널 지정  "
        "3) 신호 분석 실행  4) 전기 분석 또는 Gain 분석에서 결과 확인",
    )
    st.write(
        "코일 / 전자석 시험 데이터를 이용해 large-signal 전기 특성, 자기장 응답, Gain / 구동 요구치를 정리하는 로컬 앱이다. "
        "AE Techron 7224는 CV 모드만 가정하며 CC 모드는 제외했다."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("불러온 데이터", len(st.session_state.datasets))
    col2.metric("분석 완료 데이터", sum(1 for item in st.session_state.datasets if item.status == "analyzed"))
    status_df = match_requests()
    col3.metric("누락 테스트 포인트", int((status_df["status"] == "not tested").sum()))
    st.subheader("Reference 파일 상태")
    ref_df = pd.DataFrame([asdict(item) for item in reference_statuses])
    if not ref_df.empty and "exists" in ref_df:
        ref_df["상태"] = ref_df["exists"].map({True: "찾음", False: "없음"})
    st.dataframe(ref_df, use_container_width=True)
    st.subheader("최근 데이터셋 요약")
    st.dataframe(
        pd.DataFrame(
            [
                {"file_name": item.file_name, "frequency_hz": dataset_frequency(item), "sheet": item.selected_sheet, "status": localized_status(item.status), "notes": item.notes}
                for item in st.session_state.datasets[-10:]
            ]
        ),
        use_container_width=True,
    )
    st.subheader("테스트 완성도")
    if not status_df.empty:
        plot_df = status_df.copy()
        plot_df["status"] = plot_df["status"]
        view_df = status_df.copy()
        view_df["status"] = view_df["status"].map(localized_status)
        st.plotly_chart(status_heatmap(plot_df), use_container_width=True)
        st.dataframe(view_df, use_container_width=True)


def render_import() -> None:
    st.header("데이터 불러오기")
    render_quick_guide(
        "이 페이지에서 할 일",
        "오실로스코프/DAQ CSV 또는 XLSX를 올린 뒤, 파일별로 주파수/목표 전류/Gain 관련 메타데이터를 입력한다. "
        "시간 컬럼 이름이 제각각이어도 다음 단계에서 직접 매핑할 수 있다.",
    )
    st.caption("CSV/XLSX 지원. XLSX는 sheet 선택 가능. 메타데이터가 앞에 붙은 CSV도 자동으로 읽도록 보강했다.")
    uploaded_files = st.file_uploader("Waveform 파일 업로드", type=["csv", "xlsx"], accept_multiple_files=True)
    if uploaded_files:
        add_uploaded_files(uploaded_files)
        st.success(f"Registered {len(uploaded_files)} file(s).")
    if st.button("예제 데이터 불러오기"):
        add_example_dataset()
        st.success("예제 데이터를 추가했다.")
    if not st.session_state.datasets:
        st.info("아직 불러온 waveform 데이터가 없다. CSV/XLSX를 업로드하거나 built-in example을 로드하면 된다.")
        return
    for dataset in st.session_state.datasets:
        with st.expander(dataset.file_name, expanded=False):
            try:
                df = dataframe_for_dataset(dataset)
                dataset.last_error = None
            except Exception as exc:
                dataset.last_error = str(exc)
                dataset.status = "flagged"
                persist_state()
                st.error(f"{dataset.file_name} 미리보기 실패: {exc}")
                st.caption("파일 등록은 유지된다. 나중에 다시 올리거나 CSV 원문 형식을 확인하면 된다.")
                continue
            auto_apply_mapping(dataset, df)
            if dataset.available_sheets:
                selected_sheet = st.selectbox(
                    "시트 선택",
                    dataset.available_sheets,
                    index=dataset.available_sheets.index(dataset.selected_sheet or dataset.available_sheets[0]),
                    key=f"sheet_{dataset.dataset_id}",
                )
                if selected_sheet != dataset.selected_sheet:
                    dataset.selected_sheet = selected_sheet
                    persist_state()
                    st.rerun()
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            dataset.metadata["frequency_hz"] = meta_col1.number_input("주파수 [Hz]", min_value=0.0, value=float(dataset.metadata.get("frequency_hz") or 0.0), key=f"freq_{dataset.dataset_id}") or None
            dataset.metadata["target_ipp_a"] = meta_col2.number_input("목표 전류 pp [A]", min_value=0.0, value=float(dataset.metadata.get("target_ipp_a") or 0.0), key=f"target_{dataset.dataset_id}") or None
            dataset.notes = meta_col3.text_input("비고", value=dataset.notes, key=f"notes_{dataset.dataset_id}")
            meta_col4, meta_col5, meta_col6 = st.columns(3)
            dataset.metadata["measured_vout_pk"] = meta_col4.number_input("실측 Vout_pk [V]", min_value=0.0, value=float(dataset.metadata.get("measured_vout_pk") or 0.0), key=f"vout_{dataset.dataset_id}") or None
            dataset.metadata["gain_mode_v_per_v"] = meta_col5.selectbox("앰프 Gain 모드 [V/V]", [0.0, *list(GAIN_MODES)], index=[0.0, *list(GAIN_MODES)].index(float(dataset.metadata.get("gain_mode_v_per_v") or 0.0)), key=f"gain_{dataset.dataset_id}") or None
            dataset.metadata["vin_pk"] = meta_col6.number_input("입력 사인 Vin_pk [V]", min_value=0.0, value=float(dataset.metadata.get("vin_pk") or DEFAULT_INPUT_VIN_PK), key=f"vin_{dataset.dataset_id}") or None
            st.dataframe(df.head(20), use_container_width=True)
            st.json(
                {
                    "자동 추정 채널": infer_column_roles(list(df.columns)),
                    "자동 추정 시간 단위": infer_time_unit(df[dataset.mapping.time.column]) if dataset.mapping.time.column in df.columns else None,
                    "요약": summarize_channels(df),
                }
            )
    persist_state()


def render_requests() -> None:
    st.header("테스트 요청 관리")
    render_quick_guide("요청표 관리", "기본 preset은 0.25~5 Hz / 20 App 이다. 필요하면 요청표 CSV/XLSX를 올리거나 직접 상태를 수정한다.")
    st.caption("기본 preset: 0.25~5 Hz, 목표 current pp 20 A")
    if st.button("기본 preset으로 초기화"):
        st.session_state.request_points = default_request_points()
        persist_state()
    request_upload = st.file_uploader("요청표 업로드", type=["csv", "xlsx"], key="request_upload")
    if request_upload is not None:
        stored_path = STORE.save_upload_bytes(request_upload.name, request_upload.getvalue())
        request_df = load_frame_cached(str(stored_path), None)
        lower_map = {str(column).lower(): column for column in request_df.columns}
        freq_col = lower_map.get("frequency_hz") or lower_map.get("frequency") or next(iter(request_df.columns))
        target_col = lower_map.get("target_ipp_a") or lower_map.get("target") or freq_col
        st.session_state.request_points = [
            RequestPoint(
                frequency_hz=float(row[freq_col]),
                target_ipp_a=float(row[target_col]),
                status=str(row[lower_map["status"]]) if "status" in lower_map else "not tested",
                notes=str(row[lower_map["notes"]]) if "notes" in lower_map else "",
            )
            for _, row in request_df.iterrows()
        ]
        persist_state()
        st.success("요청표를 불러왔다.")
    status_df = match_requests()
    if not status_df.empty:
        status_df = status_df.copy()
        status_df["status"] = status_df["status"].map(localized_status)
    editor_df = st.data_editor(status_df, num_rows="dynamic", use_container_width=True, key="request_editor")
    if st.button("요청표 저장"):
        reverse_map = {"미시험": "not tested", "데이터 로드됨": "data loaded", "분석 완료": "analyzed", "확인 필요": "flagged"}
        if "status" in editor_df.columns:
            editor_df["status"] = editor_df["status"].replace(reverse_map)
        update_request_points_from_dataframe(editor_df)
        st.success("요청표를 저장했다.")


def render_mapping() -> None:
    st.header("채널 매핑 / 보정")
    render_quick_guide(
        "최소 설정",
        "시간, 전압, 전류 컬럼만 지정해도 전기 분석과 Gain 분석이 가능하다. 자기장 채널은 있으면 추가한다. "
        "시간 컬럼이 datetime 문자열이면 단위를 'datetime'으로 두면 된다.",
    )
    if not st.session_state.datasets:
        st.info("먼저 waveform 데이터를 불러와야 한다.")
        return
    dataset = dataset_by_id(
        st.selectbox(
            "데이터셋 선택",
            options=[item.dataset_id for item in st.session_state.datasets],
            format_func=lambda item: dataset_by_id(item).file_name,
        )
    )
    df = dataframe_for_dataset(dataset)
    auto_apply_mapping(dataset, df)
    columns = [""] + list(df.columns)
    st.subheader("핵심 채널")
    map_col1, map_col2, map_col3 = st.columns(3)
    dataset.mapping.time.column = map_col1.selectbox("시간 컬럼", columns, index=columns.index(dataset.mapping.time.column) if dataset.mapping.time.column in columns else 0) or None
    time_units = ["s", "ms", "us", "datetime"]
    if dataset.mapping.time.column and dataset.mapping.time.column in df.columns and dataset.mapping.time.unit in ("", "s"):
        dataset.mapping.time.unit = infer_time_unit(df[dataset.mapping.time.column])
    dataset.mapping.time.unit = map_col1.selectbox("시간 단위", time_units, index=time_units.index(dataset.mapping.time.unit or "s"))
    dataset.mapping.voltage.column = map_col2.selectbox("전압 컬럼", columns, index=columns.index(dataset.mapping.voltage.column) if dataset.mapping.voltage.column in columns else 0) or None
    dataset.mapping.current.column = map_col3.selectbox("전류 컬럼", columns, index=columns.index(dataset.mapping.current.column) if dataset.mapping.current.column in columns else 0) or None

    magnetic_selection = st.multiselect(
        "자기장 컬럼",
        options=list(df.columns),
        default=[cfg.column for cfg in dataset.mapping.magnetic.values() if cfg.column in df.columns],
    )
    new_magnetic: dict[str, ChannelConfig] = {}
    for idx, column in enumerate(magnetic_selection, start=1):
        existing_key = next((key for key, cfg in dataset.mapping.magnetic.items() if cfg.column == column), f"b{idx}")
        alias = st.text_input(f"{column} 별칭", value=existing_key, key=f"mag_alias_{dataset.dataset_id}_{idx}")
        existing_cfg = dataset.mapping.magnetic.get(existing_key, ChannelConfig(column=column, unit="mT"))
        new_magnetic[alias] = ChannelConfig(
            column=column,
            scale=existing_cfg.scale,
            offset=existing_cfg.offset,
            invert=existing_cfg.invert,
            delay_s=existing_cfg.delay_s,
            unit=existing_cfg.unit or "mT",
        )
    dataset.mapping.magnetic = new_magnetic

    st.subheader("보정값")
    all_configs = [("voltage", dataset.mapping.voltage), ("current", dataset.mapping.current)] + [(key, cfg) for key, cfg in dataset.mapping.magnetic.items()]
    for label, cfg in all_configs:
        c1, c2, c3, c4, c5 = st.columns(5)
        cfg.scale = c1.number_input(f"{label} scale", value=float(cfg.scale), key=f"scale_{dataset.dataset_id}_{label}", help="센서 배율 / 프로브 배율")
        cfg.offset = c2.number_input(f"{label} offset", value=float(cfg.offset), key=f"offset_{dataset.dataset_id}_{label}", help="DC 오프셋 보정")
        cfg.invert = c3.checkbox(f"{label} 반전", value=cfg.invert, key=f"invert_{dataset.dataset_id}_{label}")
        cfg.delay_s = c4.number_input(f"{label} 지연 [s]", value=float(cfg.delay_s), key=f"delay_{dataset.dataset_id}_{label}", format="%.8f", help="수동 시간 보정")
        cfg.unit = c5.text_input(f"{label} 단위", value=cfg.unit, key=f"unit_{dataset.dataset_id}_{label}")

    if dataset.mapping.current.column and st.button("전류 기준 자동 정렬", key=f"auto_align_{dataset.dataset_id}"):
        standardized = standardize_for_dataset(dataset)
        sample_rate = sample_rate_from_time(standardized["time_s"])
        reference = standardized["current_a"].fillna(0.0).to_numpy()
        if "voltage_v" in standardized.columns:
            dataset.mapping.voltage.delay_s = estimate_delay_cross_correlation(reference=reference, target=standardized["voltage_v"].fillna(0.0).to_numpy(), sample_rate_hz=sample_rate)
        for key in dataset.mapping.magnetic:
            column = f"magnetic_{key}"
            if column in standardized.columns:
                dataset.mapping.magnetic[key].delay_s = estimate_delay_cross_correlation(reference=reference, target=standardized[column].fillna(0.0).to_numpy(), sample_rate_hz=sample_rate)
        st.success("cross-correlation 기반 지연 보정을 반영했다.")

    if st.button("매핑 저장", key=f"save_mapping_{dataset.dataset_id}"):
        save_mapping_pattern(list(df.columns), dataset.mapping)
        persist_state()
        st.success("매핑과 보정값을 저장했다.")
    st.dataframe(df.head(20), use_container_width=True)


def render_signal_analysis() -> None:
    st.header("신호 분석")
    render_quick_guide(
        "Gain 분석 전 필수 단계",
        "이 단계에서 fundamental amplitude/phase를 먼저 계산한다. "
        "전압/전류가 매핑되어 있으면 이후 전기 분석과 Gain / 구동 요구 분석이 바로 이어진다.",
    )
    if not st.session_state.datasets:
        st.info("먼저 데이터 업로드와 채널 매핑을 완료해야 한다.")
        return
    dataset = dataset_by_id(
        st.selectbox(
            "분석할 데이터셋",
            [item.dataset_id for item in st.session_state.datasets],
            format_func=lambda item: dataset_by_id(item).file_name,
            key="signal_dataset",
        )
    )
    aw = st.session_state.analysis_window
    s1, s2, s3 = st.columns(3)
    aw.cycle_start = s1.number_input("시작 cycle", min_value=0, value=int(aw.cycle_start), step=1)
    aw.cycle_count = s2.number_input("분석 cycle 수", min_value=1, value=int(aw.cycle_count), step=1)
    freq_override = s3.number_input("주파수 override [Hz]", min_value=0.0, value=float(dataset_frequency(dataset) or 0.0), help="메타데이터를 우선 사용하고, 비어 있으면 current fundamental로 자동 추정한다.")
    opt1, opt2, opt3 = st.columns(3)
    aw.detrend = opt1.checkbox("추세 제거", value=aw.detrend)
    aw.remove_offset = opt2.checkbox("DC offset 제거", value=aw.remove_offset)
    aw.zero_phase_smoothing = opt3.checkbox("Zero-phase smoothing", value=aw.zero_phase_smoothing)
    aw.show_zero_crossing_aux = st.checkbox("zero-crossing 보조 지표 표시", value=aw.show_zero_crossing_aux)

    try:
        standardized = standardize_for_dataset(dataset)
        st.caption(f"표준화된 컬럼: {', '.join(standardized.columns)}")
    except Exception as exc:
        st.error(f"매핑 / 전처리 오류: {exc}")
        return

    representative = None
    magnetic_columns = [column for column in standardized.columns if column.startswith("magnetic_")]
    if magnetic_columns:
        representative = st.selectbox(
            "대표 자기장 채널",
            magnetic_columns,
            index=magnetic_columns.index(st.session_state.settings["representative_b"].get(dataset.dataset_id, magnetic_columns[0]))
            if st.session_state.settings["representative_b"].get(dataset.dataset_id) in magnetic_columns
            else 0,
        )
        st.session_state.settings["representative_b"][dataset.dataset_id] = representative

    if st.button("분석 실행 / 새로고침"):
        if freq_override > 0:
            dataset.request_frequency_hz = freq_override
        result = run_analysis(dataset, representative)
        if result:
            st.success("분석이 완료됐다.")

    result = st.session_state.analysis_results.get(dataset.dataset_id)
    if result is None:
        st.info("분석을 실행하면 fitted waveform과 후속 지표가 채워진다.")
        st.plotly_chart(waveform_figure(standardized), use_container_width=True)
        persist_state()
        return

    fitted = {key: value["fitted"] for key, value in result["signals"].items()}
    mask = (standardized["time_s"] >= result["window_start_s"]) & (standardized["time_s"] <= result["window_end_s"])
    window_df = standardized.loc[mask].copy()
    st.plotly_chart(waveform_figure(window_df, fitted_signals=fitted, title="원파형 + fundamental fit"), use_container_width=True)
    if result["warnings"]:
        st.warning("\n".join(sorted(set(result["warnings"]))))
    signal_table = pd.DataFrame(result["signals"]).T.drop(columns=["fitted", "detrended", "warnings"], errors="ignore")
    st.dataframe(signal_table, use_container_width=True)
    persist_state()


def render_electrical() -> None:
    st.header("전기 분석")
    if not st.session_state.analysis_results:
        st.info("먼저 신호 분석을 실행해야 한다.")
        return
    dataset_id = st.selectbox("분석 완료 데이터셋", list(st.session_state.analysis_results.keys()), format_func=lambda item: dataset_by_id(item).file_name, key="electrical_dataset")
    electrical = st.session_state.analysis_results[dataset_id].get("electrical", {})
    if not electrical:
        st.warning("전기 분석에는 전압과 전류 채널이 모두 필요하다.")
        return
    st.caption("large-signal 실측 결과이며, small-signal LCR reference 와 분리해서 표시한다.")
    st.dataframe(pd.DataFrame([electrical]), use_container_width=True)
    st.plotly_chart(phasor_summary_figure(electrical), use_container_width=True)


def render_magnetic() -> None:
    st.header("자기장 분석")
    if not st.session_state.analysis_results:
        st.info("먼저 신호 분석을 실행해야 한다.")
        return
    dataset_id = st.selectbox("분석 완료 데이터셋", list(st.session_state.analysis_results.keys()), format_func=lambda item: dataset_by_id(item).file_name, key="magnetic_dataset")
    dataset = dataset_by_id(dataset_id)
    standardized = standardize_for_dataset(dataset)
    magnetic = st.session_state.analysis_results[dataset_id].get("magnetic", {})
    if not magnetic:
        st.warning("이 데이터셋에는 자기장 채널이 없다.")
        return
    st.dataframe(pd.DataFrame([magnetic]), use_container_width=True)
    representative = magnetic["representative_channel"]
    if "current_a" in standardized.columns:
        unit = dataset.mapping.magnetic[representative.replace("magnetic_", "")].unit
        st.plotly_chart(loop_figure(standardized["current_a"], standardized[representative], "B-I loop", "Current [A]", f"{representative} [{unit}]"), use_container_width=True)
    if "voltage_v" in standardized.columns:
        st.plotly_chart(loop_figure(standardized["voltage_v"], standardized[representative], "B-V plot", "Voltage [V]", representative), use_container_width=True)


def render_advanced() -> None:
    st.header("고급 분석")
    if not st.session_state.analysis_results:
        st.info("먼저 신호 분석을 실행해야 한다.")
        return
    dataset_id = st.selectbox("분석 완료 데이터셋", list(st.session_state.analysis_results.keys()), format_func=lambda item: dataset_by_id(item).file_name, key="advanced_dataset")
    dataset = dataset_by_id(dataset_id)
    standardized = standardize_for_dataset(dataset)
    if "voltage_v" not in standardized.columns or "current_a" not in standardized.columns:
        st.warning("고급 flux-linkage 분석에는 전압과 전류가 모두 필요하다.")
        return
    rdc = st.number_input("Rdc(T) [ohm]", min_value=0.0, value=0.0, help="별도 측정/입력값이다. LCR series R에서 자동 추정하지 않는다.")
    lambda_metrics = compute_lambda_metrics(
        standardized["time_s"].to_numpy(),
        standardized["voltage_v"].to_numpy(),
        standardized["current_a"].to_numpy(),
        rdc_ohm=rdc,
    )
    st.info("결과 라벨: system-level inferred quantity. 가정과 한계를 함께 봐야 한다.")
    lambda_df = pd.DataFrame({"time_s": standardized["time_s"], "lambda": lambda_metrics["lambda"], "differential_inductance_h": lambda_metrics["differential_inductance_h"]})
    st.plotly_chart(loop_figure(standardized["current_a"], pd.Series(lambda_metrics["lambda"]), "Lambda-I plot", "Current [A]", "Lambda [V*s]"), use_container_width=True)
    st.dataframe(lambda_df.head(200), use_container_width=True)


def render_gain() -> None:
    st.header("Gain / 구동 요구 분석")
    render_quick_guide(
        "이 페이지의 핵심",
        "실측 Vout_pk가 있으면 그것을 우선 사용한다. 없으면 measured |Z1| 와 목표 Ipp로 required Vout_pk를 역산한다. "
        "alpha = Vout_pk / (Vin_pk * Gain_mode) 로 계산하고 1을 넘으면 현재 설정으로 불가능한 포인트로 표시한다.",
    )
    if not st.session_state.analysis_results:
        st.info("먼저 신호 분석을 실행해야 한다.")
        return
    g1, g2 = st.columns(2)
    st.session_state.settings["vin_pk"] = g1.number_input("기본 입력 Vin_pk [V]", min_value=0.1, value=float(st.session_state.settings.get("vin_pk", DEFAULT_INPUT_VIN_PK)))
    st.session_state.settings["gain_mode"] = g2.selectbox("기본 앰프 Gain 모드 [V/V]", list(GAIN_MODES), index=list(GAIN_MODES).index(float(st.session_state.settings.get("gain_mode", GAIN_MODES[0]))))
    rows: list[dict[str, Any]] = []
    for dataset_id, analysis in st.session_state.analysis_results.items():
        dataset = dataset_by_id(dataset_id)
        electrical = analysis.get("electrical", {})
        if not electrical:
            continue
        target_ipp = float(dataset.request_target_ipp_a or dataset.metadata.get("target_ipp_a") or DEFAULT_TARGET_IPP_A)
        measured_vout_pk = dataset.metadata.get("measured_vout_pk") or electrical.get("V1_pk")
        gain_mode = float(dataset.metadata.get("gain_mode_v_per_v") or st.session_state.settings["gain_mode"])
        vin_pk = float(dataset.metadata.get("vin_pk") or st.session_state.settings["vin_pk"])
        source = "실측 Vout" if dataset.metadata.get("measured_vout_pk") else "파형 분석 V1_pk"
        rows.append(
            {
                **compute_gain_requirement(
                frequency_hz=float(analysis["frequency_hz"]),
                target_ipp_a=target_ipp,
                electrical_metrics=electrical,
                achieved_ipp_a=float(electrical["I1_pk"] * 2.0) if electrical.get("I1_pk") is not None else None,
                measured_vout_pk=measured_vout_pk,
                gain_mode_v_per_v=gain_mode,
                vin_pk=vin_pk,
                notes=dataset.notes,
                ),
                "vout_source": source,
                "dataset": dataset.file_name,
            }
        )
    if not rows:
        st.warning("Gain 계산에 필요한 전압/전류 분석 결과가 없다.")
        return
    gain_df = pd.DataFrame(rows).sort_values("frequency_hz")
    st.dataframe(gain_df, use_container_width=True)
    st.plotly_chart(frequency_summary_figure(gain_df, "required_alpha_pct", "주파수별 required alpha [%]"), use_container_width=True)
    persist_state()


def render_reference(reference_statuses: list[Any]) -> None:
    st.header("LCR Reference 비교")
    measured_df = build_measured_summary()
    reference_file = next((item for item in reference_statuses if item.name == "all_bands_full.xlsx" and item.exists), None)
    workbook_path = st.session_state.settings.get("reference_workbook_path") or (reference_file.path if reference_file else None)
    uploaded_reference = st.file_uploader("reference workbook 직접 업로드", type=["xlsx"], key="reference_upload")
    if uploaded_reference is not None:
        stored_path = STORE.save_upload_bytes(uploaded_reference.name, uploaded_reference.getvalue())
        workbook_path = str(stored_path)
        st.session_state.settings["reference_workbook_path"] = workbook_path
        persist_state()
    if not workbook_path:
        st.warning("reference workbook을 찾지 못했다. small-signal 비교 없이 계속 진행한다.")
        return
    sheet_names, sheets = load_reference_workbook(Path(workbook_path))
    if not sheet_names:
        st.warning("reference workbook을 읽지 못했다.")
        return
    selected_sheet = st.selectbox("reference 시트", sheet_names)
    reference_df = sheets[selected_sheet]
    st.json(summarize_reference_sheet(reference_df))
    detected = infer_reference_columns(reference_df)
    frequency_col = st.selectbox("reference 주파수 컬럼", reference_df.columns, index=list(reference_df.columns).index(detected["frequency"]) if detected["frequency"] in reference_df.columns else 0)
    ref_plot_df = pd.DataFrame({"frequency_hz": reference_df[frequency_col]})
    for label, detected_col, measured_col, title in [
        ("reference |Z| 컬럼", detected["z"], "|Z1|", "Large-signal |Z| vs small-signal reference"),
        ("reference L 컬럼", detected["l"], "Leq_H", "Large-signal L vs small-signal reference"),
        ("reference R / loss 컬럼", detected["r"], "Req", "Large-signal R vs small-signal reference"),
    ]:
        selected = st.selectbox(label, [""] + list(reference_df.columns), index=([""] + list(reference_df.columns)).index(detected_col) if detected_col in reference_df.columns else 0, key=f"ref_{measured_col}")
        if selected:
            alias = f"ref_{measured_col}"
            ref_plot_df[alias] = reference_df[selected]
            st.plotly_chart(reference_comparison_figure(measured_df, ref_plot_df, measured_col, alias, title), use_container_width=True)
    st.dataframe(reference_df.head(200), use_container_width=True)


def render_export(reference_statuses: list[Any]) -> None:
    st.header("내보내기")
    measured_df = build_measured_summary()
    status_df = match_requests()
    electrical_df = pd.DataFrame([{"dataset_id": dataset_id, **result.get("electrical", {})} for dataset_id, result in st.session_state.analysis_results.items() if result.get("electrical")])
    magnetic_df = pd.DataFrame([{"dataset_id": dataset_id, **result.get("magnetic", {})} for dataset_id, result in st.session_state.analysis_results.items() if result.get("magnetic")])
    gain_df = pd.DataFrame(
        [
            compute_gain_requirement(
                frequency_hz=float(result["frequency_hz"]),
                target_ipp_a=float(dataset_by_id(dataset_id).metadata.get("target_ipp_a") or DEFAULT_TARGET_IPP_A),
                electrical_metrics=result.get("electrical", {}),
                achieved_ipp_a=float(result.get("electrical", {}).get("I1_pk", 0.0) * 2.0),
                measured_vout_pk=result.get("electrical", {}).get("V1_pk"),
                gain_mode_v_per_v=float(st.session_state.settings.get("gain_mode", GAIN_MODES[0])),
                vin_pk=float(st.session_state.settings.get("vin_pk", DEFAULT_INPUT_VIN_PK)),
                notes=dataset_by_id(dataset_id).notes,
            )
            for dataset_id, result in st.session_state.analysis_results.items()
            if result.get("electrical")
        ]
    )
    workbook_sheets = {
        "summary": measured_df,
        "per_test_result": status_df,
        "electrical_metrics": electrical_df,
        "magnetic_metrics": magnetic_df,
        "gain_requirement": gain_df,
        "missing_test_points": status_df.loc[status_df["status"] == "not tested"],
    }
    figures: dict[str, Any] = {}
    if not gain_df.empty:
        figures["gain_requirement"] = frequency_summary_figure(gain_df, "required_alpha_pct", "Required alpha [%]")
    if not status_df.empty:
        figures["test_status"] = status_heatmap(status_df)
    settings_payload = {
        "analysis_window": st.session_state.analysis_window.to_dict(),
        "settings": st.session_state.settings,
        "datasets": [item.to_dict() for item in st.session_state.datasets],
        "reference_status": [asdict(item) for item in reference_statuses],
    }
    bundle = build_export_bundle(workbook_sheets, figures, settings_payload)
    st.download_button("분석 결과 bundle 다운로드 (Excel + HTML + JSON)", data=bundle, file_name="coil_analysis_export.zip", mime="application/zip")
    st.download_button("설정 JSON 다운로드", data=json.dumps(settings_payload, ensure_ascii=False, indent=2), file_name="coil_analysis_settings.json", mime="application/json")
    imported_json = st.file_uploader("설정 JSON 불러오기", type=["json"], key="settings_upload")
    if imported_json is not None:
        payload = json.loads(imported_json.getvalue().decode("utf-8"))
        st.session_state.analysis_window = AnalysisWindow.from_dict(payload.get("analysis_window", {}))
        st.session_state.settings.update(payload.get("settings", {}))
        persist_state()
        st.success("설정을 불러왔다.")


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide")
    ensure_state()
    reference_statuses = discover_reference_files(ROOT)
    st.sidebar.title("코일 분석 앱")
    st.sidebar.caption("CV 모드 전용 워크플로우. CC 모드는 제외.")
    page = st.sidebar.radio("단계 선택", PAGE_LABELS)
    st.sidebar.progress(page_index(page) / len(PAGE_LABELS), text=f"현재 단계 {page_index(page)} / {len(PAGE_LABELS)}")
    st.sidebar.write("작업 폴더", str(ROOT))
    st.sidebar.write("등록 데이터", len(st.session_state.datasets))
    st.sidebar.write("분석 완료", sum(1 for item in st.session_state.datasets if item.status == "analyzed"))
    if page == "1. 홈 / 전체 현황":
        render_home(reference_statuses)
    elif page == "2. 데이터 불러오기":
        render_import()
    elif page == "3. 테스트 요청 관리":
        render_requests()
    elif page == "4. 채널 매핑 / 보정":
        render_mapping()
    elif page == "5. 신호 분석":
        render_signal_analysis()
    elif page == "6. 전기 분석":
        render_electrical()
    elif page == "7. 자기장 분석":
        render_magnetic()
    elif page == "8. 고급 분석":
        render_advanced()
    elif page == "9. Gain / 구동 요구 분석":
        render_gain()
    elif page == "10. LCR 비교":
        render_reference(reference_statuses)
    elif page == "11. 내보내기":
        render_export(reference_statuses)


if __name__ == "__main__":
    main()
