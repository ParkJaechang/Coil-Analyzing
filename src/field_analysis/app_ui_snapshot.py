from __future__ import annotations

"""Snapshot copy of the field-analysis Streamlit UI.

This file is an explicit duplicate of ``app_ui.py`` kept as a separate
entrypoint target so future UI changes can be tested without immediately
touching the main app shell.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .analysis import analyze_measurements, build_shape_phase_comparison, build_warning_table, combine_analysis_frames
from .compensation import synthesize_current_waveform_compensation
from .control_formula import build_control_formula
from .dataset_library import (
    build_dataset_manifest,
    get_dataset_manifest_path,
    get_default_settings_path,
    load_dataset_library_settings,
    load_dataset_manifest,
    save_dataset_library_settings,
)
from .exports import build_export_zip_bytes, export_analysis_bundle
from .hardware import apply_command_hardware_model
from .lut import TARGET_LABELS, prioritize_lut_target_metrics, recommend_voltage_waveform, target_metric_label
from .metrics import build_calculation_details, estimate_drive_for_target_field
from .models import CycleDetectionConfig, PreprocessConfig
from .parser import build_mapping_table, parse_measurement_file, preview_measurement_file
from .plotting import (
    plot_command_waveform,
    plot_current_compensation_waveforms,
    plot_formula_comparison,
    plot_frequency_support_curve,
    plot_output_compensation_waveforms,
    plot_coverage_matrix,
    plot_cycle_detection_overlay,
    plot_cycle_overlay,
    plot_drift,
    plot_frequency_comparison,
    plot_lut_lookup_curve,
    plot_loop,
    plot_metric_heatmap,
    plot_operating_map,
    plot_shape_metric_trend,
    plot_shape_overlay,
    plot_temperature_vs_drift,
    plot_waveforms,
)
from .preprocessing import apply_preprocessing
from .schema_config import dump_schema_yaml, load_schema_config
from .ui_upload_state import category_payloads, list_persisted_uploads, render_sidebar_memory_panel, render_workspace_panel
from .ui_validation_retune import render_catalogs_and_diagnostics_section, render_validation_retune_section
from .utils import first_number, infer_current_from_text, infer_frequency_from_text, infer_waveform_from_text


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "excel_mapping_template.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "field_analysis_export"
DEFAULT_QUICK_OUTPUT_DIR = REPO_ROOT / "outputs" / "field_analysis_quick_export"


@st.cache_data(show_spinner=False)
def _preview_file_cached(file_name: str, file_bytes: bytes, config_path: str | None) -> object:
    schema = load_schema_config(config_path)
    return preview_measurement_file(file_name=file_name, file_bytes=file_bytes, schema=schema)


@st.cache_data(show_spinner=False)
def _parse_file_cached(
    file_name: str,
    file_bytes: bytes,
    config_path: str | None,
    mapping_overrides_json: str,
    metadata_overrides_json: str,
    expected_cycles: int,
    target_current_mode: str,
) -> object:
    schema = load_schema_config(config_path)
    mapping_overrides = json.loads(mapping_overrides_json)
    metadata_overrides = json.loads(metadata_overrides_json)
    return parse_measurement_file(
        file_name=file_name,
        file_bytes=file_bytes,
        schema=schema,
        mapping_overrides=mapping_overrides,
        metadata_overrides=metadata_overrides,
        expected_cycles=expected_cycles,
        target_current_mode=target_current_mode,
    )


@st.cache_data(show_spinner=False)
def _analyze_measurements_cached(
    parsed_measurements: list,
    preprocess_config: PreprocessConfig,
    cycle_config: CycleDetectionConfig,
    current_channel: str,
    main_field_axis: str,
) -> object:
    return analyze_measurements(
        parsed_measurements=parsed_measurements,
        preprocess_config=preprocess_config,
        cycle_config=cycle_config,
        current_channel=current_channel,
        main_field_axis=main_field_axis,
    )


@st.cache_data(show_spinner=False)
def _preprocess_measurements_cached(
    parsed_measurements: list,
    preprocess_config: PreprocessConfig,
) -> list:
    return [apply_preprocessing(parsed.normalized_frame, preprocess_config) for parsed in parsed_measurements]


def _render_dataset_library_panel() -> None:
    settings = load_dataset_library_settings()
    session_key = "dataset_library_root"
    if session_key not in st.session_state:
        st.session_state[session_key] = str(settings.get("dataset_root") or "")

    with st.expander("Dataset Library", expanded=False):
        dataset_root_value = str(
            st.text_input(
                "Dataset root path",
                key=session_key,
                placeholder="D:/OneDrive/CoilDatasets",
                help="Save a shared sync-folder path such as OneDrive, Google Drive, Dropbox, NAS, or external SSD.",
            )
            or ""
        ).strip()
        current_manifest = None
        action_left, action_right = st.columns(2)
        if action_left.button("Save Root", use_container_width=True, key="dataset_library_save"):
            save_dataset_library_settings({"dataset_root": dataset_root_value})
            st.success("Dataset root saved.")
        if action_right.button("Manifest Refresh", use_container_width=True, key="dataset_library_refresh"):
            if not dataset_root_value:
                st.warning("Enter a dataset root path first.")
            else:
                try:
                    with st.spinner("Building dataset manifest..."):
                        current_manifest = build_dataset_manifest(dataset_root_value)
                    save_dataset_library_settings({"dataset_root": dataset_root_value})
                    st.success("Dataset manifest refreshed.")
                except (FileNotFoundError, NotADirectoryError) as exc:
                    st.error(str(exc))

        saved_settings_path = get_default_settings_path()
        st.caption(f"settings: {saved_settings_path}")
        if not dataset_root_value:
            st.caption("No dataset root is saved.")
            return

        manifest_path = get_dataset_manifest_path(dataset_root_value)
        if current_manifest is None:
            current_manifest = load_dataset_manifest(dataset_root_value)

        count_left, count_right = st.columns(2)
        count_left.metric("Registered Files", int(current_manifest.get("file_count") or 0))
        count_right.metric("Continuous", int(current_manifest.get("counts", {}).get("continuous") or 0))
        count_left.metric("Finite Cycle", int(current_manifest.get("counts", {}).get("finite_cycle") or 0))
        count_right.metric("Unknown", int(current_manifest.get("counts", {}).get("unknown") or 0))
        st.caption(f"root: {dataset_root_value}")
        if manifest_path.exists():
            st.caption(f"manifest: {manifest_path}")
        else:
            st.caption("Manifest has not been generated yet.")


def _run_app_shell(
    *,
    page_title: str,
    title: str,
    caption: str,
    initial_usage_mode: str,
    lock_usage_mode: bool,
    default_output_dir: Path,
) -> None:
    """Render the Streamlit engineering analysis app."""

    st.set_page_config(
        page_title=page_title,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title(title)
    st.caption(caption)
    render_workspace_panel()

    config_path = str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None
    schema = load_schema_config(config_path)

    with st.sidebar:
        st.header("입력")
        render_sidebar_memory_panel()
        _render_dataset_library_panel()
        continuous_files = st.file_uploader(
            "연속 cycle 데이터 업로드",
            type=["csv", "txt", "xlsx", "xlsm", "xls"],
            accept_multiple_files=True,
            key="continuous_uploads",
            help="현재 분석과 LUT의 기본 입력입니다.",
        )
        transient_files = st.file_uploader(
            "finite-cycle 전용 데이터 업로드",
            type=["csv", "txt", "xlsx", "xlsm", "xls"],
            accept_multiple_files=True,
            key="transient_uploads",
            help="1 cycle, 0.75 cycle, 1.25 cycle 같은 짧은 구동 데이터를 분리 보관합니다.",
        )
        validation_files = st.file_uploader(
            "2차 보정 검증 run 업로드",
            type=["csv", "txt", "xlsx", "xlsm", "xls"],
            accept_multiple_files=True,
            key="validation_uploads",
            help="추천 LUT/보정 결과를 실제 측정과 다시 비교하는 validation run 데이터를 적재합니다.",
        )
        lcr_files = st.file_uploader(
            "LCR 데이터 업로드",
            type=["csv", "txt", "xlsx", "xlsm", "xls"],
            accept_multiple_files=True,
            key="lcr_uploads",
            help="LCR 파일은 자동 기억 목록과 업로드 폴더 요약에 함께 남깁니다.",
        )
        if lock_usage_mode:
            usage_mode = initial_usage_mode
            st.caption(f"사용 모드: {usage_mode}")
        else:
            usage_mode = st.radio(
                "사용 모드",
                options=["간단 LUT", "전체 분석"],
                index=0 if initial_usage_mode == "간단 LUT" else 1,
                help="간단 LUT는 목표 자기장/전류 -> 추천 전압 파형 중심 화면입니다.",
            )

        st.header("핵심 설정")
        expected_cycles = int(
            st.number_input(
                "기대 cycle 수",
                min_value=1,
                max_value=100,
                value=schema.default_expected_cycles,
                step=1,
            )
        )
        target_current_mode = st.selectbox(
            "타겟 전류 해석",
            options=["auto", "peak", "pp"],
            index=["auto", "peak", "pp"].index(schema.target_current_mode)
            if schema.target_current_mode in {"auto", "peak", "pp"}
            else 0,
            help="파일 메타데이터의 Target Current(A)를 peak로 볼지, pp로 볼지, 자동 추론할지 선택합니다.",
        )
        current_channel = st.selectbox(
            "대표 전류 축",
            options=[
                "i_sum_signed",
                "i_custom_signed",
                "coil2_current_signed_a",
                "coil1_current_signed_a",
                "i_diff_signed",
                "i_sum",
                "i_custom",
                "coil1_current_a",
                "coil2_current_a",
                "i_diff",
            ],
            index=0,
        )
        main_field_axis = st.selectbox(
            "대표 자기장 축",
            options=["bz_mT", "bmag_mT", "bx_mT", "by_mT", "bproj_mT"],
            index=0,
        )
        st.subheader("구동 하드웨어")
        max_daq_voltage_pp = float(
            st.number_input(
                "DAQ 최대 Voltage PP (V)",
                min_value=0.1,
                value=20.0,
                step=1.0,
            )
        )
        amp_gain_at_100_pct = float(
            st.number_input(
                "DC AMP gain @100% (x)",
                min_value=0.1,
                value=20.0,
                step=0.5,
            )
        )
        amp_max_output_pk_v = float(
            st.number_input(
                "DC AMP 최대 출력 (±V)",
                min_value=1.0,
                value=180.0,
                step=10.0,
            )
        )
        amp_gain_limit_pct = float(
            st.number_input(
                "사용 가능 AMP gain 상한 (%)",
                min_value=1.0,
                max_value=100.0,
                value=100.0,
                step=5.0,
            )
        )
        default_support_amp_gain_pct = float(
            st.number_input(
                "데이터 기준 AMP gain (%)",
                min_value=1.0,
                max_value=100.0,
                value=100.0,
                step=5.0,
                help="파일에 amp_gain_setting이 없을 때 기준값으로 사용합니다.",
            )
        )
        allow_target_extrapolation = st.checkbox(
            "실험 범위 밖 target extrapolation 허용",
            value=True,
            help="실험 support 범위를 넘어도 하드웨어 headroom을 고려해 전압을 외삽합니다.",
        )

        with st.expander("고급 설정", expanded=usage_mode == "전체 분석"):
            baseline_seconds = float(st.number_input("Baseline 구간 (초)", min_value=0.0, value=0.0, step=0.1))
            smoothing_method = st.selectbox(
                "Smoothing",
                options=["none", "moving_average", "savitzky_golay"],
                format_func=lambda value: {
                    "none": "사용 안 함",
                    "moving_average": "Moving Average",
                    "savitzky_golay": "Savitzky-Golay",
                }[value],
            )
            smoothing_window = int(st.slider("Smoothing window", min_value=3, max_value=101, value=11, step=2))
            savgol_polyorder = int(st.slider("SG polyorder", min_value=1, max_value=5, value=2))
            outlier_threshold = float(
                st.number_input("Outlier z-score threshold (0=off)", min_value=0.0, value=0.0, step=0.5)
            )
            custom_current_alpha = float(st.number_input("사용자 정의 전류 alpha", value=1.0, step=0.1))
            custom_current_beta = float(st.number_input("사용자 정의 전류 beta", value=1.0, step=0.1))
            projection_nx = float(st.number_input("Bproj nx", value=0.0, step=0.1))
            projection_ny = float(st.number_input("Bproj ny", value=0.0, step=0.1))
            projection_nz = float(st.number_input("Bproj nz", value=1.0, step=0.1))
            sign_flip_selection = st.multiselect(
                "Sign flip 채널",
                options=["coil1_current_a", "coil2_current_a", "bx_mT", "by_mT", "bz_mT"],
            )
            apply_alignment = st.checkbox("Cross-correlation 시차 자동 보정 적용", value=False)
            cycle_reference_channel = st.selectbox(
                "Cycle detection 기준 채널",
                options=[
                    "daq_input_v",
                    "i_sum_signed",
                    "i_custom_signed",
                    "coil2_current_signed_a",
                    "coil1_current_signed_a",
                    "i_sum",
                    "i_custom",
                    "coil1_current_a",
                    "coil2_current_a",
                    "bz_mT",
                ],
                index=0,
            )
            alignment_reference = st.selectbox(
                "정렬 기준 채널",
                options=[
                    "daq_input_v",
                    "i_sum_signed",
                    "i_custom_signed",
                    "coil2_current_signed_a",
                    "coil1_current_signed_a",
                    "i_sum",
                    "i_custom",
                    "coil1_current_a",
                    "coil2_current_a",
                    "bz_mT",
                ],
                index=0,
            )
            alignment_targets = st.multiselect(
                "정렬 대상 채널",
                options=["coil1_current_a", "coil2_current_a", "bx_mT", "by_mT", "bz_mT", "daq_input_v"],
                default=["bx_mT", "by_mT", "bz_mT"],
            )
            manual_start_s = st.text_input("수동 cycle 시작 (초, 비우면 자동)", value="")
            manual_period_s = st.text_input("수동 cycle 주기 (초, 비우면 자동)", value="")
        if usage_mode == "간단 LUT":
            baseline_seconds = 0.0
            smoothing_method = "none"
            smoothing_window = 11
            savgol_polyorder = 2
            outlier_threshold = 0.0
            custom_current_alpha = 1.0
            custom_current_beta = 1.0
            projection_nx, projection_ny, projection_nz = 0.0, 0.0, 1.0
            sign_flip_selection = []
            apply_alignment = False
            cycle_reference_channel = "daq_input_v"
            alignment_reference = "daq_input_v"
            alignment_targets = ["bx_mT", "by_mT", "bz_mT"]
            manual_start_s = ""
            manual_period_s = ""

    if usage_mode == "간단 LUT":
        active_section = st.radio(
            "화면",
            options=["Quick LUT", "Validation / Retune", "Catalogs / Diagnostics", "Finite Runs", "Raw Waveforms", "Data Import", "Export"],
            horizontal=True,
            key="quick_section_nav",
        )
    else:
        active_section = st.radio(
            "분석 화면",
            options=[
                "Data Import",
                "Validation / Retune",
                "Catalogs / Diagnostics",
                "Raw Waveforms",
                "Cycle Overlay",
                "Loop Analysis",
                "Frequency/Amplitude Comparison",
                "Thermal / Drift",
                "Operating Map",
                "Calculation Details",
                "Export",
            ],
            horizontal=True,
            key="full_section_nav",
        )

    uploaded_payloads = category_payloads("continuous", continuous_files)
    transient_payloads = category_payloads("transient", transient_files)
    validation_payloads = category_payloads("validation", validation_files)
    lcr_payloads = category_payloads("lcr", lcr_files)
    lcr_records = list_persisted_uploads("lcr")

    if not uploaded_payloads and not transient_payloads and not validation_payloads and not lcr_payloads:
        st.info("CSV 또는 Excel 파일을 업로드하면 구조 인식과 분석이 시작됩니다.")
        sample_doc = REPO_ROOT / "docs" / "sample_data_structure.md"
        if sample_doc.exists():
            st.markdown(sample_doc.read_text(encoding="utf-8"))
        return

    with st.spinner("업로드 파일 구조를 확인하는 중입니다..."):
        previews = [
            _preview_file_cached(file_name, file_bytes, config_path)
            for file_name, file_bytes in uploaded_payloads
        ]
        transient_previews = [
            _preview_file_cached(file_name, file_bytes, config_path)
            for file_name, file_bytes in transient_payloads
        ]
        validation_previews = [
            _preview_file_cached(file_name, file_bytes, config_path)
            for file_name, file_bytes in validation_payloads
        ]

    editable_metadata = _build_metadata_editor_rows(previews)
    transient_edited_metadata = _build_metadata_editor_rows(transient_previews)
    validation_edited_metadata = _build_metadata_editor_rows(validation_previews)
    if not editable_metadata.empty:
        st.subheader("연속 cycle 메타데이터 편집")
        edited_metadata = st.data_editor(
            editable_metadata,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key="metadata_editor",
        )
    else:
        edited_metadata = pd.DataFrame(columns=["source_file", "sheet_name", "waveform_type", "freq_hz", "target_current_a", "notes"])

    if not transient_edited_metadata.empty:
        st.subheader("finite-cycle 메타데이터 편집")
        transient_edited_metadata = st.data_editor(
            transient_edited_metadata,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key="transient_metadata_editor",
        )
    if not validation_edited_metadata.empty:
        st.subheader("2차 보정 검증 run 메타데이터 편집")
        validation_edited_metadata = st.data_editor(
            validation_edited_metadata,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key="validation_metadata_editor",
        )
    mapping_source_previews = previews or transient_previews or validation_previews
    mapping_overrides = _render_mapping_editor(schema=schema, previews=mapping_source_previews) if mapping_source_previews else {}
    metadata_override_by_file = _group_metadata_overrides(edited_metadata)
    transient_metadata_override_by_file = _group_metadata_overrides(transient_edited_metadata)
    validation_metadata_override_by_file = _group_metadata_overrides(validation_edited_metadata)

    parsed_measurements = []
    if uploaded_payloads:
        with st.spinner("연속 cycle 데이터를 정규화하는 중입니다..."):
            for file_name, file_bytes in uploaded_payloads:
                file_preview = next(preview for preview in previews if preview.file_name == file_name)
                file_mapping = {
                    sheet_preview.sheet_name: mapping_overrides
                    for sheet_preview in file_preview.sheet_previews
                }
                file_metadata = metadata_override_by_file.get(file_name, {})
                parsed_measurements.extend(
                    _parse_file_cached(
                        file_name=file_name,
                        file_bytes=file_bytes,
                        config_path=config_path,
                        mapping_overrides_json=json.dumps(file_mapping, ensure_ascii=False),
                        metadata_overrides_json=json.dumps(file_metadata, ensure_ascii=False),
                        expected_cycles=expected_cycles,
                        target_current_mode=target_current_mode,
                    )
                )

    transient_measurements = []
    if transient_payloads:
        with st.spinner("finite-cycle 데이터를 정규화하는 중입니다..."):
            for file_name, file_bytes in transient_payloads:
                file_preview = next(preview for preview in transient_previews if preview.file_name == file_name)
                file_mapping = {
                    sheet_preview.sheet_name: mapping_overrides
                    for sheet_preview in file_preview.sheet_previews
                }
                file_metadata = transient_metadata_override_by_file.get(file_name, {})
                transient_measurements.extend(
                    _parse_file_cached(
                        file_name=file_name,
                        file_bytes=file_bytes,
                        config_path=config_path,
                        mapping_overrides_json=json.dumps(file_mapping, ensure_ascii=False),
                        metadata_overrides_json=json.dumps(file_metadata, ensure_ascii=False),
                        expected_cycles=max(1, expected_cycles),
                        target_current_mode=target_current_mode,
                    )
                )

    validation_measurements = []
    if validation_payloads:
        with st.spinner("validation run 데이터를 정규화하는 중입니다..."):
            for file_name, file_bytes in validation_payloads:
                file_preview = next(preview for preview in validation_previews if preview.file_name == file_name)
                file_mapping = {
                    sheet_preview.sheet_name: mapping_overrides
                    for sheet_preview in file_preview.sheet_previews
                }
                file_metadata = validation_metadata_override_by_file.get(file_name, {})
                validation_measurements.extend(
                    _parse_file_cached(
                        file_name=file_name,
                        file_bytes=file_bytes,
                        config_path=config_path,
                        mapping_overrides_json=json.dumps(file_mapping, ensure_ascii=False),
                        metadata_overrides_json=json.dumps(file_metadata, ensure_ascii=False),
                        expected_cycles=max(1, expected_cycles),
                        target_current_mode=target_current_mode,
                    )
                )

    if not uploaded_payloads and transient_payloads and active_section not in {"Finite Runs", "Data Import"}:
        st.info("현재는 finite-cycle 데이터만 올라와 있습니다. `Finite Runs` 화면에서 정지 응답을 확인하거나 `Data Import`에서 파싱 상태를 먼저 확인하십시오.")

    if not uploaded_payloads and not transient_payloads and not validation_payloads and not lcr_payloads:
        st.info("CSV 또는 Excel 파일을 업로드하면 구조 인식과 분석이 시작됩니다.")
        sample_doc = REPO_ROOT / "docs" / "sample_data_structure.md"
        if sample_doc.exists():
            st.markdown(sample_doc.read_text(encoding="utf-8"))
        return

    preprocess_config = PreprocessConfig(
        baseline_seconds=baseline_seconds,
        smoothing_method=smoothing_method,
        smoothing_window=smoothing_window,
        savgol_polyorder=savgol_polyorder,
        alignment_reference=alignment_reference,
        alignment_targets=tuple(alignment_targets),
        apply_alignment=apply_alignment,
        outlier_zscore_threshold=outlier_threshold,
        sign_flips={channel: -1 for channel in sign_flip_selection},
        custom_current_alpha=custom_current_alpha,
        custom_current_beta=custom_current_beta,
        projection_vector=(projection_nx, projection_ny, projection_nz),
    )
    cycle_config = CycleDetectionConfig(
        reference_channel=cycle_reference_channel,
        expected_cycles=expected_cycles,
        manual_start_s=first_number(manual_start_s),
        manual_period_s=first_number(manual_period_s),
    )

    transient_preprocess_results = []
    if transient_measurements:
        with st.spinner("finite-cycle corrected 데이터를 준비하는 중입니다..."):
            transient_preprocess_results = _preprocess_measurements_cached(
                parsed_measurements=transient_measurements,
                preprocess_config=preprocess_config,
            )

    validation_preprocess_results = []
    if validation_measurements:
        with st.spinner("validation run corrected 데이터를 준비하는 중입니다..."):
            validation_preprocess_results = _preprocess_measurements_cached(
                parsed_measurements=validation_measurements,
                preprocess_config=preprocess_config,
            )

    if not uploaded_payloads:
        if active_section == "Finite Runs":
            _render_finite_run_section(
                transient_measurements=transient_measurements,
                transient_preprocess_results=transient_preprocess_results,
                current_channel=current_channel,
                main_field_axis=main_field_axis,
            )
        else:
            _render_data_import_tab(
                previews=[],
                parsed_measurements=[],
                edited_metadata=edited_metadata,
                warning_table=pd.DataFrame(),
                transient_previews=transient_previews,
                transient_parsed_measurements=transient_measurements,
                transient_edited_metadata=transient_edited_metadata,
                validation_previews=validation_previews,
                validation_parsed_measurements=validation_measurements,
                validation_edited_metadata=validation_edited_metadata,
                lcr_uploads=lcr_records,
            )
        return

    with st.spinner("cycle 분석과 measured field 요약을 계산하는 중입니다..."):
        analyses = _analyze_measurements_cached(
            parsed_measurements=parsed_measurements,
            preprocess_config=preprocess_config,
            cycle_config=cycle_config,
            current_channel=current_channel,
            main_field_axis=main_field_axis,
        )
    analysis_lookup = {
        analysis.parsed.normalized_frame["test_id"].iloc[0]: analysis
        for analysis in analyses
        if not analysis.parsed.normalized_frame.empty
    }

    reference_options = ["없음"] + sorted(analysis_lookup.keys())
    reference_test_id = st.selectbox("Reference test", options=reference_options, index=0)
    reference_test_value = None if reference_test_id == "없음" else reference_test_id

    per_cycle_summary, per_test_summary, coverage = combine_analysis_frames(
        analyses=analyses,
        reference_test_id=reference_test_value,
        field_axis=main_field_axis,
    )
    warning_table = build_warning_table(analyses)

    test_ids = sorted(analysis_lookup.keys())
    if not test_ids:
        st.error("분석 가능한 테스트가 없습니다. 매핑과 메타데이터를 확인하십시오.")
        return

    if usage_mode == "간단 LUT":
        if active_section == "Quick LUT":
            _render_quick_lut_tab_v2(
                per_test_summary=per_test_summary,
                analysis_lookup=analysis_lookup,
                main_field_axis=main_field_axis,
                current_channel=current_channel,
                max_daq_voltage_pp=max_daq_voltage_pp,
                amp_gain_at_100_pct=amp_gain_at_100_pct,
                amp_gain_limit_pct=amp_gain_limit_pct,
                amp_max_output_pk_v=amp_max_output_pk_v,
                default_support_amp_gain_pct=default_support_amp_gain_pct,
                allow_target_extrapolation=allow_target_extrapolation,
                transient_measurements=transient_measurements,
                transient_preprocess_results=transient_preprocess_results,
            )
        elif active_section == "Validation / Retune":
            render_validation_retune_section(
                current_channel=current_channel,
                field_channel=main_field_axis,
                max_daq_voltage_pp=max_daq_voltage_pp,
                amp_gain_at_100_pct=amp_gain_at_100_pct,
                amp_gain_limit_pct=amp_gain_limit_pct,
                amp_max_output_pk_v=amp_max_output_pk_v,
                default_support_amp_gain_pct=default_support_amp_gain_pct,
                validation_measurements=validation_measurements,
                validation_preprocess_results=validation_preprocess_results,
            )
        elif active_section == "Catalogs / Diagnostics":
            render_catalogs_and_diagnostics_section()
        elif active_section == "Finite Runs":
            _render_finite_run_section(
                transient_measurements=transient_measurements,
                transient_preprocess_results=transient_preprocess_results,
                current_channel=current_channel,
                main_field_axis=main_field_axis,
            )
        elif active_section == "Raw Waveforms":
            _render_raw_waveforms_tab(
                test_ids=test_ids,
                analysis_lookup=analysis_lookup,
            )
        elif active_section == "Data Import":
            _render_data_import_tab(
                previews=previews,
                parsed_measurements=parsed_measurements,
                edited_metadata=edited_metadata,
                warning_table=warning_table,
                transient_previews=transient_previews,
                transient_parsed_measurements=transient_measurements,
                transient_edited_metadata=transient_edited_metadata,
                validation_previews=validation_previews,
                validation_parsed_measurements=validation_measurements,
                validation_edited_metadata=validation_edited_metadata,
                lcr_uploads=lcr_records,
            )
        elif active_section == "Export":
            _render_export_tab(
                parsed_measurements=parsed_measurements,
                analyses=analyses,
                per_cycle_summary=per_cycle_summary,
                per_test_summary=per_test_summary,
                coverage=coverage,
                schema=schema,
                main_field_axis=main_field_axis,
                current_channel=current_channel,
                default_output_dir=default_output_dir,
            )
        return

    if active_section == "Data Import":
        _render_data_import_tab(
            previews=previews,
            parsed_measurements=parsed_measurements,
            edited_metadata=edited_metadata,
            warning_table=warning_table,
            transient_previews=transient_previews,
            transient_parsed_measurements=transient_measurements,
            transient_edited_metadata=transient_edited_metadata,
            validation_previews=validation_previews,
            validation_parsed_measurements=validation_measurements,
            validation_edited_metadata=validation_edited_metadata,
            lcr_uploads=lcr_records,
        )
    elif active_section == "Validation / Retune":
        render_validation_retune_section(
            current_channel=current_channel,
            field_channel=main_field_axis,
            max_daq_voltage_pp=max_daq_voltage_pp,
            amp_gain_at_100_pct=amp_gain_at_100_pct,
            amp_gain_limit_pct=amp_gain_limit_pct,
            amp_max_output_pk_v=amp_max_output_pk_v,
            default_support_amp_gain_pct=default_support_amp_gain_pct,
            validation_measurements=validation_measurements,
            validation_preprocess_results=validation_preprocess_results,
        )
    elif active_section == "Catalogs / Diagnostics":
        render_catalogs_and_diagnostics_section()
    elif active_section == "Raw Waveforms":
        _render_raw_waveforms_tab(
            test_ids=test_ids,
            analysis_lookup=analysis_lookup,
        )
    elif active_section == "Cycle Overlay":
        _render_cycle_overlay_section(test_ids=test_ids, analysis_lookup=analysis_lookup)
    elif active_section == "Loop Analysis":
        _render_loop_analysis_section(test_ids=test_ids, analysis_lookup=analysis_lookup)
    elif active_section == "Frequency/Amplitude Comparison":
        _render_frequency_comparison_section(
            per_test_summary=per_test_summary,
            analysis_lookup=analysis_lookup,
            current_channel=current_channel,
            main_field_axis=main_field_axis,
        )
    elif active_section == "Thermal / Drift":
        _render_thermal_drift_section(
            test_ids=test_ids,
            analysis_lookup=analysis_lookup,
            main_field_axis=main_field_axis,
        )
    elif active_section == "Operating Map":
        _render_operating_map_section(
            per_test_summary=per_test_summary,
            coverage=coverage,
        )
    elif active_section == "Calculation Details":
        _render_calculation_details_section(
            test_ids=test_ids,
            analysis_lookup=analysis_lookup,
            current_channel=current_channel,
            main_field_axis=main_field_axis,
        )
    elif active_section == "Export":
        _render_export_tab(
            parsed_measurements=parsed_measurements,
            analyses=analyses,
            per_cycle_summary=per_cycle_summary,
            per_test_summary=per_test_summary,
            coverage=coverage,
            schema=schema,
            main_field_axis=main_field_axis,
            current_channel=current_channel,
            default_output_dir=default_output_dir,
        )


def run_app() -> None:
    _run_app_shell(
        page_title="전자기장 실험 데이터 분석 툴",
        title="전자기장 실험 데이터 분석 툴",
        caption="고정 형식 CSV/Excel을 읽어 정규화, 전처리, cycle 검출, measured field 중심 비교, export까지 연결하는 로컬 분석 앱",
        initial_usage_mode="간단 LUT",
        lock_usage_mode=False,
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )


def run_quick_lut_app() -> None:
    _run_app_shell(
        page_title="전자기장 LUT/보정 운용 툴",
        title="전자기장 LUT/보정 운용 툴",
        caption="measured field 중심으로 Quick LUT, waveform compensation, 제어 전달용 export만 빠르게 다루는 운용 전용 앱",
        initial_usage_mode="간단 LUT",
        lock_usage_mode=True,
        default_output_dir=DEFAULT_QUICK_OUTPUT_DIR,
    )


def _render_quick_lut_tab(
    per_test_summary: pd.DataFrame,
    analysis_lookup: dict,
    main_field_axis: str,
    current_channel: str,
) -> None:
    st.markdown("#### Estimate a recommended voltage waveform from a target field-oriented output.")
    st.caption("Quick LUT defaults to field-first targets. Current and gain remain available only as debug or equipment reference.")
    if per_test_summary.empty:
        st.warning("LUT 계산에 사용할 테스트 요약이 없습니다.")
        return

    waveform_options = sorted(
        value for value in per_test_summary["waveform_type"].dropna().unique().tolist() if value
    )
    freq_options = sorted(
        float(value) for value in per_test_summary["freq_hz"].dropna().unique().tolist()
    )
    metric_candidates = [
        f"achieved_{main_field_axis}_pp_mean",
        "achieved_bz_mT_pp_mean",
        "achieved_bmag_mT_pp_mean",
        "achieved_current_pp_a_mean",
    ]
    available_metric_options = [
        metric for metric in dict.fromkeys(metric_candidates) if metric in per_test_summary.columns
    ]

    left, mid, right = st.columns(3)
    with left:
        target_waveform = st.selectbox("파형", options=waveform_options or ["sine"], key="lut_waveform")
        target_freq = st.selectbox("주파수 (Hz)", options=freq_options or [0.5], key="lut_freq")
    with mid:
        target_metric = st.selectbox(
            "Target Metric",
            options=_prioritize_metric_options(
                available_metric_options,
                main_field_axis,
                include_current_debug=st.checkbox(
                    "Show current target metric (debug)",
                    value=False,
                    help="Field metrics stay first. Current remains available only for debug or fallback use.",
                    key="lut_metric_current_debug",
                ),
            ),
            format_func=target_metric_label,
            key="lut_metric",
        )
        target_value = float(
            st.number_input("목표 값", min_value=0.0, value=100.0, step=1.0, key="lut_target_value")
        )
    with right:
        st.write("")
        st.write("")
        estimate_clicked = st.button("추천 전압 파형 계산", use_container_width=True)

    if not estimate_clicked:
        st.info("파형, 주파수, 목표값을 고른 뒤 `추천 전압 파형 계산`을 누르십시오.")
        return

    recommendation = recommend_voltage_waveform(
        per_test_summary=per_test_summary,
        analyses_by_test_id=analysis_lookup,
        waveform_type=target_waveform,
        freq_hz=float(target_freq),
        target_metric=target_metric,
        target_value=target_value,
    )
    if recommendation is None:
        st.warning("선택한 파형/주파수 조합에 대해 전압 LUT를 만들 수 있는 데이터가 부족합니다.")
        return

    if recommendation["recommendation_mode"] == "single_point_only":
        st.warning(
            "현재 선택한 파형/주파수 조합에는 실험점이 1개뿐입니다. "
            "보간 LUT가 아니라 가장 가까운 실험점의 전압 개형만 보여줍니다."
        )
    elif recommendation["in_range"]:
        st.success("선택한 목표값이 실험 데이터 범위 안에 있어 보간으로 계산했습니다.")
    else:
        st.warning(
            "목표값이 실험 데이터 범위를 벗어나 nearest clamp로 계산했습니다. "
            f"사용값={recommendation['used_target_value']:.3f}"
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recommended Voltage PP", f"{recommendation['limited_voltage_pp']:.3f} V")
    c2.metric(
        str(recommendation["primary_output_label"]),
        _format_optional_metric(recommendation["primary_output_pp"], str(recommendation["primary_output_unit"])),
    )
    c3.metric("Support Freqs", f"{recommendation['frequency_support_count']}")
    c4.metric("Waveform Scope", str(recommendation["recommendation_scope_label"]))
    st.caption("Equipment and current estimates are shown below only as reference. The primary LUT target is the field waveform/output.")
    _render_lut_equipment_debug(recommendation)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            plot_lut_lookup_curve(
                recommendation["lookup_table"],
                target_metric=target_metric,
                voltage_metric="daq_input_v_pp_mean",
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            plot_command_waveform(recommendation["command_waveform"]),
            use_container_width=True,
        )

    st.markdown("#### 계산 근거")
    st.write(f"- template test: `{recommendation['template_test_id']}`")
    st.write(f"- target metric: `{target_metric_label(target_metric)}`")
    st.write(f"- recommendation scope: `{recommendation['recommendation_scope_label']}`")
    st.write(f"- requested target: `{recommendation['requested_target_value']:.3f}`")
    st.write(f"- used target: `{recommendation['used_target_value']:.3f}`")
    st.write(
        f"- support points: `{recommendation['support_point_count']}` "
        f"(range `{recommendation['available_target_min']:.3f}` ~ `{recommendation['available_target_max']:.3f}`)"
    )
    st.write(f"- mode: `{recommendation['recommendation_mode']}`")

    st.markdown("#### 근거 실험점")
    neighbor_points = recommendation["neighbor_points"].copy()
    neighbor_points = neighbor_points.loc[:, ~neighbor_points.columns.duplicated()]
    st.dataframe(neighbor_points, use_container_width=True)

    st.markdown("#### 전체 LUT 테이블")
    lookup_table = recommendation["lookup_table"].copy()
    lookup_table = lookup_table.loc[:, ~lookup_table.columns.duplicated()]
    st.dataframe(lookup_table, use_container_width=True)

    csv_bytes = recommendation["command_waveform"].to_csv(index=False).encode("utf-8-sig")
    file_name = (
        f"{recommendation['recommendation_scope']}_recommended_voltage_waveform_{target_waveform}_{float(target_freq):g}Hz_"
        f"{target_metric}_{target_value:g}.csv"
    )
    st.download_button(
        label="추천 전압 파형 CSV 다운로드",
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
    )


def _prioritize_metric_options(
    metric_options: list[str],
    main_field_axis: str,
    include_current_debug: bool = False,
) -> list[str]:
    return prioritize_lut_target_metrics(
        metric_options=metric_options,
        main_field_axis=main_field_axis,
        include_current_debug=include_current_debug,
    )


def _format_optional_metric(value: object, unit: str = "", digits: int = 3) -> str:
    numeric = first_number(value)
    if numeric is None or not np.isfinite(float(numeric)):
        return "n/a"
    suffix = f" {unit}" if unit else ""
    return f"{float(numeric):.{digits}f}{suffix}"


def _render_lut_equipment_debug(recommendation: dict[str, object]) -> None:
    with st.expander("Equipment / Debug", expanded=False):
        st.write(
            f"- modeling focus output: `{recommendation['primary_output_label']}` = "
            f"`{_format_optional_metric(recommendation['primary_output_pp'], str(recommendation['primary_output_unit']))}`"
        )
        st.write(
            f"- selected target output: `{recommendation['target_output_label']}` = "
            f"`{_format_optional_metric(recommendation['target_output_pp'], str(recommendation['target_output_unit']))}`"
        )
        st.write(f"- estimated current pp: `{_format_optional_metric(recommendation['estimated_current_pp'], 'A')}`")
        st.write(f"- raw recommended voltage pp: `{_format_optional_metric(recommendation['estimated_voltage_pp'], 'V')}`")
        st.write(f"- DAQ-limited voltage pp: `{_format_optional_metric(recommendation['limited_voltage_pp'], 'V')}`")
        st.write(
            f"- support amp gain: `{_format_optional_metric(recommendation['support_amp_gain_pct'], '%', digits=1)}`"
        )
        st.write(
            f"- required amp gain: `{_format_optional_metric(recommendation['required_amp_gain_pct'], '%', digits=1)}`"
        )
        st.write(
            f"- available amp gain: `{_format_optional_metric(recommendation['available_amp_gain_pct'], '%', digits=1)}`"
        )
        st.write(
            f"- amp output at required gain: `{_format_optional_metric(recommendation['amp_output_pp_at_required'], 'Vpp')}`"
        )
        st.write(f"- within hardware limits: `{recommendation['within_hardware_limits']}`")


def _render_quick_lut_tab_v2(
    per_test_summary: pd.DataFrame,
    analysis_lookup: dict,
    main_field_axis: str,
    current_channel: str,
    max_daq_voltage_pp: float,
    amp_gain_at_100_pct: float,
    amp_gain_limit_pct: float,
    amp_max_output_pk_v: float,
    default_support_amp_gain_pct: float,
    allow_target_extrapolation: bool,
    transient_measurements: list | None = None,
    transient_preprocess_results: list | None = None,
) -> None:
    st.info(
        "Usage: `scalar LUT` gives a first-pass voltage estimate for a target field PP value. "
        "`waveform compensation` recommends a drive command while keeping measured field waveform as the primary reference."
    )
    st.markdown("#### Review Quick LUT and waveform compensation with a field-first modeling focus.")
    st.caption(
        "현재 메인 보정은 steady-state support 기반 harmonic inverse입니다. finite-cycle은 별도 transient support가 있을 때만 제한적으로 참고하십시오."
    )
    if per_test_summary.empty:
        st.warning(
            "No steady-state summary is available for Quick LUT yet. "
            "Upload continuous measurement files first; validation-only or finite-cycle-only uploads do not populate this screen."
        )
        return

    waveform_options = sorted(
        value for value in per_test_summary["waveform_type"].dropna().unique().tolist() if value
    )
    freq_options = sorted(float(value) for value in per_test_summary["freq_hz"].dropna().unique().tolist())
    default_freq = float(freq_options[0]) if freq_options else 0.5
    frequency_labels = ", ".join(f"{value:g}" for value in freq_options) if freq_options else "없음"
    metric_candidates = [
        f"achieved_{main_field_axis}_pp_mean",
        "achieved_bz_mT_pp_mean",
        "achieved_bmag_mT_pp_mean",
        "achieved_current_pp_a_mean",
    ]
    available_metric_options = [
        metric for metric in dict.fromkeys(metric_candidates) if metric in per_test_summary.columns
    ]
    metric_options = _prioritize_metric_options(available_metric_options, main_field_axis)
    if not waveform_options or not freq_options or not metric_options:
        missing_support = []
        if not waveform_options:
            missing_support.append("waveform types")
        if not freq_options:
            missing_support.append("frequencies")
        if not metric_options:
            missing_support.append("target metrics")
        st.warning(
            "Quick LUT support data is incomplete: missing "
            f"{', '.join(missing_support)} in the steady-state summary."
        )
        st.caption(
            "Check `Data Import` and parsed metadata. Quick LUT only works from steady-state rows with usable waveform/frequency/output fields."
        )
        return

    left, mid, right = st.columns(3)
    with left:
        target_waveform = st.selectbox("파형", options=waveform_options or ["sine"], key="lut_waveform_v2")
        target_freq = float(
            st.number_input(
                "주파수 (Hz)",
                min_value=0.01,
                value=default_freq,
                step=0.25,
                key="lut_freq_v2",
            )
        )
        use_frequency_trend = st.checkbox(
            "근방 주파수 trend 사용",
            value=True,
            help="끄면 정확히 같은 주파수 실험점만 사용합니다. 켜면 같은 파형의 인접 주파수 경향까지 사용해 보간/클램프합니다.",
            key="lut_freq_trend_v2",
        )
        st.caption(f"보유 주파수: {frequency_labels} Hz")
    with mid:
        include_current_debug = st.checkbox(
            "Show current target metric (debug)",
            value=False,
            help="Field metrics stay first. Current remains available only for debug or fallback use.",
            key="lut_metric_current_debug_v2",
        )
        target_metric = st.selectbox(
            "LUT Target Metric",
            options=_prioritize_metric_options(
                available_metric_options,
                main_field_axis,
                include_current_debug=include_current_debug,
            ),
            format_func=target_metric_label,
            key="lut_metric_v2",
        )
        target_value = float(
            st.number_input("크기 LUT 목표값", min_value=0.0, value=100.0, step=1.0, key="lut_target_value_v2")
        )
        compensation_target_type = st.selectbox(
            "파형 보정 목표 항목",
            options=["field", "current"],
            format_func=lambda value: "전류" if value == "current" else f"자기장 ({main_field_axis})",
            key="comp_target_type_v2",
        )
        st.caption("Field waveform is the default recommendation target. Current mode is kept only for advanced/debug comparison.")
        compensation_target_label = (
            "파형 보정 목표 Current PP (A)"
            if compensation_target_type == "current"
            else f"파형 보정 목표 {main_field_axis} PP (mT)"
        )
        compensation_target_current_pp = float(
            st.number_input(
                compensation_target_label,
                min_value=0.0,
                value=20.0,
                step=1.0,
                key="comp_target_current_pp_v2",
            )
        )
        finite_cycle_mode = st.checkbox(
            "구동 cycle 수 제한 사용",
            value=False,
            help="끄면 기존 steady-state 1-cycle 보정 로직을 사용합니다. 켜면 0초 시작/종료를 포함한 finite run 보정을 계산합니다.",
            key="finite_cycle_mode_v2",
        )
        if finite_cycle_mode:
            target_cycle_count = float(
                st.number_input(
                    "구동 cycle 수",
                    min_value=0.25,
                    value=1.25,
                    step=0.25,
                    key="target_cycle_count_v2",
                )
            )
            preview_tail_cycles = float(
                st.number_input(
                    "종료 후 zero preview (cycle)",
                    min_value=0.0,
                    value=0.25,
                    step=0.25,
                    key="preview_tail_cycles_v2",
                )
            )
        else:
            target_cycle_count = None
            preview_tail_cycles = 0.25
    with right:
        st.write("")
        st.write("")
        estimate_clicked = st.button("크기 LUT 계산", use_container_width=True, key="lut_scalar_button_v2")
        compensation_button_label = (
            f"{main_field_axis} 파형 보정 계산" if compensation_target_type == "field" else "전류 파형 보정 계산"
        )
        compensation_clicked = st.button(
            compensation_button_label,
            use_container_width=True,
            key="lut_comp_button_v2",
        )

    if not estimate_clicked and not compensation_clicked:
        st.info("파형, 주파수, 목표값을 고른 뒤 계산 버튼을 누르십시오.")
        return

    if compensation_clicked:
        compensation_title = (
            f"{main_field_axis} 파형 보정" if compensation_target_type == "field" else "전류 파형 보정"
        )
        compensation_basis = (
            f"목표 measured {main_field_axis} waveform"
            if compensation_target_type == "field"
            else "목표 measured current waveform"
        )
        clamp_label = (
            f"목표 {main_field_axis} 크기"
            if compensation_target_type == "field"
            else "목표 current 크기"
        )
        st.markdown(f"#### {compensation_title}")
        st.caption(
            f"이 기능은 {compensation_basis}을 기준으로 recommended voltage waveform을 계산합니다. "
            "실제 구동용은 이쪽을 우선 보고, 위의 크기 LUT는 대략적인 전압 범위 확인용으로 쓰면 됩니다."
        )
        if finite_cycle_mode:
            st.warning(
                "finite-cycle은 아직 실험적 기능입니다. 현재 추천은 steady-state support를 기반으로 0초 시작/종료를 맞춘 보조 계산이며, "
                "전용 transient 데이터 없이 0.75 / 1.25 cycle 응답을 정확 모델처럼 해석하면 안 됩니다."
            )
        frequency_mode = "interpolate" if use_frequency_trend else "exact"
        compensation = synthesize_current_waveform_compensation(
            per_test_summary=per_test_summary,
            analyses_by_test_id=analysis_lookup,
            waveform_type=target_waveform,
            freq_hz=float(target_freq),
            target_current_pp_a=compensation_target_current_pp,
            current_channel=current_channel,
            field_channel=main_field_axis,
            target_output_type=compensation_target_type,
            target_output_pp=compensation_target_current_pp,
            frequency_mode=frequency_mode,
            finite_cycle_mode=finite_cycle_mode,
            target_cycle_count=target_cycle_count,
            preview_tail_cycles=preview_tail_cycles,
            max_daq_voltage_pp=max_daq_voltage_pp,
            amp_gain_at_100_pct=amp_gain_at_100_pct,
            amp_gain_limit_pct=amp_gain_limit_pct,
            amp_max_output_pk_v=amp_max_output_pk_v,
            default_support_amp_gain_pct=default_support_amp_gain_pct,
            allow_output_extrapolation=allow_target_extrapolation,
        )
        if compensation is None:
            frequency_scope = "this frequency or nearby frequencies" if use_frequency_trend else "the exact frequency"
            st.warning(
                f"Could not build waveform compensation for waveform `{target_waveform}` at {float(target_freq):.3f} Hz. "
                f"The current support data does not provide a usable basis at {frequency_scope}."
            )
            st.caption(
                "Check `Data Import` and confirm the steady-state support runs have usable waveform, frequency, and output metadata."
            )
        else:
            st.success("파형 보정 계산이 완료되었습니다.")
            if compensation["mode"] == "harmonic_inverse_single_support":
                st.warning(
                    "현재 조합에는 실험점이 1개뿐이라 단일 실험의 harmonic transfer로만 역보정했습니다."
                )
            elif not compensation["target_output_pp"] >= compensation["available_output_pp_min"] or not compensation["target_output_pp"] <= compensation["available_output_pp_max"]:
                if compensation["allow_output_extrapolation"]:
                    st.warning(
                        "목표 출력이 실험 support 범위를 벗어나 외삽으로 계산했습니다. "
                        f"support={compensation['available_output_pp_min']:.3f} ~ {compensation['available_output_pp_max']:.3f} {compensation['target_output_unit']}"
                    )
                else:
                    st.warning(
                        f"{clamp_label}가 지원 범위를 벗어나 "
                        f"harmonic transfer가 clamp되었습니다. ratio={compensation['phase_clamp_fraction']:.1%}"
                    )
            elif compensation["phase_clamp_fraction"] > 0:
                st.warning(
                    f"{clamp_label} 일부가 지원 범위를 벗어나 "
                    f"harmonic transfer가 clamp되었습니다. ratio={compensation['phase_clamp_fraction']:.1%}"
                )
            else:
                st.success("harmonic inverse compensation으로 추천 전압 파형을 계산했습니다.")
            if compensation["frequency_mode"] == "frequency_interpolated":
                st.info(
                    f"주파수 trend 보간 사용: 요청 {compensation['requested_freq_hz']:.3f} Hz, "
                    f"지원 범위 {compensation['available_freq_min']:.3f} ~ {compensation['available_freq_max']:.3f} Hz"
                )
            elif compensation["frequency_mode"] == "frequency_clamped":
                st.warning(
                    f"요청 주파수가 지원 범위를 벗어나 {compensation['used_freq_hz']:.3f} Hz 경계값으로 transfer를 추정했습니다."
                )

            command_profile = compensation["command_profile"]
            recommended_voltage_pp = float(command_profile["recommended_voltage_pp"].iloc[0])
            limited_voltage_pp = float(command_profile["limited_voltage_pp"].iloc[0])
            required_gain_multiplier = float(command_profile["required_amp_gain_multiplier"].iloc[0])
            required_gain_pct = float(command_profile["required_amp_gain_pct"].iloc[0])
            available_gain_pct = float(command_profile["available_amp_gain_pct"].iloc[0])
            amp_output_pp = float(command_profile["amp_output_pp_at_required"].iloc[0])
            output_unit = compensation["target_output_unit"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("DAQ Voltage PP", f"{limited_voltage_pp:.3f} V")
            c2.metric(
                f"목표 {compensation['target_output_label']}",
                f"{compensation['target_output_pp']:.3f} {output_unit}",
            )
            c3.metric("지원 실험점 수", f"{compensation['support_point_count']}")
            c4.metric("필요 AMP Gain", f"{required_gain_pct:.1f} %")
            c5.metric("추정 출력 lag", f"{compensation['estimated_output_lag_seconds']:.4f} s")
            st.caption(
                f"raw recommended voltage pp={recommended_voltage_pp:.3f} V, "
                f"DAQ limit={compensation['max_daq_voltage_pp']:.1f} Vpp, "
                f"AMP output={amp_output_pp:.1f} Vpp"
            )
            if compensation["within_hardware_limits"]:
                st.success(
                    f"하드웨어 가능: 필요 AMP gain {required_gain_pct:.1f}% / 사용 가능 {available_gain_pct:.1f}%"
                )
            else:
                st.error(
                    f"하드웨어 제한 초과: 필요 AMP gain {required_gain_pct:.1f}% / 사용 가능 {available_gain_pct:.1f}%"
                )

            comp_left, comp_right = st.columns(2)
            with comp_left:
                st.plotly_chart(
                    plot_output_compensation_waveforms(
                        command_profile=command_profile,
                        nearest_profile=compensation.get("nearest_profile_preview", compensation["nearest_profile"]),
                        nearest_column=(
                            "measured_current_a"
                            if compensation_target_type == "current"
                            else "measured_field_mT"
                        ),
                        title=(
                            "Current Waveform Compensation"
                            if compensation_target_type == "current"
                            else f"Field Waveform Compensation: {main_field_axis}"
                        ),
                        yaxis_title=("Current (A)" if compensation_target_type == "current" else f"{main_field_axis} (mT)"),
                    ),
                    use_container_width=True,
                )
            with comp_right:
                st.plotly_chart(
                    plot_command_waveform(command_profile, value_column="limited_voltage_v"),
                    use_container_width=True,
                )
            if finite_cycle_mode:
                st.caption(
                    "그래프의 `Target Output`은 전압 0초 시작 기준으로 정렬된 목표이고, "
                    "`Lag-Compensated Target`은 내부 보정 계산에 사용된 선행 목표입니다."
                )
            else:
                st.caption("현재는 steady-state 모드라 기존 1-cycle 보정 로직을 그대로 사용합니다.")

            st.write(f"- mode: `{compensation['mode']}`")
            st.write(f"- current axis: `{current_channel}`")
            st.write(f"- target output type: `{compensation['target_output_type']}`")
            st.write(f"- finite cycle mode: `{compensation['finite_cycle_mode']}`")
            if compensation["finite_cycle_mode"]:
                st.write(f"- active cycle count: `{compensation['target_cycle_count']:.2f}`")
                st.write(f"- preview tail cycles: `{compensation['preview_tail_cycles']:.2f}`")
            st.write(
                f"- estimated output lag: `{compensation['estimated_output_lag_seconds']:.6f}` s "
                f"(`{compensation['estimated_output_lag_cycles']:.4f}` cycle)"
            )
            st.write(
                f"- requested/used freq: `{compensation['requested_freq_hz']:.3f}` / "
                f"`{compensation['used_freq_hz']:.3f}` Hz"
            )
            st.write(
                f"- available freq range: `{compensation['available_freq_min']:.3f}` ~ "
                f"`{compensation['available_freq_max']:.3f}` Hz "
                f"({compensation['frequency_support_count']} freq)"
            )
            st.write(f"- nearest support test: `{compensation['nearest_test_id']}`")
            st.write(
                f"- available output pp: `{compensation['available_output_pp_min']:.3f}` ~ "
                f"`{compensation['available_output_pp_max']:.3f}` {output_unit}"
            )
            st.write(f"- raw recommended voltage pp: `{recommended_voltage_pp:.3f}` V")
            st.write(f"- DAQ-limited voltage pp: `{limited_voltage_pp:.3f}` V")
            st.write(f"- required dc amp gain multiplier: `{required_gain_multiplier:.3f}x`")
            st.write(f"- support amp gain: `{compensation['support_amp_gain_pct']:.1f}` %")
            st.write(f"- required amp gain: `{required_gain_pct:.1f}` %")
            st.write(f"- available amp gain: `{available_gain_pct:.1f}` %")
            st.write(f"- amp output at required gain: `{compensation['amp_output_pp_at_required']:.3f}` Vpp")
            st.write(f"- within hardware limits: `{compensation['within_hardware_limits']}`")
            if pd.notna(compensation["scale_ratio_from_nearest"]):
                st.write(f"- nearest profile scale ratio: `{compensation['scale_ratio_from_nearest']:.3f}`")

            st.markdown("#### 보정 LUT 실험점")
            st.dataframe(compensation["support_table"], use_container_width=True)

            control_formula = build_control_formula(command_profile, value_column="limited_voltage_v")
            if control_formula is not None:
                st.markdown("#### 제어 전달용 수식")
                mode_text = "finite run piecewise formula" if control_formula["finite_cycle_mode"] else "steady-state periodic formula"
                st.caption(f"제어파트 전달용 표현: {mode_text}")
                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("Fourier RMSE", f"{control_formula['rmse']:.4f} V")
                fc2.metric("Fourier NRMSE", f"{control_formula['nrmse']:.2%}")
                fc3.metric("최대 절대오차", f"{control_formula['max_abs_error']:.4f} V")
                st.plotly_chart(
                    plot_formula_comparison(control_formula["reconstruction_frame"]),
                    use_container_width=True,
                )
                st.code(control_formula["formula_text"], language="text")
                st.markdown("#### 제어용 Python 식")
                st.code(control_formula["python_snippet"], language="python")
                st.markdown("#### 조화파 계수표")
                st.dataframe(control_formula["coefficient_table"], use_container_width=True)

                formula_text_bytes = control_formula["formula_text"].encode("utf-8")
                coeff_csv_bytes = control_formula["coefficient_table"].to_csv(index=False).encode("utf-8-sig")
                formula_file_prefix = (
                    f"control_formula_{target_waveform}_{float(target_freq):g}Hz_"
                    f"{compensation['target_output_type']}_{compensation_target_current_pp:g}"
                )
                st.download_button(
                    label="제어 수식 TXT 다운로드",
                    data=formula_text_bytes,
                    file_name=f"{formula_file_prefix}.txt",
                    mime="text/plain",
                    key="download_comp_formula_txt_v2",
                )
                st.download_button(
                    label="조화파 계수 CSV 다운로드",
                    data=coeff_csv_bytes,
                    file_name=f"{formula_file_prefix}_coefficients.csv",
                    mime="text/csv",
                    key="download_comp_formula_coeffs_v2",
                )

            comp_csv = command_profile.to_csv(index=False).encode("utf-8-sig")
            comp_file_name = (
                f"compensated_voltage_waveform_{target_waveform}_{float(target_freq):g}Hz_"
                f"{compensation['target_output_type']}_{compensation_target_current_pp:g}_"
                f"{(target_cycle_count if target_cycle_count is not None else 'steady') }cycle.csv"
            )
            st.download_button(
                label="보정 전압 파형 CSV 다운로드",
                data=comp_csv,
                file_name=comp_file_name,
                mime="text/csv",
                key="download_compensation_waveform_v2",
            )

            if finite_cycle_mode:
                finite_support_entries = _build_finite_support_entries(
                    transient_measurements=transient_measurements or [],
                    transient_preprocess_results=transient_preprocess_results or [],
                    current_channel=current_channel,
                    main_field_axis=main_field_axis,
                )
                finite_model = _build_empirical_finite_model(
                    finite_support_entries=finite_support_entries,
                    waveform_type=target_waveform,
                    freq_hz=float(target_freq),
                    target_cycle_count=float(target_cycle_count) if target_cycle_count is not None else None,
                    target_output_type=compensation_target_type,
                    target_output_pp=float(compensation["target_output_pp"]),
                    current_channel=current_channel,
                    main_field_axis=main_field_axis,
                    max_daq_voltage_pp=max_daq_voltage_pp,
                    amp_gain_at_100_pct=amp_gain_at_100_pct,
                    amp_gain_limit_pct=amp_gain_limit_pct,
                    amp_max_output_pk_v=amp_max_output_pk_v,
                    default_support_amp_gain_pct=default_support_amp_gain_pct,
                )
                with st.expander("finite-cycle empirical support (실험적)", expanded=False):
                    st.caption(
                        "이 영역은 finite transient 데이터를 nearest support로 잡아 선형 스케일한 empirical 참고 모델입니다. "
                        "정식 transient 물리 모델이 아니며, steady-state harmonic 보정을 대체하지 않습니다."
                    )
                    if finite_model is None:
                        st.info(
                            "현재 finite-cycle support 데이터로는 같은 waveform / freq / cycle_count 근방 empirical 참고 모델을 만들 수 없습니다."
                        )
                    else:
                        st.write(f"- support test: `{finite_model['support_test_id']}`")
                        st.write(
                            f"- support freq/cycle: `{finite_model['support_freq_hz']:.3f}` Hz / "
                            f"`{finite_model['support_cycle_count']:.2f}` cycle"
                        )
                        st.write(
                            f"- support output pp: `{finite_model['support_output_pp']:.3f}` {finite_model['target_output_unit']}"
                        )
                        st.write(f"- scale ratio: `{finite_model['scale_ratio']:.3f}`")
                        st.write(f"- distance score: `{finite_model['distance_score']:.3f}`")
                        finite_left, finite_right = st.columns(2)
                        with finite_left:
                            st.plotly_chart(
                                plot_waveforms(
                                    finite_model["modeled_frame"],
                                    ["target_output", "modeled_output"],
                                    title="Finite empirical support comparison",
                                ),
                                use_container_width=True,
                            )
                        with finite_right:
                            st.plotly_chart(
                                plot_command_waveform(finite_model["modeled_frame"], value_column="limited_voltage_v"),
                                use_container_width=True,
                            )
                        empirical_csv = finite_model["modeled_frame"].to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="finite empirical support CSV 다운로드",
                            data=empirical_csv,
                            file_name=(
                                f"finite_empirical_model_{target_waveform}_{float(target_freq):g}Hz_"
                                f"{compensation_target_type}_{compensation_target_current_pp:g}_"
                                f"{target_cycle_count:g}cycle.csv"
                            ),
                            mime="text/csv",
                            key="download_finite_empirical_model_v2",
                        )

    if estimate_clicked:
        st.markdown("#### 크기 LUT")
        st.caption(
            "This estimate focuses on field/output magnitude first. "
            "Current remains in debug only, while the main modeling target is the field waveform/output."
        )
        frequency_mode = "interpolate" if use_frequency_trend else "exact"
        recommendation = recommend_voltage_waveform(
            per_test_summary=per_test_summary,
            analyses_by_test_id=analysis_lookup,
            waveform_type=target_waveform,
            freq_hz=float(target_freq),
            target_metric=target_metric,
            target_value=target_value,
            frequency_mode=frequency_mode,
            finite_cycle_mode=finite_cycle_mode,
            target_cycle_count=target_cycle_count,
            preview_tail_cycles=preview_tail_cycles,
            max_daq_voltage_pp=max_daq_voltage_pp,
            amp_gain_at_100_pct=amp_gain_at_100_pct,
            amp_gain_limit_pct=amp_gain_limit_pct,
            amp_max_output_pk_v=amp_max_output_pk_v,
            default_support_amp_gain_pct=default_support_amp_gain_pct,
            allow_target_extrapolation=allow_target_extrapolation,
        )
        if recommendation is None:
            frequency_scope = "this frequency or nearby frequencies" if use_frequency_trend else "the exact frequency"
            st.warning(
                f"Could not build a Quick LUT recommendation for waveform `{target_waveform}` at {float(target_freq):.3f} Hz. "
                f"The current steady-state support table has no usable rows for this combination at {frequency_scope}."
            )
            st.caption(
                "Check `Data Import` and confirm waveform/frequency metadata were inferred correctly from the uploaded continuous runs."
            )
            return
        st.success("크기 LUT 계산이 완료되었습니다.")

        if recommendation["recommendation_mode"] == "single_point_only":
            st.warning(
                "현재 선택한 파형/주파수 조합에는 실험점이 1개뿐입니다. "
                "보간 LUT가 아니라 가장 가까운 실험점의 전압 개형만 보여줍니다."
            )
        elif recommendation["in_range"]:
            st.success("선택한 목표값이 실험 데이터 범위 안에 있어 보간으로 계산했습니다.")
        else:
            if recommendation["allow_target_extrapolation"]:
                st.warning(
                    "목표값이 실험 데이터 범위를 벗어나 외삽으로 계산했습니다. "
                    f"support={recommendation['available_target_min']:.3f} ~ {recommendation['available_target_max']:.3f}"
                )
            else:
                st.warning(
                    "목표값이 실험 데이터 범위를 벗어나 nearest clamp로 계산했습니다. "
                    f"사용값={recommendation['used_target_value']:.3f}"
                )
        if recommendation["frequency_mode"] == "frequency_interpolated":
            st.info(
                f"주파수 trend 보간 사용: 요청 {recommendation['requested_freq_hz']:.3f} Hz, "
                f"지원 범위 {recommendation['available_freq_min']:.3f} ~ {recommendation['available_freq_max']:.3f} Hz"
            )
        elif recommendation["frequency_mode"] == "frequency_clamped":
            st.warning(
                f"요청 주파수가 지원 범위를 벗어나 {recommendation['used_freq_hz']:.3f} Hz 경계값으로 LUT를 계산했습니다."
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recommended Voltage PP", f"{recommendation['limited_voltage_pp']:.3f} V")
        c2.metric(
            str(recommendation["primary_output_label"]),
            _format_optional_metric(recommendation["primary_output_pp"], str(recommendation["primary_output_unit"])),
        )
        c3.metric("Support Freqs", f"{recommendation['frequency_support_count']}")
        c4.metric("Waveform Scope", str(recommendation["recommendation_scope_label"]))
        st.caption(
            f"raw recommended voltage pp={recommendation['estimated_voltage_pp']:.3f} V, "
            f"DAQ limit={recommendation['max_daq_voltage_pp']:.1f} Vpp, "
            f"AMP output={recommendation['amp_output_pp_at_required']:.1f} Vpp"
        )
        if recommendation["within_hardware_limits"]:
            st.info("Equipment note: the recommended voltage stays within current DAQ / AMP limits.")
        else:
            st.warning("Equipment note: the recommended voltage exceeds current DAQ / AMP limits.")
        _render_lut_equipment_debug(recommendation)

        lut_left, lut_right = st.columns(2)
        with lut_left:
            if (
                recommendation["frequency_mode"] != "exact"
                and len(recommendation["frequency_support_table"]) > 1
            ):
                st.plotly_chart(
                    plot_frequency_support_curve(
                        recommendation["frequency_support_table"],
                        requested_freq_hz=recommendation["requested_freq_hz"],
                        used_freq_hz=recommendation["used_freq_hz"],
                    ),
                    use_container_width=True,
                )
            else:
                st.plotly_chart(
                    plot_lut_lookup_curve(
                        recommendation["lookup_table"],
                        target_metric=target_metric,
                        voltage_metric="daq_input_v_pp_mean",
                    ),
                    use_container_width=True,
                )
        with lut_right:
            st.plotly_chart(
                plot_command_waveform(recommendation["command_waveform"], value_column="limited_voltage_v"),
                use_container_width=True,
            )

        st.write(f"- template test: `{recommendation['template_test_id']}`")
        st.write(f"- target metric: `{target_metric_label(target_metric)}`")
        st.write(f"- recommendation scope: `{recommendation['recommendation_scope_label']}`")
        st.write(f"- finite cycle mode: `{recommendation['finite_cycle_mode']}`")
        if recommendation["finite_cycle_mode"]:
            st.write(f"- active cycle count: `{recommendation['target_cycle_count']:.2f}`")
            st.write(f"- preview tail cycles: `{recommendation['preview_tail_cycles']:.2f}`")
        st.write(
            f"- requested/used freq: `{recommendation['requested_freq_hz']:.3f}` / "
            f"`{recommendation['used_freq_hz']:.3f}` Hz"
        )
        st.write(
            f"- available freq range: `{recommendation['available_freq_min']:.3f}` ~ "
            f"`{recommendation['available_freq_max']:.3f}` Hz "
            f"({recommendation['frequency_support_count']} freq)"
        )
        st.write(f"- requested target: `{recommendation['requested_target_value']:.3f}`")
        st.write(f"- used target: `{recommendation['used_target_value']:.3f}`")
        st.write(
            f"- support points: `{recommendation['support_point_count']}` "
            f"(range `{recommendation['available_target_min']:.3f}` ~ `{recommendation['available_target_max']:.3f}`)"
        )
        st.write(f"- mode: `{recommendation['recommendation_mode']}`")
        st.write(f"- raw recommended voltage pp: `{recommendation['estimated_voltage_pp']:.3f}` V")
        st.write(f"- DAQ-limited voltage pp: `{recommendation['limited_voltage_pp']:.3f}` V")
        st.write("- equipment note: gain and hardware numbers are reference-only, not the primary modeling target")

        st.markdown("#### 근거 실험점")
        neighbor_points = recommendation["neighbor_points"].copy()
        neighbor_points = neighbor_points.loc[:, ~neighbor_points.columns.duplicated()]
        st.dataframe(neighbor_points, use_container_width=True)

        st.markdown("#### 전체 LUT 테이블")
        lookup_table = recommendation["lookup_table"].copy()
        lookup_table = lookup_table.loc[:, ~lookup_table.columns.duplicated()]
        st.dataframe(lookup_table, use_container_width=True)

        if len(recommendation["frequency_support_table"]) > 1:
            st.markdown("#### 주파수 trend 지원점")
            st.dataframe(recommendation["frequency_support_table"], use_container_width=True)

        control_formula = build_control_formula(recommendation["command_waveform"], value_column="limited_voltage_v")
        if control_formula is not None:
            st.markdown("#### 제어 전달용 수식")
            mode_text = "finite run piecewise formula" if control_formula["finite_cycle_mode"] else "steady-state periodic formula"
            st.caption(f"제어파트 전달용 표현: {mode_text}")
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Fourier RMSE", f"{control_formula['rmse']:.4f} V")
            fc2.metric("Fourier NRMSE", f"{control_formula['nrmse']:.2%}")
            fc3.metric("최대 절대오차", f"{control_formula['max_abs_error']:.4f} V")
            st.plotly_chart(
                plot_formula_comparison(control_formula["reconstruction_frame"]),
                use_container_width=True,
            )
            st.code(control_formula["formula_text"], language="text")
            st.markdown("#### 조화파 계수표")
            st.dataframe(control_formula["coefficient_table"], use_container_width=True)

            formula_text_bytes = control_formula["formula_text"].encode("utf-8")
            coeff_csv_bytes = control_formula["coefficient_table"].to_csv(index=False).encode("utf-8-sig")
            formula_file_prefix = (
                f"{recommendation['recommendation_scope']}_control_formula_{target_waveform}_{float(target_freq):g}Hz_"
                f"{target_metric}_{target_value:g}"
            )
            st.download_button(
                label="제어 수식 TXT 다운로드",
                data=formula_text_bytes,
                file_name=f"{formula_file_prefix}.txt",
                mime="text/plain",
                key="download_scalar_formula_txt_v2",
            )
            st.download_button(
                label="조화파 계수 CSV 다운로드",
                data=coeff_csv_bytes,
                file_name=f"{formula_file_prefix}_coefficients.csv",
                mime="text/csv",
                key="download_scalar_formula_coeffs_v2",
            )

        csv_bytes = recommendation["command_waveform"].to_csv(index=False).encode("utf-8-sig")
        file_name = (
            f"{recommendation['recommendation_scope']}_recommended_voltage_waveform_{target_waveform}_{float(target_freq):g}Hz_"
            f"{target_metric}_{target_value:g}_{(target_cycle_count if target_cycle_count is not None else 'steady')}cycle.csv"
        )
        st.download_button(
            label="추천 전압 파형 CSV 다운로드",
            data=csv_bytes,
            file_name=file_name,
            mime="text/csv",
            key="download_scalar_waveform_v2",
        )


def _render_raw_waveforms_tab(
    test_ids: list[str],
    analysis_lookup: dict,
) -> None:
    selected_test_id = st.selectbox("테스트 선택", options=test_ids, key="raw_test_simple")
    selected_analysis = analysis_lookup[selected_test_id]
    dataset_mode = st.radio("데이터셋", options=["corrected", "raw"], horizontal=True, key="raw_dataset_simple")
    display_frame = (
        selected_analysis.preprocess.corrected_frame
        if dataset_mode == "corrected"
        else selected_analysis.parsed.normalized_frame
    )
    default_channels = [
        "daq_input_v",
        "coil1_current_a",
        "coil2_current_a",
        "temperature_c",
        "bx_mT",
        "by_mT",
        "bz_mT",
        "bmag_mT",
    ]
    selected_channels = st.multiselect(
        "표시 채널",
        options=[
            column
            for column in display_frame.columns
            if column not in {"source_file", "sheet_name", "test_id", "notes", "parse_warnings"}
        ],
        default=[channel for channel in default_channels if channel in display_frame.columns],
        key="raw_channels_simple",
    )
    st.plotly_chart(
        plot_waveforms(display_frame, selected_channels, title=f"{selected_test_id} / {dataset_mode}"),
        use_container_width=True,
    )
    st.dataframe(display_frame.head(200), use_container_width=True)


def _render_cycle_overlay_section(
    test_ids: list[str],
    analysis_lookup: dict,
) -> None:
    selected_test_id = st.selectbox("테스트 선택", options=test_ids, key="cycle_test")
    selected_analysis = analysis_lookup[selected_test_id]
    overlay_channel = st.selectbox(
        "Overlay 채널",
        options=[
            "i_sum_signed",
            "i_custom_signed",
            "coil2_current_signed_a",
            "coil1_current_signed_a",
            "i_sum",
            "i_custom",
            "coil1_current_a",
            "coil2_current_a",
            "bz_mT",
            "bmag_mT",
            "bproj_mT",
        ],
        index=0,
    )
    x_mode = st.radio("Overlay X축", options=["cycle_progress", "cycle_time_s"], horizontal=True, key="cycle_x_mode")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            plot_cycle_detection_overlay(
                selected_analysis.cycle_detection.annotated_frame,
                selected_analysis.cycle_detection.reference_channel,
                selected_analysis.cycle_detection.boundaries,
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            plot_cycle_overlay(
                selected_analysis.cycle_detection.annotated_frame,
                channel=overlay_channel,
                x_mode=x_mode,
            ),
            use_container_width=True,
        )
    boundary_frame = pd.DataFrame(
        [
            {
                "cycle_index": boundary.cycle_index,
                "start_s": boundary.start_s,
                "end_s": boundary.end_s,
                "duration_s": boundary.end_s - boundary.start_s,
                "start_index": boundary.start_index,
                "end_index": boundary.end_index,
            }
            for boundary in selected_analysis.cycle_detection.boundaries
        ]
    )
    st.dataframe(boundary_frame, use_container_width=True)


def _render_loop_analysis_section(
    test_ids: list[str],
    analysis_lookup: dict,
) -> None:
    selected_test_id = st.selectbox("테스트 선택", options=test_ids, key="loop_test")
    selected_analysis = analysis_lookup[selected_test_id]
    loop_current_axis = st.selectbox(
        "Loop X축",
        options=[
            "i_sum_signed",
            "i_custom_signed",
            "coil2_current_signed_a",
            "coil1_current_signed_a",
            "i_diff_signed",
            "i_sum",
            "i_custom",
            "coil1_current_a",
            "coil2_current_a",
            "i_diff",
        ],
        index=0,
    )
    loop_field_axis = st.selectbox(
        "Loop Y축",
        options=["bz_mT", "bmag_mT", "bx_mT", "by_mT", "bproj_mT"],
        index=0,
    )
    loop_color = st.radio("색 구분", options=["branch_direction", "cycle_index"], horizontal=True, key="loop_color")
    st.plotly_chart(
        plot_loop(
            selected_analysis.cycle_detection.annotated_frame,
            current_channel=loop_current_axis,
            field_channel=loop_field_axis,
            color_by=loop_color,
        ),
        use_container_width=True,
    )
    loop_metrics_columns = [
        column
        for column in (
            "cycle_index",
            "loop_area_main",
            "coercive_like_current_a",
            "zero_crossing_offset_mT",
            "branch_asymmetry_ratio",
            "rising_branch_pp_mT",
            "falling_branch_pp_mT",
        )
        if column in selected_analysis.per_cycle_summary.columns
    ]
    st.dataframe(selected_analysis.per_cycle_summary[loop_metrics_columns], use_container_width=True)


def _render_frequency_comparison_section(
    per_test_summary: pd.DataFrame,
    analysis_lookup: dict,
    current_channel: str,
    main_field_axis: str,
) -> None:
    comparison_metric = st.selectbox(
        "비교 지표",
        options=[
            "achieved_current_pp_a_mean",
            "achieved_bz_mT_pp_mean",
            "achieved_bmag_mT_pp_mean",
            "current_retention",
            "reference_field_ratio",
            "reference_current_ratio",
            "temperature_rise_total_c",
            "thermal_drift_ratio",
        ],
        index=1,
    )
    waveform_filter = st.selectbox("파형 필터", options=["all", "sine", "triangle"], index=0)
    filtered_test_summary = per_test_summary.copy()
    if waveform_filter != "all":
        filtered_test_summary = filtered_test_summary[filtered_test_summary["waveform_type"] == waveform_filter]
    st.plotly_chart(
        plot_frequency_comparison(filtered_test_summary, metric=comparison_metric),
        use_container_width=True,
    )
    st.plotly_chart(
        plot_metric_heatmap(filtered_test_summary, metric=comparison_metric, waveform_type=waveform_filter),
        use_container_width=True,
    )
    st.dataframe(filtered_test_summary.head(300), use_container_width=True)
    _render_shape_phase_compare_section(
        per_test_summary=per_test_summary,
        analysis_lookup=analysis_lookup,
        current_channel=current_channel,
        main_field_axis=main_field_axis,
    )


def _render_thermal_drift_section(
    test_ids: list[str],
    analysis_lookup: dict,
    main_field_axis: str,
) -> None:
    selected_test_id = st.selectbox("테스트 선택", options=test_ids, key="thermal_test")
    selected_analysis = analysis_lookup[selected_test_id]
    drift_metric = st.selectbox(
        "Drift 지표",
        options=[
            "current_pp_drift_ratio",
            f"{main_field_axis}_pp_drift_ratio",
            "gain_drift_ratio",
            "temperature_drift_ratio",
        ],
        index=1,
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            plot_waveforms(selected_analysis.preprocess.corrected_frame, ["temperature_c"], "시간 대비 온도"),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_drift(selected_analysis.per_cycle_summary, metric=drift_metric),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            plot_temperature_vs_drift(selected_analysis.per_cycle_summary, drift_metric=drift_metric),
            use_container_width=True,
        )
        st.dataframe(selected_analysis.per_cycle_summary.head(120), use_container_width=True)


def _render_operating_map_section(
    per_test_summary: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    operating_field_axis = st.selectbox(
        "Operating map field axis",
        options=["bz_mT", "bmag_mT", "bx_mT", "by_mT", "bproj_mT"],
        index=0,
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            plot_operating_map(per_test_summary, field_axis=operating_field_axis),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(plot_coverage_matrix(coverage), use_container_width=True)

    st.subheader("타겟 자기장 역추정")
    wave_options = sorted(value for value in per_test_summary["waveform_type"].dropna().unique().tolist() if value)
    freq_options = sorted(float(value) for value in per_test_summary["freq_hz"].dropna().unique().tolist())
    target_waveform = st.selectbox("파형", options=wave_options or ["sine"])
    target_freq = st.selectbox("주파수", options=freq_options or [0.5])
    target_field_pp = float(st.number_input("목표 field pp (mT)", min_value=0.0, value=100.0, step=5.0))
    estimated_drive = estimate_drive_for_target_field(
        per_test_summary=per_test_summary,
        waveform_type=target_waveform,
        freq_hz=float(target_freq),
        target_field_pp=target_field_pp,
        field_axis=operating_field_axis,
    )
    if estimated_drive:
        st.success(
            f"예상 필요 current_pp={estimated_drive['estimated_current_pp_a']:.3f} A, "
            f"input_voltage_pp={estimated_drive['estimated_input_voltage_pp_v']:.3f} V"
        )
    else:
        st.warning("현재 데이터 범위 안에서는 해당 목표 자기장을 안정적으로 보간할 수 없습니다.")


def _render_calculation_details_section(
    test_ids: list[str],
    analysis_lookup: dict,
    current_channel: str,
    main_field_axis: str,
) -> None:
    selected_test_id = st.selectbox("테스트 선택", options=test_ids, key="calc_test")
    selected_analysis = analysis_lookup[selected_test_id]
    cycle_options = sorted(int(value) for value in selected_analysis.per_cycle_summary["cycle_index"].dropna().tolist())
    selected_cycle = st.selectbox("Cycle 선택", options=cycle_options)
    detail_table, intermediate_table = build_calculation_details(
        annotated_frame=selected_analysis.cycle_detection.annotated_frame,
        per_cycle_summary=selected_analysis.per_cycle_summary,
        cycle_index=int(selected_cycle),
        current_channel=current_channel,
        main_field_axis=main_field_axis,
    )
    st.markdown("#### 계산식 / 중간값 / 결과")
    st.dataframe(detail_table, use_container_width=True)
    st.markdown("#### Intermediate Sample Table")
    st.dataframe(intermediate_table, use_container_width=True)

    offsets_df = pd.DataFrame(
        [{"channel": channel, "offset": offset} for channel, offset in selected_analysis.preprocess.offsets.items()]
    )
    lags_df = pd.DataFrame(
        [
            {
                "channel": lag.channel,
                "lag_seconds": lag.lag_seconds,
                "lag_samples": lag.lag_samples,
                "correlation": lag.correlation,
            }
            for lag in selected_analysis.preprocess.lags
        ]
    )
    st.markdown("#### 사용된 설정")
    st.write(f"- selected current: `{current_channel}`")
    st.write(f"- main field axis: `{main_field_axis}`")
    st.write(f"- cycle reference: `{selected_analysis.cycle_detection.reference_channel}`")
    st.write(f"- preprocessing logs: {' | '.join(selected_analysis.preprocess.logs)}")
    if not offsets_df.empty:
        st.dataframe(offsets_df, use_container_width=True)
    if not lags_df.empty:
        st.dataframe(lags_df, use_container_width=True)


def _render_shape_phase_compare_section(
    per_test_summary: pd.DataFrame,
    analysis_lookup: dict,
    current_channel: str,
    main_field_axis: str,
) -> None:
    st.markdown("#### 동일 주파수 개형 / 위상 비교")
    if per_test_summary.empty:
        st.info("개형 비교를 할 테스트 요약이 아직 없습니다.")
        return

    shape_wave_options = sorted(
        value for value in per_test_summary["waveform_type"].dropna().unique().tolist() if value
    )
    shape_waveform = st.selectbox(
        "비교 파형",
        options=shape_wave_options or ["sine"],
        key="shape_compare_waveform",
    )
    shape_freq_candidates = sorted(
        float(value)
        for value in per_test_summary.loc[
            per_test_summary["waveform_type"] == shape_waveform,
            "freq_hz",
        ].dropna().unique().tolist()
    )
    compare_left, compare_mid, compare_right = st.columns(3)
    with compare_left:
        shape_freq = st.selectbox(
            "비교 주파수 (Hz)",
            options=shape_freq_candidates or [0.5],
            key="shape_compare_freq",
        )
    supported_shape_channels = list(
        dict.fromkeys(
            [
                current_channel,
                "i_sum_signed",
                "i_custom_signed",
                "coil1_current_signed_a",
                "coil2_current_signed_a",
                "i_sum",
                "coil1_current_a",
                "coil2_current_a",
                "daq_input_v",
                main_field_axis,
                "bz_mT",
                "bmag_mT",
                "bx_mT",
                "by_mT",
                "bproj_mT",
            ]
        )
    )
    with compare_mid:
        shape_signal_channel = st.selectbox(
            "비교 채널",
            options=supported_shape_channels,
            key="shape_compare_channel",
        )
    with compare_right:
        normalization_mode = st.selectbox(
            "정규화 방식",
            options=["peak_to_peak", "peak", "rms"],
            format_func=lambda value: {
                "peak_to_peak": "PP 기준",
                "peak": "Peak 기준",
                "rms": "RMS 기준",
            }[value],
            key="shape_compare_norm",
        )

    shape_reference_rows = per_test_summary[
        (per_test_summary["waveform_type"] == shape_waveform)
        & np.isclose(per_test_summary["freq_hz"], float(shape_freq), equal_nan=False)
    ].sort_values(["current_pp_target_a", "achieved_current_pp_a_mean", "test_id"])
    reference_options = shape_reference_rows["test_id"].tolist()
    reference_test_id = (
        st.selectbox(
            "기준 테스트",
            options=reference_options,
            index=0,
            key="shape_compare_reference",
        )
        if reference_options
        else None
    )

    overlay_frame, shape_summary = build_shape_phase_comparison(
        analyses=analysis_lookup.values(),
        waveform_type=shape_waveform,
        freq_hz=float(shape_freq),
        signal_channel=shape_signal_channel,
        reference_test_id=reference_test_id,
        current_channel=current_channel,
        main_field_axis=main_field_axis,
        normalization_mode=normalization_mode,
    )
    if overlay_frame.empty or shape_summary.empty:
        st.info("선택한 파형/주파수 조합에서 개형 비교를 만들 수 있는 테스트가 충분하지 않습니다.")
        return

    overlay_left, overlay_right = st.columns(2)
    with overlay_left:
        st.plotly_chart(
            plot_shape_overlay(
                overlay_frame=overlay_frame,
                y_column="normalized_signal_raw",
                title="정규화 개형 비교 (원위상)",
            ),
            use_container_width=True,
        )
    with overlay_right:
        st.plotly_chart(
            plot_shape_overlay(
                overlay_frame=overlay_frame,
                y_column="normalized_signal_aligned",
                title="정규화 개형 비교 (위상 정렬)",
            ),
            use_container_width=True,
        )

    metric_left, metric_right = st.columns(2)
    with metric_left:
        st.plotly_chart(
            plot_shape_metric_trend(
                summary_frame=shape_summary,
                metric="phase_lag_deg",
                title="전류 레벨별 위상 차",
            ),
            use_container_width=True,
        )
    with metric_right:
        st.plotly_chart(
            plot_shape_metric_trend(
                summary_frame=shape_summary,
                metric="shape_nrmse_aligned",
                title="전류 레벨별 개형 차이",
            ),
            use_container_width=True,
        )

    summary_columns = [
        column
        for column in (
            "test_id",
            "current_pp_target_a",
            "achieved_current_pp_a_mean",
            "phase_lag_deg",
            "phase_lag_seconds",
            "shape_corr_raw",
            "shape_corr_aligned",
            "shape_nrmse_raw",
            "shape_nrmse_aligned",
            "signal_pp",
            "is_reference",
        )
        if column in shape_summary.columns
    ]
    st.dataframe(shape_summary[summary_columns], use_container_width=True)
    st.download_button(
        label="개형 비교 요약 CSV 다운로드",
        data=shape_summary.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"shape_phase_compare_{shape_waveform}_{shape_freq:g}Hz.csv",
        mime="text/csv",
        key="download_shape_compare_summary",
    )
    st.download_button(
        label="정규화 개형 CSV 다운로드",
        data=overlay_frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"shape_phase_overlay_{shape_waveform}_{shape_freq:g}Hz.csv",
        mime="text/csv",
        key="download_shape_compare_overlay",
    )


def _render_export_tab(
    parsed_measurements: list,
    analyses: list,
    per_cycle_summary: pd.DataFrame,
    per_test_summary: pd.DataFrame,
    coverage: pd.DataFrame,
    schema,
    main_field_axis: str,
    current_channel: str,
    default_output_dir: Path,
) -> None:
    export_dir = st.text_input("Export 폴더", value=str(default_output_dir), key="export_dir_simple")
    if st.button("Export 생성", use_container_width=True, key="export_button_simple"):
        figures = {
            "coverage_matrix": plot_coverage_matrix(coverage),
            "operating_map": plot_operating_map(per_test_summary, field_axis=main_field_axis),
            "frequency_comparison": plot_frequency_comparison(
                per_test_summary,
                metric="achieved_bz_mT_pp_mean"
                if "achieved_bz_mT_pp_mean" in per_test_summary.columns
                else "achieved_current_pp_a_mean",
            ),
        }
        artifacts = export_analysis_bundle(
            output_dir=export_dir,
            parsed_measurements=parsed_measurements,
            analyses=analyses,
            per_cycle_summary=per_cycle_summary,
            per_test_summary=per_test_summary,
            coverage=coverage,
            config_snapshot_yaml=dump_schema_yaml(schema),
            current_channel=current_channel,
            field_channel=main_field_axis,
            figures=figures,
        )
        zip_bytes = build_export_zip_bytes(artifacts.root_dir)
        st.success(f"Export 완료: {artifacts.root_dir}")
        st.info("Excel 결과물은 값을 유지하고, 열 너비 / 고정 행·열 / 필터 / 표시 자릿수를 보기 좋게 정리해 저장합니다.")
        st.caption("원본 raw 파일은 덮어쓰지 않고, corrected/derived 결과만 export 폴더에 새로 만듭니다.")
        st.write(f"- normalized_data.xlsx: {artifacts.normalized_data_path}")
        st.write(f"- per_test_summary.xlsx: {artifacts.per_test_summary_path}")
        st.write(f"- per_cycle_summary.xlsx: {artifacts.per_cycle_summary_path}")
        st.write(f"- analysis_report.md: {artifacts.report_path}")
        if artifacts.excel_formatting_report_path:
            st.write(f"- excel_formatting_check.md: {artifacts.excel_formatting_report_path}")
            st.caption("`excel_formatting_check.md`에서 freeze panes / filter / 숫자 표시 형식 검증 결과를 바로 확인할 수 있습니다.")
        st.caption("주요 시트: per_test_summary / per_cycle_summary / waveform_fit_summary / representative_cycles")
        st.write(f"- summary_plots: {artifacts.plots_dir}")
        st.download_button(
            label="Export ZIP 다운로드",
            data=zip_bytes,
            file_name="field_analysis_export.zip",
            mime="application/zip",
            key="export_download_simple",
        )


def _build_finite_run_summary(parsed_measurement, corrected_frame: pd.DataFrame) -> dict[str, object]:
    normalized = parsed_measurement.normalized_frame
    test_id = (
        str(normalized["test_id"].iloc[0])
        if "test_id" in normalized.columns and not normalized.empty
        else f"{parsed_measurement.source_file}::{parsed_measurement.sheet_name}"
    )
    freq_hz = float(pd.to_numeric(normalized.get("freq_hz"), errors="coerce").dropna().iloc[0]) if "freq_hz" in normalized.columns and pd.to_numeric(normalized.get("freq_hz"), errors="coerce").notna().any() else np.nan
    duration_s = (
        float(pd.to_numeric(corrected_frame["time_s"], errors="coerce").max())
        if "time_s" in corrected_frame.columns and not corrected_frame.empty
        else np.nan
    )
    approx_cycle_span = duration_s * freq_hz if np.isfinite(duration_s) and np.isfinite(freq_hz) else np.nan
    return {
        "test_id": test_id,
        "source_file": parsed_measurement.source_file,
        "sheet_name": parsed_measurement.sheet_name,
        "waveform_type": parsed_measurement.metadata.get("waveform"),
        "freq_hz": freq_hz,
        "target_current_a": first_number(str(parsed_measurement.metadata.get("Target Current(A)", ""))),
        "duration_s": duration_s,
        "approx_cycle_span": approx_cycle_span,
        "notes": parsed_measurement.metadata.get("notes", ""),
    }


def _signal_peak_to_peak(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.nanmax(finite) - np.nanmin(finite))


def _build_finite_support_entries(
    transient_measurements: list,
    transient_preprocess_results: list,
    current_channel: str,
    main_field_axis: str,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for parsed, preprocess in zip(transient_measurements, transient_preprocess_results, strict=False):
        corrected = preprocess.corrected_frame.copy()
        summary = _build_finite_run_summary(parsed, corrected)
        entries.append(
            {
                **summary,
                "current_pp": _signal_peak_to_peak(corrected, current_channel),
                "field_pp": _signal_peak_to_peak(corrected, main_field_axis),
                "daq_voltage_pp": _signal_peak_to_peak(corrected, "daq_input_v"),
                "frame": corrected,
            }
        )
    return entries


def _build_empirical_finite_model(
    finite_support_entries: list[dict[str, object]],
    waveform_type: str,
    freq_hz: float,
    target_cycle_count: float | None,
    target_output_type: str,
    target_output_pp: float,
    current_channel: str,
    main_field_axis: str,
    max_daq_voltage_pp: float,
    amp_gain_at_100_pct: float,
    amp_gain_limit_pct: float,
    amp_max_output_pk_v: float,
    default_support_amp_gain_pct: float,
) -> dict[str, object] | None:
    if not finite_support_entries or target_cycle_count is None:
        return None

    output_column = "field_pp" if target_output_type == "field" else "current_pp"
    unit = "mT" if target_output_type == "field" else "A"
    waveform_matches = [
        entry for entry in finite_support_entries
        if str(entry.get("waveform_type") or "") == waveform_type
    ] or finite_support_entries

    working_entries = []
    freq_values = [float(entry["freq_hz"]) for entry in waveform_matches if np.isfinite(entry["freq_hz"])]
    cycle_values = [float(entry["approx_cycle_span"]) for entry in waveform_matches if np.isfinite(entry["approx_cycle_span"])]
    output_values = [float(entry[output_column]) for entry in waveform_matches if np.isfinite(entry[output_column])]
    freq_range = max((max(freq_values) - min(freq_values)) if freq_values else 0.0, 1e-9)
    cycle_range = max((max(cycle_values) - min(cycle_values)) if cycle_values else 0.0, 1e-9)
    output_range = max((max(output_values) - min(output_values)) if output_values else 0.0, 1e-9)

    for entry in waveform_matches:
        support_output = float(entry.get(output_column, np.nan))
        if not np.isfinite(support_output) or support_output <= 0:
            continue
        freq_distance = abs(float(entry.get("freq_hz", np.nan)) - float(freq_hz)) if np.isfinite(entry.get("freq_hz", np.nan)) else 1e6
        cycle_distance = abs(float(entry.get("approx_cycle_span", np.nan)) - float(target_cycle_count)) if np.isfinite(entry.get("approx_cycle_span", np.nan)) else 1e6
        output_distance = abs(support_output - float(target_output_pp))
        distance_score = np.sqrt(
            np.square(freq_distance / freq_range)
            + np.square(cycle_distance / cycle_range)
            + np.square(output_distance / output_range)
        )
        working_entries.append((distance_score, entry))

    if not working_entries:
        return None

    working_entries.sort(key=lambda item: item[0])
    distance_score, support = working_entries[0]
    support_output_pp = float(support[output_column])
    scale_ratio = float(target_output_pp / support_output_pp) if support_output_pp else float("nan")
    if not np.isfinite(scale_ratio):
        return None

    frame = support["frame"].copy()
    if "time_s" not in frame.columns or "daq_input_v" not in frame.columns:
        return None
    if target_output_type == "field":
        source_output_column = main_field_axis if main_field_axis in frame.columns else "bz_mT"
    else:
        source_output_column = current_channel if current_channel in frame.columns else "i_sum_signed"

    modeled = pd.DataFrame({"time_s": pd.to_numeric(frame["time_s"], errors="coerce")}).dropna().copy()
    modeled["recommended_voltage_v"] = pd.to_numeric(frame.loc[modeled.index, "daq_input_v"], errors="coerce") * scale_ratio
    modeled["modeled_output"] = pd.to_numeric(frame.loc[modeled.index, source_output_column], errors="coerce") * scale_ratio
    modeled["target_output"] = np.nan
    if len(modeled) >= 2:
        target_template = _finite_target_template(
            time_grid=modeled["time_s"].to_numpy(dtype=float),
            waveform_type=waveform_type,
            freq_hz=float(freq_hz),
            target_cycle_count=float(target_cycle_count),
            target_output_pp=float(target_output_pp),
        )
        modeled["target_output"] = target_template
    support_amp_gain_pct = float(default_support_amp_gain_pct)
    modeled = apply_command_hardware_model(
        command_waveform=modeled,
        max_daq_voltage_pp=float(max_daq_voltage_pp),
        amp_gain_at_100_pct=float(amp_gain_at_100_pct),
        support_amp_gain_pct=support_amp_gain_pct,
        amp_gain_limit_pct=float(amp_gain_limit_pct),
        amp_max_output_pk_v=float(amp_max_output_pk_v),
        preserve_start_voltage=True,
    )
    return {
        "support_test_id": support["test_id"],
        "support_freq_hz": float(support.get("freq_hz", np.nan)),
        "support_cycle_count": float(support.get("approx_cycle_span", np.nan)),
        "support_output_pp": support_output_pp,
        "scale_ratio": scale_ratio,
        "distance_score": float(distance_score),
        "target_output_unit": unit,
        "modeled_frame": modeled,
    }


def _finite_target_template(
    time_grid: np.ndarray,
    waveform_type: str,
    freq_hz: float,
    target_cycle_count: float,
    target_output_pp: float,
) -> np.ndarray:
    if len(time_grid) == 0:
        return np.array([], dtype=float)
    period_s = 1.0 / float(freq_hz) if float(freq_hz) > 0 else 1.0
    active_end_s = float(target_cycle_count) * period_s
    values = np.zeros_like(time_grid, dtype=float)
    active_mask = (time_grid >= 0.0) & (time_grid <= active_end_s + 1e-12)
    if not active_mask.any():
        return values
    cycle_progress_total = np.clip(time_grid[active_mask] / period_s, 0.0, float(target_cycle_count))
    cycle_phase = np.mod(cycle_progress_total, 1.0)
    if waveform_type == "triangle":
        normalized = np.where(
            cycle_phase < 0.25,
            cycle_phase * 4.0,
            np.where(
                cycle_phase < 0.75,
                2.0 - cycle_phase * 4.0,
                cycle_phase * 4.0 - 4.0,
            ),
        )
    else:
        normalized = np.sin(2.0 * np.pi * cycle_phase)
    values[active_mask] = normalized * float(target_output_pp) / 2.0
    return values


def _render_finite_run_section(
    transient_measurements: list,
    transient_preprocess_results: list,
    current_channel: str,
    main_field_axis: str,
) -> None:
    st.markdown("#### finite-cycle 측정 확인")
    st.caption(
        "이 화면은 0.75 / 1.0 / 1.25 / 1.5 cycle처럼 시작 후 멈추는 transient 데이터를 raw/corrected 기준으로 확인하는 용도입니다. "
        "steady-state LUT와 별도로, 실제 정지 응답을 눈으로 검증하는 화면입니다."
    )
    st.info("여기서 보는 finite 데이터는 향후 transient 전용 모델 support용입니다. 현재 메인 보정 로직 자체는 steady-state 중심입니다.")
    if not transient_measurements:
        st.info("finite-cycle 전용 데이터를 업로드하면 여기서 정지 응답 파형을 확인할 수 있습니다.")
        return

    summary_rows = [
        _build_finite_run_summary(parsed, preprocess.corrected_frame)
        for parsed, preprocess in zip(transient_measurements, transient_preprocess_results, strict=False)
    ]
    summary_frame = pd.DataFrame(summary_rows)
    st.dataframe(summary_frame, use_container_width=True)

    selection_labels = [
        f"{row['test_id']} | {row['waveform_type']} | {row['freq_hz']:.3f} Hz | {row['approx_cycle_span']:.2f} cycle"
        if np.isfinite(row["freq_hz"]) and np.isfinite(row["approx_cycle_span"])
        else row["test_id"]
        for row in summary_rows
    ]
    selected_label = st.selectbox("finite-cycle 테스트 선택", options=selection_labels, key="finite_run_select")
    selected_index = selection_labels.index(selected_label)
    selected_parsed = transient_measurements[selected_index]
    selected_preprocess = transient_preprocess_results[selected_index]

    dataset_mode = st.radio(
        "데이터셋",
        options=["corrected", "raw"],
        horizontal=True,
        key="finite_run_dataset_mode",
    )
    display_frame = (
        selected_preprocess.corrected_frame
        if dataset_mode == "corrected"
        else selected_parsed.normalized_frame
    )

    default_channels = [
        "daq_input_v",
        current_channel,
        "i_sum_signed",
        main_field_axis,
        "bz_mT",
        "bmag_mT",
        "temperature_c",
    ]
    available_channels = [
        column
        for column in display_frame.columns
        if column not in {"source_file", "sheet_name", "test_id", "notes", "parse_warnings"}
    ]
    selected_channels = st.multiselect(
        "표시 채널",
        options=available_channels,
        default=[channel for channel in default_channels if channel in available_channels],
        key="finite_run_channels",
    )
    st.plotly_chart(
        plot_waveforms(display_frame, selected_channels, title=f"{summary_rows[selected_index]['test_id']} / {dataset_mode}"),
        use_container_width=True,
    )

    meta_left, meta_right = st.columns(2)
    with meta_left:
        st.markdown("#### run 정보")
        for key in ("source_file", "sheet_name", "waveform_type", "freq_hz", "target_current_a", "duration_s", "approx_cycle_span", "notes"):
            value = summary_rows[selected_index].get(key)
            st.write(f"- {key}: `{value}`")
    with meta_right:
        st.markdown("#### preprocessing 로그")
        if selected_preprocess.logs:
            st.caption(" / ".join(selected_preprocess.logs))
        if selected_preprocess.warnings:
            st.warning(" | ".join(selected_preprocess.warnings))
        if selected_parsed.warnings:
            st.warning(" | ".join(selected_parsed.warnings))

    st.dataframe(display_frame.head(300), use_container_width=True)


def _render_data_import_tab(
    previews: list,
    parsed_measurements: list,
    edited_metadata: pd.DataFrame,
    warning_table: pd.DataFrame,
    transient_previews: list | None = None,
    transient_parsed_measurements: list | None = None,
    transient_edited_metadata: pd.DataFrame | None = None,
    validation_previews: list | None = None,
    validation_parsed_measurements: list | None = None,
    validation_edited_metadata: pd.DataFrame | None = None,
    lcr_uploads: list[dict] | None = None,
) -> None:
    summary_rows = []
    for preview in previews:
        for sheet_preview in preview.sheet_previews:
            summary_rows.append(
                {
                    "file_name": preview.file_name,
                    "file_type": preview.file_type,
                    "sheet_name": sheet_preview.sheet_name,
                    "row_count": sheet_preview.row_count,
                    "column_count": sheet_preview.column_count,
                    "warning_count": len(sheet_preview.warnings),
                }
            )
    st.markdown("#### 연속 cycle 입력 요약")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
    if not edited_metadata.empty:
        st.markdown("#### 메타데이터 편집 결과")
        st.dataframe(edited_metadata, use_container_width=True)

    st.markdown("#### 구조 Preview / 추천 매핑 / 파싱 로그")
    for preview in previews:
        with st.expander(preview.file_name, expanded=False):
            for sheet_preview in preview.sheet_previews:
                st.markdown(f"**Sheet:** `{sheet_preview.sheet_name}`")
                st.write(f"- 행 수: {sheet_preview.row_count}")
                st.write(f"- 컬럼: {', '.join(sheet_preview.columns)}")
                st.write(f"- 메타데이터: {sheet_preview.metadata}")
                st.dataframe(pd.DataFrame(sheet_preview.preview_rows), use_container_width=True)
                if sheet_preview.recommended_mapping:
                    st.dataframe(pd.DataFrame([sheet_preview.recommended_mapping]), use_container_width=True)
                if sheet_preview.warnings:
                    st.warning(" | ".join(sheet_preview.warnings))
                if sheet_preview.logs:
                    st.caption(" / ".join(sheet_preview.logs))

    transient_previews = transient_previews or []
    transient_parsed_measurements = transient_parsed_measurements or []
    transient_edited_metadata = transient_edited_metadata if transient_edited_metadata is not None else pd.DataFrame()
    if transient_previews:
        transient_rows = []
        for preview in transient_previews:
            for sheet_preview in preview.sheet_previews:
                transient_rows.append(
                    {
                        "file_name": preview.file_name,
                        "file_type": preview.file_type,
                        "sheet_name": sheet_preview.sheet_name,
                        "row_count": sheet_preview.row_count,
                        "column_count": sheet_preview.column_count,
                        "warning_count": len(sheet_preview.warnings),
                    }
                )
        st.markdown("#### finite-cycle 입력 대기열")
        st.info("이 입력군은 1 cycle / 0.75 cycle / 1.25 cycle 전용 모델링 데이터용으로 분리되어 있습니다.")
        st.dataframe(pd.DataFrame(transient_rows), use_container_width=True)
        if not transient_edited_metadata.empty:
            st.markdown("#### finite-cycle 메타데이터 편집 결과")
            st.dataframe(transient_edited_metadata, use_container_width=True)
        for preview in transient_previews:
            with st.expander(f"[finite] {preview.file_name}", expanded=False):
                for sheet_preview in preview.sheet_previews:
                    st.markdown(f"**Sheet:** `{sheet_preview.sheet_name}`")
                    st.write(f"- 행 수: {sheet_preview.row_count}")
                    st.write(f"- 컬럼: {', '.join(sheet_preview.columns)}")
                    st.write(f"- 메타데이터: {sheet_preview.metadata}")
                    st.dataframe(pd.DataFrame(sheet_preview.preview_rows), use_container_width=True)
        if transient_parsed_measurements:
            st.markdown("#### finite-cycle 정규화된 매핑")
            mapping_frames = []
            for parsed in transient_parsed_measurements:
                mapping_table = build_mapping_table(parsed.mapping, load_schema_config(str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None))
                mapping_table["source_file"] = parsed.source_file
                mapping_table["sheet_name"] = parsed.sheet_name
                mapping_frames.append(mapping_table)
            st.dataframe(pd.concat(mapping_frames, ignore_index=True), use_container_width=True)

    validation_previews = validation_previews or []
    validation_parsed_measurements = validation_parsed_measurements or []
    validation_edited_metadata = validation_edited_metadata if validation_edited_metadata is not None else pd.DataFrame()
    if validation_previews:
        validation_rows = []
        for preview in validation_previews:
            for sheet_preview in preview.sheet_previews:
                validation_rows.append(
                    {
                        "file_name": preview.file_name,
                        "file_type": preview.file_type,
                        "sheet_name": sheet_preview.sheet_name,
                        "row_count": sheet_preview.row_count,
                        "column_count": sheet_preview.column_count,
                        "warning_count": len(sheet_preview.warnings),
                    }
                )
        st.markdown("#### 2차 보정 검증 run 입력 대기열")
        st.dataframe(pd.DataFrame(validation_rows), use_container_width=True)
        if not validation_edited_metadata.empty:
            st.markdown("#### 2차 보정 검증 run 메타데이터 편집 결과")
            st.dataframe(validation_edited_metadata, use_container_width=True)
        for preview in validation_previews:
            with st.expander(f"[validation] {preview.file_name}", expanded=False):
                for sheet_preview in preview.sheet_previews:
                    st.markdown(f"**Sheet:** `{sheet_preview.sheet_name}`")
                    st.write(f"- 행 수: {sheet_preview.row_count}")
                    st.write(f"- 컬럼: {', '.join(sheet_preview.columns)}")
                    st.write(f"- 메타데이터: {sheet_preview.metadata}")
                    st.dataframe(pd.DataFrame(sheet_preview.preview_rows), use_container_width=True)
        if validation_parsed_measurements:
            st.markdown("#### validation run 정규화된 매핑")
            mapping_frames = []
            for parsed in validation_parsed_measurements:
                mapping_table = build_mapping_table(parsed.mapping, load_schema_config(str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None))
                mapping_table["source_file"] = parsed.source_file
                mapping_table["sheet_name"] = parsed.sheet_name
                mapping_frames.append(mapping_table)
            st.dataframe(pd.concat(mapping_frames, ignore_index=True), use_container_width=True)

    lcr_uploads = lcr_uploads or []
    if lcr_uploads:
        st.markdown("#### LCR 업로드")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "file_name": item.get("display_name") or item.get("file_name"),
                        "cache_name": item.get("cache_name"),
                        "size_bytes": item.get("size_bytes"),
                        "source": item.get("source"),
                        "path": item.get("path"),
                    }
                    for item in lcr_uploads
                ]
            ),
            use_container_width=True,
        )

    if parsed_measurements:
        st.markdown("#### 정규화된 매핑")
        mapping_frames = []
        for parsed in parsed_measurements:
            mapping_table = build_mapping_table(parsed.mapping, load_schema_config(str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None))
            mapping_table["source_file"] = parsed.source_file
            mapping_table["sheet_name"] = parsed.sheet_name
            mapping_frames.append(mapping_table)
        st.dataframe(pd.concat(mapping_frames, ignore_index=True), use_container_width=True)

    if not warning_table.empty:
        st.markdown("#### 경고")
        st.dataframe(warning_table, use_container_width=True)


def _render_mapping_editor(schema, previews: list) -> dict[str, str | None]:
    column_pool: list[str] = []
    for preview in previews:
        for sheet_preview in preview.sheet_previews:
            column_pool.extend(sheet_preview.columns)
    options = [""] + sorted(dict.fromkeys(column_pool))
    default_mapping = previews[0].sheet_previews[0].recommended_mapping if previews and previews[0].sheet_previews else {}

    with st.expander("컬럼 매핑 조정", expanded=False):
        overrides: dict[str, str | None] = {}
        columns = st.columns(2)
        for index, (field_key, spec) in enumerate(schema.field_specs.items()):
            target_column = columns[index % 2]
            with target_column:
                default_value = default_mapping.get(field_key) or ""
                selected = st.selectbox(
                    f"{spec.label_ko} ({field_key})",
                    options=options,
                    index=options.index(default_value) if default_value in options else 0,
                    key=f"mapping_{field_key}",
                )
                overrides[field_key] = selected or None
        return overrides


def _build_metadata_editor_rows(previews: list) -> pd.DataFrame:
    rows = []
    for preview in previews:
        for sheet_preview in preview.sheet_previews:
            waveform_value = sheet_preview.metadata.get("waveform")
            if waveform_value in (None, "", "0", "0.0"):
                waveform_value = infer_waveform_from_text(preview.file_name, sheet_preview.sheet_name)
            freq_value = sheet_preview.metadata.get("frequency(Hz)")
            if freq_value in (None, "", "0", "0.0", "0.000"):
                freq_value = infer_frequency_from_text(preview.file_name, sheet_preview.sheet_name)
            target_current = sheet_preview.metadata.get("Target Current(A)")
            target_current_value = first_number(target_current)
            if target_current_value is None or target_current_value <= 0:
                target_current_value = infer_current_from_text(preview.file_name, sheet_preview.sheet_name)
            rows.append(
                {
                    "source_file": preview.file_name,
                    "sheet_name": sheet_preview.sheet_name,
                    "waveform_type": waveform_value,
                    "freq_hz": first_number(freq_value),
                    "target_current_a": target_current_value,
                    "notes": sheet_preview.metadata.get("notes", ""),
                }
            )
    return pd.DataFrame(rows)


def _group_metadata_overrides(edited_metadata: pd.DataFrame) -> dict[str, dict[str, dict[str, object]]]:
    overrides: dict[str, dict[str, dict[str, object]]] = {}
    if edited_metadata.empty:
        return overrides
    for row in edited_metadata.to_dict(orient="records"):
        file_overrides = overrides.setdefault(str(row["source_file"]), {})
        file_overrides[str(row["sheet_name"])] = {
            "waveform": row.get("waveform_type"),
            "frequency(Hz)": row.get("freq_hz"),
            "Target Current(A)": row.get("target_current_a"),
            "notes": row.get("notes"),
        }
    return overrides
