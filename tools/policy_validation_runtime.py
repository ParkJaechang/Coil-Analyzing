from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

for logger_name in (
    "streamlit.runtime.caching.cache_data_api",
    "streamlit.runtime.scriptrunner_utils.script_run_context",
):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from field_analysis.app_ui import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    _apply_provisional_finite_preview_override,
    _analyze_measurements_cached,
    _build_default_app_settings,
    _build_request_route_summary,
    _canonicalize_measurements_cached,
    _load_persisted_upload_payloads,
    _parse_file_cached,
    _preprocess_measurements_cached,
    _target_type_label,
)
from field_analysis.analysis import combine_analysis_frames  # noqa: E402
from field_analysis.canonical_runs import CANONICAL_SCHEMA_VERSION  # noqa: E402
from field_analysis.models import CycleDetectionConfig, PreprocessConfig  # noqa: E402
from field_analysis.recommendation_service import (  # noqa: E402
    LegacyRecommendationContext,
    RecommendationOptions,
    RecommendationResult,
    TargetRequest,
    recommend,
)
from field_analysis.schema_config import load_schema_config  # noqa: E402


EXACT_BUTTON_LABEL = "가장 가까운 exact 조합으로 전환"
PROVISIONAL_BUTTON_LABEL = "가장 가까운 provisional 조합으로 미리보기"


@dataclass(slots=True)
class ValidationRuntime:
    continuous_runs: list[Any]
    transient_runs: list[Any]
    validation_runs: list[Any]
    recommendation_options: RecommendationOptions
    legacy_context: LegacyRecommendationContext
    main_field_axis: str


def _canonicalize_payloads(
    *,
    upload_kind: str,
    expected_cycles: int,
    target_current_mode: str,
    regime: str,
    role: str,
    canonicalize_config_json: str,
) -> list[Any]:
    payloads = _load_persisted_upload_payloads(upload_kind)
    if not payloads:
        return []
    config_path = str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None
    parsed_measurements: list[Any] = []
    for file_name, file_bytes in payloads:
        parsed_measurements.extend(
            _parse_file_cached(
                file_name=file_name,
                file_bytes=file_bytes,
                config_path=config_path,
                mapping_overrides_json=json.dumps({}, ensure_ascii=False),
                metadata_overrides_json=json.dumps({}, ensure_ascii=False),
                expected_cycles=expected_cycles,
                target_current_mode=target_current_mode,
            )
        )
    if not parsed_measurements:
        return []
    return _canonicalize_measurements_cached(
        parsed_measurements=parsed_measurements,
        regime=regime,
        role=role,
        canonical_schema_version=CANONICAL_SCHEMA_VERSION,
        canonicalize_config_json=canonicalize_config_json,
    )


def _parse_payloads(
    *,
    upload_kind: str,
    expected_cycles: int,
    target_current_mode: str,
) -> list[Any]:
    payloads = _load_persisted_upload_payloads(upload_kind)
    if not payloads:
        return []
    config_path = str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None
    parsed_measurements: list[Any] = []
    for file_name, file_bytes in payloads:
        parsed_measurements.extend(
            _parse_file_cached(
                file_name=file_name,
                file_bytes=file_bytes,
                config_path=config_path,
                mapping_overrides_json=json.dumps({}, ensure_ascii=False),
                metadata_overrides_json=json.dumps({}, ensure_ascii=False),
                expected_cycles=expected_cycles,
                target_current_mode=target_current_mode,
            )
        )
    return parsed_measurements


@lru_cache(maxsize=1)
def load_runtime() -> ValidationRuntime:
    schema = load_schema_config(str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None)
    settings = _build_default_app_settings(schema, "간단 LUT")
    canonicalize_config_json = json.dumps(
        {
            "preferred_field_axis": str(settings["main_field_axis"]),
            "uniform_resample": True,
            "custom_current_alpha": float(settings["custom_current_alpha"]),
            "custom_current_beta": float(settings["custom_current_beta"]),
        },
        ensure_ascii=False,
    )

    continuous_measurements = _parse_payloads(
        upload_kind="continuous",
        expected_cycles=int(settings["expected_cycles"]),
        target_current_mode=str(settings["target_current_mode"]),
    )
    transient_measurements = _parse_payloads(
        upload_kind="transient",
        expected_cycles=max(1, int(settings["expected_cycles"])),
        target_current_mode=str(settings["target_current_mode"]),
    )
    validation_measurements = _parse_payloads(
        upload_kind="validation",
        expected_cycles=max(1, int(settings["expected_cycles"])),
        target_current_mode=str(settings["target_current_mode"]),
    )
    continuous_runs = (
        _canonicalize_measurements_cached(
            parsed_measurements=continuous_measurements,
            regime="continuous",
            role="train",
            canonical_schema_version=CANONICAL_SCHEMA_VERSION,
            canonicalize_config_json=canonicalize_config_json,
        )
        if continuous_measurements
        else []
    )
    transient_runs = (
        _canonicalize_measurements_cached(
            parsed_measurements=transient_measurements,
            regime="transient",
            role="train",
            canonical_schema_version=CANONICAL_SCHEMA_VERSION,
            canonicalize_config_json=canonicalize_config_json,
        )
        if transient_measurements
        else []
    )
    validation_runs = (
        _canonicalize_measurements_cached(
            parsed_measurements=validation_measurements,
            regime="transient",
            role="validation",
            canonical_schema_version=CANONICAL_SCHEMA_VERSION,
            canonicalize_config_json=canonicalize_config_json,
        )
        if validation_measurements
        else []
    )
    preprocess_config = PreprocessConfig(
        baseline_seconds=float(settings["baseline_seconds"]),
        smoothing_method=str(settings["smoothing_method"]),
        smoothing_window=int(settings["smoothing_window"]),
        savgol_polyorder=int(settings["savgol_polyorder"]),
        alignment_reference="daq_input_v",
        alignment_targets=("bx_mT", "by_mT", "bz_mT"),
        apply_alignment=False,
        outlier_zscore_threshold=float(settings["outlier_threshold"]),
        sign_flips={},
        custom_current_alpha=float(settings["custom_current_alpha"]),
        custom_current_beta=float(settings["custom_current_beta"]),
        projection_vector=(
            float(settings["projection_nx"]),
            float(settings["projection_ny"]),
            float(settings["projection_nz"]),
        ),
    )
    cycle_config = CycleDetectionConfig(
        reference_channel="daq_input_v",
        expected_cycles=int(settings["expected_cycles"]),
        manual_start_s=None,
        manual_period_s=None,
    )
    analyses = (
        _analyze_measurements_cached(
            parsed_measurements=continuous_measurements,
            canonical_runs=continuous_runs,
            preprocess_config=preprocess_config,
            cycle_config=cycle_config,
            current_channel=str(settings["current_channel"]),
            main_field_axis=str(settings["main_field_axis"]),
        )
        if continuous_measurements
        else []
    )
    analysis_lookup = {
        analysis.parsed.normalized_frame["test_id"].iloc[0]: analysis
        for analysis in analyses
        if not analysis.parsed.normalized_frame.empty
    }
    _per_cycle_summary, per_test_summary, _coverage = combine_analysis_frames(
        analyses=analyses,
        reference_test_id=None,
        field_axis=str(settings["main_field_axis"]),
    )
    transient_preprocess_results = (
        _preprocess_measurements_cached(
            parsed_measurements=transient_measurements,
            preprocess_config=preprocess_config,
        )
        if transient_measurements
        else []
    )
    validation_preprocess_results = (
        _preprocess_measurements_cached(
            parsed_measurements=validation_measurements,
            preprocess_config=preprocess_config,
        )
        if validation_measurements
        else []
    )
    recommendation_options = RecommendationOptions(
        current_channel=str(settings["current_channel"]),
        field_channel=str(settings["main_field_axis"]),
        max_daq_voltage_pp=float(settings["max_daq_voltage_pp"]),
        amp_gain_at_100_pct=float(settings["amp_gain_at_100_pct"]),
        amp_gain_limit_pct=float(settings["amp_gain_limit_pct"]),
        amp_max_output_pk_v=float(settings["amp_max_output_pk_v"]),
        default_support_amp_gain_pct=float(settings["default_support_amp_gain_pct"]),
        allow_target_extrapolation=bool(settings["allow_target_extrapolation"]),
        allow_output_extrapolation=bool(settings["allow_target_extrapolation"]),
        frequency_mode="exact",
        preview_tail_cycles=0.25,
    )
    return ValidationRuntime(
        continuous_runs=continuous_runs,
        transient_runs=transient_runs,
        validation_runs=validation_runs,
        recommendation_options=recommendation_options,
        legacy_context=LegacyRecommendationContext(
            per_test_summary=per_test_summary,
            analysis_lookup=analysis_lookup,
            transient_measurements=transient_measurements,
            transient_preprocess_results=transient_preprocess_results,
            transient_canonical_runs=transient_runs,
            validation_measurements=validation_measurements,
            validation_preprocess_results=validation_preprocess_results,
        ),
        main_field_axis=str(settings["main_field_axis"]),
    )


def run_case(
    *,
    waveform: str,
    target_type: str,
    freq_hz: float,
    target_level: float,
    finite_cycle_mode: bool = False,
    target_cycle_count: float | None = None,
) -> tuple[RecommendationResult, dict[str, Any], ValidationRuntime]:
    runtime = load_runtime()
    result = recommend(
        continuous_runs=runtime.continuous_runs,
        transient_runs=runtime.transient_runs,
        validation_runs=runtime.validation_runs,
        target=TargetRequest(
            regime="transient" if finite_cycle_mode else "continuous",
            target_waveform=waveform,
            command_waveform=waveform,
            freq_hz=float(freq_hz),
            commanded_cycles=None if target_cycle_count is None else float(target_cycle_count),
            target_type=target_type,
            target_level_value=float(target_level),
            target_level_kind="pp",
            context={
                "request_kind": "waveform_compensation",
                "frequency_mode": "exact",
                "finite_cycle_mode": bool(finite_cycle_mode),
                "preview_tail_cycles": 0.25,
            },
        ),
        options=runtime.recommendation_options,
        legacy_context=runtime.legacy_context,
    )
    result = _apply_provisional_finite_preview_override(
        result=result,
        transient_runs=runtime.transient_runs,
        waveform=waveform,
        freq_hz=float(freq_hz),
        target_level=float(target_level),
        target_cycle_count=target_cycle_count,
        finite_cycle_mode=finite_cycle_mode,
    )
    summary = _build_request_route_summary(
        continuous_runs=runtime.continuous_runs,
        transient_runs=runtime.transient_runs,
        waveform=waveform,
        freq_hz=float(freq_hz),
        target_type=target_type,
        target_level=float(target_level),
        finite_cycle_mode=finite_cycle_mode,
        target_cycle_count=target_cycle_count,
        result=result,
    )
    return result, summary, runtime


def synthesize_case_messages(
    *,
    route_summary: dict[str, Any],
    result: RecommendationResult,
    main_field_axis: str,
    target_type: str,
) -> dict[str, list[str]]:
    support_label = str(route_summary.get("status_label", "") or "").strip()
    reason = str(route_summary.get("reason", "") or "").strip()
    state = str(route_summary.get("state", "unsupported"))
    combined = f"{support_label}: {reason}".strip(": ")

    warning_messages: list[str] = []
    info_messages: list[str] = []
    success_messages: list[str] = []
    caption_messages: list[str] = []

    if state == "exact":
        success_messages.append(combined)
    elif state == "provisional":
        warning_messages.append(combined)
    elif state == "preview-only":
        info_messages.append(combined)
    else:
        warning_messages.append(combined)

    if target_type == "field":
        target_label = _target_type_label(target_type, main_field_axis)
        success_messages.append(f"{target_label} exact는 software-ready이며 bench sign-off 전 단계입니다.")

    if route_summary.get("apply_exact_enabled"):
        caption_messages.append(EXACT_BUTTON_LABEL)
    if route_summary.get("apply_provisional_enabled"):
        caption_messages.append(PROVISIONAL_BUTTON_LABEL)

    recommendations = route_summary.get("measurement_recommendations")
    if isinstance(recommendations, list) and recommendations:
        preview = recommendations[:3]
        recommendation_text = " / ".join(
            (
                f"{item.get('waveform')} {float(item.get('freq_hz')):g} Hz"
                f"{f' {float(item.get('cycles')):g} cycle' if item.get('cycles') is not None else ''}"
                f" {float(item.get('level')):g}{' pp' if route_summary.get('finite_cycle_mode') else ' A'}"
            )
            for item in preview
            if item.get("freq_hz") is not None and item.get("level") is not None
        )
        if recommendation_text:
            caption_messages.append(f"추가 측정 추천: {recommendation_text}")

    return {
        "warning_messages": warning_messages,
        "info_messages": info_messages,
        "success_messages": success_messages,
        "caption_messages": caption_messages,
    }


def button_labels_from_summary(route_summary: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    if route_summary.get("apply_exact_enabled"):
        labels.append(EXACT_BUTTON_LABEL)
    if route_summary.get("apply_provisional_enabled"):
        labels.append(PROVISIONAL_BUTTON_LABEL)
    return labels


def apply_recipe_to_state(
    *,
    recipe: dict[str, Any],
    finite_cycle_mode: bool,
    use_provisional_level: bool,
    freq_hz_before: float,
    target_before: float,
) -> dict[str, Any]:
    freq_hz = recipe.get("freq_hz", freq_hz_before)
    cycles = recipe.get("cycles")
    level_key = "target_level_pp" if use_provisional_level else "recommended_level"
    target_level = recipe.get(level_key)
    if target_level is None:
        usable_levels = recipe.get("usable_levels")
        if isinstance(usable_levels, list) and usable_levels:
            target_level = usable_levels[0]
    return {
        "freq_hz_before": float(freq_hz_before),
        "freq_hz_after": float(freq_hz),
        "target_before": float(target_before),
        "target_after": float(target_level if target_level is not None else target_before),
        "finite_mode_after": bool(finite_cycle_mode),
        "cycle_after": float(cycles) if finite_cycle_mode and cycles is not None else None,
    }
