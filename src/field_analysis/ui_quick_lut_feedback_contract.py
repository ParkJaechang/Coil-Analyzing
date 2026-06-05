from __future__ import annotations

import pandas as pd

FIELD_TARGET_LABEL = "목표 자기장"
MEASURED_FIELD_LABEL = "실측 자기장 (선택 부호, 정규화)"
RESIDUAL_LABEL = "오차 (목표 - 실측)"
FIRST_VOLTAGE_LABEL = "1차 모델링 전압"
ACTUAL_VOLTAGE_LABEL = "실제 구동 전압"
SECOND_VOLTAGE_LABEL = "2차 모델링 전압"
CORRECTION_DELTA_LABEL = "보정 전압 변화량"
ACTIVE_COMMAND_LABEL = "현재 표시 중인 전압 명령"
BASELINE_VOLTAGE_LABEL = "1차 추천/제한 전압"
FEEDBACK_LIMITED_LABEL = "피드백 보정 후 제한 전압"


def feedback_export_source_column(command_profile: pd.DataFrame) -> str:
    if "feedback_corrected_limited_voltage_v" not in command_profile.columns:
        return "limited_voltage_v"
    if "feedback_correction_status" in command_profile.columns and len(command_profile):
        if str(command_profile["feedback_correction_status"].iloc[0]) != "ok":
            return "limited_voltage_v"
    if "feedback_correction_available" in command_profile.columns and len(command_profile):
        if not bool(command_profile["feedback_correction_available"].iloc[0]):
            return "limited_voltage_v"
    return "feedback_corrected_limited_voltage_v"


def build_command_source_rows(command_profile: pd.DataFrame, metadata: dict[str, object] | None = None) -> list[dict[str, object]]:
    metadata = metadata or {}
    active_source = feedback_export_source_column(command_profile)
    predicted_valid = bool(
        metadata.get("predicted_from_plotted_command", False)
        and "feedback_corrected_predicted_field_mT" in command_profile.columns
    )
    return [
        {"field": "active_command_source", "value": active_source},
        {"field": "plotted_command_source", "value": active_source},
        {"field": "exported_voltage_source_column", "value": active_source},
        {"field": "run_waveform_voltage_source", "value": active_source},
        {"field": "feedback_used_for_correction", "value": bool(metadata.get("feedback_used_for_correction", False))},
        {"field": "predicted_from_plotted_command", "value": bool(metadata.get("predicted_from_plotted_command", False))},
        {"field": "displayed_predicted_valid", "value": predicted_valid},
        {
            "field": "command_prediction_consistency_status",
            "value": metadata.get(
                "command_prediction_consistency_status",
                "ok" if predicted_valid else "forward_prediction_unavailable_for_feedback_corrected_command",
            ),
        },
        {"field": "correction_method", "value": metadata.get("correction_method", "residual_proportional_feedback")},
    ]


def build_feedback_status_rows(metadata: dict[str, object]) -> list[dict[str, object]]:
    route = metadata.get("feedback_route") or "finite_actual_feedback_peak_correction"
    if route == "finite_feedback_symmetric_peak_correction":
        route = "finite_actual_feedback_peak_correction"
    fields = [
        ("route", route),
        ("feedback_correction_available", metadata.get("feedback_correction_available", False)),
        ("feedback_correction_status", metadata.get("feedback_correction_status", "unavailable")),
        ("supported cycles", "1.0, 1.5"),
        ("unsupported cycles", "1.25, 1.75, 2.0"),
        ("unsupported reason", "unsupported_cycle_policy_1p0_1p5_only"),
        ("production cycle policy", "1p0_1p5_cycles"),
        ("filename", metadata.get("feedback_source_file", "unavailable")),
        ("parse status", metadata.get("feedback_schema_status", "unavailable")),
        ("alignment status", metadata.get("feedback_alignment_status") or metadata.get("alignment_status", "unavailable")),
        ("target_unchanged", metadata.get("target_unchanged", True)),
    ]
    return [{"field": field, "value": value} for field, value in fields]


def build_feedback_plot_frame(command_profile: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"time_s": pd.to_numeric(command_profile["time_s"], errors="coerce")})
    columns = {
        "physical_target_output_mT": FIELD_TARGET_LABEL,
        "measured_field_normalized_mT": MEASURED_FIELD_LABEL,
        "limited_voltage_v": ACTIVE_COMMAND_LABEL,
        "baseline_limited_voltage_v": BASELINE_VOLTAGE_LABEL,
        "feedback_correction_delta_v": CORRECTION_DELTA_LABEL,
        "feedback_corrected_limited_voltage_v": FEEDBACK_LIMITED_LABEL,
        "feedback_corrected_predicted_field_mT": "피드백 보정 예측 자기장",
    }
    for source, label in columns.items():
        if source in command_profile.columns:
            frame[label] = pd.to_numeric(command_profile[source], errors="coerce")
    if {FIELD_TARGET_LABEL, MEASURED_FIELD_LABEL}.issubset(frame.columns):
        frame[RESIDUAL_LABEL] = frame[FIELD_TARGET_LABEL] - frame[MEASURED_FIELD_LABEL]
    return frame
