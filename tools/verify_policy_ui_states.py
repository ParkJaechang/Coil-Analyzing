from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest


APP_FILE = Path(__file__).resolve().parents[1] / "app_field_analysis_quick.py"

FREQ_INPUT_INDEX = 0
COMP_TARGET_INPUT_INDEX = 2
FINITE_CYCLE_INPUT_INDEX = 3
WAVEFORM_SELECT_INDEX = 21
TARGET_MODE_SELECT_INDEX = 23
TREND_CHECKBOX_INDEX = 0
FINITE_MODE_CHECKBOX_INDEX = 1
OPERATIONAL_MODE_CHECKBOX_INDEX = 2
CALCULATE_BUTTON_INDEX = 1


@dataclass(slots=True)
class UiCaseConfig:
    name: str
    waveform: str
    target_mode: str
    freq_hz: float
    compensation_target: float
    trend_enabled: bool = True
    finite_mode: bool = False
    operational_mode: bool = False
    cycle_count: float | None = None


@dataclass(slots=True)
class UiCaseResult:
    name: str
    waveform: str
    target_mode: str
    freq_hz: float
    compensation_target: float
    trend_enabled: bool
    finite_mode: bool
    operational_mode: bool
    cycle_count: float | None
    exception_messages: list[str]
    warnings: list[str]
    infos: list[str]
    successes: list[str]
    metrics: dict[str, str]
    captions: list[str]
    button_labels: list[str]
    pass_status: bool


def _string_values(elements: list[Any]) -> list[str]:
    values: list[str] = []
    for element in elements:
        value = str(getattr(element, "value", "") or "").strip()
        if value:
            values.append(value)
    return values


def _metric_map(app: AppTest) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for metric in app.metric:
        label = str(getattr(metric, "label", "") or "").strip()
        value = str(getattr(metric, "value", "") or "").strip()
        if label:
            metrics[label] = value
    return metrics


def _run_case(config: UiCaseConfig) -> UiCaseResult:
    app = AppTest.from_file(str(APP_FILE))
    app.run(timeout=120)
    app.selectbox[WAVEFORM_SELECT_INDEX].set_value(config.waveform)
    app.selectbox[TARGET_MODE_SELECT_INDEX].set_value(config.target_mode)
    app.checkbox[TREND_CHECKBOX_INDEX].set_value(config.trend_enabled)
    app.checkbox[FINITE_MODE_CHECKBOX_INDEX].set_value(config.finite_mode)
    app.checkbox[OPERATIONAL_MODE_CHECKBOX_INDEX].set_value(config.operational_mode)
    app.run(timeout=120)

    app.number_input[FREQ_INPUT_INDEX].set_value(config.freq_hz)
    app.number_input[COMP_TARGET_INPUT_INDEX].set_value(config.compensation_target)
    if config.finite_mode and config.cycle_count is not None:
        app.number_input[FINITE_CYCLE_INPUT_INDEX].set_value(config.cycle_count)
    app.run(timeout=120)

    app.button[CALCULATE_BUTTON_INDEX].click()
    app.run(timeout=120)

    exception_messages = [str(getattr(exc, "message", "") or "").strip() for exc in app.exception]
    return UiCaseResult(
        name=config.name,
        waveform=config.waveform,
        target_mode=config.target_mode,
        freq_hz=config.freq_hz,
        compensation_target=config.compensation_target,
        trend_enabled=config.trend_enabled,
        finite_mode=config.finite_mode,
        operational_mode=config.operational_mode,
        cycle_count=config.cycle_count,
        exception_messages=[message for message in exception_messages if message],
        warnings=_string_values(list(app.warning)),
        infos=_string_values(list(app.info)),
        successes=_string_values(list(app.success)),
        metrics=_metric_map(app),
        captions=_string_values(list(app.caption)),
        button_labels=[str(getattr(button, "label", "") or "").strip() for button in app.button],
        pass_status=False,
    )


def _require_metric(case: UiCaseResult, label: str, expected: str) -> None:
    actual = case.metrics.get(label)
    assert actual == expected, f"{case.name}: metric {label!r} expected {expected!r}, got {actual!r}"


def _require_any_contains(values: list[str], needle: str, *, case_name: str, kind: str) -> None:
    assert any(needle in value for value in values), f"{case_name}: {kind} missing {needle!r}"


def _assert_case_expectations(case: UiCaseResult) -> None:
    assert not case.exception_messages, f"{case.name}: unexpected exception {case.exception_messages}"
    _require_any_contains(case.successes, "파형 보정 계산이 완료됐습니다.", case_name=case.name, kind="success")
    _require_any_contains(case.captions, "policy version:", case_name=case.name, kind="caption")

    if case.name == "exact_current_auto":
        _require_metric(case, "엔진", "harmonic_surface")
        _require_metric(case, "지원 상태", "exact")
        _require_metric(case, "자동 추천", "가능")
    elif case.name == "exact_field_auto":
        _require_metric(case, "엔진", "harmonic_surface")
        _require_metric(case, "지원 상태", "exact")
        _require_metric(case, "자동 추천", "가능")
    elif case.name == "interpolated_current_preview":
        _require_metric(case, "엔진", "harmonic_surface")
        _require_metric(case, "지원 상태", "interpolated_in_hull")
        _require_metric(case, "자동 추천", "미리보기 전용")
        _require_any_contains(case.warnings, "preview only:", case_name=case.name, kind="warning")
    elif case.name == "finite_exact_sine":
        _require_metric(case, "엔진", "legacy")
        _require_metric(case, "지원 상태", "exact")
        _require_metric(case, "자동 추천", "가능")
    elif case.name == "finite_exact_triangle":
        _require_metric(case, "엔진", "legacy")
        _require_metric(case, "지원 상태", "exact")
        _require_metric(case, "자동 추천", "가능")
    elif case.name == "finite_missing_triangle_recipe":
        _require_metric(case, "엔진", "legacy")
        _require_metric(case, "지원 상태", "unsupported")
        _require_metric(case, "자동 추천", "미리보기 전용")
        _require_any_contains(case.warnings + case.infos, "exact recipe", case_name=case.name, kind="message")
        _require_any_contains(case.warnings + case.infos, "가장 가까운 사용 가능 조합", case_name=case.name, kind="message")
    else:
        raise AssertionError(f"unknown case: {case.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify exact/preview UI states with Streamlit AppTest.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/policy_eval/ui_policy_validation.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    cases = [
        UiCaseConfig(name="exact_current_auto", waveform="sine", target_mode="전류", freq_hz=0.5, compensation_target=20.0),
        UiCaseConfig(name="exact_field_auto", waveform="sine", target_mode="자기장 (bz_mT)", freq_hz=0.25, compensation_target=20.0),
        UiCaseConfig(name="interpolated_current_preview", waveform="sine", target_mode="전류", freq_hz=0.75, compensation_target=20.0),
        UiCaseConfig(
            name="finite_exact_sine",
            waveform="sine",
            target_mode="전류",
            freq_hz=0.5,
            compensation_target=20.0,
            finite_mode=True,
            operational_mode=True,
            cycle_count=1.0,
        ),
        UiCaseConfig(
            name="finite_exact_triangle",
            waveform="triangle",
            target_mode="전류",
            freq_hz=1.0,
            compensation_target=20.0,
            finite_mode=True,
            operational_mode=True,
            cycle_count=1.0,
        ),
        UiCaseConfig(
            name="finite_missing_triangle_recipe",
            waveform="triangle",
            target_mode="전류",
            freq_hz=5.0,
            compensation_target=20.0,
            finite_mode=True,
            operational_mode=True,
            cycle_count=1.0,
        ),
    ]

    results: list[UiCaseResult] = []
    for config in cases:
        result = _run_case(config)
        _assert_case_expectations(result)
        result.pass_status = True
        results.append(result)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "app_file": str(APP_FILE),
        "validation_method": "streamlit.testing.v1.AppTest",
        "scope": [
            "continuous current exact auto",
            "continuous field exact auto",
            "continuous current interpolated preview",
            "finite sine exact-supported",
            "finite triangle exact-supported",
            "finite triangle missing exact recipe preview",
        ],
        "limitations": [
            "Programmatic UI validation only; browser download buttons are validated separately by Selenium and offline export checks.",
            "AppTest indexes are tied to the current Quick LUT layout and should be refreshed if the form order changes.",
        ],
        "cases": [asdict(case) for case in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote UI validation artifact to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
