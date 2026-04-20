from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from policy_validation_runtime import (
    button_labels_from_summary,
    run_case,
    synthesize_case_messages,
)


DEFAULT_OUTPUT_JSON = Path("artifacts/policy_eval/exact_matrix_ui_validation.json")
DEFAULT_OUTPUT_MD = Path("artifacts/policy_eval/exact_matrix_ui_validation.md")


@dataclass(slots=True)
class UiCaseConfig:
    name: str
    waveform: str
    target_type: str
    freq_hz: float
    compensation_target: float
    expected_support_state: str
    expected_preview_only: bool
    expected_allow_auto_download: bool
    finite_mode: bool = False
    cycle_count: float | None = None
    required_message_substrings: tuple[str, ...] = ()
    forbidden_message_substrings: tuple[str, ...] = ()


@dataclass(slots=True)
class UiCaseResult:
    name: str
    support_state: str
    preview_only: bool
    allow_auto_download: bool
    policy_version: str
    warning_messages: list[str]
    info_messages: list[str]
    success_messages: list[str]
    caption_messages: list[str]
    button_labels: list[str]
    command_profile_rows: int
    validation_report_reasons: list[str]
    pass_status: bool = False


def _assert_case(case: UiCaseResult, config: UiCaseConfig) -> None:
    if not config.expected_preview_only or config.expected_allow_auto_download:
        assert case.command_profile_rows > 0, f"{config.name}: empty recommended waveform"
    assert case.support_state == config.expected_support_state, (
        f"{config.name}: support_state {case.support_state!r} != {config.expected_support_state!r}"
    )
    assert case.preview_only is config.expected_preview_only, (
        f"{config.name}: preview_only {case.preview_only!r} != {config.expected_preview_only!r}"
    )
    assert case.allow_auto_download is config.expected_allow_auto_download, (
        f"{config.name}: allow_auto_download {case.allow_auto_download!r} != {config.expected_allow_auto_download!r}"
    )
    assert case.policy_version, f"{config.name}: missing policy_version"

    merged_messages = [
        *case.warning_messages,
        *case.info_messages,
        *case.success_messages,
        *case.caption_messages,
        *case.validation_report_reasons,
    ]
    for needle in config.required_message_substrings:
        assert any(needle in value for value in merged_messages), f"{config.name}: missing {needle!r}"
    for needle in config.forbidden_message_substrings:
        assert not any(needle in value for value in merged_messages), f"{config.name}: unexpected {needle!r}"


def _run_case(config: UiCaseConfig) -> UiCaseResult:
    result, route_summary, runtime = run_case(
        waveform=config.waveform,
        target_type=config.target_type,
        freq_hz=config.freq_hz,
        target_level=config.compensation_target,
        finite_cycle_mode=config.finite_mode,
        target_cycle_count=config.cycle_count,
    )
    engine_summary = getattr(result, "engine_summary", {}) or {}
    debug_info = getattr(result, "debug_info", {}) or {}
    message_groups = synthesize_case_messages(
        route_summary=route_summary,
        result=result,
        main_field_axis=runtime.main_field_axis,
        target_type=config.target_type,
    )
    validation_report = getattr(result, "validation_report", None)
    command_profile = getattr(result, "command_profile", None)
    recommended_time = getattr(result, "recommended_time_s", None)
    row_count = int(len(command_profile)) if command_profile is not None else int(len(recommended_time)) if recommended_time is not None else 0

    return UiCaseResult(
        name=config.name,
        support_state=str(engine_summary.get("support_state", "unknown")),
        preview_only=bool(getattr(result, "preview_only", False)),
        allow_auto_download=bool(getattr(result, "allow_auto_download", False)),
        policy_version=str(debug_info.get("policy_version") or engine_summary.get("policy_version") or ""),
        warning_messages=message_groups["warning_messages"],
        info_messages=message_groups["info_messages"],
        success_messages=message_groups["success_messages"],
        caption_messages=message_groups["caption_messages"],
        button_labels=button_labels_from_summary(route_summary),
        command_profile_rows=row_count,
        validation_report_reasons=list(getattr(validation_report, "reasons", []) or []),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate exact-matrix operational UI states without full AppTest.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    cases = [
        UiCaseConfig(
            name="continuous_exact_current_auto",
            waveform="sine",
            target_type="current",
            freq_hz=0.5,
            compensation_target=20.0,
            expected_support_state="exact",
            expected_preview_only=False,
            expected_allow_auto_download=True,
        ),
        UiCaseConfig(
            name="continuous_exact_field_ready",
            waveform="sine",
            target_type="field",
            freq_hz=0.25,
            compensation_target=20.0,
            expected_support_state="exact",
            expected_preview_only=False,
            expected_allow_auto_download=True,
            required_message_substrings=("software-ready", "bench sign-off"),
        ),
        UiCaseConfig(
            name="continuous_interpolated_preview",
            waveform="sine",
            target_type="current",
            freq_hz=0.75,
            compensation_target=20.0,
            expected_support_state="interpolated_in_hull",
            expected_preview_only=True,
            expected_allow_auto_download=False,
            required_message_substrings=("미리보기 전용",),
        ),
        UiCaseConfig(
            name="finite_provisional_sine_preview",
            waveform="sine",
            target_type="current",
            freq_hz=1.0,
            compensation_target=20.0,
            expected_support_state="provisional_preview",
            expected_preview_only=True,
            expected_allow_auto_download=False,
            finite_mode=True,
            cycle_count=1.0,
            required_message_substrings=("임시 대체 조합", "provisional"),
        ),
    ]

    results: list[UiCaseResult] = []
    failures: list[str] = []
    for config in cases:
        try:
            result = _run_case(config)
            _assert_case(result, config)
            result.pass_status = True
            results.append(result)
        except Exception as exc:
            results.append(
                UiCaseResult(
                    name=config.name,
                    support_state="error",
                    preview_only=False,
                    allow_auto_download=False,
                    policy_version="",
                    warning_messages=[],
                    info_messages=[],
                    success_messages=[],
                    caption_messages=[],
                    button_labels=[],
                    command_profile_rows=0,
                    validation_report_reasons=[],
                    pass_status=False,
                )
            )
            failures.append(f"{config.name}: {exc}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": [asdict(result) for result in results],
        "failures": failures,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Exact Matrix UI Validation",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        "",
        "| case | support_state | preview_only | allow_auto_download | pass |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.name,
                    result.support_state,
                    str(result.preview_only),
                    str(result.allow_auto_download),
                    str(result.pass_status),
                ]
            )
            + " |"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
