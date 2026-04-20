from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from policy_validation_runtime import (
    EXACT_BUTTON_LABEL,
    PROVISIONAL_BUTTON_LABEL,
    apply_recipe_to_state,
    button_labels_from_summary,
    run_case,
)


DEFAULT_OUTPUT_JSON = Path("artifacts/policy_eval/request_router_validation.json")
DEFAULT_OUTPUT_MD = Path("artifacts/policy_eval/request_router_validation.md")


@dataclass(slots=True)
class RouterCaseConfig:
    name: str
    waveform: str
    target_type: str
    freq_hz: float
    compensation_target: float
    expected_state: str
    click_button_label: str
    expected_freq_after: float | None = None
    expected_target_after: float | None = None
    finite_mode: bool = False
    cycle_count: float | None = None
    expect_exact_button: bool = False
    expect_provisional_button: bool = False


@dataclass(slots=True)
class RouterCaseResult:
    name: str
    freq_hz_before: float
    freq_hz_after: float
    target_before: float
    target_after: float
    finite_mode_after: bool
    cycle_after: float | None
    button_labels: list[str]
    info_messages: list[str]
    warning_messages: list[str]
    success_messages: list[str]
    caption_messages: list[str]
    clicked_button_label: str
    pass_status: bool = False


def _assert_case(case: RouterCaseResult, config: RouterCaseConfig) -> None:
    assert (EXACT_BUTTON_LABEL in case.button_labels) is config.expect_exact_button, (
        f"{config.name}: exact button presence mismatch"
    )
    assert (PROVISIONAL_BUTTON_LABEL in case.button_labels) is config.expect_provisional_button, (
        f"{config.name}: provisional button presence mismatch"
    )
    if config.expected_freq_after is not None:
        assert abs(case.freq_hz_after - config.expected_freq_after) < 1e-9, (
            f"{config.name}: freq_after {case.freq_hz_after!r} != {config.expected_freq_after!r}"
        )
    if config.expected_target_after is not None:
        assert abs(case.target_after - config.expected_target_after) < 1e-9, (
            f"{config.name}: target_after {case.target_after!r} != {config.expected_target_after!r}"
        )


def _run_case(config: RouterCaseConfig) -> RouterCaseResult:
    result, route_summary, _runtime = run_case(
        waveform=config.waveform,
        target_type=config.target_type,
        freq_hz=config.freq_hz,
        target_level=config.compensation_target,
        finite_cycle_mode=config.finite_mode,
        target_cycle_count=config.cycle_count,
    )
    state = str(route_summary.get("state", "unsupported"))
    assert state == config.expected_state, f"{config.name}: state {state!r} != {config.expected_state!r}"

    button_labels = button_labels_from_summary(route_summary)
    recipe = None
    use_provisional_level = False
    if config.click_button_label == EXACT_BUTTON_LABEL:
        recipe = route_summary.get("nearest_exact")
    elif config.click_button_label == PROVISIONAL_BUTTON_LABEL:
        recipe = route_summary.get("nearest_provisional")
        use_provisional_level = True
    if not isinstance(recipe, dict):
        raise AssertionError(f"{config.name}: route recipe missing for {config.click_button_label!r}")

    applied = apply_recipe_to_state(
        recipe=recipe,
        finite_cycle_mode=config.finite_mode,
        use_provisional_level=use_provisional_level,
        freq_hz_before=config.freq_hz,
        target_before=config.compensation_target,
    )
    combined_status = f"{route_summary.get('status_label', '')}: {route_summary.get('reason', '')}".strip(": ")
    if state == "exact":
        success_messages = [combined_status]
        warning_messages = []
        info_messages = []
    elif state == "provisional":
        success_messages = []
        warning_messages = [combined_status]
        info_messages = []
    else:
        success_messages = []
        warning_messages = []
        info_messages = [combined_status]

    return RouterCaseResult(
        name=config.name,
        freq_hz_before=applied["freq_hz_before"],
        freq_hz_after=applied["freq_hz_after"],
        target_before=applied["target_before"],
        target_after=applied["target_after"],
        finite_mode_after=applied["finite_mode_after"],
        cycle_after=applied["cycle_after"],
        button_labels=button_labels,
        info_messages=info_messages,
        warning_messages=warning_messages,
        success_messages=success_messages,
        caption_messages=[],
        clicked_button_label=config.click_button_label,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate request-router transitions without full AppTest.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    cases = [
        RouterCaseConfig(
            name="continuous_preview_apply_exact",
            waveform="sine",
            target_type="current",
            freq_hz=0.75,
            compensation_target=20.0,
            expected_state="preview-only",
            click_button_label=EXACT_BUTTON_LABEL,
            expected_freq_after=0.5,
            expected_target_after=20.0,
            expect_exact_button=True,
            expect_provisional_button=False,
        ),
        RouterCaseConfig(
            name="finite_provisional_apply_exact",
            waveform="sine",
            target_type="current",
            freq_hz=1.0,
            compensation_target=20.0,
            finite_mode=True,
            cycle_count=1.0,
            expected_state="provisional",
            click_button_label=EXACT_BUTTON_LABEL,
            expected_freq_after=1.0,
            expected_target_after=10.0,
            expect_exact_button=True,
            expect_provisional_button=True,
        ),
        RouterCaseConfig(
            name="finite_provisional_apply_provisional",
            waveform="sine",
            target_type="current",
            freq_hz=1.0,
            compensation_target=20.0,
            finite_mode=True,
            cycle_count=1.0,
            expected_state="provisional",
            click_button_label=PROVISIONAL_BUTTON_LABEL,
            expected_freq_after=1.0,
            expected_target_after=20.0,
            expect_exact_button=True,
            expect_provisional_button=True,
        ),
    ]

    results: list[RouterCaseResult] = []
    failures: list[str] = []
    for config in cases:
        try:
            result = _run_case(config)
            _assert_case(result, config)
            result.pass_status = True
            results.append(result)
        except Exception as exc:
            failures.append(f"{config.name}: {exc}")
            results.append(
                RouterCaseResult(
                    name=config.name,
                    freq_hz_before=config.freq_hz,
                    freq_hz_after=float("nan"),
                    target_before=config.compensation_target,
                    target_after=float("nan"),
                    finite_mode_after=config.finite_mode,
                    cycle_after=config.cycle_count,
                    button_labels=[],
                    info_messages=[],
                    warning_messages=[],
                    success_messages=[],
                    caption_messages=[],
                    clicked_button_label=config.click_button_label,
                    pass_status=False,
                )
            )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": [asdict(result) for result in results],
        "failures": failures,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Request Router Validation",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        "",
        "| case | freq_before | freq_after | target_after | finite_mode_after | cycle_after | pass |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.name,
                    f"{result.freq_hz_before:g}",
                    f"{result.freq_hz_after:g}" if result.freq_hz_after == result.freq_hz_after else "nan",
                    f"{result.target_after:g}" if result.target_after == result.target_after else "nan",
                    str(result.finite_mode_after),
                    "" if result.cycle_after is None else f"{result.cycle_after:g}",
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
