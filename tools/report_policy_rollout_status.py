from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class PolicyEvalBundle:
    json_path: Path
    csv_path: Path


def _load_bundle(bundle: PolicyEvalBundle) -> tuple[dict[str, Any], pd.DataFrame]:
    summary = json.loads(bundle.json_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(bundle.csv_path)
    return summary, frame


def _int_count(mapping: dict[str, Any], key: str) -> int:
    return int(mapping.get(key, 0) or 0)


def _format_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def _format_float(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _bundle_row(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    confusion = dict(summary.get("confusion_counts") or {})
    return {
        "scope": name,
        "case_count": int(summary.get("case_count", 0) or 0),
        "auto_count": int(summary.get("auto_count", 0) or 0),
        "false_auto": _int_count(confusion, "false_auto"),
        "false_block": _int_count(confusion, "false_block"),
        "correct_auto": _int_count(confusion, "correct_auto"),
        "correct_block": _int_count(confusion, "correct_block"),
        "margin_source": str(summary.get("policy_margin_source", "")),
        "support_state_counts": dict(summary.get("support_state_counts") or {}),
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _ui_case_summary(ui_data: dict[str, Any]) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for case in ui_data.get("cases", []):
        metrics = case.get("metrics") or {}
        warnings = case.get("warnings") or []
        summaries.append(
            {
                "name": str(case.get("name", "")),
                "support_state": str(metrics.get("지원 상태", "")),
                "auto_state": str(metrics.get("자동 추천", "")),
                "engine": str(metrics.get("엔진", "")),
                "warning": " | ".join(str(item) for item in warnings),
            }
        )
    return summaries


def _make_markdown(
    *,
    generated_at: str,
    full_v2: dict[str, Any],
    full_v3: dict[str, Any],
    sine_v2: dict[str, Any],
    sine_v3: dict[str, Any],
    triangle_v2: dict[str, Any],
    triangle_v3: dict[str, Any],
    ui_data: dict[str, Any] | None,
) -> str:
    rows = [
        _bundle_row("full/v2", full_v2),
        _bundle_row("full/v3_candidate_p95", full_v3),
        _bundle_row("sine/v2", sine_v2),
        _bundle_row("sine/v3_candidate_p95", sine_v3),
        _bundle_row("triangle/v2", triangle_v2),
        _bundle_row("triangle/v3_candidate_p95", triangle_v3),
    ]
    lines: list[str] = []
    lines.append("# Policy Rollout Status")
    lines.append("")
    lines.append(f"- generated_at_utc: `{generated_at}`")
    lines.append("- evaluation_mode: `leave-one-frequency-out`")
    lines.append("- promotion_scope_under_review: `interpolated_in_hull + current target`")
    lines.append("- field_target: `hold`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        _markdown_table(
            rows,
            [
                "scope",
                "case_count",
                "auto_count",
                "false_auto",
                "false_block",
                "correct_auto",
                "correct_block",
                "margin_source",
                "support_state_counts",
            ],
        )
    )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("- Operational policy remains `v2`.")
    lines.append("- `v3_candidate_p95` is not eligible for promotion on the larger continuous corpus because it introduces false auto cases.")
    lines.append(
        f"- Full corpus result: `v2 false_auto={_int_count(full_v2.get('confusion_counts', {}), 'false_auto')}`, "
        f"`v3 false_auto={_int_count(full_v3.get('confusion_counts', {}), 'false_auto')}`."
    )
    lines.append(
        f"- `v3_candidate_p95` auto count rises to `{int(full_v3.get('auto_count', 0) or 0)}`, but only "
        f"`{_int_count(full_v3.get('confusion_counts', {}), 'correct_auto')}` of those are correct auto cases."
    )
    lines.append(
        f"- Sine is the failure driver: `false_auto={_int_count(sine_v3.get('confusion_counts', {}), 'false_auto')}` on `"
        f"{int(sine_v3.get('case_count', 0) or 0)}` interpolated sine cases."
    )
    lines.append(
        f"- Triangle remains preview-only in practice: `v3 auto_count={int(triangle_v3.get('auto_count', 0) or 0)}`, "
        f"`false_auto={_int_count(triangle_v3.get('confusion_counts', {}), 'false_auto')}`."
    )
    lines.append("")
    lines.append("## Margin Audit")
    lines.append("")
    lines.append(
        f"- `v2` policy margin source: `{full_v2.get('policy_margin_source')}`; "
        f"mean gain/peak/p95 margins = `{_format_pct(full_v2.get('mean_input_limit_margin'))}` / "
        f"`{_format_pct(full_v2.get('mean_peak_input_limit_margin'))}` / "
        f"`{_format_pct(full_v2.get('mean_p95_input_limit_margin'))}`."
    )
    lines.append(
        f"- `v3_candidate_p95` margin source: `{full_v3.get('policy_margin_source')}`; "
        f"mean gain/peak/p95 margins = `{_format_pct(full_v3.get('mean_input_limit_margin'))}` / "
        f"`{_format_pct(full_v3.get('mean_peak_input_limit_margin'))}` / "
        f"`{_format_pct(full_v3.get('mean_p95_input_limit_margin'))}`."
    )
    lines.append("- The p95 margin is materially less conservative than the legacy gain margin, but it is still not sufficient as a promotion rule on its own.")
    lines.append("")
    if ui_data:
        lines.append("## UI Validation")
        lines.append("")
        lines.append(f"- method: `{ui_data.get('validation_method')}`")
        for limitation in ui_data.get("limitations", []):
            lines.append(f"- limitation: `{limitation}`")
        lines.append("")
        lines.append(
            _markdown_table(
                _ui_case_summary(ui_data),
                ["name", "support_state", "auto_state", "engine", "warning"],
            )
        )
        lines.append("")
    lines.append("## Operational Scope")
    lines.append("")
    lines.append("- Promote now: none beyond current `exact` auto path.")
    lines.append("- Keep as preview-only: `interpolated_in_hull + current target`.")
    lines.append("- Keep blocked/held: `field target` auto promotion, `interpolated_edge`, `out_of_hull`.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize policy evaluation artifacts into a rollout report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/policy_eval/policy_rollout_status_2026-04-13.md"),
        help="Markdown output path.",
    )
    parser.add_argument(
        "--ui-validation",
        type=Path,
        default=Path("artifacts/policy_eval/ui_policy_validation.json"),
        help="Optional UI validation JSON path.",
    )
    args = parser.parse_args()

    base = Path("artifacts/policy_eval")
    bundles = {
        "full_v2": PolicyEvalBundle(
            json_path=base / "policy_eval_v2_continuous_corpus_l1fo.json",
            csv_path=base / "policy_eval_v2_continuous_corpus_l1fo.csv",
        ),
        "full_v3": PolicyEvalBundle(
            json_path=base / "policy_eval_v3_candidate_p95_continuous_corpus_l1fo.json",
            csv_path=base / "policy_eval_v3_candidate_p95_continuous_corpus_l1fo.csv",
        ),
        "sine_v2": PolicyEvalBundle(
            json_path=base / "policy_eval_v2_sine_l1fo.json",
            csv_path=base / "policy_eval_v2_sine_l1fo.csv",
        ),
        "sine_v3": PolicyEvalBundle(
            json_path=base / "policy_eval_v3_candidate_p95_sine_l1fo.json",
            csv_path=base / "policy_eval_v3_candidate_p95_sine_l1fo.csv",
        ),
        "triangle_v2": PolicyEvalBundle(
            json_path=base / "policy_eval_v2_triangle_l1fo.json",
            csv_path=base / "policy_eval_v2_triangle_l1fo.csv",
        ),
        "triangle_v3": PolicyEvalBundle(
            json_path=base / "policy_eval_v3_candidate_p95_triangle_l1fo.json",
            csv_path=base / "policy_eval_v3_candidate_p95_triangle_l1fo.csv",
        ),
    }

    loaded = {name: _load_bundle(bundle)[0] for name, bundle in bundles.items()}
    ui_data = None
    if args.ui_validation.exists():
        ui_data = json.loads(args.ui_validation.read_text(encoding="utf-8"))

    report = _make_markdown(
        generated_at=datetime.now(timezone.utc).isoformat(),
        full_v2=loaded["full_v2"],
        full_v3=loaded["full_v3"],
        sine_v2=loaded["sine_v2"],
        sine_v3=loaded["sine_v3"],
        triangle_v2=loaded["triangle_v2"],
        triangle_v3=loaded["triangle_v3"],
        ui_data=ui_data,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote rollout report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
