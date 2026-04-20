from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = REPO_ROOT / "artifacts" / "policy_eval"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _policy_summary_block(name: str, payload: dict) -> list[str]:
    confusion = payload.get("confusion_counts", {})
    return [
        f"### {name}",
        f"- policy_version: `{payload.get('policy_version')}`",
        f"- auto_count: `{payload.get('auto_count')}`",
        f"- false_auto_count: `{payload.get('false_auto_count')}`",
        f"- false_block_count: `{confusion.get('false_block', 0)}`",
        f"- mean_realized_shape_corr: `{payload.get('mean_realized_shape_corr'):.6f}`",
        f"- mean_realized_nrmse: `{payload.get('mean_realized_nrmse'):.6f}`",
        f"- mean_predicted_error_band: `{payload.get('mean_predicted_error_band'):.6f}`",
        "",
    ]


def main() -> None:
    v2 = _read_json(POLICY_DIR / "policy_eval_v2_continuous_corpus_l1fo.json")
    v3_candidate = _read_json(POLICY_DIR / "policy_eval_v3_candidate_p95_continuous_corpus_l1fo.json")
    v3_geom = _read_json(POLICY_DIR / "policy_eval_v3_geom_p95_continuous_corpus_l1fo.json")
    export_validation = _read_json(POLICY_DIR / "exact_export_validation.json")
    ui_validation = _read_json(POLICY_DIR / "ui_policy_validation.json")
    finite_stage2 = _read_json(POLICY_DIR / "finite_generalization_stage2.json")

    lines: list[str] = [
        "# Feature Progress Report",
        "",
        "## Interpolation Confidence L1FO Comparison",
        "",
    ]
    lines.extend(_policy_summary_block("v2 (operational)", v2))
    lines.extend(_policy_summary_block("v3_candidate_p95", v3_candidate))
    lines.extend(_policy_summary_block("v3_geom_p95", v3_geom))
    lines.extend(
        [
            "## Exact Export Validation",
            "",
        ]
    )
    for case_name, case in export_validation.get("cases", {}).items():
        lines.extend(
            [
                f"### {case_name}",
                f"- support_state: `{case.get('support_state')}`",
                f"- preview_only: `{case.get('preview_only')}`",
                f"- allow_auto_download: `{case.get('allow_auto_download')}`",
                f"- csv_size_bytes: `{case.get('csv_size_bytes')}`",
                f"- formula_size_bytes: `{case.get('formula_size_bytes')}`",
                f"- lut_size_bytes: `{case.get('lut_size_bytes')}`",
                f"- line_count: `{case.get('line_count')}`",
                f"- coeff_row_count: `{case.get('coeff_row_count')}`",
                f"- lut_row_count: `{case.get('lut_row_count')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## UI Policy Validation",
            "",
        ]
    )
    for case in ui_validation.get("cases", []):
        metrics = case.get("metrics", {})
        lines.extend(
            [
                f"### {case.get('name')}",
                f"- pass_status: `{case.get('pass_status')}`",
                f"- warnings: `{len(case.get('warnings', []))}`",
                f"- successes: `{len(case.get('successes', []))}`",
                f"- engine: `{metrics.get('엔진', 'n/a')}`",
                f"- support_state: `{metrics.get('지원 상태', 'n/a')}`",
                f"- auto_recommendation: `{metrics.get('자동 추천', 'n/a')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Finite Generalization Stage 2",
            "",
            f"- case_count: `{finite_stage2.get('case_count')}`",
            f"- preview_case_count: `{finite_stage2.get('preview_case_count')}`",
            f"- mean_shape_corr: `{finite_stage2.get('summary', {}).get('mean_shape_corr'):.6f}`",
            f"- mean_nrmse: `{finite_stage2.get('summary', {}).get('mean_nrmse'):.6f}`",
            f"- mean_phase_lag_s: `{finite_stage2.get('summary', {}).get('mean_phase_lag_s'):.6f}`",
            "",
        ]
    )
    lines.extend(
        [
            "## Current Status",
            "",
            "- continuous/current exact path: validated and unchanged",
            "- continuous interpolation: geometry-aware confidence added, but operational rollout remains closed",
            "- exact field path: export + UI policy path validated",
            "- finite exact path: export + UI policy path validated",
            "- finite preview stage 2: quantified, preview-only quality remains weak on corpus LORO",
            "- hardware smoke test: not executed in this environment",
            "",
        ]
    )

    md_path = POLICY_DIR / "feature_progress_report.md"
    json_path = POLICY_DIR / "feature_progress_report.json"
    md_path.write_text("\n".join(lines), encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(
            {
                "v2": v2,
                "v3_candidate_p95": v3_candidate,
                "v3_geom_p95": v3_geom,
                "exact_export_validation": export_validation,
                "ui_policy_validation": ui_validation,
                "finite_generalization_stage2": finite_stage2,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(str(md_path))


if __name__ == "__main__":
    main()
