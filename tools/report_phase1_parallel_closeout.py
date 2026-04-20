from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


POLICY_DIR = Path("artifacts/policy_eval")
OUTPUT_JSON = POLICY_DIR / "phase1_parallel_closeout.json"
OUTPUT_MD = POLICY_DIR / "phase1_parallel_closeout.md"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    interpolation = _read_json(POLICY_DIR / "interpolation_confidence_comparison.json")
    modelside = _read_json(POLICY_DIR / "modelside_interpolation_attempts.json")
    field_status = _read_json(POLICY_DIR / "field_exact_promotion_status.json")
    scope = _read_json(POLICY_DIR / "exact_and_finite_scope.json")
    finite_stage2 = _read_json(POLICY_DIR / "finite_generalization_stage2.json")
    finite_data = _read_json(POLICY_DIR / "finite_data_collection_status.json")
    export_validation = _read_json(POLICY_DIR / "exact_export_validation.json")
    ui_validation = _read_json(POLICY_DIR / "ui_policy_validation.json")
    browser_validation = _read_json(POLICY_DIR / "browser_export_validation.json")
    hardware_manifest = _read_json(POLICY_DIR / "hardware_smoke_test_manifest.json")

    finite_total = scope["finite_exact_scope"]["total_files"]
    missing_combinations = scope["finite_exact_scope"]["missing_combinations"]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bench_smoke_test": {
            "status": "pending_external_hardware",
            "manifest_cases": len(hardware_manifest.get("cases", [])),
            "cases": hardware_manifest.get("cases", []),
            "note": "현재 코딩 환경에서는 실제 장비를 제어할 수 없어 manifest/checklist/results template까지만 준비된 상태입니다.",
        },
        "field_exact_promotion": {
            "status": field_status["status"],
            "remaining_gate": field_status["remaining_gate"],
            "export_validation": field_status["export_validation"],
            "ui_validation_pass": field_status["ui_validation"]["pass_status"],
        },
        "finite_exact_scope": {
            "exact_recipe_count": finite_total,
            "bench_confirmed_count": finite_data["measurement_status"]["bench_confirmed_exact_recipe_count"],
            "bench_confirmed_combinations": finite_data["measurement_status"]["new_exact_recipe_measured"],
            "missing_exact_combinations": missing_combinations,
        },
        "interpolation_confidence": interpolation,
        "modelside_interpolation": modelside,
        "finite_generalization_stage2": finite_stage2,
        "additional_data_recommendation": finite_data.get("next_data_priority", []),
        "export_validation": {
            "exact_export_cases": export_validation.get("cases", {}),
            "browser_download_validation_passed": browser_validation.get("download_content_validation_passed"),
            "ui_cases_passed": all(case.get("pass_status") for case in ui_validation.get("cases", [])),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Phase 1 Closeout + Phase 2 Parallel Status",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        "",
        "## 1. bench smoke test 결과",
        "",
        "- status: `pending_external_hardware`",
        f"- manifest_cases: `{len(hardware_manifest.get('cases', []))}`",
        "- note: `현재 코딩 환경에서는 실제 장비를 제어할 수 없어 manifest/checklist/results template까지만 준비된 상태입니다.`",
        "",
        "## 2. exact field 공식 지원 승격 여부",
        "",
        f"- status: `{field_status['status']}`",
        f"- remaining_gate: `{field_status['remaining_gate']}`",
        f"- export_validation_pass: `{field_status['export_validation']['allow_auto_download'] and field_status['export_validation']['support_state'] == 'exact'}`",
        f"- ui_validation_pass: `{field_status['ui_validation']['pass_status']}`",
        "",
        "## 3. finite exact 47 recipes 중 bench 확인 완료한 조합",
        "",
        f"- exact_recipe_count: `{finite_total}`",
        f"- bench_confirmed_count: `{finite_data['measurement_status']['bench_confirmed_exact_recipe_count']}`",
        f"- bench_confirmed_combinations: `{finite_data['measurement_status']['new_exact_recipe_measured']}`",
        f"- missing_exact_combinations: `{missing_combinations}`",
        "",
        "## 4. steady-state interpolation 개선 전/후 L1FO 비교",
        "",
    ]
    for row in modelside.get("rows", []):
        lines.extend(
            [
                f"### {row.get('label')}",
                f"- policy_version: `{row.get('policy_version')}`",
                f"- case_count: `{row.get('case_count')}`",
                f"- auto_count: `{row.get('auto_count')}`",
                f"- false_auto_count: `{row.get('false_auto_count')}`",
                f"- false_block_count: `{row.get('false_block_count')}`",
                f"- mean_shape_corr: `{row.get('mean_shape_corr'):.6f}`",
                f"- mean_nrmse: `{row.get('mean_nrmse'):.6f}`",
                "",
            ]
        )
    lines.extend(
        [
            f"- conclusion: `{modelside.get('conclusion')}`",
            "",
            "## 5. sine finite generalization 2차 첫 결과",
            "",
            f"- case_count: `{finite_stage2.get('case_count')}`",
            f"- preview_case_count: `{finite_stage2.get('preview_case_count')}`",
            f"- mean_shape_corr: `{finite_stage2.get('summary', {}).get('mean_shape_corr'):.6f}`",
            f"- mean_nrmse: `{finite_stage2.get('summary', {}).get('mean_nrmse'):.6f}`",
            f"- mean_phase_lag_s: `{finite_stage2.get('summary', {}).get('mean_phase_lag_s'):.6f}`",
            "- operational_decision: `preview-only 유지`",
            "",
            "## 6. 추가 데이터가 있으면 가장 빨라지는 조합 제안",
            "",
        ]
    )
    for index, recommendation in enumerate(finite_data.get("next_data_priority", []), start=1):
        proposal = recommendation.get("proposal", recommendation)
        reason = recommendation.get("reason") if isinstance(recommendation, dict) else None
        lines.append(f"- {index}. {proposal}")
        if reason:
            lines.append(f"  reason: `{reason}`")
    lines.extend(
        [
            "",
            "## Export / UI 검증 상태",
            "",
            f"- ui_cases_all_passed: `{all(case.get('pass_status') for case in ui_validation.get('cases', []))}`",
            f"- browser_download_validation_passed: `{browser_validation.get('download_content_validation_passed')}`",
            f"- exact_export_cases: `{list(export_validation.get('cases', {}).keys())}`",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
