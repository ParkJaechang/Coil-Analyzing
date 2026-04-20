from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


POLICY_DIR = Path("artifacts/policy_eval")
MANIFEST_PATH = POLICY_DIR / "hardware_smoke_test_manifest.json"
OUTPUT_JSON = POLICY_DIR / "hardware_smoke_test_results_template.json"
OUTPUT_MD = POLICY_DIR / "hardware_smoke_test_results_template.md"


def _case_level(case: dict) -> object:
    return case.get("level_or_field", case.get("level"))


def _case_cycles(case: dict) -> object:
    return case.get("cycles", case.get("cycle_count"))


def _case_waveform(case: dict) -> object:
    return case.get("waveform", case.get("waveform_type"))


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    generated_at = datetime.now(timezone.utc).isoformat()

    cases = []
    for case in manifest.get("cases", []):
        regime = case.get("regime")
        target_type = case.get("target_type")
        cases.append(
            {
                "case_id": case["case_id"],
                "category": f"{regime}_{target_type}",
                "description": case.get("expected_mode", "bench smoke test"),
                "regime": regime,
                "target_type": target_type,
                "waveform_type": _case_waveform(case),
                "freq_hz": case.get("freq_hz"),
                "level": _case_level(case),
                "cycle_count": _case_cycles(case),
                "lut_file": case.get("lut_file"),
                "formula_file": case.get("formula_file"),
                "waveform_file": case.get("waveform_file"),
                "shape_corr": None,
                "nrmse": None,
                "phase_lag_s": None,
                "clipping_or_saturation": None,
                "operator_judgement": None,
                "pass_fail": None,
                "notes": "",
            }
        )

    payload = {
        "generated_at_utc": generated_at,
        "source_manifest": str(MANIFEST_PATH),
        "instructions": [
            "실제 장비에서 manifest에 지정된 LUT/formula/waveform 파일을 사용해 각 케이스를 실행합니다.",
            "shape_corr, nrmse, phase_lag_s는 실측 파형과 목표 파형 비교 기준으로 기록합니다.",
            "clipping_or_saturation은 true/false로 기록하고, 관찰된 경우 notes에 상세를 남깁니다.",
            "operator_judgement는 '목표 개형 재현 가능' 또는 '재현 불가'로 기록합니다.",
            "pass_fail은 최종 승인 결과를 PASS 또는 FAIL로 기록합니다.",
        ],
        "cases": cases,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Hardware Smoke Test Results Template",
        "",
        f"- generated_at_utc: `{generated_at}`",
        f"- source_manifest: `{MANIFEST_PATH}`",
        "",
        "## Recording Rules",
        "",
        "- shape_corr / nrmse / phase_lag_s는 실측 파형과 목표 파형 비교 기준으로 기록합니다.",
        "- clipping_or_saturation은 `true` 또는 `false`로 기록합니다.",
        "- operator_judgement는 `목표 개형 재현 가능` 또는 `재현 불가`로 기록합니다.",
        "- pass_fail은 `PASS` 또는 `FAIL`로 기록합니다.",
        "",
        "## Cases",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"### {case['case_id']}",
                f"- category: `{case['category']}`",
                f"- description: `{case['description']}`",
                f"- regime: `{case['regime']}`",
                f"- target_type: `{case['target_type']}`",
                f"- waveform_type: `{case['waveform_type']}`",
                f"- freq_hz: `{case['freq_hz']}`",
                f"- level: `{case['level']}`",
                f"- cycle_count: `{case['cycle_count']}`",
                f"- lut_file: `{case['lut_file']}`",
                f"- formula_file: `{case['formula_file']}`",
                f"- waveform_file: `{case['waveform_file']}`",
                "- shape_corr: ``",
                "- nrmse: ``",
                "- phase_lag_s: ``",
                "- clipping_or_saturation: ``",
                "- operator_judgement: ``",
                "- pass_fail: ``",
                "- notes: ``",
                "",
            ]
        )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
