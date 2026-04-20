from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "policy_eval"
SCOPE_JSON = ARTIFACT_DIR / "operational_support_scope.json"
CLOSEOUT_JSON = ARTIFACT_DIR / "exact_matrix_closeout.json"
PREFLIGHT_JSON = ARTIFACT_DIR / "release_preflight.json"
OUTPUT_JSON = ARTIFACT_DIR / "release_package_summary.json"
OUTPUT_MD = ARTIFACT_DIR / "release_package_summary.md"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    scope = _load_json(SCOPE_JSON)
    closeout = _load_json(CLOSEOUT_JSON)
    preflight = _load_json(PREFLIGHT_JSON)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_files": {
            "app": str(ROOT / "app_field_analysis_quick.py"),
            "launcher_general": str(ROOT / "launch_quick_lut.cmd"),
            "launcher_operational": str(ROOT / "launch_quick_lut_operational.cmd"),
            "guide": str(ROOT / "사용안내_전자기장_LUT_보정_툴.txt"),
            "readme": str(ROOT / "README.md"),
        },
        "support_scope": scope.get("user_facing", {}),
        "closeout_summary": {
            "continuous": closeout.get("continuous_support", {}),
            "finite": closeout.get("finite_support", {}),
        },
        "required_preflight_passed": preflight.get("required_passed"),
        "required_preflight_failed": preflight.get("required_failed", []),
        "known_external_work": [
            "bench sign-off",
            "remaining exact measured file: sine / 1.0 Hz / 1.0 cycle / 20 pp",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Release Package Summary",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        "",
        "## 실행 파일",
        "",
        f"- app: `{payload['execution_files']['app']}`",
        f"- launcher_general: `{payload['execution_files']['launcher_general']}`",
        f"- launcher_operational: `{payload['execution_files']['launcher_operational']}`",
        f"- guide: `{payload['execution_files']['guide']}`",
        f"- readme: `{payload['execution_files']['readme']}`",
        "",
        "## 운영 범위",
        "",
    ]
    for key, values in payload["support_scope"].items():
        lines.append(f"- {key}: `{values}`")
    lines.extend(
        [
            "",
            "## Preflight",
            "",
            f"- required_preflight_passed: `{payload['required_preflight_passed']}`",
            f"- required_preflight_failed: `{payload['required_preflight_failed']}`",
            "",
            "## 외부 작업",
            "",
            *[f"- `{item}`" for item in payload["known_external_work"]],
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
