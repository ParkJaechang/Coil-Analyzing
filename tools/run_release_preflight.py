from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "artifacts" / "policy_eval" / "release_preflight.json"
OUTPUT_MD = ROOT / "artifacts" / "policy_eval" / "release_preflight.md"
VENV_PYTHON = ROOT.parent / ".venv" / "Scripts" / "python.exe"
PYTHON_EXE = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)

BASE_STEPS = [
    {"name": "scope_refresh", "command": [str(PYTHON_EXE), str(ROOT / "tools" / "refresh_exact_supported_scope.py")], "required": True},
    {"name": "ui_exact_matrix", "command": [str(PYTHON_EXE), str(ROOT / "tools" / "validate_exact_matrix_ui_states.py")], "required": True},
    {"name": "ui_request_router", "command": [str(PYTHON_EXE), str(ROOT / "tools" / "validate_request_router_actions.py")], "required": True},
    {"name": "export_validation", "command": [str(PYTHON_EXE), str(ROOT / "tools" / "validate_exact_export_outputs.py")], "required": True},
    {"name": "closeout_report", "command": [str(PYTHON_EXE), str(ROOT / "tools" / "report_exact_matrix_closeout.py")], "required": True},
    {"name": "pytest_core", "command": [str(PYTHON_EXE), "-m", "pytest", "tests/test_recommendation_service.py", "tests/test_support_extraction.py"], "required": True},
]

OPTIONAL_BROWSER_STEP = {
    "name": "browser_export_optional",
    "command": [str(PYTHON_EXE), str(ROOT / "tools" / "validate_browser_export_ui.py")],
    "required": False,
}


def main() -> int:
    include_browser = "--include-browser" in sys.argv[1:]
    steps = list(BASE_STEPS)
    if include_browser:
        steps.append(OPTIONAL_BROWSER_STEP)
    executions: list[dict[str, object]] = []
    for step in steps:
        completed = subprocess.run(
            step["command"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        executions.append(
            {
                "name": step["name"],
                "required": bool(step["required"]),
                "returncode": int(completed.returncode),
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )

    required_failed = [item["name"] for item in executions if item["required"] and item["returncode"] != 0]
    optional_failed = [item["name"] for item in executions if not item["required"] and item["returncode"] != 0]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "python_exe": str(PYTHON_EXE),
        "include_browser": include_browser,
        "required_passed": not required_failed,
        "required_failed": required_failed,
        "optional_failed": optional_failed,
        "executions": executions,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Release Preflight",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- required_passed: `{payload['required_passed']}`",
        f"- required_failed: `{payload['required_failed']}`",
        f"- optional_failed: `{payload['optional_failed']}`",
        "",
        "| step | required | returncode |",
        "| --- | --- | --- |",
    ]
    for item in executions:
        lines.append(f"| {item['name']} | {item['required']} | {item['returncode']} |")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    return 0 if not required_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
