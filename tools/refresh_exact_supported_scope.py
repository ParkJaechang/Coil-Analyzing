from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "artifacts" / "policy_eval" / "exact_scope_refresh.json"
SCRIPTS = [
    ROOT / "tools" / "report_exact_and_finite_scope.py",
    ROOT / "tools" / "generate_bz_first_artifacts.py",
    ROOT / "tools" / "report_operational_support_scope.py",
    ROOT / "tools" / "report_finite_exact_productization.py",
    ROOT / "tools" / "report_finite_data_recommendations.py",
]


def main() -> int:
    executions: list[dict[str, object]] = []
    for script in SCRIPTS:
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        executions.append(
            {
                "script": str(script),
                "returncode": int(completed.returncode),
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )
        if completed.returncode != 0:
            break

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "success": all(item["returncode"] == 0 for item in executions),
        "executions": executions,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
