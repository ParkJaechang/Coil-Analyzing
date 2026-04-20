from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path.home() / "Desktop"
OUTPUT_JSON = ROOT / "artifacts" / "policy_eval" / "desktop_shortcuts.json"
GUIDE_FILE = ROOT / "사용안내_전자기장_LUT_보정_툴.txt"


def _create_shortcut(shortcut_path: Path, target: Path, description: str) -> None:
    powershell_script = f"""
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut('{str(shortcut_path)}')
$Shortcut.TargetPath = '{str(target)}'
$Shortcut.WorkingDirectory = '{str(ROOT)}'
$Shortcut.Description = '{description}'
$Shortcut.IconLocation = '{str(target)},0'
$Shortcut.Save()
""".strip()
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", powershell_script],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> int:
    DESKTOP.mkdir(parents=True, exist_ok=True)
    shortcuts = [
        {
            "path": DESKTOP / "전자기장 LUT 보정 툴.lnk",
            "target": ROOT / "launch_quick_lut.cmd",
            "description": "전자기장 LUT/보정 툴 일반 실행",
        },
        {
            "path": DESKTOP / "전자기장 LUT 보정 툴 (실사용 모드).lnk",
            "target": ROOT / "launch_quick_lut_operational.cmd",
            "description": "전자기장 LUT/보정 툴 실사용 모드 실행",
        },
        {
            "path": DESKTOP / "예제 1 연속 Bz exact.lnk",
            "target": ROOT / "launch_quick_lut_example_bz_exact.cmd",
            "description": "대표 예제 1: continuous Bz exact preset 실행",
        },
        {
            "path": DESKTOP / "예제 2 연속 current exact.lnk",
            "target": ROOT / "launch_quick_lut_example_current_exact.cmd",
            "description": "대표 예제 2: continuous current exact preset 실행",
        },
        {
            "path": DESKTOP / "예제 3 finite triangle exact.lnk",
            "target": ROOT / "launch_quick_lut_example_finite_triangle_exact.cmd",
            "description": "대표 예제 3: finite triangle exact preset 실행",
        },
    ]
    created: list[dict[str, object]] = []
    for item in shortcuts:
        _create_shortcut(item["path"], item["target"], item["description"])
        created.append(
            {
                "shortcut": str(item["path"]),
                "target": str(item["target"]),
                "exists": item["path"].exists(),
            }
        )

    desktop_guide = DESKTOP / GUIDE_FILE.name
    desktop_guide.write_text(GUIDE_FILE.read_text(encoding="utf-8"), encoding="utf-8")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "desktop": str(DESKTOP),
        "shortcuts": created,
        "guide_copy": str(desktop_guide),
        "guide_exists": desktop_guide.exists(),
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
