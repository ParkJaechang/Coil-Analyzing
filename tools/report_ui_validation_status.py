from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


APPTEST_JSON = Path("artifacts/policy_eval/ui_policy_validation.json")
BROWSER_JSON = Path("artifacts/policy_eval/browser_export_validation.json")
OUTPUT_MD = Path("artifacts/policy_eval/ui_export_validation_status.md")


def main() -> int:
    apptest = json.loads(APPTEST_JSON.read_text(encoding="utf-8"))
    browser = json.loads(BROWSER_JSON.read_text(encoding="utf-8"))

    case_by_name = {case["name"]: case for case in apptest.get("cases", [])}
    exact = case_by_name.get("exact_current_auto", {})
    preview = case_by_name.get("interpolated_current_preview", {})
    blocked = case_by_name.get("finite_missing_exact_recipe", {})

    exact_metrics = exact.get("metrics", {})
    preview_metrics = preview.get("metrics", {})
    blocked_metrics = blocked.get("metrics", {})

    lines: list[str] = []
    lines.append("# UI Export Validation Status")
    lines.append("")
    lines.append(f"- generated_at_utc: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("- goal: `operational path verification, not full browser automation completeness`")
    lines.append("")
    lines.append("## Current/Exact Path")
    lines.append("")
    lines.append(f"- status: `{'PASS' if exact_metrics.get('자동 추천') == '가능' else 'FAIL'}`")
    lines.append(f"- engine: `{exact_metrics.get('엔진', '')}`")
    lines.append(f"- support_state: `{exact_metrics.get('지원 상태', '')}`")
    lines.append(f"- auto_state: `{exact_metrics.get('자동 추천', '')}`")
    lines.append("- source: `AppTest`")
    lines.append("")
    lines.append("## Preview / Block Path")
    lines.append("")
    lines.append(
        f"- preview_case_status: `{'PASS' if preview_metrics.get('자동 추천') == 'preview-only' else 'FAIL'}`"
    )
    lines.append(f"- preview_support_state: `{preview_metrics.get('지원 상태', '')}`")
    lines.append(
        f"- finite_missing_exact_status: `{'PASS' if blocked_metrics.get('자동 추천') == 'preview-only' else 'FAIL'}`"
    )
    lines.append("- source: `AppTest`")
    lines.append("")
    lines.append("## Export Rendering")
    lines.append("")
    lines.append(
        f"- status: `{'PASS' if browser.get('found_completion_message') and browser.get('found_export_section') and browser.get('found_harmonic_transfer_download') else 'FAIL'}`"
    )
    lines.append(f"- export_buttons_found: `{len(browser.get('export_buttons', []))}`")
    for button in browser.get("export_buttons", []):
        lines.append(f"- export_button: `{button}`")
    lines.append("- source: `Selenium + headless Chrome`")
    lines.append(
        f"- download_content_validation: `{'PASS' if browser.get('download_content_validation_passed') else 'FAIL'}`"
    )
    for item in browser.get("downloaded_files", []):
        lines.append(
            f"- downloaded_file: `{item.get('name')}` size=`{item.get('size_bytes')}` header=`{item.get('header')}`"
        )
    lines.append("")
    lines.append("## Limits")
    lines.append("")
    lines.append("- current-target browser rerender was not stabilized in this headless route")
    lines.append("- current-target operational state is still covered by AppTest evidence")
    lines.append("- browser validation confirms export section/button rendering and basic downloaded file content")
    lines.append("")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"Wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
