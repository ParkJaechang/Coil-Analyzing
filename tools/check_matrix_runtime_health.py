from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from report_exact_and_finite_scope import run_provisional_promotion_smoke


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "bz_first_exact_matrix"
OUTPUT_JSON = OUTPUT_DIR / "matrix_runtime_healthcheck.json"
OUTPUT_MD = OUTPUT_DIR / "matrix_runtime_healthcheck.md"
LABEL_OUTPUT_JSON = OUTPUT_DIR / "runtime_display_label_healthcheck.json"
LABEL_OUTPUT_MD = OUTPUT_DIR / "runtime_display_label_healthcheck.md"
REFRESH_SCRIPT = ROOT / "tools" / "refresh_exact_supported_scope.py"
GENERATE_SCRIPT = ROOT / "tools" / "generate_bz_first_artifacts.py"
RUNTIME_ROUTE_TEST_PATH = ROOT / "tests" / "test_recommendation_service.py"
RUNTIME_ROUTE_TEST_FILTER = (
    "artifact_scope_lock or marks_above_5hz_current_runtime_as_reference_only_preview"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_script(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return {
        "script": str(path),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def run_pytest(path: Path, *, expression: str | None = None) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", str(path), "-q"]
    if expression:
        command.extend(["-k", expression])
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return {
        "script": "pytest",
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def append_check(checks: list[dict[str, Any]], *, name: str, ok: bool, details: str) -> None:
    checks.append({"name": name, "status": "pass" if ok else "fail", "details": details})


def summarize_command_output(execution: dict[str, Any]) -> str:
    stdout_lines = [line.strip() for line in str(execution.get("stdout") or "").splitlines() if line.strip()]
    stderr_lines = [line.strip() for line in str(execution.get("stderr") or "").splitlines() if line.strip()]
    if execution.get("returncode") == 0 and stdout_lines:
        return stdout_lines[-1]
    if stderr_lines:
        return stderr_lines[-1]
    if stdout_lines:
        return stdout_lines[-1]
    return "no output"


def validate_catalog_runtime_state(lut_catalog: dict[str, Any]) -> tuple[bool, str]:
    entries = list(lut_catalog.get("entries", []))
    if not entries:
        return False, "entries=0"

    duplicate_count = sum(1 for entry in entries if bool(entry.get("duplicate_runtime")))
    stale_count = sum(1 for entry in entries if bool(entry.get("stale_runtime")))
    deprecated_count = sum(1 for entry in entries if str(entry.get("status")) == "deprecated")
    summary = lut_catalog.get("summary", {})
    if duplicate_count != int(summary.get("duplicate_runtime_entries") or 0):
        return False, (
            f"duplicate_runtime_entries summary={summary.get('duplicate_runtime_entries')} actual={duplicate_count}"
        )
    if stale_count != int(summary.get("stale_runtime_entries") or 0):
        return False, f"stale_runtime_entries summary={summary.get('stale_runtime_entries')} actual={stale_count}"
    if deprecated_count != int(summary.get("deprecated_entries") or 0):
        return False, f"deprecated_entries summary={summary.get('deprecated_entries')} actual={deprecated_count}"

    runtime_groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        identity = json.dumps(entry.get("runtime_identity"), ensure_ascii=False, sort_keys=True)
        runtime_groups.setdefault(identity, []).append(entry)

    for identity, group in runtime_groups.items():
        group_size = len(group)
        preferred_ids = {entry.get("runtime_preferred_lut_id") for entry in group if entry.get("runtime_preferred_lut_id")}
        if len(preferred_ids) != 1:
            return False, f"runtime_identity={identity} preferred_ids={sorted(preferred_ids)}"
        preferred_id = next(iter(preferred_ids))
        if not any(entry.get("lut_id") == preferred_id for entry in group):
            return False, f"runtime_identity={identity} preferred_missing={preferred_id}"
        if any(int(entry.get("runtime_group_size") or 0) != group_size for entry in group):
            return False, f"runtime_identity={identity} runtime_group_size_mismatch={group_size}"
        expected_duplicate = group_size > 1
        if any(bool(entry.get("duplicate_runtime")) != expected_duplicate for entry in group):
            return False, f"runtime_identity={identity} duplicate_runtime_mismatch={expected_duplicate}"
        expected_stale_ids = sorted(
            entry.get("lut_id")
            for entry in group
            if group_size > 1 and entry.get("lut_id") != preferred_id
        )
        actual_stale_ids = sorted(entry.get("lut_id") for entry in group if bool(entry.get("stale_runtime")))
        if actual_stale_ids != expected_stale_ids:
            return False, (
                f"runtime_identity={identity} stale_ids={actual_stale_ids} expected={expected_stale_ids}"
            )

    return True, f"groups={len(runtime_groups)}, duplicate_runtime_entries={duplicate_count}, stale_runtime_entries={stale_count}"


def build_runtime_display_label_healthcheck(
    label_report: dict[str, Any],
    *,
    output_json: Path = LABEL_OUTPUT_JSON,
    output_md: Path = LABEL_OUTPUT_MD,
) -> dict[str, Any]:
    summary = label_report.get("summary", {}) if isinstance(label_report, dict) else {}
    artifact_counts = summary.get("artifact_record_counts", {}) if isinstance(summary, dict) else {}
    checks = [
        {
            "name": "label_report_present",
            "status": "pass" if bool(label_report) else "fail",
            "details": "label_sanitization_report.json loaded" if bool(label_report) else "label_sanitization_report.json missing",
        },
        {
            "name": "display_leak_check",
            "status": "pass" if int(summary.get("leak_violations") or 0) == 0 else "fail",
            "details": f"leak_violations={int(summary.get('leak_violations') or 0)}",
        },
        {
            "name": "display_name_consistency_check",
            "status": "pass" if int(summary.get("display_name_mismatches") or 0) == 0 else "fail",
            "details": f"display_name_mismatches={int(summary.get('display_name_mismatches') or 0)}",
        },
        {
            "name": "display_label_consistency_check",
            "status": "pass" if int(summary.get("display_label_mismatches") or 0) == 0 else "fail",
            "details": f"display_label_mismatches={int(summary.get('display_label_mismatches') or 0)}",
        },
        {
            "name": "label_artifact_coverage_check",
            "status": (
                "pass"
                if all(
                    int(artifact_counts.get(name) or 0) > 0
                    for name in (
                        "lut_catalog",
                        "retune_picker_catalog",
                        "measurement_roi_priority",
                        "exact_matrix.finite.summary",
                    )
                )
                else "fail"
            ),
            "details": json.dumps(artifact_counts, ensure_ascii=False, sort_keys=True),
        },
    ]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "success": all(item["status"] == "pass" for item in checks),
        "summary": summary,
        "checks": checks,
        "violations": list(label_report.get("violations", []))[:40] if isinstance(label_report, dict) else [],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Runtime Display Label Healthcheck",
        "",
        f"- success: `{payload['success']}`",
        f"- total_records: `{int(summary.get('total_records') or 0)}`",
        f"- leak_violations: `{int(summary.get('leak_violations') or 0)}`",
        f"- display_name_mismatches: `{int(summary.get('display_name_mismatches') or 0)}`",
        f"- display_label_mismatches: `{int(summary.get('display_label_mismatches') or 0)}`",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    for item in checks:
        lines.append(f"| {item['name']} | {item['status']} | {item['details']} |")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    executions = [run_script(REFRESH_SCRIPT), run_script(GENERATE_SCRIPT)]
    checks: list[dict[str, Any]] = []

    append_check(
        checks,
        name="refresh_exact_supported_scope",
        ok=executions[0]["returncode"] == 0,
        details="refresh_exact_supported_scope.py completed successfully." if executions[0]["returncode"] == 0 else executions[0]["stderr"] or "refresh failed",
    )
    append_check(
        checks,
        name="generate_bz_first_artifacts",
        ok=executions[1]["returncode"] == 0,
        details="generate_bz_first_artifacts.py completed successfully." if executions[1]["returncode"] == 0 else executions[1]["stderr"] or "artifact generation failed",
    )

    exact_matrix_path = OUTPUT_DIR / "exact_matrix_final.json"
    lut_catalog_path = OUTPUT_DIR / "lut_catalog.json"
    roi_path = OUTPUT_DIR / "measurement_roi_priority.json"
    regression_path = OUTPUT_DIR / "data_intake_regression.json"
    label_report_path = OUTPUT_DIR / "label_sanitization_report.json"
    release_summary_path = OUTPUT_DIR / "release_candidate_summary.md"

    exact_matrix = load_json(exact_matrix_path) if exact_matrix_path.exists() else {}
    lut_catalog = load_json(lut_catalog_path) if lut_catalog_path.exists() else {}
    roi_payload = load_json(roi_path) if roi_path.exists() else {}
    regression_payload = load_json(regression_path) if regression_path.exists() else {}
    label_report = load_json(label_report_path) if label_report_path.exists() else {}

    counts = exact_matrix.get("counts", {})
    finite_exact_cells = int(counts.get("finite_exact_cells") or 0)
    provisional_cells = int(counts.get("provisional_cells") or 0)
    missing_exact_cells = int(counts.get("missing_exact_cells") or 0)
    promotion_status = exact_matrix.get("finite_exact_matrix", {}).get("promotion_status", {})
    promotion_state = str(promotion_status.get("state") or "unknown")

    valid_matrix_count = finite_exact_cells in {95, 96}
    append_check(
        checks,
        name="exact_matrix_count_check",
        ok=valid_matrix_count,
        details=f"finite_exact_cells={finite_exact_cells}",
    )

    consistent_provisional_state = (
        (finite_exact_cells == 95 and provisional_cells == 1 and missing_exact_cells == 1 and promotion_state == "provisional_only")
        or (finite_exact_cells == 96 and provisional_cells == 0 and missing_exact_cells == 0 and promotion_state == "promoted_to_exact")
    )
    append_check(
        checks,
        name="provisional_count_check",
        ok=consistent_provisional_state,
        details=(
            f"finite_exact_cells={finite_exact_cells}, provisional_cells={provisional_cells}, "
            f"missing_exact_cells={missing_exact_cells}, promotion_state={promotion_state}"
        ),
    )

    catalog_total = int(lut_catalog.get("summary", {}).get("total") or 0)
    catalog_entries = lut_catalog.get("entries", [])
    catalog_status_total = sum(int(value) for value in (lut_catalog.get("summary", {}).get("by_status") or {}).values())
    append_check(
        checks,
        name="lut_catalog_count_check",
        ok=catalog_total > 0 and catalog_total == len(catalog_entries) and catalog_total == catalog_status_total,
        details=f"total={catalog_total}, entries={len(catalog_entries)}, by_status_total={catalog_status_total}",
    )
    catalog_runtime_ok, catalog_runtime_details = validate_catalog_runtime_state(lut_catalog)
    append_check(
        checks,
        name="catalog_runtime_state_check",
        ok=catalog_runtime_ok,
        details=catalog_runtime_details,
    )

    regression_checks = regression_payload.get("checks", [])
    passing_regression = all(str(item.get("status")) == "pass" for item in regression_checks)
    append_check(
        checks,
        name="data_intake_regression",
        ok=passing_regression and len(regression_checks) > 0,
        details=f"passing_checks={sum(1 for item in regression_checks if item.get('status') == 'pass')}/{len(regression_checks)}",
    )

    priorities = roi_payload.get("priorities", [])
    head_category = str(priorities[0].get("category")) if priorities else "none"
    roi_ok = False
    if missing_exact_cells > 0:
        roi_ok = head_category == "missing_exact_promotion"
    else:
        roi_ok = head_category != "missing_exact_promotion"
    validation_attention = int(roi_payload.get("summary", {}).get("validation_attention_items") or 0)
    validation_present = any(str(item.get("category")) == "validation_priority" for item in priorities)
    if validation_attention > 0:
        roi_ok = roi_ok and validation_present
    append_check(
        checks,
        name="roi_priority_check",
        ok=roi_ok,
        details=f"head_category={head_category}, validation_attention_items={validation_attention}, validation_present={validation_present}",
    )

    smoke = run_provisional_promotion_smoke()
    append_check(
        checks,
        name="promotion_smoke_check",
        ok=bool(smoke.get("pass")),
        details=str(smoke.get("details") or ""),
    )

    runtime_route_execution = run_pytest(
        RUNTIME_ROUTE_TEST_PATH,
        expression=RUNTIME_ROUTE_TEST_FILTER,
    )
    executions.append(runtime_route_execution)
    append_check(
        checks,
        name="runtime_route_consistency_check",
        ok=runtime_route_execution["returncode"] == 0,
        details=summarize_command_output(runtime_route_execution),
    )

    label_healthcheck = build_runtime_display_label_healthcheck(label_report)
    append_check(
        checks,
        name="runtime_display_label_healthcheck",
        ok=bool(label_healthcheck.get("success")),
        details=(
            f"leak_violations={int(label_healthcheck.get('summary', {}).get('leak_violations') or 0)}, "
            f"display_name_mismatches={int(label_healthcheck.get('summary', {}).get('display_name_mismatches') or 0)}, "
            f"display_label_mismatches={int(label_healthcheck.get('summary', {}).get('display_label_mismatches') or 0)}"
        ),
    )

    append_check(
        checks,
        name="closeout_summary_check",
        ok=release_summary_path.exists(),
        details=f"release_candidate_summary_exists={release_summary_path.exists()}",
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "success": all(item["status"] == "pass" for item in checks),
        "executions": executions,
        "checks": checks,
        "snapshot": {
            "finite_exact_cells": finite_exact_cells,
            "provisional_cells": provisional_cells,
            "missing_exact_cells": missing_exact_cells,
            "promotion_state": promotion_state,
            "lut_catalog_total": catalog_total,
            "roi_head_category": head_category,
        },
        "promotion_smoke": smoke,
        "runtime_display_label_healthcheck": label_healthcheck,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Matrix Runtime Healthcheck",
        "",
        f"- success: `{payload['success']}`",
        f"- finite_exact_cells: `{finite_exact_cells}`",
        f"- provisional_cells: `{provisional_cells}`",
        f"- missing_exact_cells: `{missing_exact_cells}`",
        f"- roi_head_category: `{head_category}`",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    for item in checks:
        lines.append(f"| {item['name']} | {item['status']} | {item['details']} |")
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {LABEL_OUTPUT_JSON}")
    print(f"Wrote {LABEL_OUTPUT_MD}")
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
