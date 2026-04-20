from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELD_ANALYSIS_SRC = ROOT.parent / "src"
ARTIFACT_DIR = ROOT / "artifacts" / "bz_first_exact_matrix"

LINE_COUNT_JSON = ARTIFACT_DIR / "line_count_before_after.json"
SPLIT_REPORT_JSON = ARTIFACT_DIR / "refactor_split_report.json"
SPLIT_REPORT_MD = ARTIFACT_DIR / "refactor_split_report.md"
COMPAT_JSON = ARTIFACT_DIR / "minimal_compatibility_regression.json"
COMPAT_MD = ARTIFACT_DIR / "minimal_compatibility_regression.md"

BASELINE_LINE_COUNTS = {
    "recommendation_service.py": 684,
    "validation_retune.py": 2445,
    "compensation.py": 3274,
}

SPLIT_HELPERS = {
    "recommendation_service.py": [
        "recommendation_service_runtime.py",
        "recommendation_service_finalize.py",
    ],
    "validation_retune.py": [
        "validation_retune_shared.py",
        "validation_retune_provenance.py",
        "validation_retune_alignment.py",
        "validation_retune_runtime.py",
        "validation_retune_comparison.py",
        "validation_retune_corrected_export.py",
        "validation_retune_acceptance.py",
        "validation_retune_metric_utils.py",
    ],
    "compensation.py": [],
}


def _module_path(name: str) -> Path:
    return FIELD_ANALYSIS_SRC / "field_analysis" / name


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def build_line_count_payload() -> dict[str, object]:
    targets: list[dict[str, object]] = []
    for name, before in BASELINE_LINE_COUNTS.items():
        current_path = _module_path(name)
        helpers = SPLIT_HELPERS[name]
        helper_counts = {helper: _line_count(_module_path(helper)) for helper in helpers if _module_path(helper).exists()}
        targets.append(
            {
                "file": name,
                "before_lines": int(before),
                "after_lines": _line_count(current_path),
                "delta_lines": _line_count(current_path) - int(before),
                "target_met": _line_count(current_path) <= 500,
                "helper_files": helper_counts,
            }
        )
    return {
        "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "targets": targets,
    }


def build_split_report_payload(line_counts: dict[str, object]) -> dict[str, object]:
    rows = []
    for entry in line_counts["targets"]:
        file_name = str(entry["file"])
        after_lines = int(entry["after_lines"])
        helper_files = entry["helper_files"]
        status = "split_active" if after_lines <= 500 and helper_files else ("fallback_or_unsplit" if helper_files else "unsplit")
        rows.append(
            {
                "file": file_name,
                "status": status,
                "after_lines": after_lines,
                "target_met": bool(entry["target_met"]),
                "helper_count": len(helper_files),
                "helper_files": helper_files,
            }
        )
    return {
        "generated_at_utc": line_counts["generated_at_utc"],
        "overall_status": "partial",
        "rows": rows,
    }


def render_split_report_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Refactor Split Report",
        "",
        f"- overall_status: `{payload['overall_status']}`",
        "",
    ]
    for row in payload["rows"]:
        lines.extend(
            [
                f"## {row['file']}",
                "",
                f"- status: `{row['status']}`",
                f"- after_lines: `{row['after_lines']}`",
                f"- target_met: `{row['target_met']}`",
                f"- helper_count: `{row['helper_count']}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_compatibility_payload() -> dict[str, object]:
    if str(FIELD_ANALYSIS_SRC) not in sys.path:
        sys.path.insert(0, str(FIELD_ANALYSIS_SRC))
    import_checks = []
    for module_name in (
        "field_analysis.recommendation_service",
        "field_analysis.validation_retune",
        "field_analysis.compensation",
    ):
        try:
            importlib.import_module(module_name)
            import_checks.append({"module": module_name, "status": "pass"})
        except Exception as exc:  # pragma: no cover
            import_checks.append({"module": module_name, "status": "fail", "error": repr(exc)})
    pytest_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(ROOT / "tests" / "test_recommendation_service.py"),
            str(ROOT / "tests" / "test_validation_retune.py"),
            "-q",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    return {
        "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "import_checks": import_checks,
        "pytest": {
            "returncode": int(pytest_run.returncode),
            "stdout": pytest_run.stdout.strip(),
            "stderr": pytest_run.stderr.strip(),
        },
    }


def render_compatibility_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Minimal Compatibility Regression",
        "",
        *(f"- {entry['module']}: `{entry['status']}`" for entry in payload["import_checks"]),
        "",
        f"- pytest_returncode: `{payload['pytest']['returncode']}`",
        f"- pytest_stdout: `{payload['pytest']['stdout']}`",
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    line_counts = build_line_count_payload()
    split_report = build_split_report_payload(line_counts)
    compatibility = build_compatibility_payload()

    LINE_COUNT_JSON.write_text(json.dumps(line_counts, ensure_ascii=False, indent=2), encoding="utf-8")
    SPLIT_REPORT_JSON.write_text(json.dumps(split_report, ensure_ascii=False, indent=2), encoding="utf-8")
    SPLIT_REPORT_MD.write_text(render_split_report_markdown(split_report), encoding="utf-8")
    COMPAT_JSON.write_text(json.dumps(compatibility, ensure_ascii=False, indent=2), encoding="utf-8")
    COMPAT_MD.write_text(render_compatibility_markdown(compatibility), encoding="utf-8")

    print(f"Wrote {LINE_COUNT_JSON}")
    print(f"Wrote {SPLIT_REPORT_JSON}")
    print(f"Wrote {SPLIT_REPORT_MD}")
    print(f"Wrote {COMPAT_JSON}")
    print(f"Wrote {COMPAT_MD}")


if __name__ == "__main__":
    main()
