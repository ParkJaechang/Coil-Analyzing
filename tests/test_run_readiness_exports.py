from __future__ import annotations

from src.field_analysis.ui_run_readiness_exports import (
    build_json_bytes,
    build_problem_csv_bytes,
    build_problem_rows,
    build_run_readiness_report_payload,
)


def test_build_run_readiness_report_payload_keeps_summary_and_access_sections() -> None:
    payload = build_run_readiness_report_payload(
        summary={
            "dataset_root": "D:/Datasets",
            "dataset_root_saved": True,
            "manifest_exists": True,
        },
        access_preflight={
            "selected": {"ok_count": 2, "missing_count": 1},
            "manifest": {"ok_count": 4, "missing_count": 0},
        },
        generated_at="2026-04-21T03:40:00+00:00",
    )

    assert payload["generated_at"] == "2026-04-21T03:40:00+00:00"
    assert payload["readiness_summary"]["dataset_root"] == "D:/Datasets"
    assert payload["access_preflight"]["selected"]["missing_count"] == 1
    assert b'"manifest_exists": true' in build_json_bytes(payload["readiness_summary"])


def test_build_problem_rows_and_csv_bytes_filter_out_ok_checks() -> None:
    rows = build_problem_rows(
        {
            "selected_continuous": {
                "checks": [
                    {"path": "continuous/run_ok.csv", "status": "ok", "message": "ok"},
                    {"path": "continuous/run_missing.csv", "status": "missing", "message": "missing"},
                ]
            },
            "manifest": {
                "checks": [
                    {"path": "manifest/run_blocked.csv", "status": "blocked", "message": "blocked"},
                ]
            },
        }
    )
    csv_bytes = build_problem_csv_bytes(rows)
    csv_text = csv_bytes.decode("utf-8")

    assert rows == [
        {
            "source": "selected_continuous",
            "path": "continuous/run_missing.csv",
            "status": "missing",
            "message": "missing",
        },
        {
            "source": "manifest",
            "path": "manifest/run_blocked.csv",
            "status": "blocked",
            "message": "blocked",
        },
    ]
    assert "continuous/run_ok.csv" not in csv_text
    assert "continuous/run_missing.csv" in csv_text
    assert "manifest/run_blocked.csv" in csv_text
