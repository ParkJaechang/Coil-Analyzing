from __future__ import annotations

from src.field_analysis.ui_run_readiness import build_run_readiness_summary


def test_run_readiness_summary_without_saved_dataset_root() -> None:
    summary = build_run_readiness_summary(
        dataset_root="",
        dataset_root_exists=False,
        manifest_exists=False,
    )

    assert summary["dataset_root_saved"] is False
    assert summary["dataset_root_exists"] is False
    assert summary["manifest_exists"] is False
    assert summary["manifest_counts"] == {
        "continuous": 0,
        "finite_cycle": 0,
        "unknown": 0,
    }


def test_run_readiness_summary_with_saved_root_and_missing_manifest() -> None:
    summary = build_run_readiness_summary(
        dataset_root="D:/Datasets",
        dataset_root_exists=True,
        manifest_exists=False,
    )

    assert summary["dataset_root_saved"] is True
    assert summary["dataset_root_exists"] is True
    assert summary["manifest_exists"] is False
    assert summary["selected_library_counts"]["total"] == 0


def test_run_readiness_summary_includes_manifest_and_selected_counts() -> None:
    summary = build_run_readiness_summary(
        dataset_root="D:/Datasets",
        dataset_root_exists=True,
        manifest_exists=True,
        manifest_counts={
            "continuous": 4,
            "finite_cycle": 2,
            "unknown": 1,
        },
        selected_library_counts={
            "continuous": 2,
            "finite_cycle": 1,
        },
    )

    assert summary["manifest_counts"] == {
        "continuous": 4,
        "finite_cycle": 2,
        "unknown": 1,
    }
    assert summary["selected_library_counts"] == {
        "continuous": 2,
        "finite_cycle": 1,
        "total": 3,
    }
