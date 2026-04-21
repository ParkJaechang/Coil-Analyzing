from __future__ import annotations

from pathlib import Path

from src.field_analysis.dataset_access_preflight import build_dataset_access_preflight
from src.field_analysis.dataset_library import build_dataset_manifest


def test_dataset_access_preflight_handles_missing_dataset_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "missing_root"

    summary = build_dataset_access_preflight(
        dataset_root=dataset_root,
        selected_paths_by_mode={"continuous": ["continuous/run_a.csv"]},
    )

    assert summary["dataset_root_exists"] is False
    assert summary["selected"]["ok_count"] == 0
    assert summary["selected"]["missing_count"] == 1
    assert summary["selected"]["unavailable_count"] == 1


def test_dataset_access_preflight_reports_missing_selected_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_library"
    continuous_dir = dataset_root / "continuous"
    continuous_dir.mkdir(parents=True)
    existing_path = continuous_dir / "run_a.csv"
    existing_path.write_text("time,value\n0,1\n", encoding="utf-8")
    build_dataset_manifest(dataset_root)
    existing_path.unlink()

    summary = build_dataset_access_preflight(
        dataset_root=dataset_root,
        selected_paths_by_mode={"continuous": ["continuous/run_a.csv"]},
    )

    assert summary["dataset_root_exists"] is True
    assert summary["selected"]["missing_count"] == 1
    assert summary["manifest"]["missing_count"] == 1
    assert summary["selected"]["problem_samples"][0]["path"] == "continuous/run_a.csv"


def test_dataset_access_preflight_reports_all_ok_when_files_exist(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_library"
    (dataset_root / "continuous").mkdir(parents=True)
    (dataset_root / "finite_cycle").mkdir(parents=True)
    (dataset_root / "continuous" / "run_a.csv").write_text("time,value\n0,1\n", encoding="utf-8")
    (dataset_root / "finite_cycle" / "run_b.txt").write_text("time\tvalue\n0\t2\n", encoding="utf-8")
    build_dataset_manifest(dataset_root)

    summary = build_dataset_access_preflight(
        dataset_root=dataset_root,
        selected_paths_by_mode={
            "continuous": ["continuous/run_a.csv"],
            "finite_cycle": ["finite_cycle/run_b.txt"],
        },
    )

    assert summary["selected"]["ok_count"] == 2
    assert summary["selected"]["unavailable_count"] == 0
    assert summary["manifest"]["ok_count"] == 2
    assert summary["manifest"]["problem_samples"] == []


def test_dataset_access_preflight_summarizes_mixed_selected_and_manifest_state(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_library"
    (dataset_root / "continuous").mkdir(parents=True)
    (dataset_root / "finite_cycle").mkdir(parents=True)
    (dataset_root / "continuous" / "run_ok.csv").write_text("time,value\n0,1\n", encoding="utf-8")
    missing_after_manifest = dataset_root / "finite_cycle" / "run_missing.txt"
    missing_after_manifest.write_text("time\tvalue\n0\t2\n", encoding="utf-8")
    build_dataset_manifest(dataset_root)
    missing_after_manifest.unlink()

    summary = build_dataset_access_preflight(
        dataset_root=dataset_root,
        selected_paths_by_mode={
            "continuous": ["continuous/run_ok.csv"],
            "finite_cycle": ["finite_cycle/run_missing.txt"],
        },
    )

    assert summary["selected_by_mode"]["continuous"]["ok_count"] == 1
    assert summary["selected_by_mode"]["finite_cycle"]["missing_count"] == 1
    assert summary["selected"]["ok_count"] == 1
    assert summary["selected"]["missing_count"] == 1
    assert summary["manifest"]["ok_count"] == 1
    assert summary["manifest"]["missing_count"] == 1
