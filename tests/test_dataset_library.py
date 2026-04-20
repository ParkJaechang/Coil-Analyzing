from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.field_analysis.dataset_library import (
    build_dataset_payloads,
    build_dataset_manifest,
    list_manifest_entries,
    load_dataset_library_settings,
    read_dataset_entry_bytes,
    save_dataset_library_settings,
)


def test_dataset_library_settings_round_trip(tmp_path: Path) -> None:
    settings_path = tmp_path / ".coil_analyzer" / "settings.json"
    assert load_dataset_library_settings(settings_path=settings_path) == {"dataset_root": ""}

    target_root = tmp_path / "dataset_root"
    saved = save_dataset_library_settings(
        {"dataset_root": str(target_root)},
        settings_path=settings_path,
    )

    assert saved == {"dataset_root": str(target_root)}
    assert load_dataset_library_settings(settings_path=settings_path) == {"dataset_root": str(target_root)}


def test_build_dataset_manifest_scans_csv_xlsx_and_txt(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_library"
    continuous_dir = dataset_root / "continuous"
    finite_dir = dataset_root / "finite_cycle"
    misc_dir = dataset_root / "misc"
    continuous_dir.mkdir(parents=True)
    finite_dir.mkdir(parents=True)
    misc_dir.mkdir(parents=True)

    csv_path = continuous_dir / "run_a.csv"
    csv_path.write_text("time,value\n0,1\n1,2\n", encoding="utf-8")

    xlsx_path = finite_dir / "run_b.xlsx"
    pd.DataFrame({"time": [0, 1], "value": [3, 4]}).to_excel(xlsx_path, index=False)

    unknown_path = misc_dir / "run_c.csv"
    unknown_path.write_text("time,value\n0,5\n1,6\n", encoding="utf-8")

    txt_path = misc_dir / "run_d.txt"
    txt_path.write_text("time\tvalue\n0\t7\n1\t8\n", encoding="utf-8")

    manifest = build_dataset_manifest(dataset_root)

    assert (dataset_root / "coil_analyzing_manifest.json").is_file()
    assert manifest["file_count"] == 4
    assert manifest["counts"] == {
        "continuous": 1,
        "finite_cycle": 1,
        "unknown": 2,
    }

    entries_by_name = {entry["name"]: entry for entry in manifest["files"]}
    assert entries_by_name["run_a.csv"]["dataset_mode"] == "continuous"
    assert entries_by_name["run_b.xlsx"]["dataset_mode"] == "finite_cycle"
    assert entries_by_name["run_c.csv"]["dataset_mode"] == "unknown"
    assert entries_by_name["run_d.txt"]["dataset_mode"] == "unknown"
    assert entries_by_name["run_a.csv"]["path"] == "continuous/run_a.csv"
    assert len(entries_by_name["run_b.xlsx"]["content_hash"]) == 64
    assert all(entry["content_hash"] for entry in manifest["files"])


def test_list_manifest_entries_filters_by_dataset_mode(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_library"
    (dataset_root / "continuous").mkdir(parents=True)
    (dataset_root / "finite_cycle").mkdir(parents=True)
    (dataset_root / "misc").mkdir(parents=True)

    (dataset_root / "continuous" / "run_a.csv").write_text("time,value\n0,1\n", encoding="utf-8")
    (dataset_root / "finite_cycle" / "run_b.txt").write_text("time\tvalue\n0\t2\n", encoding="utf-8")
    (dataset_root / "misc" / "run_c.csv").write_text("time,value\n0,3\n", encoding="utf-8")
    build_dataset_manifest(dataset_root)

    continuous_entries = list_manifest_entries(dataset_root, dataset_mode="continuous")
    finite_entries = list_manifest_entries(dataset_root, dataset_mode="finite_cycle")
    unknown_entries = list_manifest_entries(dataset_root, dataset_mode="unknown")

    assert [entry["path"] for entry in continuous_entries] == ["continuous/run_a.csv"]
    assert [entry["path"] for entry in finite_entries] == ["finite_cycle/run_b.txt"]
    assert [entry["path"] for entry in unknown_entries] == ["misc/run_c.csv"]


def test_read_dataset_entry_bytes_and_build_payloads(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_library"
    (dataset_root / "continuous").mkdir(parents=True)
    run_path = dataset_root / "continuous" / "run_a.csv"
    run_path.write_bytes(b"time,value\n0,1\n1,2\n")
    build_dataset_manifest(dataset_root)

    raw_bytes = read_dataset_entry_bytes(dataset_root, "continuous/run_a.csv")
    payloads = build_dataset_payloads(
        dataset_root,
        ["continuous/run_a.csv"],
    )

    assert raw_bytes == b"time,value\n0,1\n1,2\n"
    assert payloads == [("continuous/run_a.csv", b"time,value\n0,1\n1,2\n")]


def test_dataset_library_rejects_path_traversal(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_library"
    dataset_root.mkdir(parents=True)
    outside_path = tmp_path / "outside.csv"
    outside_path.write_text("time,value\n0,9\n", encoding="utf-8")
    build_dataset_manifest(dataset_root)

    with pytest.raises(ValueError):
        read_dataset_entry_bytes(dataset_root, "../outside.csv")

    with pytest.raises(ValueError):
        build_dataset_payloads(dataset_root, ["../outside.csv"])
