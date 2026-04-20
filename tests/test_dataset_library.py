from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.field_analysis.dataset_library import (
    build_dataset_manifest,
    load_dataset_library_settings,
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
