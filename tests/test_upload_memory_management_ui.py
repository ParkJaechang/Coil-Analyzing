from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class _UploadedFile:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_upload_memory_summary_preview_and_item_delete_count_refresh(tmp_path: Path) -> None:
    from field_analysis.ui_upload_memory_management import build_upload_memory_summary_rows
    from field_analysis.ui_upload_memory_management import delete_upload_memory_items
    from field_analysis.ui_upload_state import build_upload_state_paths
    from field_analysis.ui_upload_state import persist_uploaded_files

    paths = build_upload_state_paths(tmp_path)
    persist_uploaded_files(
        "continuous",
        [
            _UploadedFile("continuous_sine_1Hz.csv", b"time_s,bz_mT\n0,0\n"),
            _UploadedFile("continuous_triangle_2Hz.csv", b"time_s,bz_mT\n0,1\n"),
            _UploadedFile("continuous_square_3Hz.csv", b"time_s,bz_mT\n0,2\n"),
        ],
        paths=paths,
    )

    rows = build_upload_memory_summary_rows(paths=paths, preview_limit=2)

    assert rows[0]["count"] == 3
    assert "continuous_sine_1Hz.csv" in rows[0]["files preview"]
    assert "+1" in rows[0]["files preview"]

    upload_item_id = rows[0]["item_ids"][0]
    deleted = delete_upload_memory_items("continuous", [upload_item_id], paths=paths)
    refreshed = build_upload_memory_summary_rows(paths=paths)

    assert deleted == [upload_item_id]
    assert refreshed[0]["count"] == 2


def test_upload_memory_group_records_include_item_level_metadata(tmp_path: Path) -> None:
    from field_analysis.ui_upload_memory_management import build_upload_memory_group_records
    from field_analysis.ui_upload_state import build_upload_state_paths
    from field_analysis.ui_upload_state import persist_uploaded_files

    paths = build_upload_state_paths(tmp_path)
    persist_uploaded_files(
        "transient",
        [_UploadedFile("finite_sine_1Hz_1.5cycle.csv", b"time_s,bz_mT\n0,0\n1,1\n")],
        paths=paths,
    )

    records = build_upload_memory_group_records("transient", paths=paths)

    assert len(records) == 1
    record = records[0]
    assert record["original filename"] == "finite_sine_1Hz_1.5cycle.csv"
    assert record["parsed waveform"] == "sine"
    assert record["parsed freq_hz"] == 1.0
    assert record["parsed cycle_count"] == 1.5
    assert record["row count"] == 2
    assert record["parse status"] in {"ok", "metadata_only"}
    assert isinstance(record["upload_item_id"], str)
    assert record["internal id"] == record["upload_item_id"]


def test_upload_memory_duplicate_detection_warns_but_same_content_is_idempotent(tmp_path: Path) -> None:
    from field_analysis.ui_upload_memory_management import find_duplicate_upload_names
    from field_analysis.ui_upload_state import build_upload_memory_items
    from field_analysis.ui_upload_state import build_upload_state_paths
    from field_analysis.ui_upload_state import persist_uploaded_files

    paths = build_upload_state_paths(tmp_path)
    persist_uploaded_files("validation", [_UploadedFile("result.csv", b"a,b\n1,2\n")], paths=paths)

    duplicates = find_duplicate_upload_names("validation", ["result.csv", "new.csv"], paths=paths)
    persist_uploaded_files("validation", [_UploadedFile("result.csv", b"a,b\n1,2\n")], paths=paths)
    validation_items = [item for item in build_upload_memory_items(paths=paths) if item["category"] == "validation"]

    assert duplicates == ["result.csv"]
    assert len(validation_items) == 1
    assert not any(item.get("duplicate_of") for item in validation_items)


def test_upload_memory_management_source_uses_scalar_item_ids_not_dataframes() -> None:
    source = (REPO_ROOT / "src" / "field_analysis" / "ui_upload_memory_management.py").read_text(encoding="utf-8")
    upload_state = (REPO_ROOT / "src" / "field_analysis" / "ui_upload_state.py").read_text(encoding="utf-8")

    assert "upload_item_id" in source
    assert "selected_ids: list[str]" in source
    assert "render_upload_memory_management" in upload_state
    assert "DataFrame" not in source.split("selected_ids: list[str]", 1)[1].split("confirm_selected", 1)[0]
