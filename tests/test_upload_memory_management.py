from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.ui_upload_state import UploadStatePaths
from field_analysis.ui_upload_state import build_upload_memory_group_summary
from field_analysis.ui_upload_state import build_upload_memory_items
from field_analysis.ui_upload_state import delete_upload_memory_group
from field_analysis.ui_upload_state import delete_upload_memory_item
from field_analysis.ui_upload_state import delete_upload_memory_items
from field_analysis.ui_upload_state import category_payloads
from field_analysis.ui_upload_state import persist_uploaded_files
from field_analysis.ui_upload_state import load_upload_manifest
from field_analysis.ui_upload_memory_status import upload_memory_status
from field_analysis.ui_upload_memory_status import activate_cached_uploads


@dataclass
class _Upload:
    name: str
    payload: bytes

    def getvalue(self) -> bytes:
        return self.payload


def _paths(tmp_path: Path) -> UploadStatePaths:
    app_state = tmp_path / "outputs" / "field_analysis_app_state"
    return UploadStatePaths(
        repo_root=tmp_path,
        app_state_dir=app_state,
        uploads_dir=app_state / "uploads",
        upload_manifest_path=app_state / "upload_manifest.json",
        recommendation_library_dir=app_state / "recommendation_library",
        validation_retune_history_path=app_state / "validation_retune_history.json",
    )


def test_upload_memory_items_have_stable_scalar_identity_and_summary_counts(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    persist_uploaded_files("continuous", [_Upload("continuous_sine_1Hz.csv", b"time_s,bz_mT\n0,0\n")], paths=paths)
    persist_uploaded_files("transient", [_Upload("finite_sine_1Hz_1cycle.csv", b"time_s,bz_mT\n0,0\n")], paths=paths)

    items = build_upload_memory_items(paths=paths)
    summary = build_upload_memory_group_summary(items)

    assert {item["label"] for item in items} == {"continuous_cycle", "finite_cycle"}
    assert all(isinstance(item["upload_item_id"], str) and item["upload_item_id"] for item in items)
    assert all(item["stored_path"].startswith(str(paths.uploads_dir)) for item in items)
    assert summary["continuous_cycle"]["count"] == 1
    assert summary["finite_cycle"]["count"] == 1
    assert summary["continuous_cycle"]["files"] == ["continuous_sine_1Hz.csv"]


def test_delete_single_multi_and_group_update_manifest_and_files(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    persist_uploaded_files(
        "continuous",
        [
            _Upload("continuous_sine_1Hz.csv", b"a\n1\n"),
            _Upload("continuous_triangle_1Hz.csv", b"a\n2\n"),
        ],
        paths=paths,
    )
    persist_uploaded_files("transient", [_Upload("finite_sine_1Hz_1cycle.csv", b"a\n3\n")], paths=paths)
    items = build_upload_memory_items(paths=paths)
    by_name = {item["original_filename"]: item for item in items}

    single = delete_upload_memory_item(by_name["continuous_sine_1Hz.csv"]["upload_item_id"], paths=paths)
    assert single["deleted_count"] == 1
    assert not Path(by_name["continuous_sine_1Hz.csv"]["stored_path"]).exists()

    remaining = build_upload_memory_items(paths=paths)
    ids = [item["upload_item_id"] for item in remaining if item["label"] == "continuous_cycle"]
    multi = delete_upload_memory_items(ids, paths=paths)
    assert multi["deleted_count"] == 1
    assert build_upload_memory_group_summary(build_upload_memory_items(paths=paths))["continuous_cycle"]["count"] == 0

    group = delete_upload_memory_group("finite_cycle", paths=paths)
    assert group["deleted_count"] == 1
    assert build_upload_memory_group_summary(build_upload_memory_items(paths=paths))["finite_cycle"]["count"] == 0


def test_group_delete_removes_scanned_file_items_without_manifest_entry(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    orphan_dir = paths.category_dir("continuous")
    orphan_dir.mkdir(parents=True)
    orphan_path = orphan_dir / "continuous_sine_5Hz.csv"
    orphan_path.write_text("time_s,bz_mT\n0,0\n", encoding="utf-8")

    items = build_upload_memory_items(paths=paths)

    assert any(item["stored_filename"] == orphan_path.name for item in items)
    result = delete_upload_memory_group("continuous_cycle", paths=paths)
    summary = build_upload_memory_group_summary(build_upload_memory_items(paths=paths))

    assert result["deleted_count"] == 1
    assert result["physical_deleted_count"] == 1
    assert summary["continuous_cycle"]["count"] == 0
    assert not orphan_path.exists()


def test_delete_missing_and_outside_path_are_safe(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_text("do-not-delete", encoding="utf-8")
    paths.upload_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths.upload_manifest_path.write_text(
        '{"files":{"continuous":[{"upload_item_id":"continuous:outside","file_name":"outside.csv",'
        f'"display_name":"outside.csv","cache_name":"outside.csv","path":"{outside.as_posix()}","size_bytes":1'
        '}],"transient":[],"validation":[],"lcr":[]}}',
        encoding="utf-8",
    )

    result = delete_upload_memory_item("continuous:outside", paths=paths)

    assert result["deleted_count"] == 1
    assert result["physical_deleted_count"] == 0
    assert outside.exists()
    assert delete_upload_memory_item("missing", paths=paths)["deleted_count"] == 0


def test_repeated_same_upload_is_idempotent_and_uses_scalar_widget_state(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    uploads = [
        _Upload("finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv", b"same"),
        _Upload("finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv", b"same"),
    ]
    persist_uploaded_files("validation", uploads, paths=paths)
    persist_uploaded_files("validation", uploads, paths=paths)

    items = build_upload_memory_items(paths=paths)
    validation_items = [item for item in items if item["label"] == "actual_drive_validation_run"]
    selected_ids = [item["upload_item_id"] for item in validation_items]
    session_state = {"selected_upload_item_ids": selected_ids}

    assert len(validation_items) == 1
    assert validation_items[0]["duplicate_of"] is None
    assert all(isinstance(item_id, str) for item_id in session_state["selected_upload_item_ids"])
    assert not any(isinstance(value, pd.DataFrame) for value in session_state.values())


def test_same_filename_different_content_keeps_separate_items(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    persist_uploaded_files(
        "validation",
        [
            _Upload("finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv", b"same"),
            _Upload("finite_recommended_voltage_lut_sine_1Hz_1cycle_result.csv", b"different"),
        ],
        paths=paths,
    )

    items = build_upload_memory_items(paths=paths)
    validation_items = [item for item in items if item["label"] == "actual_drive_validation_run"]

    assert len(validation_items) == 2
    assert len({item["stored_filename"] for item in validation_items}) == 2


def test_category_payloads_does_not_duplicate_persisted_files_on_rerun(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    upload = _Upload("continuous_sine_1Hz.csv", b"time_s,bz_mT\n0,0\n")

    first_payloads = category_payloads("continuous", [upload], paths=paths)
    second_payloads = category_payloads("continuous", [upload], paths=paths)
    summary = build_upload_memory_group_summary(build_upload_memory_items(paths=paths))

    assert len(first_payloads) == 1
    assert len(second_payloads) == 1
    assert summary["continuous_cycle"]["count"] == 1


def test_category_payloads_without_current_upload_does_not_auto_load_cached_files(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    persist_uploaded_files("continuous", [_Upload("continuous_sine_1Hz.csv", b"time_s,bz_mT\n0,0\n")], paths=paths)
    manifest = json.loads(paths.upload_manifest_path.read_text(encoding="utf-8"))
    manifest["active_uploads"]["continuous"] = []
    paths.upload_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    payloads = category_payloads("continuous", None, paths=paths)
    cached_payloads = category_payloads("continuous", None, paths=paths, include_cached_uploads=True)

    assert payloads == []
    assert len(cached_payloads) == 1


def test_category_payloads_loads_active_remembered_set_without_current_upload(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    upload = _Upload("continuous_sine_1Hz.csv", b"time_s,bz_mT\n0,0\n")
    category_payloads("continuous", [upload], paths=paths)

    remembered_payloads = category_payloads("continuous", None, paths=paths)

    assert len(remembered_payloads) == 1


def test_legacy_manifest_without_active_uploads_migrates_cached_files_to_active(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    category_dir = paths.category_dir("continuous")
    category_dir.mkdir(parents=True, exist_ok=True)
    cache_name = "continuous_sine_2Hz.csv"
    (category_dir / cache_name).write_bytes(b"time_s,bz_mT\n0,0\n")
    paths.upload_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths.upload_manifest_path.write_text(
        json.dumps(
            {
                "files": {
                    "continuous": [{"cache_name": cache_name, "file_name": cache_name, "size_bytes": 16}],
                    "transient": [],
                    "validation": [],
                    "lcr": [],
                }
            }
        ),
        encoding="utf-8",
    )

    remembered_payloads = category_payloads("continuous", None, paths=paths)

    assert len(remembered_payloads) == 1


def test_activate_cached_uploads_marks_cached_files_as_active(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    upload = _Upload("continuous_triangle_1Hz.csv", b"time_s,bz_mT\n0,0\n")
    persist_uploaded_files("continuous", [upload], paths=paths)
    manifest = json.loads(paths.upload_manifest_path.read_text(encoding="utf-8"))
    manifest["active_uploads"]["continuous"] = []
    paths.upload_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = activate_cached_uploads("continuous", paths=paths)
    remembered_payloads = category_payloads("continuous", None, paths=paths)

    assert result["activated_count"] == 1
    assert len(remembered_payloads) == 1


def test_continuous_category_aliases_restore_as_canonical_continuous(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    upload = _Upload("continuous_tri_2Hz.csv", b"time_s,bz_mT\n0,0\n")

    category_payloads("continuous-cycle", [upload], paths=paths)
    remembered_payloads = category_payloads("연속 cycle", None, paths=paths)
    manifest = load_upload_manifest(paths=paths)

    assert len(remembered_payloads) == 1
    assert "continuous" in manifest["files"]
    assert len(manifest["files"]["continuous"]) == 1
    assert manifest["files"]["continuous"][0]["category"] == "continuous"
    assert manifest["files"]["continuous"][0]["upload_category_alias_applied"] is True


def test_legacy_alias_manifest_active_uploads_merge_into_continuous(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    category_dir = paths.category_dir("continuous")
    category_dir.mkdir(parents=True, exist_ok=True)
    cache_name = "abc12345_continuous_tri_3Hz.csv"
    (category_dir / cache_name).write_bytes(b"time_s,bz_mT\n0,0\n")
    paths.upload_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths.upload_manifest_path.write_text(
        json.dumps(
            {
                "files": {
                    "continuous-cycle": [
                        {"cache_name": cache_name, "file_name": "continuous_tri_3Hz.csv", "size_bytes": 16}
                    ],
                },
                "active_uploads": {"continuous-cycle": [cache_name]},
            }
        ),
        encoding="utf-8",
    )

    remembered_payloads = category_payloads("continuous", None, paths=paths)
    manifest = load_upload_manifest(paths=paths)

    assert len(remembered_payloads) == 1
    assert manifest["active_uploads"]["continuous"] == [cache_name]
    assert manifest["files"]["continuous"][0]["upload_category_original"] == "continuous-cycle"


def test_upload_memory_status_reports_remembered_but_missing_files(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.upload_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths.upload_manifest_path.write_text(
        json.dumps(
            {
                "files": {
                    "continuous": [
                        {
                            "cache_name": "missing_continuous_tri_1Hz.csv",
                            "file_name": "continuous_tri_1Hz.csv",
                            "size_bytes": 16,
                        }
                    ],
                    "transient": [],
                    "validation": [],
                    "lcr": [],
                },
                "active_uploads": {
                    "continuous": ["missing_continuous_tri_1Hz.csv"],
                    "transient": [],
                    "validation": [],
                    "lcr": [],
                },
            }
        ),
        encoding="utf-8",
    )

    status = upload_memory_status(paths=paths)
    payloads = category_payloads("continuous", None, paths=paths)

    assert payloads == []
    assert status["remembered_continuous_count"] == 1
    assert status["cached_continuous_count"] == 0
    assert status["missing_remembered_continuous_count"] == 1
    assert status["upload_memory_restore_status"] == "no_cached_files"
    assert status["continuous_upload_restore_status"] == "remembered_but_missing_files"
