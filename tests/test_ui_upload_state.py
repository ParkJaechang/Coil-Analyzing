from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELD_ANALYSIS_SRC = ROOT.parent / "src"
if str(FIELD_ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIELD_ANALYSIS_SRC))

from field_analysis.ui_upload_state import (  # noqa: E402
    build_upload_state_paths,
    category_payloads,
    category_summary_rows,
    clear_all_uploads,
    list_persisted_uploads,
    load_upload_manifest,
    persist_uploaded_files,
)


class _UploadedFileStub:
    def __init__(self, name: str, payload: bytes) -> None:
        self.name = name
        self._payload = payload

    def getvalue(self) -> bytes:
        return self._payload


def test_category_payloads_reads_stored_manifest_files_without_new_uploads(tmp_path: Path) -> None:
    paths = build_upload_state_paths(repo_root=tmp_path)
    category_dir = paths.category_dir("continuous")
    category_dir.mkdir(parents=True, exist_ok=True)
    cache_name = "0123456789abcdef_sine_1_10.csv"
    payload = b"Timestamp,Voltage1\n0,1\n"
    (category_dir / cache_name).write_bytes(payload)
    paths.upload_manifest_path.write_text(
        (
            '{\n'
            '  "files": {\n'
            '    "continuous": [\n'
            f'      {{"file_name": "sine_1_10.csv", "display_name": "sine_1_10.csv", "cache_name": "{cache_name}", "size_bytes": {len(payload)}}}\n'
            "    ]\n"
            "  }\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    restored = category_payloads("continuous", None, paths=paths)

    assert restored == [(cache_name, payload)]


def test_persist_uploaded_files_hashes_and_updates_manifest(tmp_path: Path) -> None:
    paths = build_upload_state_paths(repo_root=tmp_path)
    uploaded = _UploadedFileStub("validation_case.csv", b"Timestamp,bz_mT\n0,2\n")

    records = persist_uploaded_files("validation", [uploaded], paths=paths)

    assert len(records) == 1
    record = records[0]
    assert record["display_name"] == "validation_case.csv"
    assert record["cache_name"].endswith("_validation_case.csv")
    assert len(record["cache_name"].split("_", 1)[0]) == 16
    assert Path(record["path"]).exists()
    manifest = load_upload_manifest(paths=paths)
    assert manifest["files"]["validation"][0]["display_name"] == "validation_case.csv"


def test_list_persisted_uploads_includes_unmanifested_scanned_files(tmp_path: Path) -> None:
    paths = build_upload_state_paths(repo_root=tmp_path)
    category_dir = paths.category_dir("lcr")
    category_dir.mkdir(parents=True, exist_ok=True)
    orphan_path = category_dir / "orphan_lcr.xlsx"
    orphan_path.write_bytes(b"dummy")

    records = list_persisted_uploads("lcr", paths=paths)

    assert len(records) == 1
    assert records[0]["cache_name"] == "orphan_lcr.xlsx"
    assert records[0]["source"] == "scan"


def test_clear_all_uploads_resets_counts(tmp_path: Path) -> None:
    paths = build_upload_state_paths(repo_root=tmp_path)
    persist_uploaded_files("continuous", [_UploadedFileStub("a.csv", b"a")], paths=paths)
    persist_uploaded_files("transient", [_UploadedFileStub("b.csv", b"b")], paths=paths)

    clear_all_uploads(paths=paths)

    rows = category_summary_rows(paths=paths)
    assert all(row["count"] == 0 for row in rows)
    manifest = load_upload_manifest(paths=paths)
    assert all(not entries for entries in manifest["files"].values())
