from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.continuous_candidate_discovery import discover_continuous_candidate_frames
from field_analysis.continuous_steady_state_schema import adapt_continuous_source_frame
from field_analysis.parser import infer_dataset_filename_metadata
from field_analysis.ui_raw_waveforms_labels import infer_new_dataset_filename_metadata
from field_analysis.ui_upload_state import UploadStatePaths, build_upload_memory_items, category_payloads


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


def _measured_csv() -> bytes:
    return b"# Frequency(Hz),2.000\nRow,TimeMs,HallBz,Voltage1_V\n0,0,0,0\n1,10,-1,1\n"


def test_prefixed_finite_and_continuous_names_recover_canonical_metadata() -> None:
    finite = infer_new_dataset_filename_metadata("0b44e758841d844c_finite_tri_2Hz_1.75cycle.csv")
    continuous = infer_new_dataset_filename_metadata("166756f8b28c75c9_continuous_tri_2Hz.csv")
    second = infer_new_dataset_filename_metadata("finite_recommended_voltage_lut_sine_2Hz_1.5cycle.csv")

    assert finite["waveform_type"] == "triangle"
    assert finite["freq_hz"] == 2.0
    assert finite["cycle_count"] == 1.75
    assert continuous["waveform_type"] == "triangle"
    assert continuous["freq_hz"] == 2.0
    assert second["source_type"] is None


def test_default_triangle_lut_names_parse_without_waveform_token() -> None:
    continuous = infer_new_dataset_filename_metadata("continuous_2hz.csv")
    finite_one = infer_new_dataset_filename_metadata("finite_1cycle_2hz.csv")
    finite_one_half = infer_new_dataset_filename_metadata("finite_1.5cycle_2hz.csv")

    assert continuous["source_type"] == "continuous"
    assert continuous["waveform_type"] == "triangle"
    assert continuous["waveform_source"] == "default_triangle_filename"
    assert continuous["freq_hz"] == 2.0
    assert finite_one["source_type"] == "finite-cycle"
    assert finite_one["waveform_type"] == "triangle"
    assert finite_one["freq_hz"] == 2.0
    assert finite_one["cycle_count"] == 1.0
    assert finite_one_half["waveform_type"] == "triangle"
    assert finite_one_half["freq_hz"] == 2.0
    assert finite_one_half["cycle_count"] == 1.5


def test_parser_default_triangle_lut_names_parse_without_waveform_token() -> None:
    continuous = infer_dataset_filename_metadata("continuous_0.25hz.csv")
    finite = infer_dataset_filename_metadata("finite_1.5cycle_0.25hz.csv")

    assert continuous["source_type"] == "continuous"
    assert continuous["waveform_type"] == "triangle"
    assert continuous["waveform_source"] == "default_triangle_filename"
    assert continuous["freq_hz"] == 0.25
    assert finite["source_type"] == "finite_cycle"
    assert finite["waveform_type"] == "triangle"
    assert finite["waveform_source"] == "default_triangle_filename"
    assert finite["freq_hz"] == 0.25
    assert finite["cycle_count"] == 1.5


def test_scanned_prefixed_upload_memory_item_uses_canonical_filename_without_renaming(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    category_dir = paths.category_dir("continuous")
    category_dir.mkdir(parents=True)
    stored = category_dir / "166756f8b28c75c9_continuous_tri_2Hz.csv"
    stored.write_bytes(_measured_csv())

    items = build_upload_memory_items(paths=paths)

    assert stored.exists()
    assert items[0]["stored_filename"] == "166756f8b28c75c9_continuous_tri_2Hz.csv"
    assert items[0]["canonical_filename"] == "continuous_tri_2Hz.csv"
    assert items[0]["original_filename"] == "continuous_tri_2Hz.csv"
    assert items[0]["waveform_family"] == "triangle"
    assert items[0]["freq_hz"] == 2.0


def test_category_payloads_expose_canonical_name_for_prefixed_cached_files(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    category_dir = paths.category_dir("continuous")
    category_dir.mkdir(parents=True)
    stored = category_dir / "166756f8b28c75c9_continuous_tri_2Hz.csv"
    stored.write_bytes(_measured_csv())
    paths.upload_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths.upload_manifest_path.write_text(
        json.dumps(
            {
                "files": {
                    "continuous": [{"cache_name": stored.name, "file_name": stored.name, "size_bytes": stored.stat().st_size}],
                    "transient": [],
                    "validation": [],
                    "lcr": [],
                }
            }
        ),
        encoding="utf-8",
    )

    payloads = category_payloads("continuous", None, paths=paths)

    assert payloads[0][0] == "continuous_tri_2Hz.csv"


def test_continuous_triangle_candidate_discovered_despite_hash_prefix() -> None:
    names, candidates, scan = discover_continuous_candidate_frames(
        {},
        upload_payloads=[("166756f8b28c75c9_continuous_tri_2Hz.csv", _measured_csv())],
        dataset_library_payloads=[],
        target_freq_hz=2.0,
        source_waveform_filter="triangle",
    )

    assert names == ["upload_memory:continuous_tri_2Hz.csv"]
    assert scan["continuous_candidate_rejected_count"] == 0
    detail = scan["continuous_candidate_details"][0]
    assert detail["filename"] == "continuous_tri_2Hz.csv"
    assert detail["storage_filename"] == "166756f8b28c75c9_continuous_tri_2Hz.csv"
    assert detail["continuous_source_waveform_family"] == "triangle"
    assert candidates[names[0]].attrs["upload_filename_prefix_stripped"] is True


def test_continuous_default_triangle_candidate_discovered_without_waveform_token() -> None:
    names, candidates, scan = discover_continuous_candidate_frames(
        {},
        upload_payloads=[("continuous_2hz.csv", _measured_csv())],
        dataset_library_payloads=[],
        target_freq_hz=2.0,
        source_waveform_filter="triangle",
    )

    assert names == ["upload_memory:continuous_2hz.csv"]
    assert scan["continuous_candidate_rejected_count"] == 0
    detail = scan["continuous_candidate_details"][0]
    assert detail["filename"] == "continuous_2hz.csv"
    assert detail["continuous_source_waveform_family"] == "triangle"
    assert detail["continuous_source_waveform_source"] == "default_triangle_filename"
    assert candidates[names[0]].attrs["continuous_source_waveform_family"] == "triangle"


def test_schema_adapter_accepts_bz_mt_alias_but_rejects_final_lut() -> None:
    import pandas as pd

    adapted, metadata = adapt_continuous_source_frame(pd.DataFrame({"time_s": [0, 1], "voltage_v": [0, 1], "Bz_mT": [0, 2]}))

    assert metadata["continuous_schema_status"] == "ok"
    assert metadata["continuous_schema_hall_or_field_column"] == "Bz_mT"
    try:
        adapt_continuous_source_frame(pd.DataFrame({"sample_index": [0], "time_s": [0.0], "voltage_v": [0.0]}))
    except ValueError as exc:
        assert str(exc) == "final_voltage_lut_not_measured_input"
    else:
        raise AssertionError("final LUT schema must be rejected as measured source")
