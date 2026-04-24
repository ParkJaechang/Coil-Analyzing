from __future__ import annotations

import json
import re
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "app_field_analysis_latest.py"
APP_STATE_DIR = REPO_ROOT / "outputs" / "field_analysis_app_state"
OPAQUE_PREFIX = re.compile(r"^[0-9a-fA-F]{12,}_")


@contextmanager
def _isolated_app_state() -> Path:
    backup_dir = APP_STATE_DIR.parent / f"{APP_STATE_DIR.name}_finite_raw_backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if APP_STATE_DIR.exists():
        shutil.move(str(APP_STATE_DIR), str(backup_dir))
    APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        yield APP_STATE_DIR
    finally:
        if APP_STATE_DIR.exists():
            shutil.rmtree(APP_STATE_DIR)
        if backup_dir.exists():
            shutil.move(str(backup_dir), str(APP_STATE_DIR))


def _clear_field_analysis_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "field_analysis" or module_name.startswith("field_analysis."):
            sys.modules.pop(module_name, None)


def _write_finite_fixture(state_dir: Path) -> None:
    category_dir = state_dir / "uploads" / "transient"
    category_dir.mkdir(parents=True, exist_ok=True)
    file_name = "10d2317e131196fe_triangle_1Hz_1.25cycle_10App.csv"
    file_path = category_dir / file_name

    freq_hz = 1.0
    cycle_count = 1.25
    timestamps = np.linspace(0.0, 1.8, 181)
    active = timestamps <= cycle_count / freq_hz
    angle = 2.0 * np.pi * freq_hz * timestamps
    voltage = np.where(active, 2.0 * np.sin(angle), 0.0)
    field = np.where(active, 40.0 * np.sin(angle - 0.1), 0.0)
    base_time = datetime(2026, 4, 7, 16, 45, 31, tzinfo=timezone(timedelta(hours=9)))
    lines = [
        "# waveform,Triangle",
        "# frequency(Hz),1",
        "# cycle,1.25",
        "# Target Current(A),10",
        "#",
        "Timestamp,T1,T2,T3,T4,Current1_A,Current2_A,Current1Peak_A,Current2Peak_A,Voltage1,Voltage2,CycleNo,HallBx,HallBy,HallBz,HallBxPeak,HallByPeak,HallBzPeak",
    ]
    for index, seconds in enumerate(timestamps):
        iso_time = (base_time + timedelta(seconds=float(seconds))).isoformat()
        lines.append(
            ",".join(
                [
                    iso_time,
                    "22.0",
                    "22.0",
                    "22.0",
                    "22.0",
                    f"{0.5 * voltage[index]:.6f}",
                    "0.0",
                    "1.0",
                    "0.0",
                    f"{voltage[index]:.6f}",
                    "0.0",
                    str(int(seconds * freq_hz) + 1),
                    "0.0",
                    "0.0",
                    f"{field[index]:.6f}",
                    "0.0",
                    "0.0",
                    "40.0",
                ]
            )
        )
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "files": {
            "continuous": [],
            "transient": [
                {
                    "category": "transient",
                    "display_name": file_name,
                    "file_name": file_name,
                    "cache_name": file_name,
                    "size_bytes": file_path.stat().st_size,
                    "source": "pytest_fixture",
                }
            ],
            "validation": [],
            "lcr": [],
        }
    }
    (state_dir / "upload_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


def test_raw_waveforms_finite_cycle_runtime_view_contains_finite_controls_and_audit_helpers() -> None:
    with _isolated_app_state() as state_dir:
        _write_finite_fixture(state_dir)
        _clear_field_analysis_modules()
        app = AppTest.from_file(str(APP_PATH), default_timeout=180)
        app.run()
        for radio in app.radio:
            if getattr(radio, "key", None) == "quick_section_nav":
                radio.set_value("Raw Waveforms")
                app.run()
                break

    selectbox_by_key = {getattr(item, "key", None): item for item in app.selectbox}
    raw_selector = selectbox_by_key["raw_test_audit"]
    source_type_selector = selectbox_by_key["raw_filter_source_type"]
    raw_options = [str(option) for option in raw_selector.options]

    assert [str(option) for option in source_type_selector.options] == ["all", "continuous", "finite-cycle"]
    assert raw_options == ["finite-cycle | Triangle | 1 Hz | 1.25 cycle | 10 App | triangle_1Hz_1.25cycle_10App.csv"]
    assert not OPAQUE_PREFIX.match(str(raw_selector.value))
    assert "1.25 cycle" in str(raw_selector.value)

    captions = [str(item.value) for item in app.caption if getattr(item, "value", None) is not None]
    markdown_values = [str(item.value) for item in app.markdown if getattr(item, "value", None) is not None]

    assert any("Internal ID:" in value for value in captions)
    assert "#### Finite-cycle visual markers" in markdown_values
    assert "#### Data quality quick checks" in markdown_values
    assert any("Raw/corrected status:" in value for value in captions)
