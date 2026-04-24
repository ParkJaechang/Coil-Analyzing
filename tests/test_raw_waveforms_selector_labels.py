from __future__ import annotations

import json
import re
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_analysis.ui_raw_waveforms import build_raw_waveform_label_lookup


APP_PATH = REPO_ROOT / "app_field_analysis_latest.py"
APP_STATE_DIR = REPO_ROOT / "outputs" / "field_analysis_app_state"
OPAQUE_PREFIX = re.compile(r"^[0-9a-fA-F]{12,}_")


def _analysis_fixture() -> SimpleNamespace:
    normalized = pd.DataFrame(
        {
            "test_id": ["10d2317e131196fe_sine_2_20_sine_2Hz_20App__sine__2Hz__20App"] * 3,
            "source_file": ["10d2317e131196fe_sine_2_20_sine_2Hz_20App.csv"] * 3,
            "sheet_name": ["main"] * 3,
            "waveform_type": ["Sine"] * 3,
            "freq_hz": [2.0] * 3,
            "current_pp_target_a": [20.0] * 3,
            "cycle_total_expected": [0.0] * 3,
            "time_s": [0.0, 0.5, 1.0],
            "daq_input_v": [0.0, 1.0, 0.0],
            "bz_mT": [0.0, 20.0, 0.0],
        }
    )
    parsed = SimpleNamespace(
        source_file="10d2317e131196fe_sine_2_20_sine_2Hz_20App.csv",
        sheet_name="main",
        metadata={"waveform": "Sine", "frequency(Hz)": "2", "Target Current(A)": "20", "cycle": "0"},
        normalized_frame=normalized,
    )
    preprocess = SimpleNamespace(corrected_frame=normalized.copy())
    return SimpleNamespace(parsed=parsed, preprocess=preprocess)


def test_selector_label_builder_uses_human_label_as_primary_option() -> None:
    test_id = "10d2317e131196fe_sine_2_20_sine_2Hz_20App__sine__2Hz__20App"
    label_by_id, id_by_label = build_raw_waveform_label_lookup([test_id], {test_id: _analysis_fixture()})
    label = label_by_id[test_id]

    assert not OPAQUE_PREFIX.match(label)
    assert label == "continuous | Sine | 2 Hz | 20 App | ±5V | Gain 100% | sine_2_20_sine_2Hz_20App.csv"
    assert "10d2317e131196fe" not in label
    assert "continuous" in label
    assert "Sine" in label
    assert "2 Hz" in label
    assert "20 App" in label
    assert "±5V" in label
    assert "Gain 100%" in label
    assert "sine_2_20_sine_2Hz_20App.csv" in label
    assert id_by_label[label] == test_id


@contextmanager
def _isolated_app_state() -> Path:
    backup_dir = APP_STATE_DIR.parent / f"{APP_STATE_DIR.name}_raw_selector_backup"
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


def _write_runtime_fixture(state_dir: Path) -> None:
    category_dir = state_dir / "uploads" / "continuous"
    category_dir.mkdir(parents=True, exist_ok=True)
    file_name = "10d2317e131196fe_sine_2_20_sine_2Hz_20App.csv"
    file_path = category_dir / file_name

    base_time = datetime(2026, 4, 7, 16, 45, 31, tzinfo=timezone(timedelta(hours=9)))
    timestamps = np.linspace(0.0, 5.0, 101)
    angle = 2.0 * np.pi * 2.0 * timestamps
    lines = [
        "# waveform,Sine",
        "# frequency(Hz),2",
        "# cycle,0",
        "# Target Current(A),20",
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
                    f"{np.sin(angle[index]):.6f}",
                    "0.0",
                    "1.0",
                    "0.0",
                    f"{2.0 * np.sin(angle[index]):.6f}",
                    "0.0",
                    str(int(seconds * 2.0) + 1),
                    "0.0",
                    "0.0",
                    f"{20.0 * np.sin(angle[index]):.6f}",
                    "0.0",
                    "0.0",
                    "20.0",
                ]
            )
        )
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "files": {
            "continuous": [
                {
                    "category": "continuous",
                    "display_name": file_name,
                    "file_name": file_name,
                    "cache_name": file_name,
                    "size_bytes": file_path.stat().st_size,
                    "source": "pytest_fixture",
                }
            ],
            "transient": [],
            "validation": [],
            "lcr": [],
        }
    }
    (state_dir / "upload_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


def test_raw_waveforms_runtime_selector_uses_label_options_without_hash_prefix() -> None:
    with _isolated_app_state() as state_dir:
        _write_runtime_fixture(state_dir)
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
    option_labels = [str(option) for option in raw_selector.options]

    assert raw_selector.label == "테스트 선택 (metadata label)"
    assert str(raw_selector.value) == "continuous | Sine | 2 Hz | 20 App | ±5V | Gain 100% | sine_2_20_sine_2Hz_20App.csv"
    assert option_labels == ["continuous | Sine | 2 Hz | 20 App | ±5V | Gain 100% | sine_2_20_sine_2Hz_20App.csv"]
    assert all(not OPAQUE_PREFIX.match(label) for label in option_labels)
    assert any(item.label == "비교 기준 테스트 (선택)" for item in app.selectbox)

    captions = [str(item.value) for item in app.caption if getattr(item, "value", None) is not None]
    assert any("Internal ID:" in value for value in captions)
    assert any("선택한 파형과 겹쳐 비교할 기준 테스트입니다" in value for value in captions)
