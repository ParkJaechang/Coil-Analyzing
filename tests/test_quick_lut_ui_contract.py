from __future__ import annotations

import json
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


APP_PATH = REPO_ROOT / "app_field_analysis_quick.py"
APP_STATE_DIR = REPO_ROOT / "outputs" / "field_analysis_app_state"


def _collect_text_values(app: AppTest) -> list[str]:
    values: list[str] = []
    for collection_name in ("markdown", "caption", "info", "success", "warning", "text"):
        for item in getattr(app, collection_name, []):
            value = getattr(item, "value", None)
            if value is not None:
                values.append(str(value))
    return values


def _selectbox_labels(app: AppTest) -> list[str]:
    return [str(item.label) for item in app.selectbox]


def _number_input_labels(app: AppTest) -> list[str]:
    return [str(item.label) for item in app.number_input]


def _write_sample_continuous_fixture(state_dir: Path, *, freq_hz: float, gain_label: str) -> dict[str, object]:
    category_dir = state_dir / "uploads" / "continuous"
    category_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{freq_hz:g}Hz_9V_{gain_label}.csv"
    file_path = category_dir / file_name

    total_duration_s = 20.0
    sample_count = 401
    timestamps = np.linspace(0.0, total_duration_s, sample_count)
    angle = 2.0 * np.pi * float(freq_hz) * timestamps
    voltage = 9.0 * np.sin(angle)
    current_1 = 0.8 * np.sin(angle - 0.15)
    current_2 = 0.05 * np.cos(angle)
    hall_bx = 8.0 * np.sin(angle - 0.05)
    hall_by = 5.0 * np.cos(angle)
    hall_bz = 50.0 * np.sin(angle - 0.2)

    base_time = datetime(2026, 4, 7, 16, 45, 31, tzinfo=timezone(timedelta(hours=9)))
    lines = [
        f"# Date,{base_time.strftime('%Y-%m-%d %H:%M:%S')}",
        "# waveform,Sine",
        f"# frequency(Hz),{freq_hz:.3f}",
        "# cycle,0.00",
        "# Target Current(A),0.000",
        "# Resting Time(ms),0",
        "# Max Temp(C),99999.0",
        "# Start Temp(C),T1=22.42 T2=23.53 T3=21.81 T4=24.21",
        "#",
        "Timestamp,T1,T2,T3,T4,Current1_A,Current2_A,Current1Peak_A,Current2Peak_A,Voltage1,Voltage2,CycleNo,HallBx,HallBy,HallBz,HallBxPeak,HallByPeak,HallBzPeak",
    ]
    for index, seconds in enumerate(timestamps):
        iso_time = (base_time + timedelta(seconds=float(seconds))).isoformat()
        cycle_no = int(np.floor(float(seconds) * float(freq_hz))) + 1
        lines.append(
            ",".join(
                [
                    iso_time,
                    "22.400",
                    "23.500",
                    "21.800",
                    "24.200",
                    f"{current_1[index]:.6f}",
                    f"{current_2[index]:.6f}",
                    f"{abs(current_1[index]):.6f}",
                    f"{abs(current_2[index]):.6f}",
                    f"{voltage[index]:.6f}",
                    "0.000000",
                    str(cycle_no),
                    f"{hall_bx[index]:.6f}",
                    f"{hall_by[index]:.6f}",
                    f"{hall_bz[index]:.6f}",
                    f"{abs(hall_bx[index]):.6f}",
                    f"{abs(hall_by[index]):.6f}",
                    f"{abs(hall_bz[index]):.6f}",
                ]
            )
        )
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "category": "continuous",
        "display_name": file_name,
        "file_name": file_name,
        "cache_name": file_name,
        "size_bytes": file_path.stat().st_size,
        "source": "pytest_fixture",
    }


def _write_upload_manifest(state_dir: Path, continuous_records: list[dict[str, object]]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "files": {
            "continuous": continuous_records,
            "transient": [],
            "validation": [],
            "lcr": [],
        }
    }
    (state_dir / "upload_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@contextmanager
def _isolated_app_state() -> Path:
    backup_dir = APP_STATE_DIR.parent / f"{APP_STATE_DIR.name}_pytest_backup"
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


def test_quick_lut_initial_screen_shows_field_only_banner_without_legacy_targets() -> None:
    with _isolated_app_state():
        app = AppTest.from_file(str(APP_PATH), default_timeout=180)
        app.run()

    text_values = _collect_text_values(app)
    selectbox_labels = _selectbox_labels(app)
    number_input_labels = _number_input_labels(app)

    assert any("FIELD-ONLY Quick LUT" in value for value in text_values)
    assert any("rounded triangle" in value for value in text_values)
    assert any("100 mT pp fixed" in value or "100pp fixed" in value for value in text_values)
    assert any("current / gain / hardware / LCR" in value for value in text_values)
    assert any("Runtime: Quick LUT field-only renderer v2" in value for value in text_values)
    assert any("source=repo-local src" in value for value in text_values)
    assert "크기 LUT 목표값" not in number_input_labels
    assert "파형 보정 목표 항목" not in selectbox_labels
    assert not any(label.startswith("파형 보정 목표") for label in number_input_labels)


def test_quick_lut_data_present_runtime_contract_hides_legacy_targets_and_limits_finite_cycles() -> None:
    with _isolated_app_state() as state_dir:
        records = [
            _write_sample_continuous_fixture(state_dir, freq_hz=0.5, gain_label="38gain"),
            _write_sample_continuous_fixture(state_dir, freq_hz=1.0, gain_label="43gain"),
        ]
        _write_upload_manifest(state_dir, records)

        app = AppTest.from_file(str(APP_PATH), default_timeout=180)
        app.run()

        text_values = _collect_text_values(app)
        selectbox_labels = _selectbox_labels(app)
        number_input_labels = _number_input_labels(app)

        assert any("FIELD-ONLY 운용 모드입니다." in value for value in text_values)
        assert any("rounded triangle" in value for value in text_values)
        assert any("100pp fixed" in value or "100 mT pp fixed" in value for value in text_values)
        assert any("support/input waveform family" in value for value in text_values)
        assert any("current / gain / hardware / LCR" in value for value in text_values)
        assert any("Runtime: Quick LUT field-only renderer v2" in value for value in text_values)
        assert any("source=repo-local src" in value for value in text_values)
        assert "지원 입력 파형 family" in selectbox_labels
        assert "파형" not in selectbox_labels
        assert "크기 LUT 목표값" not in number_input_labels
        assert "파형 보정 목표 항목" not in selectbox_labels
        assert not any(label.startswith("파형 보정 목표") for label in number_input_labels)

        finite_toggle = next(item for item in app.checkbox if getattr(item, "key", None) == "finite_cycle_mode_v2")
        finite_toggle.check().run()

        selectbox_by_key = {getattr(item, "key", None): item for item in app.selectbox}
        number_input_keys = {getattr(item, "key", None) for item in app.number_input}
        finite_cycle_select = selectbox_by_key["target_cycle_count_v2"]

        assert [str(option) for option in finite_cycle_select.options] == ["0.75", "1.0", "1.25", "1.5"]
        assert "target_cycle_count_v2" not in number_input_keys
