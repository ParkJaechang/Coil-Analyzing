from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    yaml = None

from .models import FieldSpec, SchemaConfig


STANDARD_FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        key="timestamp",
        label_ko="타임스탬프",
        aliases=("Timestamp", "Time", "Datetime", "DateTime", "시간"),
        required=True,
        description="절대 또는 상대 시간축",
    ),
    FieldSpec(
        key="daq_input_v",
        label_ko="DAQ 입력전압",
        aliases=("DAQInput_V", "InputVoltage", "Voltage1", "Voltage", "입력전압"),
        unit="V",
        description="DAQ 입력 또는 구동 전압",
    ),
    FieldSpec(
        key="daq_input_v_secondary",
        label_ko="보조 입력전압",
        aliases=("Voltage2", "InputVoltage2", "DAQInput2_V"),
        unit="V",
        description="보조 전압 채널",
    ),
    FieldSpec(
        key="coil1_current_a",
        label_ko="코일 전류 1",
        aliases=("Current1_A", "I1", "Coil1Current", "CurrentCh1"),
        unit="A",
        description="코일 전류 1",
    ),
    FieldSpec(
        key="coil2_current_a",
        label_ko="코일 전류 2",
        aliases=("Current2_A", "I2", "Coil2Current", "CurrentCh2"),
        unit="A",
        description="코일 전류 2",
    ),
    FieldSpec(
        key="coil1_peak_a",
        label_ko="코일 피크 전류 1",
        aliases=("Current1Peak_A", "I1Peak"),
        unit="A",
        description="이동 최대 기반 피크 전류 1",
    ),
    FieldSpec(
        key="coil2_peak_a",
        label_ko="코일 피크 전류 2",
        aliases=("Current2Peak_A", "I2Peak"),
        unit="A",
        description="이동 최대 기반 피크 전류 2",
    ),
    FieldSpec(
        key="temperature_t1_c",
        label_ko="온도 T1",
        aliases=("T1", "Temp1", "Temperature1"),
        unit="C",
        description="온도 채널 1",
    ),
    FieldSpec(
        key="temperature_t2_c",
        label_ko="온도 T2",
        aliases=("T2", "Temp2", "Temperature2"),
        unit="C",
        description="온도 채널 2",
    ),
    FieldSpec(
        key="temperature_t3_c",
        label_ko="온도 T3",
        aliases=("T3", "Temp3", "Temperature3"),
        unit="C",
        description="온도 채널 3",
    ),
    FieldSpec(
        key="temperature_t4_c",
        label_ko="온도 T4",
        aliases=("T4", "Temp4", "Temperature4"),
        unit="C",
        description="온도 채널 4",
    ),
    FieldSpec(
        key="temperature_c",
        label_ko="평균 온도",
        aliases=("Temperature", "TempAvg", "AverageTemperature", "온도"),
        unit="C",
        description="직접 제공된 평균 온도",
    ),
    FieldSpec(
        key="source_cycle_no",
        label_ko="원본 CycleNo",
        aliases=("CycleNo", "Cycle", "CycleIndex"),
        description="원본 파일에 들어 있는 cycle 라벨",
    ),
    FieldSpec(
        key="bx_mT",
        label_ko="Bx",
        aliases=("HallBx", "Bx", "Bx_mT"),
        unit="mT",
        description="시료 표면 자기장 X축",
    ),
    FieldSpec(
        key="by_mT",
        label_ko="By",
        aliases=("HallBy", "By", "By_mT"),
        unit="mT",
        description="시료 표면 자기장 Y축",
    ),
    FieldSpec(
        key="bz_mT",
        label_ko="Bz",
        aliases=("HallBz", "Bz", "Bz_mT"),
        unit="mT",
        description="시료 표면 자기장 Z축",
    ),
    FieldSpec(
        key="bx_peak_mT",
        label_ko="Bx 피크",
        aliases=("HallBxPeak", "BxPeak"),
        unit="mT",
        description="피크 처리된 Bx",
    ),
    FieldSpec(
        key="by_peak_mT",
        label_ko="By 피크",
        aliases=("HallByPeak", "ByPeak"),
        unit="mT",
        description="피크 처리된 By",
    ),
    FieldSpec(
        key="bz_peak_mT",
        label_ko="Bz 피크",
        aliases=("HallBzPeak", "BzPeak"),
        unit="mT",
        description="피크 처리된 Bz",
    ),
    FieldSpec(
        key="amp_gain_setting",
        label_ko="AMP 게인",
        aliases=("AmpGain", "GainSetting", "DCAmpGain", "AmpGainSetting"),
        description="DC AMP 게인 또는 설정값",
    ),
)


DEFAULT_SCHEMA_DICT = {
    "comment_prefix": "#",
    "header_search_rows": 25,
    "default_expected_cycles": 10,
    "target_current_mode": "auto",
    "preferred_sheet_names": ["data", "raw", "sheet1"],
    "default_main_field_axis": "bz_mT",
    "default_current_axis": "i_sum",
    "metadata_aliases": {
        "waveform_type": ["waveform", "파형"],
        "freq_hz": ["frequency(Hz)", "frequency", "freq_hz", "주파수"],
        "target_current_a": ["Target Current(A)", "TargetCurrent", "목표전류"],
        "cycle_hint": ["cycle", "cycles", "CycleCount", "반복횟수"],
        "amp_gain_setting": ["gain", "gain setting", "amp gain", "dc amp gain"],
        "resting_time_ms": ["Resting Time(ms)", "resting_time_ms", "휴지시간"],
        "start_temp_c": ["Start Temp(℃)", "Start Temp", "시작온도"],
        "max_temp_c": ["Max Temp(℃)", "Max Temp", "최대온도"],
        "notes": ["notes", "memo", "비고"],
    },
}


def build_default_schema() -> SchemaConfig:
    """Return the built-in fixed-format schema."""

    field_specs = {spec.key: spec for spec in STANDARD_FIELD_SPECS}
    column_aliases = {spec.key: spec.aliases for spec in STANDARD_FIELD_SPECS}
    return SchemaConfig(
        comment_prefix=DEFAULT_SCHEMA_DICT["comment_prefix"],
        header_search_rows=DEFAULT_SCHEMA_DICT["header_search_rows"],
        default_expected_cycles=DEFAULT_SCHEMA_DICT["default_expected_cycles"],
        target_current_mode=DEFAULT_SCHEMA_DICT["target_current_mode"],
        preferred_sheet_names=tuple(DEFAULT_SCHEMA_DICT["preferred_sheet_names"]),
        metadata_aliases={
            key: tuple(values)
            for key, values in DEFAULT_SCHEMA_DICT["metadata_aliases"].items()
        },
        column_aliases=column_aliases,
        field_specs=field_specs,
        default_main_field_axis=DEFAULT_SCHEMA_DICT["default_main_field_axis"],
        default_current_axis=DEFAULT_SCHEMA_DICT["default_current_axis"],
    )


def load_schema_config(config_path: str | Path | None = None) -> SchemaConfig:
    """Load schema settings from YAML, falling back to the built-in template."""

    schema = build_default_schema()
    if config_path is None or yaml is None:
        return schema

    path = Path(config_path)
    if not path.exists():
        return schema

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    preferred_sheet_names = tuple(
        loaded.get("preferred_sheet_names", list(schema.preferred_sheet_names))
    )
    metadata_aliases = {
        **schema.metadata_aliases,
        **{
            key: tuple(values)
            for key, values in (loaded.get("metadata_aliases") or {}).items()
        },
    }
    column_aliases = {
        **schema.column_aliases,
        **{
            key: tuple(values)
            for key, values in (loaded.get("column_aliases") or {}).items()
        },
    }

    field_specs = dict(schema.field_specs)
    for key, aliases in column_aliases.items():
        if key in field_specs:
            field_specs[key] = replace(field_specs[key], aliases=tuple(aliases))

    return SchemaConfig(
        comment_prefix=loaded.get("comment_prefix", schema.comment_prefix),
        header_search_rows=int(loaded.get("header_search_rows", schema.header_search_rows)),
        default_expected_cycles=int(
            loaded.get("default_expected_cycles", schema.default_expected_cycles)
        ),
        target_current_mode=str(
            loaded.get("target_current_mode", schema.target_current_mode)
        ),
        preferred_sheet_names=preferred_sheet_names,
        metadata_aliases=metadata_aliases,
        column_aliases=column_aliases,
        field_specs=field_specs,
        default_main_field_axis=str(
            loaded.get("default_main_field_axis", schema.default_main_field_axis)
        ),
        default_current_axis=str(
            loaded.get("default_current_axis", schema.default_current_axis)
        ),
    )


def dump_schema_yaml(schema: SchemaConfig) -> str:
    """Serialize a schema config into YAML for export snapshots."""

    payload = {
        "comment_prefix": schema.comment_prefix,
        "header_search_rows": schema.header_search_rows,
        "default_expected_cycles": schema.default_expected_cycles,
        "target_current_mode": schema.target_current_mode,
        "preferred_sheet_names": list(schema.preferred_sheet_names),
        "default_main_field_axis": schema.default_main_field_axis,
        "default_current_axis": schema.default_current_axis,
        "metadata_aliases": {
            key: list(value)
            for key, value in schema.metadata_aliases.items()
        },
        "column_aliases": {
            key: list(value)
            for key, value in schema.column_aliases.items()
        },
    }
    if yaml is None:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
