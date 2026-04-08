"""Typed data models used across the application."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ChannelConfig:
    column: str | None = None
    scale: float = 1.0
    offset: float = 0.0
    invert: bool = False
    delay_s: float = 0.0
    unit: str = ""


@dataclass
class ChannelMapping:
    time: ChannelConfig = field(default_factory=lambda: ChannelConfig(unit="s"))
    voltage: ChannelConfig = field(default_factory=lambda: ChannelConfig(unit="V"))
    current: ChannelConfig = field(default_factory=lambda: ChannelConfig(unit="A"))
    magnetic: dict[str, ChannelConfig] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChannelMapping":
        return cls(
            time=ChannelConfig(**payload.get("time", {})),
            voltage=ChannelConfig(**payload.get("voltage", {})),
            current=ChannelConfig(**payload.get("current", {})),
            magnetic={
                key: ChannelConfig(**value)
                for key, value in payload.get("magnetic", {}).items()
            },
        )


@dataclass
class DatasetMeta:
    dataset_id: str
    file_name: str
    stored_path: str
    file_type: str
    selected_sheet: str | None = None
    available_sheets: list[str] = field(default_factory=list)
    detected_frequency_hz: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    mapping: ChannelMapping = field(default_factory=ChannelMapping)
    notes: str = ""
    request_frequency_hz: float | None = None
    request_target_ipp_a: float | None = None
    status: str = "data loaded"
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mapping"] = self.mapping.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetMeta":
        data = dict(payload)
        data["mapping"] = ChannelMapping.from_dict(payload.get("mapping", {}))
        return cls(**data)


@dataclass
class RequestPoint:
    frequency_hz: float
    target_ipp_a: float
    status: str = "not tested"
    linked_dataset_id: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisWindow:
    cycle_start: int = 0
    cycle_count: int = 3
    detrend: bool = True
    remove_offset: bool = True
    zero_phase_smoothing: bool = False
    smoothing_order: int = 2
    smoothing_cutoff_ratio: float = 0.2
    show_zero_crossing_aux: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnalysisWindow":
        return cls(**payload)
