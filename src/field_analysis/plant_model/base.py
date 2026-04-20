from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ModelContext:
    """Common model context passed to plant predictors."""

    waveform_type: str
    freq_hz: float
    target_level_value: float | None = None
    target_level_kind: str | None = None
    commanded_cycles: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelPrediction:
    """Standard model prediction bundle."""

    time_s: np.ndarray
    input_v: np.ndarray
    predicted_current_a: np.ndarray | None = None
    predicted_field_mT: np.ndarray | None = None
    debug_frame: pd.DataFrame | None = None
    debug_info: dict[str, Any] = field(default_factory=dict)


class PlantModel(Protocol):
    """Protocol for future steady-state and transient plant models."""

    def fit(self, runs: list[Any]) -> Any:
        ...

    def predict(self, input_v: np.ndarray, context: ModelContext) -> ModelPrediction:
        ...
