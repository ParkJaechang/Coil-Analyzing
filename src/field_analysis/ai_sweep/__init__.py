"""Experimental offline AI sweep planning helpers.

This package is not a production Quick LUT route. It does not execute hardware,
does not call the modeling core automatically, does not mutate the Streamlit
runtime, and does not export a production LUT unless a reviewed future path
explicitly calls it. User approval is required before any runtime or hardware
use.
"""

from .schema import (
    ManifestValidationResult,
    SweepSegmentManifestRow,
    SweepSegmentSpec,
    SweepTargetConfig,
)
from .segment_parser import SegmentMeasurement, SegmentSplitResult, split_long_measurement_by_manifest
from .sweep_plan import SweepPlanConfig, build_sweep_plan, plan_to_dataframe
from .sweep_lut_generator import (
    SegmentCommandInput,
    SweepLutBuildResult,
    build_sweep_lut_from_segment_commands,
)

__all__ = [
    "ManifestValidationResult",
    "SegmentMeasurement",
    "SegmentCommandInput",
    "SegmentSplitResult",
    "SweepPlanConfig",
    "SweepLutBuildResult",
    "SweepSegmentManifestRow",
    "SweepSegmentSpec",
    "SweepTargetConfig",
    "build_sweep_lut_from_segment_commands",
    "build_sweep_plan",
    "plan_to_dataframe",
    "split_long_measurement_by_manifest",
]
