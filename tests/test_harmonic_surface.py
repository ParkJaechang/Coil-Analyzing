from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIELD_ANALYSIS_SRC = ROOT.parent / "src"
if str(FIELD_ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIELD_ANALYSIS_SRC))

from field_analysis.plant_model.harmonic_surface import (
    HarmonicSurfaceModel,
    build_harmonic_observation_frame,
    build_harmonic_transfer_frame,
    extract_harmonic_components,
    harmonic_cap,
)
from field_analysis.plant_model.base import ModelContext


def test_harmonic_cap_scales_with_sampling_and_frequency() -> None:
    assert harmonic_cap(61.2, 0.5, user_cap=31) == 31
    assert harmonic_cap(61.2, 1.0, user_cap=31) == 24
    assert harmonic_cap(61.2, 2.0, user_cap=31) == 12


def test_extract_harmonic_components_identifies_fundamental() -> None:
    time_s = np.linspace(0.0, 4.0, 4000, endpoint=False)
    values = 3.0 * np.sin(2.0 * np.pi * 1.0 * time_s)

    frame = extract_harmonic_components(time_s, values, fundamental_freq_hz=1.0, max_harmonics=5)

    fundamental = frame.loc[frame["harmonic"] == 1].iloc[0]
    higher = frame.loc[frame["harmonic"] == 3].iloc[0]
    assert fundamental["magnitude"] > 2.9
    assert higher["magnitude"] < 0.05


def test_build_harmonic_transfer_frame_returns_expected_gain() -> None:
    time_s = np.linspace(0.0, 4.0, 4000, endpoint=False)
    input_v = np.sin(2.0 * np.pi * 1.0 * time_s)
    output = 2.0 * np.sin(2.0 * np.pi * 1.0 * time_s - np.pi / 2.0)

    frame = build_harmonic_transfer_frame(
        time_s=time_s,
        input_v=input_v,
        output_signal=output,
        fundamental_freq_hz=1.0,
        max_harmonics=5,
    )

    fundamental = frame.loc[frame["harmonic"] == 1].iloc[0]
    assert np.isclose(fundamental["gain_magnitude"], 2.0, atol=0.05)
    assert np.isclose(abs(fundamental["gain_phase_rad"]), np.pi / 2.0, atol=0.05)


def test_harmonic_surface_model_inverts_and_predicts_fundamental() -> None:
    time_s = np.linspace(0.0, 1.0, 256)
    input_v = np.sin(2.0 * np.pi * time_s)
    output = 2.5 * np.sin(2.0 * np.pi * time_s - np.pi / 4.0)

    observation_frame = build_harmonic_observation_frame(
        run_id="run-1",
        waveform_type="sine",
        freq_hz=1.0,
        target_level_value=5.0,
        sample_rate_hz=256.0,
        reference_axis="current",
        output_type="current",
        time_s=time_s,
        input_v=input_v,
        output_signal=output,
        max_harmonics=5,
    )
    model = HarmonicSurfaceModel()
    fitted = model.fit(observation_frame)

    assert not fitted.empty
    context = ModelContext(
        waveform_type="sine",
        freq_hz=1.0,
        target_level_value=5.0,
        metadata={"allow_output_extrapolation": True, "max_harmonics": 5, "field_channel": "bz_mT"},
    )
    inverse = model.invert_target(
        target_output=output,
        output_type="current",
        context=context,
        max_harmonics=5,
        reference_axis="current",
    )

    assert inverse is not None
    recommended_voltage, inverse_debug, inverse_meta = inverse
    assert len(recommended_voltage) == len(output)
    assert inverse_meta["usable_harmonic_count"] >= 1
    assert not inverse_debug.empty

    prediction = model.predict(recommended_voltage, context)
    assert prediction.predicted_current_a is not None
    modeled_pp = float(np.nanmax(prediction.predicted_current_a) - np.nanmin(prediction.predicted_current_a))
    target_pp = float(np.nanmax(output) - np.nanmin(output))
    assert np.isclose(modeled_pp, target_pp, atol=0.25)
