# Finite Feedback Correction Feature Inventory

This inventory records runtime ownership after the Quick LUT finite feedback peak correction route.

## Active Runtime

- `finite_feedback_symmetric_peak_correction`
  - Quick LUT finite feedback correction backend route.
  - Supported cycles: `1.0`, `1.5`.
  - Input: actual-drive result CSV schema with `TimeMs`, `Voltage1_V`, `HallBz`.
  - Output command columns: `feedback_correction_delta_v`, `feedback_corrected_recommended_voltage_v`, `feedback_corrected_limited_voltage_v`.
  - Final LUT export may use `feedback_corrected_limited_voltage_v` when status is `ok`.

- Final modeled voltage LUT export
  - Active export path.
  - Output columns remain `sample_index`, `time_s`, `voltage_v`.
  - Export source is recorded in `exported_voltage_source_column`.
  - Fourier or harmonic re-synthesis is not involved.

- Actual-drive review normalization
  - Active input/review path for actual-drive feedback data.
  - Raw HallBz and raw voltage are preserved.
  - Shape-review normalization maps field to peak `50mT` and voltage to peak/limit `5V`.

- Support Reference active-segment alignment
  - Active diagnostic path.
  - Support Reference remains diagnostic only and must not become a command target.

## Test-Only

- Synthetic actual-drive CSV fixtures in tests.
- Synthetic forward model callbacks used by feedback correction tests.
- Raw waveform normalization fixtures used to guard shape-only review behavior.

## Legacy Fallback

- Baseline Quick LUT finite route without feedback files.
  - Still valid when feedback is unavailable.
  - Final LUT export falls back to `limited_voltage_v`.

- Feedback peak correction for unsupported cycles.
  - `1.25` and `1.75` return `unsupported_cycle_phase_delay`.
  - No fake correction is produced.

## Deletion Candidates For Separate Cleanup PR

- Old corrected/second-LUT artifact paths and reports from prior Phase 2 experiments.
- Old validation retune flows that generate second-correction LUT artifacts, if no active UI path still depends on them.
- Any stale UI helper that exports Fourier/reconstructed command data instead of final plotted time-voltage arrays.
- Any stale finite 1.25/1.75 amplitude-correction path that implies peak-amplitude correction is production-supported.

Do not remove these in a feature PR. Confirm active imports, UI reachability, and tests first.
