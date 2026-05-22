# PR61 Runtime Gap Report

Runtime checked from local app at `http://127.0.0.1:8504/` and code paths that render the user-visible Quick LUT result panels. Current head before fixes: `f21acddac3d4970b478e28e1e733d2c51adf4331`.

| item | expected | current runtime | status | screenshot/evidence | fix required |
|---|---|---|---|---|---|
| Legacy target PP wording | Main UI must not show `100mT pp fixed`, `100pp fixed`, `목표 bz_mT PP`. | Result summary still renders `목표 {target_output_label}` and can show `목표 bz_mT PP 100.000 mT`. | FAIL | User screenshot and `app_ui_snapshot.py` compensation result metrics. | Replace main target-output metric with target shape / target peak / normalization semantics. |
| Bz_mT waveform compensation wording | Main graph should reflect current modeling meaning, not legacy compensation/extrapolation. | Main plot title can be `Field Waveform Compensation: bz_mT`; support range warning mentions extrapolation. | FAIL | `plot_output_compensation_waveforms(... title=f"Field Waveform Compensation: {main_field_axis}")`; warning branch for `allow_output_extrapolation`. | Rename section/plot and move support/extrapolation/hardware details into Advanced/Legacy. |
| Legacy hardware terms | `필요 AMP gain`, `DAQ Voltage`, `AMP output`, hardware-limit messages must not occupy main flow. | Hardware metrics are now in expander, but legacy text still exists in user path and older recommendation path. | PARTIAL | `Advanced / Legacy hardware diagnostics` exists; source still has strings for diagnostics. | Keep only in collapsed Advanced/Legacy; ensure no main metric uses these labels. |
| First command plot | Main plot should show only final first command with trace `1차 모델링 command`. | `_retitle_command_waveform_figure` adds a second recommended trace and uses garbled/ambiguous trace name. | FAIL | `_retitle_command_waveform_figure` adds `recommended_voltage_v` trace. | Remove extra trace in main plot and set exact trace name/title. |
| Phase sync residual active-end support | Phase-aligned measured/residual must remain finite through active end. | Finite first metadata exists, but active-end finite ratio/status are not explicit in UI summary. | PARTIAL | `finite_first_phase_sync.py`, `ui_finite_first_phase_sync.py`. | Add metadata/UI markers and block/warn on incomplete active residual. |
| Measured normalization scale | UI must show measured peak, scale to ±50mT, residual gain/headroom/clipping. | Finite first summary has scale/gain/headroom, but labels are partially garbled. | PARTIAL | `ui_finite_first_phase_sync.py`. | Rename labels clearly and add normalized peak/status fields. |
| Target rounded triangle template | Analytic fixed rounded triangle; straight segments should not ripple. | `lut.py` uses Hann convolution and `compensation.py` uses 1/3/5 harmonic approximation, both can ripple. | FAIL | `build_fixed_field_target_template`, `_rounded_triangle_normalized`. | Replace with analytic piecewise rounded-corner template and diagnostics. |
| Support/Provenance/Consistency | Main UI should show Korean summary; verbose details only in Debug. | Some details are under Debug, but main captions still use Physical Target / Support Reference / Predicted Output English terms. | PARTIAL | `app_ui_snapshot.py` captions after finite review. | Move/rename those captions into Debug and show concise Korean summary in main flow. |
| Startup Compensation Review | Should not appear as duplicate main workflow section. | Finite path places it in Debug; non-finite path still calls it directly. | PARTIAL | `render_startup_compensation_review(compensation, command_profile)` in steady/non-finite branch. | Keep in Advanced/Legacy only. |

## Fix Pass Evidence

Local verification after the hotfix:

- Runtime screenshot: `outputs/runtime_evidence/pr61_ui_cleanup_runtime_8505.png`
- Runtime text checks: `outputs/runtime_evidence/pr61_ui_cleanup_runtime_8505_checks.json`
- Full test suite: `477 passed, 217 warnings`

Observed startup/runtime checks:

- `100mT pp fixed`: absent
- `100pp fixed`: absent
- `Target metric fixed`: absent
- `Support Family Selection`: absent
- `Support Reference Provenance`: absent
- `Command Prediction Consistency`: absent
- Legacy hardware terms remain only inside collapsed `Advanced / Legacy` diagnostics.
