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

## Follow-up Evidence At `52dce561cf85c071b80ec2244bd96b1041b28e70`

The user runtime screenshot after the previous cleanup still showed finite first phase sync using the wrong alignment reference and cutting the aligned measured trace near the active end. The follow-up fix changes finite first phase sync from "first positive peak" to "dominant absolute peak with polarity"; if the measured dominant peak is negative, the voltage negative peak is used as the sync reference.

| item | expected | current result | status | evidence |
|---|---|---|---|---|
| Legacy DAQ/AMP/extrapolation copy | Not default-visible in Quick LUT main flow. | Exact prohibited labels are removed from default labels; remaining calibration controls are under collapsed `Advanced / Legacy` diagnostics and explicitly marked legacy. | PASS | `tests/test_target_semantics_ui_contract.py`, runtime text capture `outputs/runtime_evidence/pr61_runtime_text_8504.txt` |
| Startup Compensation Review policy | Not part of default production workflow. | Kept only as Advanced/Legacy diagnostic and renamed `startup 과도응답 진단 / Advanced Legacy`. | PASS | `docs/pr61_acceptance_inventory.md`, `tests/test_startup_compensation_ui_contract.py` |
| Support/Provenance/Consistency | Korean summary in main flow; verbose internals Debug-only. | English main heading `Finite Signal Consistency` removed; debug heading is localized. | PASS | `tests/test_finite_signal_status_ui_contract.py` |
| Finite phase sync peak reference | Align by dominant peak, including negative peak. | Synthetic runtime-path evidence uses `phase_sync_peak_polarity=negative`, voltage negative peak at `0.7498s`, measured negative peak at `0.8397s`. | PASS | `outputs/runtime_evidence/pr61_phase_sync_negative_peak_evidence.json` |
| Finite active-end support | Aligned measured/residual finite through active end; no silent zero-fill. | `required_phase_aligned_source_end_s=1.0943`, `actual_source_time_end_s=1.1378`, `active_residual_finite_ratio=1.0`, `active_end_kink_detected=false`. | PASS | `outputs/runtime_evidence/pr61_phase_sync_negative_peak_evidence.json` |
| Target template quality | Analytic fixed rounded triangle and ripple check available. | Source-level quality tests pass; runtime target template diagnostic is exposed in finite first review expander. | PASS | `tests/test_target_template_quality.py`, `ui_finite_first_phase_sync.py` |

Headless Selenium could capture the app shell on `8504`, but it starts a fresh Streamlit browser session and did not inherit the user's loaded modeling panel state. User-launched runtime review is still required for the exact screenshot path after the pushed fix.

## Follow-up After User Recheck

The user-launched screenshot still showed the same cutoff. Root cause: finite first phase sync was still building smoothed/aligned measured traces from the active-only `command_profile` grid. The compensation route already preserved full native measured support in `selected_support_source_time_s` / `selected_support_source_mT`, but the phase-sync kernel ignored it.

Fix added after this report:

- `apply_finite_first_phase_sync_modeling()` now uses `selected_support_source_time_s` / `selected_support_source_mT` as the native measurement support grid when available.
- Smoothing is performed on the native support grid, not on the active output command grid.
- Aligned measured is sampled back onto the output grid using `native_time + phase_delay`.
- Metadata now marks `measurement_support_grid_separate_from_output_grid=True` and `measurement_support_source=selected_support_source_native`.
- If native support is unavailable or insufficient, the UI stops before plotting a misleading partial phase-sync result and shows a support warning.
- Regression test `test_finite_first_phase_sync_uses_native_support_beyond_active_output_grid` covers the exact active-only-output + longer-native-support case.
