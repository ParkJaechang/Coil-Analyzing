# PR61 Acceptance Inventory

PR management document. This is not a generated runtime artifact and does not change modeling behavior.

- Repository: `ParkJaechang/Coil-Analyzing`
- PR: [#61 Core Quick LUT 1.0/1.5-cycle workflow stabilization](https://github.com/ParkJaechang/Coil-Analyzing/pull/61)
- Branch: `codex/finite-feedback-cycle-policy-backend`
- Base: `codex/finite-actual-drive-second-correction`
- Checked local head before docs update: `a5c72dc92d954cf0ff074e9e5100811553874846`
- Remote head before docs update: `a5c72dc92d954cf0ff074e9e5100811553874846`
- GitHub CI before docs update: pass x2
- Merge state before docs update: `CLEAN`
- Runtime acceptance rule: tests and source markers are not final acceptance. User launched-runtime review is required before merge.

## Summary

| Status | Count |
|---|---:|
| PASS | 8 |
| PARTIAL | 12 |
| FAIL | 3 |
| NOT VERIFIED | 1 |

## Inventory

| ID | Requirement | Current status | Evidence files | UI location | Test files | Runtime evidence | Remaining work | Owner recommendation |
|---:|---|---|---|---|---|---|---|---|
| 1 | Quick LUT main workflow cleanup | PARTIAL | `src/field_analysis/app_ui_snapshot.py`, `src/field_analysis/ui_voltage_lut_review.py` | Quick LUT / LUT Review / Data-Cache Status | `tests/test_quick_lut_ui_contract.py`, `tests/test_button_gated_calculation_contract.py` | Partial historical browser note only; no current launched-runtime packet | Main labels improved, but debug/gain/upload controls still leak into default flow | A: UI grouping and default-screen cleanup |
| 2 | Hide/remove manual actual-drive CSV upload / feedback run | PARTIAL | `src/field_analysis/ui_quick_lut_feedback.py`, `src/field_analysis/ui_continuous_steady_state.py` | Finite feedback section / Continuous actual-drive panel | `tests/test_quick_lut_feedback_correction_ui.py`, `tests/test_continuous_steady_state_quick_lut.py` | Not verified | Finite manual upload is under Legacy expander, but continuous uploader and sidebar uploaders remain visible | A: hide legacy uploaders behind Advanced/Legacy |
| 3 | 2nd folder / upload memory actual-drive flow | PASS | `src/field_analysis/ui_quick_lut_feedback.py`, `src/field_analysis/ui_quick_lut_feedback_second_sources.py` | `2차 보정 입력 source` | `tests/test_quick_lut_feedback_correction_ui.py` | Not verified on current head | Needs launched-runtime confirmation with real `uploads/2nd` files | A/B: runtime packet only unless broken |
| 4 | Remove `100mT pp fixed` / `100pp fixed` user-facing wording | PARTIAL | `src/field_analysis/app_ui_snapshot.py` | Quick LUT captions and debug rows | `tests/test_quick_lut_ui_contract.py`, `tests/test_target_semantics_ui_contract.py` | Not verified | Exact `100mT pp fixed` appears removed, but `fixed 100pp rounded-triangle field target` remains in main Quick LUT caption | A: remove/reword remaining 100pp copy |
| 5 | Separate target shape / target peak / normalization semantics | PARTIAL | `src/field_analysis/app_ui_snapshot.py`, `src/field_analysis/quick_lut_target_config.py` | Quick LUT target summary | `tests/test_quick_lut_target_config.py`, `tests/test_target_semantics_ui_contract.py` | Not verified | UI displays separation, but target peak user input is not clearly exposed; config defaults to 50mT | A/B: expose user target peak or explicitly mark fixed normalized modeling peak |
| 6 | Remove Bz_mT legacy extrapolation/gain/DAQ wording from main UI | FAIL | `src/field_analysis/app_ui_snapshot.py` | Sidebar hardware settings and Quick LUT result metrics | `tests/test_simplified_user_workflow_ui_contract.py` | Source evidence shows remaining default-visible copy | `DAQ 최대 Voltage PP`, `DC AMP gain`, `target extrapolation`, required gain metrics still visible in default flow | A: move hardware/gain/extrapolation to Advanced/Debug or reword |
| 7 | Reduce/localize Support/Provenance/Consistency UI | PARTIAL | `src/field_analysis/app_ui_snapshot.py` | `데이터 선택 상세 / Debug` expander | Support/reference contract tests | Not verified | Much of it moved to expander, but headings and some rows remain English/internal | A: Korean summary plus Advanced details |
| 8 | Decide Startup Compensation Review fate | PARTIAL | `src/field_analysis/ui_startup_compensation_review.py`, `src/field_analysis/app_ui_snapshot.py` | Debug expander in finite; direct call remains in non-finite/scalar paths | `tests/test_startup_compensation_ui_contract.py` | Not verified | Hidden in one path but not merged/renamed/removed; policy still unclear | Head/A/B: decide keep-vs-merge-vs-hide, then A implements |
| 9 | Sidebar legacy uploader cleanup | FAIL | `src/field_analysis/app_ui_snapshot.py`, `src/field_analysis/ui_upload_memory_management.py` | Sidebar | `tests/test_upload_memory_management_ui.py` | Source evidence only | Continuous, finite, validation, LCR file uploaders are still default-visible in sidebar | A: move under Advanced/Legacy or Data Import |
| 10 | Remove rounded-triangle target template ripple | NOT VERIFIED | `src/field_analysis/compensation.py`, target generation helpers | Target / plot output | Existing semantic tests only | No analytic/runtime ripple evidence | Needs dedicated analytic template/ripple test and visual packet | B: add analytic target template evidence; user performs visual check |
| 11 | Clean first modeling command main plot | PARTIAL | `src/field_analysis/app_ui_snapshot.py`, `src/field_analysis/ui_finite_first_phase_sync.py` | Quick LUT first command plot | `tests/test_finite_first_phase_sync_modeling.py` | Not verified | Main plot uses `limited_voltage_v`, but surrounding debug/gain rows still visible | A: keep one main final command graph, move diagnostics |
| 12 | Finite phase-aligned residual active-end support | PARTIAL | `src/field_analysis/finite_first_phase_sync.py`, `src/field_analysis/finite_second_modeling_active_support.py` | Finite phase sync review / second modeling | `tests/test_finite_first_phase_sync_modeling.py`, `tests/test_finite_second_modeling_active_support.py` | Not verified | Source/tests cover support blocking and finite residual, but no launched plot evidence | B: if user still sees truncation, provide runtime CSV/plot evidence |
| 13 | Measured field scale-to-50mT and gain reflection | PASS | `src/field_analysis/finite_first_phase_sync.py`, `src/field_analysis/ui_finite_first_phase_sync.py`, `src/field_analysis/ui_quick_lut_feedback.py` | Finite phase sync / feedback review | `tests/test_finite_first_phase_sync_modeling.py`, `tests/test_finite_actual_drive_review.py` | Not verified | Needs user confirmation in runtime panel | A/B: no code unless runtime disagrees |
| 14 | Continuous source waveform family selection | PASS | `src/field_analysis/app_ui_snapshot.py`, `src/field_analysis/continuous_candidate_discovery.py` | Continuous steady-state runtime | `tests/test_continuous_steady_state_quick_lut.py` | Not verified | Runtime source selection must be checked with upload memory data | A/B: runtime packet |
| 15 | Continuous terminal cycle guard | PASS | `src/field_analysis/continuous_phase_support.py`, `src/field_analysis/continuous_steady_state_extraction.py` | Continuous extraction | `tests/test_continuous_steady_state_extraction.py`, `tests/test_continuous_steady_state_quick_lut.py` | Not verified | Needs 1Hz/3Hz/5Hz visual evidence | B: runtime evidence if needed |
| 16 | Continuous first modeling actual wiring | PARTIAL | `src/field_analysis/app_ui_snapshot.py`, `src/field_analysis/continuous_first_modeling.py`, `src/field_analysis/ui_continuous_first_modeling.py` | Continuous 1차 modeling | `tests/test_continuous_steady_state_quick_lut.py` | Not verified | Source/tests show wiring, but no launched result evidence | B/A: verify runtime extraction -> modeling -> plot |
| 17 | Continuous first/second final LUT export | PASS | `src/field_analysis/ui_continuous_final_lut_export.py`, `src/field_analysis/ui_voltage_lut_review.py` | Continuous final LUT export / LUT Review | `tests/test_continuous_steady_state_quick_lut.py`, `tests/test_final_voltage_lut_export_review_ui.py` | Not verified | Needs user download check for first/second result | A: runtime validation only |
| 18 | Finite tail auto/on/off policy | PARTIAL | `src/field_analysis/finite_second_modeling_tail.py`, `src/field_analysis/ui_finite_tail_policy.py`, `src/field_analysis/ui_second_modeling.py` | Finite 2차 modeling controls | `tests/test_finite_second_modeling_tail.py`, `tests/test_finite_second_modeling_tail_controller.py` | Not verified | Code/tests exist; current runtime plot/export evidence missing | B/A: runtime packet |
| 19 | Finite tail threshold dynamic copy | PARTIAL | `src/field_analysis/ui_finite_tail_policy.py`, `src/field_analysis/finite_second_modeling_tail.py` | Finite tail policy UI | Tail tests | Not verified | Need user-facing copy check in launched app | A: polish copy if still confusing |
| 20 | 1Hz finite 1.0 / 1.5 modeling | PARTIAL | `src/field_analysis/quick_lut_target_config.py`, `src/field_analysis/finite_first_phase_sync.py`, `src/field_analysis/ui_second_modeling.py` | Quick LUT target/debug and first/second modeling | `tests/test_finite_cycle_selector_policy.py`, `tests/test_quick_lut_target_config.py`, `tests/test_finite_first_phase_sync_modeling.py` | Not verified | Needs launched 1Hz 1.0 and 1.5 test with config snapshot | B/A: runtime evidence |
| 21 | Target config source-of-truth | PASS | `src/field_analysis/quick_lut_target_config.py`, `src/field_analysis/ui_quick_lut_target_debug.py`, `src/field_analysis/app_ui_snapshot.py` | Quick LUT target/debug | `tests/test_quick_lut_target_config.py`, `tests/test_quick_lut_ui_contract.py` | Not verified | Runtime evidence still required for stale selection cases | B/A: runtime packet |
| 22 | Upload memory/cache restore | PASS | `src/field_analysis/upload_active_records.py`, `src/field_analysis/upload_manifest_normalization.py`, `src/field_analysis/ui_upload_state.py` | Sidebar memory / Data Cache Status | `tests/test_upload_memory_management.py`, `tests/test_upload_memory_management_ui.py` | Historical browser evidence only | Need current head runtime check if user reports missing data | A: no change unless runtime disagrees |
| 23 | Prevent return to initial screen after button click | PARTIAL | `src/field_analysis/app_ui_snapshot.py`, continuous runtime UI | Quick LUT / continuous panels | `tests/test_button_gated_calculation_contract.py`, `tests/test_continuous_steady_state_quick_lut.py` | Not verified | Session state markers exist; runtime flow still needs verification | A: fix only with reproducible runtime issue |
| 24 | Final LUT CSV contract: `sample_index,time_s,voltage_v` only | PASS | `src/field_analysis/ui_final_voltage_lut_export.py`, `src/field_analysis/ui_continuous_final_lut_export.py`, `src/field_analysis/final_modeled_lut.py` | Finite/continuous final LUT export | `tests/test_final_modeled_lut_export.py`, `tests/test_final_voltage_lut_export_review_ui.py`, `tests/test_continuous_steady_state_quick_lut.py` | Not verified | Needs user download check, but source/tests pass | A: runtime validation |

## Merge blockers

- Runtime evidence is insufficient for several user-visible workflows.
- Sidebar legacy uploaders remain visible in default flow.
- Main UI still includes some `100pp`, DAQ, AMP gain, and extrapolation language.
- Rounded-triangle target template ripple removal is not verified.
- PR remains draft until user launched-runtime review is complete.
