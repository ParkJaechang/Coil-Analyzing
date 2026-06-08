# PR61 Acceptance Inventory

PR management document. This is not a generated runtime artifact and does not change modeling behavior.

- Repository: `ParkJaechang/Coil-Analyzing`
- PR: [#61 Core Quick LUT 1.0/1.5-cycle workflow stabilization](https://github.com/ParkJaechang/Coil-Analyzing/pull/61)
- Branch: `codex/finite-feedback-cycle-policy-backend`
- Checked head at this inventory pass: `70263b58d446160b46ffc9196163736be3d7e020`
- Runtime acceptance rule: source/tests are not final acceptance. User launched-runtime review is required before merge.
- Current policy source of truth: [pr61_user_feedback_resolution_log.md](./pr61_user_feedback_resolution_log.md)
- Cleanup inventory: [pr61_cleanup_inventory.md](./pr61_cleanup_inventory.md)

## PR61 Acceptance Boundary

PR61 acceptance is only for Streamlit WebApp / Quick LUT / `src.field_analysis` behavior.

| external track | PR61 acceptance status |
|---|---|
| WinApp repository work | Separate repository/thread. Do not use as PR61 acceptance evidence. |
| AI/RL modeling app repository work | Separate repository/thread. Do not use as PR61 production acceptance evidence. |
| Cross-repo notes/reports | Historical/reference only unless explicitly wired in a reviewed future PR. |

## Latest Policy Snapshot

- Target shape: `fixed_rounded_triangle`.
- Target peak field: user setting.
- Field normalization follows the user target peak field.
- Measured field normalization is scale-only against the target peak; offset shifting is not normalization.
- HallBz convention: effective field = `-HallBz raw`.
- Command voltage limit / normalization policy: +/-10V.
- Final LUT export columns: `sample_index,time_s,voltage_v` only.
- Fourier/harmonic resynthesis is not used for final export.
- Heavy calculations remain button-triggered.

## Focused Inventory

| ID | Requirement | Current status | Evidence files | Runtime evidence | Remaining work |
|---:|---|---|---|---|---|
| 1 | Remove prohibited fixed-PP target wording | PASS | `tests/test_target_semantics_ui_contract.py` | User runtime still required | Report any stale launched-session copy. |
| 2 | Separate target shape / target peak / normalization | PARTIAL | `quick_lut_target_config.py`, UI contracts | User runtime still required | Confirm target peak UI clarity. |
| 3 | Remove legacy target-output / AMP / DAQ / extrapolation from main flow | PASS/PARTIAL | `app_ui_snapshot.py`, simplified workflow tests | User runtime still required | Keep only Advanced/Legacy diagnostics. |
| 4 | Rounded triangle target template quality | PASS | `target_templates.py`, `tests/test_target_template_quality.py` | Plot evidence still user-reviewed | Keep analytic template as source of truth. |
| 5 | First command plot clarity | PARTIAL | `ui_finite_first_phase_sync.py` | User runtime still required | Continue reducing noisy debug rows. |
| 6 | Finite phase-sync residual active-end support | PASS | finite phase sync tests | User runtime still required | Re-test screenshots after rerun. |
| 7 | Measured field scale / gain / headroom metadata | PASS | `ui_modeling_error_summary.py`, phase sync metadata | User runtime still required | Keep concise UI summary. |
| 8 | Support/Provenance/Consistency cleanup | PARTIAL | Debug expanders | User runtime still required | Keep detailed internals under Debug only. |
| 9 | Startup Compensation Review policy | PASS | `ui_startup_compensation_review.py` | Not part of default flow | Decision: `keep_advanced_only`. |
| 10 | Continuous final export finite-like contract | PASS | continuous export tests | User runtime still required | Confirm CSV download path. |
| 11 | WinApp / AI-RL cross-repo separation | PASS | PR docs and cleanup inventory | N/A | Keep separate from PR61 acceptance. |
| 12 | `ai_sweep` production isolation | PASS | `src/field_analysis/ai_sweep`, ai_sweep tests | N/A | Do not wire into Quick LUT production without reviewed future PR. |
| 13 | CI status | NOT VERIFIED | GitHub Actions | N/A | Re-run after latest push. |

## Merge Blockers

- PR remains draft.
- CI must be green after the latest push.
- User launched-runtime evidence is still required for final acceptance.
- WinApp and AI/RL app work must remain separated from this Streamlit PR acceptance scope.
