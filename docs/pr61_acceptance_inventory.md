# PR61 Acceptance Inventory

PR management document. This is not a generated runtime artifact and does not change modeling behavior.

- Repository: `ParkJaechang/Coil-Analyzing`
- PR: [#61 Core Quick LUT 1.0/1.5-cycle workflow stabilization](https://github.com/ParkJaechang/Coil-Analyzing/pull/61)
- Branch: `codex/finite-feedback-cycle-policy-backend`
- Base: `codex/finite-actual-drive-second-correction`
- Checked head: `4e8c250d9f9dbfe4978ebb192d56c9b8ffb06bd0`
- GitHub CI at check time: pass x2
- Merge state at check time: `CLEAN`
- Runtime acceptance rule: source/tests are not final acceptance. User launched-runtime review is required before merge.

## Latest UI/Semantics/Modeling Cleanup Summary

| Status | Count |
|---|---:|
| PASS | 6 |
| PARTIAL | 4 |
| FAIL | 1 |
| NOT VERIFIED | 0 |

## Focused Inventory

| ID | Requirement | Current status | Evidence files | UI location | Test files | Runtime evidence | Remaining work | Owner recommendation |
|---:|---|---|---|---|---|---|---|---|
| 1 | Remove `100mT pp fixed`, `100pp fixed`, `목표 bz_mT PP` wording | PASS | `src/field_analysis/app_ui_snapshot.py` | Quick LUT target captions / summaries | `tests/test_target_semantics_ui_contract.py`, `tests/test_quick_lut_ui_contract.py` | Not verified | Exact prohibited strings are absent from source-level UI contract; user should still report any runtime copy that appears stale | A: no code unless runtime shows stale copy |
| 2 | Separate target shape / target peak / +/-50mT normalization | PARTIAL | `src/field_analysis/app_ui_snapshot.py`, `src/field_analysis/quick_lut_target_config.py`, `src/field_analysis/ui_quick_lut_feedback.py` | Quick LUT target summary / finite feedback review | `tests/test_target_semantics_ui_contract.py`, `tests/test_quick_lut_target_config.py`, `tests/test_simplified_user_workflow_ui_contract.py` | Not verified | Shape and normalization are separated in copy; target peak user setting still needs runtime confirmation for clarity | A: verify UI wording with user, then simplify if ambiguous |
| 3 | Remove legacy target-output / AMP / DAQ / extrapolation copy from main UI | FAIL | `src/field_analysis/app_ui_snapshot.py` | Sidebar hardware settings and scalar/route result rows | `tests/test_simplified_user_workflow_ui_contract.py` | Source evidence only | `DAQ 최대 Voltage PP`, `DC AMP gain`, `AMP output`, `target extrapolation`, and related hardware language remain default-visible in source | A: move hardware/extrapolation controls and metrics to Advanced/Debug or remove from core Quick LUT path |
| 4 | Verify rounded triangle target template ripple removal | PASS | `src/field_analysis/target_templates.py`, `src/field_analysis/finite_first_phase_sync.py` | Target metadata / phase sync review | `tests/test_target_template_quality.py`, `tests/test_finite_first_phase_sync_modeling.py` | Not verified | Analytic test passes; user visual review still decides graph acceptance | B: no code unless user sees ripple in launched plot |
| 5 | First command plot shows only final first command in main view | PARTIAL | `src/field_analysis/app_ui_snapshot.py`, `src/field_analysis/ui_finite_first_phase_sync.py` | Quick LUT first command / phase sync review | `tests/test_finite_first_phase_sync_modeling.py`, `tests/test_quick_lut_ui_contract.py` | Not verified | Main command source is `limited_voltage_v`, but surrounding debug/hardware rows still make the screen noisy | A: keep one primary command plot; move diagnostics under Advanced expanders |
| 6 | Finite phase sync residual remains finite through active end | PASS | `src/field_analysis/finite_first_phase_sync.py`, `src/field_analysis/ui_finite_first_phase_sync.py` | Finite phase sync review | `tests/test_finite_first_phase_sync_modeling.py` | Not verified | Source/test metadata covers `active_residual_finite_through_end`; user still needs runtime plot check | B: provide runtime CSV/plot only if user reports truncation |
| 7 | Measured field scale-to-50mT and correction gain/headroom visible | PASS | `src/field_analysis/finite_first_phase_sync.py`, `src/field_analysis/ui_finite_first_phase_sync.py`, `src/field_analysis/ui_second_modeling_cards.py` | Finite phase sync / second modeling metadata | `tests/test_finite_first_phase_sync_modeling.py` | Not verified | UI surfaces scale/headroom/clipping metadata; runtime visibility still needs user check | A/B: no code unless runtime panel lacks the fields |
| 8 | Support / Provenance / Consistency moved to Korean summary or Advanced | PARTIAL | `src/field_analysis/app_ui_snapshot.py` | Debug / data selection detail expanders | Support/reference contract tests | Not verified | Much is behind expanders, but some English/internal labels remain (`Support Reference`, `Finite Signal Consistency`, provenance rows) | A: localize main summary and keep internal rows Advanced-only |
| 9 | Startup Compensation Review disposition decided | PARTIAL | `src/field_analysis/ui_startup_compensation_review.py`, `src/field_analysis/app_ui_snapshot.py` | Finite Debug expander and non-finite/scalar unavailable paths | `tests/test_startup_compensation_ui_contract.py` | Not verified | It is still present as a separate review component; keep/merge/hide/remove policy is not fully closed | Head/A/B: decide whether to merge into residual review or keep as Advanced diagnostics |
| 10 | Runtime evidence packet exists for current cleanup | PARTIAL | PR comments/docs only | User-launched app required | N/A | Insufficient current launched-runtime evidence | CI/source checks are available, but no current screenshot/CSV packet for these exact UI cleanup items | User: run checklist and report visible copy/plots |
| 11 | GitHub CI status | PASS | GitHub Actions | PR checks | `gh pr checks 61` | CI pass x2 | Continue to re-check after docs commit | PR Manager: update CI comment after push |

## Targeted checks run by PR Manager

```powershell
python -m pytest -q tests/test_target_semantics_ui_contract.py tests/test_target_template_quality.py tests/test_simplified_user_workflow_ui_contract.py tests/test_finite_first_phase_sync_modeling.py tests/test_quick_lut_ui_contract.py
```

Result: `22 passed`.

GitHub CI at source check time: `test` pass x2.

## Merge blockers

- PR remains draft.
- Current launched-runtime evidence for the latest UI cleanup is insufficient.
- Main UI still contains default-visible hardware/legacy language around DAQ, AMP, gain, and extrapolation.
- Support/Provenance/Consistency and Startup Compensation Review need final UX disposition before merge or explicit user acceptance as Advanced diagnostics.
- User must verify in launched runtime that no stale text or old UI path is still visible.
