# PR61 Acceptance Inventory

PR management document. This is not a generated runtime artifact and does not change modeling behavior.

- Repository: `ParkJaechang/Coil-Analyzing`
- PR: [#61 Core Quick LUT 1.0/1.5-cycle workflow stabilization](https://github.com/ParkJaechang/Coil-Analyzing/pull/61)
- Branch: `codex/finite-feedback-cycle-policy-backend`
- Base: `codex/finite-actual-drive-second-correction`
- Checked head: `52dce561cf85c071b80ec2244bd96b1041b28e70`
- GitHub CI at check time: pass x2
- Merge state at check time: `CLEAN`
- Runtime acceptance rule: source/tests are not final acceptance. User launched-runtime review is required before merge.

## Latest UI/Semantics/Modeling Cleanup Summary

| Status | Count |
|---|---:|
| PASS | 9 |
| PARTIAL | 2 |
| FAIL | 0 |
| NOT VERIFIED | 0 |

## Focused Inventory

| ID | Requirement | Current status | Evidence files | UI location | Test files | Runtime evidence | Remaining work | Owner recommendation |
|---:|---|---|---|---|---|---|---|---|
| 1 | Remove `100mT pp fixed`, `100pp fixed`, `목표 bz_mT PP` wording | PASS | `src/field_analysis/app_ui_snapshot.py` | Quick LUT target captions / summaries | `tests/test_target_semantics_ui_contract.py`, `tests/test_quick_lut_ui_contract.py` | Not verified | Exact prohibited strings are absent from source-level UI contract; user should still report any runtime copy that appears stale | A: no code unless runtime shows stale copy |
| 2 | Separate target shape / target peak / +/-50mT normalization | PARTIAL | `src/field_analysis/app_ui_snapshot.py`, `src/field_analysis/quick_lut_target_config.py`, `src/field_analysis/ui_quick_lut_feedback.py` | Quick LUT target summary / finite feedback review | `tests/test_target_semantics_ui_contract.py`, `tests/test_quick_lut_target_config.py`, `tests/test_simplified_user_workflow_ui_contract.py` | Not verified | Shape and normalization are separated in copy; target peak user setting still needs runtime confirmation for clarity | A: verify UI wording with user, then simplify if ambiguous |
| 3 | Remove legacy target-output / AMP / DAQ / extrapolation copy from main UI | PASS | `src/field_analysis/app_ui_snapshot.py` | Collapsed `Advanced / Legacy hardware calibration` / diagnostics only | `tests/test_simplified_user_workflow_ui_contract.py` | Source/runtime check pending latest screenshot | Exact legacy strings `DAQ 최대 Voltage PP`, `DC AMP gain`, `AMP output`, `target extrapolation` were removed from default-facing labels; remaining calibration controls are explicitly marked legacy diagnostics | Keep collapsed Advanced/Legacy only |
| 4 | Verify rounded triangle target template ripple removal | PASS | `src/field_analysis/target_templates.py`, `src/field_analysis/finite_first_phase_sync.py` | Target metadata / phase sync review | `tests/test_target_template_quality.py`, `tests/test_finite_first_phase_sync_modeling.py` | Not verified | Analytic test passes; user visual review still decides graph acceptance | B: no code unless user sees ripple in launched plot |
| 5 | First command plot shows only final first command in main view | PARTIAL | `src/field_analysis/app_ui_snapshot.py`, `src/field_analysis/ui_finite_first_phase_sync.py` | Quick LUT first command / phase sync review | `tests/test_finite_first_phase_sync_modeling.py`, `tests/test_quick_lut_ui_contract.py` | Not verified | Main command source is `limited_voltage_v`, but surrounding debug/hardware rows still make the screen noisy | A: keep one primary command plot; move diagnostics under Advanced expanders |
| 6 | Finite phase sync residual remains finite through active end | PASS | `src/field_analysis/finite_first_phase_sync.py`, `src/field_analysis/ui_finite_first_phase_sync.py` | Finite phase sync review | `tests/test_finite_first_phase_sync_modeling.py` | User screenshot showed truncation before this follow-up fix | Phase sync now uses dominant absolute peak with polarity, not first positive peak; positive delay requires post-active source support or blocks with `insufficient_phase_sync_support` | Re-test runtime plot after push |
| 7 | Measured field scale-to-50mT and correction gain/headroom visible | PASS | `src/field_analysis/finite_first_phase_sync.py`, `src/field_analysis/ui_finite_first_phase_sync.py`, `src/field_analysis/ui_second_modeling_cards.py` | Finite phase sync / second modeling metadata | `tests/test_finite_first_phase_sync_modeling.py` | Not verified | UI surfaces scale/headroom/clipping metadata; runtime visibility still needs user check | A/B: no code unless runtime panel lacks the fields |
| 8 | Support / Provenance / Consistency moved to Korean summary or Advanced | PASS | `src/field_analysis/app_ui_snapshot.py` | `데이터 선택 상세 / Debug` expander | Support/reference contract tests | Source evidence | Main summary remains Korean; old English headings were removed or localized under Debug (`데이터 선택 기준`, `참조 데이터 출처`, `전압 예측 일관성`, `finite 신호 일관성`) | Keep as Debug-only diagnostics |
| 9 | Startup Compensation Review disposition decided | PASS | `src/field_analysis/ui_startup_compensation_review.py`, `src/field_analysis/app_ui_snapshot.py` | `Advanced / Legacy startup compensation review` and finite Debug expander | `tests/test_startup_compensation_ui_contract.py` | Source evidence | Decision: keep only as Advanced/Legacy diagnostic. It is not part of the default Quick LUT production workflow and is renamed `startup 과도응답 진단 / Advanced Legacy`. | Do not merge into production residual review unless requested later |
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
- Runtime screenshot evidence is still required after the dominant-peak phase sync follow-up fix.
- User must verify in launched runtime that no stale text or old UI path is still visible.
