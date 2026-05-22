# PR61 Current Status

PR management status document. This document is meant for GitHub PR readers and does not change application behavior.

## PR

- Repository: `ParkJaechang/Coil-Analyzing`
- PR: [#61 Core Quick LUT 1.0/1.5-cycle workflow stabilization](https://github.com/ParkJaechang/Coil-Analyzing/pull/61)
- Branch: `codex/finite-feedback-cycle-policy-backend`
- Base: `codex/finite-actual-drive-second-correction`
- Checked remote head before docs update: `a5c72dc92d954cf0ff074e9e5100811553874846`
- Checked local head before docs update: `a5c72dc92d954cf0ff074e9e5100811553874846`
- PR state before docs update: open draft
- Merge state before docs update: `CLEAN`
- GitHub CI before docs update: pass x2
- Changed files before docs update: 97

## Local workspace note

At the time of this PR-manager update, the local worktree also had unrelated dirty files:

- docs/report files from previous report work
- cleanup audit markdown files
- uncommitted source changes in `src/field_analysis/app_ui_snapshot.py`, `src/field_analysis/quick_lut_target_config.py`, and `src/field_analysis/ui_continuous_steady_state.py`

This PR-manager update must not commit core source changes. Only `docs/pr61_*.md` status documents should be committed for this update.

## Current scope

PR61 is the draft integration branch for:

- Quick LUT workflow cleanup and button-gated modeling.
- Upload memory and cache restore.
- Target config source-of-truth for current/applied/result configs.
- Finite startup-aware first modeling.
- Finite 1.0 / 1.5 production cycle policy.
- Finite actual-drive review and second correction workflow.
- Finite tail policy and second command export.
- Continuous steady-state 1cycle extraction.
- Continuous first modeling.
- Continuous second modeling / validation where implemented.
- Continuous first/second final LUT export.
- Final LUT CSV contract using raw plotted/generated voltage samples, not Fourier/harmonic resynthesis.

## Current policy

- Finite production cycles: `1.0` and `1.5`.
- Finite `1.25`, `1.75`, `2.0`: review-only or unsupported for production correction/export.
- Continuous production cycle: stable `1cycle` only.
- Continuous tail / zero-return tail: off by default.
- Target shape: fixed rounded triangle.
- Field review/modeling normalization: +/-50mT.
- Command voltage limit/export basis: +/-5V.
- Hall field convention for actual-drive review: effective field = `-HallBz raw`.
- Final LUT export columns: `sample_index,time_s,voltage_v`.
- Final LUT export must not use Fourier/harmonic resynthesis.

## Acceptance summary

See [pr61_acceptance_inventory.md](./pr61_acceptance_inventory.md) for itemized status.

Current summary before this docs update:

- PASS: 8
- PARTIAL: 12
- FAIL: 3
- NOT VERIFIED: 1

## Main remaining blockers

- Current launched-runtime evidence is insufficient.
- Sidebar legacy uploaders are still default-visible in the source checked before this docs update.
- Some main Quick LUT copy still contains `fixed 100pp`, DAQ, AMP gain, and extrapolation wording.
- Startup Compensation Review is not fully resolved as keep/merge/hide/remove.
- Rounded-triangle target template ripple removal has not been verified with dedicated analytic/runtime evidence.
- Continuous extraction/modeling and finite 1Hz 1.0/1.5 target config flows require user runtime verification.

## Tests reported before this docs update

- GitHub CI: pass x2.
- Local targeted tests recently run by PR Manager: 70 passed.
- Local full tests recently run by PR Manager: 475 passed, 217 warnings.

## Merge decision

PR61 should remain draft.

Do not merge until:

- User launched-runtime review is complete.
- Remaining UI/semantics blockers are either fixed or explicitly accepted.
- Runtime checklist results are posted back to the PR.
