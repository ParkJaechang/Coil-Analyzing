# PR61 Current Status

PR management status document. This document is meant for GitHub PR readers and does not change application behavior.

## PR

- Repository: `ParkJaechang/Coil-Analyzing`
- PR: [#61 Core Quick LUT 1.0/1.5-cycle workflow stabilization](https://github.com/ParkJaechang/Coil-Analyzing/pull/61)
- Branch: `codex/finite-feedback-cycle-policy-backend`
- Base: `codex/finite-actual-drive-second-correction`
- Checked head: `4e8c250d9f9dbfe4978ebb192d56c9b8ffb06bd0`
- PR state: open draft
- Merge state: `CLEAN`
- GitHub CI at check time: pass x2
- Changed files at check time: 107

## Local workspace note

At this PR-manager update, the local worktree contains dirty docs/report/cleanup files unrelated to core code. No core source file is staged by this status update.

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
- Final LUT CSV contract using plotted/generated voltage samples, not Fourier/harmonic resynthesis.
- Latest UI/semantics cleanup around target wording, target normalization, first command plots, and phase-sync metadata.

## Current policy

- Finite production cycles: `1.0` and `1.5`.
- Finite `1.25`, `1.75`, `2.0`: review-only or unsupported for production correction/export.
- Continuous production cycle: stable `1cycle` only.
- Continuous tail / zero-return tail: off by default.
- Target shape: fixed rounded triangle.
- Target peak and internal normalization must be described separately.
- Field review/modeling normalization: +/-50mT.
- Command voltage limit/export basis: +/-5V.
- Hall field convention for actual-drive review: effective field = `-HallBz raw`.
- Final LUT export columns: `sample_index,time_s,voltage_v`.
- Final LUT export must not use Fourier/harmonic resynthesis.

## Latest acceptance summary

See [pr61_acceptance_inventory.md](./pr61_acceptance_inventory.md) for itemized status.

- PASS: 6
- PARTIAL: 4
- FAIL: 1
- NOT VERIFIED: 0

## Current pass items

- Prohibited target wording (`100mT pp fixed`, `100pp fixed`, `목표 bz_mT PP`) is absent from source-level UI contract.
- Rounded triangle target template has an analytic ripple quality test.
- Finite phase-sync residual metadata reports finite active-end support.
- Measured field scale-to-50mT and gain/headroom/clipping metadata are present in source/UI paths.
- GitHub CI is passing on PR head.

## Main remaining blockers

- Current launched-runtime evidence is insufficient for the latest UI/semantics cleanup.
- Main UI still contains default-visible hardware/legacy language around DAQ, AMP, gain, and extrapolation.
- Support/Provenance/Consistency cleanup is partial: internal/English details still appear in debug/detail areas.
- Startup Compensation Review still needs a final UX policy: keep as Advanced diagnostics, merge into residual review, hide, or remove.
- User must confirm in launched runtime that the first command plot, target semantics copy, and phase-sync metadata appear as intended.

## Tests checked by PR Manager for this status update

```powershell
python -m pytest -q tests/test_target_semantics_ui_contract.py tests/test_target_template_quality.py tests/test_simplified_user_workflow_ui_contract.py tests/test_finite_first_phase_sync_modeling.py tests/test_quick_lut_ui_contract.py
```

Result: `22 passed`.

GitHub CI at source check time:

- `test`: pass
- `test`: pass

## Merge decision

PR61 should remain draft.

Do not merge until:

- User launched-runtime review is complete.
- Remaining UI/semantics blockers are fixed or explicitly accepted.
- Runtime checklist results are posted back to the PR.
