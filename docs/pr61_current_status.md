# PR61 Current Status

This document is a status snapshot for PR61. It does not change application behavior.

## PR

- Repository: `ParkJaechang/Coil-Analyzing`
- PR: [#61 Core Quick LUT 1.0/1.5-cycle workflow stabilization](https://github.com/ParkJaechang/Coil-Analyzing/pull/61)
- Branch: `codex/finite-feedback-cycle-policy-backend`
- Checked head: `910cf096a80badc0debfaf96b3f179d8997d491d`
- PR state: open draft
- Merge state at check time: `UNSTABLE`
- GitHub CI at check time: failing before this cleanup pass
- Source of truth for current user/modeling policy: [pr61_user_feedback_resolution_log.md](./pr61_user_feedback_resolution_log.md)

## Current Policy

- Target field shape: `fixed_rounded_triangle`.
- Target peak field: user-configured value.
- Field normalization follows the user target peak field, not a fixed +/-50mT production policy.
- Measured field normalization is scale-only against the target peak; do not confuse amplitude scaling with offset shifting.
- HallBz convention: effective field = `-HallBz raw`.
- Command voltage limit / normalization policy: +/-10V.
- Finite production cycles: `1.0` and `1.5`.
- Continuous production: steady-state loop-safe `1cycle` only.
- Final LUT export columns: `sample_index,time_s,voltage_v` only.
- Fourier/harmonic resynthesis is not used for final export.
- Heavy calculations remain button-triggered.

## CI Failure Root Cause Before This Cleanup Pass

- `tests/test_file_size_guardrails.py`: PR61 bridge/stabilization modules exceeded the 600-line guardrail and were missing from the temporary oversized allowlist.
- `tests/test_finite_actual_drive_response.py`: stale expectation still required HallBz sign auto-selection, while current policy fixes effective field to `-HallBz raw`.
- `tests/test_finite_second_modeling_tail.py`: stale expectation still required default-visible second-modeling tail UI controls, while current UI policy hides those controls from the main 2nd command flow.

## Cleanup Status

- Cleanup inventory: [pr61_cleanup_inventory.md](./pr61_cleanup_inventory.md)
- Legacy hardware calibration: keep Advanced/Legacy only; do not expose as primary modeling input.
- Startup Compensation Review: classify as `keep_advanced_only` until it is explicitly merged into finite residual review or archived.
- Duplicate/stale status docs: archive candidates; do not delete in this pass.
- Source/test files: keep; do not delete production bridges during cleanup.

## Merge Decision

Keep PR61 as draft until:

- GitHub CI is green at the latest head.
- User runtime confirms 1st/2nd modeling UI and export behavior.
- Stale PR body/status docs no longer conflict with `pr61_user_feedback_resolution_log.md`.
