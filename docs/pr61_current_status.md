# PR61 Current Status

This document is a Streamlit / Quick LUT status snapshot for PR61. It does not change application behavior.

## PR

- Repository: `ParkJaechang/Coil-Analyzing`
- PR: [#61 Core Quick LUT 1.0/1.5-cycle workflow stabilization](https://github.com/ParkJaechang/Coil-Analyzing/pull/61)
- Branch: `codex/finite-feedback-cycle-policy-backend`
- Checked head at this status pass: `70263b58d446160b46ffc9196163736be3d7e020`
- PR state: open draft
- Source of truth for current user/modeling policy: [pr61_user_feedback_resolution_log.md](./pr61_user_feedback_resolution_log.md)

## PR61 Acceptance Boundary

PR61 acceptance is limited to the Streamlit WebApp / Quick LUT / `src.field_analysis` runtime behavior.

The following work is tracked separately and must not be used as PR61 Streamlit acceptance evidence:

- WinApp repository work.
- AI/RL modeling app repository work.
- Cross-repo implementation notes, report drafts, or generated deliverables.

Cross-repo notes in this repository are historical/reference material only unless a later PR explicitly wires them into the Streamlit Quick LUT runtime.

## Current Policy

- Target field shape: `fixed_rounded_triangle`.
- Target peak field: user-configured value.
- Field normalization follows the user target peak field.
- Measured field normalization is scale-only against the target peak; do not confuse amplitude scaling with offset shifting.
- HallBz convention: effective field = `-HallBz raw`.
- Command voltage limit / normalization policy: +/-10V.
- Finite production cycles: `1.0` and `1.5`.
- Continuous production: steady-state loop-safe `1cycle` only.
- Final LUT export columns: `sample_index,time_s,voltage_v` only.
- Fourier/harmonic resynthesis is not used for final export.
- Heavy calculations remain button-triggered.

## Production Import Boundary

- `src/field_analysis/ai_sweep/*` is experimental/offline sweep planning only.
- Production Streamlit / Quick LUT runtime must not import `field_analysis.ai_sweep` by default.
- WinApp modules must not be imported by PR61 Streamlit runtime.
- Final LUT export continues to use generated voltage samples only and keeps the three-column CSV contract.

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
