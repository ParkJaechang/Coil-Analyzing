# Windows App Current Status

Last updated: 2026-06-01 KST

## PR

- PR URL: https://github.com/ParkJaechang/Coil-Analyzing/pull/61
- PR number: 61
- State: OPEN
- Draft: yes
- Branch: `codex/finite-feedback-cycle-policy-backend`
- Base branch: `codex/finite-actual-drive-second-correction`
- Head SHA at audit start: `d3eed1f970ffe52e70568ed4d06da1908e564dde`
- CI status at audit start: GitHub Actions CI succeeded for `d3eed1f970ffe52e70568ed4d06da1908e564dde`

## Current Scope

- Quick LUT-centered finite and continuous workflow stabilization.
- Finite 1.0-cycle and 1.5-cycle production path.
- Finite first modeling, actual-drive review, finite second modeling, tail policy, and final LUT export.
- Continuous steady-state one-cycle extraction, continuous first modeling, continuous second modeling route, and final LUT export.
- UI copy/policy cleanup around target shape, target peak, field normalization, voltage normalization/limit, and final export contract.

## Implemented Pages

- Quick LUT workflow and target configuration.
- Raw Waveforms / actual-drive review.
- Finite first phase-sync modeling.
- Finite second modeling.
- Continuous steady-state extraction.
- Continuous first modeling.
- Continuous final LUT export.
- Final voltage LUT export/review.
- Upload memory/cache management.

## Core Adapter Status

- Core modules import without Streamlit when loaded with `PYTHONPATH=src`.
- Verified modules:
  - `field_analysis.final_modeled_lut`
  - `field_analysis.finite_first_phase_sync`
  - `field_analysis.finite_second_modeling`
  - `field_analysis.continuous_steady_state_schema`
  - `field_analysis.continuous_first_modeling`
  - `field_analysis.voltage_policy`
- UI modules still intentionally import Streamlit.
- Continuous schema adapter rejects final voltage LUT-shaped CSV input as a measurement source.

## Policy Status

- Target peak and normalization are documented as separate concepts.
- Field review/modeling normalization is target-peak based and uses the 50 mT normalized display/fit convention where applicable.
- Command voltage normalization/limit policy is `±10V`.
- Final LUT export contract is exactly `sample_index,time_s,voltage_v`.
- Harmonic inverse output is not allowed as the final export route; final export must use the modeled/limited command profile.

## Known Limitations

- User-launched Windows/runtime evidence is still incomplete for this exact HEAD.
- PR is still draft.
- Some previously modified source/test files are already present in the working tree outside this PR-manager documentation pass.
- The dependency pin to the Streamlit/core reference was not found in `requirements.txt`, `pyproject.toml`, or a submodule before this documentation update.

## Merge Blockers

- Complete user runtime checklist on the Windows App path.
- Confirm packaging smoke test result.
- Confirm GitHub Actions for the final pushed documentation HEAD.
- Decide whether remaining UI/semantics limitations are accepted or require a Coder follow-up.
- Keep generated artifacts, local state, upload/export caches, and user measurement data out of commits.
