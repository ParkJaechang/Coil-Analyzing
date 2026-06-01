# Windows App Core Dependency

Last updated: 2026-06-01 KST

## Streamlit/Core Repo

- Reference repo: `ParkJaechang/Coil-Analyzing`
- Local reference path inspected: `D:\programs\Codex\Coil Analyzing`
- Reference branch: `main`
- Reference commit SHA: `f55fc878ac5d669fe3f0c1481ce8851fb0110de6`
- Reference working tree status: clean

## Windows App / PR Checkout

- PR checkout path inspected: `D:\programs\Codex\Coil Analyzing_clean`
- PR URL: https://github.com/ParkJaechang/Coil-Analyzing/pull/61
- PR branch: `codex/finite-feedback-cycle-policy-backend`
- Implementation audit target SHA: `8f8a881b040429a4d6451f15831602d2e76added`
- CI status for implementation audit target: PASS.
- Latest exact PR head may be a later documentation-only commit; see the newest `WinApp Implementation Status` PR comment for the exact head SHA.

## Dependency Method

- No git submodule is configured.
- No package pin for a separate Streamlit/core dependency was found in `requirements.txt`.
- The current dependency record is documentation-based: Windows App work must reference Streamlit/core repo SHA `f55fc878ac5d669fe3f0c1481ce8851fb0110de6`.

## Windows App Adapter Audit

Result: documentation only / implementation not present.

- MISSING: `src/coil_win_app/core_adapter.py`
- MISSING: `src/coil_win_app/main.py`
- MISSING: `src/coil_win_app/project_state.py`
- MISSING: `src/coil_win_app/ui/`

Because `src/coil_win_app/core_adapter.py` does not exist, the Windows App core adapter cannot be imported and cannot be considered implemented.

## Existing Field-Analysis Imports

These existing field-analysis modules imported without Streamlit in the previous PR-manager smoke check with `PYTHONPATH=src`:

- `field_analysis.final_modeled_lut`
- `field_analysis.finite_first_phase_sync`
- `field_analysis.finite_second_modeling`
- `field_analysis.continuous_steady_state_schema`
- `field_analysis.continuous_first_modeling`
- `field_analysis.voltage_policy`

This does not replace the missing Windows App core adapter.

## Required Contract

- Windows App adapter must import without Streamlit.
- Windows App adapter must not vendor-copy Streamlit UI code.
- Final LUT export contract must remain `sample_index,time_s,voltage_v`.
- Voltage limit/normalization policy must remain `+/-10V`.
- Harmonic inverse must not be used as the final export route.

## Upstream Requests

- Publish or document a stable non-Streamlit core API surface for Windows App use.
- Add a durable dependency pin, tag, or package reference for the Windows App to consume.
- Document any future algorithm or policy change before updating the Windows App dependency SHA.

## Parity Test Status

- Windows App core adapter contract test: MISSING.
- Windows App no-Streamlit dependency test: MISSING.
- Windows App final LUT export contract test: MISSING.
- Runtime parity: PARTIAL / blocked because the Windows App skeleton is not present.
