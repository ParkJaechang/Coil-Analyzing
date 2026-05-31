# Windows App Core Dependency

Last updated: 2026-06-01 KST

## Streamlit/Core Repo

- Reference repo: `ParkJaechang/Coil-Analyzing`
- Local reference path inspected: `D:\programs\Codex\Coil Analyzing`
- Reference branch: `main`
- Reference commit SHA: `f55fc878ac5d669fe3f0c1481ce8851fb0110de6`
- Reference working tree status at audit: clean

## Windows App / PR Checkout

- PR checkout path inspected: `D:\programs\Codex\Coil Analyzing_clean`
- PR branch: `codex/finite-feedback-cycle-policy-backend`
- Current documented PR head SHA: `6858c9613fa4be1c2d805b81b5a63276882ec4aa`
- Previous implementation head SHA before PR-manager docs: `d3eed1f970ffe52e70568ed4d06da1908e564dde`
- PR URL: https://github.com/ParkJaechang/Coil-Analyzing/pull/61

## Dependency Method

- No git submodule was configured.
- No package pin for a separate Streamlit/core dependency was found in `requirements.txt`.
- The dependency is currently documented as a source/reference relationship against the Streamlit/core repo and commit above.
- This document is the current PR-visible record of the Streamlit/core reference SHA.

## Imported APIs

Core/non-UI imports verified without importing Streamlit:

- `field_analysis.final_modeled_lut`
- `field_analysis.finite_first_phase_sync`
- `field_analysis.finite_second_modeling`
- `field_analysis.continuous_steady_state_schema`
- `field_analysis.continuous_first_modeling`
- `field_analysis.voltage_policy`

UI modules intentionally depend on Streamlit and are not part of the core-adapter import surface.

## Placeholder APIs

- No separate placeholder package API was found for the Streamlit/core dependency.
- Current placeholder boundary is documentation/process based: Coder should not vendor-copy Streamlit/core changes into the Windows App path without an explicit dependency update plan.

## Upstream Requests

- Add a durable dependency record if Windows App and Streamlit/core are split into separate repositories or packages.
- Publish a stable core adapter interface that can be imported without Streamlit.
- Keep final LUT export contract stable as `sample_index,time_s,voltage_v`.
- Keep voltage policy stable as peak-based `±10V` normalization/limit unless explicitly approved.
- Keep harmonic inverse out of the final export route.

## Parity Test Status

- Core import smoke: PASS with `PYTHONPATH=src`.
- Final LUT schema policy: covered by existing tests and source contracts.
- Continuous schema adapter reject-final-LUT policy: covered by existing tests.
- Full Windows runtime parity: PARTIAL, pending user-launched checklist and packaging smoke test.

## Streamlit Repo Isolation

- `D:\programs\Codex\Coil Analyzing` was inspected as the Streamlit/core reference checkout.
- It remained on `main` at `f55fc878ac5d669fe3f0c1481ce8851fb0110de6`.
- `git status -sb` reported no local changes.
- No commit or push was made to the Streamlit/core reference repo during this PR-manager pass.
