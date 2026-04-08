# Coil Analyzing Master Notes

## Baseline
- Date: 2026-04-08
- Workspace: `D:\programs\Codex\Coil Analyzing`
- Product: Streamlit-based coil large-signal analyzer for CSV/XLSX waveform imports, calibration, electrical/magnetic analysis, LCR comparison, and export packaging

## Current State
- Core app entrypoint is `app.py`.
- Source modules are organized under `src/coil_analyzer`.
- Runtime workspace state exists under `.coil_analyzer`.
- Local manifest currently tracks 7 uploaded CSV datasets at 0.25 Hz, 0.5 Hz, 1 Hz, 1.25 Hz, 2 Hz, 3 Hz, and 4 Hz.
- Default request board tracks 0.25 Hz through 5 Hz with target `20.0 App`.
- Tests pass: `python -m pytest -q` -> `6 passed`.
- Syntax check passes: `python -m py_compile app.py`.

## Architecture
- `app.py`: Streamlit UI, page routing, session state, workflow orchestration
- `src/coil_analyzer/io`: CSV/XLSX loading, reference workbook loading, workspace persistence
- `src/coil_analyzer/preprocessing`: time-axis parsing, scaling, inversion, delay correction, channel standardization
- `src/coil_analyzer/analysis`: frequency estimation, sine-fit fundamentals, impedance/magnetic/lambda/gain metrics
- `src/coil_analyzer/plotting`: waveform, phasor, loop, heatmap, and comparison figures
- `src/coil_analyzer/export`: Excel workbook + HTML figure + JSON settings bundle packaging
- `tests`: regression checks for dataset standardization, datetime parsing, CSV fallback parsing, impedance analysis, and lambda metrics

## Workflow
1. Import waveform CSV/XLSX or use example data.
2. Infer or assign time, voltage, current, and magnetic channels.
3. Apply scale, offset, polarity, unit, and delay corrections.
4. Standardize signals into `time_s`, `voltage_v`, `current_a`, and `magnetic_*`.
5. Estimate frequency and analyze a selected cycle window.
6. Derive electrical, magnetic, advanced, and gain metrics.
7. Compare measured results against reference workbook data when available.
8. Export results as Excel, HTML figures, and JSON settings bundle.

## Live Workspace Observations
- Uploaded datasets in `.coil_analyzer/uploads` appear to use `Timestamp`, `Voltage1`, `Current1_A`, `HallBx`, `HallBy`, and `HallBz`.
- Time mapping is already configured as `datetime` for the loaded datasets.
- Manifest shows datasets are loaded but not yet marked as analyzed.
- `TASK_BOARD.md` previously claimed the workspace had no code and no git repository; that was stale and required correction.

## Git / GitHub
- `git.exe` is available at `C:\Program Files\Git\cmd\git.exe`, but the shell PATH does not expose `git` directly.
- The workspace was not initialized as a git repository when inspected.
- No installed GitHub repository matching `Coil Analyzing` was found through the GitHub connector.
- Runtime artifacts under `.coil_analyzer` should stay out of version control.

## Notion
- Notion authentication is available for the current user.
- No obvious existing Notion page dedicated to `Coil Analyzing` was found from a direct workspace search.
- A destination page or database still needs to be chosen before syncing structured project notes there.

## Operating Rules
- Treat source-of-truth code as `app.py`, `src/`, `tests/`, `README.md`, and project docs in the repo root.
- Treat `.coil_analyzer/` as runtime state, uploaded measurements, and generated artifacts.
- Keep task tracking aligned with actual repo state before making GitHub or Notion updates.
- Prefer updating this file and `TASK_BOARD.md` when project flow or operating assumptions change.
