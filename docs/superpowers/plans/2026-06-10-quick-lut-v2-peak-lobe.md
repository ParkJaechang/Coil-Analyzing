# Quick LUT v2 Peak-Lobe Modeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the current Quick LUT app as legacy, build a smaller Streamlit v2 shell, and move finite 1st modeling toward phase-synced peak-lobe gain control for 1.0 and 1.5 cycle commands.

**Architecture:** Keep the legacy app entrypoint separate from the v2 app entrypoint. Put new modeling behavior in focused core modules first, then wire it into the v2 Streamlit shell once tests prove the peak-lobe command contract.

**Tech Stack:** Python, Streamlit, pandas, numpy, Plotly, pytest.

---

### Task 1: Preserve Legacy And Add v2 Entrypoint

**Files:**
- Create: `app_field_analysis_quick_legacy.py`
- Create: `app_field_analysis_quick_v2.py`
- Create: `src/field_analysis/app_ui_v2.py`
- Create: `launch_quick_lut_legacy_local.cmd`
- Create: `launch_quick_lut_v2_local.cmd`
- Test: `tests/test_quick_lut_v2_entrypoints.py`

- [x] **Step 1: Write the failing entrypoint contract test**

Run: `.\\.venv\\Scripts\\python.exe -m pytest -q tests/test_quick_lut_v2_entrypoints.py`

Expected before implementation: FAIL because the legacy/v2 entrypoints and launchers do not exist.

- [x] **Step 2: Add explicit legacy and v2 entrypoints**

`app_field_analysis_quick_legacy.py` keeps importing `run_quick_lut_app` from `field_analysis.app_ui` / `field_analysis.app_ui_snapshot`.

`app_field_analysis_quick_v2.py` imports only `run_quick_lut_v2_app` from `field_analysis.app_ui_v2`.

- [x] **Step 3: Add the first v2 Streamlit shell**

`src/field_analysis/app_ui_v2.py` exposes `run_quick_lut_v2_app()`, an ordered workflow contract, and a first-model policy table documenting 1.0-cycle two-peak and 1.5-cycle three-peak modeling.

- [x] **Step 4: Verify entrypoints**

Run: `.\\.venv\\Scripts\\python.exe -m pytest -q tests/test_quick_lut_v2_entrypoints.py tests/test_entrypoint_import_path_contract.py tests/test_repository_smoke.py`

Expected after implementation: PASS.

### Task 2: Add Peak-Lobe Gain Core

**Files:**
- Create: `src/field_analysis/finite_first_peak_lobe.py`
- Test: `tests/test_finite_first_peak_lobe_modeling.py`

- [x] **Step 1: Write the failing 1.5-cycle lobe gain test**

The test should construct an already phase-synced active waveform where the effective measured field peaks are `+40mT`, `-20mT`, and `+30mT`, the base voltage peaks are `+1V`, `-1V`, and `+1V`, and the target peak is `50mT`.

Expected lobe gain result:

```python
assert gains == pytest.approx([1.25, 2.5, 50.0 / 30.0], rel=0.02)
assert command_peaks == pytest.approx([1.25, -2.5, 50.0 / 30.0], rel=0.02)
```

- [x] **Step 2: Write the failing 1.0-cycle lobe gain test**

The test should construct two lobes and assert only `+peak1` and `-peak1` are used.

Expected lobe count:

```python
assert [lobe.polarity for lobe in result.lobes] == ["positive", "negative"]
```

- [x] **Step 3: Implement the helper**

Create a pure helper that accepts `time_s`, `target_field_mT`, `aligned_measured_field_mT`, `base_voltage_v`, `active_mask`, and `cycle_count`. It returns a lobe gain envelope, lobe metadata, `peak_lobe_base_voltage_v`, and `peak_lobe_predicted_field_mT`.

- [x] **Step 4: Verify the helper**

Run: `.\\.venv\\Scripts\\python.exe -m pytest -q tests/test_finite_first_peak_lobe_modeling.py`

Expected: PASS.

### Task 3: Wire Peak-Lobe Modeling Into Finite 1st Modeling

**Files:**
- Modify: `src/field_analysis/finite_first_phase_sync.py`
- Modify: `tests/test_finite_first_phase_sync_modeling.py`

- [ ] **Step 1: Add a failing integration test**

Use `apply_finite_first_phase_sync_modeling()` on a synthetic 1.5-cycle command profile and assert metadata includes:

```python
assert metadata["finite_first_modeling_peak_lobe_enabled"] is True
assert metadata["finite_first_peak_lobe_count"] == 3
assert metadata["finite_first_peak_lobe_cycle_policy"] == "1.5cycle_three_peak"
```

- [ ] **Step 2: Wire the helper after phase sync and before residual correction**

Use the current phase sync result as the source for peak detection. Replace the base voltage used by residual correction with `peak_lobe_base_voltage_v`, and compute residual from `target - peak_lobe_predicted_field_mT`.

- [ ] **Step 3: Preserve raw HallBz polarity policy**

Keep `coerce_measured_field_centered()` as the only place that converts `HallBz`, `HallZ`, or `raw_hallbz_mT` to effective field by sign inversion.

- [ ] **Step 4: Verify finite 1st modeling**

Run: `.\\.venv\\Scripts\\python.exe -m pytest -q tests/test_finite_first_phase_sync_modeling.py tests/test_finite_second_modeling.py`

Expected: PASS.

### Task 4: Promote v2 UI From Shell To Operating Flow

**Files:**
- Modify: `src/field_analysis/app_ui_v2.py`
- Test: `tests/test_quick_lut_v2_entrypoints.py`

- [ ] **Step 1: Add v2 UI contract checks**

Assert the v2 app exposes upload state, first modeling state, and export state as separate rendering functions:

```python
assert {"render_data_stage", "render_first_model_stage", "render_export_stage"}.issubset(function_names)
```

- [ ] **Step 2: Wire shared upload memory read-only summary**

Read existing upload manifest state from `outputs/field_analysis_app_state/uploads` without mutating legacy state.

- [ ] **Step 3: Add first-model run controls after the peak-lobe core is integrated**

The v2 UI should call the same tested core path used by `apply_finite_first_phase_sync_modeling()` and show lobe metadata before export.

- [ ] **Step 4: Verify v2 app contract and relevant Quick LUT regressions**

Run: `.\\.venv\\Scripts\\python.exe -m pytest -q tests/test_quick_lut_v2_entrypoints.py tests/test_quick_lut_feedback_correction_ui.py tests/test_support_reference_integrity.py`

Expected: PASS.
