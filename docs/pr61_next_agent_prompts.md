# PR61 Next Agent Prompts

These prompts are for follow-up agents. They are intentionally detailed so the next agent can act without reinterpreting the user requirements. Do not treat this document as implementation code.

## Prompt A1: Hide legacy uploaders and simplify default Quick LUT UI

```text
You are Agent A for Coil-Analyzing PR61.

Problem:
- The default Quick LUT runtime still exposes legacy uploaders and hardware/debug controls in the sidebar/main flow.
- Users report that old features and current workflow are mixed together.

User runtime symptom:
- On the default Quick LUT app, users can still see continuous/finite/validation/LCR uploaders and DAQ/AMP/extrapolation controls before they are needed.

Suspected cause:
- `app_ui_snapshot.py` renders file uploaders and hardware settings directly in the sidebar.
- Some Quick LUT debug/status rows are still default-visible rather than under Advanced / Debug.

Modification requirements:
- Move sidebar file uploaders (`continuous_uploads`, `transient_uploads`, `validation_uploads`, `lcr_uploads`) behind an Advanced / Legacy expander or Data Import mode.
- Keep Upload Memory / Dataset Library status visible, but do not make legacy uploaders the primary path.
- Move DAQ/AMP gain/extrapolation controls behind Advanced / Debug for Quick LUT unless they are essential for the current user workflow.
- Do not remove underlying functionality; hide/reorganize only.
- Keep Quick LUT primary nav focused on Quick LUT / Raw Waveforms / LUT Review / Data-Cache Status.

Forbidden:
- Do not delete source/test files.
- Do not remove upload memory, dataset library, or actual-drive processing.
- Do not alter modeling thresholds or correction math.

Test requirements:
- Update/add UI contract tests verifying default Quick LUT does not expose legacy uploaders.
- Preserve button-gated calculation tests.
- Run targeted UI tests and full pytest if feasible.

Runtime evidence requirements:
- Launch Quick LUT from the PR branch.
- Capture or summarize the default sidebar and Quick LUT first screen.
- Confirm legacy uploaders are hidden until Advanced/Legacy is opened.
```

## Prompt A2: Remove remaining `100pp`, DAQ, AMP gain, and extrapolation wording from main UI

```text
You are Agent A for Coil-Analyzing PR61.

Problem:
- The main Quick LUT flow still contains legacy wording such as `fixed 100pp rounded-triangle field target`, `DAQ`, `AMP gain`, and `target extrapolation`.
- User wants target shape fixed, but target peak semantics and normalization separated.

User runtime symptom:
- User sees old 100pp/gain/DAQ wording and cannot tell whether target peak is user-set, normalized, or fixed.

Suspected cause:
- `app_ui_snapshot.py` still contains old captions/metrics in the Quick LUT scalar and waveform compensation sections.

Modification requirements:
- Replace `fixed 100pp` user-facing text with current semantics:
  - target shape = fixed rounded triangle
  - field review/modeling normalization = +/-50mT
  - target peak field = user-facing setting or explicitly normalized review peak
  - command voltage limit = +/-5V
- Move DAQ/AMP gain/extrapolation details into Advanced / Debug or clearly mark as reference-only.
- Preserve schema/metadata keys even if UI labels change.

Forbidden:
- Do not change final LUT CSV schema.
- Do not change finite/continuous modeling math.
- Do not remove debug metadata; only hide or reword in UI.

Test requirements:
- Add/update tests that fail if `fixed 100pp`, `100mT pp fixed`, or equivalent old wording appears in main Quick LUT source.
- Add/update tests for current Korean target semantics copy.

Runtime evidence requirements:
- Launch Quick LUT and record whether old wording is absent from the default path.
```

## Prompt A3/B1: Decide and clean up Startup Compensation Review

```text
You are Agent A or B for Coil-Analyzing PR61.

Problem:
- Startup Compensation Review overlaps conceptually with residual/second correction UI and remains partially visible.

User runtime symptom:
- Users see Startup Compensation Review and cannot tell whether it is current workflow, legacy diagnostic, or redundant.

Suspected cause:
- `render_startup_compensation_review` is still called from multiple paths in `app_ui_snapshot.py`.

Modification requirements:
- Decide one of:
  1. Keep: label it clearly as Advanced diagnostic only.
  2. Merge: fold useful fields into residual/second modeling diagnostics.
  3. Hide/remove from default path: expose only under Advanced / Debug.
- Use Korean user-facing labels.
- Do not overclaim model quality.

Forbidden:
- Do not change startup compensation backend behavior unless the chosen policy requires B review.
- Do not delete tests blindly; update contracts intentionally.

Test requirements:
- UI contract test for chosen policy.
- Ensure default Quick LUT does not show conflicting startup review in primary path.

Runtime evidence requirements:
- Launch finite and non-finite/scalar paths and confirm Startup Compensation Review visibility matches the chosen policy.
```

## Prompt B1: Verify or fix rounded-triangle target template ripple

```text
You are Agent B for Coil-Analyzing PR61.

Problem:
- The user requires the target template itself to be ideal.
- Current acceptance inventory marks rounded-triangle ripple removal as NOT VERIFIED.

User runtime symptom:
- Target waveform may visually contain ripple or non-ideal linear segments, making modeling assessment unreliable.

Suspected cause:
- Target generation helpers may inherit support/plot artifacts, interpolation artifacts, or non-ideal smoothing.

Modification requirements:
- Identify the exact target template generation path for Quick LUT finite and continuous.
- Add analytic checks for ideal rounded triangle:
  - monotonic linear segment quality
  - smooth rounded corners
  - no support-derived ripple
  - no data-source contamination
- If needed, fix target generation so Physical Target is generated from an analytic template, not measured/support traces.

Forbidden:
- Do not stretch target duration.
- Do not alter target shape based on support data.
- Do not use measured support as target.

Test requirements:
- Add a dedicated target template ripple/linearity test.
- Existing target immutability and semantic tests must pass.

Runtime evidence requirements:
- Produce user-inspection packet or CSV summary for target template at 1Hz finite 1.0 and 1.5.
```

## Prompt A/B: Runtime evidence packet for PR61

```text
You are a validation agent for Coil-Analyzing PR61.

Problem:
- Source and tests pass, but user-facing acceptance requires launched-runtime evidence.

Required runtime cases:
- Quick LUT default screen.
- Finite 1Hz / 1.0 cycle first modeling.
- Finite 1Hz / 1.5 cycle first modeling.
- Finite second modeling with matching uploads/2nd actual-drive source if available.
- Continuous steady-state extraction.
- Continuous first modeling.
- Continuous final LUT first export.
- Continuous final LUT second export if second result exists.

Check:
- Workspace path, branch, HEAD SHA, process command.
- Stale process ruled out.
- Wrong workspace ruled out.
- Current/applied/result target config visible and unchanged.
- Final LUT exports have exactly sample_index,time_s,voltage_v.

Forbidden:
- Do not judge final model quality.
- Do not create screenshots unless the user asks; user may perform visual inspection.

Deliverable:
- Runtime status report with PASS/PARTIAL/FAIL and exact missing evidence.
```
