# PR61 Runtime Review Checklist

PR management document for user-launched runtime review. This file is intended to be read from the GitHub PR.

- PR: [#61](https://github.com/ParkJaechang/Coil-Analyzing/pull/61)
- Branch: `codex/finite-feedback-cycle-policy-backend`
- Baseline checked: `4e8c250d9f9dbfe4978ebb192d56c9b8ffb06bd0`
- Rule: PR Manager does not decide model/graph quality. User runtime review decides acceptance.

## Before launching

Record:

- Workspace path:
- Branch:
- `git rev-parse HEAD`:
- Streamlit command:
- Port:
- Stale process ruled out: yes / no
- Wrong workspace ruled out: yes / no

Recommended launch:

```powershell
cd "D:\programs\Codex\Coil Analyzing_clean"
git branch --show-current
git rev-parse HEAD
.\launch_quick_lut_local.cmd
```

## 1. Quick LUT default UI cleanup

Check:

- Quick LUT is the primary default tab.
- Raw Waveforms, LUT Review, and Data / Cache Status remain easy to reach.
- Advanced / Debug content is not mixed into the main user flow.
- Sidebar legacy uploaders do not dominate the default workflow.
- Hardware/legacy wording (`DAQ`, `AMP`, `gain`, `extrapolation`) is not shown as core modeling semantics.

Result:

- PASS / PARTIAL / FAIL:
- Notes:

## 2. Target semantics

Check:

- Target shape is described as fixed rounded triangle.
- Target peak field and internal normalization are separate concepts.
- Field review/modeling normalization is +/-50mT.
- Command voltage limit/export basis is +/-5V.
- No user-facing `100mT pp fixed`, `100pp fixed`, `100pp`, or `목표 bz_mT PP` wording appears.
- If stale old wording appears, record exact tab/section/text.

Result:

- PASS / PARTIAL / FAIL:
- Notes:

## 3. Rounded triangle target template

Check:

- Target line segments look ideal and not support-data-rippled.
- The target template does not inherit measurement noise/ripple.
- User visual judgment decides whether the target graph is acceptable.

Result:

- PASS / PARTIAL / FAIL:
- Notes:

## 4. Finite first modeling command plot

Run:

- Mode: finite startup-aware
- Frequency: 1Hz
- Cycle: 1.0
- Run first modeling.
- Repeat for cycle 1.5.

Check:

- Selected frequency/cycle remain unchanged after modeling.
- Main command plot shows the final first modeled command only.
- Second/final command plots do not overwrite the first modeling section.
- Diagnostic traces and internal metadata are under expanders.
- Phase sync panel shows actual measured source, measured peak, scale-to-50mT, gain/headroom/clipping metadata.
- Phase-aligned residual remains finite through active end.

Result:

- 1Hz / 1.0 PASS / PARTIAL / FAIL:
- 1Hz / 1.5 PASS / PARTIAL / FAIL:
- Notes:

## 5. Finite second modeling and tail policy

Run:

- Use a matching actual-drive result from `uploads/2nd` or upload memory.
- Generate second modeled voltage LUT.

Check:

- 2nd folder/source scan is visible.
- Manual upload is secondary/legacy, not the main path.
- 1.0 and 1.5 are production-supported.
- 1.25, 1.75, and 2.0 are review-only or blocked for production correction/export.
- Final export source clearly distinguishes first vs second command.
- Exported CSV columns are exactly `sample_index,time_s,voltage_v`.

Result:

- PASS / PARTIAL / FAIL:
- Notes:

## 6. Support / Provenance / Consistency UI

Check:

- Main screen uses Korean summary text.
- Internal source/provenance/consistency rows are in Advanced or Debug expanders.
- Support Reference is not presented as command target.
- User can still inspect details when Advanced/Debug is opened.

Result:

- PASS / PARTIAL / FAIL:
- Notes:

## 7. Startup Compensation Review disposition

Check one of the following is true:

- Kept only as Advanced diagnostics.
- Merged into residual/second modeling review.
- Hidden from the default user flow.
- Removed from this workflow.

Record:

- Current disposition:
- PASS / PARTIAL / FAIL:
- Notes:

## 8. Continuous steady-state workflow

Run:

- Mode: Continuous steady-state
- Select a continuous source waveform family and target frequency.
- Extract steady-state 1cycle.
- Run Continuous first modeling.

Check:

- Candidate labels show source waveform/frequency.
- Mismatched source frequency is not silently used.
- Startup transient cycles are excluded.
- Terminal/stop-influenced cycle is not selected as the stable 1cycle.
- Selected 1cycle plot appears after extraction.
- Continuous first modeling plot appears after execution.
- Continuous mode stays 1cycle only and tail off by default.

Result:

- PASS / PARTIAL / FAIL:
- Notes:

## 9. Continuous final LUT export

Check:

- Continuous final voltage LUT export section appears after continuous result exists.
- First result export option appears when first command exists.
- Second result export is unavailable until second command exists.
- If second command exists, second export uses the explicit second command source.
- Export filename includes continuous, first or second, frequency, 1cycle, and loop-safe intent.
- Exported CSV columns are exactly `sample_index,time_s,voltage_v`.

Result:

- PASS / PARTIAL / FAIL:
- Notes:

## 10. Evidence to attach to PR

Attach or summarize:

- Runtime branch/head SHA:
- Quick LUT default screen notes:
- Target semantics notes:
- Finite 1Hz 1.0 result notes:
- Finite 1Hz 1.5 result notes:
- Phase sync residual/scale metadata notes:
- Continuous extraction/modeling notes:
- Continuous final LUT export notes:
- Any screenshot paths if the user chooses to share them:

## Decision

- User runtime review complete: yes / no
- Ready to undraft: yes / no
- Ready to merge: yes / no
- Remaining blockers:
