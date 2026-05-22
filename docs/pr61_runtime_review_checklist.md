# PR61 Runtime Review Checklist

PR management document for user-launched runtime review. This file is intended to be read from the GitHub PR.

- PR: [#61](https://github.com/ParkJaechang/Coil-Analyzing/pull/61)
- Branch: `codex/finite-feedback-cycle-policy-backend`
- Baseline checked before docs update: `a5c72dc92d954cf0ff074e9e5100811553874846`
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

## 1. Quick LUT default UI

Check:

- Quick LUT is the primary default tab.
- Raw Waveforms, LUT Review, Data / Cache Status are primary tabs.
- Advanced / Debug tabs are hidden unless explicitly enabled.
- Sidebar legacy uploaders are not dominating the default workflow.
- If sidebar uploaders are visible, record which ones:

Result:

- PASS / PARTIAL / FAIL:
- Notes:

## 2. Target semantics

Check:

- Target shape is described as fixed rounded triangle.
- Target peak field and internal normalization are separate concepts.
- Field review/modeling normalization is +/-50mT.
- Command voltage limit/export basis is +/-5V.
- No user-facing `100mT pp fixed`, `100pp fixed`, or equivalent old wording remains.
- If `fixed 100pp` or DAQ/AMP/gain/extrapolation language appears in the main flow, record location.

Result:

- PASS / PARTIAL / FAIL:
- Notes:

## 3. Finite first modeling

Run:

- Mode: finite startup-aware
- Frequency: 1Hz
- Cycle: 1.0
- Apply Quick LUT settings.
- Run first modeling.
- Repeat for cycle 1.5.

Check:

- Selected frequency/cycle remain unchanged after modeling.
- First command main plot shows the final first modeling command clearly.
- Diagnostic traces are in expanders, not the main graph.
- Phase sync panel shows actual measured source, measured peak, scale-to-50mT, gain/headroom/clipping metadata.
- Phase-aligned residual remains finite through active end.

Result:

- 1Hz / 1.0 PASS / PARTIAL / FAIL:
- 1Hz / 1.5 PASS / PARTIAL / FAIL:
- Notes:

## 4. Finite second modeling and tail policy

Run:

- Use a matching actual-drive result from `uploads/2nd` or upload memory.
- Generate second modeled voltage LUT.

Check:

- 2nd folder/source scan is visible.
- Manual upload is secondary/legacy, not the main path.
- 1.0 and 1.5 are production-supported.
- 1.25, 1.75, and 2.0 are review-only or blocked for production correction/export.
- Tail mode is clear: auto/on/off or finite-time zero-return policy as intended.
- Tail threshold text is understandable.
- Final export source clearly distinguishes first vs second command.

Result:

- PASS / PARTIAL / FAIL:
- Notes:

## 5. Continuous steady-state workflow

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

## 6. Continuous final LUT export

Check:

- `Continuous 최종 전압 LUT 추출` section appears after continuous result exists.
- First result export option appears when first command exists.
- Second result export is unavailable until second command exists.
- If second command exists, second export uses `second_limited_voltage_v`.
- Export filename includes `continuous`, `first` or `second`, frequency, `1cycle`, and `loop`.
- Exported CSV columns are exactly `sample_index,time_s,voltage_v`.
- Export is loop-safe period-exclusive 1cycle.

Result:

- PASS / PARTIAL / FAIL:
- Notes:

## 7. LUT Review

Check:

- Final exported LUT can be loaded into LUT Review.
- LUT Review shows time/voltage plots only after button action.
- Time axis is monotonic and not misinterpreted as milliseconds.
- Cache edit/delete works per uploaded LUT item.

Result:

- PASS / PARTIAL / FAIL:
- Notes:

## 8. Evidence to attach to PR

Attach or summarize:

- Runtime branch/head SHA:
- Quick LUT default screen notes:
- Finite 1Hz 1.0 result notes:
- Finite 1Hz 1.5 result notes:
- Continuous extraction/modeling notes:
- Continuous final LUT export notes:
- Any screenshot paths if the user chooses to share them:

## Decision

- User runtime review complete: yes / no
- Ready to undraft: yes / no
- Ready to merge: yes / no
- Remaining blockers:
