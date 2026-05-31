# Windows App Runtime Checklist

Last updated: 2026-06-01 KST

Status legend: PASS means verified on the current PR path, PARTIAL means source/CI evidence exists but user-launched runtime evidence is missing, FAIL means a blocking issue is known.

| Item | Status | Evidence / Expected Result | Blocker |
| --- | --- | --- | --- |
| App launch | PARTIAL | Source and CI are present; user must launch the Windows App/entrypoint from the PR branch. | Need fresh user-launched runtime evidence. |
| Project folder selection | PARTIAL | UI/workflow docs mention project selection, but current audit did not run a desktop clickthrough. | Need runtime confirmation. |
| Target config | PARTIAL | Target shape, target peak, and normalization are separated in source/policy docs. | Need visual confirmation in app. |
| Finite 1차 modeling | PARTIAL | `field_analysis.finite_first_phase_sync` imports without Streamlit. | Need runtime modeling run. |
| Finite 2차 modeling | PARTIAL | `field_analysis.finite_second_modeling` imports without Streamlit. | Need runtime modeling run. |
| Continuous extraction | PARTIAL | `field_analysis.continuous_steady_state_schema` imports without Streamlit and rejects final LUT as measurement source. | Need runtime extraction run. |
| Continuous 1차 modeling | PARTIAL | `field_analysis.continuous_first_modeling` imports without Streamlit. | Need runtime modeling run. |
| Final LUT export | PARTIAL | Export contract is documented and tested as `sample_index,time_s,voltage_v`. | Need exported CSV from runtime. |
| CSV preview | PARTIAL | Final LUT review/preview source paths exist. | Need visual/user confirmation. |
| Packaging smoke test | PARTIAL | Not executed during this PR-manager pass. | Need Coder or user packaging smoke result. |

## Required Runtime Evidence

- Screenshot or text capture of app launch and selected project folder.
- Screenshot of target config showing target shape, target peak, and normalization as separate concepts.
- Finite first modeling result for supported 1.0-cycle and/or 1.5-cycle case.
- Finite second modeling result using actual-drive input.
- Continuous one-cycle extraction result.
- Continuous first modeling result.
- Final LUT CSV preview or downloaded CSV header proving exactly `sample_index,time_s,voltage_v`.
- Packaging smoke test command/result.

## Policy Checks To Confirm In Runtime

- Voltage limit/normalization copy uses `±10V`.
- Target peak normalization policy is visible and not mixed with target shape.
- Final LUT export does not expose harmonic inverse as the final export method.
- Generated exports, upload caches, and user measurement files are not staged for commit.
