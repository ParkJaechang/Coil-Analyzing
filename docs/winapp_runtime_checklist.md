# Windows App Runtime Checklist

Last updated: 2026-06-01 KST

Latest PR-manager recheck: 2026-06-01 KST. Runtime remains blocked because the Windows App skeleton is still missing.

Status legend: PASS means verified on the current PR head, PARTIAL means partial evidence exists but runtime evidence is missing, FAIL means a blocking issue is known.

## Implementation Gate

| Item | Status | Evidence / Expected Result | Blocker |
| --- | --- | --- | --- |
| Windows App skeleton | FAIL | `src/coil_win_app/` is missing from the audited implementation head. | Coder must implement and push skeleton. |
| `main.py` | FAIL | `src/coil_win_app/main.py` is missing. | Needed before app launch testing. |
| `core_adapter.py` | FAIL | `src/coil_win_app/core_adapter.py` is missing. | Needed before non-Streamlit adapter testing. |
| `project_state.py` | FAIL | `src/coil_win_app/project_state.py` is missing. | Needed before project folder state testing. |
| `ui/` | FAIL | `src/coil_win_app/ui/` is missing. | Needed before Windows App UI testing. |

## Required Tests

| Test | Status | Blocker |
| --- | --- | --- |
| `tests/test_core_adapter_contract.py` | FAIL | Missing. |
| `tests/test_winapp_no_streamlit_dependency.py` | FAIL | Missing. |
| `tests/test_final_lut_export_contract.py` | FAIL | Missing. |

## Runtime Checklist

| Item | Status | Evidence / Expected Result | Blocker |
| --- | --- | --- | --- |
| App launch | PARTIAL | Cannot validate the actual Windows App until skeleton exists. | Missing `src/coil_win_app/main.py`. |
| Project folder selection | PARTIAL | Cannot validate Windows App project state until skeleton exists. | Missing `project_state.py`. |
| Target config | PARTIAL | Existing field-analysis docs/policy exist, but Windows App UI is missing. | Missing Windows App UI. |
| Finite first modeling | PARTIAL | Existing field-analysis module exists; Windows App route is missing. | Missing core adapter. |
| Finite second modeling | PARTIAL | Existing field-analysis module exists; Windows App route is missing. | Missing core adapter. |
| Continuous extraction | PARTIAL | Existing field-analysis module exists; Windows App route is missing. | Missing core adapter. |
| Continuous first modeling | PARTIAL | Existing field-analysis module exists; Windows App route is missing. | Missing core adapter. |
| Final LUT export | PARTIAL | Existing contract is `sample_index,time_s,voltage_v`; Windows App export route is missing. | Missing final export adapter/UI. |
| CSV preview | PARTIAL | Cannot validate Windows App CSV preview until UI exists. | Missing Windows App UI. |
| Packaging smoke test | PARTIAL | Not executed. | Missing Windows App skeleton. |

## User Runtime Checklist After Skeleton Lands

1. Launch the Windows App from the PR branch.
2. Select a project folder and confirm state persists.
3. Load target config and verify target shape is separate from target peak normalization.
4. Run finite first modeling.
5. Run finite second modeling.
6. Run continuous extraction.
7. Run continuous first modeling.
8. Export final LUT and confirm the CSV header is exactly `sample_index,time_s,voltage_v`.
9. Confirm voltage limit/normalization is `+/-10V`.
10. Confirm harmonic inverse is not exposed as final export.
11. Run packaging smoke and record the command/result.
12. Confirm no generated exports, upload caches, local state, user CSV/XLSX data, `dist/`, `build/`, `__pycache__/`, or `.pytest_cache/` are committed.
