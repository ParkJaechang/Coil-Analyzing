# Windows App Next Agent Prompts

Last updated: 2026-06-01 KST

Latest PR-manager recheck: 2026-06-01 KST. Required implementation tasks remain unchanged because `src/coil_win_app/` is still missing.

## Prompt For Next Coder

You are the next Coder for the Windows App PR. The current PR head is documentation only for Windows App status: `src/coil_win_app/` is not present.

Do not modify Streamlit/core algorithms casually. Do not push to the Streamlit/core reference repo. Work only on the Windows App PR branch unless explicitly instructed otherwise.

Required implementation tasks:

1. Add `src/coil_win_app/main.py`.
2. Add `src/coil_win_app/core_adapter.py`.
3. Add `src/coil_win_app/project_state.py`.
4. Add `src/coil_win_app/ui/`.
5. Add `tests/test_core_adapter_contract.py`.
6. Add `tests/test_winapp_no_streamlit_dependency.py`.
7. Add `tests/test_final_lut_export_contract.py`.
8. Ensure the Windows App core adapter imports without Streamlit.
9. Ensure final LUT export contract is exactly `sample_index,time_s,voltage_v`.
10. Ensure voltage limit/normalization policy is `+/-10V`.
11. Ensure harmonic inverse is not used as the final export route.

## Upstream Streamlit/Core Requests

1. Provide a stable, documented, non-Streamlit core adapter surface for Windows App imports.
2. Provide a durable dependency pin, release tag, or package reference for Windows App.
3. Keep final LUT export contract stable: `sample_index,time_s,voltage_v`.
4. Keep command voltage policy stable as peak-based `+/-10V` normalization/limit.
5. Keep harmonic inverse out of final LUT export.
6. Document any intentional algorithm/policy changes before Windows App updates its dependency reference.

## Test TODO

- Run `tests/test_core_adapter_contract.py`.
- Run `tests/test_winapp_no_streamlit_dependency.py`.
- Run `tests/test_final_lut_export_contract.py`.
- Run targeted field-analysis tests around final LUT export and continuous schema adapter if adapter code calls those APIs.
- Add routing tests only if the Windows App UI introduces routing logic.

## Runtime TODO

- Launch the Windows App from the PR branch.
- Select a project folder.
- Verify target config.
- Run finite first modeling and finite second modeling through the Windows App route.
- Run continuous extraction and continuous first modeling through the Windows App route.
- Export final LUT and verify `sample_index,time_s,voltage_v`.
- Confirm generated/user data is not committed.

## Packaging TODO

- Run a packaging smoke test and record the command, result, artifact path, and whether the app launches.
- Confirm packaged app does not include user measurement data, generated exports, upload caches, or local state.
- Confirm packaged app still exports final LUT CSV with only `sample_index,time_s,voltage_v`.
