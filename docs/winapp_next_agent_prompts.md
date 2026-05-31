# Windows App Next Agent Prompts

Last updated: 2026-06-01 KST

## Prompt For Next Coder

You are the next Coder for the Windows App PR. Do not modify Streamlit/core algorithms casually. Work only on the Windows App PR branch unless explicitly instructed otherwise.

Tasks:

1. Run the Windows App from the PR branch and complete `docs/winapp_runtime_checklist.md`.
2. Capture runtime evidence for app launch, project folder selection, target config, finite first modeling, finite second modeling, continuous extraction, continuous first modeling, CSV preview, final LUT export, and packaging smoke.
3. Confirm final LUT export columns are exactly `sample_index,time_s,voltage_v`.
4. Confirm voltage normalization/limit copy and behavior use `±10V`.
5. Confirm target peak normalization remains separate from target shape.
6. Confirm harmonic inverse is not exposed or used as the final export route.
7. Do not commit generated artifacts, local state, upload caches, export caches, real measurement CSV/XLSX, `dist/`, `build/`, `__pycache__/`, or `.pytest_cache/`.

## Upstream Streamlit/Core Requests

1. Provide a stable, documented, non-Streamlit core adapter surface for Windows App imports.
2. Provide a durable dependency pin or release tag for the Windows App to reference.
3. Keep final LUT export contract stable: `sample_index,time_s,voltage_v`.
4. Keep command voltage policy stable as peak-based `±10V` normalization/limit.
5. Keep harmonic inverse out of final LUT export.
6. Document any intentional algorithm/policy changes before the Windows App updates its dependency reference.

## Test TODO

- Run targeted core tests around final LUT export, continuous schema adapter, finite first modeling, and finite second modeling.
- Run Streamlit/AppTest source contract tests if the UI copy changes.
- Add or update routing tests only if the runtime checklist exposes a routing gap.
- Preserve current retune policy and acceptance thresholds unless explicitly approved.

## Routing TODO

- Verify Windows App launch path uses the intended latest/full entrypoint or Quick LUT entrypoint.
- Keep `app_field_analysis_latest.py` as the latest/full field-analysis entrypoint.
- Keep `app_field_analysis_quick.py` as the Quick LUT entrypoint.
- Treat `src/field_analysis/app_ui_snapshot.py` as the practical UI source of truth.
- Do not add new feature bodies to `app_ui_snapshot.py`; extract to `ui_*.py` modules if code changes are later needed.

## Packaging TODO

- Run a packaging smoke test and record the command, result, artifact path, and whether the app launches.
- Confirm packaged app does not include user measurement data, generated exports, upload caches, or local state.
- Confirm packaged app still exports final LUT CSV with only `sample_index,time_s,voltage_v`.
