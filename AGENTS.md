# Codex Guardrails

- Work on `workspace1`; do not push directly to `main`.
- `app_field_analysis_latest.py` is the latest/full field-analysis entrypoint.
- `app_field_analysis_quick.py` is the Quick LUT entrypoint.
- `src/field_analysis/app_ui.py` is a wrapper/loader.
- `src/field_analysis/app_ui_snapshot.py` is the practical UI source of truth.
- Do not commit generated artifacts, local state, upload/export caches, or change retune policy/acceptance thresholds casually.
