# Codex Guardrails

- Work on `workspace1`; do not push directly to `main`.
- `app_field_analysis_latest.py` is the latest/full field-analysis entrypoint.
- `app_field_analysis_quick.py` is the Quick LUT entrypoint.
- `src/field_analysis/app_ui.py` is a wrapper/loader.
- `src/field_analysis/app_ui_snapshot.py` is the practical UI source of truth.
- New Python modules should stay under 600 lines.
- If a file approaches 600 lines, split by feature.
- Do not grow `app_ui_snapshot.py` with new feature bodies; extract to `ui_*.py` modules.
- Do not commit generated artifacts, local state, upload/export caches, or change retune policy/acceptance thresholds casually.
