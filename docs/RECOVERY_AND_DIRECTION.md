# Recovery And Development Direction

This file exists to reduce future “why does the app look old” or “which code is authoritative” recovery work.

## Canonical Working Assumption

This repository clone is the source of truth for current development.

If local experiments exist elsewhere, they are not authoritative unless their changes are deliberately copied into this repository and committed.

## Authoritative Paths

Repository root:

- this clone

Primary app entrypoints:

- `app_field_analysis_latest.py`
- `app_field_analysis_quick.py`

Authoritative package paths:

- `src/field_analysis/`
- `src/coil_analyzer/`

UI source of truth:

- `src/field_analysis/app_ui_snapshot.py`

Wrapper module:

- `src/field_analysis/app_ui.py`

That wrapper should be treated as loader glue.
The actual working UI should be changed in `app_ui_snapshot.py` and its helpers unless there is a clear reason not to.

## Current Development Direction

Current emphasis:

- stable Streamlit app execution
- self-contained repository layout
- field analysis and validation/retune backend inside `src/field_analysis/`
- launchers that avoid local port collisions
- Git-friendly working flow that does not depend on hidden local folders

## What Should Not Become Source Of Truth Again

Do not rely on:

- temporary worktrees as the only copy of important code
- generated `artifacts/`
- runtime `outputs/`
- local upload caches
- logs or screenshots
- external folders that are not part of this repository

If something matters, it should be:

- copied into this repository
- committed
- pushed to GitHub

## Recovery Checklist

If the app seems wrong, check in this order.

1. Confirm the folder.

```powershell
pwd
```

2. Confirm the active branch.

```powershell
git branch --show-current
git status -sb
```

3. Confirm the clone is up to date.

```powershell
git pull
```

4. Confirm the launcher is running this repository’s entrypoint.

Expected app path should point at this repository clone, not another folder.

5. Confirm the UI source file exists.

- `src/field_analysis/app_ui_snapshot.py`

6. Confirm the package import resolves to this clone.

If needed:

```powershell
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))
import field_analysis.app_ui_snapshot as app
print(app.__file__)
PY
```

The printed path should point into this repository clone.

## Safe Change Rules

When making changes:

- keep entrypoints simple
- keep generated files out of Git
- add helper modules instead of overloading one giant file when practical
- commit small logical units
- push important state, do not leave critical code only in local experiments

## Recommended Human Workflow

For normal work:

1. open this clean clone
2. pull latest `main`
3. create `feature/...`
4. work with Codex
5. commit and push

That is the simplest path that minimizes future recovery work.
