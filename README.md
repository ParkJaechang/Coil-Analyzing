# Coil Analyzing

`Coil Analyzing` is the trimmed application repository for the current Streamlit-based coil analysis and field-analysis workflow.

This repository is intended to be the stable source of truth for:

- `app_field_analysis_latest.py`
- `app_field_analysis_quick.py`
- `src/field_analysis/`
- `src/coil_analyzer/`

## Start Here

Use this repository clone as the working copy for Codex and Git work.

Recommended local working folder:

- a fresh clone of GitHub `main`
- one folder per active working copy
- feature branches for non-trivial changes

Do not treat ad-hoc folders outside this repository as authoritative source code.

## Source Of Truth

Application entrypoints:

- `app_field_analysis_latest.py`: latest full field-analysis app
- `app_field_analysis_quick.py`: quick LUT-focused app

Core packages:

- `src/field_analysis/`: current production app/backend logic
- `src/coil_analyzer/`: coil analyzer subsystem used by `app.py`

UI loading detail:

- `src/field_analysis/app_ui.py` is a wrapper/loader
- `src/field_analysis/app_ui_snapshot.py` is the editable UI source currently used as the practical UI source of truth

## Run

From repo root:

```powershell
.\launch_field_analysis_latest.cmd
```

or:

```powershell
.\launch_quick_lut.cmd
```

The launcher chooses an available local Streamlit port automatically.

## Run On Another PC

From a fresh clone:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.\launch_quick_lut_local.cmd
```

Use `.\launch_field_analysis_latest_local.cmd` for the full app. These local launchers prefer the repo `.venv`, then PATH `streamlit`, then `python -m streamlit`.

## Git Workflow

Safe default workflow:

```powershell
git pull
git switch -c feature/my-task
```

After edits:

```powershell
git status
git add .
git commit -m "Describe the change"
git push -u origin feature/my-task
```

If the change is very small and you intentionally want to work on `main`, that is possible, but feature branches are the safer default.

## Files That Should Stay Out Of Git

Generated and machine-local state should not be committed:

- `artifacts/`
- `outputs/`
- runtime logs
- `.coil_analyzer/uploads/`
- `.coil_analyzer/exports/`

## Detailed Guides

- [Korean Guide / 한국어 안내](README.ko.md)
- [Git and Codex Workflow](docs/GIT_AND_CODEX_WORKFLOW.md)
- [Recovery and Development Direction](docs/RECOVERY_AND_DIRECTION.md)
- [Codex Handoff](docs/CODEX_HANDOFF.md)
