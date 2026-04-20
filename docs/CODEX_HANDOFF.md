# Codex Handoff

This file is the recovery point for continuing work from another PC or a new Codex thread.

The goal is simple:

- keep important context inside the repository
- avoid relying on one desktop thread as the only source of development intent
- let a new Codex session recover the current branch, priorities, and next actions quickly

## Current Working Assumption

Current active development may happen directly on `workspace1`.

That means:

- code changes can be committed on `workspace1`
- `workspace1` can be pushed to `origin/workspace1`
- changes can be merged into `main` through a PR when ready

If the active branch changes later, update this file.

## Source Of Truth

For development context, trust these in order:

1. the current checked-out repository clone
2. Git history and the active branch
3. this handoff document
4. the recovery and workflow docs in `docs/`

Do not treat old desktop chat threads, screenshots, local notes, or ad-hoc folders as authoritative.

## Branch Flow

Typical current flow:

1. work on `workspace1`
2. commit meaningful checkpoints locally
3. push with `git push -u origin workspace1` the first time, then `git push`
4. open a PR from `workspace1` into `main` when the change is ready
5. merge the PR
6. update local `main` with `git switch main` and `git pull origin main`

To bring the latest `main` back into `workspace1`:

```powershell
git fetch origin
git switch workspace1
git merge origin/main
```

To merge `workspace1` into local `main` directly:

```powershell
git switch main
git pull origin main
git merge workspace1
git push origin main
```

Prefer a PR when you want a clean review point.

## Cross-PC Continuation Rule

If work continues on another PC, do not rely on the old Codex thread still being available.

Instead:

1. commit and push code changes
2. update this file with the latest goal, status, and next step
3. on the other PC, clone or pull the repository
4. switch to the intended branch
5. ask Codex to read this file and the related docs before making changes

## Session Update Template

Update these fields at the end of an important working session.

`Branch`

- `workspace1`

`Current Goal`

- describe the current feature, bugfix, or evaluation target in one or two lines

`What Changed Recently`

- list the most recent meaningful code or document changes

`Open Issues`

- list blockers, known bugs, or uncertainty

`Next Action`

- state the next concrete task Codex or a human should do first

`Validation Status`

- note what was actually checked: import smoke, app launch, manual review, tests, or not run

## Recommended Prompt On Another PC

Start a new Codex thread with something like:

```text
Read README.md, docs/RECOVERY_AND_DIRECTION.md, docs/GIT_AND_CODEX_WORKFLOW.md, and docs/CODEX_HANDOFF.md.
Then check the current branch and git status.
Summarize the current development context, the active branch intent, and the next task.
After that, continue the implementation on the current branch.
```

## Current Summary

`Branch`

- `workspace1`

`Current Goal`

- use this branch as the active development line for development, fixes, and evaluation
- preserve enough repository context that work can continue on another PC without depending on one old thread

`What Changed Recently`

- repository workflow and recovery guardrails were added
- the repository clone was documented as the source of truth
- `src/field_analysis/app_ui_snapshot.py` was documented as the practical editable UI source of truth
- generated outputs, logs, and local runtime state were explicitly kept out of Git

`Open Issues`

- automated tests are still effectively absent
- much of the app logic remains concentrated in large modules
- future sessions still need a human or Codex to keep this file current

`Next Action`

- update this summary whenever the active task or branch strategy changes materially

`Validation Status`

- repository structure reviewed
- branch and remote state reviewed
- no automated test coverage currently present
