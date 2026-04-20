# Git And Codex Workflow

This file is for day-to-day use, especially when working with Codex.

## Core Terms In Plain Language

`repository`

- the project folder tracked by Git
- here, this folder is the repository

`branch`

- a work line
- `main` is the stable baseline
- `feature/...` is a safe branch for new work

`commit`

- a saved snapshot of code changes

`push`

- upload local commits to GitHub

`pull`

- download and apply the latest remote changes

`PR` (pull request)

- a request to merge one branch into another
- useful when you want review, a clean merge record, or a safe checkpoint before touching `main`

`worktree`

- a second local folder using the same Git repository but a different checked-out branch
- powerful, but easy to confuse with normal folders

## Recommended Default Workflow

For most tasks:

```powershell
git pull
git switch -c feature/my-task
```

Ask Codex to make the change in this branch.

After Codex changes files:

```powershell
git status
git add .
git commit -m "Describe the change"
git push -u origin feature/my-task
```

Then either:

- open a PR into `main`, or
- merge yourself later if you work solo

## Current Branch Practice

Right now, active development may intentionally happen on `workspace1`.

If you are using that branch as the active line:

```powershell
git switch workspace1
git status -sb
git push -u origin workspace1
```

That means:

- you can keep developing on `workspace1`
- you can push checkpoints to `origin/workspace1`
- you can open a PR from `workspace1` into `main` when ready

This is not the default recommendation for every future task, but it is a valid current workflow.

## When To Use `main`

Use `main` when:

- you just cloned the repo
- you are reviewing the current stable baseline
- you are pulling the latest shared state
- you are making a very small hotfix and intentionally want it directly on the baseline

Do not do large exploratory work directly on `main` unless you intentionally accept that risk.

## When To Create A New Branch

Create a new branch when:

- you are starting a new feature
- you are changing UI and backend together
- you are not sure the change is safe
- you want a clean history for later review

Recommended branch naming:

- `feature/validation-ui`
- `feature/retune-catalog`
- `fix/launcher-port-selection`
- `docs/recovery-guide`

## When To Use A PR

Use a PR when:

- you want a clear review point before merging to `main`
- you want to compare what changed
- you want a safe “merge decision” step
- you want a permanent record of why a branch existed

If you are working solo, PRs are optional, but still useful.

## Example PR Flow For `workspace1`

If work was done on `workspace1` and you want to merge it into `main` through GitHub:

```powershell
git switch workspace1
git status
git add .
git commit -m "Describe the change"
git push -u origin workspace1
```

Then on GitHub:

- open the repository page
- create a PR with base `main` and compare `workspace1`
- review the diff
- merge the PR

After the PR is merged:

```powershell
git switch main
git pull origin main
```

If you no longer need the branch later, you can delete it locally and remotely.

## Bringing `main` Into `workspace1`

If `main` moved forward and `workspace1` needs the latest baseline:

```powershell
git fetch origin
git switch workspace1
git merge origin/main
```

This updates `workspace1` with the latest `main` changes without leaving the working branch.

## When Not To Use A Worktree

Do not use a worktree unless you intentionally need:

- two branches open at once
- one folder for stable baseline and another for a risky refactor
- parallel work without repeated checkout/switch operations

If you are still getting used to Git, prefer:

- one clone folder
- one active branch at a time

## Codex Best Practices

When using Codex:

1. open the repository root as the workspace
2. confirm branch first
3. ask Codex to edit within that branch
4. inspect `git status`
5. commit only what you actually want to keep

Useful commands:

```powershell
git branch --show-current
git status -sb
git diff
```

## Continuing On Another PC

Use GitHub as the handoff point.

On the other PC:

```powershell
git clone https://github.com/ParkJaechang/Coil-Analyzing.git
cd Coil-Analyzing
git switch main
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Then either work on `main` or create a feature branch there.

Do not copy random local folders unless you specifically need uncommitted local-only state.

## Continuing On Another PC With Codex Context

Git restores code state, but not the full reasoning context from an older Codex desktop thread.

To avoid losing development context:

- commit and push important changes
- keep a short handoff summary in `docs/CODEX_HANDOFF.md`
- start the next Codex session by asking it to read the handoff and recovery docs first

That gives the next session a repository-native context source instead of depending on one old thread.

## If Codex Or Git Says “Checkout The Feature Branch”

That usually means:

- the changes you want are on another branch
- the current folder is still on `main`
- Git wants the intended branch to be the active branch before push or PR

First check:

```powershell
git branch --show-current
git branch -vv
```

## Practical Rule

If confused, always answer these three questions first:

1. which folder am I in
2. which branch is active
3. which remote branch is this supposed to update

Those three answers solve most Git confusion.
