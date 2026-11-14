---
name: git-workflow
description: Use when committing, branching, pulling, pushing, or resolving merge conflicts in this repo.
---
# Git workflow
- Branch names: `pose/<thing>`, `logic/<thing>`, `render/<thing>`, `server/<thing>`, `cards/<thing>`.
- Commit small and often. Prefer a "wip" commit over `git stash`.
- Before pushing: `git pull --rebase origin main`, run `pytest`, then push your branch and open a PR.
- Stay inside your own folder. `shared/` and `requirements.txt` are lead-only.
- On a conflict, ask the owner of that file. Do not guess. `git rebase --abort` undoes a bad rebase.
- Never commit: models/, recordings, config.local.json, calibration.local.json, outbox.jsonl, .venv/.
- `main` must always run. The fest runs from a tag (`v1-fest`), not from `main`.
