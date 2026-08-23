---
description: Compile the report, stage code/config/md only, and commit with a repo-style message
---

SHIP step. Compile the report and commit the current work.

Steps:
1. Run `git status` and `git diff` to see what changed.
2. Compile the report: `cd latex && latexmk -pdf main.tex` (from repo root).
   If it fails, stop and report; do not commit a broken report.
3. Stage only code + config + `.md` files. Never stage `data/`, `input/`,
   `notebooks/`, `.env`, `.pem`, or build artifacts.
4. Review `git log --oneline -10` and write a concise commit message matching
   the repo style.
5. Commit and report the resulting hash + one-line summary.

Rules: stage intended files only; never commit secrets or generated outputs.
