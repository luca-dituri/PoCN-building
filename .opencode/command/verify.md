---
description: Re-run every check in VERIFY.md and report pass/fail against expected results
---

AUDIT step. Re-run every check in `VERIFY.md` from the repo root and report
results. Do not fix anything yet — report first, then fix only if the user asks.

Steps:
1. Read `VERIFY.md` and `GOAL.md`.
2. Run each check command exactly as written, from the repo root.
3. Compare actual output to the "Expected" column.
4. Report a pass/fail table: check #, command, expected, actual, PASS/FAIL.
5. For each FAIL, state the likely cause (read the relevant file/script).
6. Append a PROGRESS.md row summarizing: which checks passed, which failed, and
   the blocking cause for any failure.
7. Tell the user the outcome in 3-6 lines.

Rules: evidence-based — every PASS/FAIL must cite a real command result, never
belief. If a check command is out of date, say so explicitly and propose the
corrected command rather than silently substituting it.
