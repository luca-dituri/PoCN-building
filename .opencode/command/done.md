---
description: Close out a small task: verify what changed, then update PROGRESS.md and GOAL.md
---

Fast-path close-out. A small task was just completed; record the evidence and
update the bookkeeping files so the next session starts current.

Steps:
1. Read `GOAL.md` and the tail of `PROGRESS.md` for context.
2. Run the single verification command that matches what was changed (e.g. the
   `run_*`/`plot_*` script or the relevant `VERIFY.md` check) and capture the
   real result.
3. Append one PROGRESS.md row: `| n. <what changed> | <one-line change> |
   \`<command>\` -> <result> | none/blocker |`.
4. If the change satisfies a GOAL.md item, mark that item done in GOAL.md; if
   all items are done, set status to COMPLETE and draft the next goal.
5. Report back in 2-4 lines: what was verified and what the files now say.

Rules: evidence-based — the PROGRESS.md row must carry a real command result.
Do not invent a result if the command was not run.
