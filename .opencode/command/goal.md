---
description: Define a new goal: writes GOAL.md + VERIFY.md and records the handoff in PROGRESS.md
---

DEFINE step. Define a new goal from the objective below and stop. Do NOT start
implementing.

Objective: $ARGUMENTS

Steps:
1. Read `projects_info/` specs, `AGENTS.md`, and the
   `complex-networks-project` skill (`.opencode/skills/complex-networks-project/SKILL.md`)
   for context.
2. Read the current `GOAL.md` and `PROGRESS.md` to confirm the previous goal is
   closed (or note it is being superseded).
3. Rewrite `GOAL.md` with: a one-line objective, numbered items (each item is a
   concrete deliverable), and "Status: ACTIVE". Keep it short.
4. Rewrite `VERIFY.md` with a table mapping each GOAL item to (a) a runnable
   check command and (b) the expected result. Checks must be runnable from the
   repo root and produce objective evidence (exit code / grep match / file
   written).
5. Append a "Handoff" row to `PROGRESS.md` noting the new goal is defined and
   no checkpoint has run yet.
6. Report the goal summary (items + how each is verified) back to the user.

Rules: contract-first — the goal must define a stopping condition before any
implementation. No file beyond the three above is touched in this step.
