---
name: complex-networks-project
description: Use when working inside this repo — cascading-failure and SOC simulations (sandpile, R(z)-B(p)-R(z) interconnected networks, Brummitt), running run_*/plot_*/fit_exponents scripts, editing config YAML, or the LaTeX report. Encodes this project's task split, env, run commands, and plot conventions.
---

# Complex Networks Project (Cascading Failures & SOC)

Project-specific conventions for this repository. This skill is the canonical
source for run commands, code conventions, and plot conventions. `AGENTS.md` at
the repo root is a thin always-on stub (project summary, env, workflow SOP,
hard boundaries); `README.md` is the human-facing overview. Read this skill when
doing task work.

## Two tasks, keep them separate

- **Task 15 (Theoretical).** Sandpile-model cascading failures on synthetic
  topologies (Gaussian, Uniform, Scale-Free) vs multiplicative
  branching-process predictions, plus interconnected R(z)-B(p)-R(z) two-layer
  networks (Brummitt et al.).
- **Task 47 (Data).** Spatial covariance / Landau-Ginzburg on empirical spatial
  networks (GridKit). Documented in `TASK_47.md`.

Current status (what is ACTIVE right now) lives in `GOAL.md`, never here — read
it before task work.

## Bookkeeping (always on)

End every completed task by updating the files, without being asked: append one
`PROGRESS.md` row with a real command result; tick done `GOAL.md` items and set
status COMPLETE (then draft the next goal) when all are done. Update this skill
or `AGENTS.md` only when a durable convention changed, not for routine status.

## Environment

- Conda env `complex-networks-env` (Python 3.10): numpy, scipy, matplotlib,
  pandas, networkx, numba, tqdm, pyyaml.
- Activate with `conda activate complex-networks-env`.
- No test suite / linter. Verify by running the scripts and recompiling the report.

## Run commands (from repo root)

```bash
python code/task_15/run_part1.py                  # Gaussian + Uniform
python code/task_15/run_part2.py                  # scale-free, all gammas
python code/task_15/run_part3.py                  # R(3)-B(p)-R(3) interconnected
python code/task_15/plot_results.py               # all plots -> latex/figures/task_15/
python code/task_15/fit_exponents.py              # exponent fits -> data/task_15/fits.csv
python code/task_15/fit_exponents.py --sweep      # sensitivity to fitting window
cd latex && latexmk -pdf main.tex                 # report
```

## Config, never hardcode

- Simulation parameters live in `config/task_15/config.yaml` (N, gammas, f,
  realizations, p_values, fig6 windows) and `config/shared.yaml` (seed,
  plot_style). Edit the YAML, not the scripts.
- Report text must stay consistent with config values (N=1e5 for tasks 1-2,
  N=2e3 per layer with f=0.01 and 2e6 grains after 20% transient for part 3).
  If config changes, update the report too.

## Code conventions

- Scripts `sys.path.insert(0, base_dir/'code')` and import as `common.*` /
  `task_15.*`. Use `common.utils.load_config` / `set_seed`.
- Seeds: base seed from `config/shared.yaml`; derive per-realization seeds.
  Call `set_seed` per worker process.
- Distributions use monotonic log-binning (`get_log_binned_distribution`,
  densities), stored as `{size,lifetime}_{centers,probs}` arrays in `.npz`.
- All plot scripts call `common.plot_utils.setup_plot_style` — do NOT set
  per-script rcParams; share font-size constants from `plot_utils`.

## Plot conventions (task 15)

- Variables: `s` = avalanche size, `t` = lifetime.
- Part 1 combined 1x2 PDF: left = size distribution, right = lifetime.
  Branch-process reference lines (`tau=1.5`, `z=2.0`) inside the axes.
- Part 2 (scale-free): gamma legend only on the right panel, outside the plot, theoretical prediction legend displayed on the left panel, in the plot.
- Part 3 (Brummitt Figs 4-6): "big" cascade = `ta > 0.5*N` (topplings split by
  network: `ta`/`tb` + origin). Primary axes linear in `p`; Fig 4/5 insets are
  log-log rank-size plots (top-right `ax.inset_axes`) of the **per-run average**
  of the largest 10^4 events; short y-labels (`T_a`/`t`), no title, lines kept
  between points. Main legends at 'upper left'. Fig 5 mean-field reference
  `s(t) ~ 0.5 t^{-3/2}`. Fig 6: shared `supxlabel('Interconnectivity p')`, bold
  colored row labels "Small cascades" (blue) / "Large cascades" (red) outside,
  one rotated "Probability" label inside.
- Figures → `latex/figures/task_15/`; report sections → `latex/sections/`
  (`task15.tex` = task 15, `task47.tex` = task 47).

## Git hygiene

- `data/` git-ignored (path still exist locally).
- `.npz` results are generated artifacts; code + config are the source of truth.
- Never commit the `.pem` keys or any cluster IP credentials.
