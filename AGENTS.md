# AGENTS.md

Project context for AI agents and contributors working on this repository.

## Project overview

**Cascading Failures and Self-Organized Criticality in Complex Networks** — an
academic project with two independent tasks:

- **Task 15 (Theoretical):** Sandpile-model cascading failures across synthetic
  topologies (Gaussian, Uniform, Scale-Free) compared with multiplicative
  branching-process predictions. This task is the active development focus.
- **Task 47 (Data):** Spatial covariance and Landau-Ginzburg models on empirical
  spatial networks (GridKit). **Not started** — leave its folders untouched
  unless explicitly asked.

Specs live in `projects_info/` (`theoretical_task/task-15.md`,
`theoretical_task/prof_requirements.md`, `data_task/project_47.md`) plus the
reference PDFs in `projects_info/theoretical_task/`.

## Environment

- Conda env: `complex-networks-env` (Python 3.10; numpy, scipy, matplotlib,
  pandas, networkx, numba, tqdm, pyyaml).
- Activate from bash: `conda activate complex-networks-env`
  (`conda init bash` already run; new shells load it via `~/.bash_profile`).
- PATH-independent fallback — run any script through the wrapper:
  `./tools/run.sh <script.py> [args...]` (calls the env interpreter by
  absolute path). Use this in automation; plain `python` may resolve to a
  Windows Store stub.
- No standalone test suite and no linter configured. Verify by executing the
  relevant `run_*`/`plot_*` scripts and re-compiling the report.

## How to run

All commands from the repository root unless noted.

```bash
# Task 15 simulations (parallel realizations -> data/task_15/*.npz)
./tools/run.sh code/task_15/run_part1.py
./tools/run.sh code/task_15/run_part2.py

# Plots (log-binned distributions -> latex/figures/task_15/)
./tools/run.sh code/task_15/plot_results.py

# Exponent fitting on existing binned data (-> data/task_15/fits.csv, tables)
./tools/run.sh code/task_15/fit_exponents.py

# Sensitivity of the fitted exponents to the fitting window (--sweep)
./tools/run.sh code/task_15/fit_exponents.py --sweep

# Report
cd latex && latexmk -pdf main.tex
```

Parameters (N, gamma, f, realizations, ...) are set in
`config/task_15/config.yaml` and `config/shared.yaml` — edit the YAML, do not
hardcode values in scripts.

## Conventions

- **Separation of concerns:** each top-level folder splits by task
  (`task_15/`, `task_47/`). Top-level folders: `code/`, `config/`, `data/`,
  `input/`, `latex/`, `projects_info/`.
- `code/common/` holds shared helpers: `utils.py` (`load_config`, `set_seed`),
  `plot_utils.py` (plot style, `get_log_binned_distribution`).
- Scripts add the repo root to `sys.path` (`sys.path.insert(0, base_dir/'code')`)
  and import as `common.*` / `task_15.*`.
- Fixed seeds come from `config/shared.yaml`; per-realization seeds are derived
  from the base seed. `set_seed` must be called per worker process.
- Avalanche size/lifetime histograms use monotonic log-binning
  (`get_log_binned_distribution`) with densities; distributions are stored as
  `{size,lifetime}_{centers,probs}` arrays in `.npz`.
- Plot conventions (task 15): variables are `s` (avalanche size) and `t`
  (lifetime); one combined 1x2 PDF per part — left panel size distribution,
  right panel lifetime distribution; branch-process reference lines
  (`tau=1.5`, `z=2.0`) plotted inside the axes. Part 2 (scale-free): gamma
  legend appears only on the left panel, inside the plot area.
- Figures go to `latex/figures/task_15/`; report sections in
  `latex/sections/` (`task1.tex` = task 15, `task2.tex` = task 47 placeholder).
- Report states `N=10^5` nodes for task 15 (matching the config); keep text and
  config consistent.
- `data/`, `input/`, `notebooks/` are git-ignored (gitignored paths still exist
  locally).
- **Task 47** folders (`code/task_47/`, `config/task_47/`, `input/task_47/`,
  `data/task_47/`, `latex/figures/task_47/`) are intentionally empty. Do not
  populate them unless the user requests task 47 work.

## State of development (task 15)

- Simulations complete: `data/task_15/` contains part1 (gaussian/uniform) and
  part2 (all gamma values) binned distributions.
- Plots regenerated per the combined-1x2 convention (see Plot conventions).
- Exponent fitting (`fit_exponents.py`) available; outputs `data/task_15/fits.csv`.
- Report (`latex/sections/task1.tex`) includes both parts and the fits table.
- TODO tracker: `projects_info/todo.md` (currently all items resolved).
- Known follow-up (deferred): two-layer / interdependent networks
  (Brummitt et al.), and MLE exponent fitting on raw (unbinned) avalanche sizes.
