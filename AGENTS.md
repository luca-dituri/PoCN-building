# AGENTS.md

Project context for AI agents and contributors working on this repository.

## Project overview

**Cascading Failures and Self-Organized Criticality in Complex Networks** — an
academic project with two independent tasks:

- **Task 15 (Theoretical):** Sandpile-model cascading failures across synthetic
  topologies (Gaussian, Uniform, Scale-Free) compared with multiplicative
  branching-process predictions, plus interconnected (two-layer) R(z)-B(p)-R(z)
  networks following Brummitt et al. This task is the active development focus.
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
- No standalone test suite and no linter configured. Verify by executing the
  relevant `run_*`/`plot_*` scripts and re-compiling the report.

## How to run

All commands from the repository root unless noted.

```bash
# Task 15 simulations (parallel realizations -> data/task_15/*.npz)
python code/task_15/run_part1.py
python code/task_15/run_part2.py
python code/task_15/run_part3.py   # R(3)-B(p)-R(3) interconnected networks

# Plots (all parts + Brummitt Figs 4-6 -> latex/figures/task_15/)
python code/task_15/plot_results.py

# Exponent fitting on existing binned data (-> data/task_15/fits.csv, tables)
python code/task_15/fit_exponents.py

# Sensitivity of the fitted exponents to the fitting window (--sweep)
python code/task_15/fit_exponents.py --sweep

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
  `plot_utils.py` (shared theme via `setup_plot_style` — reads `plot_style` from
  `config/shared.yaml`, standard font sizes, `get_figure_dir`,
  `get_log_binned_distribution`). All plot scripts call `setup_plot_style` and
  import font-size constants from here so every figure shares one consistent
  style; do not set per-script rcParams.
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
  config consistent. Part 3 uses `N=2*10^3` nodes per network (two layers) with
  `f=0.01` and `2*10^6` grains after a 20% transient.
- Part 3 (interconnected) tracks per-avalanche topplings split by network
  (`ta`, `tb`) plus the origin network; "big" cascade = `ta > 0.5*N`. Figures
  4-6 follow the Brummitt paper (primary axes linear in `p`; the Fig 4/5 insets
  are log-log rank-size plots of the largest 10^4 events). Fig 4/5 insets are
  INSIDE the main plot (top-right, via `ax.inset_axes`) in the empty band left
  by expanding the y-axis (downward translation of the data); they show the
  **per-run average** rank-size curve (mean over the `realizations` of each
  run's sorted top-`top_k`), short y-labels (`T_a` / `t`), no title, lines kept
  between points; `run_part3.py` stores `top_{ta,total}_mean_{j}/std_{j}`.
  Main legends sit at 'upper left'. Left margins in `fig.add_axes` must be wide
  enough for the long rotated y-labels (previously clipped). Fig 5 mean-field
  reference is `s(t) ~ 0.5 t^{-3/2}`.
  Fig 6 has no per-panel axis labels: shared `supxlabel('Interconnectivity p')`,
  bold colored row labels "Small cascades" (blue) / "Large cascades" (red) on
  the outside, and a single rotated "Probability" label inside the plot area.
- `data/`, `input/`, `notebooks/` are git-ignored (gitignored paths still exist
  locally).
- **Task 47** folders (`code/task_47/`, `config/task_47/`, `input/task_47/`,
  `data/task_47/`, `latex/figures/task_47/`) are intentionally empty. Do not
  populate them unless the user requests task 47 work.

## State of development (task 15)

- Simulations complete: `data/task_15/` contains part1 (gaussian/uniform),
  part2 (all gamma values) binned distributions, and part3 (interconnected
  R(3)-B(p)-R(3)) summary npz (`part3_{fig4,fig5,fig6}.npz`).
- Plots regenerated per the combined-1x2 convention (see Plot conventions);
  part 3 figures `task_15_part3_fig{4,5,6}.pdf` reproduce Brummitt Figs 4-6
  (P(big cascade) minimized at p*~0.075, global cascades grow with p, Fig 6
  size-window trade-off).
- Exponent fitting (`fit_exponents.py`) available; outputs `data/task_15/fits.csv`.
- Report (`latex/sections/task1.tex`) includes all three parts and the fits table.
