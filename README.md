# Project Name: Cascading Failures and Self-Organized Criticality in Complex Networks

## 1. Project Overview & Objectives
This project aims to study Self-Organized Criticality (SOC) through the Sandpile Model across different network topologies. The project is split into two main independent tasks:
- **Task 15 (Theoretical):** Simulates the behavior of dynamical models of cascading failures and analyzes their critical behavior across synthetic topologies (Gaussian, Uniform, Scale-Free), comparing them with theoretical Multiplicative Branching Process exponents. It also studies cascading failures in interconnected (two-layer) networks following Brummitt et al. This task is the active development focus.
- **Task 47 (Data):** Spatial covariance and Landau-Ginzburg models on empirical spatial networks (GridKit). **Not started** — its folders are placeholders.

## 2. Directory Structure & Content Descriptions

This repository follows a strict separation of concerns, with a further split by task (`task_15/`, `task_47/`) inside each top-level folder.

### `config/`
- `shared.yaml`: parameters common to both tasks (seed, plot style).
- `task_15/config.yaml`: task-specific parameters for theoretical simulations (network sizes, thresholds, gamma ranges).
- `task_47/`: task-specific parameters for data analysis.

### `input/`
- `task_47/`: raw input data for the empirical data task.

### `data/`
- `task_15/`: post-processed simulation outputs (e.g., cascade-size and lifetime probability arrays, fitted exponents).
- `task_47/`: post-processed datasets used for analysis.

### `code/`
- `common/`: utilities shared by both tasks (config loaders, plotting theme).
- `task_15/`: modular scripts for theoretical subtasks (Numba sandpile engine, Configuration Model generators).
- `task_47/`: modular scripts for empirical data subtasks.

### `latex/`
- `figures/task_15/`, `figures/task_47/`: publication-ready figures, one folder per task.
- `main.tex`: source including both task sections; compiles to `latex/main.pdf`.

### `projects_info/`
Task specifications and reference material (`theoretical_task/`, `data_task/`).

## 3. Main features description

### Task 15 — Theoretical
Implements the Sandpile Model dynamics to test Self-Organized Criticality. It features:
- Networks efficiently generated using the Configuration Model.
- Simulates load dissipation at specific boundary nodes for Gaussian and Uniform networks (Buldyrev et al., 2010).
- Simulates grain loss probability $f$ on Scale-Free networks for various exponents $\gamma$ (Motter & Lai, 2002).
- Interconnected R(3)-B(p)-R(3) networks: suppression of cascading failures at small interconnectivity $p$ (Brummitt et al., 2012).
- Highly optimized execution in Python via `numba` JIT compilation and parallelization over independent realizations via `multiprocessing`.

### Task 47 — Data
*(Not started — see the specification in `projects_info/data_task/project_47.md`.)*

---

## How to Run

All commands from the repository root. Uses the project's conda environment `complex-networks-env` (`conda activate complex-networks-env`).

### Setup Conda Environment
To ensure all dependencies are met, create and activate the project environment:
```bash
conda env create -f environment.yml
conda activate complex-networks-env
```

### Running the Simulations
Run the simulations for Part 1 (Gaussian & Uniform), Part 2 (Scale-Free), and Part 3 (interconnected networks). These scripts run parallel realizations and output the data arrays in `data/task_15/`:
```bash
python code/task_15/run_part1.py
python code/task_15/run_part2.py
python code/task_15/run_part3.py
```
*Note: parameters (like $N$, $\gamma$, or number of realizations) can be changed in `config/task_15/config.yaml` before running the code.*

### Generating Plots
Once the data is generated, plot the log-binned probability distributions and the interconnected-network figures (automatically saved in `latex/figures/task_15/`):
```bash
python code/task_15/plot_results.py
```

### Fitting Exponents
Fit power-law exponents on the binned distributions and print LaTeX tables (writes `data/task_15/fits.csv`):
```bash
python code/task_15/fit_exponents.py
python code/task_15/fit_exponents.py --sweep
```

### Building the Report
```bash
cd latex && latexmk -pdf main.tex
```
