---
name: general-guidlines
description: General guidelines to structure the project and for the management of it. Always follow these guidelines. These guidelines are meant to ensure reproducibility and good scientific practices.
---

# Scientific Project Architecture

## Objective
Establish a clean, reproducible, and production-ready workspace for scientific computation and network analysis. This structure enforces a strict unidirectional data flow, separates exploratory chaos from stable code, and isolates compute-heavy state files. The project covers **two independent tasks** (Task 15 – theoretical, Task 47 – data), which must be kept cleanly separated at the level of code, data, and figures, while sharing the same top-level skeleton and a single final report. Always try to use this architecture for your projects.

## Project Folder Structure

The top-level skeleton is fixed. Inside `data/`, `notebooks/`, `input/`, `code/` and `latex/`, content is further split into one subfolder per task (`task_15/`, `task_47/`), so the two tasks never mix files.

```
my_project/
├── config/                  # Global parameters, thresholds, environment variables
│   ├── shared.yaml          # Parameters common to both tasks (paths, seeds, plotting style...)
│   ├── task_15/              # Config specific to Task 15 (theoretical)
│   └── task_47/              # Config specific to Task 47 (data)
│
├── data/                    # Output data only (post-processed, ready to plot/report)
│   ├── task_15/
│   └── task_47/
│
├── notebooks/                # Non-production sandbox for exploration and prototyping Not committed to git
│   ├── task_15/
│   └── task_47/
│
├── input/                    # Raw/input working data, NEVER committed to git
│   ├── task_15/
│   └── task_47/
│
├── code/                      # Modular, reusable source code to run the analysis
│   ├── task_15/
│   └── task_47/
│
├── latex/                     # LaTeX source and figures for the report (use the given template)
│   ├── figures/
│   │   ├── task_15/            # Figures used only in the Task 15 section
│   │   └── task_47/            # Figures used only in the Task 47 section
│   └── main.tex                # Single source that includes both task sections
│
└── report.pdf                 # Compiled LaTeX report, covering BOTH tasks
```

**Rules for the split:**
- Nothing inside `task_15/` may import from or write to `task_47/`, and vice versa (and the same for their `input/`, `data/`, `code/`, `config/`, `latex/figures/` subfolders). If something is genuinely shared (e.g. a generic plotting utility, a common network-generation helper), put it in a `code/common/` folder instead of duplicating it into both tasks.
- Every figure filename should make the task explicit even though it already lives in a task-specific folder (e.g. `task_15_gcc_vs_p.pdf`), to avoid ambiguity once figures are copied into `latex/`.
- `report.pdf` remains a single compiled document with clearly separated sections/chapters for Task 15 and Task 47 (see Report Writing below).

## Conda Environment

Always run scripts and, if necessary, install packages in a project-specific environment (called e.g. `projname-env`).
If it is not present, create it; otherwise activate and use it.
Generate if not present, and keep updated, a `.yml` file describing the content of the conda environment.
Use a single shared environment for both tasks unless dependencies genuinely conflict; if they do, document two environments (`projname-task15-env`, `projname-task47-env`) and note which one each script requires.

## Script Runs

Run the scripts using the project-specific conda environment.
When possible, put also a progress bar indicating the advancement of the run, or if not possible, an estimate of the remaining time.
Each script should make clear (via its path, e.g. `code/task_15/...`, and/or a header comment) which task it belongs to.

## Gitignore

Automatically populate the root `.gitignore` file to ensure huge datasets, binary files, or local IDE metadata are not committed to version control. Ensure it includes at least:

```
# Data and heavy storage boundaries
/input/

# All the development code
/notebooks/

# Environments and heavy distributions
.conda/
env/
venv/
.env
*.pyc
__pycache__/

# IDEs and OS overhead
.ipynb_checkpoints/
.DS_Store
.vscode/
.idea/

# .tex build files
*.aux
*.log
*.out
*.toc
*.bbl
*.blg
*.synctex.gz
```

## README

Keep updated the `README.md` file, structured like the template below and reflecting the two-task split:

```
# Project Name: [Clear, Descriptive Title of the Research Project]

[![DOI](https://img.shields.io/badge/DOI-10.1000%2Fxyz123-blue.svg)](https://doi.org/10.1000/xyz123)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Project Overview & Objectives
*Concise abstract covering both Task 15 (theoretical) and Task 47 (data): core objectives and computational methods used to test them.*

## 2. Directory Structure & Content Descriptions

This repository follows a strict separation of concerns, with a further split by task (`task_15/`, `task_47/`) inside each top-level folder.

### `config/`
- `shared.yaml`: parameters common to both tasks.
- `task_15/`, `task_47/`: task-specific parameters, thresholds, environment variables.

### `input/`
- `task_15/`: *raw input for the theoretical/simulation task (e.g. reference parameter sets)*.
- `task_47/`: *raw input data for the data task*.

### `data/`
- `task_15/`: *post-processed simulation outputs (e.g. cascade-size statistics per topology)*.
- `task_47/`: *post-processed datasets used for analysis*.

### `code/`
- `common/`: utilities shared by both tasks (network generators, stats helpers, plotting style).
- `task_15/`: modular scripts for each theoretical subtask (model implementation, experiment runner).
- `task_47/`: modular scripts for each data-task subtask.

### `notebooks/`
- `task_15/`, `task_47/`: exploratory notebooks, not production code.

### `latex/`
- `figures/task_15/`, `figures/task_47/`: publication-ready figures, one folder per task.
- `main.tex`: source including both task sections; compiles to `report.pdf`.

### `report.pdf`
Final compiled report covering both Task 15 and Task 47.

## 3. Main features description
### Task 15 — Theoretical
*Logic of the cascading-failure models implemented, and the assumptions behind them.*

### Task 47 — Data
*Logic of the data pipeline/network creation, and the analysis hypotheses tested.*
```

## Work To Implement:

### Task 15 — Theoretical: Cascading Failures
The objective is to simulate the behavior of dynamical models of cascading failures and analyze their critical behavior across various synthetic (and potentially real-world) topologies.

The project is based on the following three papers, found in `theoretical_task/`:
- **Paper 1:** [Suppressing Cascades of Load in Interdependent Networks (Brummitt et al. 2012)](./theoretical_task/brummitt-et-al-2012-suppressing-cascades-of-load-in-interdependent-networks.pdf)
- **Paper 2:** [Cascading Failures in Complex Networks (Motter & Lai 2002)](./theoretical_task/PhysRevLett.91.148701.pdf)
- **Paper 3:** [Catastrophic Cascade of Failures in Interdependent Networks (Buldyrev et al. 2010)](./theoretical_task/jpsj.64.327.pdf)

### Implementation:

Work only on the coding part of this project, I want you to read the articles and implement the following analysis:


---
