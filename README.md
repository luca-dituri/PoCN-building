# Cascading Failures and Self-Organized Criticality in Complex Networks

## Overview & Objectives

This project studies Self-Organized Criticality (SOC) through the Sandpile Model across different network topologies. It is split into two independent tasks:

- **Task 15 (Theoretical):** Sandpile-model cascading failures on synthetic topologies (Gaussian, Uniform, Scale-Free), compared with multiplicative branching-process predictions, plus interconnected two-layer R(3)-B(p)-R(3) networks following Brummitt et al. This task is the active development focus.
- **Task 47 (Data):** Spatial covariance and Landau-Ginzburg models on empirical spatial networks (GridKit). Implemented — see `TASK_47.md`.

## Directory Structure

```
config/          parameters (shared.yaml + per-task YAML) — source of truth
code/
  common/        shared helpers (config loader, seeds, plot theme, log-binning)
  task_15/       simulation/plot/fit scripts (numba sandpile, configuration model)
  task_47/       spatial-covariance / Landau-Ginzburg pipeline (see TASK_47.md)
data/            post-processed outputs (git-ignored)
input/           raw inputs (git-ignored)
latex/           report: main.tex, sections/, figures/, bibliography.bib
projects_info/   task specifications and reference papers
.opencode/       project skill (canonical run/convention reference for agents)
```

## Setup

```bash
conda env create -f environment.yml
conda activate complex-networks-env
```

## How to Run

See `AGENTS.md` for the agentic workflow, and the `complex-networks-project`
skill (`.opencode/skills/complex-networks-project/SKILL.md`) for the exact run
commands, code conventions, and plot conventions.
