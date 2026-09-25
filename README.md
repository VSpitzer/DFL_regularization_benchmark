# Benchmarking Predict-then-Optimize Problems

This repository is intended to support the reproducibility of the numerical experiments presented in the article "Managing Solution Stability in Decision-Focused Learning with Cost Regularization". It is primarily based on an adaptation of the code originally developed by Jayanta Mandi: https://github.com/PredOpt/predopt-benchmarks

## Installation

### Setup
Prerequisite : conda package manager

1. Install Conda by following the [official installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. Create and activate the environment:
```bash
# Create environment
conda create -n benchmarking_env python=3.9 -y

# Activate on Linux/macOS
conda activate benchmarking_env

# Activate on Windows
pip install -r requirements.txt
```

## Running Experiments

Navigate to the corresponding experiment directory to run specific benchmarks.

### Data

- **Matching**: download the preprocessed CORA dataset before running any experiment. From the `Matching/` directory, run:
  ```bash
  ./get_data.sh
  ```
  (On Windows, there is no bash to run this script directly — run its commands one by one instead: `pip install gdown`, then `gdown 1MNy9HCVkJykRbXf6XXI9D7lggF0UF8MP`, `tar -xvzf data.tar.gz`, delete `data.tar.gz`, and finally `cd data` followed by `python make_cora_dataset.py`.) This creates a `data/` folder with the files the experiments expect. Alternatively, download the dataset directly from https://doi.org/10.48804/KT2P3Z and extract the `tar.gz` archive into `Matching/data/`.

- **ShortestPath**: download the shortest path dataset from https://doi.org/10.48804/KT2P3Z, then, from the `ShortestPath/` directory, extract it:
  ```bash
  tar -xvzf ShortestPathData.tar.gz
  ```

### Run configurations

`Matching/test_matching.py` and `ShortestPath/test_sp.py` both accept a `--config` flag selecting which JSON file of hyperparameter settings to run:

- `config.json` (default) — the best/tuned hyperparameters for each model and problem instance.
- `config_grid.json` — the full hyperparameter grid searched in the paper.

```bash
# Best/tuned hyperparameters (default)
python test_matching.py --scheduler True --config config.json
python test_sp.py --scheduler True --config config.json

# Full hyperparameter grid from the paper
python test_matching.py --scheduler True --config config_grid.json
python test_sp.py --scheduler True --config config_grid.json
```

### The instability problem

`instability_problem/test_instability_problem.py` takes the same `--config`
flag. Its data ships with the repository, so nothing has to be downloaded. Each
run writes one CSV per model/loss/instance into `instability_problem/Rslt/`, and
`best_results.py` reduces those to one line per (model, instance): the
hyperparameter combination with the lowest mean validation regret over the ten
seeds, and that combination's mean test regret.

```bash
cd instability_problem

# Best/tuned hyperparameters (default)
python test_instability_problem.py --scheduler True --config config.json

# Full hyperparameter grid from the paper
python test_instability_problem.py --scheduler True --config config_grid.json

# Summarise a grid run
python best_results.py
```

### Solution stability during training

`instability_problem/stability_study/` measures, epoch by epoch, the ratio the
perturbation-based methods are governed by,

    rho_eta = ||delta|| / eta(theta~)

the perturbation norm in units of the stability radius of the point being
perturbed, and produces the two figures comparing each standard run with its
regularized counterpart:

```
stability_study/figs/rho_eta_drift_panel.png    how far rho_eta travels during training
stability_study/figs/rho_eta_level_panel.png    where rho_eta sits
```

Drawing them from the stored results takes one command, from anywhere in the
repository:

```bash
python instability_problem/stability_study/paired_rho_figures.py
```

Regenerating the results behind them first (315 configurations x 10 seeds, about
five CPU-hours; the run is resumable, so rerunning the same command picks up
where it stopped, and `--shard i --nshards n` splits it across processes):

```bash
cd instability_problem/stability_study
python run_experiments.py --spec specs/paper_grid.json --out results/paper_grid.jsonl
gzip -f results/paper_grid.jsonl
python paired_rho_figures.py
```

Training is deterministic and the probe draws its noise from its own generator,
so a rerun reproduces every record of `results/paper_grid.jsonl.gz` exactly.
`Trainer/PO_models.py` is not modified: `instrument.py` reads it, applies its
additive edits to the source text and executes the result in memory. See
`instability_problem/stability_study/README.md` for the definition of `rho_eta`,
of the two summaries plotted, and of the scope of the comparison.

