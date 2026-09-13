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

