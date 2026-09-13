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

