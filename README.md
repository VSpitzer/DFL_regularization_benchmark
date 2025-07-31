# Benchmarking Predict-then-Optimize (PtO) Problems

This repository's purpose is the reproducibility of the numerical experiments led in the article: "Managing Solution Stability in Decision-Focused Learning: On the Importance of Cost Vector Regularization". It consists mostly in an adaptation of the code proposed by Jayanta Mandi: https://github.com/PredOpt/predopt-benchmarks

## Installation

### Prerequisites
- Python 3.7.3 (recommended)
- pip or conda package manager

### Option 1: Using venv (Recommended)

1. Create and activate a virtual environment:
```bash
python3 -m venv benchmarking_env
source benchmarking_env/bin/activate
```

2. Upgrade pip:
```bash
pip install --upgrade pip
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Option 2: Using Conda

1. Install Conda by following the [official installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. Create and activate the environment:
```bash
# Create environment
conda env create -n benchmarking_env --file environment.yml

# Activate on Linux/macOS
conda activate benchmarking_env

# Activate on Windows
source activate benchmarking_env
```

## Running Experiments

Navigate to the corresponding experiment directory to run specific benchmarks.

