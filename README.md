# Controlling Scale Mismatch in Decision-Focused Learning via Cost Regularization

Code and results to reproduce the numerical experiments of the article (Section 6 and Appendices C–E):

| Article | What | Script |
|---|---|---|
| Table 1 | Three-vertex problem, regret of SPO / DPO / DBB with each regularization | `three_vertex/make_table1.py` |
| Figure 1 (a, b) | Three-vertex problem, stability ratio $\rho_\eta$ with and without shrinkage | `three_vertex/make_figure1.py` |
| Table 2 | Shortest-path and knapsack benchmarks, regret by degree $q$ | `benchmarks/make_table2.py` |
| Tables 4, 5 | Benchmarks, decision-change rate $\Delta$ and norm of the estimates | `benchmarks/make_tables4_5.py` |

The three-vertex code is adapted from the benchmark of Jayanta Mandi
(https://github.com/PredOpt/predopt-benchmarks); the benchmark data follow the PyEPO generator
(Tang & Khalil, 2024).

## Installation

Python 3.9:

```bash
conda create -n dfl_reg python=3.9 -y
conda activate dfl_reg
pip install -r requirements.txt
```

## Layout

```
three_vertex/                   Section 6.1 (three-vertex problem)
  run_three_vertex.py           trains a grid of configurations (ten seeds each), records rho_eta
  make_table1.py                Table 1
  make_figure1.py               Figure 1a and 1b
  rho_eta.py                    exact stability radius and stability ratio rho_eta of the problem
  Trainer/                      models (PO_models_rebuttal.py: every method x regularization),
                                regularizations (regularizers.py), problem and data loading
  DPO/                          perturbed optimizers (DPO)
  data/                         the 300 instances (rotation angles and true costs)
  specs/three_vertex.json       the grid of the article (2700 configurations)
  results/three_vertex.jsonl.gz the 27000 runs behind Table 1 and Figure 1
benchmarks/                     Section 6.2 (shortest path and knapsack)
  pyepo_benchmark.py            data generators, exact solvers, training of SPO / DPO / DBB
  regularizers.py               regularizations
  run_benchmarks.py             trains the grid of the article (ten seeds each)
  analysis.py                   selection protocol
  make_table2.py                Table 2 (and the two-stage baseline, the spread of the cost norms)
  make_tables4_5.py             Tables 4 and 5
  results/{sp,knapsack}.jsonl.gz the 54000 runs behind Tables 2, 4 and 5
```

## Reproducing the tables and figures from the stored runs

Each command takes a few seconds and writes LaTeX tables or PNG figures to the `outputs/` folder next to
the script:

```bash
cd three_vertex
python make_table1.py          # outputs/table1.tex  (--show_options: selected configurations)
python make_figure1.py         # outputs/rho_eta_drift_panel_rs.png, outputs/rho_eta_level_panel_rs.png

cd ../benchmarks
python make_table2.py          # outputs/table2.tex
python make_tables4_5.py       # outputs/table4.tex, outputs/table5.tex
```

## Rerunning the experiments

Both runners are resumable (runs already present in the output file are skipped), so an interrupted run is
continued by repeating the same command.

**Three-vertex problem** (27000 runs, about 25 CPU-hours):

```bash
cd three_vertex
python run_three_vertex.py --make_spec            # specs/three_vertex.json (already provided)
python run_three_vertex.py --spec specs/three_vertex.json --out results/three_vertex.jsonl \
    --shard 0 --nshards 4                         # one process per shard, 0..3
cat results/three_vertex.jsonl | gzip > results/three_vertex.jsonl.gz
```

**Benchmarks** (54000 runs, about 50 CPU-hours; `--deg` and `--method` split the work):

```bash
cd benchmarks
python run_benchmarks.py --problem sp --out results/sp.jsonl
python run_benchmarks.py --problem knapsack --out results/knapsack.jsonl
gzip -f results/sp.jsonl results/knapsack.jsonl
```

Training is seeded and deterministic: on CPU, a rerun reproduces the stored validation and test regrets,
up to the last digits of floating-point reductions.

## Experimental protocol

* **Methods**: SPO (perturbation $\alpha\theta - \bar\theta$), DPO (Gaussian noise $\sigma$, 5 samples),
  DBB (interpolation $\theta + \beta\bar\theta$).
* **Regularizations** of the estimate $\theta$ fed to the perturbation: shrinkage
  $r^s(\theta) = \theta/(1 + \|\theta\|/\kappa)$, clipping $r^c(\theta) = \min(1, \kappa/\|\theta\|)\,\theta$,
  normalization $r^n(\theta) = \theta/\|\theta\|$, and the $L_2$ penalty
  $\lambda\|\theta\|^2/\overline{\|\bar\theta\|}$ added to the loss, $\kappa$ being an absolute value.
* **Target normalization**, an optional preprocessing selected like any other hyperparameter: the model is
  trained on $\bar\theta\,\overline{\|\bar\theta\|}/\|\bar\theta\|$ ($\overline{\|\bar\theta\|}$: mean norm
  over the training set), and the regret is computed with the original costs.
* **Grids** (Appendix C):

  | | Three-vertex | Benchmarks |
  |---|---|---|
  | learning rate | $10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1$ | $10^{-2}, 10^{-1}, 1$ |
  | $\alpha$ (SPO) / $\sigma$ (DPO) / $\beta$ (DBB) | $\{0.5, 2, 10\}$ / $\{0.1, 0.5, 1\}$ / $\{0.1, 1, 10\}$ | same |
  | $\kappa$ (shrinkage, clipping) | $1, 100, 10^4$ | same |
  | $\lambda$ ($L_2$ penalty) | $0.1, 1$ | same |
  | target normalization | without, with | same |
  | training | MLP (10 hidden units), Adam, 30 epochs, scheduler | linear model, Adam, 30 epochs, batches of 32 |

* **Selection and testing**: every configuration is trained with ten seeds; for each method,
  regularization and problem, the configuration with the lowest mean validation regret is selected (target
  normalization included), and the mean test regret of the same ten runs is reported. Significance: one-sided
  paired Wilcoxon signed-rank test over the seeds against the standard method selected in the same way.
* **Three-vertex categories**: instances 1, 2, 3 of `Trainer/instability_problem.py` place vertex $C$ at
  $b = 1, 10, 100$ (categories c1, c2, c3).
* **Benchmarks**: shortest path on a $5\times5$ grid and knapsack with 32 items, PyEPO generator with degree
  $q \in \{1, 2, 4, 6, 8\}$, 1000 training, 250 validation and 1000 test samples, exact dynamic-programming
  solvers.
