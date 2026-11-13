"""Selection protocol shared by the benchmark tables.

For each (problem, degree, method, regularization), the configuration (learning rate, perturbation,
kappa or lambda) with the lowest mean validation regret over the ten seeds is selected, separately
without and with target normalization; the option with the lower validation regret is kept (target
normalization as a hyperparameter). The cell is the mean test regret of the ten runs of the selection.
"""
import gzip
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
DEGS = [1, 2, 4, 6, 8]
PROBLEMS = [("sp", "Shortest path"), ("knapsack", "Knapsack")]
METHODS = ["SPO", "DPO", "DBB"]
# regularization -> (family without, family with target normalization)
PAIRS = [("none", "tn"), ("rs", "tn_rs"), ("rc", "tn_rc"), ("rn", "tn_rn"), ("l2p", "tn_l2p")]
TEX = {"none": "standard", "rs": "$r^s$ (shrinkage)", "rc": "$r^c$ (clipping)", "rn": "$r^n$ (normalization)",
       "l2p": "$L_2$ penalty"}
SHORT = {"none": "standard", "rs": "$r^s$", "rc": "$r^c$", "rn": "$r^n$", "l2p": "$L_2$"}


def load(paths=None):
    paths = paths or [os.path.join(HERE, "results", f"{p}.jsonl.gz") for p, _ in PROBLEMS]
    by = defaultdict(dict)
    ls = {}
    for path in paths:
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                cfg = (r["problem"], r["deg"], r["method"], r["reg"], r["lr"], r["pert"], r["kappa_abs"], r["lam"])
                by[cfg][r["seed"]] = r
                ls[(r["problem"], r["deg"])] = 100 * r["ls_test"]
    return by, ls


def select(by, n_seeds=10):
    """Best configuration per (problem, degree, method, family) on mean validation regret."""
    best = {}
    for cfg, runs in by.items():
        if len(runs) < n_seeds:
            continue
        k = cfg[:4]
        v = np.mean([r["val"] for r in runs.values()])
        if k not in best or v < best[k][0]:
            best[k] = (v, cfg, runs)
    return best


def choose(best):
    """Target normalization as a hyperparameter: (option, (val, cfg, runs)) per regularization."""
    res = {}
    for (p, q, m, f) in list(best):
        for raw, tn in PAIRS:
            if f != raw:
                continue
            opts = [(o, best[(p, q, m, fam)]) for o, fam in (("raw", raw), ("tn", tn)) if (p, q, m, fam) in best]
            res[(p, q, m, raw)] = min(opts, key=lambda ob: ob[1][0])
    return res


def test_mean(runs):
    return 100 * np.mean([r["test"] for r in runs.values()])


def p_less(a, b):
    """One-sided paired Wilcoxon signed-rank test over the common seeds (nan if identical runs)."""
    s = sorted(set(a) & set(b))
    x = np.array([a[i]["test"] for i in s])
    y = np.array([b[i]["test"] for i in s])
    return np.nan if np.allclose(x, y) else wilcoxon(x, y, alternative="less").pvalue
