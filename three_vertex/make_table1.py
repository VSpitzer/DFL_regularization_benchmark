"""Table 1 of the article: regret (%) on the three-vertex problem, categories c1, c2, c3 (b = 1, 10, 100).

For each (method, regularization, category), the best configuration is selected on mean validation
regret over the ten seeds, separately without and with target normalization, over the learning rate,
the perturbation (alpha / sigma / beta), kappa (shrinkage, clipping) or lambda (L2 penalty); the option
with the lower validation regret is kept, and its mean test regret over the same ten runs is reported.
Markers: one-sided paired Wilcoxon signed-rank test over the seeds against the standard method selected
in the same way (** p <= 0.01, * p < 0.05); bold: best variant; italic: significantly worse.

    python make_table1.py [--results results/three_vertex.jsonl.gz] [--show_options]

Writes outputs/table1.tex and prints a text version.
"""
import argparse
import warnings
import gzip
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PERT = {"SPO": "alpha", "DPO": "sigma", "DBB": "lambda_val"}
REGS = [("std", "tn"), ("rs", "tnrs"), ("rc", "tnrc"), ("rn2", "tnrn"), ("l2p", "tnl2p")]
NAMES = ["standard", "r^s", "r^c", "r^n", "L2"]
COLS = ["", "-$r^s$", "-$r^c$", "-$r^n$", "-$L_2$"]
CATS = [(1, "c1"), (2, "c2"), (3, "c3")]


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    for r in rows:
        r.pop("probe_history", None)
    return pd.DataFrame(rows)


def best(d, model, inst):
    """Configuration with the lowest mean validation regret (complete configurations only)."""
    M = model.split("_")[0]
    x = d[(d.model == model) & (d.instance == inst)]
    hp = ["lr", PERT[M]] + [c for c in ("kappa", "lam") if c in x and x[c].notna().any()]
    g = x.groupby(hp)
    val = g.val_regret.mean()[g.seed.nunique() >= 10]
    if val.empty:
        return None
    cfg = val.idxmin()
    cfg = cfg if isinstance(cfg, tuple) else (cfg,)
    sel = x.loc[(x[hp] == pd.Series(cfg, index=hp)).all(axis=1)].sort_values("seed")
    return {"val": val.min(), "test": sel.test_regret.values, "cfg": dict(zip(hp, cfg))}


def choose(d, M, pair, inst):
    """Target normalization as a hyperparameter: keep the option with the lower validation regret."""
    opts = [(o, best(d, f"{M}_{v}", inst)) for o, v in zip(("raw", "tn"), pair)]
    opts = [(o, b) for o, b in opts if b is not None]
    o, b = min(opts, key=lambda ob: ob[1]["val"])
    return dict(b, option=o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results", "three_vertex.jsonl.gz"))
    ap.add_argument("--show_options", action="store_true", help="print the selected option and configuration")
    a = ap.parse_args()
    d = load(a.results)
    res = {(M, i, c): choose(d, M, pair, c) for M in PERT for i, pair in enumerate(REGS) for c, _ in CATS}

    rows, text = [], []
    for c, lab in CATS:
        cells, tcells = [], []
        for M in PERT:
            std = res[(M, 0, c)]["test"]
            means = [100 * res[(M, i, c)]["test"].mean() for i in range(len(REGS))]
            for i in range(len(REGS)):
                x = res[(M, i, c)]
                s = f"{means[i]:.2f}"
                t = s
                pl = pw = None
                if i > 0 and not np.allclose(x["test"], std):
                    pl = wilcoxon(x["test"], std, alternative="less").pvalue
                    pw = wilcoxon(x["test"], std, alternative="greater").pvalue
                if np.isclose(means[i], min(means)):
                    s = r"\textbf{" + s + "}"
                elif pw is not None and pw < 0.05:
                    s = r"\textit{" + s + "}"
                if pl is not None and pl < 0.05:
                    st = "**" if pl <= 0.01 else "*"
                    s += "$^{" + st.replace("*", r"\ast") + "}$"
                    t += st
                cells.append(s)
                tcells.append(t)
        rows.append(f"    {lab} & " + " & ".join(cells) + r" \\")
        text.append((lab, tcells))

    head = " & ".join(" & ".join([M] + [f"{M}{c}" for c in COLS[1:]]) for M in PERT)
    tex = r"""\begin{table*}[h]
  \centering
  \caption{Regret (\%) for the three-vertex problem. $^{\ast\ast}$: $p \le 0.01$, $^{\ast}$: $p < 0.05$.
  Bold: best variant; italic: significantly worse than the standard method.}
  \setlength{\tabcolsep}{2pt}
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{c|ccccc|ccccc|ccccc}
    & """ + head + r""" \\ \midrule
""" + "\n".join(rows) + r"""
  \end{tabular}}
  \label{tab:perf_3v}
\end{table*}
"""
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    open(os.path.join(HERE, "outputs", "table1.tex"), "w").write(tex)

    print("Table 1 -- regret (%)")
    for k, M in enumerate(PERT):
        print(f"\n{M:4} " + " ".join(f"{n:>10}" for n in NAMES))
        for lab, cells in text:
            print(f"{lab:4} " + " ".join(f"{v:>10}" for v in cells[5 * k:5 * k + 5]))
    if a.show_options:
        print("\nselected option and configuration")
        for (M, i, c), x in sorted(res.items()):
            print(f"  {M} {NAMES[i]:8} c{c}: {x['option']:3} {x['cfg']}")
    print(f"\nwrote {os.path.join(HERE, 'outputs', 'table1.tex')}")


if __name__ == "__main__":
    main()
