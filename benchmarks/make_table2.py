"""Table 2 of the article: test regret (%) on the shortest-path and knapsack problems, by degree q.

Selection: see analysis.py (target normalization chosen on validation). Markers: one-sided paired
Wilcoxon signed-rank test over the seeds against the standard method selected in the same way
(** p <= 0.01, * p < 0.05); bold: best variant per method and degree. Also prints the two-stage
least-squares regret and the spread of the cost norms quoted in Appendix E.

    python make_table2.py [--show_options]

Writes outputs/table2.tex.
"""
import argparse
import os
import warnings

import numpy as np

from analysis import DEGS, METHODS, PAIRS, PROBLEMS, TEX, HERE, load, select, choose, test_mean, p_less
from pyepo_benchmark import make_problem

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show_options", action="store_true", help="print the selected option and configuration")
    a = ap.parse_args()
    by, ls = load()
    res = choose(select(by))
    rows, text = [], []
    for m in METHODS:
        for i, (f, _) in enumerate(PAIRS):
            cells, tcells = [], []
            for p, _ in PROBLEMS:
                for q in DEGS:
                    runs = res[(p, q, m, f)][1][2]
                    v = test_mean(runs)
                    allv = [test_mean(res[(p, q, m, g)][1][2]) for g, _ in PAIRS]
                    s = t = f"{v:.2f}"
                    if np.isclose(v, min(allv)):
                        s = r"\textbf{" + s + "}"
                    if f != "none":
                        pv = p_less(runs, res[(p, q, m, "none")][1][2])
                        if not np.isnan(pv) and pv < 0.05:
                            st = "**" if pv <= 0.01 else "*"
                            s += "$^{" + st.replace("*", r"\ast") + "}$"
                            t += st
                    cells.append(s)
                    tcells.append(t)
            lead = rf"\multirow{{5}}{{*}}{{{m}}}" if i == 0 else ""
            rows.append(f"{lead} & {TEX[f]} & " + " & ".join(cells) + r" \\")
            text.append((m, f, tcells))
        rows.append(r"\midrule")
    rows = rows[:-1]
    tex = r"""\begin{table}[t]
\centering
\caption{Test regret (\%) on the shortest-path and knapsack problems, by degree $q$.
$^{\ast\ast}$: $p \le 0.01$, $^{\ast}$: $p < 0.05$. Bold: best variant per method and degree.}
\label{tab:degree_sweep}
\small
\setlength{\tabcolsep}{2.5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrrrrr}
\toprule
 & & \multicolumn{5}{c}{Shortest path} & \multicolumn{5}{c}{Knapsack} \\
\cmidrule(lr){3-7}\cmidrule(lr){8-12}
Method & Regularization & $q=1$ & $q=2$ & $q=4$ & $q=6$ & $q=8$ & $q=1$ & $q=2$ & $q=4$ & $q=6$ & $q=8$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    open(os.path.join(HERE, "outputs", "table2.tex"), "w").write(tex)

    print("Table 2 -- test regret (%)          shortest path q=1..8 | knapsack q=1..8")
    for m, f, cells in text:
        print(f"{m:4} {f:5} " + " ".join(f"{c:>8}" for c in cells[:5]) + " | " +
              " ".join(f"{c:>8}" for c in cells[5:]))
    print("two-stage " + " ".join(f"{ls[('sp', q)]:8.2f}" for q in DEGS) + " | " +
          " ".join(f"{ls[('knapsack', q)]:8.2f}" for q in DEGS))
    for p, name in PROBLEMS:
        stats = []
        for q in DEGS:
            (_, Ctr, *_), _ = make_problem(p, q)
            nrm = np.linalg.norm(Ctr, axis=1)
            stats.append(f"q={q}: mean {nrm.mean():.1f}, max/min {nrm.max() / nrm.min():.1f}")
        print(f"cost norms, {name} training set: " + "; ".join(stats))
    if a.show_options:
        print("\nselected option and configuration (lr, perturbation, kappa, lambda)")
        for (p, q, m, f), (o, (v, cfg, _)) in sorted(res.items()):
            print(f"  {p:8} q={q} {m} {f:5}: {o:3} lr={cfg[4]} pert={cfg[5]} kappa={cfg[6]} lambda={cfg[7]}")
    print(f"\nwrote {os.path.join(HERE, 'outputs', 'table2.tex')}")


if __name__ == "__main__":
    main()
