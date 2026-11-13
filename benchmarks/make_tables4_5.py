"""Tables 4 and 5 of the article (Appendix E.1): decision-change rate Delta and norm of the estimates
during training, for the configurations selected in Table 2.

Table 4: growth factor of the mean ||theta|| over training and Delta after the first and the last epoch,
mean over q in {4, 6, 8} and the ten seeds. Table 5: Delta at the end of training, by degree.

    python make_tables4_5.py

Writes outputs/table4.tex and outputs/table5.tex.
"""
import os

import numpy as np

from analysis import DEGS, METHODS, PAIRS, PROBLEMS, TEX, SHORT, HERE, load, select, choose


def main():
    by, _ = load()
    res = choose(select(by))

    def runs(p, m, q, f):
        return res[(p, q, m, f)][1][2].values()

    t4, t5 = [], []
    print("Table 4 -- growth of ||theta|| and Delta first -> last (mean over q = 4, 6, 8)")
    for m in METHODS:
        for i, (f, _) in enumerate(PAIRS):
            cells, txt = [], []
            for p, _ in PROBLEMS:
                g, d0, d1 = [], [], []
                for q in (4, 6, 8):
                    for r in runs(p, m, q, f):
                        h = r["hist"]
                        g.append(h[-1]["norm"] / h[0]["norm"])
                        d0.append(h[0]["dist_same"])
                        d1.append(h[-1]["dist_same"])
                cells += [f"{np.mean(g):.1f}", f"{np.mean(d0):.2f} $\\to$ {np.mean(d1):.2f}"]
                txt.append(f"x{np.mean(g):.1f}  {np.mean(d0):.2f} -> {np.mean(d1):.2f}")
            lead = rf"\multirow{{5}}{{*}}{{{m}}}" if i == 0 else ""
            t4.append(f"{lead} & {TEX[f]} & " + " & ".join(cells) + r" \\")
            print(f"  {m:4} {f:5} shortest path {txt[0]:22} knapsack {txt[1]}")
            d = [np.mean([r["hist"][-1]["dist_same"] for r in runs(p, m, q, f)]) for p, _ in PROBLEMS for q in DEGS]
            t5.append(f"{lead} & {SHORT[f]} & " + " & ".join(f"{x:.2f}" for x in d) + r" \\")
        t4.append(r"\midrule")
        t5.append(r"\midrule")
    hdr = r"""\toprule
 & & \multicolumn{5}{c}{Shortest path} & \multicolumn{5}{c}{Knapsack} \\
\cmidrule(lr){3-7}\cmidrule(lr){8-12}
Method & Regularization & $q=1$ & $q=2$ & $q=4$ & $q=6$ & $q=8$ & $q=1$ & $q=2$ & $q=4$ & $q=6$ & $q=8$ \\
\midrule
"""
    tab4 = r"""\begin{table}[h]
\centering
\caption{Training diagnostics for the configurations of Table~\ref{tab:degree_sweep}: growth factor of
the mean $\|\vtheta\|$ over training and decision-change rate $\Delta$ after the first and the last epoch
(mean over $q \in \{4, 6, 8\}$ and the ten seeds).}
\label{tab:diagnostics}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrcrc}
\toprule
 & & \multicolumn{2}{c}{Shortest path} & \multicolumn{2}{c}{Knapsack} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Method & Regularization & Growth & $\Delta$: first $\to$ last & Growth & $\Delta$: first $\to$ last \\
\midrule
""" + "\n".join(t4[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    tab5 = r"""\begin{table}[h]
\centering
\caption{Decision-change rate $\Delta$ at the end of training for the configurations of
Table~\ref{tab:degree_sweep}, by degree $q$ (mean over the ten seeds).}
\label{tab:delta_final}
\small
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrrrrr}
""" + hdr + "\n".join(t5[:-1]) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    open(os.path.join(HERE, "outputs", "table4.tex"), "w").write(tab4)
    open(os.path.join(HERE, "outputs", "table5.tex"), "w").write(tab5)
    print(f"\nwrote {os.path.join(HERE, 'outputs', 'table4.tex')} and table5.tex")


if __name__ == "__main__":
    main()
