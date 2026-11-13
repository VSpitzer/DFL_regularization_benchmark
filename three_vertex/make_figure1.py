"""Figure 1 of the article: cost vector stability ratio rho_eta with and without the shrinkage r^s.

One point per (method, category, learning rate, perturbation, target-normalization option, kappa) on the
seven (learning rate, perturbation) settings below, comparing r^s with the standard method under the same
setting and option, both averaged over the ten seeds:
    (a) drift = max_t log10 rho_eta - min_t log10 rho_eta over training, plotted as the fold range;
    (b) level = mean of log10 rho_eta over the second half of training, plotted as 10^(.).
rho_eta is the per-epoch history logged by run_three_vertex.py (see rho_eta.py).

    python make_figure1.py [--results results/three_vertex.jsonl.gz] [--width 4.0]

Writes outputs/rho_eta_drift_panel_rs.png and outputs/rho_eta_level_panel_rs.png, and prints the numbers
quoted in Section 6.1.
"""
import argparse
import gzip
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERT = {"SPO": "alpha", "DPO": "sigma", "DBB": "lambda_val"}
# the seven (learning rate, perturbation) settings per method
SLICE = {("SPO", 0.1, 0.5), ("SPO", 0.1, 2.0), ("SPO", 0.1, 10.0), ("SPO", 1e-4, 2.0), ("SPO", 1e-3, 2.0),
         ("SPO", 1e-2, 2.0), ("SPO", 1.0, 2.0),
         ("DPO", 1e-2, 0.1), ("DPO", 1e-2, 0.5), ("DPO", 1e-2, 1.0), ("DPO", 1e-4, 1.0), ("DPO", 1e-3, 1.0),
         ("DPO", 0.1, 1.0), ("DPO", 1.0, 1.0),
         ("DBB", 0.1, 0.1), ("DBB", 0.1, 1.0), ("DBB", 0.1, 10.0), ("DBB", 1e-4, 1.0), ("DBB", 1e-3, 1.0),
         ("DBB", 1e-2, 1.0), ("DBB", 1.0, 1.0)}
COLK = {1.0: "#2a78d6", 100.0: "#1baf7a", 10000.0: "#d9622b"}
LABK = {1.0: r"$\kappa = 1$", 100.0: r"$\kappa = 100$", 10000.0: r"$\kappa = 10^4$"}
MARK = {"SPO": "o", "DBB": "s", "DPO": "^"}
C_TEXT, C_MUTED, C_GRID, C_SURF = "#0b0b0b", "#52514e", "#e3e1dc", "#ffffff"


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def summarise(runs):
    level, drift = [], []
    for r in runs:
        h = [e["log10_rho_eta"] for e in r["probe_history"] if np.isfinite(e.get("log10_rho_eta", np.nan))]
        if len(h) > 1:
            level.append(np.mean(h[len(h) // 2:]))
            drift.append(max(h) - min(h))
    return (float(np.mean(level)), float(np.mean(drift))) if level else None


def pairs(recs):
    by = defaultdict(list)
    for r in recs:
        M, v = r["model"].split("_")
        if v not in ("std", "tn", "rs", "tnrs") or "probe_history" not in r:
            continue
        if (M, round(r["lr"], 8), r[PERT[M]]) not in SLICE:
            continue
        opt = "tn" if v.startswith("tn") else "raw"
        reg = "rs" if v.endswith("rs") else "std"
        by[(M, r["instance"], r["lr"], r[PERT[M]], opt, reg, r.get("kappa"))].append(r)
    out = []
    for (M, inst, lr, pert, opt, reg, kappa), runs in by.items():
        if reg != "rs":
            continue
        std = by.get((M, inst, lr, pert, opt, "std", None))
        if not std:
            continue
        seeds = sorted({r["seed"] for r in runs} & {r["seed"] for r in std})
        if len(seeds) < 10:
            continue
        a = summarise([r for r in std if r["seed"] in seeds])
        b = summarise([r for r in runs if r["seed"] in seeds])
        if a and b:
            out.append(dict(method=M, instance=inst, option=opt, kappa=float(kappa),
                            std_level=a[0], reg_level=b[0], std_drift=a[1], reg_drift=b[1]))
    return out


def panel(P, key, path, xlabel, ylabel, width, fs=9, dpi=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({
        "figure.facecolor": C_SURF, "axes.facecolor": C_SURF, "axes.edgecolor": "#8c8880",
        "axes.linewidth": 0.7, "axes.labelcolor": C_TEXT, "text.color": C_TEXT,
        "xtick.color": C_MUTED, "ytick.color": C_MUTED, "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1,
        "axes.labelsize": fs, "legend.fontsize": fs - 2, "axes.grid": True, "grid.color": C_GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "font.size": fs})
    a = np.array([p["std_" + key] for p in P])
    b = np.array([p["reg_" + key] for p in P])
    fig, ax = plt.subplots(figsize=(width, width))
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    lim = [10 ** (lo - 0.15), 10 ** (hi + 0.15)]
    ax.plot(lim, lim, color=C_MUTED, lw=1.0, ls=(0, (3, 3)), zorder=2)
    for p in P:
        ax.scatter(10 ** p["std_" + key], 10 ** p["reg_" + key], s=22, marker=MARK[p["method"]],
                   color=COLK[p["kappa"]], edgecolors=C_SURF, linewidths=0.5, alpha=0.8, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    h = [Line2D([], [], color=COLK[k], marker="o", ls="", ms=5, label=LABK[k]) for k in COLK]
    h += [Line2D([], [], color=C_MUTED, marker=MARK[m], ls="", ms=5, label=m) for m in ("SPO", "DBB", "DPO")]
    h += [Line2D([], [], color=C_MUTED, lw=1.0, ls=(0, (3, 3)), label="unchanged")]
    ax.legend(handles=h, loc="upper left", handletextpad=0.5, borderaxespad=0.4, labelspacing=0.3)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results", "three_vertex.jsonl.gz"))
    ap.add_argument("--width", type=float, default=4.0, help="figure side in inches")
    a = ap.parse_args()
    P = pairs(load(a.results))
    out = os.path.join(HERE, "outputs")
    os.makedirs(out, exist_ok=True)
    print("wrote", panel(P, "drift", os.path.join(out, "rho_eta_drift_panel_rs.png"),
                         r"$\max_t \rho_\eta / \min_t \rho_\eta$   (unregularized)",
                         r"$\max_t \rho_\eta / \min_t \rho_\eta$   (regularized)", a.width))
    print("wrote", panel(P, "level", os.path.join(out, "rho_eta_level_panel_rs.png"),
                         r"mean $\rho_\eta$   (unregularized)", r"mean $\rho_\eta$   (regularized)", a.width))
    print(f"\n{len(P)} pairs")
    for k in (1.0, 100.0, 10000.0):
        q = [p for p in P if p["kappa"] == k]
        sd = np.array([p["std_drift"] for p in q])
        rd = np.array([p["reg_drift"] for p in q])
        big = sd > np.log10(5)
        print(f"kappa = {k:g}: max-min ratio decreases in {int((rd < sd - 0.02).sum())} of {len(q)} "
              f"configurations; where it exceeds 5 ({int(big.sum())}), median {10 ** np.median(sd[big]):.1f} -> "
              f"{10 ** np.median(rd[big]):.1f}")


if __name__ == "__main__":
    main()
