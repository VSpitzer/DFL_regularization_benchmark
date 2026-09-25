"""The two rho_eta figures, one panel each: axis labels and legend only.

    figs/rho_eta_drift_panel.png    drift of rho_eta, unregularized vs regularized
    figs/rho_eta_level_panel.png    level of rho_eta, unregularized vs regularized

One point per matched (standard, regularized) pair: a standard run and a
regularized run sharing the DFL method, the problem instance, the learning
rate and the perturbation hyperparameter, over the same ten seeds.  Colour is
the regularization map, marker is the DFL method, the dashed line is y = x, so
a point below the diagonal is a pair in which regularization reduced the
quantity plotted.

Both quantities are computed from the per-epoch history of rho_eta that
run_experiments.py logs (rho_eta.py defines it), then averaged over the pair's
ten seeds:

    level = mean of log10 rho_eta over the second half of training
            (epochs 15-29; the first half is initialisation transient),
            plotted as 10^(.) -- i.e. the geometric mean of rho_eta
    drift = max_t log10 rho_eta - min_t log10 rho_eta over all 30 epochs,
            plotted as the fold range 10^(.)

results/paper_grid.jsonl.gz already contains exactly the runs behind these
figures -- Table 3's hyperparameter grid, seeds 0-9, the two regularization
maps as the repository implements them -- so no filtering is needed here.

    python paired_rho_figures.py
    python paired_rho_figures.py --width 3.25      # NeurIPS column width
"""

import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np

SUFFIXES = ("_rn", "_rp")
# The fields that identify the shared configuration of a matched pair. The
# regularization's own knob (kappa) and the variant name are what differ
# within a pair, so they are excluded; everything else must agree.
PAIR_KEYS = ("instance", "lr", "alpha", "lambda_val", "sigma", "num_samples",
             "max_epochs")

# Colours checked for contrast in light mode.
COL = {"_rp": "#2a78d6", "_rn": "#1baf7a"}
LAB = {"_rp": r"$r^s$", "_rn": r"$r^n$"}
MARK = {"SPO": "o", "DBB": "s", "DPO": "^"}
METHODS = ["SPO", "DBB", "DPO"]

C_TEXT, C_MUTED, C_GRID, C_SURF = "#0b0b0b", "#52514e", "#e3e1dc", "#ffffff"


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------

def load(patterns):
    recs = []
    for p in patterns:
        for f in sorted(set(glob.glob(p)) | set(glob.glob(p + ".gz"))):
            op = gzip.open if f.endswith(".gz") else open
            with op(f, "rt") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        recs.append(json.loads(line))
    return recs


def split_model(name):
    for suf in SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)], suf
    return name, ""


def pair_key(r):
    return tuple((k, r.get(k)) for k in PAIR_KEYS)


def summarise(runs):
    """Level and drift of rho_eta for one configuration, meaned over its seeds."""
    level, drift = [], []
    for r in runs:
        h = [e["log10_rho_eta"] for e in (r.get("probe_history") or [])
             if np.isfinite(e.get("log10_rho_eta", np.nan))]
        if not h:
            continue
        level.append(float(np.mean(h[len(h) // 2:])))
        if len(h) > 1:
            drift.append(float(max(h) - min(h)))
    if not level or not drift:
        return None
    return {"level": float(np.mean(level)), "drift": float(np.mean(drift))}


def build_pairs(recs, n_seeds=10):
    by_cfg = defaultdict(list)
    for r in recs:
        base, suf = split_model(r["model"])
        by_cfg[(base, suf, pair_key(r), r.get("kappa"))].append(r)
    std = {(k[0], k[2]): v for k, v in by_cfg.items() if k[1] == ""}

    pairs = []
    for (base, suf, pk, kappa), runs in by_cfg.items():
        if suf == "":
            continue
        s_runs = std.get((base, pk))
        if not s_runs:
            continue
        seeds = sorted({r["seed"] for r in runs} & {r["seed"] for r in s_runs})
        if len(seeds) < n_seeds:
            continue
        a = summarise([r for r in s_runs if r["seed"] in seeds])
        b = summarise([r for r in runs if r["seed"] in seeds])
        if a is None or b is None:
            continue
        pairs.append({"base": base, "variant": suf, "kappa": kappa,
                      "instance": dict(pk)["instance"], "n_seeds": len(seeds),
                      "std_level": a["level"], "reg_level": b["level"],
                      "std_drift": a["drift"], "reg_drift": b["drift"]})
    return pairs


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def style(fs):
    return {
        "figure.facecolor": C_SURF, "axes.facecolor": C_SURF,
        "axes.edgecolor": "#8c8880", "axes.linewidth": 0.7,
        "axes.labelcolor": C_TEXT, "text.color": C_TEXT,
        "xtick.color": C_MUTED, "ytick.color": C_MUTED,
        "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1,
        "axes.labelsize": fs, "legend.fontsize": fs - 1,
        "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "font.size": fs,
    }


def panel(pairs, key, path, xlabel, ylabel, width=4.0, fs=9, dpi=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update(style(fs))

    a = np.array([p["std_" + key] for p in pairs])
    b = np.array([p["reg_" + key] for p in pairs])

    fig, ax = plt.subplots(figsize=(width, width))

    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    lim = [10 ** (lo - 0.15), 10 ** (hi + 0.15)]
    ax.plot(lim, lim, color=C_MUTED, lw=1.0, ls=(0, (3, 3)), zorder=2)

    for p in pairs:
        ax.scatter(10 ** p["std_" + key], 10 ** p["reg_" + key], s=20,
                   marker=MARK[p["base"]], color=COL[p["variant"]], alpha=0.75,
                   linewidths=0.5, edgecolors=C_SURF, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    handles = [Line2D([], [], color=COL[s], marker="o", ls="", ms=5, label=LAB[s])
               for s in ("_rp", "_rn") if any(p["variant"] == s for p in pairs)]
    handles += [Line2D([], [], color=C_MUTED, marker=MARK[m], ls="", ms=5, label=m)
                for m in METHODS if any(p["base"] == m for p in pairs)]
    handles.append(Line2D([], [], color=C_MUTED, lw=1.0, ls=(0, (3, 3)),
                          label="unchanged"))
    ax.legend(handles=handles, loc="upper left", handletextpad=0.5,
              borderaxespad=0.4, labelspacing=0.35)

    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*",
                    default=[os.path.join(here, "results", "paper_grid.jsonl")],
                    help="result files (.jsonl or .jsonl.gz)")
    ap.add_argument("--outdir", default=os.path.join(here, "figs"))
    ap.add_argument("--width", type=float, default=4.0,
                    help="figure side in inches (3.25 for a NeurIPS column)")
    ap.add_argument("--fontsize", type=float, default=9)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    recs = load(args.results)
    if not recs:
        raise SystemExit("no records found in {}".format(" ".join(args.results)))
    pairs = build_pairs(recs)
    if not pairs:
        raise SystemExit("no matched pairs -- check --results")

    print("{} runs -> {} matched pairs  |  {}  |  {}".format(
        len(recs), len(pairs),
        ", ".join("{} {}".format(m, sum(p["base"] == m for p in pairs))
                  for m in METHODS),
        ", ".join("{} {}".format(LAB[v].replace("$", ""),
                                 sum(p["variant"] == v for p in pairs))
                  for v in ("_rp", "_rn"))))

    os.makedirs(args.outdir, exist_ok=True)
    print("wrote", panel(
        pairs, "drift", os.path.join(args.outdir, "rho_eta_drift_panel.png"),
        r"$\max_t \rho_\eta / \min_t \rho_\eta$   (unregularized)",
        r"$\max_t \rho_\eta / \min_t \rho_\eta$   (regularized)",
        width=args.width, fs=args.fontsize, dpi=args.dpi))
    print("wrote", panel(
        pairs, "level", os.path.join(args.outdir, "rho_eta_level_panel.png"),
        r"mean $\rho_\eta$   (unregularized)",
        r"mean $\rho_\eta$   (regularized)",
        width=args.width, fs=args.fontsize, dpi=args.dpi))


if __name__ == "__main__":
    main()
