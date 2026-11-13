"""Shortest-path and knapsack benchmarks (Section 6.2): train the grid of the article with ten seeds.

Grid (Appendix C): problem in {sp, knapsack}, degree q in {1, 2, 4, 6, 8}, method in {SPO, DPO, DBB},
learning rate in {0.01, 0.1, 1}, perturbation alpha (SPO) in {0.5, 2, 10}, sigma (DPO) in {0.1, 0.5, 1},
beta (DBB) in {0.1, 1, 10}, and, each without and with target normalization:
    standard; shrinkage r^s and clipping r^c with kappa in {1, 100, 1e4}; normalization r^n;
    L2 penalty with lambda in {0.1, 1}.
That is 5400 configurations, 54000 runs. Each record holds the configuration, the seed, the validation
and test regrets, the two-stage least-squares regret of the data set and the training diagnostics
("hist": mean norm of the estimates and decision-change rate every five epochs).

    python run_benchmarks.py --problem sp --out results/sp.jsonl [--deg 1 2] [--method DPO]
    python run_benchmarks.py --problem knapsack --out results/knapsack.jsonl

The run is resumable: runs already present in --out are skipped. Training is deterministic, so a rerun
reproduces the records of results/*.jsonl.gz.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pyepo_benchmark import make_problem, train, two_stage  # noqa: E402

DEGS = [1, 2, 4, 6, 8]
LRS = [0.01, 0.1, 1.0]
PERT = {"SPO": [0.5, 2.0, 10.0], "DPO": [0.1, 0.5, 1.0], "DBB": [0.1, 1.0, 10.0]}
KAPPAS = [1.0, 100.0, 10000.0]
LAMS = [0.1, 1.0]


def variants():
    """(reg, kappa, lam) for every regularization, without and with target normalization."""
    out = []
    for tn in ("", "tn_"):
        out.append(("tn" if tn else "none", None, None))
        out += [(tn + "rs", k, None) for k in KAPPAS]
        out += [(tn + "rc", k, None) for k in KAPPAS]
        out.append((tn + "rn", None, None))
        out += [(tn + "l2p", None, lam) for lam in LAMS]
    return out


def key(problem, deg, method, lr, pert, reg, kappa, lam, seed):
    return json.dumps([problem, deg, method, lr, pert, reg, kappa, lam, seed])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["sp", "knapsack"], required=True)
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--deg", type=int, nargs="+", default=DEGS)
    ap.add_argument("--method", nargs="+", default=["SPO", "DPO", "DBB"])
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add(key(r["problem"], r["deg"], r["method"], r["lr"], r["pert"], r["reg"], r["kappa_abs"],
                         r["lam"], r["seed"]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    for deg in a.deg:
        (Xtr, Ctr, Xva, Cva, Xte, Cte), solve = make_problem(a.problem, deg)
        ls = two_stage(Xtr, Ctr, Xte, Cte, solve)
        print(f"[{a.problem}] q = {deg}: two-stage (least squares) test regret {100 * ls:.2f}%", flush=True)
        for method in a.method:
            for lr in LRS:
                for pert in PERT[method]:
                    for reg, kappa, lam in variants():
                        for seed in range(a.seeds):
                            if key(a.problem, deg, method, lr, pert, reg, kappa, lam, seed) in done:
                                continue
                            te, va, h = train(method, reg, Xtr, Ctr, Xva, Cva, Xte, Cte, solve, lr=lr, pert=pert,
                                              lam=0.1 if lam is None else lam, seed=seed, kappa_abs=kappa)
                            with open(a.out, "a") as f:
                                f.write(json.dumps({"problem": a.problem, "deg": deg, "method": method, "reg": reg,
                                                    "lr": lr, "pert": pert, "kappa_rel": None, "lam": lam,
                                                    "kappa_abs": kappa, "seed": seed, "test": te, "val": va,
                                                    "ls_test": ls, "hist": h}) + "\n")
                        print(f"  q={deg} {method} lr={lr} pert={pert}: done", flush=True)


if __name__ == "__main__":
    main()
