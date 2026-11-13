"""Three-vertex problem (Section 6.1): train every configuration of a spec with ten seeds.

Each record holds the configuration, the seed, the validation and test regrets, and the per-epoch history
of the cost vector stability ratio rho_eta (rho_eta.py), so that the same runs give Table 1 and Figure 1.

    python run_three_vertex.py --make_spec                     # writes specs/three_vertex.json
    python run_three_vertex.py --spec specs/three_vertex.json --out results/three_vertex.jsonl \
        [--shard i --nshards n]

The run is resumable: configurations already present in --out are skipped. Training is deterministic
(seeded generator, deterministic algorithms), and the rho_eta probe draws its noise from its own
generator, so a rerun reproduces every record of results/three_vertex.jsonl.gz.

Models (Trainer/PO_models_rebuttal.py): {SPO,DPO,DBB}_{variant}, with variant
    std   standard method                 tn    standard method, normalized targets
    rs    shrinkage r^s (kappa)           tnrs  shrinkage, normalized targets
    rc    clipping r^c (kappa)            tnrc  clipping, normalized targets
    rn2   normalization r^n               tnrn  normalization, normalized targets
    l2p   L2 penalty (lam)                tnl2p L2 penalty, normalized targets
"""
import argparse
import json
import os
import random
import sys
import time
import types

import numpy as np
import torch
import pytorch_lightning as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rho_eta import RhoEtaProbe                          # noqa: E402
from Trainer.data_utils import DataModule                # noqa: E402
from Trainer.instability_problem import InstabilityProblem   # noqa: E402
from Trainer.utils import regret_fn                      # noqa: E402
import Trainer.PO_models_rebuttal as R                    # noqa: E402

N_SEEDS = 10
PERT = {"SPO": "alpha", "DPO": "sigma", "DBB": "lambda_val"}
PERT_GRID = {"SPO": [0.5, 2.0, 10.0], "DPO": [0.1, 0.5, 1.0], "DBB": [0.1, 1.0, 10.0]}
LR_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
KAPPA_GRID = [1.0, 100.0, 10000.0]
LAM_GRID = [0.1, 1.0]
INSTANCES = [1, 2, 3]                                    # b = 1, 10, 100
VARIANTS = ("std", "rs", "rc", "rn2", "l2p", "tn", "tnrs", "tnrc", "tnrn", "tnl2p")
PROBE_REG = {"none": "none", "rs": "rp", "rc": "rc", "rn": "rn", "l2p": "none"}


def make_spec():
    """The grid of Table 3 (Appendix C) for the three-vertex problem: 2700 configurations."""
    spec = []
    for inst in INSTANCES:
        for M in ("SPO", "DPO", "DBB"):
            for lr in LR_GRID:
                for pert in PERT_GRID[M]:
                    base = {"instance": inst, "lr": lr, PERT[M]: pert, "max_epochs": 30, "scheduler": True}
                    if M == "DPO":
                        base["num_samples"] = 5
                    for v in VARIANTS:
                        model = f"{M}_{v}"
                        if v in ("rs", "rc", "tnrs", "tnrc"):
                            spec += [dict(base, model=model, kappa=k) for k in KAPPA_GRID]
                        elif v in ("l2p", "tnl2p"):
                            spec += [dict(base, model=model, lam=lam) for lam in LAM_GRID]
                        else:
                            spec.append(dict(base, model=model))
    return spec


def _probed(method, base):
    class Probed(base):
        def __init__(self, *a, probe=True, probe_noise=32, seed=0, ybar=1.0, n_dim=2, **kw):
            kw.setdefault("diag", False)
            kw.setdefault("y_norm_mean", ybar)
            kw.setdefault("n_dim", n_dim)
            super().__init__(*a, seed=seed, **kw)
            self._probe = RhoEtaProbe(num_noise=int(probe_noise), seed=987654 + int(seed)) if probe else None
            self._probe_history = []

        def training_step(self, batch, batch_idx):
            # the probe sees y_hat = model(x), recomputed without gradient (deterministic, no random
            # draw), the cost r(y_hat) fed to the perturbation, and the targets seen by the loss
            if self._probe is not None:
                x, y, sol, m = batch
                with torch.no_grad():
                    y_hat = self(x).squeeze()
                self._probe.observe(method=method, y_hat=y_hat, y=self._tn(y), m=m, solver=self.solver,
                                    reg=PROBE_REG[self.reg], kappa=float(self.kappa_eff),
                                    alpha=float(getattr(self, "alpha", 2.0)),
                                    lam=float(getattr(self, "lambda_val", 1.0)),
                                    sigma=float(getattr(self, "sigma", 1.0)))
            return super().training_step(batch, batch_idx)

        def on_train_epoch_end(self):
            if self._probe is None:
                return
            s = self._probe.epoch_summary()
            if s:
                s["epoch"] = int(self.current_epoch)
                self._probe_history.append(s)
    Probed.__name__ = base.__name__
    return Probed


MODELS = types.SimpleNamespace(**{f"{M}_{v}": _probed(M, getattr(R, f"{M}_{v}"))
                                  for M in ("SPO", "DBB", "DPO") for v in VARIANTS})


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def evaluate(model, loader, solver):
    model.eval()
    tot, n = 0.0, 0
    for x, y, sol, m in loader:
        y_hat = model(x).squeeze()
        tot += float(regret_fn(solver, y_hat, y, sol, m)) * len(y)
        n += len(y)
    model.train()
    return tot / max(n, 1)


def run_one(params, seed, probe_noise=32, verbose=False):
    t0 = time.time()
    solver = InstabilityProblem(int(params["instance"]))
    torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(seed)
    seed_all(seed)
    data = DataModule(generator=g, num_workers=0, solver=solver)
    kwargs = {k: v for k, v in params.items() if k != "model"}
    model = getattr(MODELS, params["model"])(
        solver=solver, seed=seed, probe_noise=probe_noise,
        ybar=float(data.train_df.y.norm(dim=-1).mean()), n_dim=int(data.train_df.y.shape[1]), **kwargs)
    trainer = pl.Trainer(max_epochs=int(kwargs["max_epochs"]), min_epochs=3, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         check_val_every_n_epoch=1, num_sanity_val_steps=0)
    trainer.fit(model, datamodule=data)
    rec = dict(params, seed=seed)
    rec["val_regret"] = evaluate(model, data.val_dataloader(), solver)
    rec["test_regret"] = evaluate(model, data.test_dataloader(), solver)
    rec["probe_history"] = model._probe_history
    if verbose:
        print("  seed {:>2}  val {:.4f}  test {:.4f}  ({:.1f}s)".format(
            seed, rec["val_regret"], rec["test_regret"], time.time() - t0), flush=True)
    return rec


def config_key(rec):
    measured = ("val_regret", "test_regret", "probe_history")
    return json.dumps({k: v for k, v in rec.items() if k not in measured}, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_spec", action="store_true", help="write specs/three_vertex.json and exit")
    ap.add_argument("--spec", help="JSON list of configurations")
    ap.add_argument("--out", help="output .jsonl path")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--probe_noise", type=int, default=32, help="DPO only: draws used to estimate E||d||")
    ap.add_argument("--shard", type=int, default=0, help="this worker's index")
    ap.add_argument("--nshards", type=int, default=1, help="number of workers")
    args = ap.parse_args()

    if args.make_spec:
        os.makedirs(os.path.join(HERE, "specs"), exist_ok=True)
        path = os.path.join(HERE, "specs", "three_vertex.json")
        spec = make_spec()
        json.dump(spec, open(path, "w"), indent=1)
        print(f"wrote {path}: {len(spec)} configurations")
        return
    if not args.spec or not args.out:
        ap.error("--spec and --out are required (or --make_spec)")

    # Trainer/data_utils.py reads data/ relative to the working directory
    args.spec, args.out = os.path.abspath(args.spec), os.path.abspath(args.out)
    os.chdir(HERE)
    specs = json.load(open(args.spec))
    specs = [s for i, s in enumerate(specs) if i % args.nshards == args.shard]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                done.add(config_key(json.loads(line)))
            except Exception:
                continue
    with open(args.out, "a") as fh:
        for i, params in enumerate(specs):
            todo = [s for s in range(args.seeds) if config_key(dict(params, seed=s)) not in done]
            if not todo:
                continue
            print("[{}/{}] {}".format(i + 1, len(specs), params), flush=True)
            for seed in todo:
                fh.write(json.dumps(run_one(params, seed, probe_noise=args.probe_noise, verbose=True)) + "\n")
                fh.flush()


if __name__ == "__main__":
    main()
