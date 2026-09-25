"""Train the repository's DFL models and log rho_eta once per epoch.

One record per (configuration, seed), appended as a line of JSON to --out:

    {model, <the spec's hyperparameters>, seed,
     test_regret,                     final regret on the test set
     probe_history: [{epoch, log10_rho_eta, n_samples}, ...]}

``log10_rho_eta`` is the mean over that epoch's training samples of
log10(||delta|| / eta(theta~)); see rho_eta.py for the definition and
instrument.py for how the probe is attached without touching
Trainer/PO_models.py.

This driver deliberately does NOT go through test_instability_problem.py: no
checkpointing, no TensorBoard, no appending to Rslt/*.csv, so nothing in the
repository's existing result set is touched.  Model selection is equivalent --
the ModelCheckpoint in test_instability_problem.py carries no `monitor`, so the
"best" path it reloads is simply the last epoch, which is the model evaluated
here.

It is resumable: a (configuration, seed) already present in --out is skipped,
so an interrupted run can simply be started again.  Runs are independent, so
--shard/--nshards split a spec across several processes.

    python run_experiments.py --spec specs/E8_article_grid.json \
                              --out results/E8_article_grid.jsonl
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import pytorch_lightning as pl

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from instrument import load_instrumented                 # noqa: E402
from Trainer.data_utils import DataModule                # noqa: E402
from Trainer.instability_problem import InstabilityProblem   # noqa: E402
from Trainer.utils import regret_fn                      # noqa: E402

MODELS = load_instrumented()          # Trainer/PO_models.py + the rho_eta probe

# The seed range of the original experiments: test_instability_problem.py
# runs `for seed in range(10)`.
N_SEEDS = 10


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
    kwargs.setdefault("max_epochs", 30)
    model = getattr(MODELS, params["model"])(
        solver=solver, seed=seed, probe=True, probe_noise=probe_noise, **kwargs)

    trainer = pl.Trainer(
        max_epochs=int(kwargs["max_epochs"]), min_epochs=3,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data)

    rec = dict(params)
    rec["seed"] = seed
    rec["test_regret"] = evaluate(model, data.test_dataloader(), solver)
    rec["probe_history"] = model._probe_history
    if verbose:
        print("  seed {:>2}  test_regret={:.4f}  ({:.1f}s)".format(
            seed, rec["test_regret"], time.time() - t0), flush=True)
    return rec


def config_key(rec):
    """Identity of a (configuration, seed), ignoring everything measured."""
    measured = ("test_regret", "probe_history")
    return json.dumps({k: v for k, v in rec.items() if k not in measured},
                      sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="JSON list of parameter dicts")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--probe_noise", type=int, default=32,
                    help="DPO only: draws used to estimate E||d||")
    ap.add_argument("--shard", type=int, default=0, help="this worker's index")
    ap.add_argument("--nshards", type=int, default=1, help="number of workers")
    args = ap.parse_args()

    # Trainer/data_utils.py opens data/data_instability_problem.json by a path
    # relative to the working directory, so the run has to happen from the
    # instability_problem directory. Paths given on the command line are
    # resolved first, so they stay relative to where the user actually is.
    args.spec = os.path.abspath(args.spec)
    args.out = os.path.abspath(args.out)
    os.chdir(REPO)

    with open(args.spec) as f:
        specs = json.load(f)
    specs = [s for i, s in enumerate(specs) if i % args.nshards == args.shard]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(config_key(json.loads(line)))
                except Exception:
                    continue

    with open(args.out, "a") as fh:
        for i, params in enumerate(specs):
            print("[{}/{}] {}".format(i + 1, len(specs), params), flush=True)
            for seed in range(args.seeds):
                if json.dumps(dict(params, seed=seed), sort_keys=True) in done:
                    continue
                rec = run_one(params, seed, probe_noise=args.probe_noise,
                              verbose=True)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()


if __name__ == "__main__":
    main()
