"""Add the rho_eta probe to the repository's own DFL models, without editing them.

``Trainer/PO_models.py`` is left exactly as it is on disk.  This module reads
it, applies a short list of purely additive edits to the source text, and
executes the result as a module in memory.  The edits are:

  1. import RhoEtaProbe from rho_eta.py (next to this file);
  2. give ``baseline_mse`` a ``probe`` constructor flag, a probe object, a
     ``probe_observe()`` helper and an ``on_train_epoch_end`` hook that appends
     one summary per epoch to ``self._probe_history``;
  3. tell each of the nine DFL classes which perturbation and which
     regularization map it uses (PROBE_METHOD / PROBE_REG / PROBE_REG_TRUE),
     remember alpha / lambda_val, forward **kwd to ``baseline_mse.__init__``
     so the flag reaches it, and call ``probe_observe`` in ``training_step``;
  4. drop ``verbose=True`` from the ReduceLROnPlateau built in
     ``configure_optimizers``, which torch >= 2.9 rejects.  That object is
     never used -- ``configure_optimizers`` builds a fresh scheduler below it
     -- so this changes nothing about training.

Everything in (1)-(3) runs under ``torch.no_grad()`` on detached tensors and
draws its noise from a dedicated generator, so a probed run is bit-identical
to an unprobed one.  (4) is the only edit that is not purely additive, and it
is a no-op by inspection.

Every substitution asserts how many times it must match, so if
``Trainer/PO_models.py`` ever changes in a way that breaks an anchor this
fails loudly instead of silently instrumenting nothing.

    python instrument.py --diff     # show the edits as a unified diff
    python instrument.py --dump instrumented.py
"""

import argparse
import difflib
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                       # .../instability_problem
PO_MODELS = os.path.join(REPO, "Trainer", "PO_models.py")

# method, regularization map applied to y_hat, and whether the variant also
# normalizes the ground-truth cost vector (the _rn variants of this repo do).
VARIANTS = [
    ("SPO",     "SPO", None, False),
    ("SPO_rn",  "SPO", "rn", True),
    ("SPO_rp",  "SPO", "rp", False),
    ("DBB",     "DBB", None, False),
    ("DBB_rn",  "DBB", "rn", True),
    ("DBB_rp",  "DBB", "rp", False),
    ("DPO",     "DPO", None, False),
    ("DPO_rn",  "DPO", "rn", True),
    ("DPO_rp",  "DPO", "rp", False),
]

_IMPORT = '''import pytorch_lightning as pl
import sys as _probe_sys
if {here!r} not in _probe_sys.path:
    _probe_sys.path.insert(0, {here!r})
from rho_eta import RhoEtaProbe
'''

_CLASS_HEAD = '''class baseline_mse(pl.LightningModule):
    # Set by each DFL subclass so the shared helper below knows which of
    # Section 4.1's perturbations applies and which regularization map the
    # subclass puts in front of the differentiable layer.
    PROBE_METHOD = None
    PROBE_REG = None
    PROBE_REG_TRUE = False

    def __init__(self,solver,lr=1e-1,seed=0,scheduler=False, probe=False, probe_noise=32, **kwd):
'''

_PROBE_INIT = '''        self.scheduler = scheduler
        self._probe = RhoEtaProbe(num_noise=int(probe_noise),
                                  seed=987654 + int(seed)) if probe else None
        self._probe_history = []
        self.save_hyperparameters("lr")
'''

_METHODS = '''    def forward(self,x):
        h = self.relu(self.fc1(x))
        return self.fc2(h)

    def probe_observe(self, y_hat, y, m):
        """Record one training batch's rho_eta (a no-op unless probing)."""
        if self._probe is None or self.PROBE_METHOD is None:
            return
        self._probe.observe(
            method=self.PROBE_METHOD, y_hat=y_hat, y=y, m=m, solver=self.solver,
            reg=self.PROBE_REG, kappa=float(getattr(self, "kappa", 1.0)),
            alpha=float(getattr(self, "_alpha", 2.0)),
            lam=float(getattr(self, "_lambda_val", 1.0)),
            sigma=float(getattr(self, "sigma", 1.0)),
            reg_true=self.PROBE_REG_TRUE,
        )

    def on_train_epoch_end(self):
        if self._probe is None:
            return
        summary = self._probe.epoch_summary()
        if summary:
            summary["epoch"] = int(self.current_epoch)
            self._probe_history.append(summary)

    def training_step(self, batch, batch_idx):
'''


def _sub(src, old, new, n):
    got = src.count(old)
    if got != n:
        raise RuntimeError(
            "instrumentation anchor matched {} times, expected {}:\n{}".format(
                got, n, old[:200]))
    return src.replace(old, new)


def instrument(src):
    """Return the instrumented source text of Trainer/PO_models.py."""
    src = _sub(src, "import pytorch_lightning as pl\n",
               _IMPORT.format(here=HERE), 1)

    src = _sub(src,
               "class baseline_mse(pl.LightningModule):\n"
               "    def __init__(self,solver,lr=1e-1,seed=0,scheduler=False, **kwd):\n",
               _CLASS_HEAD, 1)
    src = _sub(src,
               "        self.scheduler = scheduler\n"
               '        self.save_hyperparameters("lr")\n',
               _PROBE_INIT, 1)
    src = _sub(src,
               "    def forward(self,x):\n"
               "        h = self.relu(self.fc1(x))\n"
               "        return self.fc2(h)\n"
               "    def training_step(self, batch, batch_idx):\n",
               _METHODS, 1)

    # one probe call per training_step, right after the batch's y_hat
    src = _sub(src,
               "        log_cost_norm(self, y_hat)\n",
               "        log_cost_norm(self, y_hat)\n"
               "        self.probe_observe(y_hat, y, m)\n", 10)

    # the probe flag has to reach baseline_mse.__init__
    src = _sub(src, "super().__init__(solver,lr,seed,scheduler)",
               "super().__init__(solver,lr,seed,scheduler, **kwd)", 6)
    src = _sub(src, "super().__init__(solver,lr,seed, scheduler)",
               "super().__init__(solver,lr,seed, scheduler, **kwd)", 3)

    # the perturbation hyperparameters, which the classes otherwise hand
    # straight to the layer without keeping
    src = _sub(src, "        self.layer = SPOlayer(solver, alpha=alpha)\n",
               "        self.layer = SPOlayer(solver, alpha=alpha)\n"
               "        self._alpha = alpha\n", 3)
    src = _sub(src, "        self.layer = DBBlayer(solver, lambda_val=lambda_val)\n",
               "        self.layer = DBBlayer(solver, lambda_val=lambda_val)\n"
               "        self._lambda_val = lambda_val\n", 3)

    for name, method, reg, reg_true in VARIANTS:
        src = _sub(src, "class {}(baseline_mse):\n".format(name),
                   "class {}(baseline_mse):\n"
                   "    PROBE_METHOD = {!r}\n"
                   "    PROBE_REG = {!r}\n"
                   "    PROBE_REG_TRUE = {!r}\n".format(name, method, reg, reg_true), 1)

    # torch >= 2.9 removed ReduceLROnPlateau's verbose argument. The object
    # built here is never used; configure_optimizers builds a fresh scheduler
    # a few lines below.
    src = _sub(src,
               "            min_lr=1e-6,\n            verbose=True\n        )\n",
               "            min_lr=1e-6,\n        )\n", 1)
    return src


def read_source(path=PO_MODELS):
    with open(path, "r") as f:                     # universal newlines
        return f.read()


def load_instrumented(path=PO_MODELS, name="PO_models_probe"):
    """Execute the instrumented source as a module and return it."""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    src = instrument(read_source(path))
    mod = types.ModuleType(name)
    mod.__file__ = path + "  [instrumented in memory by stability_study/instrument.py]"
    sys.modules[name] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=PO_MODELS)
    ap.add_argument("--dump", help="write the instrumented source here")
    ap.add_argument("--diff", action="store_true", help="print a unified diff")
    args = ap.parse_args()

    original = read_source(args.source)
    patched = instrument(original)
    if args.diff:
        try:
            sys.stdout.writelines(difflib.unified_diff(
                original.splitlines(True), patched.splitlines(True),
                "Trainer/PO_models.py", "Trainer/PO_models.py  [instrumented]"))
        except BrokenPipeError:        # piped into head/less
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            return
    if args.dump:
        with open(args.dump, "w") as f:
            f.write(patched)
        print("wrote", args.dump)
    if not args.diff and not args.dump:
        load_instrumented(args.source)
        print("instrumented {} lines -> {} lines; imports cleanly".format(
            len(original.splitlines()), len(patched.splitlines())))


if __name__ == "__main__":
    main()
