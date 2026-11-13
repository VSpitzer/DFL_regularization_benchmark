"""SPO / DBB / DPO with cost regularization for the three-vertex problem (Section 6.1).

Each method has one generic class, parameterised by the regularization mode (see
Trainer/regularizers.py), and named subclasses, one per variant of the article:

    {M}_std   standard method                    {M}_tn    standard method, normalized targets
    {M}_rs    shrinkage r^s                      {M}_tnrs  shrinkage, normalized targets
    {M}_rc    clipping r^c                       {M}_tnrc  clipping, normalized targets
    {M}_rn2   normalization r^n                  {M}_tnrn  normalization, normalized targets
    {M}_l2p   L2 penalty lam * ||theta||^2/ybar  {M}_tnl2p L2 penalty, normalized targets

with M in {SPO, DBB, DPO}; kappa (absolute) and lam are passed as keyword arguments. With normalized
targets, the model is trained on c / ||c|| * ybar (ybar: mean norm of the training costs) and evaluated
on the true costs. The training steps mirror the classes of PO_models.py, with the cost fed to the
perturbation replaced by ``regularize``. The variants rsg / rng (stop-gradient on the scale) are not
used in the article.
"""
import math

import numpy as np
import torch
from torch import nn

from Trainer.PO_models import baseline_mse, log_cost_norm
from Trainer.utils import regret_fn, abs_regret_fn, batch_solve
from Trainer.diff_layer import SPOlayer, DBBlayer
from Trainer.regularizers import regularize, effective_kappa, DiagMixin
from DPO import perturbations


class _RegBase(DiagMixin, baseline_mse):
    reg_mode = "none"
    target_norm = False  # tn variants: train on c / ||c|| * ybar (validation/test on true costs)

    def __init__(self, solver, lr=1e-1, seed=0, scheduler=False, kappa=1., kappa_rel=None,
                 lam=0., y_norm_mean=1., n_dim=2, diag=True, diag_n=32, diag_mc=5, diag_batches=1, reg=None, **kwd):
        super().__init__(solver, lr, seed, scheduler)
        self.reg = reg or self.reg_mode
        self.lam = float(lam)
        self.ybar = float(y_norm_mean)
        self.n_dim = int(n_dim)
        self._kappa_abs = float(kappa)
        self._kappa_rel = None if kappa_rel is None else float(kappa_rel)
        self._diag_init(diag, diag_n, diag_mc, diag_batches=diag_batches)

    def _set_kappa(self, nu):
        self.kappa_eff = effective_kappa(self._kappa_abs, self._kappa_rel, nu)

    def _tn(self, y):
        if not self.target_norm:
            return y
        return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-12) * self.ybar

    def reg_apply(self, y_hat):
        return regularize(y_hat, self.reg, self.kappa_eff, self.lam, self.ybar)

    def _solve_idx(self, m):
        return lambda c, idx: batch_solve(self.solver, c, m[idx])

    def _log_val(self, y_hat, y, sol, m, val_loss):
        val_regret = regret_fn(self.solver, y_hat, y, sol, m)
        abs_val_loss = abs_regret_fn(self.solver, y_hat, y, sol, m)
        mseloss = nn.MSELoss(reduction='mean')(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_regret", val_regret, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_abs_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        return {"val_loss": val_loss, "val_regret": val_regret}


# ----------------------------------------------------------------------------- SPO
class SPO_reg(_RegBase):
    def __init__(self, solver, lr=1e-1, alpha=2, seed=0, scheduler=False, **kwd):
        super().__init__(solver, lr, seed, scheduler, **kwd)
        self.alpha = float(alpha)
        self.layer = SPOlayer(solver, alpha=alpha)
        self._set_kappa(self.ybar / self.alpha)

    def training_step(self, batch, batch_idx):
        x, y, sol, m = batch
        y = self._tn(y)
        y_hat = self(x).squeeze()
        log_cost_norm(self, y_hat)
        train_regret = regret_fn(self.solver, y_hat, y, sol, m)
        u, pen = self.reg_apply(y_hat)
        self._diag_track(y_hat, u)
        # SPOlayer returns a sum over the batch: the penalty (a batch mean) is scaled to match
        loss = self.layer(u, y, sol, m) + pen * len(y)
        self._diag_perturb(batch_idx, u, self._solve_idx(m),
                           lambda uu, idx, g: self.alpha * uu - y[idx],
                           lambda d, idx: -y[idx], 1, true_dec=sol)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("train_regret", train_regret, prog_bar=True, on_step=False, on_epoch=True, )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, sol, m = batch
        y_hat = self(x).squeeze()
        u, _ = self.reg_apply(y_hat)
        return self._log_val(y_hat, y, sol, m, self.layer(u, y, sol, m))


# ----------------------------------------------------------------------------- DBB
class DBB_reg(_RegBase):
    def __init__(self, solver, lr=1e-1, lambda_val=1., seed=0, scheduler=False, **kwd):
        super().__init__(solver, lr, seed, scheduler, **kwd)
        self.lambda_val = float(lambda_val)
        self.layer = DBBlayer(solver, lambda_val=lambda_val)
        # DBBlayer here perturbs y_hat by lambda_val * y_true (it ignores grad_output)
        self._set_kappa(self.lambda_val * self.ybar)

    def training_step(self, batch, batch_idx):
        x, y, sol, m = batch
        y = self._tn(y)
        y_hat = self(x).squeeze()
        log_cost_norm(self, y_hat)
        train_regret = regret_fn(self.solver, y_hat, y, sol, m)
        u, pen = self.reg_apply(y_hat)
        self._diag_track(y_hat, u)
        sol_hat = self.layer(u, y, sol, m)
        loss = ((sol - sol_hat) * y).sum(-1).mean() + pen
        self._diag_perturb(batch_idx, u, self._solve_idx(m),
                           lambda uu, idx, g: uu + self.lambda_val * y[idx],
                           lambda d, idx: y[idx], 1, true_dec=sol)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("train_regret", train_regret, prog_bar=True, on_step=False, on_epoch=True, )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, sol, m = batch
        y_hat = self(x).squeeze()
        u, _ = self.reg_apply(y_hat)
        sol_hat = self.layer(u, y, sol, m)
        return self._log_val(y_hat, y, sol, m, ((sol - sol_hat) * y).sum(-1).mean())


# ----------------------------------------------------------------------------- DPO
class DPO_reg(_RegBase):
    def __init__(self, solver, instance=1, sigma=0.1, num_samples=10, lr=1e-1, seed=0,
                 scheduler=False, logger_backend="tb", **kwd):
        self.sigma = sigma
        self.num_samples = num_samples
        self.instance = instance
        self.logger_backend = logger_backend
        super().__init__(solver, lr, seed, scheduler, **kwd)
        self._set_kappa(self.sigma * math.sqrt(self.n_dim))

    def _dpo_loss(self, u, y, sol, m):
        loss = 0
        for i in range(len(u)):
            def solver(y_):
                out = []
                for j in range(len(y_)):
                    out.append(batch_solve(self.solver, y_[j], m[i], batched=False).unsqueeze(0))
                return torch.cat(out).float()
            op = perturbations.perturbed(solver, num_samples=self.num_samples, sigma=self.sigma,
                                         noise='normal', batched=False)(u[i])
            loss += y[i].dot(sol[i] - op)
        return loss / len(u)

    def training_step(self, batch, batch_idx):
        x, y, sol, m = batch
        y = self._tn(y)
        y_hat = self(x).squeeze()
        log_cost_norm(self, y_hat)
        train_regret = regret_fn(self.solver, y_hat, y, sol, m)
        u, pen = self.reg_apply(y_hat)
        self._diag_track(y_hat, u)
        loss = self._dpo_loss(u, y, sol, m) + pen
        self._diag_perturb(batch_idx, u, self._solve_idx(m),
                           lambda uu, idx, g: uu + self.sigma * torch.randn(uu.shape, generator=g),
                           lambda d, idx: d, self.diag_mc, true_dec=sol)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("train_regret", train_regret, prog_bar=True, on_step=False, on_epoch=True, )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, sol, m = batch
        y_hat = self(x).squeeze()
        u, _ = self.reg_apply(y_hat)
        return self._log_val(y_hat, y, sol, m, self._dpo_loss(u, y, sol, m))


# ----------------------------------------------------------------------------- named variants
def _variant(base, name, mode, **attrs):
    cls = type(name, (base,), {"reg_mode": mode, **attrs})
    globals()[name] = cls
    return cls


for _M, _B in (("SPO", SPO_reg), ("DBB", DBB_reg), ("DPO", DPO_reg)):
    for _suffix, _mode in (("std", "none"), ("rs", "rs"), ("rsg", "rsg"), ("rng", "rng"),
                           ("rc", "rc"), ("l2p", "l2p"), ("rn2", "rn")):
        _variant(_B, f"{_M}_{_suffix}", _mode)
    _variant(_B, f"{_M}_tn", "none", target_norm=True)     # target normalization
    _variant(_B, f"{_M}_tnrs", "rs", target_norm=True)     # target normalization + shrinkage
    _variant(_B, f"{_M}_tnrn", "rn", target_norm=True)     # target normalization + normalization
    _variant(_B, f"{_M}_tnrc", "rc", target_norm=True)     # target normalization + clipping
    _variant(_B, f"{_M}_tnl2p", "l2p", target_norm=True)   # target normalization + L2 penalty

REBUTTAL_MODELS = [f"{M}_{s}" for M in ("SPO", "DBB", "DPO")
                   for s in ("std", "rs", "rsg", "rng", "rc", "l2p", "rn2", "tn", "tnrs", "tnrn", "tnrc", "tnl2p")]
