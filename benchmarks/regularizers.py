"""Cost regularizations and training diagnostics (shrinkage, clipping, normalization, L2 penalty).

The same file is used by three_vertex/ (Trainer/regularizers.py) and benchmarks/, so that each folder
stays standalone. The experiments of the article use none, rs, rc, rn and l2p, with an absolute kappa.

Regularizations r(theta) applied to the predicted cost vector theta = h_v(x)
before it enters a DFL method (``mode``):

    none   theta                                    (standard method)
    rn     theta / ||theta||                        (normalization, Sahoo et al. 2022)
    rs     theta / (1 + ||theta|| / kappa)          (shrinkage, the paper's r_s)
    rc     theta * min(1, kappa / ||theta||)        (hard norm clipping)
    l2p    theta, plus a penalty  lam * ||theta||^2 / ybar  on the loss
    rsg    theta / sg(1 + ||theta|| / kappa)        (shrinkage, stop-gradient on the scale)
    rng    theta / sg(||theta||)                    (normalization, stop-gradient on the scale)

sg(.) is the stop-gradient (detach).  By positive scaling invariance,
f(r(theta) + delta) = f(theta + s(theta) * delta) where s(theta) is the scale
divided out, so ``rsg`` and ``rng`` are exactly *sample-dependent perturbation
scaling*: the perturbation is multiplied by (1 + ||theta||/kappa), resp.
||theta||, with no other change to the gradient (the Jacobian of r reduces to
I / s).  ``rs``/``rn`` differ from them only through the full Jacobian of r,
which also damps the radial component of the gradient.  Comparing rs vs rsg
and rn vs rng therefore isolates that effect.

kappa can be given in absolute terms (``kappa``) or relative to the scale of
the method's perturbation (``kappa_rel``): kappa = kappa_rel * nu, with

    DPO   nu = sigma * sqrt(n)            (norm of the noise sigma * d)
    SPO   nu = ybar / alpha               (the cost alpha*theta is perturbed by -theta_bar)
    DBB   nu = lambda * ybar * scale_dbb  (the cost theta is perturbed by lambda*theta_bar*scale_dbb)

where ybar is the mean norm of the true training cost vectors, n their
dimension, and scale_dbb is 1 or 1/batch_size depending on how the problem's
DBB layer uses the incoming gradient.  kappa_rel is dimensionless, so the same
grid can be used on every problem.

Diagnostics (``DiagMixin``), recorded once per epoch in ``self.diag_history``:

    norm_raw_*     mean/median/min/max of ||theta|| over the epoch's training batches
    norm_used_mean mean of ||r(theta)||
    radial_push    mean over training samples of  -<g, theta> / (||g|| ||theta||),
                   g the loss gradient w.r.t. the raw network output theta: > 0 means
                   that a gradient step increases ||theta|| (norm-inflating incentive)
    p_same         Monte-Carlo estimate of P[f(r(theta) + delta) = f(r(theta))]:
                   the perturbation leaves the decision unchanged (rho_eta regime < 1)
    p_noise        estimate of P[f(r(theta) + delta) = f(delta)]: the decision is the
                   one of the perturbation alone (xi_eta regime < 1)
    dist_same      mean relative L1 distance |f(r+delta) - f(r)|_1 / (|f(r+delta)|_1 + |f(r)|_1)
                   in [0, 1]: a graded version of 1 - p_same, informative when decisions are
                   large (e.g. a 72-hour plan) and exact equality is rare
    dist_noise     the same between f(r(theta) + delta) and f(delta)
    frac_wrong     fraction of the probed samples whose decision f(r(theta)) differs from
                   the optimal decision f(theta_bar), i.e. those with positive regret
    p_same_wrong, dist_same_wrong
                   p_same / dist_same restricted to those samples: a perturbation that
                   leaves a *wrong* decision unchanged gives no descent direction where
                   one is needed (p_same alone is also high on correctly decided samples,
                   where an unchanged decision is harmless)

p_same / p_noise are measured on the first ``diag_n`` samples of each of the
first ``diag_batches`` training batches of each epoch, with the method's own perturbation.  They need
only calls to the exact solver, so they are available on every problem,
including those where the stability radius has no closed form.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch

REG_MODES = ("none", "rn", "rs", "rc", "l2p", "rsg", "rng")


def effective_kappa(kappa: float, kappa_rel: Optional[float], nu: float) -> float:
    """kappa_rel * nu if kappa_rel is given, else the absolute kappa."""
    if kappa_rel is None:
        return float(kappa)
    return float(kappa_rel) * float(nu)


def regularize(y_hat: torch.Tensor, mode: str, kappa: float = 1.0, lam: float = 0.0,
               ybar: float = 1.0, eps: float = 1e-12):
    """Return (r(y_hat), penalty).  y_hat has shape (B, n) or (n,)."""
    if mode not in REG_MODES:
        raise ValueError(f"unknown regularization {mode!r}, expected one of {REG_MODES}")
    squeeze = y_hat.dim() == 1
    y = y_hat.unsqueeze(0) if squeeze else y_hat
    norms = y.norm(dim=-1, keepdim=True).clamp_min(eps)
    penalty = y.new_zeros(())
    if mode == "none":
        out = y
    elif mode == "rn":
        out = y / norms
    elif mode == "rng":
        out = y / norms.detach()
    elif mode == "rs":
        out = y / (1.0 + norms / kappa)
    elif mode == "rsg":
        out = y / (1.0 + norms / kappa).detach()
    elif mode == "rc":
        out = y * torch.clamp(kappa / norms, max=1.0)
    elif mode == "l2p":
        out = y
        penalty = lam * (norms.squeeze(-1) ** 2).mean() / max(ybar, eps)
    return (out.squeeze(0) if squeeze else out), penalty


def rel_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise relative L1 distance in [0, 1] between decision vectors."""
    den = a.abs().sum(-1) + b.abs().sum(-1)
    return torch.where(den > 0, (a - b).abs().sum(-1) / den.clamp_min(1e-12), torch.zeros_like(den))


def same_decision(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
    """Row-wise equality of decision vectors, shape (B,) bool."""
    return (a - b).abs().amax(dim=-1) <= atol * (1.0 + b.abs().amax(dim=-1))


class DiagMixin:
    """Per-epoch diagnostics.  Call ``_diag_init`` in __init__, ``_diag_track``
    in training_step on the raw output, and ``_diag_perturb`` on the first batch."""

    def _diag_init(self, diag: bool = True, diag_n: int = 32, diag_mc: int = 5, diag_seed: int = 12345,
                   diag_batches: int = 1):
        self.diag = bool(diag)
        self.diag_n = int(diag_n)
        self.diag_batches = int(diag_batches)
        self.diag_mc = int(diag_mc)
        self.diag_history = []
        self._diag_gen = torch.Generator()
        self._diag_gen.manual_seed(int(diag_seed))
        self._diag_reset()

    def _diag_reset(self):
        self._d_norm_raw = []
        self._d_norm_used = []
        self._d_radial = []
        self._d_same = []
        self._d_noise = []
        self._d_dsame = []
        self._d_dnoise = []
        self._d_wrong = []
        self._d_same_w = []
        self._d_dsame_w = []

    # -- hooks ------------------------------------------------------------
    def on_train_epoch_start(self):
        if getattr(self, "diag", False):
            self._diag_reset()

    def on_train_epoch_end(self):
        if not getattr(self, "diag", False):
            return
        rec = {"epoch": int(self.current_epoch)}
        if self._d_norm_raw:
            nr = np.concatenate(self._d_norm_raw)
            rec.update(norm_raw_mean=float(nr.mean()), norm_raw_median=float(np.median(nr)),
                       norm_raw_min=float(nr.min()), norm_raw_max=float(nr.max()))
        if self._d_norm_used:
            rec["norm_used_mean"] = float(np.concatenate(self._d_norm_used).mean())
        if self._d_radial:
            rec["radial_push"] = float(np.concatenate(self._d_radial).mean())
        if self._d_same:
            rec["p_same"] = float(np.mean(self._d_same))
        if self._d_noise:
            rec["p_noise"] = float(np.mean(self._d_noise))
        if self._d_dsame:
            rec["dist_same"] = float(np.mean(self._d_dsame))
        if self._d_dnoise:
            rec["dist_noise"] = float(np.mean(self._d_dnoise))
        if self._d_wrong:
            rec["frac_wrong"] = float(np.mean(self._d_wrong))
        if self._d_same_w:
            rec["p_same_wrong"] = float(np.mean(self._d_same_w))
            rec["dist_same_wrong"] = float(np.mean(self._d_dsame_w))
        self.diag_history.append(rec)
        for k, v in rec.items():
            if k != "epoch":
                self.log("diag_" + k, v, on_step=False, on_epoch=True)

    # -- collection -------------------------------------------------------
    def _diag_track(self, y_hat: torch.Tensor, y_used: torch.Tensor):
        """Record norms and register the radial-push hook on the raw output."""
        if not getattr(self, "diag", False):
            return
        y2 = y_hat if y_hat.dim() > 1 else y_hat.unsqueeze(0)
        u2 = y_used if y_used.dim() > 1 else y_used.unsqueeze(0)
        self._d_norm_raw.append(y2.detach().norm(dim=-1).cpu().numpy())
        self._d_norm_used.append(u2.detach().norm(dim=-1).cpu().numpy())
        if y_hat.requires_grad:
            theta = y2.detach()

            def _hook(g, theta=theta):
                g2 = g if g.dim() > 1 else g.unsqueeze(0)
                num = -(g2 * theta).sum(-1)
                den = g2.norm(dim=-1) * theta.norm(dim=-1)
                ok = den > 0
                if ok.any():
                    self._d_radial.append((num[ok] / den[ok]).detach().cpu().numpy())
                return g

            y_hat.register_hook(_hook)

    def _diag_perturb(self, batch_idx: int, y_used: torch.Tensor,
                      solve: Callable[[torch.Tensor, slice], torch.Tensor],
                      perturbed: Callable[[torch.Tensor, slice, torch.Generator], torch.Tensor],
                      noise_only: Callable[[torch.Tensor, slice], torch.Tensor],
                      n_draws: int, true_dec: Optional[torch.Tensor] = None):
        """Estimate p_same and p_noise on the first diag_n samples of the first diag_batches batches.

        solve(costs, idx)          -> decisions for the samples ``idx`` of the batch
        perturbed(u, idx, gen)     -> the perturbed costs the method feeds the solver
        noise_only(pert_minus_u, idx) -> the costs whose decision is f(delta)
        true_dec                   optimal decisions f(theta_bar) of the batch (optional)
        """
        if not getattr(self, "diag", False) or batch_idx >= self.diag_batches:
            return
        with torch.no_grad():
            u2 = y_used.detach() if y_used.dim() > 1 else y_used.detach().unsqueeze(0)
            n = min(self.diag_n, u2.shape[0])
            idx = slice(0, n)
            u = u2[idx]
            base = solve(u, idx)
            wrong = None
            if true_dec is not None:
                wrong = rel_l1(base, true_dec[idx].detach().to(base.dtype)) > 1e-6
                self._d_wrong.extend(wrong.float().tolist())
            for _ in range(n_draws):
                pc = perturbed(u, idx, self._diag_gen)
                dec = solve(pc, idx)
                self._d_same.extend(same_decision(dec, base).float().tolist())
                self._d_dsame.extend(rel_l1(dec, base).tolist())
                if wrong is not None and wrong.any():
                    self._d_same_w.extend(same_decision(dec, base)[wrong].float().tolist())
                    self._d_dsame_w.extend(rel_l1(dec, base)[wrong].tolist())
                dn = solve(noise_only(pc - u, idx), idx)
                self._d_noise.extend(same_decision(dec, dn).float().tolist())
                self._d_dnoise.extend(rel_l1(dec, dn).tolist())


def dump_diag(path: str, model, explicit: dict, seed: int, **extra):
    """Append one JSON line per epoch of ``model.diag_history`` to ``path``."""
    import json
    import os
    hist = getattr(model, "diag_history", None)
    if not hist:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    params = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
              for k, v in explicit.items()}
    with open(path, "a") as f:
        for rec in hist:
            f.write(json.dumps({**params, **extra, "seed": int(seed), **rec}) + "\n")
