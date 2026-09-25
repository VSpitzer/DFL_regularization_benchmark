"""The governing ratio rho_eta, measured during training.

Section 4.1 of "Managing Solution Stability in Decision-Focused Learning with
Cost Regularization" states the condition under which a perturbation-based DFL
method learns anything at all: the perturbation applied to the predicted cost
vector has to be comparable to the stability radius of the point it is applied
to.  Written as a single number, that condition is

    rho_eta  =  ||delta|| / eta(theta~)

where theta~ = r(theta_hat) is the (possibly regularized) cost estimate handed
to the differentiable layer, delta is the perturbation the method applies to
it, and eta is the stability radius of Section 3.3.

    rho_eta << 1   the perturbation is absorbed: f(theta~ + delta) = f(theta~),
                   so the method sees no difference and the gradient degenerates
    rho_eta >> 1   the perturbation dominates: theta~ is irrelevant to the
                   perturbed decision
    rho_eta ~ 1    the only regime in which the perturbation explores a
                   neighbourhood of f(theta~), as intended

The perturbation scale (alpha, lambda, sigma) is a fixed hyperparameter, while
eta(theta~) is proportional to ||theta~|| (Proposition 4.1) and ||theta_hat||
moves freely during training.  rho_eta is therefore uncontrolled unless
something pins ||theta~|| down -- which is what cost regularization does.

For the three-vertex problem the optimal-solution set is an argmax over a
finite vertex set, so eta is available in closed form rather than having to be
estimated.  For  max_{v in V} theta.v  with a unique maximiser v*,

    eta(theta) = min_{v != v*}  theta.(v* - v) / ||v* - v||                 (A)

the distance from theta to the nearest facet of its normal cone.  (A) is
homogeneous of degree one in theta, so it reproduces Proposition 4.1,
eta(theta) = ||theta|| eta(theta/||theta||), by construction, and
Cauchy-Schwarz gives eta(theta/||theta||) <= 1, the bound of Appendix B.

Per method, with theta_bar' the ground-truth cost vector as the layer receives
it, the perturbation and its base point are:

    SPO   p = alpha*theta~ - theta_bar'    delta = theta_bar',
                                           base  = alpha*theta~
    DBB   p = theta~ + lambda*theta_bar'   delta = lambda*theta_bar',
                                           base  = theta~
    DPO   p = theta~ + sigma*d,  d~N(0,I)  delta = sigma*E||d||,
                                           base  = theta~

Nothing here touches training.  Every routine runs under torch.no_grad() on
detached tensors, and the DPO noise is drawn from a dedicated torch.Generator,
so the global RNG stream -- and hence the trained model -- is bit-identical
whether the probe is switched on or off.
"""

import math

import numpy as np
import torch


# --------------------------------------------------------------------------
# Geometry of the three-vertex problem
# --------------------------------------------------------------------------

def vertex_set(solver, m):
    """The mapped vertex set {M(m) v : v in {B, C, A}} for one sample.

    Mirrors InstabilityProblem.solve exactly (same vertex order, same map),
    so the argmax taken here is the argmax the solver takes.
    """
    from Trainer.instability_problem import rotate

    theta = float(np.asarray(m).reshape(-1)[0])
    b = solver.b
    tab_vertex = [
        [np.cos(0), np.sin(0)],
        [b * np.cos(2 * np.pi / 3), b * np.sin(2 * np.pi / 3)],
        [np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3)],
    ]
    return np.array([rotate(theta, v) for v in tab_vertex])


def stability_radius(V, theta):
    """Exact stability radius eta(theta); formula (A) of the module docstring.

    Returns 0.0 on the degenerate inputs where eta is not defined or is zero:
    theta = 0 (every vertex optimal) and theta exactly on a cone boundary.
    """
    theta = np.asarray(theta, dtype=float)
    n = np.linalg.norm(theta)
    if not np.isfinite(n) or n == 0.0:
        return 0.0
    vals = V @ theta
    i = int(np.argmax(vals))
    best = math.inf
    for j in range(len(V)):
        if j == i:
            continue
        d = V[i] - V[j]
        nd = np.linalg.norm(d)
        if nd == 0.0:
            continue
        best = min(best, float(theta @ d) / nd)
    if not np.isfinite(best):
        return 0.0
    return max(best, 0.0)


# --------------------------------------------------------------------------
# Regularization maps (Definitions 5.2 and 5.3), as numpy
# --------------------------------------------------------------------------

def apply_reg(theta, reg, kappa=1.0):
    theta = np.asarray(theta, dtype=float)
    if reg in (None, "", "none"):
        return theta
    n = float(np.linalg.norm(theta))
    if reg == "rn":                      # Definition 5.2, L2 normalization
        return theta if n == 0.0 else theta / n
    if reg == "rp":                      # Definition 5.3, shrinkage into an L2 ball
        return theta / (1.0 + n / kappa)
    raise ValueError("unknown regularization {!r}".format(reg))


# --------------------------------------------------------------------------
# The probe
# --------------------------------------------------------------------------

_EPS = 1e-12


def _safe_ratio(num, den):
    return float(num) / float(den) if den > _EPS else float("inf")


class RhoEtaProbe:
    """Accumulates rho_eta over the samples of one training epoch.

    Used from a training_step as

        probe.observe(method="SPO", y_hat=y_hat, y=y, m=m, solver=self.solver,
                      reg=None, alpha=self.alpha)

    and then ``epoch_summary()`` once per epoch, which returns the mean of
    log10 rho_eta over the epoch's samples and resets.  The mean is taken in
    the log because rho_eta spans decades within a single epoch.
    """

    def __init__(self, num_noise=32, seed=12345):
        self.num_noise = num_noise
        # Dedicated generator: the probe must never advance the global RNG,
        # otherwise switching it on would change the trained model.
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)
        self.reset()

    def reset(self):
        self._sum = 0.0
        self._n = 0

    # -- per-sample kernels: (perturbation norm, base point) ----------------

    def _spo(self, V, th, tb, alpha):
        return _safe_ratio(np.linalg.norm(tb), stability_radius(V, alpha * th))

    def _dbb(self, V, th, tb, lam):
        return _safe_ratio(lam * np.linalg.norm(tb), stability_radius(V, th))

    def _dpo(self, V, th, sigma):
        d = torch.randn(self.num_noise, len(th), generator=self._gen).numpy()
        mean_dn = float(np.linalg.norm(d, axis=1).mean())
        return _safe_ratio(sigma * mean_dn, stability_radius(V, th))

    # -- public -------------------------------------------------------------

    @torch.no_grad()
    def observe(self, method, y_hat, y, m, solver, reg=None, kappa=1.0,
                alpha=2.0, lam=1.0, sigma=1.0, reg_true=False):
        """Record one training batch.

        ``reg``/``kappa`` describe the regularization the model applies to
        y_hat before the differentiable layer; ``reg_true=True`` reproduces
        the _rn variants of this repository, which normalize the ground-truth
        cost vector as well.
        """
        yh = y_hat.detach().cpu().numpy().astype(float)
        yt = y.detach().cpu().numpy().astype(float)
        mm = m.detach().cpu().numpy()
        if yh.ndim == 1:
            yh, yt, mm = yh[None, :], yt[None, :], mm[None, :]

        for i in range(len(yh)):
            V = vertex_set(solver, mm[i])
            th = apply_reg(yh[i], reg, kappa)
            tb = apply_reg(yt[i], "rn", kappa) if reg_true else yt[i]

            if method == "SPO":
                rho = self._spo(V, th, tb, alpha)
            elif method == "DBB":
                rho = self._dbb(V, th, tb, lam)
            elif method == "DPO":
                rho = self._dpo(V, th, sigma)
            else:
                raise ValueError("unknown method {!r}".format(method))

            # Samples on which eta is 0 give rho_eta = inf; they are left out
            # of the sum but still counted, exactly as in the run that
            # produced results/.
            if np.isfinite(rho):
                self._sum += float(np.log10(max(rho, 1e-12)))
            self._n += 1

    def epoch_summary(self, reset=True):
        if self._n == 0:
            return {}
        out = {"log10_rho_eta": self._sum / self._n, "n_samples": self._n}
        if reset:
            self.reset()
        return out
