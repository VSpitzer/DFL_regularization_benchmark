"""Shortest-path and knapsack benchmarks of Section 6.2 (PyEPO data generator, exact solvers).

0/1 knapsack (maximisation) with 32 items, integer weights in {3, ..., 8} and a capacity of half the total
weight; shortest path on a 5 x 5 grid (40 edges, moves to the right and downwards). Costs are generated
from 5 standard Gaussian features by PyEPO's polynomial generator of degree q (Tang & Khalil, 2024;
Elmachtoub & Grigas, 2022),
    c_j = [((B x)_j / sqrt(5) + 3)^q / 3.5^q + 1] * U(0.5, 1.5),
B a fixed random binary matrix. Both problems are solved exactly by dynamic programming (numba).

Methods: SPO+ (perturbation alpha * theta - c), DBB (theta + beta * c / batch, with the 1/beta gradient),
DPO (Gaussian noise sigma, 5 samples). Regularizations (regularizers.py): shrinkage "rs", clipping "rc"
(absolute kappa), normalization "rn", L2 penalty "l2p" (lambda); prefix "tn_" (or "tn" alone) trains on
normalized targets c * mean||c|| / ||c||, the regret being always computed with the true costs. Linear
model, Adam, 30 epochs, batches of 32; normalized regret on the validation and test sets, and the
decision-change rate Delta every five epochs ("hist").
"""
import math

import numba as nb
import numpy as np
import torch

from regularizers import regularize, effective_kappa, rel_l1


# ----------------------------------------------------------------------------- data
def gen_knapsack(n, m=32, p=5, deg=1, ebar=0.5, seed=0):
    """PyEPO knapsack generator; returns features, item values (to maximise), weights, capacity."""
    rng = np.random.default_rng(seed)
    B = rng.binomial(1, 0.5, (m, p))
    X = rng.normal(0, 1, (n, p))
    C = ((X @ B.T) / np.sqrt(p) + 3) ** deg / 3.5 ** deg + 1
    C *= rng.uniform(1 - ebar, 1 + ebar, C.shape)
    w = rng.integers(3, 9, m)
    return X, C, w, int(w.sum() // 2)


def gen_sp(n, k=5, p=5, deg=1, ebar=0.5, seed=0):
    """PyEPO shortest-path generator; returns features and values = -costs (the code maximises)."""
    rng = np.random.default_rng(seed)
    d = 2 * k * (k - 1)
    B = rng.binomial(1, 0.5, (d, p))
    X = rng.normal(0, 1, (n, p))
    C = ((X @ B.T) / np.sqrt(p) + 3) ** deg / 3.5 ** deg + 1
    C *= rng.uniform(1 - ebar, 1 + ebar, C.shape)
    return X, -C


def make_problem(problem, deg, n=2250):
    """Data set of the article for a problem and degree: 1000 training, 250 validation, 1000 test samples
    (the generator is seeded with the degree), and the corresponding exact solver."""
    if problem == "knapsack":
        X, C, w, cap = gen_knapsack(n, deg=deg, seed=deg)
        solve = Solver(w, cap)
    elif problem == "sp":
        X, C = gen_sp(n, deg=deg, seed=deg)
        solve = SPSolver(5)
    else:
        raise ValueError(problem)
    return (X[:1000], C[:1000], X[1000:1250], C[1000:1250], X[1250:], C[1250:]), solve


# ----------------------------------------------------------------------------- solvers
@nb.njit(cache=True)
def _knap(v, w, cap, out):
    m = v.shape[0]
    best = np.zeros(cap + 1)
    take = np.zeros((m, cap + 1), dtype=np.bool_)
    for i in range(m):
        if v[i] <= 0:
            continue
        for c in range(cap, w[i] - 1, -1):
            cand = best[c - w[i]] + v[i]
            if cand > best[c]:
                best[c] = cand
                take[i, c] = True
    c = cap
    for i in range(m - 1, -1, -1):
        out[i] = 0.0
        if take[i, c]:
            out[i] = 1.0
            c -= w[i]


@nb.njit(parallel=True, cache=True)
def _knap_batch(V, w, cap, OUT):
    for b in nb.prange(V.shape[0]):
        _knap(V[b], w, cap, OUT[b])



@nb.njit(cache=True)
def _sp_grid(v, k, out):
    """Longest path for values v (= shortest path for costs -v) on a k x k grid, moves right/down.
    Edge order: right edges (i, j)->(i, j+1) at i*(k-1)+j, then down edges (i, j)->(i+1, j)
    at k*(k-1) + i*k + j."""
    nr = k * (k - 1)
    best = np.full((k, k), -np.inf)
    frm = np.zeros((k, k), dtype=np.int64)  # 0: from left, 1: from top
    best[0, 0] = 0.0
    for i in range(k):
        for j in range(k):
            if i == 0 and j == 0:
                continue
            b = -np.inf
            f = -1
            if j > 0:
                cand = best[i, j - 1] + v[i * (k - 1) + j - 1]
                if cand > b:
                    b, f = cand, 0
            if i > 0:
                cand = best[i - 1, j] + v[nr + (i - 1) * k + j]
                if cand > b:
                    b, f = cand, 1
            best[i, j] = b
            frm[i, j] = f
    for e in range(out.shape[0]):
        out[e] = 0.0
    i, j = k - 1, k - 1
    while i > 0 or j > 0:
        if frm[i, j] == 0:
            out[i * (k - 1) + j - 1] = 1.0
            j -= 1
        else:
            out[nr + (i - 1) * k + j] = 1.0
            i -= 1


@nb.njit(parallel=True, cache=True)
def _sp_batch(V, k, OUT):
    for b in nb.prange(V.shape[0]):
        _sp_grid(V[b], k, OUT[b])


class SPSolver:
    def __init__(self, k=5):
        self.k = k

    def __call__(self, V):
        Vn = np.ascontiguousarray(V.detach().cpu().numpy().astype(np.float64))
        out = np.empty_like(Vn)
        _sp_batch(Vn, self.k, out)
        return torch.from_numpy(out).float()



class Solver:
    def __init__(self, w, cap):
        self.w, self.cap = w.astype(np.int64), cap

    def __call__(self, V):
        Vn = np.ascontiguousarray(V.detach().cpu().numpy().astype(np.float64))
        out = np.empty_like(Vn)
        _knap_batch(Vn, self.w, self.cap, out)
        return torch.from_numpy(out).float()


# ----------------------------------------------------------------------------- training
def _reg(th, reg, kappa, lam, ybar):
    return regularize(th, reg, kappa, lam, ybar)


def regret(net, X, C, solve):
    with torch.no_grad():
        Ct = torch.tensor(C, dtype=torch.float32)
        y_hat = solve(net(torch.tensor(X, dtype=torch.float32)))
        y_opt = solve(Ct)
        return float((((y_opt - y_hat) * Ct).sum(1) / (y_opt * Ct).sum(1).abs()).mean())


def train(method, reg, Xtr, Ctr, Xva, Cva, Xte, Cte, solve, lr=0.1, epochs=30, bs=32, pert=1.0, kappa_rel=1.0,
          lam=0.1, n_samples=5, seed=0, diag_every=5, kappa_abs=None):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    if reg.startswith("tn"):
        # target normalization: train on c / ||c||, rescaled to the mean norm so that the
        # perturbation calibration (ybar) is unchanged; evaluation always uses the true costs
        nrm = np.linalg.norm(Ctr, axis=1, keepdims=True)
        Ctr = Ctr / nrm * nrm.mean()
        reg = {"tn": "none", "tn_rs": "rs", "tn_rn": "rn", "tn_rc": "rc", "tn_l2p": "l2p"}[reg]
    Xtr_t, Ctr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Ctr, dtype=torch.float32)
    Str = solve(Ctr_t)
    n, m = Ctr.shape
    ybar = float(np.linalg.norm(Ctr, axis=1).mean())
    nu = {"DPO": pert * math.sqrt(m), "SPO": ybar / pert, "DBB": pert * ybar / bs}[method]
    # absolute kappa (as in the paper) if given, else kappa relative to the perturbation scale nu
    kappa = float(kappa_abs) if kappa_abs is not None else effective_kappa(1.0, kappa_rel, nu)
    net = torch.nn.Linear(Xtr.shape[1], m)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for k in range(0, n, bs):
            idx = perm[k:k + bs]
            x, c, s = Xtr_t[idx], Ctr_t[idx], Str[idx]
            th = net(x)
            u, pen = _reg(th, reg, kappa, lam, ybar)
            if method == "SPO":
                # maximisation c^T y == minimisation with costs -c and prediction -u; the SPO+
                # subgradient w.r.t. u is then solve(alpha*u - c) - y*(c)
                grad = solve(pert * u.detach() - c) - s
                loss = (u * grad).sum() / len(idx) + pen
            elif method == "DBB":
                y_hat = solve(u.detach())
                y_pert = solve(u.detach() + pert * c / len(idx))
                ghat = (y_pert - y_hat) / pert                       # ascent direction for the regret reduction
                loss = -(u * ghat).sum() / len(idx) + pen
            else:  # DPO, Gaussian noise; gradient E[y(u + sigma Z) Z^T] c  (no 1/sigma, as in the repo)
                Z = torch.randn((n_samples,) + tuple(u.shape), generator=g)
                Y = solve((u.detach().unsqueeze(0) + pert * Z).reshape(-1, m)).reshape(n_samples, len(idx), m)
                # c^T dE[y]/du with dE[y]/du ~ mean_s Y_s Z_s^T, computed as mean_s (c.Y_s) Z_s
                # (same quantity, O(m) instead of O(m^2) memory/time)
                gu = ((Y * c.unsqueeze(0)).sum(-1, keepdim=True) * Z).mean(0)
                loss = -(u * gu).sum() / len(idx) + pen
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % diag_every == 0 or ep == 0:
            with torch.no_grad():
                th = net(Xtr_t[:256])
                u, _ = _reg(th, reg, kappa, lam, ybar)
                base = solve(u)
                if method == "DPO":
                    pc = u + pert * torch.randn(u.shape, generator=g)
                elif method == "DBB":
                    pc = u + pert * Ctr_t[:256] / bs
                else:
                    pc = pert * u - Ctr_t[:256]
                d = rel_l1(solve(pc), base).mean().item()
                hist.append({"epoch": ep + 1, "norm": th.norm(dim=1).mean().item(), "dist_same": d})
    return regret(net, Xte, Cte, solve), regret(net, Xva, Cva, solve), hist




def two_stage(Xtr, Ctr, Xte, Cte, solve):
    """Least-squares two-stage baseline: affine regression of the costs, then optimization."""
    A = np.c_[Xtr, np.ones(len(Xtr))]
    W = np.linalg.lstsq(A, Ctr, rcond=None)[0]

    class _LS(torch.nn.Module):
        def forward(self, x):
            return torch.cat([x, torch.ones(len(x), 1)], 1) @ torch.tensor(W, dtype=torch.float32)
    return regret(_LS(), Xte, Cte, solve)
