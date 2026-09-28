"""Structural Onsager quantities, hydraulic coupling and Langevin dynamics (Sections 4, 6).

Force        X_e = -dPhi/dx_e = 2 lambda b C_e (z_e^2 - 1)                    (Eq. 11)
Mobility     L_e = kappa / [2 lambda b C_e (z_e + 1)]                          (Eq. 12)
Coupling     L_eff = (Gamma^{-1} + K)^{-1},  eps_e = Gamma_e K_ee              (Eq. 16)
"""
from collections import deque
import numpy as np
from .dynamics import XMIN


def force_and_mobility(net, x, b, lam, kappa=1.0):
    st = net.state(x, b)
    z = st["s"] / np.sqrt(lam)
    X = 2 * lam * b * st["C"] * (z**2 - 1)
    L = kappa / (2 * lam * b * st["C"] * (z + 1))
    return X, L, st


def active_edges(x):
    return x > XMIN + 1e-9


def coupling_scale(net, x, L, act, eps):
    """Volume prefactor nu such that median(Gamma_e K_ee) = eps."""
    K0 = net.K_matrix(x, 1.0)[np.ix_(act, act)]
    return np.sqrt(eps / np.median(L[act] * np.diag(K0)))


def effective_mobility(L_active, K_active):
    return np.linalg.inv(np.diag(1 / L_active) + K_active)


def langevin_increments(net, xs, lam, b, rng, eps=0.0, Tfac=1e-4, dt=2e-3, nsteps=40000, kappa=1.0):
    """Euler-Maruyama around the optimal tree with noise covariance 2 T L_eff dt.

    Returns measured mobilities (increment covariance / 2 T dt) and their theoretical values.
    """
    act = active_edges(xs)
    X, L, st = force_and_mobility(net, xs, b, lam, kappa)
    T = Tfac * lam * st["C"][act].mean()
    nu = coupling_scale(net, xs, L, act, eps) if eps > 0 else 0.0
    x = xs.copy()
    inc = np.zeros((nsteps, act.sum()))
    for k in range(nsteps):
        X, L, _ = force_and_mobility(net, x, b, lam, kappa)
        La = L[act]
        Leff = effective_mobility(La, net.K_matrix(x, nu)[np.ix_(act, act)]) if nu > 0 else np.diag(La)
        dx = dt * Leff @ X[act] + np.linalg.cholesky(2 * T * dt * Leff) @ rng.standard_normal(act.sum())
        x[act] += dx
        inc[k] = dx
    X, L, st = force_and_mobility(net, xs, b, lam, kappa)
    Lth = (effective_mobility(L[act], net.K_matrix(xs, nu)[np.ix_(act, act)])
           if nu > 0 else np.diag(L[act]))
    return dict(C=st["C"][act], r=st["r"][act], Lmeas=np.cov(inc.T) / (2 * T * dt),
                Lth=Lth, T=T, act=act)


def edge_distance(net, act):
    """Distance between active edges (number of steps through shared nodes)."""
    ids = np.where(act)[0]
    nb = {i: [] for i in range(len(ids))}
    for i, e in enumerate(ids):
        for j, f in enumerate(ids):
            if i < j and set(net.edges[e]) & set(net.edges[f]):
                nb[i].append(j)
                nb[j].append(i)
    Dm = np.zeros((len(ids), len(ids)), int)
    for s in range(len(ids)):
        dist = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in nb[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for v, dd in dist.items():
            Dm[s, v] = dd
    return Dm
