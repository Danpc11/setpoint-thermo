"""Metropolis sampling of P(x) ~ exp[-(D + lambda C_b)/T] (canonical ensemble, Section 6.3).

D is updated with rank-one Sherman-Morrison updates of G = L_r^{-1}, since changing w_e
changes L_r by dw b_e b_e^T and D = d^T G d.
"""
import numpy as np
from .dynamics import XMIN


def metropolis(net, b, lam, T, xs, sweeps, rng, thr=0.1, step=0.35, burn=0.3, refresh=50):
    """Returns (mean loop count, sampled configurations after burn-in, acceptance rate)."""
    x = xs.copy()
    w = np.exp(4 * x) / net.ell
    G = np.linalg.inv((net.Br * w) @ net.Br.T)
    d = net.d
    Ce = np.exp(2 * b * x) * net.ell**b
    loops, samples, acc = [], [], 0
    for sw in range(sweeps):
        for e in rng.permutation(net.E):
            xn = x[e] + step * rng.standard_normal()
            if xn < XMIN:
                xn = 2 * XMIN - xn                  # reflection at the floor
            wn = np.exp(4 * xn) / net.ell[e]
            dw = wn - w[e]
            be = net.Br[:, e]
            Gb = G @ be
            bGb, bGd = be @ Gb, Gb @ d
            dD = -dw * bGd**2 / (1 + dw * bGb)
            Cn = np.exp(2 * b * xn) * net.ell[e]**b
            dE = dD + lam * (Cn - Ce[e])
            if dE <= 0 or rng.random() < np.exp(-dE / T):
                G -= dw * np.outer(Gb, Gb) / (1 + dw * bGb)
                x[e], w[e], Ce[e] = xn, wn, Cn
                acc += 1
        if sw % refresh == 0:
            G = np.linalg.inv((net.Br * w) @ net.Br.T)
        if sw > burn * sweeps:
            loops.append(net.cycles(x, thr))
            samples.append(x.copy())
    return float(np.mean(loops)), np.array(samples), acc / (sweeps * net.E)


def hessian(net, b, lam, xs, act, h=1e-4):
    """Finite-difference Hessian of Phi = D + lambda C_b over the active edges."""
    def E(x):
        s = net.state(x, b)
        return s["D"] + lam * s["C"].sum()
    ids = np.where(act)[0]
    H = np.zeros((len(ids), len(ids)))
    for i, e in enumerate(ids):
        for j, f in enumerate(ids):
            if j < i:
                continue
            def ev(de, df):
                y = xs.copy()
                y[e] += de
                y[f] += df
                return E(y)
            H[i, j] = H[j, i] = (ev(h, h) - ev(h, -h) - ev(-h, h) + ev(-h, -h)) / (4 * h * h)
    return H
