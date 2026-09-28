"""Time-reversal test of reciprocity (Section 6, Eq. 26; Appendix A.10).

Linearized dynamics around the optimal tree (fixed lambda*):
    d(dx) = -A dx dt + sqrt(2) B dW,  A = (I + gamma G) L_eff H,  B B^T = N = T L_eff
G_ee' = exp(-d_ee'/xi) if e' is downstream of e (upstream conduction), else 0.
The noise obeys the fluctuation-dissipation relation with respect to L_eff only.
"""
from collections import deque
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, expm, cholesky
from .dynamics import XMIN


def linearize(net, xs, lam, b, kappa=1.0, h=1e-6):
    """Hessian, mobilities and tree orientation at the fixed point (xs, lam)."""
    act = np.where(xs > XMIN + 1e-9)[0]
    st = net.state(xs, b)
    z = st["s"] / np.sqrt(lam)
    L = kappa / (2 * lam * b * st["C"] * (z + 1))

    def grad(x):
        s = net.state(x, b)
        zz = s["s"] / np.sqrt(lam)
        return -2 * lam * b * s["C"] * (zz**2 - 1)

    H = np.zeros((len(act), len(act)))
    for j, e in enumerate(act):
        xp, xm = xs.copy(), xs.copy()
        xp[e] += h
        xm[e] -= h
        H[:, j] = (grad(xp)[act] - grad(xm)[act]) / (2 * h)
    H = 0.5 * (H + H.T)

    f = st["f"]
    tail = np.where(f >= 0, net.edges[:, 0], net.edges[:, 1])
    head = np.where(f >= 0, net.edges[:, 1], net.edges[:, 0])
    children = {e: [e2 for e2 in act if tail[e2] == head[e]] for e in act}
    idx = {e: i for i, e in enumerate(act)}
    Ddown = np.full((len(act), len(act)), np.inf)
    for e in act:
        q = deque([(e, 0)])
        while q:
            u, d = q.popleft()
            Ddown[idx[e], idx[u]] = d
            for c in children[u]:
                q.append((c, d + 1))
    K0 = net.K_matrix(xs, 1.0)[np.ix_(act, act)]
    pairs = [(idx[e], idx[c]) for e in act for c in children[e]]
    return dict(act=act, L=L[act], H=H, Ddown=Ddown, K0=K0, pairs=pairs)


def build(S, gamma, eps=0.0, xi=3.0, T=1.0):
    """Drift matrix A, noise matrix N, stationary covariance C, conduction matrix G, L_eff."""
    La = S["L"]
    if eps == 0:
        Leff = np.diag(La)
    else:
        Leff = np.linalg.inv(np.diag(1 / La) + S["K0"] * eps / np.median(La * np.diag(S["K0"])))
    G = np.where(np.isfinite(S["Ddown"]) & (S["Ddown"] > 0), np.exp(-S["Ddown"] / xi), 0.0)
    A = (np.eye(len(La)) + gamma * G) @ Leff @ S["H"]
    N = T * Leff
    C = solve_continuous_lyapunov(A, 2 * N)
    return A, N, 0.5 * (C + C.T), G, Leff


def entropy_production(A, N, C):
    """Stationary (housekeeping) entropy production of the OU process."""
    V = A - N @ np.linalg.inv(C)
    return float(np.trace(V.T @ np.linalg.solve(N, V) @ C))


def relax_time(A):
    return 1.0 / np.linalg.eigvals(A).real.min()


def lagged_correlation(A, C, tau):
    """<dx(t+tau) dx(t)^T> = exp(-A tau) C."""
    return expm(-A * tau) @ C


def simulate(A, C, dt, nsteps, ntrials, eta, rng):
    """Exact discrete-time OU trajectories with additive measurement noise eta * sd."""
    n = A.shape[0]
    F = expm(-A * dt)
    Q = C - F @ C @ F.T
    Lq = cholesky(0.5 * (Q + Q.T) + 1e-14 * np.eye(n), lower=True)
    x = rng.standard_normal((ntrials, n)) @ cholesky(C, lower=True).T
    X = np.empty((nsteps, ntrials, n))
    for k in range(nsteps):
        X[k] = x
        x = x @ F.T + rng.standard_normal((ntrials, n)) @ Lq.T
    return X + eta * np.sqrt(np.diag(C)) * rng.standard_normal(X.shape)


def asymmetry_statistic(X, pairs, lag, signs):
    """Sum over parent-child pairs of the sign-weighted normalized lagged asymmetry."""
    P = np.array(pairs)
    Xc = X - X.mean(0)
    sd = Xc.std(0)
    a, b = P[:, 0], P[:, 1]
    fwd = np.mean(Xc[lag:, :, a] * Xc[:-lag, :, b], 0)
    bwd = np.mean(Xc[lag:, :, b] * Xc[:-lag, :, a], 0)
    return (((fwd - bwd) / (sd[:, a] * sd[:, b])) * signs).sum(1)
