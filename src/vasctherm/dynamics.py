"""Deterministic structural dynamics (Sections 2 and 5).

Local rule           x'_e = kappa (z_e - 1),                z_e = s_e / sqrt(lambda)
Gaussian multiplier  sqrt(lambda(t)) = sum_e C_e s_e / sum_e C_e      (Eq. 16, closed form)
Retraction           uniform shift of free edges restoring C_b = C_0 exactly (Appendix A.5)
"""
import numpy as np

RMIN = 1e-2                 # floor radius
XMIN = np.log(RMIN)


def velocity(net, x, b, kappa=1.0, lam=None):
    """Velocity of the local rule.

    If lam is None the Gaussian multiplier is used, computed over the edges that are not
    clamped at the floor. Returns (velocity, lambda, state).
    """
    st = net.state(x, b)
    s, C = st["s"], st["C"]
    floor = x <= XMIN + 1e-12
    active = np.ones(net.E, bool)
    for _ in range(5):
        sl = np.sum(C[active] * s[active]) / np.sum(C[active]) if lam is None else np.sqrt(lam)
        v = kappa * (s / sl - 1.0)
        blocked = floor & (v < 0)
        if np.array_equal(~blocked, active):
            break
        active = ~blocked
    v[blocked] = 0.0
    return v, sl**2, st


def retract(net, x, b, C0):
    """Exact retraction onto C_b = C_0: since C_e ~ exp(2 b x_e), shift the free edges."""
    C = np.exp(x)**(2 * b) * net.ell**b
    free = x > XMIN + 1e-12
    delta = np.log((C0 - C[~free].sum()) / C[free].sum()) / (2 * b)
    x = x.copy()
    x[free] += delta
    return np.maximum(x, XMIN)


def run(net, x0, b, scheme, dt=0.05, kappa=1.0, lam_fixed=None, steps=4000, retraction=False):
    """Integrate the local rule with explicit Euler.

    scheme : "gauss" (Gaussian multiplier) or "quenched" (lambda = lam_fixed).
    Returns final state, budget drift |C_b/C_0 - 1| per step, lambda(t), D(t), loop count.
    """
    x = x0.copy()
    C0 = net.cost(x0, b)
    drift, lams, Ds = [], [], []
    for _ in range(steps):
        lam = lam_fixed if scheme == "quenched" else None
        v, lam_t, st = velocity(net, x, b, kappa, lam)
        x = np.maximum(x + dt * v, XMIN)
        if retraction:
            x = retract(net, x, b, C0)
        drift.append(abs(net.cost(x, b) / C0 - 1.0))
        lams.append(lam_t)
        Ds.append(st["D"])
    st = net.state(x, b)
    return dict(x=x, drift=np.array(drift), lam=np.array(lams), D=np.array(Ds),
                Dfin=st["D"], loops=net.cycles(x, 10 * RMIN),
                nact=int((np.exp(x) > 10 * RMIN).sum()))


def fixed_point(net, x0, b, dt=0.005, steps=12000):
    """Optimal tree reached by the Gaussian dynamics with retraction: returns (x*, lambda*)."""
    ref = run(net, x0, b, "gauss", dt=dt, steps=steps, retraction=True)
    return ref["x"], ref["lam"][-1]


def initial_state(net, seed, scale=0.1):
    """Dense initial state x_e ~ N(0, scale^2)."""
    return scale * np.random.default_rng(seed).standard_normal(net.E)
