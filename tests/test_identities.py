"""Numerical checks of the identities derived in the paper (Appendix A). Run with `pytest`."""
import numpy as np
import pytest
from vasctherm import Network, run, retract, fixed_point, initial_state, force_and_mobility, active_edges
from vasctherm.nonreciprocal import linearize, build, entropy_production


@pytest.fixture(scope="module")
def small():
    net = Network(m=4, seed=0)
    x0 = initial_state(net, seed=1)
    return net, x0


def test_setpoint_and_force(small):
    """A.1/A.3: (dp)^2 = z^2 lam alpha C/w and dPhi/dx = 4 lam alpha C (1 - z^2)."""
    net, x0 = small
    b, lam, h = 1.3, 2.0, 1e-6
    st = net.state(x0, b)
    z = st["s"] / np.sqrt(lam)
    assert np.allclose(st["dp"]**2, z**2 * lam * (b / 2) * st["C"] / st["w"])
    Phi = lambda x: (lambda s: s["D"] + lam * s["C"].sum())(net.state(x, b))
    num = np.array([(Phi(x0 + h * np.eye(net.E)[e]) - Phi(x0 - h * np.eye(net.E)[e])) / (2 * h)
                    for e in range(net.E)])
    assert np.allclose(num, 4 * lam * (b / 2) * st["C"] * (1 - z**2), rtol=1e-5, atol=1e-7)


def test_lyapunov_identity(small):
    """Eq. (6): dPhi/dt = -2 b lam kappa sum_e C_e (z-1)^2 (z+1) under the local rule."""
    net, x0 = small
    b, lam = 1.0, 3.0
    X, L, st = force_and_mobility(net, x0, b, lam)
    z = st["s"] / np.sqrt(lam)
    v = L * X
    assert np.allclose(v, z - 1)
    assert np.isclose(-(X * v).sum(), -2 * b * lam * np.sum(st["C"] * (z - 1)**2 * (z + 1)))


def test_gaussian_multiplier_conserves_budget(small):
    """Eq. (16): with sqrt(lam) = sum C s / sum C, sum_e h_e x'_e = 0 identically."""
    net, x0 = small
    b = 0.8
    st = net.state(x0, b)
    sl = np.sum(st["C"] * st["s"]) / np.sum(st["C"])
    assert abs(np.sum(st["C"] * (st["s"] / sl - 1))) < 1e-10


def test_retraction_is_exact(small):
    """A.5: the uniform shift restores C_b = C_0 to machine precision."""
    net, x0 = small
    b = 1.5
    C0 = net.cost(x0, b)
    x = retract(net, x0 + 0.3, b, C0)
    assert abs(net.cost(x, b) / C0 - 1) < 1e-13


def test_coupling_symmetric_and_negative(small):
    """Section 4.5: K symmetric PSD; first-order cross-mobilities negative."""
    net, x0 = small
    b = 1.0
    xs, lam = fixed_point(net, x0, b, dt=0.01, steps=3000)
    act = active_edges(xs)
    X, L, _ = force_and_mobility(net, xs, b, lam)
    K = net.K_matrix(xs, 1.0)[np.ix_(act, act)]
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-12
    La = L[act]
    first = -np.outer(La, La) * K * 1e-3 / np.median(La * np.diag(K))
    off = first[~np.eye(len(La), dtype=bool)]
    assert np.all(off[np.abs(off) > 1e-14] < 0)


def test_power_balance_identity(small):
    """A.6: x'^T K x' equals the viscous dissipation of the flows induced by wall motion."""
    net, x0 = small
    rng = np.random.default_rng(0)
    xdot = rng.standard_normal(net.E)
    r, w, f, dp, Lr = net.solve(x0)
    vp = 2 * r**2 * net.ell
    phi = -np.linalg.solve(Lr, net.Ar @ (vp * xdot))
    fdyn = w * (net.Br.T @ phi)
    assert np.isclose(np.sum(fdyn**2 / w), xdot @ net.K_matrix(x0, 1.0) @ xdot)


def test_detailed_balance_without_conduction(small):
    """A.10: sigma_hk = 0 for gamma = 0 (with or without hydraulic coupling), > 0 otherwise."""
    net, x0 = small
    xs, lam = fixed_point(net, x0, 1.0, dt=0.01, steps=3000)
    S = linearize(net, xs, lam, 1.0)
    for eps in (0.0, 0.3):
        A, N, C, _, _ = build(S, 0.0, eps=eps)
        assert abs(entropy_production(A, N, C)) < 1e-15 * max(1.0, np.abs(A).max())
        assert np.allclose(C, np.linalg.inv(S["H"]), atol=1e-12)
    A, N, C, _, _ = build(S, 0.2)
    assert entropy_production(A, N, C) > 0


def test_quenched_drift_vs_gaussian(small):
    """Table 1: quenched lambda violates the budget during the transient; retraction does not."""
    net, x0 = small
    b = 1.0
    ref = run(net, x0, b, "gauss", dt=0.01, steps=2000, retraction=True)
    q = run(net, x0, b, "quenched", dt=0.01, steps=2000, lam_fixed=ref["lam"][-1])
    assert q["drift"].max() > 1e-2
    assert ref["drift"].max() < 1e-12
