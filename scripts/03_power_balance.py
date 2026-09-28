"""Eq. (17): power balance of the coupled sectors along relaxations towards the optimal tree."""
import numpy as np
from _common import parse, dump
from vasctherm import fixed_point, force_and_mobility, active_edges
from vasctherm.config import main_network, RESULTS

args = parse(__doc__)
net, x0 = main_network()
b = 1.0
xs, lam = fixed_point(net, x0, b)
act = active_edges(xs)
rng = np.random.default_rng(5)
X, L, st = force_and_mobility(net, xs, b, lam)
K0 = net.K_matrix(xs, 1.0)[np.ix_(act, act)]
med = np.median(L[act] * np.diag(K0))
Phi = lambda x: (lambda s: s["D"] + lam * s["C"].sum())(net.state(x, b))
nsteps, ncheck = (600, 400) if args.quick else (3000, 2000)
out = {}
for eps in [0.0, 0.01, 0.1, 0.3, 1.0]:
    x = xs.copy()
    x[act] += 0.15 * rng.standard_normal(act.sum())
    dt, worst, share, monotone = 1e-3, 0.0, [], True
    for k in range(nsteps):
        X, L, _ = force_and_mobility(net, x, b, lam)
        La = L[act]
        Ka = net.K_matrix(x, 1.0)[np.ix_(act, act)] * eps / med
        v = np.linalg.solve(np.diag(1 / La) + Ka, X[act])
        wall, hyd = v @ (v / La), v @ Ka @ v
        P1 = Phi(x)
        x2 = x.copy()
        x2[act] += dt * v
        P2 = Phi(x2)
        if k < ncheck and wall + hyd > 1e-8:
            worst = max(worst, abs(-(P2 - P1) / dt / (wall + hyd) - 1))
            share.append(hyd / (wall + hyd))
        monotone &= P2 <= P1 + 1e-12
        x = x2
    out[str(eps)] = dict(max_rel_error_balance=worst, mean_hydraulic_share=float(np.mean(share)),
                         Phi_monotone=bool(monotone))
    print(eps, out[str(eps)])
dump(out, RESULTS / "power_balance.json")
