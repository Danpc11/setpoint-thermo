"""Sign of the cross-mobilities L_eff_ee' across networks, cost exponents and coupling strengths."""
import numpy as np
from _common import parse, dump
from vasctherm import Network, run, initial_state, active_edges
from vasctherm.config import RESULTS, B_VALUES

args = parse(__doc__)
seeds = range(2) if args.quick else range(6)
rows = []
for seed in seeds:
    for src in ["corner", "center"]:
        net = Network(m=6, seed=seed, source=src)
        x0 = initial_state(net, seed=seed + 10)
        for b in B_VALUES:
            o = run(net, x0, b, "gauss", dt=0.01, steps=6000, retraction=True)
            x, lam = o["x"], o["lam"][-1]
            act = active_edges(x)
            st = net.state(x, b)
            z = st["s"] / np.sqrt(lam)
            La = (1 / (2 * lam * b * st["C"] * (z + 1)))[act]
            K = net.K_matrix(x, 1.0)[np.ix_(act, act)]
            for eps in [0.01, 1.0, 10.0]:
                Le = np.linalg.inv(np.diag(1 / La) + K * eps / np.median(La * np.diag(K)))
                off = Le[~np.eye(len(La), dtype=bool)]
                nz = np.abs(off) > 1e-12 * np.abs(np.diag(Le)).max()
                rows.append(dict(seed=seed, source=src, b=b, eps=eps,
                                 n_positive=int((off[nz] > 0).sum()), n_nonzero=int(nz.sum())))
summary = {str(eps): dict(cases=sum(r["eps"] == eps for r in rows),
                          cases_with_positive=sum(r["eps"] == eps and r["n_positive"] > 0 for r in rows))
           for eps in [0.01, 1.0, 10.0]}
print(summary)
dump(dict(summary=summary, rows=rows), RESULTS / "sign_cross_mobility.json")
