"""Fig. 3 (a,b): mean loop count vs T_eff by Metropolis sampling, for three cost exponents."""
import numpy as np
from _common import parse, dump
from vasctherm import fixed_point, active_edges
from vasctherm.montecarlo import metropolis, hessian
from vasctherm.config import mc_network, RESULTS, B_VALUES

args = parse(__doc__)
net, x0 = mc_network()
rng = np.random.default_rng(7)
thetas = np.logspace(-3.5, -0.5, 4 if args.quick else 9)     # T_eff / Phi*
sweeps = 150 if args.quick else 1500
res = {}
for b in B_VALUES:
    xs, lam = fixed_point(net, x0, b)
    act = active_edges(xs)
    st = net.state(xs, b)
    Phis = st["D"] + lam * st["C"].sum()
    Hinv = np.linalg.inv(hessian(net, b, lam, xs, act))
    ids = np.where(act)[0]
    iref = int(np.argmax(np.diag(Hinv)))                        # reference edge for calibration
    rows = []
    for th in thetas:
        T = th * Phis
        beta, samples, acc = metropolis(net, b, lam, T, xs, sweeps, rng)
        Tcal = np.var(samples[:, ids[iref]]) / Hinv[iref, iref]  # equipartition
        rows.append(dict(theta=th, T=T, Tcal=Tcal, beta=beta, acceptance=acc))
        print(b, f"T/Phi*={th:.2e} <beta>={beta:.2f} Tcal/T={Tcal / T:.2f}")
    res[str(b)] = dict(Phistar=Phis, lam=lam, rows=rows, loops_max=net.E - net.n + 1)
dump(res, RESULTS / "fig3_loop_count.json")
