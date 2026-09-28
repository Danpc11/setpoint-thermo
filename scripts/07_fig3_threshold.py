"""Fig. 3 (c): dependence of the loop count on the detection threshold r_th."""
import numpy as np
from _common import parse, dump
from vasctherm import fixed_point
from vasctherm.montecarlo import metropolis
from vasctherm.config import mc_network, RESULTS, B_VALUES

args = parse(__doc__)
net, x0 = mc_network()
rng = np.random.default_rng(9)
theta, thresholds = 1.78e-3, [0.03, 0.1, 0.3]
out = {}
for b in B_VALUES:
    xs, lam = fixed_point(net, x0, b)
    st = net.state(xs, b)
    Phis = st["D"] + lam * st["C"].sum()
    _, samples, _ = metropolis(net, b, lam, theta * Phis, xs, 100 if args.quick else 1000, rng)
    out[str(b)] = {str(t): float(np.mean([net.cycles(x, t) for x in samples[::5]])) for t in thresholds}
    print(b, out[str(b)])
dump(dict(theta=theta, loops=out), RESULTS / "fig3_threshold.json")
