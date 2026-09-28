"""Table 1: budget violation for quenched vs Gaussian multiplier, with and without retraction."""
import numpy as np
from _common import parse, dump
from vasctherm import run
from vasctherm.config import main_network, RESULTS, B_VALUES

args = parse(__doc__)
net, x0 = main_network()
Ttot = 6.0 if args.quick else 60.0
dts = [0.05, 0.02] if args.quick else [0.05, 0.02, 0.005]
schemes = [("quenched", "quenched", False), ("quenched+retraction", "quenched", True),
           ("gauss", "gauss", False), ("gauss+retraction", "gauss", True)]
rows = []
for b in B_VALUES:
    ref = run(net, x0, b, "gauss", dt=0.005, steps=int(Ttot / 0.005), retraction=True)
    lam_star = ref["lam"][-1]
    for dt in dts:
        for name, sch, ret in schemes:
            o = run(net, x0, b, sch, dt=dt, steps=int(Ttot / dt), lam_fixed=lam_star, retraction=ret)
            rows.append(dict(b=b, dt=dt, scheme=name, max_drift=float(o["drift"].max()),
                             final_drift=float(o["drift"][-1]), D_rel=float(o["Dfin"] / ref["Dfin"] - 1),
                             loops=o["loops"],
                             same_topology=bool(np.array_equal(np.exp(o["x"]) > 0.1, np.exp(ref["x"]) > 0.1))))
            print(rows[-1])
dump(rows, RESULTS / "table1_budget_drift.json")


def fmt(v):
    if v < 1e-14:
        return r"$<10^{-15}$"
    e = int(np.floor(np.log10(v)))
    return f"${v / 10**e:.1f}\\times10^{{{e}}}$"


get = lambda b, dt, s, k="max_drift": [r for r in rows if r["b"] == b and r["dt"] == dt and r["scheme"] == s][0][k]
lines = []
for b in B_VALUES:
    for dt in dts:
        lines.append(f"{b} & {dt} & {fmt(get(b, dt, 'quenched'))} & {fmt(get(b, dt, 'quenched', 'final_drift'))} & "
                     f"{fmt(get(b, dt, 'gauss'))} & {fmt(get(b, dt, 'gauss+retraction'))} \\\\")
    lines.append(r"\hline")
(RESULTS / "table1_rows.tex").write_text("\n".join(lines) + "\n")
print("wrote", RESULTS / "table1_rows.tex")
