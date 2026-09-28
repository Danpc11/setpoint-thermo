"""Fig. 1: structural mobilities from fluctuations (FDT), cross-mobilities and range of the coupling."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import parse, dump
from vasctherm import (fixed_point, force_and_mobility, active_edges, langevin_increments,
                       edge_distance)
from vasctherm.config import main_network, RESULTS, FIGURES, B_VALUES, B_COLORS, BLUE, RED, GREY

args = parse(__doc__)
net, x0 = main_network()
rng = np.random.default_rng(11)
nsteps = 4000 if args.quick else 40000
res = {}
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))

for b in B_VALUES:
    xs, lam = fixed_point(net, x0, b)
    o = langevin_increments(net, xs, lam, b, rng, nsteps=nsteps)
    Lm = np.diag(o["Lmeas"])
    pC = np.polyfit(np.log(o["C"]), np.log(Lm), 1)[0]
    pr = np.polyfit(np.log(o["r"]), np.log(Lm), 1)[0]
    res[f"b={b}"] = dict(slope_vs_C=pC, slope_vs_r=pr, expected_slope_vs_r=-2 * b,
                         median_rel_error=float(np.median(np.abs(Lm / np.diag(o["Lth"]) - 1))))
    ax[0].loglog(o["C"], Lm * lam, "o", ms=4, color=B_COLORS[b], label=f"$b={b}$: slope {pC:.3f}")
    cc = np.sort(o["C"])
    ax[0].loglog(cc, 1 / (4 * b * cc), "-", color=B_COLORS[b], lw=1)
ax[0].set_xlabel("local maintenance cost $C_e=r_e^{2b}\\ell_e^{b}$")
ax[0].set_ylabel("$\\lambda L_e/\\kappa$ (measured, FDT)")
ax[0].set_title("(a) mobility vs. cost, uncoupled", fontsize=10)
ax[0].legend(fontsize=8)

b = 1.0
xs, lam = fixed_point(net, x0, b)
o = langevin_increments(net, xs, lam, b, rng, eps=0.3, nsteps=nsteps)
Lm, Lt = o["Lmeas"], o["Lth"]
dg = np.sqrt(np.outer(np.diag(Lt), np.diag(Lt)))
iu = np.triu_indices_from(Lt, 1)
ax[1].plot(Lt[iu] / dg[iu], Lm[iu] / dg[iu], ".", ms=3, color=RED, label="off-diagonal")
ax[1].plot([-0.4, 0.05], [-0.4, 0.05], "-", color=GREY, lw=0.8)
ax[1].set_xlabel("$L^{\\rm eff}_{ee'}/\\sqrt{L_{ee}L_{e'e'}}$ (theory)")
ax[1].set_ylabel("measured (increment covariance)")
ax[1].set_title("(b) cross-mobilities, $\\varepsilon=0.3$", fontsize=10)
res["coupled"] = dict(min_offdiag_normalized=float((Lt[iu] / dg[iu]).min()),
                      fraction_negative=float((Lt[iu] < 0).mean()),
                      corr_theory_measured=float(np.corrcoef(Lt[iu], Lm[iu])[0, 1]))

act = active_edges(xs)
X, L, st = force_and_mobility(net, xs, b, lam)
Dm = edge_distance(net, act)
K0 = net.K_matrix(xs, 1.0)[np.ix_(act, act)]
La = L[act]
med = np.median(La * np.diag(K0))
for eps, c in [(0.01, BLUE), (0.1, GREY), (1.0, RED)]:
    Lt = np.linalg.inv(np.diag(1 / La) + K0 * eps / med)
    dgn = np.sqrt(np.outer(np.diag(Lt), np.diag(Lt)))
    ds = np.arange(1, Dm.max() + 1)
    m_ = np.array([np.mean(np.abs(Lt[Dm == d]) / dgn[Dm == d]) for d in ds])
    ok = m_ > 1e-8
    ax[2].semilogy(ds[ok], m_[ok], "o-", ms=4, color=c, label=f"$\\varepsilon={eps}$")
ax[2].axvspan(12.5, Dm.max() + 0.5, color="0.9")
ax[2].text(12.7, 3e-3, "branches separated\nby the source:\nno coupling", fontsize=7)
ax[2].set_xlabel("edge-to-edge distance along the tree")
ax[2].set_ylabel("$|L^{\\rm eff}_{ee'}|/\\sqrt{L_{ee}L_{e'e'}}$")
ax[2].set_title("(c) range of the coupling", fontsize=10)
ax[2].legend(fontsize=8)
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(FIGURES / f"fig_mobility.{ext}", dpi=130)
dump(res, RESULTS / "fig1_mobility.json")
