"""Fig. 3: plot of the loop-count test from the results of scripts 06 and 07."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import parse
from vasctherm.config import RESULTS, FIGURES, BLUE, RED, GREY

parse(__doc__)
R = json.loads((RESULTS / "fig3_loop_count.json").read_text())
Th = json.loads((RESULTS / "fig3_threshold.json").read_text())
cols = [BLUE, RED, GREY]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for (b, v), c in zip(R.items(), cols):
    T = np.array([r["T"] for r in v["rows"]])
    Tc = np.array([r["Tcal"] for r in v["rows"]])
    be = np.array([r["beta"] for r in v["rows"]])
    ok = np.abs(Tc / T - 1) < 0.35
    ax[0].semilogx(T, be, "-", color=c, lw=1)
    ax[0].semilogx(T[ok], be[ok], "o", color=c, label=f"$b={b}$")
    ax[0].semilogx(T[~ok], be[~ok], "o", mfc="none", color=c)
    ax[1].semilogx(T / v["Phistar"], be, "o-", color=c, label=f"$b={b}$")
lm = list(R.values())[0]["loops_max"]
for a in ax[:2]:
    a.axhline(lm, ls=":", c="gray", lw=0.8)
    a.set_ylabel("$\\langle\\beta\\rangle$")
    a.legend(fontsize=8)
ax[0].set_xlabel("$T_{\\rm eff}$")
ax[0].set_title("(a) $\\langle\\beta\\rangle$ vs $T_{\\rm eff}$ (open: equipartition fails)", fontsize=9)
ax[1].set_xlabel("$T_{\\rm eff}/\\Phi^\\star(b)$")
ax[1].set_title("(b) rescaled by $\\Phi^\\star(b)$", fontsize=9)
thr = ["0.03", "0.1", "0.3"]
wdt = 0.25
for i, (b, v) in enumerate(Th["loops"].items()):
    ax[2].bar(np.arange(3) + (i - 1) * wdt, [v[t] for t in thr], wdt, color=cols[i], label=f"$b={b}$")
ax[2].set_xticks(range(3))
ax[2].set_xticklabels([f"$r_{{\\rm th}}={t}$" for t in thr])
ax[2].set_ylabel(f"$\\langle\\beta\\rangle$ at $T_{{\\rm eff}}/\\Phi^\\star={Th['theta']:.1e}$")
ax[2].set_title("(c) dependence on detection threshold", fontsize=9)
ax[2].legend(fontsize=8)
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(FIGURES / f"fig_collapse.{ext}", dpi=130)
print("wrote", FIGURES / "fig_collapse.pdf")
