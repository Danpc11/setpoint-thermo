"""Fig. 2: time-reversal test of reciprocity with an upstream conducted signal (Eq. 26)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import parse, dump
from vasctherm import fixed_point
from vasctherm.nonreciprocal import (linearize, build, entropy_production, relax_time,
                                     lagged_correlation, simulate, asymmetry_statistic)
from vasctherm.config import main_network, RESULTS, FIGURES, BLUE, RED, GREY

args = parse(__doc__)
rng = np.random.default_rng(21)
net, x0 = main_network()
xs, lam = fixed_point(net, x0, 1.0)
S = linearize(net, xs, lam, 1.0)
pairs = S["pairs"]
out = {}

# (a) housekeeping entropy production vs gamma
gammas = np.array([0.02, 0.05, 0.1, 0.2, 0.4])
A0, N0, C0, _, _ = build(S, 0.0)
tau = relax_time(A0)
sig = {eps: [entropy_production(*build(S, g, eps=eps)[:3]) * tau for g in gammas] for eps in (0.0, 0.3)}
out["tau_relax"] = tau
out["sigma_tau"] = {str(k): v for k, v in sig.items()}
out["sigma_at_gamma0"] = {str(eps): entropy_production(*build(S, 0.0, eps=eps)[:3]) * tau for eps in (0.0, 0.3)}
out["fitted_exponent"] = float(np.polyfit(np.log(gammas), np.log(sig[0.0]), 1)[0])

# (b) strongest parent-child pair
A1, N1, C1, _, _ = build(S, 0.2)
As = lagged_correlation(A1, C1, tau)
As = As - As.T
nrm = np.sqrt(np.outer(np.diag(C1), np.diag(C1)))
ip, jc = pairs[int(np.argmax([abs(As[i, j]) / nrm[i, j] for i, j in pairs]))]
lags = np.linspace(0, 4 * tau, 60)

# (c) detection power
dt, lag = tau / 2, 2
Nts = [10, 30] if args.quick else [10, 30, 100, 300]
ntr = 40 if args.quick else 300
power = {}
for g in (0.1, 0.3):
    A, N, C, _, _ = build(S, g)
    Ct = lagged_correlation(A, C, lag * dt)
    signs = np.sign([(Ct - Ct.T)[i, j] for i, j in pairs])
    for eta in (0.0, 0.5, 1.0):
        pw = []
        for Nt in Nts:
            nsteps = int(Nt * tau / dt)
            s1 = asymmetry_statistic(simulate(A, C, dt, nsteps, ntr, eta, rng), pairs, lag, signs)
            s0 = asymmetry_statistic(simulate(A0, C0, dt, nsteps, ntr, eta, rng), pairs, lag, signs)
            pw.append(float((s1 > np.quantile(s0, 0.95)).mean()))
        power[f"gamma={g},eta={eta}"] = pw
        print(g, eta, pw)
out["observation_times"] = Nts
out["power"] = power
dump(out, RESULTS / "fig2_reciprocity.json")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
ax[0].loglog(gammas, sig[0.0], "o-", color=BLUE, label="$\\varepsilon=0$")
ax[0].loglog(gammas, sig[0.3], "s--", color=RED, mfc="none", label="$\\varepsilon=0.3$")
ax[0].loglog(gammas, sig[0.0][2] * (gammas / gammas[2])**2, ":", color=GREY, label="$\\propto\\gamma^2$")
ax[0].set_xlabel("conducted-signal strength $\\gamma$")
ax[0].set_ylabel("$\\sigma_{\\rm hk}\\,\\tau_{\\rm relax}$")
ax[0].set_title("(a) housekeeping entropy production", fontsize=10)
ax[0].text(0.03, 0.9, "$\\gamma=0$: $\\sigma<10^{-26}$ (both $\\varepsilon$)", transform=ax[0].transAxes, fontsize=8)
ax[0].legend(fontsize=8, loc="lower right")
for g, e_, c in [(0.0, 0.3, GREY), (0.2, 0.0, RED)]:
    A, N, C, _, _ = build(S, g, eps=e_)
    nm = np.sqrt(C[ip, ip] * C[jc, jc])
    fw = [lagged_correlation(A, C, t)[ip, jc] / nm for t in lags]
    bw = [lagged_correlation(A, C, t)[jc, ip] / nm for t in lags]
    ax[1].plot(lags / tau, fw, "-", color=c, label=f"$\\gamma={g},\\ \\varepsilon={e_}$: parent$(t+\\tau)$, child$(t)$")
    ax[1].plot(lags / tau, bw, "--", color=c, label=f"$\\gamma={g},\\ \\varepsilon={e_}$: child$(t+\\tau)$, parent$(t)$")
ax[1].set_xlabel("lag $\\tau/\\tau_{\\rm relax}$")
ax[1].set_ylabel("normalized cross-correlation")
ax[1].set_title("(b) time-reversal asymmetry, parent--child pair", fontsize=10)
ax[1].legend(fontsize=7)
for eta, c in [(0.0, BLUE), (0.5, GREY), (1.0, RED)]:
    ax[2].semilogx(Nts, power[f"gamma=0.1,eta={eta}"], "o-", color=c, label=f"$\\gamma=0.1$, $\\eta={eta}$")
    ax[2].semilogx(Nts, power[f"gamma=0.3,eta={eta}"], "s--", color=c, mfc="none", label=f"$\\gamma=0.3$, $\\eta={eta}$")
ax[2].axhline(0.05, color=GREY, lw=0.8, ls=":")
ax[2].set_xlabel("observation time $T_{\\rm obs}/\\tau_{\\rm relax}$")
ax[2].set_ylabel("detection power (5% false positives)")
ax[2].set_title("(c) detectability", fontsize=10)
ax[2].legend(fontsize=7, ncol=2, loc="upper left")
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(FIGURES / f"fig_nonreciprocal.{ext}", dpi=130)
