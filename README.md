# Vascular thermodynamics

Companion code for

> E. Hernández-Lemus and D. Pérez-Calixto, *A non-equilibrium thermodynamics formulation of shear
> set-points in adaptively remodeling vascular networks*, (2026).

The package implements the adaptive flow-network model of the paper (Poiseuille hydraulics, power-law
maintenance cost, local shear-set-point rule) together with the quantities introduced by the
non-equilibrium thermodynamic formulation: structural Onsager coefficients, the Gaussian metabolic
multiplier, the hydraulic–structural coupling through the displaced volume, the Langevin extension,
the partition function over topologies, and the time-reversal test of reciprocity. Every figure and
table of the paper is produced by one script, and the identities derived in Appendix A are checked by
the test suite.

## Installation

```bash
git clone https://github.com/Danpc11/setpoint-thermo.git
cd setpoint-thermo
python -m pip install -e ".[test]"
```

Requires Python ≥ 3.9 with NumPy, SciPy and Matplotlib. The Python package is imported as
`vasctherm`.

## Reproducing the paper

```bash
make test     # identities of Appendix A (seconds)
make all      # every table and figure (about 3 minutes on a laptop)
make quick    # smoke run of every script with reduced statistics (about 1 minute)
```

| Paper | Script | Output |
|---|---|---|
| Table 1 — budget violation, quenched vs Gaussian multiplier | `scripts/01_table1_budget_drift.py` | `results/table1_budget_drift.json`, `results/table1_rows.tex` |
| Fig. 1 — mobilities from fluctuations, cross-mobilities, range of the coupling | `scripts/02_fig1_mobility.py` | `figures/fig_mobility.pdf`, `results/fig1_mobility.json` |
| Eq. (17) — power balance of the coupled sectors | `scripts/03_power_balance.py` | `results/power_balance.json` |
| Sec. 4.5 (ii) — sign of the cross-mobilities over 12 networks | `scripts/04_sign_cross_mobility.py` | `results/sign_cross_mobility.json` |
| Fig. 2 — time-reversal test of reciprocity | `scripts/05_fig2_reciprocity_test.py` | `figures/fig_nonreciprocal.pdf`, `results/fig2_reciprocity.json` |
| Fig. 3 (a, b) — loop count vs effective temperature (Monte Carlo) | `scripts/06_fig3_loop_count_mc.py` | `results/fig3_loop_count.json` |
| Fig. 3 (c) — dependence on the detection threshold | `scripts/07_fig3_threshold.py` | `results/fig3_threshold.json` |
| Fig. 3 — plot | `scripts/08_plot_fig3.py` | `figures/fig_collapse.pdf` |

All random number generators are seeded, so the outputs are bit-for-bit reproducible on a given
platform. Each script accepts `--quick` for a short run.

## Model in brief

For an edge $e$ with radius $r_e$ and length $\ell_e$, in reduced units,

| Quantity | Definition |
|---|---|
| conductance | $w_e = r_e^4/\ell_e$ |
| wall shear stress | $\tau_e = \lvert f_e\rvert/r_e^3$ |
| maintenance cost | $C_e = r_e^{2b}\ell_e^{b}$, $C_b=\sum_e C_e = C_0$ |
| composite functional | $\Phi = D + \lambda C_b$, $D=\sum_e f_e^2/w_e$ |
| shear set-point | $\tau_{\rm set} = \sqrt{\lambda b/2}\ r_e^{b-1}\ell_e^{(b-1)/2}$ |
| local rule | $\dot x_e = \kappa(z_e-1)$, $x_e=\ln r_e$, $z_e=\lvert\tau_e\rvert/\tau_{\rm set}$ |
| structural force and mobility | $X_e = 2\lambda b C_e(z_e^2-1)$, $L_e = \kappa/[2\lambda b C_e(z_e+1)]$ |
| Gaussian multiplier | $\sqrt{\lambda(t)} = \sum_e C_e s_e/\sum_e C_e$, $s_e=\sqrt{\lambda} z_e$ |
| coupled mobility | $L^{\rm eff} = (\Gamma^{-1}+K)^{-1}$, $K = V'A_{\rm r}^{\mathsf T}L_{\rm r}^{-1}A_{\rm r}V'$ |
| conducted signal | $\dot x = (\mathbb I+\gamma G)L^{\rm eff}X + \xi$ |

Networks are jittered square lattices with a single corner source and unit sink demands; a floor
radius $r_{\min}=10^{-2}$ prevents singular conductances. See Appendix B of the paper for all
numerical parameters.

## Repository layout

```
src/vasctherm/
    network.py        lattice, hydraulics, costs, loop counting, coupling matrix K
    dynamics.py       local rule, quenched and Gaussian multipliers, exact retraction
    structural.py     forces, mobilities, effective mobility, Langevin increments
    montecarlo.py     Metropolis sampling with Sherman-Morrison updates, Hessian
    nonreciprocal.py  linearized dynamics with conduction, housekeeping entropy, detection test
    config.py         networks used in the paper, output paths, colour palette
scripts/              one script per table or figure (numbered in running order)
tests/                numerical checks of the identities of Appendix A
results/              JSON outputs and LaTeX table rows
figures/              PDF and PNG figures
```

## Tests

`tests/test_identities.py` checks, on a small network:

- the set-point relation and the structural force (A.1, A.3);
- the Lyapunov identity of the local rule (Eq. 6);
- exact budget conservation by the Gaussian multiplier (Eq. 16) and by the retraction (A.5);
- symmetry, positivity and first-order sign of the hydraulic coupling (A.7);
- the identity between $\dot x^{\mathsf T}K\dot x$ and the viscous dissipation of the induced flows (A.6);
- vanishing of the housekeeping entropy production without conduction (A.10);
- the transient budget violation of the quenched multiplier (Table 1).
