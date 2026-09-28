"""Shared settings: networks, output paths and figure palette."""
from pathlib import Path
from .network import Network
from .dynamics import initial_state

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

BLUE, RED, GREY = "#1f4e9c", "#c0392b", "#6b6b6b"
B_COLORS = {0.5: BLUE, 1.0: RED, 1.5: GREY}
B_VALUES = (0.5, 1.0, 1.5)


def main_network():
    """6x6 jittered lattice used in Table 1 and Figs. 1-2."""
    net = Network(m=6, seed=1)
    return net, initial_state(net, seed=3)


def mc_network():
    """5x5 jittered lattice used in Fig. 3 (Monte Carlo)."""
    net = Network(m=5, seed=2)
    return net, initial_state(net, seed=4)
