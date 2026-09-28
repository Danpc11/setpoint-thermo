"""vasctherm: non-equilibrium thermodynamics of adaptively remodeling vascular networks.

Companion code for Hernandez-Lemus & Perez-Calixto, "A non-equilibrium thermodynamics
formulation of shear set-points in adaptively remodeling vascular networks".
"""
from .network import Network
from .dynamics import RMIN, XMIN, velocity, retract, run, fixed_point, initial_state
from .structural import (force_and_mobility, active_edges, coupling_scale,
                         effective_mobility, langevin_increments, edge_distance)

__version__ = "1.0.0"
