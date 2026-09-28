"""Flow network model.

Reduced units (Section 2 of the paper):
    w_e = r_e^4 / l_e        Poiseuille conductance
    tau_e = |f_e| / r_e^3    wall shear stress
    C_e = r_e^(2b) l_e^b     maintenance cost of edge e   (= c_e w_e^alpha, alpha = b/2)

The source pressure is fixed to zero and every other node withdraws one unit of flow,
so pressures are obtained from the grounded (reduced) Laplacian.
"""
import numpy as np


class Network:
    """Jittered square lattice with a single source.

    Parameters
    ----------
    m : int
        Number of nodes per side (m*m nodes, 2m(m-1) edges).
    jitter : float
        Maximum uniform displacement of node positions, in lattice units.
    seed : int
        Seed for the node displacements.
    source : {"corner", "center"}
        Location of the source node.
    """

    def __init__(self, m=6, jitter=0.2, seed=0, source="corner"):
        rng = np.random.default_rng(seed)
        xs, ys = np.meshgrid(np.arange(m), np.arange(m))
        pos = np.c_[xs.ravel(), ys.ravel()].astype(float)
        pos += jitter * rng.uniform(-1, 1, pos.shape)
        edges = []
        for i in range(m):
            for j in range(m):
                k = i * m + j
                if j + 1 < m:
                    edges.append((k, k + 1))
                if i + 1 < m:
                    edges.append((k, k + m))
        self.pos, self.edges = pos, np.array(edges)
        self.n, self.E = len(pos), len(edges)
        self.ell = np.linalg.norm(pos[self.edges[:, 0]] - pos[self.edges[:, 1]], axis=1)
        B = np.zeros((self.n, self.E))
        B[self.edges[:, 0], np.arange(self.E)] = 1.0
        B[self.edges[:, 1], np.arange(self.E)] = -1.0
        self.B = B
        self.src = 0 if source == "corner" else (m // 2) * m + m // 2
        self.keep = np.array([i for i in range(self.n) if i != self.src])
        self.Br = B[self.keep]              # reduced incidence matrix (source removed)
        self.Ar = 0.5 * np.abs(self.Br)     # assigns half of each volume change to each end node
        self.d = np.ones(self.n - 1)        # sink demands

    # ------------------------------------------------------------------ hydraulics
    def solve(self, x):
        """Flows and pressure drops for log-radii x."""
        r = np.exp(x)
        w = r**4 / self.ell
        Lr = (self.Br * w) @ self.Br.T
        phi = np.linalg.solve(Lr, -self.d)          # (B f)_i = I_i = -1 at sinks
        dp = self.Br.T @ phi
        f = w * dp
        return r, w, f, dp, Lr

    def state(self, x, b):
        """Hydraulic and cost quantities.

        Returns a dict with r, w, f, dp, Lr, C (edge costs), D (dissipation) and
        s = sqrt(lambda) * z, the lambda-independent shear ratio of Section 5.
        """
        a = b / 2
        r, w, f, dp, Lr = self.solve(x)
        C = r**(2 * b) * self.ell**b
        tau = np.abs(f) / r**3
        s = tau / (np.sqrt(a) * r**(b - 1) * self.ell**((b - 1) / 2))
        D = np.sum(f**2 / w)
        return dict(r=r, w=w, f=f, dp=dp, Lr=Lr, C=C, s=s, D=D, a=a)

    def cost(self, x, b):
        return float((np.exp(x)**(2 * b) * self.ell**b).sum())

    def cycles(self, x, thr):
        """Cycle rank (number of independent loops) of the subgraph with r_e > thr."""
        act = np.exp(x) > thr
        parent = list(range(self.n))

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        comps = self.n
        for (u, v) in self.edges[act]:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
                comps -= 1
        return int(act.sum() - self.n + comps)

    # ------------------------------------------------------ hydraulic-structural coupling
    def K_matrix(self, x, nu):
        """Hydraulic friction of wall motion, K = V' A_r^T L_r^{-1} A_r V' (Section 4.5).

        v_e = nu r_e^2 l_e is the lumen volume and V' = diag(dv_e/dx_e) = diag(2 v_e).
        """
        r, w, f, dp, Lr = self.solve(x)
        vp = 2 * nu * r**2 * self.ell
        M = self.Ar * vp
        return M.T @ np.linalg.solve(Lr, M)
