from pyiron_nodes.controls import IterToDataFrame
from pyiron_nodes.math_utils import Linspace
from core import Workflow
from core import group_node
from core import as_function_node

# ── Local node definitions ──────────────────────


@as_function_node(["a", "volume", "energy"])
def fcc_energy(a: float = 3.6, D: float = 0.35, alpha: float = 1.6, r0: float = 2.55):
    """Cohesive energy per atom of an fcc crystal at lattice parameter ``a``.

    A self-contained template node for an equation-of-state (E–V) sweep. The
    per-atom energy is evaluated from an analytical **Morse pair potential**
    summed over the first five fcc coordination shells — no external simulation
    engine is required, so the workflow is fully reproducible in the GUI. Sweeping
    ``a`` with ``IterToDataFrame`` traces out the binding curve; fitting the
    resulting (volume, energy) columns to a Birch–Murnaghan equation of state
    then yields the equilibrium lattice constant, cohesive energy, and bulk
    modulus.

    In production, replace this node with an ``ApplyEngine``/``Relax`` node backed
    by LAMMPS, an ASE calculator, or a machine-learned potential — the surrounding
    sweep topology is unchanged (the paper's "pluggable potential" point).

    Parameters
    ----------
    a : float
        Conventional fcc lattice parameter in Å — the swept quantity.
    D : float
        Morse well depth (eV): the depth of the pair-interaction minimum.
    alpha : float
        Morse decay constant (1/Å): controls curvature/stiffness of the well.
    r0 : float
        Morse equilibrium pair distance (Å).

    Returns
    -------
    a : float
        The lattice parameter evaluated, echoed as a column.
    volume : float
        Volume per atom (Å³) = a³ / 4 for the 4-atom conventional fcc cell.
    energy : float
        Cohesive energy per atom (eV); the ½ prefactor avoids double-counting
        the shared pair bonds.
    """
    import numpy as np

    # fcc coordination shells: (multiplicity, neighbour distance in units of a)
    shells = [
        (12, np.sqrt(0.5)),
        (6, 1.0),
        (24, np.sqrt(1.5)),
        (12, np.sqrt(2.0)),
        (24, np.sqrt(2.5)),
    ]

    def morse(r):
        x = np.exp(-alpha * (r - r0))
        return D * (x * x - 2.0 * x)

    energy = 0.5 * sum(mult * morse(dist * a) for mult, dist in shells)
    volume = a ** 3 / 4.0
    return float(a), float(volume), float(energy)


wf = Workflow("fcc_ev_curve")

wf.a_values = Linspace(x_min=3.0, x_max=4.0, num_points=21)

wf.template = fcc_energy(D=0.35, alpha=1.6, r0=2.55)

wf.sweep = IterToDataFrame(
    node=wf.template,
    input_label="a",
    values=wf.a_values,
    debug=False,
    executor=None,
    store=False,
)
