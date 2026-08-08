"""
Application 3 — High-throughput atomistic materials modeling with aiflow
========================================================================
Backs the "high-throughput atomistic materials modeling" application in
``docs/paper_draft.md``.

A lattice-parameter (equation-of-state) sweep for an fcc metal, expressed as a
single ``IterToDataFrame`` call over an energy node, followed by a
Birch-Murnaghan fit that extracts the equilibrium volume, cohesive energy, and
bulk modulus.

To keep the example reproducible with no external simulation engine (LAMMPS/DFT),
the per-atom energy is evaluated from an **analytical Morse pair potential**
summed over fcc coordination shells. In production the same workflow topology
holds with ``RunLammps`` or an ASE/DFT calculator node in place of ``fcc_energy``
— swapping the engine is a one-line change (the paper's "pluggable potential"
point).
"""

import sys

sys.path.insert(0, "pyiron_core/src")
sys.path.insert(0, "/Users/jorgneugebauer/git_libs/pyiron_nodes")

import numpy as np

from core import Workflow, as_function_node
from pyiron_nodes.controls import IterToDataFrame


# fcc coordination shells: (multiplicity, neighbour distance in units of a)
_FCC_SHELLS = [
    (12, np.sqrt(1 / 2)),
    (6, 1.0),
    (24, np.sqrt(3 / 2)),
    (12, np.sqrt(2.0)),
    (24, np.sqrt(5 / 2)),
]


@as_function_node(["a", "volume", "energy"])
def fcc_energy(a: float = 3.6, D: float = 0.35, alpha: float = 1.6, r0: float = 2.55):
    """Cohesive energy per atom of an fcc lattice with parameter *a* (Angstrom).

    Analytical Morse pair sum over fcc shells. *D* (eV), *alpha* (1/Ang) and
    *r0* (Ang) are Morse parameters chosen to give a sensible metal-like curve.
    """

    def morse(r):
        x = np.exp(-alpha * (r - r0))
        return D * (x * x - 2.0 * x)

    e = 0.5 * sum(mult * morse(dist * a) for mult, dist in _FCC_SHELLS)
    volume = a ** 3 / 4.0  # volume per atom (4 atoms per conventional fcc cell)
    return float(a), float(volume), float(e)


def ev_sweep(a_values=None):
    """Equation-of-state sweep: one IterToDataFrame call over lattice parameter."""
    if a_values is None:
        a_values = list(np.linspace(3.0, 4.0, 21))
    wf = Workflow("ev_sweep")
    wf.template = fcc_energy(D=0.35, alpha=1.6, r0=2.55)
    wf.sweep = IterToDataFrame(
        node=wf.template, input_label="a", values=a_values
    )
    return wf.run()


def birch_murnaghan_fit(volume, energy):
    """3rd-order Birch-Murnaghan fit → (E0, V0, B0 in GPa, B0_prime)."""
    from scipy.optimize import curve_fit

    volume = np.asarray(volume, float)
    energy = np.asarray(energy, float)

    def bm(V, E0, V0, B0, Bp):
        eta = (V0 / V) ** (2.0 / 3.0)
        return E0 + 9.0 * V0 * B0 / 16.0 * (
            (eta - 1.0) ** 3 * Bp + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
        )

    imin = int(np.argmin(energy))
    p0 = [energy[imin], volume[imin], 0.5, 4.0]  # B0 here in eV/Ang^3
    popt, _ = curve_fit(bm, volume, energy, p0=p0, maxfev=10000)
    E0, V0, B0_ev, Bp = popt
    B0_GPa = B0_ev * 160.21766208  # eV/Ang^3 -> GPa
    return {"E0": E0, "V0": V0, "B0_GPa": B0_GPa, "Bp": Bp, "popt": popt, "bm": bm}


if __name__ == "__main__":
    df = ev_sweep()
    print("=== Energy-volume sweep (fcc Morse) ===")
    print(df[["a", "volume", "energy"]].round(4).to_string(index=False))
    fit = birch_murnaghan_fit(df["volume"], df["energy"])
    print("\n=== Birch-Murnaghan fit ===")
    print(f"  E0 = {fit['E0']:.4f} eV/atom")
    print(f"  V0 = {fit['V0']:.4f} Ang^3/atom  (a0 = {(4*fit['V0'])**(1/3):.4f} Ang)")
    print(f"  B0 = {fit['B0_GPa']:.1f} GPa")
    print(f"  B0'= {fit['Bp']:.2f}")
