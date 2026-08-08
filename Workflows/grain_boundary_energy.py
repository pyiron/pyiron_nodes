from __future__ import annotations

import numpy as np

from core import Workflow, as_function_node
from pyiron_nodes.atomistic.structure.build_gb import GrainBoundaryOptions, BuildGrainBoundary
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.engine.ase import EMT
from pyiron_nodes.atomistic.calculator.ase import StaticEnergy, Minimize
from pyiron_nodes.atomistic.calculator.output import GetEnergyLast
from pyiron_nodes.atomistic.structure.view import PlotCNA


@as_function_node("energy_per_atom")
def BulkEnergyPerAtom(total_energy: float = 0.0, structure=None):
    """Divide total bulk energy by number of atoms to get the per-atom reference."""
    n_atoms = len(structure)
    energy_per_atom = total_energy / n_atoms
    return energy_per_atom


@as_function_node("gamma_meV_per_A2")
def GrainBoundaryEnergy(
    gb_total_energy: float = 0.0,
    bulk_energy_per_atom: float = 0.0,
    gb_structure=None,
):
    """
    Compute the grain boundary energy γ [meV/Å²].

    γ = (E_GB - N_GB · e_bulk) / (2 · A)   [eV/Å²] × 1000  →  meV/Å²

    The factor 2 accounts for the two equivalent boundary planes in the
    fully periodic bicrystal supercell.
    """
    n_gb = len(gb_structure.symbols)
    cell = np.array(gb_structure.cell)
    v1, v2 = cell[0], cell[1]
    area_A2 = np.linalg.norm(np.cross(v1, v2))

    excess_eV = gb_total_energy - n_gb * bulk_energy_per_atom
    gamma_meV_per_A2 = (excess_eV / (2.0 * area_A2)) * 1000.0
    return gamma_meV_per_A2


wf = Workflow("grain_boundary_energy")

# ── Enumerate all Σ5 FCC Al CSL configurations ───────────────────────────────
wf.gb_options = GrainBoundaryOptions(
    sigma=5,
    crystalstructure="fcc",
    a=4.05,
)

# ── Build the smallest Σ5 bicrystal (index 0 = fewest atoms) ─────────────────
wf.gb_structure = BuildGrainBoundary(
    options=wf.gb_options.outputs.options,
    index=0,
    symbol="Al",
    min_slab_thickness=15.0,
    vacuum=0.0,
    merge_tol=0.5,
)

# ── Bulk reference unit cell ─────────────────────────────────────────────────
wf.bulk = Bulk(name="Al", crystalstructure="fcc", a=4.05)

# ── Calculator ───────────────────────────────────────────────────────────────
wf.engine = EMT()

# ── Relax GB supercell, then extract final energy ─────────────────────────────
wf.relax_gb = Minimize(
    structure=wf.gb_structure.outputs.structure,
    engine=wf.engine.outputs.engine,
    fmax=0.005,
)
wf.gb_energy_total = GetEnergyLast(calculator=wf.relax_gb.outputs.out)

# ── Static energy of bulk reference unit cell ─────────────────────────────────
wf.bulk_energy = StaticEnergy(
    structure=wf.bulk.outputs.structure,
    engine=wf.engine.outputs.engine,
)

# ── Per-atom bulk reference energy ───────────────────────────────────────────
wf.bulk_energy_per_atom = BulkEnergyPerAtom(
    total_energy=wf.bulk_energy.outputs.energy,
    structure=wf.bulk.outputs.structure,
)

# ── Grain boundary energy γ [meV/Å²] ─────────────────────────────────────────
wf.gb_energy = GrainBoundaryEnergy(
    gb_total_energy=wf.gb_energy_total.outputs.energy_last,
    bulk_energy_per_atom=wf.bulk_energy_per_atom.outputs.energy_per_atom,
    gb_structure=wf.gb_structure.outputs.structure,
)

# ── Visualise with CNA colouring (GB atoms appear as non-FCC) ────────────────
wf.view = PlotCNA(structure=wf.gb_structure.outputs.structure)
