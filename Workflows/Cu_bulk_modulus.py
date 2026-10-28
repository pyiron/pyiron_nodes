"""
# Bulk modulus of fcc Cu from an energy-volume curve (EMT)

Equation-of-state route to the bulk modulus: build the conventional fcc Cu cell,
scan its volume, evaluate the energy of every strained cell with the same
engine, and fit a Birch-Murnaghan EOS to the resulting E(V) curve.

The engine here is ASE's built-in EMT potential, so the whole workflow runs in
well under a second and needs no downloads. EMT is a cheap effective-medium
model, not a first-principles method - it is used here because it makes the
workflow instant. Swapping `engine` for a GRACE or LAMMPS node changes the
physics and leaves the rest of the graph untouched; see the `EOS_Bulk_modulus`
workflow for the same graph driven by a foundation potential.

## Steps
1. `cu_fcc` - conventional (cubic, 4-atom) fcc Cu unit cell.
2. `engine` - EMT calculator, shared by every single-point evaluation.
3. `volume_factors` -> `strained_cells` - 13 cells spanning +/-15 % in volume.
4. `ev_container` -> `ev_energies` -> `ev_results` - single-point energies for
   every strained cell.
5. `ev_table` - (volume, energy) table, per atom.
6. `eos_fit` - Birch-Murnaghan fit. `B0_GPa` is the bulk modulus in GPa,
   `V0` the equilibrium volume in A^3/atom, `E0` the minimum energy in eV/atom.
7. `ev_plot` - the E(V) data the fit is based on.

## Expected result
fcc Cu with EMT: `B0_GPa` ~ 134 GPa (experiment ~140) and `V0` ~ 11.57
A^3/atom, i.e. a lattice constant of ~3.59 A (experiment 3.615).

## Key inputs
- `cu_fcc.name` - the element. EMT also covers Ni, Ag, Au and Al. Changing this
  to `Ni` gives ~177 GPa (experiment ~180).
- `volume_factors` - range and density of the volume scan.

## A note on the scan range
The +/-15 % range is deliberately wide. A Birch-Murnaghan fit is only
meaningful if the energy minimum lies *inside* the sampled volumes; if it does
not, the fit extrapolates and still returns a confident-looking number. The
wide range keeps the minimum bracketed for both Cu and Ni. It does *not* for Ag
or Au, whose EMT equilibrium volume is near 16.4 A^3/atom - for those, raise
`cu_fcc.a` as well. Always check `ev_plot` before trusting `eos_fit`.
"""

from pyiron_nodes.atomistic.calculator.data import OutputSEFS
from pyiron_nodes.atomistic.calculator.generic import ApplyEngine, CreateSEFSContainer
from pyiron_nodes.atomistic.engine.ase import EMT
from pyiron_nodes.atomistic.property.bulk import FitBirchMurnaghanEOS, PlotEVCurve
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import GenerateStrainedStructures
from pyiron_nodes.math_utils import Linspace
from core import Workflow
from core import as_function_node

# ── Local node definitions ──────────────────────


@as_function_node("ev_curve_df")
def EnergyVolumeTable(structures, energies, per_atom: bool = True):
    """Tabulate an energy-volume curve from structures and their energies.

    Turns the two parallel lists an ``ApplyEngine`` sweep produces into the
    ``volume``/``energy`` DataFrame the Birch-Murnaghan and plotting nodes take,
    sorted by volume.

    Parameters
    ----------
    structures : list of Atoms
        The structures the energies were computed for, in the same order.
    energies : list of float
        Total energy of each structure (eV).
    per_atom : bool
        Divide both volume and energy by the number of atoms, so the fit returns
        an energy per atom and a volume per atom.
    """
    import pandas as pd

    from pyiron_nodes.atomistic.structure._atoms import _resolve_atoms

    volume_lst, energy_lst = [], []
    for structure, energy in zip(structures, energies):
        atoms = _resolve_atoms(structure)
        n_atoms = len(atoms) if per_atom else 1
        volume_lst.append(atoms.get_volume() / n_atoms)
        energy_lst.append(float(energy) / n_atoms)

    ev_curve_df = pd.DataFrame({"volume": volume_lst, "energy": energy_lst})
    ev_curve_df = ev_curve_df.sort_values("volume").reset_index(drop=True)
    return ev_curve_df


wf = Workflow("Cu_bulk_modulus")

wf.cu_fcc = Bulk(name="Cu", crystalstructure="fcc", a=3.615, cubic=True)

wf.engine = EMT()

wf.volume_factors = Linspace(x_min=0.85, x_max=1.15, num_points=13)

wf.strained_cells = GenerateStrainedStructures(
    structure=wf.cu_fcc, strain_lst=wf.volume_factors
)

wf.ev_container = CreateSEFSContainer(structures=wf.strained_cells)

wf.ev_energies = ApplyEngine(
    sefs_container=wf.ev_container, engine=wf.engine, store=False
)

wf.ev_results = OutputSEFS(input=wf.ev_energies)

wf.ev_table = EnergyVolumeTable(
    structures=wf.ev_results.outputs.structures, energies=wf.ev_results.outputs.energies
)

wf.eos_fit = FitBirchMurnaghanEOS(ev_curve_df=wf.ev_table)

wf.ev_plot = PlotEVCurve(
    ev_curve_df=wf.ev_table,
    xlabel="Volume per atom (Å³)",
    ylabel="Energy per atom (eV)",
    title="fcc Cu energy vs volume (EMT)",
)
