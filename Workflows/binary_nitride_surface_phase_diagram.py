"""
Generic binary wurtzite (0001) surface phase diagram.

Switch system by changing only the two Identity nodes and lattice constants:

    GaN  → cation="Ga", anion="N",  COMPOUND_A=3.19, COMPOUND_C_OVER_A=1.627, CATION_A=4.05
    AlN  → cation="Al", anion="N",  COMPOUND_A=3.11, COMPOUND_C_OVER_A=1.601, CATION_A=4.046
    ZnO  → cation="Zn", anion="O",  COMPOUND_A=3.25, COMPOUND_C_OVER_A=1.602, CATION_A=2.665

The stoichiometric constraint μ_cation + μ_anion = E_compound/fu fixes the
second degree of freedom, so the entire phase diagram is swept by varying μ_cation.
"""

from __future__ import annotations

from core import Workflow
from pyiron_nodes.atomistic.calculator.ase import Minimize
from pyiron_nodes.atomistic.calculator.output import GetEnergyLast
from pyiron_nodes.atomistic.engine.ase import GRACE
from pyiron_nodes.atomistic.property.surface import BinarySlabConfigurations, RelaxSlabsDataFrame
from pyiron_nodes.atomistic.structure.build import Bulk, BinaryWurtziteBulk
from pyiron_nodes.atomistic.structure.calc import EnergyPerFormulaUnit, NumberOfAtoms
from pyiron_nodes.atomistic.thermodynamics.defect_phases import (
    AddDefectConcentrationColumns,
    AddElementCountColumns,
    BinaryChemicalPotentialSweep,
    ComputeDefectFormationEnergy,
    PlotConvexHull,
    PlotDefectFormationEnergy,
    SelectStableStructures,
)
from pyiron_nodes.math_utils import Identity

# ── Material constants — update these when switching system ──────────────────
# GaN defaults
COMPOUND_A = 3.19        # in-plane lattice constant of the compound (Å)
COMPOUND_C_OVER_A = 1.627  # c/a ratio of the compound
CATION_A = 4.05          # lattice constant of the elemental cation reference (Å)
CATION_CRYSTAL = "fcc"   # crystal structure of the cation reference

wf = Workflow("binary_nitride_surface_phase_diagram")

# ── Element identity — single source of truth ────────────────────────────────
wf.cation = Identity(x="Ga")   # change to "Al" for AlN, "Zn" for ZnO, etc.
wf.anion = Identity(x="N")

# ── Engine ────────────────────────────────────────────────────────────────────
wf.grace_engine = GRACE(model="GRACE-2L-OAM")

# ── Bulk reference structures ─────────────────────────────────────────────────
wf.compound_bulk = BinaryWurtziteBulk(
    cation=wf.cation.outputs.x,
    anion=wf.anion.outputs.x,
    a=COMPOUND_A,
    c_over_a=COMPOUND_C_OVER_A,
)
wf.cation_bulk = Bulk(
    name=wf.cation.outputs.x,
    crystalstructure=CATION_CRYSTAL,
    a=CATION_A,
)

# ── Relax bulk references ─────────────────────────────────────────────────────
wf.compound_min = Minimize(
    structure=wf.compound_bulk,
    engine=wf.grace_engine,
    fmax=0.01,
    log_file="compound_bulk.log",
)
wf.cation_min = Minimize(
    structure=wf.cation_bulk,
    engine=wf.grace_engine,
    fmax=0.01,
    log_file="cation_bulk.log",
)

# ── Reference energies ────────────────────────────────────────────────────────
wf.compound_n_atoms = NumberOfAtoms(structure=wf.compound_bulk)
wf.cation_n_atoms = NumberOfAtoms(structure=wf.cation_bulk)

wf.compound_energy_per_fu = EnergyPerFormulaUnit(
    total_energy=GetEnergyLast(calculator=wf.compound_min).outputs.energy_last,
    n_atoms=wf.compound_n_atoms,
    atoms_per_fu=2,
)
wf.cation_energy_per_atom = EnergyPerFormulaUnit(
    total_energy=GetEnergyLast(calculator=wf.cation_min).outputs.energy_last,
    n_atoms=wf.cation_n_atoms,
    atoms_per_fu=1,
)

# ── Surface slab configurations ───────────────────────────────────────────────
wf.slab_configs = BinarySlabConfigurations(
    cation=wf.cation.outputs.x,
    anion=wf.anion.outputs.x,
    a=COMPOUND_A,
    c_over_a=COMPOUND_C_OVER_A,
    n_layers=4,
    repeat=2,
    vacuum=12.0,
    fix_bottom_fraction=0.5,
)

# ── Relax all slab configurations ─────────────────────────────────────────────
wf.slab_df = RelaxSlabsDataFrame(
    engine=wf.grace_engine,
    structures=wf.slab_configs.outputs.structures,
    names=wf.slab_configs.outputs.names,
    fmax=0.05,
    max_steps=300,
)

# ── Defect formation energy analysis ──────────────────────────────────────────
wf.df_counts = AddElementCountColumns(df=wf.slab_df)

wf.df_deltas = AddDefectConcentrationColumns(
    df=wf.df_counts.outputs.df,
    pristine_row=0,
)

wf.mu = BinaryChemicalPotentialSweep(
    mu_cation=wf.cation_energy_per_atom.outputs.energy_per_fu,
    mu_compound_per_fu=wf.compound_energy_per_fu.outputs.energy_per_fu,
    cation=wf.cation.outputs.x,
    anion=wf.anion.outputs.x,
    n_points=80,
    delta_mu_range=3.0,
)

wf.Ef = ComputeDefectFormationEnergy(
    df=wf.df_deltas.outputs.df,
    chemical_potentials=wf.mu,
    pristine_row=0,
)

# ── Visualization ─────────────────────────────────────────────────────────────
wf.plot_Ef = PlotDefectFormationEnergy(
    formation_energies=wf.Ef,
    ef_label="Relative surface energy (eV)",
    title="(0001) surface phase diagram",
)

wf.plot_hull = PlotConvexHull(
    formation_energies=wf.Ef,
    ef_label="Relative surface energy (eV)",
    title="(0001) convex hull",
    exclude_pristine=False,
)

wf.stable_df = SelectStableStructures(formation_energies=wf.Ef)
