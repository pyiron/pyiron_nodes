from pyiron_nodes.atomistic.calculator.data import InputCalcMD, OutputCalcMD
from pyiron_nodes.atomistic.diffusion import (
    AddInterstitialH,
    AugmentWithSymmetry,
    ComputeFreeEnergySurface,
    ComputeMSD,
    DiffusionConstant,
    ExtractMigrationBarrier,
    FoldPositionsToUnitCell,
    PlotFreeEnergySurface,
    PlotHPositions,
    PlotMigrationPath,
    RunASEMD,
)
from pyiron_nodes.atomistic.engine.ase import GRACE
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.dpg2026.atomistic.structure.transform import Repeat
from pyiron_nodes.plotting import Plot
from core import Workflow

wf = Workflow("h_diffusion_GRACE_md")

wf.GRACE = GRACE()

wf.al_unit = Bulk(name="Al", cubic=True)

wf.al_supercell = Repeat(structure=wf.al_unit, repeat_scalar=3)

wf.al_h_structure = AddInterstitialH(structure=wf.al_supercell, repeat_scalar=3)

wf.md_params = InputCalcMD(
    temperature=800.0, n_ionic_steps=100000, n_print=100, time_step=1.0
)

wf.md_run = RunASEMD(
    structure=wf.al_h_structure, engine=wf.GRACE, md_input=wf.md_params
)
wf.md_run.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)

# ── Split raw output into named ports ─────────────────────────────────────────

wf.md_output = OutputCalcMD(input=wf.md_run)

# ── Diagnostics ───────────────────────────────────────────────────────────────

wf.temp_plot = Plot(y=wf.md_output.outputs.temperatures, x=wf.md_output.outputs.steps)

wf.h_pos_plot = PlotHPositions(md_output=wf.md_run, md_input=wf.md_params)

# ── MSD and diffusion constant ────────────────────────────────────────────────

wf.msd = ComputeMSD(md_output=wf.md_run)

wf.diffusion = DiffusionConstant(msd=wf.msd, md_input=wf.md_params)

wf.msd_plot = Plot(y=wf.msd, x=wf.diffusion.outputs.times)

# ── Free energy surface (Boltzmann inversion + symmetry augmentation) ─────────

wf.folded_h_pos = FoldPositionsToUnitCell(md_output=wf.md_run, al_bulk=wf.al_unit)

wf.augmented_h_pos = AugmentWithSymmetry(
    folded_positions=wf.folded_h_pos, al_bulk=wf.al_unit
)

wf.free_energy_surface = ComputeFreeEnergySurface(
    augmented_positions=wf.augmented_h_pos, md_input=wf.md_params
)

wf.migration_barrier = ExtractMigrationBarrier(
    free_energy=wf.free_energy_surface.outputs.free_energy,
    grid_centers=wf.free_energy_surface.outputs.grid_centers,
)

wf.free_energy_plot = PlotFreeEnergySurface(
    free_energy=wf.free_energy_surface.outputs.free_energy,
    grid_centers=wf.free_energy_surface.outputs.grid_centers,
    al_bulk=wf.al_unit,
)

wf.migration_path_plot = PlotMigrationPath(
    free_energy=wf.free_energy_surface.outputs.free_energy,
    grid_centers=wf.free_energy_surface.outputs.grid_centers,
    augmented_positions=wf.augmented_h_pos,
    al_bulk=wf.al_unit,
    md_input=wf.md_params,
)
