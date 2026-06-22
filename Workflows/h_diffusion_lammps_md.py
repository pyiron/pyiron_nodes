from pyiron_nodes.atomistic.calculator.data import InputCalcMD, OutputCalcMD
from pyiron_nodes.atomistic.engine.lammps import CreateLammpsMDInput, CreateLammpsStructure, ListPotentials, ParseLammpsOutput, RunLammpsCalculation
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import FixSpecies, Repeat
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
)
from pyiron_nodes.controls import pick_element
from pyiron_nodes.plotting import Plot
from core import Workflow


wf = Workflow("h_diffusion_lammps_md")

wf.al_bulk = Bulk(name='Al', cubic=True)

wf.md_params = InputCalcMD(temperature=800.0, n_ionic_steps=100000)

wf.al_supercell = Repeat(structure=wf.al_bulk, repeat_scalar=3)

wf.al_h_structure = AddInterstitialH(structure=wf.al_supercell, repeat_scalar=3)

wf.list_potentials = ListPotentials(structure=wf.al_h_structure)

wf.fix_species = FixSpecies(structure=wf.al_h_structure)

wf.potential = pick_element(lst=wf.list_potentials, index=0)

wf.lammps_structure = CreateLammpsStructure(structure=wf.fix_species, potential=wf.potential, working_directory='./h_diffusion_md')

wf.lammps_input = CreateLammpsMDInput(io_bundle=wf.lammps_structure, calc_dataclass=wf.md_params)

wf.lammps_run = RunLammpsCalculation(io_bundle=wf.lammps_input)
wf.lammps_run.inputs.add("debug", port_type=bool, default=False, value=False, has_explicit_default=True)

wf.parsed_output = ParseLammpsOutput(io_bundle=wf.lammps_run.outputs.io_bundle)
wf.parsed_output.inputs.add("store", port_type=bool, default=False, value=True, has_explicit_default=True)

wf.md_output = OutputCalcMD(input=wf.parsed_output)

wf.msd = ComputeMSD(md_output=wf.parsed_output)

wf.h_pos_plot = PlotHPositions(md_output=wf.parsed_output, md_input=wf.md_params)

wf.temp_plot = Plot(y=wf.md_output.outputs.temperatures, x=wf.md_output.outputs.steps)

wf.diffusion = DiffusionConstant(msd=wf.msd, md_input=wf.md_params)

wf.msd_plot = Plot(y=wf.msd, x=wf.diffusion.outputs.times)

# ── Free energy surface (Boltzmann inversion + symmetry augmentation) ──────────
wf.folded_h_pos = FoldPositionsToUnitCell(md_output=wf.parsed_output, al_bulk=wf.al_bulk)
wf.augmented_h_pos = AugmentWithSymmetry(folded_positions=wf.folded_h_pos, al_bulk=wf.al_bulk)
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
    al_bulk=wf.al_bulk,
)
wf.migration_path_plot = PlotMigrationPath(
    free_energy=wf.free_energy_surface.outputs.free_energy,
    grid_centers=wf.free_energy_surface.outputs.grid_centers,
    augmented_positions=wf.augmented_h_pos,
    al_bulk=wf.al_bulk,
    md_input=wf.md_params,
)
