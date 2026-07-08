from pyiron_nodes.atomistic.calculator.data import OutputCalcMD
from pyiron_nodes.atomistic.diffusion import (
    ComputeMSD,
    DiffusionConstant,
    ExtractMigrationBarrier,
    PlotFreeEnergySurface,
    PlotHPositions,
    PlotMigrationPath,
)
from pyiron_nodes.plotting import Plot
from core import Workflow
from core import group_node

# ── Group node factories ─────────────────────────────


@group_node("new_structure", "structure")
def Supercell(name, cubic=False, repeat_scalar=1, al_h_structure__repeat_scalar=1):
    from pyiron_nodes.atomistic.diffusion import AddInterstitialH
    from pyiron_nodes.atomistic.structure.build import Bulk
    from pyiron_nodes.atomistic.structure.transform import FixSpecies, Repeat
    from core import Workflow

    inner_wf = Workflow("Supercell")
    inner_wf.al_bulk = Bulk(name=name, cubic=cubic)
    inner_wf.al_supercell = Repeat(
        structure=inner_wf.al_bulk, repeat_scalar=repeat_scalar
    )
    inner_wf.al_h_structure = AddInterstitialH(
        structure=inner_wf.al_supercell, repeat_scalar=al_h_structure__repeat_scalar
    )
    inner_wf.fix_species = FixSpecies(structure=inner_wf.al_h_structure)
    return inner_wf.fix_species, inner_wf.al_bulk


@group_node("out", "output")
def Lammps(
    structure,
    list_potentials__structure,
    temperature,
    n_ionic_steps,
    n_print,
    pressure,
    time_step,
    temperature_damping_timescale,
    pressure_damping_timescale,
    seed,
    tloop,
    initial_temperature,
    langevin,
    delta_temp,
    delta_press,
    index,
    working_directory=".",
    store=False,
):
    from pyiron_nodes.atomistic.calculator.data import InputCalcMD
    from pyiron_nodes.atomistic.engine.lammps import (
        CreateLammpsMDInput,
        CreateLammpsStructure,
        ListPotentials,
        ParseLammpsOutput,
        RunLammpsCalculation,
    )
    from pyiron_nodes.controls import pick_element
    from core import Workflow

    inner_wf = Workflow("Lammps")
    inner_wf.list_potentials = ListPotentials(structure=list_potentials__structure)
    inner_wf.md_params = InputCalcMD(
        temperature=temperature,
        n_ionic_steps=n_ionic_steps,
        n_print=n_print,
        pressure=pressure,
        time_step=time_step,
        temperature_damping_timescale=temperature_damping_timescale,
        pressure_damping_timescale=pressure_damping_timescale,
        seed=seed,
        tloop=tloop,
        initial_temperature=initial_temperature,
        langevin=langevin,
        delta_temp=delta_temp,
        delta_press=delta_press,
    )
    inner_wf.potential = pick_element(lst=inner_wf.list_potentials, index=index)
    inner_wf.lammps_structure = CreateLammpsStructure(
        structure=structure,
        potential=inner_wf.potential,
        working_directory=working_directory,
    )
    inner_wf.lammps_input = CreateLammpsMDInput(
        io_bundle=inner_wf.lammps_structure, calc_dataclass=inner_wf.md_params
    )
    inner_wf.lammps_run = RunLammpsCalculation(
        io_bundle=inner_wf.lammps_input, debug=False
    )
    inner_wf.parsed_output = ParseLammpsOutput(
        io_bundle=inner_wf.lammps_run.outputs.io_bundle, store=store
    )
    return inner_wf.parsed_output, inner_wf.md_params


@group_node("free_energy", "grid_centers", "augmented_positions")
def FreeEnergySurface(al_bulk, augmented_h_pos__al_bulk, md_output, md_input):
    from pyiron_nodes.atomistic.diffusion import (
        AugmentWithSymmetry,
        ComputeFreeEnergySurface,
        FoldPositionsToUnitCell,
    )
    from core import Workflow

    inner_wf = Workflow("FreeEnergySurface")
    inner_wf.folded_h_pos = FoldPositionsToUnitCell(
        md_output=md_output, al_bulk=al_bulk
    )
    inner_wf.augmented_h_pos = AugmentWithSymmetry(
        folded_positions=inner_wf.folded_h_pos, al_bulk=augmented_h_pos__al_bulk
    )
    inner_wf.free_energy_surface = ComputeFreeEnergySurface(
        augmented_positions=inner_wf.augmented_h_pos, md_input=md_input
    )
    return (
        inner_wf.free_energy_surface,
        inner_wf.free_energy_surface,
        inner_wf.augmented_h_pos,
    )


wf = Workflow("h_diffusion_md")

wf.Supercell = Supercell(
    name="Al", cubic=True, repeat_scalar=3, al_h_structure__repeat_scalar=3
)

wf.Lammps = Lammps(
    structure=wf.Supercell.outputs.new_structure,
    list_potentials__structure=wf.Supercell.outputs.new_structure,
    temperature=800.0,
    n_ionic_steps=100000,
    index=0,
    working_directory="./h_diffusion_md",
    store=True,
)
wf.Lammps.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)

wf.md_output = OutputCalcMD(input=wf.Lammps.outputs.out)

wf.msd = ComputeMSD(md_output=wf.Lammps.outputs.out)

wf.h_pos_plot = PlotHPositions(
    md_output=wf.Lammps.outputs.out, md_input=wf.Lammps.outputs.output
)

wf.FreeEnergySurface = FreeEnergySurface(
    al_bulk=wf.Supercell.outputs.structure,
    augmented_h_pos__al_bulk=wf.Supercell.outputs.structure,
    md_output=wf.Lammps.outputs.out,
    md_input=wf.Lammps.outputs.output,
)

wf.temp_plot = Plot(y=wf.md_output.outputs.temperatures, x=wf.md_output.outputs.steps)

wf.diffusion = DiffusionConstant(msd=wf.msd, md_input=wf.Lammps.outputs.output)

wf.migration_barrier = ExtractMigrationBarrier(
    free_energy=wf.FreeEnergySurface.outputs.free_energy,
    grid_centers=wf.FreeEnergySurface.outputs.grid_centers,
)

wf.free_energy_plot = PlotFreeEnergySurface(
    free_energy=wf.FreeEnergySurface.outputs.free_energy,
    grid_centers=wf.FreeEnergySurface.outputs.grid_centers,
    al_bulk=wf.Supercell.outputs.structure,
)

wf.migration_path_plot = PlotMigrationPath(
    free_energy=wf.FreeEnergySurface.outputs.free_energy,
    grid_centers=wf.FreeEnergySurface.outputs.grid_centers,
    augmented_positions=wf.FreeEnergySurface.outputs.augmented_positions,
    al_bulk=wf.Supercell.outputs.structure,
    md_input=wf.Lammps.outputs.output,
)

wf.msd_plot = Plot(y=wf.msd, x=wf.diffusion.outputs.times)
