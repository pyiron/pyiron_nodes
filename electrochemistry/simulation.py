from core import group_node


@group_node("new_structure")
def ElectrochemicalCell(
    element,
    size="1 1 1",
    vacuum=1.0,
    orthogonal=False,
    water_width=10.0,
    cation="Cl",
    number=0,
    seed=1234,
    density=1e-24,
):
    from pyiron_nodes.atomistic.structure.build import Surface
    from pyiron_nodes.atomistic.structure.transform import FixSpecies
    from pyiron_nodes.electrochemistry.structure.build import (
        AddIonPair,
        add_neon_layer,
        add_water_film,
    )
    from core import Workflow

    inner_wf = Workflow("ElectrochemicalCell")
    inner_wf.al_slab = Surface(
        element=element, size=size, vacuum=vacuum, orthogonal=orthogonal
    )
    inner_wf.water_cell = add_water_film(
        electrode=inner_wf.al_slab, water_width=water_width, density=density
    )
    inner_wf.electrolyte_with_ions = AddIonPair(
        structure=inner_wf.water_cell, cation=cation, no_of_pairs=number, seed=seed
    )
    inner_wf.full_cell = add_neon_layer(structure=inner_wf.electrolyte_with_ions)
    inner_wf.fixed_cell_inner = FixSpecies(
        structure=inner_wf.full_cell, fixed_species='["Al", "Ne"]'
    )
    return inner_wf.fixed_cell_inner.outputs.new_structure


@group_node("sim_setup", "structure")
def SimulationSetup(
    element,
    size="1 1 1",
    vacuum=1.0,
    orthogonal=False,
    water_width=10.0,
    cation="Cl",
    number=0,
    seed=1234,
    density=1e-24,
    metal_charge=0.0,
    neon_charge=0.0,
    quasi_2d=False,
):
    from pyiron_nodes.atomistic.calculator.data import SimSetupBundleInp
    from pyiron_nodes.electrochemistry.structure.build import ConfigurePBC
    from pyiron_nodes.electrochemistry.structure.equilibrate import IonPotential
    from core import Workflow

    inner_wf = Workflow("SimulationSetup")
    inner_wf.cell = ElectrochemicalCell(
        element=element,
        size=size,
        vacuum=vacuum,
        orthogonal=orthogonal,
        water_width=water_width,
        cation=cation,
        number=number,
        seed=seed,
        density=density,
    )
    inner_wf.potential = IonPotential(
        metal_charge=metal_charge, neon_charge=neon_charge, quasi_2d=quasi_2d
    )
    inner_wf.pbc = ConfigurePBC(
        structure=inner_wf.cell, charges=inner_wf.potential.outputs.charges
    )
    inner_wf.bundle = SimSetupBundleInp(
        structure=inner_wf.pbc,
        water_potential=inner_wf.potential.outputs.water_potential,
        bond_dict=inner_wf.potential.outputs.bond_dict,
        charges=inner_wf.potential.outputs.charges,
    )
    return inner_wf.bundle.outputs.output, inner_wf.pbc.outputs.structure


@group_node("out", "sim_setup")
def Lammps(sim_setup, calc_dataclass, threads_per_core=1):
    from pyiron_nodes.atomistic.calculator.data import SimSetupBundle, SimSetupBundleInp
    from pyiron_nodes.atomistic.engine.lammps import (
        CreateLammpsMDInput,
        CreateLammpsStructure,
        ParseElectrodeForce,
        ParseLammpsOutput,
        RunLammpsCalculation,
    )
    from core import Workflow

    inner_wf = Workflow("Lammps")
    inner_wf.unpacked = SimSetupBundle(input=sim_setup)
    inner_wf.lammps_structure = CreateLammpsStructure(
        structure=inner_wf.unpacked.outputs.structure,
        potential=inner_wf.unpacked.outputs.water_potential,
        bond_dict=inner_wf.unpacked.outputs.bond_dict,
    )
    inner_wf.md_input = CreateLammpsMDInput(
        io_bundle=inner_wf.lammps_structure, calc_dataclass=calc_dataclass
    )
    inner_wf.md_run = RunLammpsCalculation(
        io_bundle=inner_wf.md_input,
        threads_per_core=threads_per_core,
        debug=False,
        executor=None,
    )
    inner_wf.lammps_output = ParseLammpsOutput(
        io_bundle=inner_wf.md_run.outputs.io_bundle
    )
    inner_wf.electrode_force = ParseElectrodeForce(
        io_bundle=inner_wf.md_run.outputs.io_bundle
    )
    inner_wf.result_bundle = SimSetupBundleInp(
        structure=inner_wf.unpacked.outputs.structure,
        water_potential=inner_wf.unpacked.outputs.water_potential,
        bond_dict=inner_wf.unpacked.outputs.bond_dict,
        charges=inner_wf.unpacked.outputs.charges,
        electrode_forces=inner_wf.electrode_force,
    )
    return inner_wf.lammps_output.outputs.out, inner_wf.result_bundle.outputs.output
