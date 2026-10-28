from pyiron_nodes.atomistic.engine.lammps import (
    CreateLammpsStructure,
    ListPotentials,
    RunLammpsCalculation,
)
from pyiron_nodes.atomistic.engine.lammps_input import LammpsInit, WriteFile
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.controls import pick_element
from pyiron_nodes.executors import SlurmAdvancedSettings, SlurmExecutor
from core import Workflow
from core import group_node

# ── Group node factories ─────────────────────────────


@group_node("io_bundle")
def LammpsMiniimze(
    io_bundle,
    init,
    e_tol,
    f_tol,
    pressure=None,
    run=None,
    write_restart=None,
):
    from pyiron_nodes.atomistic.engine.lammps_input import (
        AssembleLammpsInput,
        LammpsDump,
        LammpsMinimize,
        LammpsPotential,
        LammpsThermo,
    )
    from core import Workflow

    inner_wf = Workflow("LammpsMiniimze")
    inner_wf.LammpsDump = LammpsDump(io_bundle=io_bundle)
    inner_wf.LammpsMinimize = LammpsMinimize(
        io_bundle=io_bundle, e_tol=e_tol, f_tol=f_tol, pressure=pressure
    )
    inner_wf.LammpsPotential = LammpsPotential(io_bundle=io_bundle)
    inner_wf.LammpsThermo = LammpsThermo(io_bundle=io_bundle)
    inner_wf.AssembleLammpsInput = AssembleLammpsInput(
        io_bundle=io_bundle,
        init=init,
        potential=inner_wf.LammpsPotential.outputs.potential,
        dump=inner_wf.LammpsDump,
        thermo=inner_wf.LammpsThermo,
        calculation=inner_wf.LammpsMinimize,
        run=run,
        write_restart=write_restart,
    )
    return inner_wf.AssembleLammpsInput


@group_node("io_bundle")
def LammpsMD(io_bundle, run=None, write_restart=None):
    from pyiron_nodes.atomistic.engine.lammps_input import (
        AssembleLammpsInput,
        LammpsEnsemble,
        LammpsVelocity,
    )
    from core import Workflow

    inner_wf = Workflow("LammpsMD")
    inner_wf.LammpsEnsemble = LammpsEnsemble(io_bundle=io_bundle)
    inner_wf.LammpsVelocity = LammpsVelocity(io_bundle=io_bundle)
    inner_wf.AssembleLammpsInput_1 = AssembleLammpsInput(
        io_bundle=io_bundle,
        calculation=inner_wf.LammpsEnsemble.outputs.ensemble,
        calculation_fix_ids=inner_wf.LammpsEnsemble.outputs.fix_ids,
        velocity=inner_wf.LammpsVelocity,
        run=run,
        write_restart=write_restart,
    )
    return inner_wf.AssembleLammpsInput_1


wf = Workflow("modular_example_grouped")

wf.Bulk = Bulk(name="Al", cubic=True)

wf.SlurmAdvancedSettings = SlurmAdvancedSettings(
    threads_per_core=4,
    submission_template="",
    pysqa_config_directory="/cmmc/u/sanand/pyiron/projects/03_DFT/02_02_Formalizing_Workflow/pysqa_config",
)

wf.ListPotentials = ListPotentials(structure=wf.Bulk)

wf.SlurmExecutor = SlurmExecutor(
    partition="s.cmmg", cache_directory="./cache2", advanced=wf.SlurmAdvancedSettings
)

wf.pick_element = pick_element(lst=wf.ListPotentials, index=0)

wf.CreateLammpsStructure = CreateLammpsStructure(
    structure=wf.Bulk, potential=wf.pick_element, working_directory="./modLammps"
)

wf.LammpsInit = LammpsInit(io_bundle=wf.CreateLammpsStructure)

wf.LammpsMiniimze = LammpsMiniimze(
    io_bundle=wf.CreateLammpsStructure,
    init=wf.LammpsInit,
    e_tol=0.001,
    f_tol=0.001,
    pressure=0,
    run=0,
    write_restart="minimize.restart",
)

wf.LammpsMD = LammpsMD(io_bundle=wf.LammpsMiniimze, run=1000, write_restart="")

wf.WriteFile = WriteFile(io_bundle=wf.LammpsMiniimze)

wf.RunLammpsCalculation = RunLammpsCalculation(
    io_bundle=wf.LammpsMD, debug=False, executor=wf.SlurmExecutor
)
