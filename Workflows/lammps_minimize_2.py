from pyiron_nodes.atomistic.calculator.data import InputCalcMinimize, OutputCalcMinimize
from pyiron_nodes.atomistic.engine.lammps import (
    CreateLammpsMinimizeInput,
    CreateLammpsStructure,
    ListPotentials,
    ParseLammpsMinimizeOutput,
    RunLammpsCalculation,
)
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import Repeat
from pyiron_nodes.controls import pick_element
from core import Workflow

wf = Workflow("lammps_minimize")

wf.Bulk = Bulk(name="Al", cubic=True)

wf.Repeat = Repeat(structure=wf.Bulk, repeat_scalar=3)

wf.ListPotentials = ListPotentials(structure=wf.Repeat)

wf.pick_element = pick_element(lst=wf.ListPotentials, index=0)

wf.InputCalcMinimize = InputCalcMinimize()

wf.CreateLammpsStructure = CreateLammpsStructure(
    structure=wf.Repeat,
    potential=wf.pick_element,
    working_directory="./lammps_minimize",
)

wf.CreateLammpsMinimizeInput = CreateLammpsMinimizeInput(
    io_bundle=wf.CreateLammpsStructure, calc_dataclass=wf.InputCalcMinimize
)

wf.RunLammpsCalculation = RunLammpsCalculation(io_bundle=wf.CreateLammpsMinimizeInput)
wf.RunLammpsCalculation.inputs.add(
    "debug", port_type=bool, default=False, value=False, has_explicit_default=True
)

wf.ParseLammpsMinimizeOutput = ParseLammpsMinimizeOutput(
    io_bundle=wf.RunLammpsCalculation.outputs.io_bundle
)

wf.OutputCalcMinimize = OutputCalcMinimize(
    input=wf.ParseLammpsMinimizeOutput.outputs.out
)
