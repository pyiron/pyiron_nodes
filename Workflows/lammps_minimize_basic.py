from pyiron_nodes.atomistic.calculator.data import (
    InputCalcMinimize,
    OutputCalcMinimize,
    OutputCalcStatic,
)
from pyiron_nodes.atomistic.engine.lammps import (
    CreateLammpsMinimizeInput,
    CreateLammpsStructure,
    ListPotentials,
    ParseLammpsOutput,
    RunLammpsCalculation,
)
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import Repeat
from pyiron_nodes.controls import pick_element
from core import Workflow
from core import group_node

wf = Workflow("lammps_minimize_basic")

wf.Bulk = Bulk(name="Al", cubic=True)

wf.InputCalcMinimize = InputCalcMinimize()

wf.Repeat = Repeat(structure=wf.Bulk, repeat_scalar=3)

wf.ListPotentials = ListPotentials(structure=wf.Repeat)

wf.pick_element = pick_element(lst=wf.ListPotentials, index=0)

wf.CreateLammpsStructure = CreateLammpsStructure(
    structure=wf.Repeat,
    potential=wf.pick_element,
    working_directory="./lammps_minimize_basic",
)

wf.CreateLammpsMinimizeInput = CreateLammpsMinimizeInput(
    io_bundle=wf.CreateLammpsStructure, calc_dataclass=wf.InputCalcMinimize
)

wf.RunLammpsCalculation = RunLammpsCalculation(
    io_bundle=wf.CreateLammpsMinimizeInput, debug=False, executor=None
)

wf.ParseLammpsOutput = ParseLammpsOutput(
    io_bundle=wf.RunLammpsCalculation.outputs.io_bundle
)

wf.OutputCalcMinimize = OutputCalcMinimize(input=wf.ParseLammpsOutput)

wf.OutputCalcStaticInitial = OutputCalcStatic(
    input=wf.OutputCalcMinimize.outputs.initial
)

wf.OutputCalcStaticFinal = OutputCalcStatic(input=wf.OutputCalcMinimize.outputs.final)
