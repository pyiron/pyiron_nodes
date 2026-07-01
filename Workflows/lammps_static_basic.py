from pyiron_nodes.atomistic.calculator.data import OutputCalcStatic
from pyiron_nodes.atomistic.engine.lammps import (
    CreateLammpsStaticInput,
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

wf = Workflow("lammps_static_basic")

wf.Bulk = Bulk(name="Al", cubic=True)

wf.Repeat = Repeat(structure=wf.Bulk, repeat_scalar=3)

wf.ListPotentials = ListPotentials(structure=wf.Repeat)

wf.pick_element = pick_element(lst=wf.ListPotentials, index=0)

wf.CreateLammpsStructure = CreateLammpsStructure(
    structure=wf.Repeat, potential=wf.pick_element, working_directory="./lammps_static_basic"
)

wf.CreateLammpsStaticInput = CreateLammpsStaticInput(io_bundle=wf.CreateLammpsStructure)

wf.RunLammpsCalculation = RunLammpsCalculation(
    io_bundle=wf.CreateLammpsStaticInput, debug=False
)

wf.ParseLammpsOutput = ParseLammpsOutput(
    io_bundle=wf.RunLammpsCalculation.outputs.io_bundle
)

wf.OutputCalcStatic = OutputCalcStatic(input=wf.ParseLammpsOutput)
