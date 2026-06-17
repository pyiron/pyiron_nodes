from pyiron_nodes.atomistic.calculator.data import InputCalcMD, OutputCalcMD
from pyiron_nodes.atomistic.engine.lammps import CreateLammpsMDInput, CreateLammpsStructure, ListPotentials, ParseLammpsOutput, RunLammpsCalculation
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import FixSpecies, Repeat
from pyiron_nodes.atomistic.structure.view import Animate
from pyiron_nodes.controls import pick_element
from core import Workflow
from core import group_node

wf = Workflow("lammps_md")

wf.Bulk = Bulk(name='Al', cubic=True)

wf.InputCalcMD = InputCalcMD()

wf.Repeat = Repeat(structure=wf.Bulk, repeat_scalar=2)

wf.ListPotentials = ListPotentials(structure=wf.Repeat)

wf.FixSpecies = FixSpecies(structure=wf.Repeat)

wf.pick_element = pick_element(lst=wf.ListPotentials, index=0)

wf.CreateLammpsStructure = CreateLammpsStructure(structure=wf.FixSpecies, potential=wf.pick_element, working_directory='./test')

wf.CreateLammpsMDInput = CreateLammpsMDInput(io_bundle=wf.CreateLammpsStructure, calc_dataclass=wf.InputCalcMD)

wf.RunLammpsCalculation = RunLammpsCalculation(io_bundle=wf.CreateLammpsMDInput)
wf.RunLammpsCalculation.inputs.add("debug", port_type=bool, default=False, value=False, has_explicit_default=True)

wf.ParseLammpsOutput = ParseLammpsOutput(io_bundle=wf.RunLammpsCalculation.outputs.io_bundle)

wf.OutputCalcMD = OutputCalcMD(input=wf.ParseLammpsOutput)

wf.Animate = Animate(trajectory=wf.ParseLammpsOutput)
