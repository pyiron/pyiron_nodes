from pyiron_nodes.atomistic.calculator.data import InputCalcMD, OutputCalcMD
from pyiron_nodes.atomistic.engine.lammps import CreateLammpsMDInput, CreateLammpsStructure, ListPotentials, ParseLammpsMDOutput, RunLammpsCalculation
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import Repeat
from pyiron_nodes.controls import pick_element
from pyiron_nodes.plotting import Plot
from core import Workflow
from core import group_node

wf = Workflow("lammps_MD")

wf.Bulk = Bulk(name='Al', cubic=True)

wf.InputCalcMD = InputCalcMD(initial_temperature=600, langevin=True)

wf.ListPotentials = ListPotentials(structure=wf.Bulk, resource_path='/cmmc/ptmp/janj/mambaforge/envs/aiflow/share/iprpy')

wf.Repeat = Repeat(structure=wf.Bulk, repeat_scalar='4')

wf.pick_element = pick_element(lst=wf.ListPotentials, index=5)

wf.CreateLammpsStructure = CreateLammpsStructure(structure=wf.Repeat, potential=wf.pick_element, working_directory='./test_1')

wf.CreateLammpsMDInput = CreateLammpsMDInput(io_bundle=wf.CreateLammpsStructure, calc_dataclass=wf.InputCalcMD, read_restart_filename='', write_restart_filename='')

wf.RunLammpsCalculation = RunLammpsCalculation(io_bundle=wf.CreateLammpsMDInput, debug=False)

wf.ParseLammpsMDOutput = ParseLammpsMDOutput(io_bundle=wf.RunLammpsCalculation.outputs.io_bundle)

wf.OutputCalcMD = OutputCalcMD(input=wf.ParseLammpsMDOutput)

wf.Plot = Plot(y=wf.OutputCalcMD.outputs.temperatures)

wf.Plot_1 = Plot(y=wf.OutputCalcMD.outputs.energies_pot)
