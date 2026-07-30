from core import Workflow, group_node
from pyiron_nodes.atomistic.calculator.data import InputCalcMD, OutputCalcMD
from pyiron_nodes.atomistic.engine.lammps import (
    CreateLammpsMDInput,
    CreateLammpsStructure,
    ParseLammpsOutput,
    RunLammpsCalculation,
)
from pyiron_nodes.atomistic.structure.build import Surface
from pyiron_nodes.atomistic.structure.view import Animate
from pyiron_nodes.electrochemistry.structure.build import AddNeonLayer, AddWaterFilm
from pyiron_nodes.electrochemistry.structure.equilibrate import TIP3PSlabPotential
from pyiron_nodes.plotting import Plot

wf = Workflow("tip3p_slab_md")

wf.InputCalcMD = InputCalcMD(n_ionic_steps=1000)

wf.Surface = Surface(element="Al", size="3 4 4", vacuum=20, orthogonal=True)

wf.TIP3PSlabPotential = TIP3PSlabPotential()

wf.AddWaterFilm = AddWaterFilm(electrode=wf.Surface)

wf.AddNeonLayer = AddNeonLayer(structure=wf.AddWaterFilm)

wf.CreateLammpsStructure = CreateLammpsStructure(
    structure=wf.AddNeonLayer,
    potential=wf.TIP3PSlabPotential.outputs.slab_potential,
    working_directory="./tip3p_slab_md",
    bond_dict=wf.TIP3PSlabPotential.outputs.bond_dict,
)

wf.CreateLammpsMDInput = CreateLammpsMDInput(
    io_bundle=wf.CreateLammpsStructure, calc_dataclass=wf.InputCalcMD
)

wf.RunLammpsCalculation = RunLammpsCalculation(
    io_bundle=wf.CreateLammpsMDInput, debug=False, executor=None
)

wf.ParseLammpsOutput = ParseLammpsOutput(
    io_bundle=wf.RunLammpsCalculation.outputs.io_bundle
)

wf.OutputCalcMD = OutputCalcMD(input=wf.ParseLammpsOutput)

wf.Animate = Animate(trajectory=wf.ParseLammpsOutput)

wf.Plot = Plot(y=wf.OutputCalcMD.outputs.temperatures)
