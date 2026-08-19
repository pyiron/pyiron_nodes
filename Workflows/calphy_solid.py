from core import Workflow
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import Repeat
from pyiron_nodes.dpg2026.atomistic.calculator.calphy import (
    InputClass,
    SolidFreeEnergyWithTemp,
)
from pyiron_nodes.dpg2026.atomistic.engine.lammps import ListPotentials
from pyiron_nodes.plotting import InputPlotOptions, Plot

wf = Workflow("calphy_solid")

wf.Bulk = Bulk(name="Al", cubic=True)

wf.InputClass = InputClass(temperature_stop=500)

wf.InputPlotOptions = InputPlotOptions(title="Free energy vs Temperature")

wf.Repeat = Repeat(structure=wf.Bulk, repeat_scalar=5)

wf.ListPotentials = ListPotentials(structure=wf.Bulk)

wf.SolidFreeEnergyWithTemp = SolidFreeEnergyWithTemp(
    inp=wf.InputClass,
    structure=wf.Repeat,
    potential="1995--Angelo-J-E--Ni-Al-H--LAMMPS--ipr1",
)
wf.SolidFreeEnergyWithTemp.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)

wf.Plot = Plot(
    y=wf.SolidFreeEnergyWithTemp.outputs.free_energy,
    x=wf.SolidFreeEnergyWithTemp.outputs.temperature,
    options=wf.InputPlotOptions,
)
