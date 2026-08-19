from core import Workflow
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import Repeat
from pyiron_nodes.dpg2026.atomistic.calculator.calphy import (
    InputClass,
    LiquidFreeEnergyWithTemp,
    SolidFreeEnergyWithTemp,
)
from pyiron_nodes.dpg2026.atomistic.engine.lammps import ListPotentials
from pyiron_nodes.dpg2026.atomistic.structure.transform import Rattle
from pyiron_nodes.plotting import InputPlotOptions, MergePlots, Plot

wf = Workflow("calph_melt")

wf.Bulk = Bulk(name="Al", cubic=True)

wf.InputClass = InputClass(temperature_stop=400)

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

wf.Rattle = Rattle(structure=wf.Repeat, stdev=0.5)

wf.Plot_solid = Plot(
    y=wf.SolidFreeEnergyWithTemp.outputs.free_energy,
    x=wf.SolidFreeEnergyWithTemp.outputs.temperature,
    options=wf.InputPlotOptions,
)

wf.LiquidFreeEnergyWithTemp = LiquidFreeEnergyWithTemp(
    inp=wf.InputClass,
    structure=wf.Rattle,
    potential="1995--Angelo-J-E--Ni-Al-H--LAMMPS--ipr1",
)
wf.LiquidFreeEnergyWithTemp.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)

wf.Plot_liquid = Plot(
    y=wf.LiquidFreeEnergyWithTemp.outputs.free_energy,
    x=wf.LiquidFreeEnergyWithTemp.outputs.temperature,
)

wf.MergePlots = MergePlots(fig1=wf.Plot_liquid, fig2=wf.Plot_solid)
