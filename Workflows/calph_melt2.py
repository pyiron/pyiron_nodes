from pyiron_nodes.atomistic.engine.lammps import GetPotential
from pyiron_nodes.atomistic.property.calphy import (
    CalphyDiagnosticsTable,
    InputClass,
    LiquidFreeEnergyWithTemp,
    PlotCalphyHysteresis,
    SolidFreeEnergyWithTemp,
)
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import Rattle, Repeat
from pyiron_nodes.plotting import InputPlotOptions, MergePlots, Plot
from core import Workflow
from core import group_node

wf = Workflow("calph_melt2")

wf.Bulk = Bulk(name="Al", cubic=True)

wf.InputClassLiquid = InputClass(temperature=700, temperature_stop=1200)

wf.InputClassSolid = InputClass(pressure=1, temperature=700, temperature_stop=1200)

wf.InputPlotOptions = InputPlotOptions(title="Free energy vs Temperature")

wf.Repeat = Repeat(structure=wf.Bulk, repeat_scalar=5)

wf.GetPotential = GetPotential(structure=wf.Bulk)

wf.Rattle = Rattle(structure=wf.Repeat, stdev=0.5)

wf.SolidFreeEnergyWithTemp = SolidFreeEnergyWithTemp(
    inp=wf.InputClassSolid,
    structure=wf.Repeat,
    potential=wf.GetPotential.outputs.potential_name,
    store=True,
)

wf.LiquidFreeEnergyWithTemp = LiquidFreeEnergyWithTemp(
    inp=wf.InputClassLiquid,
    structure=wf.Rattle,
    potential=wf.GetPotential.outputs.potential_name,
    store=True,
)

wf.Plot_solid = Plot(
    y=wf.SolidFreeEnergyWithTemp.outputs.free_energy,
    x=wf.SolidFreeEnergyWithTemp.outputs.temperature,
    options=wf.InputPlotOptions,
)

wf.CalphyDiagnosticsTable_1 = CalphyDiagnosticsTable(
    diagnostics=wf.SolidFreeEnergyWithTemp.outputs.diagnostics
)

wf.PlotCalphyHysteresis_1 = PlotCalphyHysteresis(
    diagnostics=wf.SolidFreeEnergyWithTemp.outputs.diagnostics
)

wf.Plot_liquid = Plot(
    y=wf.LiquidFreeEnergyWithTemp.outputs.free_energy,
    x=wf.LiquidFreeEnergyWithTemp.outputs.temperature,
)

wf.PlotCalphyHysteresis = PlotCalphyHysteresis(
    diagnostics=wf.LiquidFreeEnergyWithTemp.outputs.diagnostics
)

wf.CalphyDiagnosticsTable = CalphyDiagnosticsTable(
    diagnostics=wf.LiquidFreeEnergyWithTemp.outputs.diagnostics
)

wf.MergePlots = MergePlots(fig1=wf.Plot_liquid, fig2=wf.Plot_solid)
