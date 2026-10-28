from core import Workflow
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import Repeat
from pyiron_nodes.atomistic.property.calphy import (
    CalphyDiagnosticsTable,
    FindMeltingTemperature,
    InputClass,
    LiquidFreeEnergyWithTemp,
    PlotCalphyHysteresis,
    PlotSolidLiquidFreeEnergy,
    SolidFreeEnergyWithTemp,
)
from pyiron_nodes.atomistic.engine.lammps import ListPotentials
from pyiron_nodes.atomistic.structure.transform import Rattle

wf = Workflow("calph_melt")

wf.Bulk = Bulk(name="Al", cubic=True)

# The window has to bracket the expected melting point -- Mishin Al melts near
# 930 K, so sweep 700-1200 K.  A window far below T_m yields a "liquid" that is
# not a physical phase and a meaningless crossing.
wf.InputClass = InputClass(temperature=700, temperature_stop=1200)

wf.Repeat = Repeat(structure=wf.Bulk, repeat_scalar=5)

wf.ListPotentials = ListPotentials(structure=wf.Bulk)

wf.SolidFreeEnergyWithTemp = SolidFreeEnergyWithTemp(
    inp=wf.InputClass,
    structure=wf.Repeat,
    potential="1999--Mishin-Y--Al--LAMMPS--ipr1",
)
wf.SolidFreeEnergyWithTemp.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)

wf.Rattle = Rattle(structure=wf.Repeat, stdev=0.5)

wf.LiquidFreeEnergyWithTemp = LiquidFreeEnergyWithTemp(
    inp=wf.InputClass,
    structure=wf.Rattle,
    potential="1999--Mishin-Y--Al--LAMMPS--ipr1",
)
wf.LiquidFreeEnergyWithTemp.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)

wf.FindMeltingTemperature = FindMeltingTemperature(
    temp_solid=wf.SolidFreeEnergyWithTemp.outputs.temperature,
    fe_solid=wf.SolidFreeEnergyWithTemp.outputs.free_energy,
    temp_liquid=wf.LiquidFreeEnergyWithTemp.outputs.temperature,
    fe_liquid=wf.LiquidFreeEnergyWithTemp.outputs.free_energy,
)

wf.PlotSolidLiquidFreeEnergy = PlotSolidLiquidFreeEnergy(
    temp_solid=wf.SolidFreeEnergyWithTemp.outputs.temperature,
    fe_solid=wf.SolidFreeEnergyWithTemp.outputs.free_energy,
    temp_liquid=wf.LiquidFreeEnergyWithTemp.outputs.temperature,
    fe_liquid=wf.LiquidFreeEnergyWithTemp.outputs.free_energy,
    T_melt=wf.FindMeltingTemperature,
)

# Run quality: the hysteresis plots should show forward and backward curves on
# top of each other, and rs_max_dissipation in the tables should stay below
# calphy's 1e-4 eV/atom threshold.
wf.SolidDiagnostics = CalphyDiagnosticsTable(
    diagnostics=wf.SolidFreeEnergyWithTemp.outputs.diagnostics
)
wf.LiquidDiagnostics = CalphyDiagnosticsTable(
    diagnostics=wf.LiquidFreeEnergyWithTemp.outputs.diagnostics
)
wf.SolidHysteresis = PlotCalphyHysteresis(
    diagnostics=wf.SolidFreeEnergyWithTemp.outputs.diagnostics
)
wf.LiquidHysteresis = PlotCalphyHysteresis(
    diagnostics=wf.LiquidFreeEnergyWithTemp.outputs.diagnostics
)
