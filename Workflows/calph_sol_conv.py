from pyiron_nodes.atomistic.engine.lammps import GetPotential
from pyiron_nodes.atomistic.property.calphy import InputClass, SolidFreeEnergyWithTemp
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import Repeat
from pyiron_nodes.plotting import InputPlotOptions, MergePlots, Plot
from core import Workflow
from core import group_node

wf = Workflow("calph_sol_conv")

wf.Bulk = Bulk(name="Al", cubic=True)

wf.InputClass = InputClass(
    temperature=10, n_equilibration_steps=5000, n_switching_steps=5000
)

wf.InputClass_1 = InputClass(
    temperature=200,
    temperature_stop=500,
    n_equilibration_steps=5000,
    n_switching_steps=5000,
)

wf.InputPlotOptions = InputPlotOptions(title="Free energy vs Temperature", color="g")

wf.Repeat = Repeat(structure=wf.Bulk, repeat_scalar=5)

wf.GetPotential = GetPotential(structure=wf.Bulk)

wf.SolidFreeEnergyWithTemp_1 = SolidFreeEnergyWithTemp(
    inp=wf.InputClass_1,
    structure=wf.Repeat,
    potential=wf.GetPotential.outputs.potential_name,
    store=True,
)

wf.SolidFreeEnergyWithTemp = SolidFreeEnergyWithTemp(
    inp=wf.InputClass,
    structure=wf.Repeat,
    potential=wf.GetPotential.outputs.potential_name,
    store=True,
)

wf.Plot_solid1 = Plot(
    y=wf.SolidFreeEnergyWithTemp_1.outputs.free_energy,
    x=wf.SolidFreeEnergyWithTemp_1.outputs.temperature,
)

wf.Plot_solid = Plot(
    y=wf.SolidFreeEnergyWithTemp.outputs.free_energy,
    x=wf.SolidFreeEnergyWithTemp.outputs.temperature,
    options=wf.InputPlotOptions,
)

wf.MergePlots = MergePlots(fig1=wf.Plot_solid1, fig2=wf.Plot_solid)
