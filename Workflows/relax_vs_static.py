from core import Workflow, group_node
from pyiron_nodes.atomistic.engine.ase import GRACE
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.controls import IterToDataFrame
from pyiron_nodes.dpg2026.atomistic.calculator.optimize import (
    GenericOptimizerSettings,
    MapCalculatorOnStructures,
    Relax,
    Static,
)
from pyiron_nodes.math_utils import Linspace
from pyiron_nodes.plotting import InputPlotOptions, MergePlots, PlotDataFrame

wf = Workflow("relax_vs_static")

wf.Bulk = Bulk(name="Al", cubic=True)

wf.ConvergenceParameters = GenericOptimizerSettings()

wf.GRACE = GRACE(model="GRACE-1L-MP-r6")

wf.InputPlotOptions = InputPlotOptions(symbol="v", legend_label="relax")

wf.InputPlotOptions_1 = InputPlotOptions(color="g", legend_label="static")

wf.Linspace = Linspace(x_min=3, x_max=5, num_points=5)

wf.Static = Static(engine=wf.GRACE)
wf.Static.inputs.add(
    "store", port_type=bool, default=False, value=False, has_explicit_default=True
)

wf.Relax = Relax(engine=wf.GRACE, opt_parameters=wf.ConvergenceParameters)
wf.Relax.inputs.add(
    "store", port_type=bool, default=False, value=False, has_explicit_default=True
)

wf.CreateStrainedStructures = IterToDataFrame(
    node=wf.Bulk, input_label="a", values=wf.Linspace
)
wf.CreateStrainedStructures.inputs.add(
    "debug", port_type=bool, default=False, value=False, has_explicit_default=True
)
wf.CreateStrainedStructures.inputs.add(
    "executor", port_type=object, default=None, value=None, has_explicit_default=True
)
wf.CreateStrainedStructures.inputs.add(
    "store", port_type=bool, default=False, value=False, has_explicit_default=True
)

wf.GetRelaxedEnergies = MapCalculatorOnStructures(
    structures=wf.CreateStrainedStructures, calculator=wf.Relax
)
wf.GetRelaxedEnergies.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)

wf.GetStaticEnergies = MapCalculatorOnStructures(
    structures=wf.CreateStrainedStructures, calculator=wf.Static
)
wf.GetStaticEnergies.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)

wf.PlotRelaxed = PlotDataFrame(
    df=wf.GetRelaxedEnergies, x="a", options=wf.InputPlotOptions
)

wf.PlotDataFrame = PlotDataFrame(df=wf.GetRelaxedEnergies, x="a")

wf.PlotStatic = PlotDataFrame(
    df=wf.GetStaticEnergies, x="a", options=wf.InputPlotOptions_1
)

wf.MergePlots = MergePlots(fig1=wf.PlotStatic, fig2=wf.PlotRelaxed)
