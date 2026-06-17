from pyiron_nodes.atomistic.engine.ase import GRACE
from pyiron_nodes.dataframe import GetColumnFromDataFrame
from pyiron_nodes.dpg2026.atomistic.assyst.stoichiometry import ElementInput, SpaceGroupSampling
from pyiron_nodes.dpg2026.atomistic.calculator.optimize import GenericOptimizerSettings, MapCalculatorOnStructures, Relax, Static
from pyiron_nodes.dpg2026.basic.math import Multiply
from pyiron_nodes.plotting import Histogram, InputPlotOptions, LinearFittingCurve, MergePlots, Scatter
from core import Workflow
from core import group_node

wf = Workflow("assyst")

wf.ElementInput = ElementInput(element='Al', min_ion=2, max_ion=4)

wf.ElementInput_1 = ElementInput(element='Ca', min_ion=2, max_ion=4)

wf.GRACE = GRACE(model='GRACE-1L-MP-r6')

wf.GenericOptimizerSettings = GenericOptimizerSettings()

wf.GenericOptimizerSettings_1 = GenericOptimizerSettings(max_steps=1000)

wf.InputPlotOptions = InputPlotOptions(legend_label='static')

wf.InputPlotOptions_1 = InputPlotOptions(legend_label='fully relaxed')

wf.Multiply = Multiply(x=wf.ElementInput, y=wf.ElementInput_1)

wf.Static = Static(structure='NotData', engine=wf.GRACE)
wf.Static.inputs.add("store", port_type=bool, default=False, value=False, has_explicit_default=True)

wf.Relax = Relax(structure='NotData', engine=wf.GRACE, opt_parameters=wf.GenericOptimizerSettings)
wf.Relax.inputs.add("store", port_type=bool, default=False, value=False, has_explicit_default=True)

wf.RelaxFull = Relax(structure='NotData', engine=wf.GRACE, opt_parameters=wf.GenericOptimizerSettings_1, opt_mode='full')
wf.RelaxFull.inputs.add("store", port_type=bool, default=False, value=False, has_explicit_default=True)

wf.SpaceGroupSampling = SpaceGroupSampling(elements=wf.Multiply, max_atoms=2, max_structures=5)
wf.SpaceGroupSampling.inputs.add("store", port_type=bool, default=False, value=True, has_explicit_default=True)

wf.GetStaticEnergies = MapCalculatorOnStructures(structures=wf.SpaceGroupSampling, calculator=wf.Static, store_structures=True)
wf.GetStaticEnergies.inputs.add("store", port_type=bool, default=False, value=True, has_explicit_default=True)

wf.GetRelaxedConstVolEnergies = MapCalculatorOnStructures(structures=wf.SpaceGroupSampling, calculator=wf.Relax, store_structures=True)
wf.GetRelaxedConstVolEnergies.inputs.add("store", port_type=bool, default=False, value=True, has_explicit_default=True)

wf.GetRelaxedEnergies = MapCalculatorOnStructures(structures=wf.SpaceGroupSampling, calculator=wf.RelaxFull, store_structures=True)
wf.GetRelaxedEnergies.inputs.add("store", port_type=bool, default=False, value=True, has_explicit_default=True)

wf.GetColumnFromDataFrame = GetColumnFromDataFrame(df=wf.GetStaticEnergies, column_name='energy', as_array=True)

wf.GetColumnFromDataFrame_1 = GetColumnFromDataFrame(df=wf.GetRelaxedEnergies, column_name='energy')

wf.HistogramStatic = Histogram(x=wf.GetColumnFromDataFrame, options=wf.InputPlotOptions)

wf.HistogramFull = Histogram(x=wf.GetColumnFromDataFrame_1, options=wf.InputPlotOptions_1)

wf.Scatter = Scatter(x=wf.GetColumnFromDataFrame, y=wf.GetColumnFromDataFrame_1)

wf.LinearFittingCurve = LinearFittingCurve(x=wf.GetColumnFromDataFrame, y=wf.GetColumnFromDataFrame_1)

wf.MergePlots = MergePlots(fig1=wf.HistogramStatic, fig2=wf.HistogramFull)
