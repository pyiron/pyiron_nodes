from pyiron_nodes.atomistic.engine.ase import GRACE
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.controls import iterate
from pyiron_nodes.dpg2026.atomistic.calculator.optimize import (
    GenericOptimizerSettings,
    MapCalculatorOnStructures,
    Relax,
)
from pyiron_nodes.math_utils import Linspace
from pyiron_nodes.plotting import PlotDataFrameXY
from core import Workflow
from core import group_node

wf = Workflow("optimize_loop")

wf.Bulk = Bulk(name="Al", cubic=True)

wf.GRACE = GRACE(model="GRACE-1L-MP-r6")

wf.GenericOptimizerSettings = GenericOptimizerSettings()

wf.LatticeConstants = Linspace(x_min=3, x_max=7, num_points=15)

wf.Relax = Relax(
    engine=wf.GRACE, opt_parameters=wf.GenericOptimizerSettings, store=False
)

wf.CreateStructures = iterate(
    node=wf.Bulk,
    input_label="a",
    values=wf.LatticeConstants,
    debug=False,
    executor=None,
)

wf.MapCalculatorOnStructures = MapCalculatorOnStructures(
    structures=wf.CreateStructures, calculator=wf.Relax, store=True
)

wf.PlotDataFrameXY = PlotDataFrameXY(df=wf.MapCalculatorOnStructures)
