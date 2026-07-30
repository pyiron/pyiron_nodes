from core import Workflow, group_node
from pyiron_nodes.atomistic.engine.ase import GRACE
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.controls import iterate
from pyiron_nodes.dpg2026.atomistic.calculator.optimize import (
    GenericOptimizerSettings,
    MapCalculatorOnStructures,
    Relax,
)
from pyiron_nodes.math_utils import Linspace

wf = Workflow("optimize_loop")

wf.Bulk = Bulk(name="Al", cubic=True)

wf.GRACE = GRACE(model="GRACE-1L-MP-r6")

wf.GenericOptimizerSettings = GenericOptimizerSettings()

wf.Linspace_1 = Linspace(x_min=3, x_max=5, num_points=5)

wf.Relax = Relax(
    structure="NotData", engine=wf.GRACE, opt_parameters=wf.GenericOptimizerSettings
)
wf.Relax.inputs.add(
    "store", port_type=bool, default=False, value=False, has_explicit_default=True
)

wf.iterate = iterate(node=wf.Bulk, input_label="a", values=wf.Linspace_1)
wf.iterate.inputs.add(
    "debug", port_type=bool, default=False, value=False, has_explicit_default=True
)
wf.iterate.inputs.add(
    "executor", port_type=object, default=None, value=None, has_explicit_default=True
)

wf.MapCalculatorOnStructures = MapCalculatorOnStructures(
    structures=wf.iterate, calculator=wf.Relax
)
wf.MapCalculatorOnStructures.inputs.add(
    "store", port_type=bool, default=False, value=False, has_explicit_default=True
)
