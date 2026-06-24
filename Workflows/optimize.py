from pyiron_nodes.atomistic.engine.ase import GRACE
from pyiron_nodes.dpg2026.atomistic.calculator.optimize import (
    GenericOptimizerSettings,
    Relax,
)
from pyiron_nodes.atomistic.structure.build import Bulk
from core import Workflow
from core import group_node

wf = Workflow("optimize")

wf.Bulk = Bulk(name="Al", cubic=True)

wf.GRACE = GRACE(model="GRACE-1L-MP-r6")

wf.GenericOptimizerSettings = GenericOptimizerSettings()

wf.Relax = Relax(
    structure=wf.Bulk, engine=wf.GRACE, opt_parameters=wf.GenericOptimizerSettings
)
wf.Relax.inputs.add(
    "store", port_type=bool, default=False, value=False, has_explicit_default=True
)
