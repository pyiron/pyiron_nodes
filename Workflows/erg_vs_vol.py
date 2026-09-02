from core import Workflow, group_node
from pyiron_nodes.controls import IterToDataFrame
from pyiron_nodes.math_utils import Linspace

# ── Group node factories ─────────────────────────────


@group_node("out")
def group_Bulk_GRACE_Static(name, a=None, cubic=False):
    from core import Workflow
    from pyiron_nodes.atomistic.calculator.ase import StaticEnergy
    from pyiron_nodes.atomistic.engine.ase import GRACE
    from pyiron_nodes.atomistic.structure.build import Bulk

    inner_wf = Workflow("group_Bulk_GRACE_Static")
    inner_wf.Bulk = Bulk(name=name, a=a, cubic=cubic)
    inner_wf.GRACE = GRACE()
    inner_wf.Static = StaticEnergy(structure=inner_wf.Bulk, engine=inner_wf.GRACE)
    return inner_wf.Static


wf = Workflow("erg_vs_vol")

wf.Linspace = Linspace(x_min=3, x_max=6, num_points=5)

wf.group_Bulk_GRACE_Static = group_Bulk_GRACE_Static(name="Al", a=2, cubic=True)

wf.IterToDataFrame = IterToDataFrame(
    node=wf.group_Bulk_GRACE_Static, input_label="a", values=wf.Linspace
)
wf.IterToDataFrame.inputs.add(
    "debug", port_type=bool, default=False, value=False, has_explicit_default=True
)
wf.IterToDataFrame.inputs.add(
    "executor", port_type=object, default=None, value=None, has_explicit_default=True
)
wf.IterToDataFrame.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)
