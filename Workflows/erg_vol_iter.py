from core import Workflow, group_node
from pyiron_nodes.controls import IterToDataFrame
from pyiron_nodes.math_utils import Linspace

# ── Group node factories ─────────────────────────────


@group_node("out")
def Bulk_GRACE_Static_Energy(name, a=None, cubic=False):
    from core import Workflow
    from pyiron_nodes.atomistic.calculator.ase import Static
    from pyiron_nodes.atomistic.engine.ase import GRACE
    from pyiron_nodes.atomistic.structure.build import Bulk

    inner_wf = Workflow("Bulk_GRACE_Static_Energy")
    inner_wf.Bulk = Bulk(name=name, a=a, cubic=cubic)
    inner_wf.GRACE = GRACE()
    inner_wf.Static = Static(structure=inner_wf.Bulk, engine=inner_wf.GRACE)
    return inner_wf.Static


wf = Workflow("erg_vol_iter")

wf.Bulk_GRACE_Static_Energy = Bulk_GRACE_Static_Energy(name="Al", a=2, cubic=True)

wf.Linspace_1 = Linspace(x_min=3, x_max=5, num_points=5)

wf.IterToDataFrame = IterToDataFrame(
    node=wf.Bulk_GRACE_Static_Energy, input_label="a", values=wf.Linspace_1
)
wf.IterToDataFrame.inputs.add(
    "debug", port_type=bool, default=False, value=True, has_explicit_default=True
)
wf.IterToDataFrame.inputs.add(
    "executor", port_type=object, default=None, value=None, has_explicit_default=True
)
wf.IterToDataFrame.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)
