from core import Workflow
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.controls import IterToDataFrame
from pyiron_nodes.math_utils import Linspace

wf = Workflow("iter_demo")

wf.Bulk = Bulk(name="Al", cubic=True)

wf.Linspace = Linspace()

wf.IterToDataFrame = IterToDataFrame(node=wf.Bulk, input_label="a", values=wf.Linspace)
wf.IterToDataFrame.inputs.add(
    "debug", port_type=bool, default=False, value=False, has_explicit_default=True
)
wf.IterToDataFrame.inputs.add(
    "executor", port_type=object, default=None, value=None, has_explicit_default=True
)
wf.IterToDataFrame.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)
