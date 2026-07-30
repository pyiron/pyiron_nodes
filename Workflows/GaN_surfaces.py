from core import Workflow, group_node
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import Repeat
from pyiron_nodes.atomistic.structure.view import Plot3d

wf = Workflow("GaN_surfaces")

wf.Bulk = Bulk(
    name="GaN", crystalstructure="wurtzite", a=3.189, c=5.125, orthorhombic=True
)

wf.Repeat = Repeat(structure=wf.Bulk, repeat_scalar=2)

wf.Plot3d = Plot3d(structure=wf.Repeat, particle_size=2)
