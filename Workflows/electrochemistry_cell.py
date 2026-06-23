from pyiron_nodes.atomistic.structure.build import Surface
from pyiron_nodes.atomistic.structure.view import Plot3d
from pyiron_nodes.electrochemistry.structure.build import add_neon_layer, add_water_film
from core import Workflow
from core import group_node

wf = Workflow("electrochemistry_cell")

wf.Surface = Surface(element="Al", size="3 4 4", vacuum=20, orthogonal=True)

wf.add_water_film = add_water_film(electrode=wf.Surface)

wf.add_neon_layer = add_neon_layer(structure=wf.add_water_film)

wf.Plot3d = Plot3d(structure=wf.add_neon_layer)
