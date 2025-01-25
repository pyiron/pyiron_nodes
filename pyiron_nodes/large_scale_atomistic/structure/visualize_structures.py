from pyiron_workflow import as_function_node
import ase as _ase
import numpy as _numpy
import ipywidgets as _ipywidgets
from typing import Optional


@as_function_node("plot")
def Ase2OvitoViz(ase_atoms: _ase.Atoms):
    """Visualize ase.Atoms with ovito widget"""
    from ovito.pipeline import StaticSource, Pipeline
    from ovito.io.ase import ase_to_ovito

    data = ase_to_ovito(ase_atoms)
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()
    from ovito.vis import Viewport, CoordinateTripodOverlay
    from ipywidgets import Layout

    vp = Viewport()
    tripod = CoordinateTripodOverlay(size=0.07)
    vp.overlays.append(tripod)
    return vp.create_jupyter_widget(layout=Layout(width="100%"))


@as_function_node("image")
def Ase2OvitoTachyonRenderPersp(
    structure: _ase.Atoms,
    particle_size: Optional[float | int] = 0.8,
    camera_dir_x: Optional[float | int] = -1.0,
    camera_dir_y: Optional[float | int] = -1.0,
    camera_dir_z: Optional[float | int] = -1.0,
):
    from ovito.pipeline import StaticSource, Pipeline
    from ovito.io.ase import ase_to_ovito

    data = ase_to_ovito(structure)
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()

    from ovito.vis import ParticlesVis

    vis_element = pipeline.compute().particles.vis
    vis_element.scaling = particle_size

    from ovito.vis import Viewport, TachyonRenderer
    from ipywidgets import Layout

    vp = Viewport(
        type=Viewport.Type.Perspective,
        camera_dir=(camera_dir_x, camera_dir_x, camera_dir_x),
    )
    vp.zoom_all()
    import os

    os.remove("temp_viz.png")
    vp.render_image(
        filename="temp_viz.png",
        size=(800, 600),
        background=(1, 1, 1),
        renderer=TachyonRenderer(ambient_occlusion=False, shadows=False),
    )
    pipeline.remove_from_scene()
    from IPython.display import Image, display

    return display(Image(filename="temp_viz.png", width=300))
