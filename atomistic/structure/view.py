from __future__ import annotations
from typing import Literal, Optional

import numpy as np
from ase import Atoms as _Atoms

from core import as_function_node
from pyiron_nodes.atomistic.structure._atoms import OutputAtoms, _data_to_ase


@as_function_node("plot")
def Plot3d(
    structure: _Atoms | OutputAtoms,
    camera: str = "orthographic",
    particle_size: float = 1.0,
    background: Literal["white", "black"] = "white",
    select_atoms: Optional[np.ndarray | list] = None,
    view_plane: Optional[list] = None,
    distance_from_camera: Optional[float] = 1.0,
):
    """
    Display atomistic structure (ase.Atoms or OutputAtoms) using nglview.

    Task
    ----
    Visualise a static atomic structure, e.g., after building a bulk cell,
    creating a surface slab, or after a geometry optimisation. This node is
    typically used when the user wants to inspect the geometry, defects, or
    surface features directly in a Jupyter notebook.

    Parameters
    ----------
    structure: ase.Atoms or OutputAtoms
        The atomic structure to visualise. Can be either an ASE Atoms object
        or an OutputAtoms dataclass instance.
    camera: str, optional
        Camera mode, either "orthographic" or "perspective".
    particle_size: float, optional
        Size of the rendered atoms.
    background: {"white", "black"}, optional
        Background colour of the view.
    select_atoms: np.ndarray or list, optional
        Indices of atoms to highlight.
    view_plane: list, optional
        Plane normal for the view.
    distance_from_camera: float, optional
        Distance of the camera from the structure.
    """

    if view_plane is None:
        view_plane = [1, 1, 1]

    # Convert OutputAtoms to ASE Atoms if necessary
    if isinstance(structure, OutputAtoms):
        structure = _data_to_ase(structure)

    from structuretoolkit.visualize import plot3d

    return plot3d(
        structure,
        camera=camera,
        particle_size=particle_size,
        background=background,
        select_atoms=select_atoms,
        view_plane=view_plane,
        distance_from_camera=distance_from_camera,
    )


@as_function_node("view")
def Animate(
    trajectory,
    gui: bool = False,
    spacefill: bool = True,
    show_cell: bool = True,
    particle_size: float = 0.5,
    camera: str = "orthographic",
):
    """
    Animate a list of ASE Atoms frames using nglview.

    Parameters
    ----------
    ase_trajectory : list of ase.Atoms
    Frames to animate, as returned by ParseLammpsOutput.
    gui : bool, default=False
    Whether to show the nglview GUI controls panel.
    """
    import nglview
    from ase import Atoms

    ase_trajectory = []

    all_symbols = trajectory.species
    for frame_idx in range(len(trajectory.positions)):

        if hasattr(trajectory, "unwrapped_positions"):
            positions = trajectory.unwrapped_positions
        else:
            positions = trajectory.positions

        cell = trajectory.cells[frame_idx] if trajectory.cells is not None else None
        frame = Atoms(
            symbols=all_symbols,
            positions=positions[frame_idx],
            cell=cell,
            pbc=cell is not None,
        )
        ase_trajectory.append(frame)

    animation = nglview.show_asetraj(ase_trajectory, gui=gui)

    if spacefill:
        animation.add_spacefill(radius_type="vdw", scale=0.5, radius=particle_size)
        animation.remove_ball_and_stick()
    else:
        animation.add_ball_and_stick()
    if show_cell:
        animation.add_unitcell()
    animation.camera = camera

    return animation
