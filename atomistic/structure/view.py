from __future__ import annotations
from typing import Literal, Optional

import numpy as np
from ase import Atoms as _Atoms

from core import as_function_node
#from pyiron_nodes.atomistic.engine.lammps import LammpsIOBundle
from pyiron_nodes.atomistic.structure._atoms import OutputAtoms, _data_to_ase
from pyiron_nodes.atomistic.calculator.data import OutputCalcMD


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
def AnimateAse(
    ase_trajectory: list,
    gui: bool = False,
    spacefill: bool = True,
    show_cell: bool = True,
    particle_size: float = 0.5,
    camera: str = "orthographic"
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


@as_function_node("fig")
def VisualizeMultipleStructures(
    ase_structure_list: list,
    columns: int = 3,
    figure_size: float = 4.0,
    rotation: str = "0x,0y,0z"
):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from ase.io import write
    import tempfile
    import os
    import math

    # Suppress any intermediate plots
    plt.ioff()  # turn off interactive mode

    columns = int(columns)
    figure_size = float(figure_size)

    n = len(ase_structure_list)
    rows = math.ceil(n / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(figure_size * columns, figure_size * rows))
    
    # Make axes always 2D array for consistent indexing
    if rows == 1 and columns == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif columns == 1:
        axes = [[ax] for ax in axes]

    for i, struct in enumerate(ase_structure_list):
        row, col = divmod(i, columns)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        
        try:
            write(tmp_path, struct, rotation=rotation)
            plt.close('all')  # close any figures ASE opened internally
            img = mpimg.imread(tmp_path)
            axes[row][col].imshow(img)
            axes[row][col].set_title(f"Structure {i}")
            axes[row][col].axis("off")
        finally:
            os.unlink(tmp_path)
    
    # Hide unused axes
    for i in range(n, rows * columns):
        row, col = divmod(i, columns)
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.ion()  # turn interactive mode back on
    return fig


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
