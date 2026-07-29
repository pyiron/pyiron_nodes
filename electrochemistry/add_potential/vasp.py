from dataclasses import dataclass, field
import os
from typing import Optional
from ase import Atoms
import numpy as np
import re

from core import as_function_node

from pyiron_nodes.atomistic.engine.vasp_new import (
    VaspInputResources,
    _build_incar,
    _ordered_elements,
    _get_potcar_paths,
)


@dataclass
class CEParameters:
    path_to_plugin: str  # Default path to plugin template
    phi0: float  # Target voltage (V)
    Q0: float  # Adjusted charge for plugin (C), includes precision-matched ZVAL adjustment
    nelect_neutral: int  # Number of electrons of the neutral system
    grid_position_frac: float  # Where dipole correction is applied
    grid_roll_frac: float  # Grid roll fraction (try 0.1 or 0.2)
    tau: float  # Thermostat time constant (units of MD time step)
    temperature: float  # Temperature (K)
    ax: float  # Cell dimension a (Å), filled in by node
    ay: float  # Cell dimension b (Å), filled in by node
    az: float  # Cell dimension c (Å), filled in by node
    d_electrode: float  # Electrode distance (Å), filled in by node
    # CCE-specific parameters
    i_Ne: Optional[int] = None  # Index of Ne in POTCAR, filled in by node
    n_elements: Optional[int] = None  # Number of unique elements, filled in by node
    n_Ne: Optional[int] = None  # Number of Ne atoms, filled in by node
    # CDCE-specific parameters
    Q_pos: Optional[np.array] = None  # Position of the Gaussian charge (Å)
    width_wall: Optional[float] = None  # Width of the wall potential (Å)
    pos_right_wall: Optional[float] = None  # Position of the right wall (


def _modify_potcar(working_directory: str, original_line: str, zval_ne: float) -> None:

    potcar_path = os.path.join(working_directory, "POTCAR")

    if not os.path.exists(potcar_path):
        raise FileNotFoundError(
            f"POTCAR not found in {working_directory}. "
            "Make sure CreateVaspInputResources has been called first."
        )

    new_line = re.sub(
        r"(ZVAL\s*=\s*)[\d.]+", f"ZVAL   =    {zval_ne:.7f}", original_line
    )

    with open(potcar_path, "r") as f:
        lines = f.readlines()

    if original_line not in lines:
        raise ValueError(
            f"Could not find Ne ZVAL line in POTCAR.\n"
            f"Expected : {original_line.strip()}\n"
            f"Make sure the structure contains Ne atoms and "
            f"the correct POTCAR is being used."
        )

    lines[lines.index(original_line)] = new_line

    with open(potcar_path, "w") as f:
        f.writelines(lines)


def _write_plugin_file(working_directory: str, ce_params: CEParameters) -> None:
    """
    Fill the vasp_plugin.py template with CCE parameters and write
    it to the working directory.

    Must be called after CreateVaspInputResources has created the working directory.

    Parameters
    ----------
    path_to_plugin : str
        Path to the plugin template file.
    working_directory : str
        Path to the VASP job working directory.
    ce_params : CEParameters
        CE parameter bundle containing all values to fill into the template.
    """
    # -------------------------------------------------------------------------
    # Read plugin template file
    # -------------------------------------------------------------------------
    plugin_path = os.path.expanduser(ce_params.path_to_plugin)
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(f"Plugin template not found: {plugin_path}")
    with open(plugin_path, "r") as f:
        plugin_string = f.read()

    plugin_params = {
        "phi0": ce_params.phi0,
        "Q0": ce_params.Q0,
        "nelect_neutral": ce_params.nelect_neutral,
        "grid_roll_frac": ce_params.grid_roll_frac,
        "grid_position_frac": ce_params.grid_position_frac,
        "tau": ce_params.tau,
        "temperature": ce_params.temperature,
        "ax": ce_params.ax,
        "ay": ce_params.ay,
        "az": ce_params.az,
        "d_electrode": ce_params.d_electrode,
        "i_Ne": ce_params.i_Ne,
        "n_elements": ce_params.n_elements,
        "n_Ne": ce_params.n_Ne,
        "Q_pos": ce_params.Q_pos,
        "width_wall": ce_params.width_wall,
        "pos_right_wall": ce_params.pos_right_wall,
    }

    try:
        plugin_content = plugin_string.format(**plugin_params)
    except KeyError as e:
        raise KeyError(
            f"Missing parameter in plugin template: {e}. "
            f"Available parameters: {list(plugin_params.keys())}"
        )

    plugin_out_path = os.path.join(working_directory, "vasp_plugin.py")
    with open(plugin_out_path, "w") as f:
        f.write(plugin_content)


@as_function_node
def CCESetup(
    io_bundle: VaspInputResources,
    electrode: Atoms,
    potential: float = 0.0,
    path_to_plugin: str = "pyiron_nodes/electrochemistry/add_potential/vasp_plugin-CCE.plugin",
    Q0: float = 0.0,
    dipole_position: float = 0.85,
    grid_roll_frac: float = 0.1,
    tau: float = 50.0,
):
    """
    Build INCAR dictionary and plugin parameters for an electrochemistry
    VASP calculation using the Ne-CCE thermopotentiostat plugin.

    Computes all structure-dependent quantities from the ASE structure, then
    writes the electrochemistry INCAR tags and the plugin file into the working
    directory created by CreateVaspInputResources.

    """

    if io_bundle.calc.md is None:
        raise ValueError(
            "CCE plugin is designed for MD simulations, but no MD settings were "
            "found on the calc. Set an InputMDVASP on the `md` port of "
            "MergeVaspInput before running CCESetup."
        )

    # -------------------------------------------------------------------------
    # Calculate nelect_neutral from individual POTCAR files
    # Uses existing _get_potcar_paths() helper — reads before concatenation
    # -------------------------------------------------------------------------
    temperature = io_bundle.calc.md.temperature

    potcar_paths = _get_potcar_paths(
        io_bundle.structure,
        io_bundle.calc.scf.functional,
        io_bundle.potcar_lib_path,
    )

    # Read ZVAL for each element from its individual POTCAR file
    # POTCAR line format: "   POMASS =  196.970; ZVAL   =   11.000    mass and valenz"
    zval_per_element = {}
    unique_elements = _ordered_elements(io_bundle.structure)

    for el, potcar_path in zip(unique_elements, potcar_paths):
        with open(potcar_path, "r") as f:
            content = f.read()
        # Take first match — each individual POTCAR has exactly one ZVAL
        match = re.search(r"ZVAL\s*=\s*([\d.]+)", content)
        if match is None:
            raise ValueError(
                f"Could not find ZVAL in POTCAR for element {el}: {potcar_path}"
            )
        zval_per_element[el] = float(match.group(1))

        if el == "Ne":
            line_match = re.search(
                r".*POMASS.*ZVAL.*8\.000.*mass and valenz.*\n", content
            )
            ne_zval_original_line = line_match.group(0)

    # nelect_neutral = sum of ZVAL over all atoms
    nelect_neutral = int(
        sum(zval_per_element[sym] for sym in io_bundle.structure.get_chemical_symbols())
    )

    # -------------------------------------------------------------------------
    # Structure-derived quantities
    # -------------------------------------------------------------------------

    unique_elements = _ordered_elements(io_bundle.structure)
    n_elements = len(unique_elements)

    # Cell dimensions - must be orthogonal
    cell = io_bundle.structure.get_cell()
    if (np.abs(cell[0] @ cell[2]) + np.abs(cell[1] @ cell[2])) > 1e-6:
        raise ValueError(
            "Cell must be orthogonal (a3 perpendicular to a1 and a2) "
            "for the electrochemistry plugin."
        )
    ax, ay, az = np.diag(cell)

    # Ne CCE atoms
    ne_indices = [
        i
        for i, sym in enumerate(io_bundle.structure.get_chemical_symbols())
        if sym == "Ne"
    ]
    if len(ne_indices) == 0:
        raise ValueError(
            "No Ne atoms found in structure. Ne atoms are required for CCE."
        )

    n_Ne = len(ne_indices)
    i_Ne = unique_elements.index("Ne")

    # Get electrode elements directly from the electrode structure
    electrode_elements = list(set(electrode.get_chemical_symbols()))

    # Get all electrode atom indices in the full structure
    electrode_indices = np.concatenate(
        [
            [
                i
                for i, sym in enumerate(io_bundle.structure.get_chemical_symbols())
                if sym in electrode_elements
            ]
        ]
    )

    d_electrode = float(
        np.max(io_bundle.structure[ne_indices].positions[:, 2])
    ) - float(np.max(io_bundle.structure[electrode_indices].positions[:, 2]))

    # Dipole correction center
    dipole_center = [
        0.0,
        0.0,
        float(np.max(io_bundle.structure.get_scaled_positions()[:, 2]) / 2.0),
    ]

    #  This is to make sure NELECT and the Ne ZVAL cancel out exactly
    zval_ne = float(f"{(8 + np.round(Q0 / n_Ne, 8)):.7f}")
    nelect_adjusted = float(nelect_neutral) + float(n_Ne) * (
        np.float64(f"{zval_ne:.7f}") - 8
    )

    # -------------------------------------------------------------------------
    # INCAR dictionary
    # -------------------------------------------------------------------------
    additional_incar = {
        "PLUGINS/LOCAL_POTENTIAL": "T",
        "PLUGINS/OCCUPANCIES": "T",
        "NELECT": nelect_adjusted,
        "IDIPOL": 3,
        "DIPOL": " ".join([str(c) for c in dipole_center]),
        "LDIPOL": ".TRUE.",
    }

    cce_params = CEParameters(
        path_to_plugin=path_to_plugin,
        phi0=potential,
        Q0=Q0,
        nelect_neutral=nelect_neutral,
        grid_position_frac=dipole_position,
        grid_roll_frac=grid_roll_frac,
        ax=ax,
        ay=ay,
        az=az,
        d_electrode=d_electrode,
        i_Ne=i_Ne,
        n_elements=n_elements,
        n_Ne=n_Ne,
        tau=tau,
        temperature=temperature,
    )

    if io_bundle.extra_incar is not None:
        io_bundle.extra_incar.update(additional_incar)
    else:
        io_bundle.extra_incar = additional_incar

    incar = _build_incar(io_bundle.calc, io_bundle.extra_incar, io_bundle.structure)
    incar.write_file(os.path.join(io_bundle.working_directory, "INCAR"))

    _modify_potcar(io_bundle.working_directory, ne_zval_original_line, zval_ne)

    _write_plugin_file(io_bundle.working_directory, cce_params)

    output_files = ["el_pot_z.dat", "Q.dat", "phi.dat"]

    # Remove existing output files to avoid confusion with previous runs
    existing_files = [
        filename
        for filename in output_files
        if os.path.exists(os.path.join(io_bundle.working_directory, filename))
    ]

    if existing_files:
        raise FileExistsError(
            f"The following file(s) already exist in '{io_bundle.working_directory}': "
            f"{', '.join(existing_files)}. "
            f"Please choose a new/empty working directory instead of overwriting existing results."
        )

    return io_bundle


@as_function_node
def CDCESetup(
    io_bundle: VaspInputResources,
    electrode: Atoms,
    potential: float = 0.0,
    path_to_plugin: str = "pyiron_nodes/electrochemistry/add_potential/vasp_plugin-CDCE_MD.plugin",
    Q0: float = 0.0,
    dipole_position: float = 0.85,
    grid_roll_frac: float = 0.1,
    pos_right_wall: float = 0.75,  # for now this is in fractional coordinates
    width_wall: float = 6.5,
    tau: float = 50.0,
):
    """
    Build INCAR dictionary and plugin parameters for an electrochemistry
    VASP calculation using the Ne-CCE thermopotentiostat plugin.

    Computes all structure-dependent quantities from the ASE structure, then
    writes the electrochemistry INCAR tags and the plugin file into the working
    directory created by CreateVaspInputResources.

    """

    if io_bundle.calc.md is None:
        raise ValueError(
            "CDCE plugin is designed for MD simulations, but no MD settings were "
            "found on the calc. Set an InputMDVASP on the `md` port of "
            "MergeVaspInput before running CDCESetup."
        )

    temperature = io_bundle.calc.md.temperature

    # -------------------------------------------------------------------------
    # Calculate nelect_neutral from individual POTCAR files
    # Uses existing _get_potcar_paths() helper — reads before concatenation
    # -------------------------------------------------------------------------
    potcar_paths = _get_potcar_paths(
        io_bundle.structure,
        io_bundle.calc.scf.functional,
        io_bundle.potcar_lib_path,
    )

    # Read ZVAL for each element from its individual POTCAR file
    # POTCAR line format: "   POMASS =  196.970; ZVAL   =   11.000    mass and valenz"
    zval_per_element = {}
    unique_elements = _ordered_elements(io_bundle.structure)

    for el, potcar_path in zip(unique_elements, potcar_paths):
        with open(potcar_path, "r") as f:
            content = f.read()
        # Take first match — each individual POTCAR has exactly one ZVAL
        match = re.search(r"ZVAL\s*=\s*([\d.]+)", content)
        if match is None:
            raise ValueError(
                f"Could not find ZVAL in POTCAR for element {el}: {potcar_path}"
            )
        zval_per_element[el] = float(match.group(1))

    # nelect_neutral = sum of ZVAL over all atoms
    nelect_neutral = int(
        sum(zval_per_element[sym] for sym in io_bundle.structure.get_chemical_symbols())
    )

    # -------------------------------------------------------------------------
    # Structure-derived quantities
    # -------------------------------------------------------------------------

    # Cell dimensions - must be orthogonal
    cell = io_bundle.structure.get_cell()
    if (np.abs(cell[0] @ cell[2]) + np.abs(cell[1] @ cell[2])) > 1e-6:
        raise ValueError(
            "Cell must be orthogonal (a3 perpendicular to a1 and a2) "
            "for the electrochemistry plugin."
        )
    ax, ay, az = np.diag(cell)

    # Get electrode elements directly from the electrode structure
    electrode_elements = list(set(electrode.get_chemical_symbols()))

    # Get all electrode atom indices in the full structure
    electrode_indices = np.concatenate(
        [
            [
                i
                for i, sym in enumerate(io_bundle.structure.get_chemical_symbols())
                if sym in electrode_elements
            ]
        ]
    )

    Q_pos = np.array([ax * 0.5, ay * 0.5, az * pos_right_wall - 2.0])

    d_electrode = float(
        Q_pos[2] - float(np.max(io_bundle.structure[electrode_indices].positions[:, 2]))
    )

    # Conwering to string to ensure the format in vasp_plugin.py is correct
    Q_pos = f"np.array({Q_pos.tolist()})"

    nelect_adjusted = float(nelect_neutral) + np.round(Q0)

    # -------------------------------------------------------------------------
    # INCAR dictionary
    # -------------------------------------------------------------------------
    additional_incar = {
        "PLUGINS/LOCAL_POTENTIAL": "T",
        "PLUGINS/OCCUPANCIES": "T",
        "PLUGINS/FORCE_AND_STRESS": "T",
        "NELECT": nelect_adjusted,
        "LREMOVE_DRIFT": "F",
    }

    cdce_params = CEParameters(
        path_to_plugin=path_to_plugin,
        phi0=potential,
        Q0=Q0,
        Q_pos=Q_pos,
        nelect_neutral=nelect_neutral,
        grid_position_frac=dipole_position,
        grid_roll_frac=grid_roll_frac,
        width_wall=width_wall,
        pos_right_wall=pos_right_wall,
        ax=ax,
        ay=ay,
        az=az,
        d_electrode=d_electrode,
        tau=tau,
        temperature=temperature,
    )

    if io_bundle.extra_incar is not None:
        io_bundle.extra_incar.update(additional_incar)
    else:
        io_bundle.extra_incar = additional_incar

    incar = _build_incar(io_bundle.calc, io_bundle.extra_incar, io_bundle.structure)
    incar.write_file(os.path.join(io_bundle.working_directory, "INCAR"))

    _write_plugin_file(io_bundle.working_directory, cdce_params)

    output_files = ["el_pot_z.dat", "Q.dat", "phi.dat", "dipole_corr.dat"]

    # Remove existing output files to avoid confusion with previous runs
    for filename in output_files:
        filepath = os.path.join(io_bundle.working_directory, filename)
        if os.path.exists(filepath):
            os.remove(filepath)

    return io_bundle


@as_function_node
def ParsePotential(
    io_bundle: VaspInputResources,
):
    """
    Reads electrode charge (Q.dat), potential (phi.dat), and electrostatic
    potential (el_pot_z.dat) from VASP working directory and saves the data.

    Parameters
    ----------
    io_bundle : VaspInputResources
        Input dataclass containing VASP resources.

    """
    working_dir = io_bundle.working_directory

    # --- Check files exist ---
    files = {
        "Q.dat": os.path.join(working_dir, "Q.dat"),
        "phi.dat": os.path.join(working_dir, "phi.dat"),
        "el_pot_z.dat": os.path.join(working_dir, "el_pot_z.dat"),
    }
    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found in working directory: {working_dir}"
            )

    # --- Load data ---
    charge_list = np.loadtxt(files["Q.dat"])
    pot_list = np.loadtxt(files["phi.dat"])
    electrostatic_potential_z = np.loadtxt(files["el_pot_z.dat"])

    # --- Check for NSW ---
    nsw = None
    if io_bundle.extra_incar is not None:
        nsw = io_bundle.extra_incar.get("NSW") or io_bundle.extra_incar.get("nsw")
    if nsw is None and io_bundle.calc is not None:
        md = io_bundle.calc.md
        if md is not None:
            nsw = md.n_ionic_steps
    if nsw is None:
        raise ValueError(
            "Could not determine NSW: not present in extra_incar and no MD "
            "settings found on the calc."
        )

    # --- Reshape electrostatic potential ---
    nz = electrostatic_potential_z.shape[0] // nsw
    electrostatic_potential_z_2d = electrostatic_potential_z.reshape([nsw, nz])
    # cell_z       = io_bundle.vasp_resources.structure.cell[2][2]
    # z_coords     = np.linspace(0, cell_z, nz)

    return electrostatic_potential_z_2d, charge_list, pot_list
