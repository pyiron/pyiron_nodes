from dataclasses import dataclass, field
import os
from typing import Optional
from ase import Atoms
import numpy as np
import re
import warnings

from core import as_function_node
from pathlib import Path

from pyiron_nodes.atomistic.engine.vasp_new import (
    VaspInputResources,
    VaspPlugin,
    VaspInput,
    _ordered_elements,
    _get_potcar_paths,
    read_potcar_config,
)

_potcar_config = read_potcar_config(Path.home() / ".pyiron_vasp_config")
_default_potcar_lib_path: str = _potcar_config.get("default_POTCAR_path", "")


@dataclass
class CEParameters:
    """Values substituted into a constant-potential VASP plugin template.

    One bundle carries the parameters for both the CCE and CDCE plugins. The
    fields above ``i_Ne`` are common to both; the ``CCE-specific`` and
    ``CDCE-specific`` blocks are only filled by the corresponding setup node and
    default to ``None`` otherwise. The ``ax``/``ay``/``az``, ``d_electrode`` and
    the species-count fields are derived from the structure inside ``CCESetup`` /
    ``CDCESetup`` rather than passed in by the user.
    """

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


def _plugin_content(ce_params: CEParameters):
    """
    Fill the vasp_plugin.py template with CCE parameters and write
    it to the working directory.

    Must be called after CreateVaspInputResources has created the working directory.

    Parameters
    ----------
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

    return plugin_content


@as_function_node
def CCESetup(
    structure: Atoms,
    electrode: Atoms,
    calc: VaspInput,
    potcar_lib_path: str = _default_potcar_lib_path,
    path_to_plugin: str = "pyiron_nodes/electrochemistry/add_potential/vasp_plugin-CCE.plugin",
    potential: float = 0.0,
    Q0: float = 0.0,
    grid_position_frac: float = 0.85,
    grid_roll_frac: float = 0.1,
    tau: float = 50.0,
):
    """Build the Ne-CCE thermopotentiostat plugin setup for a VASP MD calculation.

    The counter-charge electrode (CCE) method holds the electrode at a fixed
    potential during MD by adjusting the electron count against a Ne
    counter-charge. This computes every structure-dependent quantity from the
    ASE structure and the individual element POTCARs (cell dimensions,
    electrode–Ne distance, Ne count, neutral electron count), derives the
    electrochemistry INCAR tags and the adjusted Ne ``ZVAL`` needed to keep
    ``NELECT`` consistent, and renders the filled CCE plugin file content.
    The calc must carry MD settings, since ``calc.md.temperature`` is read directly.

    Parameters
    ----------
    structure
        The atomic structure (ASE ``Atoms``) for which to set up the CCE
        plugin. Must contain at least one Ne atom.
    electrode
        The electrode structure; its elements identify the electrode atoms in
        the full cell, used to measure the electrode–Ne distance.
    calc
        The VASP input settings (``VaspInput``); must carry MD settings
        (``calc.md.temperature``) and SCF settings (``calc.scf.functional``).
    potcar_lib_path
        Path to the POTCAR library used to look up ZVAL values for each
        element, including Ne.
    path_to_plugin
        Path to the CCE plugin template.
    potential
        Target electrode potential in volts (``phi0`` in the plugin).
    Q0
        Extra charge (electrons) added to the electrode, spread over the Ne
        atoms by shifting their ``ZVAL``.
    grid_position_frac
        Fractional z where the the potential is measured.
    grid_roll_frac
        Grid roll fraction passed to the plugin.
    tau
        Thermostat time constant, in units of the MD time step.

    Returns
    -------
    Atoms, VaspInput, VaspPlugin
        The unmodified ``structure`` and ``calc``, together with a
        ``VaspPlugin`` bundling the rendered plugin content, the Ne POTCAR
        ``ZVAL`` override, and the extra INCAR tags (``NELECT`` and the
        plugin activation flags) needed to run the CCE setup.

    """

    # -------------------------------------------------------------------------
    # Calculate nelect_neutral from individual POTCAR files
    # Uses existing _get_potcar_paths() helper — reads before concatenation
    # -------------------------------------------------------------------------

    temperature = calc.md.temperature
    functional = calc.scf.functional

    potcar_paths = _get_potcar_paths(structure, functional, potcar_lib_path)

    # Read ZVAL for each element from its individual POTCAR file
    # POTCAR line format: "   POMASS =  196.970; ZVAL   =   11.000    mass and valenz"
    zval_per_element = {}
    unique_elements = _ordered_elements(structure)

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
        sum(zval_per_element[sym] for sym in structure.get_chemical_symbols())
    )

    # -------------------------------------------------------------------------
    # Structure-derived quantities
    # -------------------------------------------------------------------------

    unique_elements = _ordered_elements(structure)
    n_elements = len(unique_elements)

    # Cell dimensions - must be orthogonal
    cell = structure.get_cell()
    if (np.abs(cell[0] @ cell[2]) + np.abs(cell[1] @ cell[2])) > 1e-6:
        raise ValueError(
            "Cell must be orthogonal (a3 perpendicular to a1 and a2) "
            "for the electrochemistry plugin."
        )
    ax, ay, az = np.diag(cell)

    # Ne CCE atoms
    ne_indices = [
        i for i, sym in enumerate(structure.get_chemical_symbols()) if sym == "Ne"
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
                for i, sym in enumerate(structure.get_chemical_symbols())
                if sym in electrode_elements
            ]
        ]
    )

    d_electrode = float(np.max(structure[ne_indices].positions[:, 2])) - float(
        np.max(structure[electrode_indices].positions[:, 2])
    )

    #  This is to make sure NELECT and the Ne ZVAL cancel out exactly
    zval_ne = float(f"{(8 + np.round(Q0 / n_Ne, 8)):.7f}")
    nelect_adjusted = float(nelect_neutral) + float(n_Ne) * (
        np.float64(f"{zval_ne:.7f}") - 8
    )

    ne_zval_new_line = re.sub(
        r"(ZVAL\s*=\s*)[\d.]+", f"ZVAL   =    {zval_ne:.7f}", ne_zval_original_line
    )

    # -------------------------------------------------------------------------
    # OUTPUT variables
    # -------------------------------------------------------------------------

    cce_params = CEParameters(
        path_to_plugin=path_to_plugin,
        temperature=temperature,
        phi0=potential,
        Q0=Q0,
        nelect_neutral=nelect_neutral,
        grid_position_frac=grid_position_frac,
        grid_roll_frac=grid_roll_frac,
        ax=ax,
        ay=ay,
        az=az,
        d_electrode=d_electrode,
        i_Ne=i_Ne,
        n_elements=n_elements,
        n_Ne=n_Ne,
        tau=tau,
    )

    plugin_content = _plugin_content(cce_params)

    extra_incar = {
        "NELECT": nelect_adjusted,
        "PLUGINS/LOCAL_POTENTIAL": "T",
        "PLUGINS/OCCUPANCIES": "T",
    }

    plugin_data = VaspPlugin(
        plugin_content=plugin_content,
        override_potcar={ne_zval_original_line: ne_zval_new_line},
        extra_incar=extra_incar,
        potcar_lib_path=potcar_lib_path,
    )

    return structure, calc, plugin_data


@as_function_node
def CDCESetup(
    structure: Atoms,
    electrode: Atoms,
    calc: VaspInput,
    potcar_lib_path: str = _default_potcar_lib_path,
    path_to_plugin: str = "pyiron_nodes/electrochemistry/add_potential/vasp_plugin-CDCE_MD.plugin",
    potential: float = 0.0,
    Q0: float = 0.0,
    grid_position_frac: float = 0.85,
    grid_roll_frac: float = 0.1,
    pos_right_wall: float = 0.75,  # for now this is in fractional coordinates
    width_wall: float = 6.5,
    tau: float = 50.0,
):
    """Build the CDCE (charged-dielectric counter-charge) plugin setup for a VASP MD run.

    The CDCE variant places a Gaussian counter-charge behind a wall potential
    instead of a Ne layer, so unlike ``CCESetup`` it needs no Ne atoms and does
    not touch the POTCAR. This computes every structure-dependent quantity
    from the ASE structure and the individual element POTCARs (cell
    dimensions, counter-charge position, electrode–counter-charge distance,
    neutral electron count), derives the electrochemistry INCAR tags, and
    renders the filled CDCE plugin file content. These are packaged into a
    ``VaspPlugin`` object, to be written out later (e.g. by
    ``CreateVaspInputResources``). The calc must carry MD settings, since
    ``calc.md.temperature`` is read directly.

    Parameters
    ----------
    structure
        The atomic structure (ASE ``Atoms``) for which to set up the CDCE
        plugin.
    electrode
        The electrode structure; its elements identify the electrode atoms in
        the full cell, used to measure the electrode–counter-charge distance.
    calc
        The VASP input settings (``VaspInput``); must carry MD settings
        (``calc.md.temperature``) and SCF settings (``calc.scf.functional``).
    potcar_lib_path
        Path to the POTCAR library used to look up ZVAL values for each
        element in the structure.
    path_to_plugin
        Path to the CDCE plugin template.
    potential
        Target electrode potential in volts (``phi0`` in the plugin).
    Q0
        Extra charge (electrons) added to the electrode; shifts ``NELECT``
        directly.
    grid_position_frac
        Fractional z where the dipole/grid correction is applied.
    grid_roll_frac
        Grid roll fraction passed to the plugin.
    pos_right_wall
        Fractional z position of the right wall potential.
    width_wall
        Width of the wall potential, in Å.
    tau
        Thermostat time constant, in units of the MD time step.

    Returns
    -------
    Atoms, VaspInput, VaspPlugin
        The unmodified ``structure`` and ``calc``, together with a
        ``VaspPlugin`` bundling the rendered plugin content and the extra
        INCAR tags (``NELECT`` and the plugin activation flags) needed to run
        the CDCE setup. No POTCAR override is included, since CDCE does not
        modify any POTCAR.

    """

    temperature = calc.md.temperature
    functional = calc.scf.functional

    potcar_paths = _get_potcar_paths(structure, functional, potcar_lib_path)

    # Read ZVAL for each element from its individual POTCAR file
    # POTCAR line format: "   POMASS =  196.970; ZVAL   =   11.000    mass and valenz"
    zval_per_element = {}
    unique_elements = _ordered_elements(structure)

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
        sum(zval_per_element[sym] for sym in structure.get_chemical_symbols())
    )

    # -------------------------------------------------------------------------
    # Structure-derived quantities
    # -------------------------------------------------------------------------

    # Cell dimensions - must be orthogonal
    cell = structure.get_cell()
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
                for i, sym in enumerate(structure.get_chemical_symbols())
                if sym in electrode_elements
            ]
        ]
    )

    Q_pos = np.array([ax * 0.5, ay * 0.5, az * pos_right_wall - 2.0])

    d_electrode = float(
        Q_pos[2] - float(np.max(structure[electrode_indices].positions[:, 2]))
    )

    # Conwering to string to ensure the format in vasp_plugin.py is correct
    Q_pos = f"np.array({Q_pos.tolist()})"

    nelect_adjusted = float(nelect_neutral) + np.round(Q0)

    # -------------------------------------------------------------------------
    # INCAR dictionary
    # -------------------------------------------------------------------------

    cdce_params = CEParameters(
        path_to_plugin=path_to_plugin,
        phi0=potential,
        Q0=Q0,
        Q_pos=Q_pos,
        nelect_neutral=nelect_neutral,
        grid_position_frac=grid_position_frac,
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

    plugin_content = _plugin_content(cdce_params)

    extra_incar = {
        "PLUGINS/LOCAL_POTENTIAL": "T",
        "PLUGINS/OCCUPANCIES": "T",
        "PLUGINS/FORCE_AND_STRESS": "T",
        "NELECT": nelect_adjusted,
        "LREMOVE_DRIFT": "F",
    }

    plugin_data = VaspPlugin(
        plugin_content=plugin_content,
        extra_incar=extra_incar,
        potcar_lib_path=potcar_lib_path,
    )

    return structure, calc, plugin_data


@as_function_node
def ParsePotential(
    io_bundle: VaspInputResources,
):
    """Read the constant-potential plugin's output traces from the working dir.

    Loads the three files the CCE/CDCE plugin writes over the MD run — electrode
    charge (``Q.dat``), electrode potential (``phi.dat``) and the planar-averaged
    electrostatic potential (``el_pot_z.dat``) — and reshapes the flat
    electrostatic-potential trace into ``(NSW, nz)``, one profile per ionic step.
    ``NSW`` is taken from ``extra_incar`` if present, otherwise from the MD
    settings on the calc.

    Parameters
    ----------
    io_bundle : VaspInputResources
        Bundle whose ``working_directory`` holds the plugin output files.

    Returns
    -------
    electrostatic_potential_z_2d : numpy.ndarray
        Planar-averaged electrostatic potential, shape ``(NSW, nz)``.
    charge_list : numpy.ndarray
        Electrode charge per step, from ``Q.dat``.
    pot_list : numpy.ndarray
        Electrode potential per step, from ``phi.dat``.

    Raises
    ------
    FileNotFoundError
        If any of ``Q.dat`` / ``phi.dat`` / ``el_pot_z.dat`` is missing.
    ValueError
        If ``NSW`` can be found neither in ``extra_incar`` nor on the MD calc.
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
