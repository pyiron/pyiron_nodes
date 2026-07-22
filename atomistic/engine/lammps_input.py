from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from ase.atoms import Atoms

from core import as_function_node
from lammpsparser.units import LAMMPS_UNIT_CONVERSIONS

from pyiron_nodes.atomistic.engine.lammps import LammpsIOBundle

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _get_env():
    """Lazily build the Jinja2 environment (defers the import for optional-dep hygiene)."""
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )


_env = None


def _render(template_name: str, **context) -> str:
    """Render a section template.

    Always returns a string ending with '\\n' so that the core port's
    _coerce() method (which strips leading/trailing quotes) never truncates
    LAMMPS format strings that end with '"'.
    """
    global _env
    if _env is None:
        _env = _get_env()
    rendered = _env.get_template(template_name).render(**context)
    return rendered.rstrip("\n") + "\n"


@as_function_node("init")
def LammpsInit(
    structure: Atoms,
    units: str = "metal",
    atom_style: str = "atomic",
    structure_filename: str = "lammps.data",
    read_restart: Optional[str] = None,
) -> str:
    """Initialization block: units, dimension, boundary, atom_style, read_data/read_restart.

    This is typically the first section of a LAMMPS input and is common to all
    calculation types (MD, minimize, static). Feed the result into
    `AssembleLammpsInput(init=...)`.
    """
    dim = int(sum(structure.pbc))
    boundary = " ".join("p" if p else "f" for p in structure.pbc)

    return _render(
        "init.j2",
        units=units,
        dim=dim,
        boundary=boundary,
        atom_style=atom_style,
        read_restart=read_restart,
        structure_filename=structure_filename,
    )


@as_function_node(["potential", "potential_file_content"])
def LammpsPotential(
    potential: str | pd.DataFrame,
    resource_path: Optional[str] = None,
    potential_filename: str = "potential.inp",
) -> tuple[str, str]:
    """Potential block: pair_style, pair_coeff (or `include potential.inp`).

    For DataFrame potentials the full potential text is returned as
    `potential_file_content` and the input file references it via `include`.
    For string potentials the commands are inlined and `potential_file_content`
    is empty.

    Connect `potential_file_content` to `SetLammpsInputString` so the file gets
    written to disk before the LAMMPS run.
    """
    from lammpsparser.compatibility.file import _get_potential

    potential_list, _, _ = _get_potential(
        potential=potential, resource_path=resource_path
    )

    if isinstance(potential, pd.DataFrame):
        section = _render("potential_include.j2", potential_filename=potential_filename)
        potential_file_content = "".join(potential_list)
    else:
        commands = "\n".join(line.rstrip() for line in potential_list if line.strip())
        section = _render("potential_inline.j2", commands=commands)
        potential_file_content = ""

    return section, potential_file_content


@as_function_node("dump")
def LammpsDump(
    n_print: int = 100,
    filename: str = "dump.out",
) -> str:
    """Dump block: trajectory output configuration.

    Writes wrapped fractional coordinates (xsu ysu zsu) plus forces and
    velocities in the standard pyiron format.
    """
    return _render("dump.j2", n_print=n_print, filename=filename)


@as_function_node("thermo")
def LammpsThermo(
    n_print: int = 100,
) -> str:
    """Thermo block: output frequency, style, and format for log.lammps."""
    return _render("thermo.j2", n_print=n_print)


@as_function_node(["ensemble", "fix_ids"])
def LammpsEnsemble(
    temperature: Optional[float] = 300.0,
    pressure: Optional[float] = None,
    temperature_damping: float = 100.0,
    pressure_damping: float = 1000.0,
    langevin: bool = False,
    seed: int = 42,
    units: str = "metal",
) -> tuple[str, list[str]]:
    """Ensemble block: fix NVE / NVT / NPT (Nosé–Hoover or Langevin).

    Parameters are in pyiron natural units:
    - temperature [K], pressure [GPa], damping timescales [fs].
    Unit conversion to the target LAMMPS unit system is applied automatically.

    - temperature=None, pressure=None → NVE
    - temperature set, pressure=None → NVT
    - temperature set, pressure set  → NPT (isotropic)
    - langevin=True                  → NVE+Langevin (NVT) or NPH+Langevin (NPT)

    Unlike `LammpsMinimize` (whose box/relax fix is unfixed immediately after
    its self-executing `minimize` command), this fix stays active until a
    later `run` executes the dynamics — potentially in a separate
    `AssembleLammpsInput` call. So `fix_ids` (the LAMMPS fix ID(s) this
    section creates) is returned alongside the rendered block: connect it to
    `AssembleLammpsInput`'s `calculation_fix_ids` so the matching `unfix` gets
    emitted after the `run` that uses it.
    """
    conversions = LAMMPS_UNIT_CONVERSIONS[units]
    t_damp = temperature_damping * conversions["time"]
    p_damp = pressure_damping * conversions["time"]

    t = p = None
    if temperature is None and pressure is None:
        mode = "nve"
    elif pressure is not None:
        t = temperature * conversions["temperature"]
        p = pressure * conversions["pressure"]
        mode = "nph_langevin" if langevin else "npt"
    else:
        t = temperature * conversions["temperature"]
        mode = "nve_langevin" if langevin else "nvt"

    fix_ids = ["ensemble", "langevin"] if "langevin" in mode else ["ensemble"]

    ensemble = _render(
        "ensemble.j2",
        mode=mode,
        t=t,
        p=p,
        t_damp=t_damp,
        p_damp=p_damp,
        seed=seed,
    )
    return ensemble, fix_ids


@as_function_node("velocity")
def LammpsVelocity(
    temperature: float = 300.0,
    seed: int = 42,
    time_step: float = 1.0,
    units: str = "metal",
) -> str:
    """Velocity + timestep block.

    Sets the integration timestep and initialises velocities from a Gaussian
    distribution at 2×temperature (equipartition: half kinetic, half potential).

    Parameters: temperature [K], time_step [fs].
    """
    conversions = LAMMPS_UNIT_CONVERSIONS[units]
    ts = time_step * conversions["time"]
    init_temp = 2.0 * temperature * conversions["temperature"]

    return _render("velocity.j2", time_step=ts, init_temp=init_temp, seed=seed)


@as_function_node("minimize")
def LammpsMinimize(
    e_tol: float,
    f_tol: float,
    max_iter: int = 1_000_000,
    style: str = "cg",
    pressure: Optional[float] = None,
    units: str = "metal",
) -> str:
    """Minimization block: optional box relaxation, min_style, minimize.

    Parameters: e_tol [eV], f_tol [eV/Å], pressure [GPa].
    When pressure is set, adds `fix ensemble all box/relax iso P` before
    the minimize command to allow cell volume to change, and `unfix ensemble`
    immediately after — unlike `LammpsEnsemble`, `minimize` is self-executing
    (it doesn't wait for a later `run`), so the fix is torn down within this
    same section rather than needing `AssembleLammpsInput` to track it.
    """
    pressure_relax = None
    if pressure is not None:
        pressure_relax = pressure * LAMMPS_UNIT_CONVERSIONS[units]["pressure"]

    return _render(
        "minimize.j2",
        pressure_relax=pressure_relax,
        style=style,
        e_tol=e_tol,
        f_tol=f_tol,
        max_iter=max_iter,
    )


@as_function_node
def AssembleLammpsInput(
    io_bundle: LammpsIOBundle,
    init: str = "",
    potential: str = "",
    potential_file_content: str = "",
    dump: str = "",
    thermo: str = "",
    calculation: str = "",
    calculation_fix_ids: list[str] = [],
    velocity: str = "",
    run: Optional[int] = None,
    write_restart: Optional[str] = None,
) -> LammpsIOBundle:
    """Assemble named sections into a LAMMPS input, in the fixed order LAMMPS
    requires (init, potential, dump, thermo, velocity, calculation, run), and
    inject/append it into `io_bundle` — mirroring how `CreateLammpsMDInput`
    builds `lammps_input_string` and returns the updated bundle. No separate
    bridge node is needed to reach `RunLammpsCalculation`.

    Chainable across multiple simulation stages: call this node again on the
    same `io_bundle` to append a further stage (e.g. an NVT equilibration
    followed by an NPT production run). The first call (`io_bundle`'s
    `lammps_input_string` still empty) is where `init`/`potential` apply;
    later calls ignore them (with a printed note) since read_data/pair_style
    belong to the first stage only — just leave those two unconnected for
    stage 2+.

    `calculation` is a single slot fed by *either* `LammpsEnsemble`'s or
    `LammpsMinimize`'s output — the two are mutually exclusive ways to set up
    what the run actually does (an MD ensemble fix, or a box/relax fix plus
    `minimize`), so they share one port rather than needing two.
    `LammpsMinimize`'s box/relax fix (if any) is unfixed immediately after its
    own self-executing `minimize` command, so it needs nothing further here.
    `LammpsEnsemble`'s fix(es) stay active until a `run` actually executes the
    dynamics, so its second output, `fix_ids`, should be connected to
    `calculation_fix_ids` — this node remembers them on `io_bundle` and emits
    the matching `unfix` line(s) right after the `run` (and any
    `write_restart`) that uses them, even if that `run` arrives in a later
    call than the one that set up `calculation`.

    Omit a string section (leave it as the default empty string) to skip it,
    e.g. leave `calculation`/`velocity` empty for a static run.

    `run` is the number of MD steps to execute (0 for a static single-point
    evaluation), rendered directly here rather than via a separate node since
    every input string is already a fully-rendered LAMMPS section. Leave
    `run=None` to omit the run command for this stage — but an ensemble
    `calculation` (one with `calculation_fix_ids`) left without a `run` is
    only valid as the *last* stage: if a later call supplies another
    `calculation` before this one ever got a `run` (and its `unfix`), that fix
    was set up and then abandoned, so this raises a `ValueError`.

    `write_restart` does *not* require `run` to be set — leaving `run=None`
    while supplying `write_restart` still emits `write_restart <filename>`
    (with no preceding `run` line), so a restart snapshot can be requested for
    a stage that has no MD steps of its own (e.g. right after a minimize).

    `potential_file_content` (from `LammpsPotential`'s second output) is
    written to `io_bundle.lammps_potential_string` when non-empty — needed
    for DataFrame potentials that produce an `include potential.inp` line.
    """
    is_first_stage = not io_bundle.lammps_input_string

    if calculation and io_bundle.lammps_pending_fix_ids:
        raise ValueError(
            "The previous `calculation` section's fix(es) "
            f"({io_bundle.lammps_pending_fix_ids}) were never `unfix`ed via a "
            "`run` before this new stage — add a `run` to that stage before "
            "adding another `calculation`."
        )

    if not is_first_stage and (init or potential):
        print(
            "AssembleLammpsInput: ignoring `init`/`potential` — they only "
            "apply to the first stage of an io_bundle; later stages reuse "
            "the existing read_data/pair_style setup."
        )
        init = ""
        potential = ""

    if calculation:
        io_bundle.lammps_pending_fix_ids = list(calculation_fix_ids)

    render_run_section = run is not None or write_restart
    run_section = (
        _render(
            "run.j2",
            n_steps=run,
            write_restart=write_restart,
            fix_ids=io_bundle.lammps_pending_fix_ids,
        )
        if render_run_section
        else ""
    )
    new_block = _render(
        "master.j2",
        init=init,
        potential=potential,
        dump=dump,
        thermo=thermo,
        calculation=calculation,
        velocity=velocity,
        run=run_section,
    )
    if is_first_stage:
        io_bundle.lammps_input_string = new_block
    else:
        io_bundle.lammps_input_string += "\n" + new_block

    if render_run_section:
        io_bundle.lammps_pending_fix_ids = []

    if potential_file_content:
        io_bundle.lammps_potential_string = potential_file_content
    return io_bundle


@as_function_node
def SetLammpsInputString(
    io_bundle: LammpsIOBundle,
    lammps_input_string: str,
    potential_file_content: str = "",
) -> LammpsIOBundle:
    """Bridge: inject an assembled input string into an existing LammpsIOBundle.

    Connects `AssembleLammpsInput`'s output to the existing RunLammpsCalculation
    pipeline. Optionally also sets the potential file content (needed when using
    DataFrame potentials that produce an `include potential.inp` line).
    """
    io_bundle.lammps_input_string = lammps_input_string
    if potential_file_content:
        io_bundle.lammps_potential_string = potential_file_content
    return io_bundle


@as_function_node("path")
def WriteFile(io_bundle: LammpsIOBundle, filename: str = "output.txt") -> str:
    """Write `io_bundle.lammps_input_string` to a file and return the file path."""
    with open(filename, "w") as f:
        f.write(io_bundle.lammps_input_string)
    path = filename
    return path
