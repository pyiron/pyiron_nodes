"""Elastic constants from stress-strain slopes.

The elastic tensor is obtained by straining the cell along each of the six
Voigt directions, reading the stress from the calculator, and taking the
central-difference slope ``C[i, j] = dsigma_i / d eps_j``.  No symmetry
analysis is required, which makes the approach applicable to any structure
whose calculator provides stresses (EMT, GRACE, M3GNet, LAMMPS, ...).
"""

from dataclasses import field
from typing import Optional

import numpy as np

from core import (
    Node,
    as_function_node,
    as_inp_dataclass_node,
    as_out_dataclass_node,
)

# eV/Angstrom^3 -> GPa
EV_PER_ANG3_TO_GPA = 160.21766208

# Voigt index -> pair of Cartesian indices
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


@as_inp_dataclass_node
class InputElasticTensor:
    num_of_point: int = 5
    eps_range: float = 0.005
    sqrt_eta: bool = True
    fit_order: int = 2


@as_out_dataclass_node
class DataStructureContainer:
    structure: list = field(default_factory=lambda: [])
    job_name: list = field(default_factory=lambda: [])
    energy: list = field(default_factory=lambda: [])
    forces: list = field(default_factory=lambda: [])
    stress: list = field(default_factory=lambda: [])


@as_out_dataclass_node
class OutputElasticAnalysis:
    strain_energy: list = field(default_factory=lambda: [])
    C: np.ndarray = field(default_factory=lambda: np.zeros(0))
    A2: list = field(default_factory=lambda: [])
    C_eigval: np.ndarray = field(default_factory=lambda: np.zeros(0))
    C_eigvec: np.ndarray = field(default_factory=lambda: np.zeros(0))

    BV: int | float = 0
    GV: int | float = 0
    EV: int | float = 0
    nuV: int | float = 0
    S: int | float = 0
    BR: int | float = 0
    GR: int | float = 0
    ER: int | float = 0
    nuR: int | float = 0
    BH: int | float = 0
    GH: int | float = 0
    EH: int | float = 0
    nuH: int | float = 0
    AVR: int | float = 0
    energy_0: float = 0

    _skip_default_values = False


@as_function_node  # ("structure_container")
def AddEnergies(
    structure_container: DataStructureContainer,
    engine: Node,
) -> DataStructureContainer:
    for structure in structure_container.structure:
        engine.inputs.structure = structure
        out = engine.run()
        structure_container.energy.append(out.energies_pot[-1])

    return structure_container


@as_function_node("forces")
def ExtractFinalEnergy(df):
    # Looks an awful lot like phonons.ExtractFinalForce -- room for abstraction here
    return [e.energy[-1] for e in df["out"].tolist()]


def voigt_strain_matrix(component: int, magnitude: float) -> np.ndarray:
    """Return the 3x3 strain tensor for one Voigt component.

    Components 0-2 are the normal strains ``eps_xx, eps_yy, eps_zz``;
    components 3-5 are the *engineering* shear strains
    ``gamma_yz, gamma_xz, gamma_xy``, so the corresponding tensor entries are
    ``magnitude / 2``.
    """
    if not 0 <= int(component) <= 5:
        raise ValueError(f"Voigt component must be in 0..5, got {component}")
    component = int(component)
    eps = np.zeros((3, 3))
    i, j = _VOIGT_PAIRS[component]
    if component < 3:
        eps[i, j] = magnitude
    else:
        eps[i, j] = eps[j, i] = magnitude / 2.0
    return eps


@as_function_node("structure")
def ApplyVoigtStrain(structure, component: int = 0, magnitude: float = 0.005):
    """Strain a structure along a single Voigt direction.

    Parameters
    ----------
    structure : Atoms or OutputAtoms
        Reference structure.
    component : int
        Voigt index 0..5, i.e. ``xx, yy, zz, yz, xz, xy``.  Components 3-5
        apply an engineering shear strain.
    magnitude : float
        Strain amplitude; may be negative.

    Returns
    -------
    Atoms
        A copy of *structure* with the cell deformed by ``I + eps`` and the
        atomic positions scaled along with it.
    """
    from pyiron_nodes.atomistic.structure._atoms import _resolve_atoms

    atoms = _resolve_atoms(structure).copy()
    deformation = np.eye(3) + voigt_strain_matrix(component, magnitude)
    atoms.set_cell(atoms.get_cell() @ deformation, scale_atoms=True)
    return atoms


def _relax_cell(atoms, fmax: float = 1e-4, steps: int = 200):
    """Relax cell and positions to (nearly) zero stress, in place."""
    from ase.optimize import LBFGS

    try:  # ase >= 3.23
        from ase.filters import FrechetCellFilter as CellFilter
    except ImportError:  # pragma: no cover - older ase
        from ase.constraints import ExpCellFilter as CellFilter

    opt = LBFGS(CellFilter(atoms), logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return atoms


@as_function_node
def StressStrainElasticConstants(
    structure,
    engine=None,
    eps: float = 0.005,
    relax_reference: bool = False,
):
    """Elastic constants from the stress-strain slopes.

    For each Voigt component ``j`` the cell is strained by ``+eps`` and
    ``-eps`` and the full stress tensor is read from the calculator, giving
    the central-difference slope ``C[:, j] = (sigma(+eps) - sigma(-eps)) /
    (2 eps)``.  The result is symmetrised as ``0.5 * (C + C.T)``.

    Parameters
    ----------
    structure : Atoms or OutputAtoms
        Reference structure.  Residual stress in the reference cell biases
        the slopes, so either supply a relaxed cell or set
        *relax_reference*.
    engine : OutputEngine or None
        Calculator engine; defaults to EMT when ``None``.
    eps : float
        Strain amplitude used for the central difference.
    relax_reference : bool
        Relax cell and positions to zero stress before straining.

    Returns
    -------
    elastic_analysis : OutputElasticAnalysis
        Full 6x6 tensor ``C`` in GPa plus the Voigt-Reuss-Hill moduli.
    C11 : float
        ``C[0, 0]`` in GPa.
    C12 : float
        ``C[0, 1]`` in GPa.
    C44 : float
        ``C[3, 3]`` in GPa.
    """
    from pyiron_nodes.atomistic.engine.generic import OutputEngine
    from pyiron_nodes.atomistic.structure._atoms import _resolve_atoms

    if engine is None:
        from ase.calculators.emt import EMT

        engine = OutputEngine(calculator=EMT())

    reference = _resolve_atoms(structure).copy()
    reference.calc = engine.calculator
    if relax_reference:
        _relax_cell(reference)

    C = np.zeros((6, 6))
    for component in range(6):
        stresses = []
        for sign in (+1, -1):
            strained = reference.copy()
            deformation = np.eye(3) + voigt_strain_matrix(component, sign * eps)
            strained.set_cell(reference.get_cell() @ deformation, scale_atoms=True)
            strained.calc = engine.calculator
            stresses.append(np.asarray(strained.get_stress(voigt=True), dtype=float))
        C[:, component] = (stresses[0] - stresses[1]) / (2 * eps)

    C = 0.5 * (C + C.T) * EV_PER_ANG3_TO_GPA

    elastic_analysis = OutputElasticAnalysis.pure_dataclass()
    elastic_analysis.C = C
    elastic_analysis.energy_0 = float(reference.get_potential_energy())
    calculate_modulus(elastic_analysis)

    C11 = float(C[0, 0])
    C12 = float(C[0, 1])
    C44 = float(C[3, 3])
    return elastic_analysis, C11, C12, C44


@as_function_node
def ComputeElasticConstants(
    structure,
    engine=None,
    input_elastic_tensor: Optional[InputElasticTensor] = None,
    relax_reference: bool = False,
):
    """Elastic constants of *structure* from the stress-strain slopes.

    Thin wrapper around :func:`StressStrainElasticConstants` that takes the
    strain amplitude from an :class:`InputElasticTensor` when one is given.

    Returns
    -------
    elastic_analysis : OutputElasticAnalysis
    C11 : float
    C12 : float
    C44 : float
    """
    eps = 0.005 if input_elastic_tensor is None else input_elastic_tensor.eps_range

    elastic_analysis, C11, C12, C44 = StressStrainElasticConstants(
        structure=structure,
        engine=engine,
        eps=eps,
        relax_reference=relax_reference,
    ).run()

    return elastic_analysis, C11, C12, C44


def calculate_modulus(out: OutputElasticAnalysis):
    C = out.C

    BV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9
    GV = (
        (C[0, 0] + C[1, 1] + C[2, 2])
        - (C[0, 1] + C[0, 2] + C[1, 2])
        + 3 * (C[3, 3] + C[4, 4] + C[5, 5])
    ) / 15
    EV = (9 * BV * GV) / (3 * BV + GV)
    nuV = (1.5 * BV - GV) / (3 * BV + GV)
    out.BV = BV
    out.GV = GV
    out.EV = EV
    out.nuV = nuV

    try:
        S = np.linalg.inv(C)

        BR = 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
        GR = 15 / (
            4 * (S[0, 0] + S[1, 1] + S[2, 2])
            - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
            + 3 * (S[3, 3] + S[4, 4] + S[5, 5])
        )
        ER = (9 * BR * GR) / (3 * BR + GR)
        nuR = (1.5 * BR - GR) / (3 * BR + GR)

        BH = 0.50 * (BV + BR)
        GH = 0.50 * (GV + GR)
        EH = (9.0 * BH * GH) / (3.0 * BH + GH)
        nuH = (1.5 * BH - GH) / (3.0 * BH + GH)

        AVR = 100.0 * (GV - GR) / (GV + GR)
        out.S = S

        out.BR = BR
        out.GR = GR
        out.ER = ER
        out.nuR = nuR

        out.BH = BH
        out.GH = GH
        out.EH = EH
        out.nuH = nuH

        out.AVR = AVR
    except np.linalg.LinAlgError as e:
        print("LinAlgError:", e)

    eigval, eigvec = np.linalg.eig(C)
    out.C_eigval = eigval
    out.C_eigvec = eigvec

    return out
