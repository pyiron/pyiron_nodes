"""
Multi‑species sp‑tight‑binding builder for an ASE Atoms object.
The neighbour‑list construction is compatible with all ASE releases.
"""

import numpy as np
from scipy.sparse import csr_matrix
from ase import Atoms
from ase.neighborlist import neighbor_list
from typing import Tuple, Dict, List, Literal, Optional
from core import as_function_node


# -------------------------------------------------------------------------
def _pair_key(el1: str, el2: str) -> Tuple[str, str]:
    """Order‑independent dict key for a pair of elements."""
    return tuple(sorted((el1, el2)))


def _sk_matrix(
    l: float, m: float, n: float, Vssσ: float, Vspσ: float, Vppσ: float, Vppπ: float
) -> np.ndarray:
    """4×4 (s,px,py,pz) Slater‑Koster hopping block."""
    Hss = Vssσ
    Hsp = np.array([Vspσ * l, Vspσ * m, Vspσ * n])  # s ↔ p
    dir_vec = np.array([l, m, n])
    outer = np.outer(dir_vec, dir_vec)  # l_i l_j
    delta = np.eye(3)
    Hpp = Vppσ * outer + Vppπ * (delta - outer)  # p‑p block

    block = np.zeros((4, 4), dtype=float)
    block[0, 0] = Hss
    block[0, 1:] = Hsp
    block[1:, 0] = Hsp
    block[1:, 1:] = Hpp
    return block


@as_function_node
def tbHamilton(
    atoms: Atoms,
    rcut: float = 5.0,
    onsite: Dict[str, Tuple[float, float, float, float]] | None = None,
    hoppings: Dict[Tuple[str, str], Tuple[float, float, float, float]] | None = None,
    scaling: Literal["harrison", "none"] = "harrison",
    H_format: Literal["sparse", "full"] = "full",
    verbose: bool = False,
    charges: np.ndarray | None = None,
    chi: np.ndarray | None = None,
    U: np.ndarray | None = None,
) -> Tuple[csr_matrix, List[Tuple[int, str]]]:
    """
    Convert an ASE Atoms object (any composition) into a real‑space sp‑TB Hamiltonian.

    Parameters
    ----------
    atoms
        ASE Atoms object describing the system.
    rcut
        Cut‑off radius (Å) for neighbour‑list generation.
    onsite
        Dictionary ``{symbol: (Es, Epx, Epy, Epz)}``.  If ``None`` a crude
        Harrison estimate is used.
    hoppings
        Dictionary ``{(el1, el2): (Vssσ, Vspσ, Vppσ, Vppπ)}``.  If ``None`` a minimal
        default table is employed.
    scaling
        ``"harrison"`` (default) – scale hoppings as 1/d²; ``"none"`` – no scaling.
    H_format
        Return a dense ``numpy.matrix`` (``"full"``) or a ``scipy.sparse.csr_matrix``
        (``"sparse"``).
    verbose
        Print a short summary of the construction.
    charges
        Optional per‑atom charge (electron population) array of shape ``(n_atoms,)``.
        If supplied the on‑site diagonal is shifted according to the
        charge‑equilibration model (see :func:`ShiftDiagonal`).
    chi, U
        Optional linear (`chi`) and quadratic (`U`) coefficients for the
        charge‑dependent shift.  If omitted they are obtained from
        :func:`ChiUData(atoms)`.

    Returns
    -------
    H, orbital_map
        ``H`` is the Hamiltonian in the requested format and ``orbital_map``
        maps each row/column index to ``(atom_index, orbital)``.
    """
    # -----------------------------------------------------------------
    # 0. Basic checks
    # -----------------------------------------------------------------
    if not isinstance(atoms, Atoms):
        raise TypeError("`atoms` must be an ASE Atoms object")
    N = len(atoms)
    if N == 0:
        raise ValueError("Empty Atoms object")

    # -----------------------------------------------------------------
    # 1. Neighbour list – works with every ASE version
    # -----------------------------------------------------------------
    i_idx, j_idx = neighbor_list("ij", atoms, rcut, self_interaction=False)

    distances = atoms.get_distances(i_idx, j_idx, mic=True)
    vectors = atoms.get_distances(i_idx, j_idx, mic=True, vector=True)

    # -----------------------------------------------------------------
    # 2. On‑site energies (default = crude Harrison estimate)
    # -----------------------------------------------------------------
    if onsite is None:
        from ase.data import covalent_radii

        onsite = {}
        for Z in np.unique(atoms.numbers):
            elem = atoms[atoms.numbers.tolist().index(Z)].symbol
            a0 = covalent_radii[Z]
            α_s, α_p = 1.0, 0.5
            Es = -α_s / a0
            Ep = -α_p / a0
            onsite[elem] = (Es, Ep, Ep, Ep)

    # -----------------------------------------------------------------
    # 3. Hopping parameters (default minimal table)
    # -----------------------------------------------------------------
    if hoppings is None:
        hoppings = {
            _pair_key("H", "H"): (-2.0, 2.5, 3.0, -1.0),
            _pair_key("H", "O"): (-1.5, 2.2, 2.8, -0.9),
            _pair_key("O", "O"): (-1.2, 2.0, 2.5, -0.8),
            _pair_key("Pt", "Pt"): (-3.0, 3.5, 4.0, -1.2),
            _pair_key("Pt", "H"): (-2.5, 3.0, 3.5, -1.0),
            _pair_key("Pt", "O"): (-2.2, 2.8, 3.2, -0.9),
        }

    # -----------------------------------------------------------------
    # 4. Helper: global index from (atom,orbital)
    # -----------------------------------------------------------------
    def gid(atom_idx: int, orb: int) -> int:
        """orbital: 0=s, 1=px, 2=py, 3=pz."""
        return 4 * atom_idx + orb

    rows, cols, data = [], [], []

    # -----------------------------------------------------------------
    # 5a. On‑site blocks
    # -----------------------------------------------------------------
    for a_idx, atom in enumerate(atoms):
        elem = atom.symbol
        Es, Epx, Epy, Epz = onsite[elem]
        for orb, val in enumerate([Es, Epx, Epy, Epz]):
            ii = gid(a_idx, orb)
            rows.append(ii)
            cols.append(ii)
            data.append(val)

    # -----------------------------------------------------------------
    # 5b. Hopping blocks
    # -----------------------------------------------------------------
    for i, j, d, vec in zip(i_idx, j_idx, distances, vectors):
        if d < 1e-3:  # skip spurious intra‑atomic pairs
            continue

        l, m, n = vec / d  # direction cosines

        elem_i = atoms[i].symbol
        elem_j = atoms[j].symbol
        key = _pair_key(elem_i, elem_j)

        if key not in hoppings:
            raise KeyError(f"No hopping parameters for pair {key}")

        Vssσ, Vspσ, Vppσ, Vppπ = hoppings[key]

        if scaling == "harrison":
            scale = (1.0 / d) ** 2  # V ∝ 1/d²
            Vssσ *= scale
            Vspσ *= scale
            Vppσ *= scale
            Vppπ *= scale
        elif scaling != "none":
            raise ValueError("`scaling` must be 'harrison' or 'none'")

        block = _sk_matrix(l, m, n, Vssσ, Vspσ, Vppσ, Vppπ)

        # Insert both (i→j) and (j→i) – Hamiltonian is real‑symmetric
        for orb_i in range(4):
            for orb_j in range(4):
                ii = gid(i, orb_i)
                jj = gid(j, orb_j)
                val = block[orb_i, orb_j]

                rows.append(ii)
                cols.append(jj)
                data.append(val)
                rows.append(jj)
                cols.append(ii)
                data.append(val)

    # -----------------------------------------------------------------
    # 6. Assemble CSR matrix
    # -----------------------------------------------------------------
    dim = 4 * N
    H = csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=float)
    if verbose:
        print("H: ", charges is None, H)

    # -----------------------------------------------------------------
    # 7. OPTIONAL: shift diagonal according to a charge density
    # -----------------------------------------------------------------
    if charges is not None:
        # --- sanity checks -------------------------------------------------
        charges = np.asarray(charges, dtype=float)
        if charges.ndim != 1 or charges.shape[0] != N:
            raise ValueError(
                "`charges` must be a 1‑D array of length equal to the number of atoms"
            )

        # --- obtain chi and U if the user did not supply them -------------
        if chi is None or U is None:
            # ChiUData returns two 1‑D arrays of length N
            chi_default, U_default = ChiUData(atoms).run()
            if chi is None:
                chi = chi_default
            if U is None:
                U = U_default

        # --- compute shifted diagonal --------------------------------------
        eps0 = H.diagonal()  # shape (dim,)
        shifted_diag = ShiftDiagonal(eps0, charges, chi, U).run()  # also (dim,)

        # --- replace the diagonal (CSR) ------------------------------------
        # csr_matrix.setdiag returns a new matrix, so we reassign.
        H.setdiag(shifted_diag)

    # -----------------------------------------------------------------
    # 8. Orbital map (optional)
    # -----------------------------------------------------------------
    orb_labels = ["s", "px", "py", "pz"]
    orbital_map = [
        (atom_idx, orb_labels[orb]) for atom_idx in range(N) for orb in range(4)
    ]

    # -----------------------------------------------------------------
    # 9. Verbose output
    # -----------------------------------------------------------------
    if verbose:
        print(f"sp‑TB Hamiltonian for {N} atoms → {dim} orbitals")
        print(f"  Cut‑off radius : {rcut:.2f} Å")
        print(f"  Scaling        : {scaling}")
        print(f"  Non‑zero entries: {H.nnz}")
        print("  On‑site energies (eV):")
        for el, vals in onsite.items():
            print(
                f"    {el:>2}:  s={vals[0]:6.3f},  px={vals[1]:6.3f}, "
                f"py={vals[2]:6.3f},  pz={vals[3]:6.3f}"
            )

    if H_format == "full":
        H = H.todense()

    return H, orbital_map


@as_function_node
def OnsiteData():
    onsite_elements = {
        "Pt": (-5.5, -2.5, -2.5, -2.5),
        "O": (-7.0, -3.0, -3.0, -3.0),
        "H": (-13.6, -5.0, -5.0, -5.0),
    }
    return onsite_elements


@as_function_node
def Hopping(scale: float = 1):
    """Return hopping parameters optionally scaled.

    The original hopping matrix elements are defined for a unit scale.
    Multiplying each element by ``scale`` allows the user to uniformly
    adjust the strength of all hopping terms.
    """
    base_hopping_elements = {
        _pair_key("Pt", "Pt"): (-2.8, 3.2, 3.8, -1.1),
        _pair_key("Pt", "O"): (-2.0, 2.6, 3.0, -0.9),
        _pair_key("Pt", "H"): (-1.8, 2.4, 2.8, -0.8),
        _pair_key("O", "O"): (-1.1, 2.0, 2.4, -0.7),
        _pair_key("O", "H"): (-1.0, 1.9, 2.3, -0.6),
        _pair_key("H", "H"): (-0.9, 1.8, 2.2, -0.5),
    }

    # Apply scaling factor to each tuple of hopping parameters
    hopping_elements = {
        pair: tuple(val * scale for val in params)
        for pair, params in base_hopping_elements.items()
    }
    return hopping_elements


@as_function_node
def FermiOccupations(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    electronic_temperature: float,
    n_electrons: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Fermi occupations and orbital‑resolved DOS.

    Parameters
    ----------
    eigenvalues
        1‑D array of eigenenergies (eV).
    eigenvectors
        2‑D array where each column is the eigenvector corresponding to the
        eigenvalue at the same index. Shape ``(n_orb, n_state)``.
    electronic_temperature
        Electronic temperature in Kelvin. A value of ``0`` yields a step
        occupation at the Fermi level.
    n_electrons
        Total number of electrons that should be occupied.

    Returns
    -------
    occupations
        1‑D array of Fermi‑Dirac occupation numbers for each eigenstate.
    orbital_dos
        2‑D array of the orbital‑resolved density of states. For each orbital
        ``i`` the contribution of state ``j`` is ``|psi_{ij}|^2 * f_j`` where
        ``f_j`` is the occupation of state ``j``.
    """
    # Ensure proper numpy arrays
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=complex)

    mu_conv = 1e-10

    # -----------------------------------------------------------------
    # 1. Determine chemical potential (mu) such that sum_i f_i = n_electrons
    # -----------------------------------------------------------------
    if electronic_temperature <= 0:
        # Zero‑temperature: occupy the lowest‑energy states
        sorted_idx = np.argsort(eigenvalues)
        occupations = np.zeros_like(eigenvalues)
        occupations[sorted_idx[: int(round(n_electrons))]] = 1.0
        mu = eigenvalues[sorted_idx[int(round(n_electrons)) - 1]]
    else:
        # Use Boltzmann constant in eV/K
        k_B = 8.617333262145e-5  # eV/K
        beta = 1.0 / (k_B * electronic_temperature)

        # Initial bounds for mu
        mu_low = eigenvalues.min() - 10.0 * k_B * electronic_temperature
        mu_high = eigenvalues.max() + 10.0 * k_B * electronic_temperature

        def occupation_sum(mu):
            f = 2.0 / (1.0 + np.exp(beta * (eigenvalues - mu)))
            return f.sum()

        # Bisection to find mu
        for i in range(100):
            mu_mid = 0.5 * (mu_low + mu_high)
            s = occupation_sum(mu_mid)
            if s < n_electrons:
                mu_low = mu_mid
            else:
                mu_high = mu_mid
            if mu_high - mu_low < mu_conv:
                break
        mu = 0.5 * (mu_low + mu_high)
        occupations = 2.0 / (1.0 + np.exp(beta * (eigenvalues - mu)))

    # -----------------------------------------------------------------
    # 2. Orbital‑resolved DOS: |psi|^2 weighted by occupations
    # -----------------------------------------------------------------
    prob = np.abs(eigenvectors) ** 2  # shape (n_orb, n_state)
    orbital_dos = prob * occupations  # broadcasting over states
    e_Fermi = mu

    return occupations, orbital_dos, e_Fermi


@as_function_node
def AtomCharges(orbital_dos: np.ndarray) -> np.ndarray:
    """Compute the net charge (electron count) on each atom.

    The ``orbital_dos`` array returned by :func:`FermiOccupations` already
    contains the orbital‑resolved density of states weighted by the Fermi
    occupations. Summing over the state dimension yields the electron
    population per orbital. Each atom contributes four orbitals (s, px, py,
    pz). The total electron count per atom is obtained by summing the four
    associated orbital populations.

    Parameters
    ----------
    orbital_dos
        2‑D array of shape ``(n_orbital, n_state)`` where each element is the
        contribution of a given orbital to the DOS. The first dimension is
        ordered by atom then orbital (i.e., ``atom * 4 + orbital``).

    Returns
    -------
    charges
        1‑D ``np.ndarray`` containing the electron population for each atom.
    """
    # Sum over states to obtain population per orbital
    orbital_population = orbital_dos.sum(axis=1)  # shape (n_orbital,)

    # Determine number of atoms assuming 4 orbitals per atom
    if orbital_population.size % 4 != 0:
        raise ValueError(
            "orbital_dos size is not a multiple of 4; cannot infer atom count"
        )
    n_atoms = orbital_population.size // 4

    # Reshape to (n_atoms, 4) and sum over the four orbitals per atom
    charges = orbital_population.reshape(n_atoms, 4).sum(axis=1)
    return charges


@as_function_node
def ChargeEquilibration(
    charges: np.ndarray, target_total_charge: float = 0.0
) -> np.ndarray:
    """Uniformly shift atomic charges to achieve a target total charge.

    In many tight‑binding workflows the raw atomic charges (electron
    populations) obtained from the density of states do not sum exactly to the
    desired total charge (often zero for a neutral system). This helper node
    applies a simple charge‑equilibration scheme: it subtracts the same amount
    from every atom such that the sum of the returned charges equals
    ``target_total_charge``.

    Parameters
    ----------
    charges
        1‑D array containing the per‑atom electron counts (or charges).
    target_total_charge
        Desired total charge of the system. The default ``0.0`` corresponds to
        a neutral system.

    Returns
    -------
    equilibrated_charges
        Adjusted per‑atom charges whose sum equals ``target_total_charge``.
    """
    charges = np.asarray(charges, dtype=float)
    current_total = charges.sum()
    if np.isclose(current_total, target_total_charge):
        return charges
    delta = (current_total - target_total_charge) / charges.size
    equilibrated_charges = charges - delta
    return equilibrated_charges


@as_function_node
def ShiftDiagonal(
    eps0: np.ndarray, charges: np.ndarray, chi: np.ndarray, U: np.ndarray
) -> np.ndarray:
    """Compute shifted diagonal on‑site elements with atom‑specific coefficients.

    In charge‑equilibrated tight‑binding the diagonal (on‑site) energy ``eps0``
    for each orbital is modified by the atomic charge ``q`` according to

    ``eps = eps0 + chi_i * q_i + U_i * q_i**2``,

    where ``chi_i`` and ``U_i`` are linear and quadratic coefficients specific
    to atom *i*.

    Parameters
    ----------
    eps0
        Base diagonal elements (eV). Shape ``(n_orb,)`` where each entry
        corresponds to an orbital (ordered as in ``tbHamilton``: atom * 4 +
        orbital).
    charges
        Per‑atom charge (electron count) array of shape ``(n_atoms,)``.
    chi
        Linear coefficient array (electronic polarizability) with length
        ``n_atoms``. Units: eV per charge.
    U
        Quadratic coefficient array (electronic stiffness) with length
        ``n_atoms``. Units: eV per charge².

    Returns
    -------
    np.ndarray
        Shifted diagonal elements with the same shape as ``eps0``.
    """
    eps0 = np.asarray(eps0, dtype=float)
    charges = np.asarray(charges, dtype=float)
    chi = np.asarray(chi, dtype=float)
    U = np.asarray(U, dtype=float)

    if eps0.size % 4 != 0:
        raise ValueError("eps0 size must be a multiple of 4 (4 orbitals per atom)")
    n_atoms = eps0.size // 4
    if charges.size != n_atoms:
        raise ValueError(
            "Length of charges does not match number of atoms inferred from eps0"
        )
    if chi.shape != (n_atoms,):
        raise ValueError("chi must be a vector of length equal to the number of atoms")
    if U.shape != (n_atoms,):
        raise ValueError("U must be a vector of length equal to the number of atoms")

    # Expand per‑atom quantities to per‑orbital arrays (repeat each value 4 times)
    charge_per_orb = np.repeat(charges, 4)
    chi_per_orb = np.repeat(chi, 4)
    U_per_orb = np.repeat(U, 4)

    shifted = eps0 + chi_per_orb * charge_per_orb + U_per_orb * charge_per_orb**2
    return shifted


@as_function_node(["chi", "U"])
def ChiUData(
    atoms: Atoms,
    chi_params: Dict[str, float] | None = None,
    U_params: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate atom‑specific ``chi`` and ``U`` arrays from the atomic structure.

    The function mirrors the style of :func:`OnsiteData` by providing a simple
    lookup table for species‑dependent parameters. Users can supply custom
    dictionaries ``chi_params`` and ``U_params`` that map element symbols to the
    desired coefficients. If a dictionary is omitted, a minimal default set
    (covering the elements currently used in the TB model) is employed.

    Parameters
    ----------
    atoms
        ASE ``Atoms`` object describing the system.
    chi_params
        Optional mapping ``element -> chi`` (linear polarizability). If ``None``
        a default table is used.
    U_params
        Optional mapping ``element -> U`` (quadratic stiffness). If ``None``
        a default table is used.

    Returns
    -------
    chi
        1‑D ``np.ndarray`` of length ``n_atoms`` containing the ``chi`` value for
        each atom.
    U
        1‑D ``np.ndarray`` of length ``n_atoms`` containing the ``U`` value for
        each atom.
    """
    # Default parameter tables – extend as needed for additional species
    default_chi = {
        "Pt": 0.5,
        "O": 0.3,
        "H": 0.1,
    }
    default_U = {
        "Pt": 5.0,
        "O": 3.0,
        "H": 2.0,
    }

    chi_lookup = chi_params if chi_params is not None else default_chi
    U_lookup = U_params if U_params is not None else default_U

    chi_list: List[float] = []
    U_list: List[float] = []
    for atom in atoms:
        symbol = atom.symbol
        if symbol not in chi_lookup:
            raise KeyError(f"chi parameter for element '{symbol}' not provided")
        if symbol not in U_lookup:
            raise KeyError(f"U parameter for element '{symbol}' not provided")
        chi_list.append(chi_lookup[symbol])
        U_list.append(U_lookup[symbol])

    return np.asarray(chi_list, dtype=float), np.asarray(U_list, dtype=float)


@as_function_node("chi_dict")
def ChiData(
    chi_Pt: float = 0.5, chi_O: float = 0.3, chi_H: float = 0.1
) -> Dict[str, float]:
    """Return a dictionary of ``chi`` parameters for supported elements.

    This helper node mirrors the style of :func:`OnsiteData` and provides a
    convenient way to construct the ``chi_params`` dictionary required by
    :func:`ChiUData`. The function accepts individual ``chi`` values for each
    element used in the TB model (Pt, O, H) and returns a mapping suitable for
    direct use as the ``chi_params`` argument.

    Parameters
    ----------
    chi_Pt
        Linear polarizability for platinum atoms.
    chi_O
        Linear polarizability for oxygen atoms.
    chi_H
        Linear polarizability for hydrogen atoms.

    Returns
    -------
    Dict[str, float]
        Mapping from element symbol to the provided ``chi`` value.
    """
    return {
        "Pt": chi_Pt,
        "O": chi_O,
        "H": chi_H,
    }


@as_function_node("u_dict")
def UData(U_Pt: float = 5.0, U_O: float = 3.0, U_H: float = 2.0) -> Dict[str, float]:
    """Return a dictionary of ``U`` parameters for supported elements.

    Mirrors :func:`ChiData` but for the quadratic stiffness coefficients ``U``.
    Provides a convenient way to construct the ``U_params`` dictionary required
    by :func:`ChiUData`.

    Parameters
    ----------
    U_Pt
        Quadratic stiffness for platinum atoms.
    U_O
        Quadratic stiffness for oxygen atoms.
    U_H
        Quadratic stiffness for hydrogen atoms.

    Returns
    -------
    Dict[str, float]
        Mapping from element symbol to the provided ``U`` value.
    """
    return {
        "Pt": U_Pt,
        "O": U_O,
        "H": U_H,
    }


# --------------------------------------------------------------
# Compatibility shim for ASE valence‑electron data
# --------------------------------------------------------------
try:
    # Newer ASE versions (≥ 3.22) provide this dictionary.
    from ase.data import valence_electrons as _VALENCE_ELECTRONS  # type: ignore
except Exception:  # pragma: no cover
    # Very small fallback that is sufficient for the elements that appear
    # in the present TB parametrisation.  Keys are atomic numbers.
    _VALENCE_ELECTRONS = {
        1: 1,  # H
        8: 6,  # O (group 16 → 6 valence electrons)
        78: 4,  # Pt (group 10 → 10 valence electrons) (for sp only Hamilton))
    }

    # expose the name used later in the code
    valence_electrons = _VALENCE_ELECTRONS  # noqa: N816
else:
    # If the import succeeded we just re‑export the name.
    valence_electrons = _VALENCE_ELECTRONS  # noqa: N816


# --------------------------------------------------------------
# NEW NODE: initial charge density generator (robust)
# --------------------------------------------------------------
@as_function_node
def InitialChargeDensity(atoms: Atoms, total_charge: float = 0.0) -> np.ndarray:
    """Return a simple first‑guess per‑atom charge (electron count).

    The default guess is the neutral valence‑electron count taken from
    ``ase.data.valence_electrons`` (or from a minimal fallback table if that
    dictionary is not available).  If a non‑zero ``total_charge`` is requested the
    function distributes the excess/deficit uniformly over all atoms so that the
    returned array satisfies

    ``sum(charges) = Σ(valence_electrons) – total_charge``.

    This array can be fed directly into :func:`tbHamilton` via its ``charges``
    argument.

    Parameters
    ----------
    atoms
        ASE ``Atoms`` object describing the system.
    total_charge
        Desired net charge of the whole system (in elementary‑charge units;
        ``+1`` means one electron missing, ``‑1`` means one extra electron).

    Returns
    -------
    np.ndarray
        1‑D array of length ``len(atoms)`` containing the initial electron
        population per atom.
    """
    # --------------------------------------------------------------
    # 1. Neutral valence‑electron count per atom
    # --------------------------------------------------------------
    # ``valence_electrons`` is a dict keyed by atomic number (Z).
    # If an element is missing we raise a clear error – the user can extend the
    # fallback dictionary above or supply a custom charge density.
    neutral = np.array(
        [valence_electrons.get(atom.number, None) for atom in atoms],
        dtype=float,
    )
    if None in neutral:
        missing = [atom.symbol for atom, v in zip(atoms, neutral) if v is None]
        raise KeyError(
            f"Valence‑electron information missing for element(s) "
            f"{missing}.  Upgrade ASE or extend the fallback "
            f"`valence_electrons` dictionary in the source file."
        )

    # --------------------------------------------------------------
    # 2. Uniform shift to enforce the requested total charge
    # --------------------------------------------------------------
    if total_charge != 0.0:
        # Total electrons in the neutral reference
        neutral_total = neutral.sum()
        # Target electron number = neutral_total – total_charge
        target_total = neutral_total - total_charge
        # Uniform shift per atom (the same as total_charge / N)
        delta = (neutral_total - target_total) / len(atoms)
        neutral -= delta

    return neutral


# ---------------------------------------------------------------
# NEW NODE: self‑consistent charge (SCC) SCF loop
# ---------------------------------------------------------------
@as_function_node
def SCC_SCF(
    atoms: Atoms,
    rcut: float = 5.0,
    onsite: Dict[str, Tuple[float, float, float, float]] | None = None,
    hoppings: Dict[Tuple[str, str], Tuple[float, float, float, float]] | None = None,
    scaling: Literal["harrison", "none"] = "harrison",
    H_format: Literal["sparse", "full"] = "full",
    max_iter: int = 20,
    mixing: float = 0.3,
    mixing_scheme: Literal["linear", "anderson", "broyden"] = "linear",
    tol: float = 1e-6,
    electronic_temperature: float = 300.0,
    total_charge: float = 0.0,
    verbose: bool = False,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, list, float]:
    """
    Perform a simple self‑consistent charge (SCC) SCF loop.

    The routine repeatedly

    1. builds the tight‑binding Hamiltonian with the current charge density,
    2. diagonalises it,
    3. evaluates Fermi occupations,
    4. extracts a new per‑atom charge from the orbital‑resolved DOS,
    5. enforces the global charge constraint,
    6. mixes the new charge with the old one,

    until the charge density converges (or ``max_iter`` is reached).

    Parameters
    ----------
    atoms, rcut, onsite, hoppings, scaling, H_format, verbose
        Same arguments as :func:`tbHamilton`.
    max_iter
        Maximum number of SCF iterations.
    mixing
        Linear mixing factor ``0 < mixing ≤ 1``.
        The updated charge density is

        ``q = (1‑mixing) * q_old + mixing * q_new``.
        Typical values are 0.2–0.5.
    tol
        Convergence tolerance for the charge density (absolute difference).
    electronic_temperature
        Temperature (K) used in :func:`FermiOccupations`.
    total_charge
        Desired net charge of the whole system (in elementary‑charge units).
        ``0.0`` corresponds to a neutral system.
    verbose
        If ``True`` prints iteration information.

    Returns
    -------
    H
        Final Hamiltonian (format determined by ``H_format``).
    q
        Converged per‑atom electron population (1‑D array, length ``len(atoms)``).
    """
    # -----------------------------------------------------------
    # 1. Initial charge guess (neutral valence distribution shifted by total_charge)
    # -----------------------------------------------------------
    q = InitialChargeDensity(atoms, total_charge=total_charge).run()
    total_electrons = q.sum()  # keep this number constant

    # ---------------------------------------------------------------------
    # 2. Mixing helpers (Anderson / Broyden)
    # ---------------------------------------------------------------------
    # History buffers for Anderson mixing (DIIS)
    q_history: list[np.ndarray] = []
    r_history: list[np.ndarray] = []  # residual = q - q_new
    # Inverse Jacobian for Broyden (initially identity)
    J_inv = np.eye(len(q)) if mixing_scheme == "broyden" else None
    prev_residual = None

    diff_list = []
    for it in range(max_iter):
        # -------------------------------------------------------
        # 2. Build Hamiltonian with the current charge density
        # -------------------------------------------------------
        H, _ = tbHamilton(
            atoms,
            rcut=rcut,
            onsite=onsite,
            hoppings=hoppings,
            scaling=scaling,
            H_format=H_format,
            verbose=verbose,
            charges=q,
        ).run()  # <-- charge‑dependent diagonal

        # -------------------------------------------------------
        # 3. Diagonalise (dense matrix required for eigh)
        # -------------------------------------------------------
        if H_format == "sparse":
            H_dense = np.asarray(H.todense())
        else:
            H_dense = H

        eps, vecs = np.linalg.eigh(H_dense)

        # -------------------------------------------------------
        # 4. Fermi occupations (finite‑temperature smearing)
        # -------------------------------------------------------
        occ, dos, e_Fermi = FermiOccupations(
            eps,
            vecs,
            electronic_temperature=electronic_temperature,
            n_electrons=total_electrons,
        ).run()

        # -------------------------------------------------------
        # 5. Extract new atomic charges from the DOS
        # -------------------------------------------------------
        q_new = AtomCharges(dos).run()
        # print("scf-q_new: ", q_new, q_new.sum())

        # -------------------------------------------------------
        # 6. Enforce the global charge constraint
        # -------------------------------------------------------
        # q_new = ChargeEquilibration(q_new, target_total_charge=total_electrons).run()
        # print("scf-q_new2: ", q_new, q_new.sum())

        # -------------------------------------------------------
        # 7. Check convergence (pre‑mixing)
        # -------------------------------------------------------
        residual = q - q_new  # residual that should vanish
        diff = np.linalg.norm(residual)
        if diff < tol:
            if verbose:
                print(f"SCC converged after {it+1} iteration(s).")
            q = q_new
            break

        # -------------------------------------------------------
        # 8. Mixing of the charge density according to the selected scheme
        # -------------------------------------------------------
        if mixing_scheme == "linear":
            # Simple linear mixing (original implementation)
            q = (1.0 - mixing) * q + mixing * q_new
        elif mixing_scheme == "anderson":
            # Store history (max length 5)
            q_history.append(q.copy())
            r_history.append(residual.copy())
            if len(q_history) > 10:
                q_history.pop(0)
                r_history.pop(0)
            # Perform DIIS mixing if we have at least 2 histories
            if len(r_history) >= 2:
                m = len(r_history)
                # Build B matrix (inner products of residuals)
                B = np.empty((m, m))
                for i in range(m):
                    for j in range(m):
                        B[i, j] = np.dot(r_history[i], r_history[j])
                # Solve for coefficients c with constraint sum(c)=1
                # Augment B and RHS to enforce the constraint
                A = np.empty((m + 1, m + 1))
                A[:m, :m] = B
                A[:m, m] = 1.0
                A[m, :m] = 1.0
                A[m, m] = 0.0
                rhs = np.zeros(m + 1)
                rhs[m] = 1.0
                coeff = np.linalg.solve(A, rhs)[:m]
                print("Anderson mixing: ", coeff)
                # New charge = linear combination of past charges
                q = sum(c * qh for c, qh in zip(coeff, q_history))
            else:
                # Fallback to simple linear mixing for the first step
                q = (1.0 - mixing) * q + mixing * q_new
        elif mixing_scheme == "broyden":
            # Broyden's second method (updates inverse Jacobian)
            if J_inv is None:
                J_inv = np.eye(len(q))
            # Solve for delta_q using current inverse Jacobian
            delta_q = -J_inv @ residual
            q = q + delta_q
            # Update inverse Jacobian if we have a previous residual
            if prev_residual is not None:
                delta_r = residual - prev_residual
                dq = delta_q
                # Broyden update for inverse Jacobian
                J_inv += np.outer((dq - J_inv @ delta_r), dq) / np.dot(dq, dq)
            prev_residual = residual.copy()
        else:
            raise ValueError(f"Unsupported mixing_scheme: {mixing_scheme}")

        diff = np.linalg.norm(q - q_new)
        diff_list.append(diff)
        if verbose:
            print(
                f"SCC iteration {it+1:2d}: charge change = {diff:.3e} (scheme={mixing_scheme})"
            )

    else:
        print(f"Warning: SCC did not converge within {max_iter} iterations.")

    # -----------------------------------------------------------
    # 9. Final Hamiltonian (built with the converged charge density)
    # -----------------------------------------------------------
    H_final, _ = tbHamilton(
        atoms,
        rcut=rcut,
        onsite=onsite,
        hoppings=hoppings,
        scaling=scaling,
        H_format=H_format,
        verbose=verbose,
        charges=q,
    ).run()

    print("scf iterations: ", it, max_iter)

    return H_final, q, eps, diff_list, e_Fermi


@as_function_node
def ShiftAtom(
    structure: Atoms,
    species: Optional[str] = None,
    atomic_index: int = 0,
    x: float = 0,
    y: float = 0,
    z: float = 0,
):
    structure.positions[atomic_index] += np.array([x, y, z])
    if species is not None:
        structure.symbols[atomic_index] = species
    return structure


@as_function_node
def PlotDos(
    x: Optional[list | np.ndarray],
    bins: int = 50,
    e_Fermi: Optional[float] = None,
    color: str = "b",
):
    """
    Plot a histogram.
    """
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.hist(x, bins=bins, color=color)
    if e_Fermi is not None:
        ax.axvline(e_Fermi, color="k", linestyle="--")

    return fig
