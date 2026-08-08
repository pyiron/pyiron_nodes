"""
pyiron_nodes/atomistic/structure/build_gb.py
────────────────────────────────────────────
Grain-boundary (GB) builder nodes based on a pure-Python CSL engine.

Two public nodes are provided:

GrainBoundaryOptions(sigma, crystalstructure, a, [c], ...)
    Enumerate all geometrically distinct coincidence-site-lattice (CSL)
    grain-boundary configurations for a target Σ value and return a
    ranked list of option dicts (sorted by estimated atom count).

BuildGrainBoundary(options, index, symbol, ...)
    Select one option from the list returned by GrainBoundaryOptions and
    construct the full bicrystal supercell, returning a picklable
    OutputAtoms instance ready for further aiflow processing.
"""

from __future__ import annotations

import numpy as np
from itertools import product, combinations
from math import gcd
from functools import reduce
from typing import Literal, Optional

from core import as_function_node
from pyiron_nodes.atomistic.structure._atoms import OutputAtoms, _ase_to_data


# =====================================================================
# Internal CSL / geometry engine (private helpers)
# =====================================================================

def _cubic_lattice(a: float = 1.0) -> np.ndarray:
    return a * np.eye(3)


def _hexagonal_lattice(a: float = 1.0, c: float = 1.633) -> np.ndarray:
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([-a / 2, a * np.sqrt(3) / 2, 0.0])
    a3 = np.array([0.0, 0.0, c])
    return np.column_stack([a1, a2, a3])


def _fcc_primitive_lattice(a: float = 1.0) -> np.ndarray:
    # columns = FCC primitive vectors: (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)
    return a / 2 * np.column_stack([[0, 1, 1], [1, 0, 1], [1, 1, 0]])


def _bcc_primitive_lattice(a: float = 1.0) -> np.ndarray:
    # columns = BCC primitive vectors: (-a/2,a/2,a/2), (a/2,-a/2,a/2), (a/2,a/2,-a/2)
    return a / 2 * np.column_stack([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])


_STRUCTURE_BASES = {
    'sc':      [(0, 0, 0)],
    'bcc':     [(0, 0, 0)],                       # 1-atom BCC primitive cell
    'fcc':     [(0, 0, 0)],                       # 1-atom FCC primitive cell
    'hcp':     [(0, 0, 0), (1 / 3, 2 / 3, 1 / 2)],
    'diamond': [(0, 0, 0), (0.25, 0.25, 0.25)],  # 2-atom FCC primitive cell;
                                                  # Lm_prim @ [1/4,1/4,1/4] = a*(1/4,1/4,1/4) ✓
}
_CUBIC_STRUCTURES = {'sc', 'bcc', 'fcc', 'diamond'}
_HEX_STRUCTURES = {'hcp'}


def _rotation_matrix(axis, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(angle_deg)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _hex_dir_to_cart(Lm: np.ndarray, u, v, w) -> np.ndarray:
    return Lm @ np.array([u, v, w], dtype=float)


# ── CSL search ────────────────────────────────────────────────────────

def _csl_from_rotation(Lm, R, target_sigma=None, max_search=None, tol=1e-4, pool_size=100):
    M = np.linalg.inv(Lm) @ R @ Lm
    if max_search is None:
        max_search = 40 if target_sigma is None else int(3 * np.sqrt(target_sigma)) + 6
    candidates = []
    rng = range(-max_search, max_search + 1)
    for v in product(rng, repeat=3):
        if v == (0, 0, 0):
            continue
        mv = M @ np.array(v, dtype=float)
        if np.all(np.abs(mv - np.round(mv)) < tol):
            candidates.append(np.array(v, dtype=int))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x @ x)
    pool = candidates[:pool_size]
    best = None
    for combo in combinations(range(len(pool)), 3):
        v1, v2, v3 = pool[combo[0]], pool[combo[1]], pool[combo[2]]
        det = int(round(np.linalg.det(np.array([v1, v2, v3], dtype=float))))
        if det == 0:
            continue
        sigma = abs(det)
        if target_sigma is not None and sigma == target_sigma:
            return sigma, np.array([v1, v2, v3])
        if best is None or sigma < best[0]:
            best = (sigma, np.array([v1, v2, v3]))
    if target_sigma is not None:
        return None, None
    return best if best else (None, None)


def _calc_sigma_from_quat(a, b, c, d) -> int:
    n = a * a + b * b + c * c + d * d
    sigma = n
    while sigma % 4 == 0:
        sigma //= 4
    if sigma % 2 == 0:
        sigma //= 2
    return int(sigma)


def _cubic_sigma_search(target_sigma: int, max_index: int = 8) -> list:
    results = {}
    for a, b, c, d in product(range(0, max_index + 1), repeat=4):
        if (a, b, c, d) == (0, 0, 0, 0):
            continue
        if reduce(gcd, [a, b, c, d]) > 1:
            continue
        sigma = _calc_sigma_from_quat(a, b, c, d)
        if sigma != target_sigma:
            continue
        n = a * a + b * b + c * c + d * d
        angle = 2 * np.degrees(np.arccos(np.clip(a / np.sqrt(n), -1, 1)))
        bcd = [b, c, d]
        nz = [abs(x) for x in bcd if x != 0]
        if not nz:
            continue
        g2 = reduce(gcd, nz)
        axis = tuple(x // g2 for x in bcd)
        key = (round(angle, 4), axis)
        if key not in results:
            results[key] = {'sigma': sigma, 'angle_deg': round(angle, 4), 'axis': axis}
    return sorted(results.values(), key=lambda r: (r['axis'], r['angle_deg']))


def _hexagonal_caxis_sigma_search(target_sigma: int, max_hk: int = 40) -> list:
    results = {}
    for h in range(0, max_hk + 1):
        for k in range(0, max_hk + 1):
            if h == 0 and k == 0:
                continue
            if gcd(h, k) != 1:
                continue
            n = h * h - h * k + k * k
            sigma = n
            while sigma % 3 == 0:
                sigma //= 3
            if sigma != target_sigma:
                continue
            z = complex(h - 0.5 * k, k * np.sqrt(3) / 2.0)
            angle = np.degrees(2 * np.angle(z)) % 60.0
            if angle < 1e-6:
                continue
            key = round(angle, 4)
            if key not in results:
                results[key] = {'sigma': sigma, 'angle_deg': round(angle, 4),
                                 'axis': (0, 0, 0, 1)}
    return sorted(results.values(), key=lambda r: r['angle_deg'])


def _rational_residual(M, q):
    qM = q * M
    return np.max(np.abs(qM - np.round(qM)))


def _general_axis_sigma_search(Lm, axis, target_sigma, angle_step=0.05,
                                coarse_tol=0.06, max_denom=None, verify_tol=1e-4) -> list:
    try:
        from scipy.optimize import minimize_scalar
        have_scipy = True
    except ImportError:
        have_scipy = False

    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    if max_denom is None:
        max_denom = int(2.5 * np.sqrt(target_sigma)) + 8

    thetas = np.arange(angle_step, 180.0, angle_step)
    hits = []
    for theta in thetas:
        R = _rotation_matrix(axis, theta)
        M = np.linalg.inv(Lm) @ R @ Lm
        for q in range(1, max_denom + 1):
            if _rational_residual(M, q) < coarse_tol:
                hits.append(theta)
                break

    if not hits:
        return []

    clusters = [[hits[0]]]
    for t in hits[1:]:
        if t - clusters[-1][-1] <= 3 * angle_step:
            clusters[-1].append(t)
        else:
            clusters.append([t])
    seeds = [np.mean(cl) for cl in clusters]

    def residual_fn(theta):
        R = _rotation_matrix(axis, theta)
        M = np.linalg.inv(Lm) @ R @ Lm
        return min(_rational_residual(M, q) for q in range(1, max_denom + 1))

    found = {}
    for seed in seeds:
        lo, hi = max(seed - 3 * angle_step, 1e-6), seed + 3 * angle_step
        if have_scipy:
            res = minimize_scalar(residual_fn, bounds=(lo, hi), method='bounded',
                                   options={'xatol': 1e-7})
            refined = res.x
        else:
            for _ in range(60):
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                if residual_fn(m1) < residual_fn(m2):
                    hi = m2
                else:
                    lo = m1
            refined = (lo + hi) / 2

        R = _rotation_matrix(axis, refined)
        sigma, basis = _csl_from_rotation(Lm, R, target_sigma=target_sigma, tol=verify_tol)
        if sigma == target_sigma:
            key = round(refined, 3)
            if key not in found:
                found[key] = {'sigma': sigma, 'angle_deg': round(refined, 4),
                              'axis': tuple(np.round(axis, 4)), 'basis': basis}

    return sorted(found.values(), key=lambda r: r['angle_deg'])


# ── Geometry / supercell helpers ──────────────────────────────────────

def _short_inplane_vectors(csl_cart_rows, normal, max_coeff=6, tol=1e-6):
    normal = normal / np.linalg.norm(normal)
    cands = []
    rng = range(-max_coeff, max_coeff + 1)
    for c in product(rng, repeat=3):
        if c == (0, 0, 0):
            continue
        v = np.array(c) @ csl_cart_rows
        if abs(np.dot(v, normal)) < tol:
            cands.append(v)
    if len(cands) < 2:
        return None
    cands.sort(key=lambda v: v @ v)
    v1 = cands[0]
    v2 = next((v for v in cands[1:] if np.linalg.norm(np.cross(v1, v)) > tol), None)
    return (v1, v2) if v2 is not None else None


def _short_stacking_vector(Lm, normal, in_plane_vecs, max_coeff=6, tol=1e-6):
    normal = normal / np.linalg.norm(normal)
    best, best_score = None, None
    rng = range(-max_coeff, max_coeff + 1)
    for c in product(rng, repeat=3):
        if c == (0, 0, 0):
            continue
        v = Lm @ np.array(c, dtype=float)
        proj = np.dot(v, normal)
        if abs(proj) < tol:
            continue
        score = (v @ v) / proj ** 2
        if best is None or score < best_score - 1e-10 or (
                abs(score - best_score) < 1e-10 and v @ v < best @ best - 1e-10):
            best, best_score = v, score
    v1, v2 = in_plane_vecs
    A = np.array([v1, v2]).T
    coeff, *_ = np.linalg.lstsq(A, best, rcond=None)
    result = best - A @ np.round(coeff)
    # The in-plane correction preserves the normal projection, so flipping here is safe.
    # Ensures both grains always get stacking vectors pointing in the same direction,
    # preventing c_total = n1*c1 + n2*c2 from cancelling to near-zero.
    if np.dot(result, normal) < 0:
        result = -result
    return result


def _lattice_points_in_box(Lm, basis_frac, box_vectors, tol=1e-6):
    inv_lat = np.linalg.inv(Lm)
    corners = np.array([box_vectors @ np.array(bc, dtype=float)
                         for bc in product([0, 1], repeat=3)])
    frac_corners = (inv_lat @ corners.T).T
    lo = np.floor(frac_corners.min(axis=0)).astype(int) - 2
    hi = np.ceil(frac_corners.max(axis=0)).astype(int) + 2
    inv_box = np.linalg.inv(box_vectors)

    positions = []
    for i in range(lo[0], hi[0] + 1):
        for j in range(lo[1], hi[1] + 1):
            for k in range(lo[2], hi[2] + 1):
                base = Lm @ np.array([i, j, k], dtype=float)
                for b in basis_frac:
                    pos = base + Lm @ np.array(b, dtype=float)
                    fb = inv_box @ pos
                    if np.all(fb > -tol) and np.all(fb < 1 - tol):
                        positions.append(pos)
    return np.array(positions)


def _estimate_supercell(Lm, atoms_per_cell, R, basis, gb_normal,
                         min_slab_thickness=15.0):
    csl_cart_rows = basis @ Lm.T
    inplane = _short_inplane_vectors(csl_cart_rows, gb_normal)
    if inplane is None:
        return None
    v1, v2 = inplane
    area = np.linalg.norm(np.cross(v1, v2))
    Lm2 = R @ Lm
    c1 = _short_stacking_vector(Lm, gb_normal, (v1, v2))
    c2 = _short_stacking_vector(Lm2, gb_normal, (v1, v2))
    nhat = gb_normal / np.linalg.norm(gb_normal)
    proj1, proj2 = abs(np.dot(c1, nhat)), abs(np.dot(c2, nhat))
    n1 = max(1, int(np.ceil(min_slab_thickness / proj1)))
    n2 = max(1, int(np.ceil(min_slab_thickness / proj2)))
    vol_prim = abs(np.linalg.det(Lm))
    vol1 = abs(np.linalg.det(np.array([v1, v2, n1 * c1])))
    vol2 = abs(np.linalg.det(np.array([v1, v2, n2 * c2])))
    n_atoms_1 = round(vol1 * atoms_per_cell / vol_prim)
    n_atoms_2 = round(vol2 * atoms_per_cell / vol_prim)
    return {
        'inplane_area': area,
        'thickness_grain1': n1 * proj1,
        'thickness_grain2': n2 * proj2,
        'n1': n1, 'n2': n2, 'c1': c1, 'c2': c2, 'v1': v1, 'v2': v2,
        'total_atoms_estimate': int(n_atoms_1 + n_atoms_2),
    }


def _default_gb_planes(Lm, axis_cart):
    normals = [('twist', axis_cart)]
    ref_dirs = [Lm[:, 0], Lm[:, 1], Lm[:, 2],
                Lm[:, 0] + Lm[:, 1], Lm[:, 0] - Lm[:, 1]]
    seen = set()
    for d in ref_dirs:
        n = np.cross(axis_cart, d)
        if np.linalg.norm(n) < 1e-8:
            continue
        n_hat = n / np.linalg.norm(n)
        key = tuple(np.round(n_hat, 3))
        if key in seen or tuple(np.round(-n_hat, 3)) in seen:
            continue
        seen.add(key)
        normals.append(('tilt', n))
    return normals


# =====================================================================
# Public function nodes
# =====================================================================

@as_function_node("options")
def GrainBoundaryOptions(
    sigma: int,
    crystalstructure: Literal["sc", "bcc", "fcc", "diamond", "hcp"] = "fcc",
    a: float = 1.0,
    c: Optional[float] = None,
    min_slab_thickness: float = 15.0,
    max_index: int = 8,
    angle_step: float = 0.05,
) -> object:
    """
    Enumerate coincidence-site-lattice (CSL) grain-boundary configurations.

    **Scientific purpose**
    For a given target Σ value and crystal structure, find all
    geometrically distinct tilt and twist grain-boundary orientations.
    Results are returned as a pandas DataFrame sorted by estimated supercell
    size (smallest first) and can be passed directly to ``BuildGrainBoundary``.

    **Required inputs**
    - ``sigma``: Target Σ (coincidence index), e.g. ``5`` for Σ5.
    - ``crystalstructure``: One of ``"sc"``, ``"bcc"``, ``"fcc"``,
      ``"diamond"``, ``"hcp"`` (default ``"fcc"``).
    - ``a``: Lattice constant in Å (default ``1.0``).
    - ``c``: Second lattice constant in Å — required for ``"hcp"``,
      ignored otherwise.
    - ``min_slab_thickness``: Minimum grain thickness in Å used for the
      quick atom-count estimate (default ``15.0``).
    - ``max_index``: Maximum quaternion component searched in the cubic
      exact solver (default ``8``).
    - ``angle_step``: Angular step in degrees for the hexagonal general-axis
      search (default ``0.05``).

    **Typical use-cases**
    * Screening all distinct Σ5 boundaries in FCC Al to pick the smallest
      supercell before running DFT.
    * Exploring twist vs. tilt boundaries for a given sigma in HCP Mg.

    Returns
    -------
    pandas.DataFrame
        Rows sorted by ``total_atoms_estimate`` (ascending).  Human-readable
        columns (``sigma``, ``axis``, ``angle_deg``, ``boundary_type``,
        ``total_atoms_estimate``, ``thickness_grain1``, ``thickness_grain2``)
        appear alongside internal geometry tensors (``R``, ``basis``, ``Lm``,
        ``gb_normal``, ``c1``, ``c2``, ``v1``, ``v2``) consumed by
        ``BuildGrainBoundary``.
    """
    structure = crystalstructure.lower()
    if structure in _CUBIC_STRUCTURES:
        if structure == 'sc':
            Lm = _cubic_lattice(a)
        elif structure in ('fcc', 'diamond'):
            Lm = _fcc_primitive_lattice(a)
        else:  # bcc
            Lm = _bcc_primitive_lattice(a)
        atoms_per_cell = len(_STRUCTURE_BASES[structure])
        raw = _cubic_sigma_search(sigma, max_index=max_index)
    elif structure in _HEX_STRUCTURES:
        if c is None:
            raise ValueError("HCP requires the 'c' lattice parameter.")
        Lm = _hexagonal_lattice(a, c)
        atoms_per_cell = len(_STRUCTURE_BASES[structure])
        raw = _hexagonal_caxis_sigma_search(sigma)
        for uvw in [(1, 0, 0), (1, 1, 0), (2, -1, 0), (1, 0, 1), (1, 1, 1)]:
            axis_cart = _hex_dir_to_cart(Lm, *uvw)
            for f in _general_axis_sigma_search(Lm, axis_cart, sigma,
                                                 angle_step=angle_step):
                f['axis'] = uvw
                raw.append(f)
    else:
        raise ValueError(
            f"Unsupported crystalstructure '{crystalstructure}'. "
            f"Choose from: sc, bcc, fcc, diamond, hcp."
        )

    options = []
    for item in raw:
        axis_label, angle = item['axis'], item['angle_deg']
        if structure in _HEX_STRUCTURES and axis_label == (0, 0, 0, 1):
            axis_cart = np.array([0.0, 0.0, 1.0])
        elif structure in _HEX_STRUCTURES:
            axis_cart = _hex_dir_to_cart(Lm, *axis_label)
        else:
            axis_cart = np.array(axis_label, dtype=float)

        R = _rotation_matrix(axis_cart, angle)
        sig_confirmed, basis = _csl_from_rotation(Lm, R, target_sigma=sigma)
        if sig_confirmed != sigma:
            continue

        axis_norm = axis_cart / np.linalg.norm(axis_cart)
        for plane_type, normal in _default_gb_planes(Lm, axis_norm):
            info = _estimate_supercell(Lm, atoms_per_cell, R, basis,
                                        normal, min_slab_thickness)
            if info is None:
                continue
            options.append({
                'sigma': sigma,
                'crystalstructure': structure,
                'axis': axis_label,
                'angle_deg': angle,
                'boundary_type': plane_type,
                'gb_normal': normal / np.linalg.norm(normal),
                'R': R,
                'basis': basis,
                'Lm': Lm,
                'atoms_per_cell': atoms_per_cell,
                'basis_frac': _STRUCTURE_BASES[structure],
                **info,
            })

    import pandas as pd

    df = pd.DataFrame(sorted(options, key=lambda o: o['total_atoms_estimate']))
    df = df.reset_index(drop=True)
    return df


@as_function_node("structure")
def BuildGrainBoundary(
    options: object,
    index: int = 0,
    symbol: str = "Al",
    min_slab_thickness: float = 15.0,
    vacuum: float = 0.0,
    merge_tol: float = 0.5,
) -> OutputAtoms:
    """
    Build a bicrystal supercell from a grain-boundary option.

    **Scientific purpose**
    Construct the full atomistic bicrystal from one entry of the list
    produced by ``GrainBoundaryOptions``.  Grain 1 occupies the lower half
    of the supercell; grain 2 sits on top, misoriented by the rotation
    defined in the chosen option.  Atoms closer than ``merge_tol`` (Å) at
    the boundary planes are merged to avoid unphysical overlaps.

    **Required inputs**
    - ``options``: DataFrame returned by ``GrainBoundaryOptions``.
    - ``index``: Which entry to build (default ``0`` — the smallest cell).
    - ``symbol``: Chemical symbol of the element (e.g. ``"Al"``).
    - ``min_slab_thickness``: Minimum grain thickness in Å (default
      ``15.0``); overrides the thickness encoded in the option.
    - ``vacuum``: Vacuum layer thickness in Å added along the stacking
      direction (default ``0.0`` — fully periodic bicrystal).
    - ``merge_tol``: Minimum inter-atomic distance below which duplicate
      boundary atoms are removed (default ``0.5`` Å).

    **Typical use-cases**
    * Building the Σ5 (210) tilt boundary in FCC Al for LAMMPS relaxation.
    * Generating a series of bicrystals across different Σ values for
      grain-boundary energy calculations.

    Returns
    -------
    OutputAtoms
        Picklable bicrystal structure compatible with aiflow storage and
        downstream calculator nodes.
    """
    from ase import Atoms

    option = options.iloc[index]
    Lm = option['Lm']
    R = option['R']
    basis = option['basis']
    normal = option['gb_normal']
    basis_frac = option['basis_frac']

    # Symmetric convention: split the misorientation equally between the two grains.
    # Extract rotation axis from R (real eigenvector with eigenvalue ≈ 1), then
    # rotate the entire frame by -θ/2 so grain 1 = R(-θ/2)·Lm, grain 2 = R(+θ/2)·Lm.
    vals, vecs = np.linalg.eig(R)
    axis_cart = vecs[:, np.argmin(np.abs(vals - 1.0))].real
    axis_cart = axis_cart / np.linalg.norm(axis_cart)
    R_sym = _rotation_matrix(axis_cart, -option['angle_deg'] / 2)

    Lm1 = R_sym @ Lm
    Lm2 = R_sym @ R @ Lm          # = R(+θ/2) @ Lm

    csl_cart_rows = (R_sym @ (basis @ Lm.T).T).T
    normal = R_sym @ normal
    v1, v2 = _short_inplane_vectors(csl_cart_rows, normal)
    c1 = _short_stacking_vector(Lm1, normal, (v1, v2))
    c2 = _short_stacking_vector(Lm2, normal, (v1, v2))

    proj1 = abs(np.dot(c1, normal))
    proj2 = abs(np.dot(c2, normal))
    n1 = max(1, int(np.ceil(min_slab_thickness / proj1)))
    n2 = max(1, int(np.ceil(min_slab_thickness / proj2)))

    c_total_check = n1 * c1 + n2 * c2
    if np.linalg.det(np.column_stack([v1, v2, c_total_check])) < 0:
        v2 = -v2

    box1 = np.column_stack([v1, v2, n1 * c1])
    box2 = np.column_stack([v1, v2, n2 * c2])

    pos1 = _lattice_points_in_box(Lm1, basis_frac, box1)
    pos2 = _lattice_points_in_box(Lm2, basis_frac, box2)
    pos2 = pos2 + n1 * c1

    c_total = n1 * c1 + n2 * c2
    if vacuum > 0:
        c_total = c_total + vacuum * normal

    cell_cols = np.column_stack([v1, v2, c_total])
    inv_cell = np.linalg.inv(cell_cols)

    all_pos = np.vstack([pos1, pos2])
    frac = (inv_cell @ all_pos.T).T
    frac[:, 0:2] %= 1.0
    if vacuum == 0:
        frac[:, 2] %= 1.0
    cart = (cell_cols @ frac.T).T

    keep = np.ones(len(cart), dtype=bool)
    for i in range(len(cart)):
        if not keep[i]:
            continue
        d = (np.linalg.norm(cart[i + 1:] - cart[i], axis=1)
             if i + 1 < len(cart) else np.array([]))
        close = np.where(d < merge_tol)[0] + (i + 1)
        keep[close] = False
    cart = cart[keep]

    cell_rows = cell_cols.T
    pbc = (True, True, vacuum == 0)
    atoms = Atoms(symbols=[symbol] * len(cart), positions=cart,
                  cell=cell_rows, pbc=pbc)

    # Rotate to upper-triangular standard form: a along x, b in xy-plane.
    # Required by nglview (and most visualisers) to display the unit-cell box.
    _c = np.array(atoms.cell)
    _e1 = _c[0] / np.linalg.norm(_c[0])
    _e3 = np.cross(_c[0], _c[1])
    _e3 = _e3 / np.linalg.norm(_e3)
    _e2 = np.cross(_e3, _e1)
    _Q = np.array([_e1, _e2, _e3])          # rotation: row vectors = new axes
    atoms.set_cell(_c @ _Q.T)
    atoms.set_positions(atoms.positions @ _Q.T, apply_constraint=False)

    structure = _ase_to_data(atoms)
    return structure
