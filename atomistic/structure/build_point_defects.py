import numpy as np
import pandas as pd
from ase import Atoms
from ase.geometry import find_mic

# %%
from ase.io import read
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from ase.build import bulk
import nglview as nv

# %%
# from atomistics.calculators.lammps.libcalculator import (
#     optimize_positions_and_volume_with_lammpslib,
#     optimize_positions_with_lammpslib,
#     calc_static_with_lammpslib,
# )
from structuretoolkit.visualize import plot3d
from pyiron_lammps import get_potential_by_name

from typing import Optional
from core import as_function_node

# %%
import sys

# sys.path.append("/cmmc/ptmp/pchilaka/Packages/pymatgen-analysis-defects")
# from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
from structuretoolkit.common import ase_to_pymatgen

# %% [markdown]
# # Functions
# %%
UID_KEY = "uid"


# %% [markdown]
# ## Uid handling
# %%
def ensure_uids(atoms: Atoms, uid_key: str = UID_KEY) -> Atoms:
    """Attach stable uids if missing. Does not modify existing uids."""
    if uid_key in atoms.arrays:
        return atoms
    atoms = atoms.copy()
    atoms.arrays[uid_key] = np.arange(len(atoms), dtype=int)
    return atoms


def next_uid(atoms: Atoms, uid_key: str = UID_KEY) -> int:
    """Return a fresh uid greater than all existing ones."""
    if uid_key not in atoms.arrays or len(atoms) == 0:
        return 0
    return int(np.max(atoms.arrays[uid_key])) + 1


# def append_atom_with_uid(atoms: Atoms, symbol: str, position, uid_key: str = UID_KEY) -> Atoms:
#     """
#     Append a new atom (e.g. interstitial) with a new uid.
#     Existing atoms' uids are untouched.
#     """
#     atoms = ensure_uids(atoms, uid_key=uid_key).copy()
#     new_id = next_uid(atoms, uid_key=uid_key)
#     atoms += Atoms(symbols=[symbol], positions=[position], cell=atoms.cell, pbc=atoms.pbc)
#     atoms.arrays[uid_key] = np.append(atoms.arrays[uid_key], new_id).astype(int)
#     return atoms
def append_atom_with_uid(
    atoms: Atoms, symbol: str, position, uid_key: str = "uid"
) -> Atoms:
    atoms = ensure_uids(atoms, uid_key=uid_key).copy()
    new_id = next_uid(atoms, uid_key=uid_key)
    # append atom via Atoms.append (simpler) then set uid array length exactly
    atoms.append(Atoms(symbols=[symbol], positions=[position])[0])
    # If append created/modified arrays, enforce uid length = len(atoms)
    u = atoms.arrays[uid_key]
    if u.shape[0] == len(atoms) - 1:
        atoms.arrays[uid_key] = np.append(u, new_id).astype(int)
    elif u.shape[0] == len(atoms):
        # already extended somehow; overwrite last entry to be safe
        atoms.arrays[uid_key][-1] = new_id
    else:
        raise ValueError(
            f"uid array has unexpected length {u.shape[0]} for len(atoms)={len(atoms)}"
        )
    return atoms


def uid_to_index(atoms: Atoms, uid: int, uid_key: str = UID_KEY):
    """Return current atom index for a given uid, or None if that uid no longer exists."""
    if uid_key not in atoms.arrays:
        return None
    hits = np.where(atoms.arrays[uid_key] == int(uid))[0]
    return int(hits[0]) if len(hits) else None


def element_uids(atoms: Atoms, element: str, uid_key: str = UID_KEY):
    atoms = ensure_uids(atoms, uid_key=uid_key)
    syms = np.array(atoms.get_chemical_symbols(), dtype=object)
    uids = atoms.arrays[uid_key].astype(int)
    return uids[syms == element].tolist()


def validate_atoms_arrays(atoms, uid_key="uid"):
    n = len(atoms)
    bad = {k: v.shape[0] for k, v in atoms.arrays.items() if v.shape[0] != n}
    if bad:
        raise ValueError(f"Inconsistent per-atom arrays: len(atoms)={n}, bad={bad}")


def _protected_uids_from_events(events):
    """
    Protect atoms that were explicitly created/modified by defects:
      - substitution: protect the substituted atom/site (atom_uid preferred)
      - interstitial: protect the inserted atom (atom_uid)
    """
    forbid = set()
    for ev in events or []:
        t = ev.get("type")
        if t == "substitution":
            if "atom_uid" in ev:
                forbid.add(int(ev["atom_uid"]))
            elif "site_uid" in ev:
                forbid.add(int(ev["site_uid"]))
        elif t == "interstitial":
            if "atom_uid" in ev:
                forbid.add(int(ev["atom_uid"]))
    return forbid


# %% [markdown]
# ## Counts and Tags
# %%
def element_counts(atoms: Atoms):
    out = {}
    for s in atoms.get_chemical_symbols():
        out[s] = out.get(s, 0) + 1
    return out


def element_fractions(atoms: Atoms):
    syms = atoms.get_chemical_symbols()
    n = len(syms)
    if n == 0:
        return {}
    return {s: syms.count(s) / n for s in set(syms)}


def defect_counts_from_events(events):
    """
    Count defect events by type:
      substitution -> s
      vacancy      -> v
      interstitial -> i
      antisite     -> a (optional)
    """
    c = dict(s=0, v=0, i=0, a=0)
    for ev in events or []:
        t = ev.get("type")
        if t == "substitution":
            c["s"] += 1
        elif t == "vacancy":
            c["v"] += 1
        elif t == "interstitial":
            c["i"] += 1
        elif t == "antisite":
            c["a"] += 1
    return c


def make_tag(events):
    c = defect_counts_from_events(events)
    tag = f"s{c['s']}v{c['v']}i{c['i']}"
    if c["a"] > 0:
        tag += f"a{c['a']}"
    return tag


@as_function_node
def make_pristine_reference(atoms: Atoms):
    """
    Create a pristine reference:
      - assigns uid if missing
      - returns (atoms_with_uids, pristine_positions_dict)
    """
    atoms0 = ensure_uids(atoms)
    uids = atoms0.arrays[UID_KEY].astype(int)
    pos = atoms0.get_positions()
    pristine_positions = {int(u): pos[i].tolist() for i, u in enumerate(uids)}

    return atoms0, pristine_positions


def make_config_row(
    atoms: Atoms,
    structure_id="pristine",
    parent_id=None,
    events=None,
    seed=None,
    phase_type=None,
    reference_phase=None,
    pristine_n_sites=None,
    pristine_positions=None,
):
    """
    Create a DataFrame row dict for one configuration.
    `pristine_positions` is a dict {site_uid: [x,y,z]} from the pristine reference.
    """
    atoms = ensure_uids(atoms)
    row = {}
    row["atoms"] = atoms.copy()
    row["symbol"] = atoms.get_chemical_formula()
    row["structure_id"] = structure_id
    row["parent_id"] = parent_id
    row["seed"] = seed
    row["phase_type"] = phase_type
    row["reference_phase"] = reference_phase
    row["events"] = list(events) if events is not None else []
    row["tag"] = make_tag(row["events"])
    row["n_sites_final"] = int(len(atoms))
    row["counts"] = element_counts(atoms)
    row["fractions"] = element_fractions(atoms)
    row["n_sites_pristine"] = (
        int(pristine_n_sites) if pristine_n_sites is not None else int(len(atoms))
    )
    row["pristine_positions"] = pristine_positions  # dict or None
    return row


# %% [markdown]
# ## Distances helpers
# %%
def mic_distance_between_positions(atoms: Atoms, pos_a, pos_b) -> float:
    """Minimum-image distance between two Cartesian positions using atoms cell/pbc."""
    dvec, _ = find_mic(np.array(pos_a) - np.array(pos_b), atoms.cell, pbc=atoms.pbc)
    return float(np.linalg.norm(dvec))


def pair_distance_same_element(atoms: Atoms, element: str, require_n: int = 2) -> float:
    """
    MIC distance between two atoms of the same element.
    By default requires exactly 2 atoms of that element; else NaN.
    """
    syms = atoms.get_chemical_symbols()
    idx = [i for i, s in enumerate(syms) if s == element]
    if len(idx) != int(require_n):
        return np.nan
    if require_n != 2:
        raise ValueError(
            "pair_distance_same_element currently supports require_n=2 only."
        )
    return float(atoms.get_distance(idx[0], idx[1], mic=True))


def pair_distance_by_uids(
    atoms: Atoms, uid_a: int, uid_b: int, uid_key: str = UID_KEY
) -> float:
    """MIC distance between two existing atoms identified by uid."""
    atoms = ensure_uids(atoms, uid_key=uid_key)
    ia = uid_to_index(atoms, uid_a, uid_key=uid_key)
    ib = uid_to_index(atoms, uid_b, uid_key=uid_key)
    if ia is None or ib is None:
        return np.nan
    return float(atoms.get_distance(ia, ib, mic=True))


def vacancy_site_distance_from_row(row, vac_event_idx1=0, vac_event_idx2=1) -> float:
    """Distance between two vacancy sites using stored site_pos0 in events."""
    vacs = [ev for ev in row.get("events", []) if ev.get("type") == "vacancy"]
    if len(vacs) <= max(vac_event_idx1, vac_event_idx2):
        return np.nan
    pos1 = vacs[vac_event_idx1]["site_pos0"]
    pos2 = vacs[vac_event_idx2]["site_pos0"]
    return mic_distance_between_positions(row["atoms"], pos1, pos2)


def substitution_site_distance_from_row(
    row, sub_event_idx1=0, sub_event_idx2=1
) -> float:
    """Distance between two substitution lattice sites using stored site_pos0."""
    subs = [ev for ev in row.get("events", []) if ev.get("type") == "substitution"]
    if len(subs) <= max(sub_event_idx1, sub_event_idx2):
        return np.nan
    pos1 = subs[sub_event_idx1]["site_pos0"]
    pos2 = subs[sub_event_idx2]["site_pos0"]
    return mic_distance_between_positions(row["atoms"], pos1, pos2)


def substitution_to_vacancy_site_distance_from_row(
    row, sub_event_idx=0, vac_event_idx=0
) -> float:
    """Distance between a substitution site and a vacancy site using stored site_pos0."""
    subs = [ev for ev in row.get("events", []) if ev.get("type") == "substitution"]
    vacs = [ev for ev in row.get("events", []) if ev.get("type") == "vacancy"]
    if len(subs) <= sub_event_idx or len(vacs) <= vac_event_idx:
        return np.nan
    return mic_distance_between_positions(
        row["atoms"], subs[sub_event_idx]["site_pos0"], vacs[vac_event_idx]["site_pos0"]
    )


@as_function_node("structures")
def op_substitute(
    row,
    from_element: str,
    to_element: str,
    n: int = 1,
    seed: Optional[int] = None,
    forbid_uids=None,
    protect_history: bool = False,
):
    """
    Substitute n atoms: from_element -> to_element.
    Records event with:
      - atom_uid: uid of the atom whose species was changed (still exists)
      - site_uid: lattice site identity (same as atom_uid here)
      - site_pos0: pristine site position (if provided)
    """
    rng = np.random.default_rng(seed if seed is not None else row.get("seed", 0))
    atoms = ensure_uids(row["atoms"]).copy()
    uids = atoms.arrays[UID_KEY].astype(int)
    syms = np.array(atoms.get_chemical_symbols(), dtype=object)
    forbid = set() if forbid_uids is None else set(map(int, forbid_uids))
    if protect_history:
        forbid |= _protected_uids_from_events(row.get("events", []))

    cand = [
        i
        for i in range(len(atoms))
        if syms[i] == from_element and int(uids[i]) not in forbid
    ]
    if int(n) > len(cand):
        raise ValueError(
            f"Not enough candidates to substitute {from_element}->{to_element}: need {n}, have {len(cand)}"
        )
    pick_idx = rng.choice(cand, size=int(n), replace=False)
    events_new = list(row.get("events", []))
    pristine_positions = row.get("pristine_positions", None)
    for i in pick_idx:
        atom_uid = int(uids[i])
        site_uid = atom_uid
        site_pos0 = (
            pristine_positions.get(site_uid)
            if isinstance(pristine_positions, dict)
            else atoms.positions[i].tolist()
        )
        events_new.append(
            {
                "type": "substitution",
                "from": str(from_element),
                "to": str(to_element),
                "atom_uid": atom_uid,
                "site_uid": site_uid,
                "site_pos0": site_pos0,
                "pos_at_creation": atoms.positions[i].tolist(),
            }
        )
    syms[pick_idx] = to_element
    atoms.set_chemical_symbols(syms.tolist())
    return make_config_row(
        atoms=atoms,
        structure_id=row.get("structure_id"),
        parent_id=row.get("structure_id"),
        events=events_new,
        seed=seed,
        phase_type=row.get("phase_type"),
        reference_phase=row.get("reference_phase"),
        pristine_n_sites=row.get("n_sites_pristine"),
        pristine_positions=row.get("pristine_positions"),
    )


# %% [markdown]
# ## Vacancies
# %%
def op_vacancy(
    row, vacancy_element=None, n=1, seed=None, forbid_uids=None, protect_history=False
):
    """
    Remove n atoms (vacancies). If vacancy_element is None, can remove any element.
    Records vacancy event with:
      - removed_element
      - site_uid (uid of removed atom at time of removal)
      - site_pos0 (pristine site position if provided)
    """
    rng = np.random.default_rng(seed if seed is not None else row.get("seed", 0))
    atoms = ensure_uids(row["atoms"]).copy()
    validate_atoms_arrays(atoms)
    uids = atoms.arrays[UID_KEY].astype(int)
    syms = np.array(atoms.get_chemical_symbols(), dtype=object)
    forbid = set() if forbid_uids is None else set(map(int, forbid_uids))
    if protect_history:
        forbid |= _protected_uids_from_events(row.get("events", []))
    if vacancy_element is None:
        cand = [i for i in range(len(atoms)) if int(uids[i]) not in forbid]
    else:
        cand = [
            i
            for i in range(len(atoms))
            if syms[i] == vacancy_element and int(uids[i]) not in forbid
        ]
    if int(n) > len(cand):
        raise ValueError(
            f"Not enough candidates to remove (vacancy_element={vacancy_element}): need {n}, have {len(cand)}"
        )
    pick_idx = rng.choice(cand, size=int(n), replace=False)
    events_new = list(row.get("events", []))
    pristine_positions = row.get("pristine_positions", None)
    # record events BEFORE deletion
    for i in pick_idx:
        site_uid = int(uids[i])
        site_pos0 = (
            pristine_positions.get(site_uid)
            if isinstance(pristine_positions, dict)
            else atoms.positions[i].tolist()
        )
        events_new.append(
            {
                "type": "vacancy",
                "removed_element": str(syms[i]),
                "site_uid": site_uid,
                "site_pos0": site_pos0,
                "pos_at_removal": atoms.positions[i].tolist(),
            }
        )
    for i in sorted(map(int, pick_idx), reverse=True):
        del atoms[i]
    validate_atoms_arrays(atoms)
    return make_config_row(
        atoms=atoms,
        structure_id=row.get("structure_id"),
        parent_id=row.get("structure_id"),
        events=events_new,
        seed=seed,
        phase_type=row.get("phase_type"),
        reference_phase=row.get("reference_phase"),
        pristine_n_sites=row.get("n_sites_pristine"),
        pristine_positions=row.get("pristine_positions"),
    )


def op_vacancy_protect_element(row, vacancy_element, protect_element, n=1, seed=None):
    forbid = element_uids(row["atoms"], protect_element)
    return op_vacancy(
        row, vacancy_element=vacancy_element, n=n, seed=seed, forbid_uids=forbid
    )


# def atoms_with_vac_marker(row, marker="H"):
#     at = row["atoms"].copy()
#     vac = [e for e in row["events"] if e["type"]=="vacancy"][0]
#     at += Atoms(marker, positions=[vac["site_pos0"]], cell=at.cell, pbc=at.pbc)
#     return at
def atoms_with_vac_marker(row, marker="H", uid_key="uid", marker_uid=-1):
    at = row["atoms"].copy()
    vac = [e for e in row["events"] if e["type"] == "vacancy"][0]
    # build marker Atoms with the SAME array keys as `at`
    mk = Atoms([marker], positions=[vac["site_pos0"]], cell=at.cell, pbc=at.pbc)
    # ensure marker has uid array (and any other per-atom arrays) so ASE can extend safely
    if uid_key in at.arrays:
        mk.arrays[uid_key] = np.array([marker_uid], dtype=at.arrays[uid_key].dtype)
    # If you have more arrays on `at`, mirror them here (set sensible defaults)
    for name, arr in at.arrays.items():
        if name not in mk.arrays:
            # create a 1-row default of the correct dtype/shape
            mk.arrays[name] = np.zeros((1,) + arr.shape[1:], dtype=arr.dtype)
    at.extend(mk)
    return at


# %% [markdown]
# ## Interstitials
# %%
def voronoi_interstitial_positions_from_row(row, element):
    at = row["atoms"]
    pmg = ase_to_pymatgen(at)
    gen = VoronoiInterstitialGenerator()  # <-- no symprec
    defects = gen.get_defects(pmg, {str(element)})
    frac_list = []
    for d in defects:
        if hasattr(d, "site") and hasattr(d.site, "frac_coords"):
            frac_list.append(np.array(d.site.frac_coords, float))
        elif hasattr(d, "defect_site") and hasattr(d.defect_site, "frac_coords"):
            frac_list.append(np.array(d.defect_site.frac_coords, float))
        else:
            raise AttributeError(
                "Could not extract frac_coords from Voronoi interstitial defect object."
            )
    frac = np.array(frac_list, float)
    cart = frac @ at.get_cell().array
    return cart, frac, defects


def op_interstitial_voronoi(row, element, n=1, seed=None):
    rng = np.random.default_rng(seed if seed is not None else row.get("seed", 0))
    atoms = ensure_uids(row["atoms"]).copy()
    validate_atoms_arrays(atoms)
    cart, frac, defects = voronoi_interstitial_positions_from_row(row, element)
    if len(cart) == 0:
        raise ValueError("VoronoiInterstitialGenerator returned no candidate sites.")
    if int(n) > len(cart):
        raise ValueError(
            f"Requested n={n} interstitials, but only {len(cart)} candidate sites available."
        )
    pick = rng.choice(np.arange(len(cart)), size=int(n), replace=False)
    events_new = list(row.get("events", []))
    for idx in pick:
        pos = cart[idx].tolist()
        frac0 = frac[idx].tolist()
        new_uid = next_uid(atoms)
        atoms = append_atom_with_uid(atoms, str(element), pos)
        validate_atoms_arrays(atoms)
        events_new.append(
            {
                "type": "interstitial",
                "element": str(element),
                "atom_uid": int(new_uid),
                "pos0": pos,
                "frac0": frac0,
                "site_label": f"voronoi_{int(idx)}",
            }
        )
    return make_config_row(
        atoms=atoms,
        structure_id=row.get("structure_id"),
        parent_id=row.get("structure_id"),
        events=events_new,
        seed=seed,
        phase_type=row.get("phase_type"),
        reference_phase=row.get("reference_phase"),
        pristine_n_sites=row.get("n_sites_pristine"),
        pristine_positions=row.get("pristine_positions"),
    )


# %%
def op_interstitial(row, element, positions, n=1, seed=None):
    """
    Add n interstitial atoms of 'element' at positions sampled from `positions`.
    Parameters
    ----------
    row : dict
        One config-row (has at least row["atoms"] and row["events"]).
    element : str
        Chemical symbol to insert (e.g. "Mg").
    positions : array-like, shape (M, 3)
        Candidate Cartesian positions (Å) where an interstitial could be placed.
        Typically precomputed once (e.g. Voronoi sites on the pristine cell).
    n : int
        How many interstitial atoms to add (default 1).
    seed : int or None
        Random seed to reproducibly pick which candidate site(s) are used.
    Returns
    -------
    new_row : dict
        New config-row with updated atoms and appended interstitial events.
    """
    rng = np.random.default_rng(seed if seed is not None else row.get("seed", 0))
    # Work on a copy, and ensure existing atoms already have stable uids
    atoms = ensure_uids(row["atoms"]).copy()
    validate_atoms_arrays(atoms)
    # Convert positions to a numeric array and sanity-check
    positions = np.asarray(positions, float)
    if len(positions) == 0:
        raise ValueError("No interstitial candidate positions provided.")
    if int(n) > len(positions):
        raise ValueError(
            f"Requested n={n} interstitials, but only {len(positions)} candidate positions provided."
        )
    # Choose which candidate sites to use (without replacement)
    pick = rng.choice(np.arange(len(positions)), size=int(n), replace=False)
    # Copy old events and append new interstitial events
    events_new = list(row.get("events", []))
    for idx in pick:
        pos = positions[idx].tolist()
        # Assign a fresh uid for the NEW atom (existing uids remain unchanged)
        new_uid = next_uid(atoms)
        # Append the new atom + extend atoms.arrays["uid"]
        atoms = append_atom_with_uid(atoms, element, pos)
        validate_atoms_arrays(atoms)
        # Log the event (so you can track what happened later)
        events_new.append(
            {
                "type": "interstitial",
                "element": str(element),
                "atom_uid": int(new_uid),
                "pos0": pos,  # where it was inserted (cartesian)
                "site_label": f"cand_{int(idx)}",  # which candidate site was used
            }
        )
    # Build and return a new config-row dict (updates tag, counts, etc.)
    return make_config_row(
        atoms=atoms,
        structure_id=row.get("structure_id"),
        parent_id=row.get("structure_id"),
        events=events_new,
        seed=seed,
        phase_type=row.get("phase_type"),
        reference_phase=row.get("reference_phase"),
        pristine_n_sites=row.get("n_sites_pristine"),
        pristine_positions=row.get("pristine_positions"),
    )


@as_function_node("structures")
def expand_configs(base, operator, operator_kwargs_list, keep_input=True):
    """
    base: a config-row dict or a DataFrame of rows
    operator: function(row, **kwargs) -> new_row dict
    operator_kwargs_list: list of kwargs dicts (each produces one new config per input row)
    keep_input: include original rows in output
    """
    if isinstance(base, dict):
        base_rows = [base]
    elif isinstance(base, pd.DataFrame):
        base_rows = base.to_dict("records")
    else:
        raise TypeError("base must be a config-row dict or a pandas DataFrame")
    out = []
    if keep_input:
        out.extend(base_rows)
    for row in base_rows:
        for kw in operator_kwargs_list:
            out.append(operator(row, **kw))
    return pd.DataFrame(out)


# # use the optimized Al structure as the reference lattice
# atoms0, pristine_pos = make_pristine_reference(atoms_Al_108_optimized)
# base = make_config_row(
#     atoms=atoms0,
#     structure_id="Al_fcc_108_opt",
#     parent_id=None,
#     events=[],
#     seed=0,
#     phase_type="fcc",
#     reference_phase="solid",
#     pristine_n_sites=len(atoms0),
#     pristine_positions=pristine_pos,
# )
# base_df = pd.DataFrame([base])
# base_df[["tag", "n_sites_final", "counts"]]
# s1_kwargs = [
#     {
#         "from_element": "Al",
#         "to_element": "Mg",
#         "n": 1,
#         "seed": s,
#         "protect_history": True,
#     }
#     for s in range(1000, 1100, 10)
# ]
# # s1_kwargs = [{"from_element": "Al", "to_element": "Mg", "n": 1, "seed": s, "protect_history": True} for s in range(1000, 1100, 1)]
# df_s1 = expand_configs(base_df, op_substitute, s1_kwargs, keep_input=False)
# df_s1["tag"].value_counts()
