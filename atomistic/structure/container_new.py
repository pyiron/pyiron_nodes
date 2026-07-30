"""
Point defect (vacancy, substitution, interstitial) structure generation,
tracking, and analysis.

StructureContainer
-------------------
A dataclass holding a list of structure rows. Each row is a dict with:

    structure
        The ase.Atoms object, tagged with a per-atom "uid" array so an
        atom stays identifiable across insertions/deletions.
    is_pristine, stoichiometry, generation
        generation 0 is a pristine row; a defect built on a pristine row
        is generation 1, a defect built on that defect is generation 2,
        etc.
    pristine_structure_index, parent_index
        Absolute row indices of this row's pristine ancestor and its
        immediate parent.
    events
        The list of defect operations (one dict per vacancy /
        substitution / interstitial) applied to reach this row.
    operations_short, operation
        Short and full text summaries built from events, e.g.
        "vacancy[5]|substitution[Al->Mg]".

Its methods cover: adding rows (add_pristine, add_defect), exporting to a
DataFrame (to_dataframe, get_structure_table, get_defect_table,
get_pristine_table), filtering (filter_by_indices, filter_by_generation,
filter_by_max_generation, filter_by_operations_short,
filter_by_operations_contains, filter_by_condition, filter_by_unique_id,
filter_by_number_of_atoms, filter_by_element_count,
filter_by_stoichiometry, filter_by_parent), and index/lineage resolution
(find_structure_index, get_structure, _find_pristine_index,
latest_pristine_index, resolve_defect_row, resolve_any_row).

Module-level functions
-----------------------
UID helpers: ensure_uids, next_uid, uid_to_index, element_uids,
validate_atoms_arrays, append_atom_with_uid, _protected_uids_from_events.
These manage the per-atom "uid" array used to track individual atoms
through insertions and deletions.

Other plain helpers: validate_structure (checks for atoms sitting too
close together), get_stoichiometry, make_operations_short.

Node functions (as_function_node)
-----------------------------------
Structure/table utilities: GetStoichiometry, ValidateStructure,
ElementUids, GetStructureTable, GetDefectTable, GetPristineTable.

Adding and creating structures: AddPristine adds a pristine row.
CreateDefectFromIds, CreateDefectFromSeed, CreateDefectBatchFromIds, and
CreateDefectBatchFromSeed each create vacancy / substitution /
interstitial defects, selected via a defect_type argument
("vacancy" / "substitution" / "interstitial") rather than three separate
functions per type. "FromIds" takes explicit atom_ids/site_ids; "FromSeed"
samples randomly given a seed; the "Batch" variants apply either across
multiple target rows in one call. parent_defect_index / input_structure
select which row a new defect is built on top of; the private helper
_resolve_parent implements that resolution.

Filtering and selection nodes: FilterByIndices, FilterByGeneration,
FilterByMaxGeneration, FilterByOperationsShort, FilterByOperationsContains,
FilterByUniqueId, FilterByNumberOfAtoms, FilterByElementCount,
FilterByStoichiometry, FilterByParent, GetStructure, FindStructureIndex,
GetPristineStructures, GetDefectStructures, LatestPristineIndex,
ResolveDefectRow, ResolveAnyRow. filter_by_condition is a plain function,
not a node, since it takes a Callable.

Distance nodes: GetVacancyDistances, GetSubstitutionDistances, and
GetInterstitialDistances compute pairwise minimum-image distances between
same-type defects from their event positions (pre-relaxation).
GetSubstitutionDistancesRelaxed and GetInterstitialDistancesRelaxed do the
same from a relaxed ase.Atoms plus its events list.

Interstitial site-finding: GetVoronoiInterstitialSites and
GetDelaunayInterstitialSites find candidate interstitial positions using
scipy (Voronoi vertices or Delaunay circumcenters, respectively).
GetVoronoiInterstitialSitesPymatgen finds them via pymatgen's
symmetry-aware VoronoiInterstitialGenerator, gated behind two
independent optional-dependency flags: STRUCTURETOOLKIT_AVAILABLE and
PYMATGEN_ANALYSIS_DEFECTS_AVAILABLE. _wrap_frac, _frac_equiv, _deduplicate_frac, and
_extract_defect_frac_coords are private helpers used only by that
function. _periodic_image_points and
_filter_cluster_tile_interstitial_candidates are shared by the two scipy
based site finders. Any of the three site finders produces the
sublattice array that the interstitial branch of the Create* functions
expects.
"""

from __future__ import annotations  # Enables lazy imports for type hints

from ase import Atoms
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from collections.abc import Callable

from core import as_function_node

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

# Optional dependencies for GetVoronoiInterstitialSitesPymatgen. structuretoolkit
# and pymatgen-analysis-defects are independent packages -- either can be
# present or absent regardless of the other, so each is checked and reported
# separately rather than folded into one combined flag.
try:
    from structuretoolkit.common import ase_to_pymatgen

    STRUCTURETOOLKIT_AVAILABLE = True
except ImportError:
    STRUCTURETOOLKIT_AVAILABLE = False

try:
    from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    PYMATGEN_ANALYSIS_DEFECTS_AVAILABLE = True
except ImportError:
    PYMATGEN_ANALYSIS_DEFECTS_AVAILABLE = False


# ============================================================================
# UID Helper Functions
# ============================================================================

UID_KEY = "uid"


def ensure_uids(atoms: Atoms, uid_key: str = UID_KEY) -> Atoms:
    """Attach stable uids if missing. Does not modify existing uids."""
    import numpy as np

    if uid_key in atoms.arrays:
        return atoms
    atoms = atoms.copy()
    atoms.arrays[uid_key] = np.arange(len(atoms), dtype=int)
    return atoms


def next_uid(atoms: Atoms, uid_key: str = UID_KEY) -> int:
    """Return a fresh uid greater than all existing ones."""
    import numpy as np

    if uid_key not in atoms.arrays or len(atoms) == 0:
        return 0
    return int(np.max(atoms.arrays[uid_key])) + 1


def uid_to_index(atoms: Atoms, uid: int, uid_key: str = UID_KEY):
    """Return current atom index for a given uid, or None if that uid no longer exists."""
    import numpy as np

    if uid_key not in atoms.arrays:
        return None
    hits = np.where(atoms.arrays[uid_key] == int(uid))[0]
    return int(hits[0]) if len(hits) else None


def element_uids(atoms: Atoms, element: str, uid_key: str = UID_KEY):
    """Get all UIDs for a given element."""
    import numpy as np

    atoms = ensure_uids(atoms, uid_key=uid_key)
    syms = np.array(atoms.get_chemical_symbols(), dtype=object)
    uids = atoms.arrays[uid_key].astype(int)
    return uids[syms == element].tolist()


def validate_atoms_arrays(atoms, uid_key="uid"):
    """Validate that all per-atom arrays have the correct length."""
    n = len(atoms)
    bad = {k: v.shape[0] for k, v in atoms.arrays.items() if v.shape[0] != n}
    if bad:
        raise ValueError(f"Inconsistent per-atom arrays: len(atoms)={n}, bad={bad}")


def append_atom_with_uid(
    atoms: Atoms, symbol: str, position, uid_key: str = "uid"
) -> Atoms:
    """Append a new atom with a fresh UID."""
    import numpy as np

    atoms = ensure_uids(atoms, uid_key=uid_key).copy()
    new_id = next_uid(atoms, uid_key=uid_key)
    atoms.append(Atoms(symbols=[symbol], positions=[position])[0])

    u = atoms.arrays[uid_key]
    if u.shape[0] == len(atoms) - 1:
        atoms.arrays[uid_key] = np.append(u, new_id).astype(int)
    elif u.shape[0] == len(atoms):
        atoms.arrays[uid_key][-1] = new_id
    else:
        raise ValueError(
            f"uid array has unexpected length {u.shape[0]} for len(atoms)={len(atoms)}"
        )

    return atoms


def _protected_uids_from_events(events):
    """Get UIDs of atoms that were explicitly created/modified by defects."""
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


def validate_structure(atoms: Atoms, min_distance: float = 0.5) -> bool:
    """
    Validate a structure for common issues like atoms too close together.

    Parameters
    ----------
    atoms : Atoms
        The structure to validate
    min_distance : float
        Minimum allowed interatomic distance in Angstroms

    Returns
    -------
    bool
        True if structure is valid

    Raises
    ------
    ValueError
        If atoms are too close together

    Examples
    --------
    >>> validate_structure(atoms_Al)  # Returns True if valid
    >>> validate_structure(atoms_bad, min_distance=0.8)  # Check with strict cutoff
    """
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            dist = atoms.get_distance(i, j, mic=True)
            if dist < min_distance:
                raise ValueError(
                    f"Structure validation failed: Atoms {i} and {j} are too close "
                    f"({dist:.3f} < {min_distance:.3f} Å). "
                    f"This may cause numerical issues in calculations."
                )
    return True


def get_stoichiometry(atoms: Atoms) -> str:
    """
    Get stoichiometry string from an Atoms object.

    Parameters
    ----------
    atoms : Atoms
        Atomic structure

    Returns
    -------
    str
        Chemical formula (e.g., "Al107Mg1")

    Examples
    --------
    >>> from ase.build import bulk
    >>> atoms = bulk('Al', cubic=True)
    >>> get_stoichiometry(atoms)
    'Al4'
    """
    from collections import Counter

    counts = Counter(atoms.get_chemical_symbols())
    return "".join(f"{el}{counts[el]}" for el in sorted(counts))


def make_operations_short(events: list[dict]) -> str:
    """
    Create pipe-separated short form from events.

    Examples
    --------
    >>> events = [{"type": "vacancy", "site_uid": 5}]
    >>> make_operations_short(events)
    'vacancy[5]'

    >>> multiple = [
    ...     {"type": "vacancy", "site_uid": 5},
    ...     {"type": "substitution", "from": "Al", "to": "Mg", "site_uid": 10}
    ... ]
    >>> make_operations_short(multiple)
    'vacancy[5]|substitution[Al->Mg]'

    Parameters
    ----------
    events : List[dict]
        List of defect events

    Returns
    -------
    str
        Pipe-separated operation short form
    """
    if not events:
        return "no_operations"

    short_ops = []
    for ev in events:
        t = ev.get("type")
        if t == "vacancy":
            uid = ev.get("site_uid", "?")
            short_ops.append(f"vacancy[{uid}]")
        elif t == "substitution":
            from_el = ev.get("from", "?")
            to_el = ev.get("to", "?")
            uid = ev.get("site_uid", "?")
            short_ops.append(f"substitution[{from_el}->{to_el}]")
        elif t == "interstitial":
            el = ev.get("element", "?")
            uid = ev.get("atom_uid", "?")
            short_ops.append(f"interstitial[{el}]")

    return "|".join(short_ops) if short_ops else "no_operations"


# ============================================================================
# StructureContainer Class
# ============================================================================


@dataclass
class StructureContainer:
    """
    Enhanced container for pristine and defect structures.

    Features:
    - Unambiguous operation tracking with pipe-separated short form
    - Clear lineage tracking with top-level parent/pristine indices
    - Duplicate checking for pristine structures
    - Flexible parent resolution
    - Comprehensive filtering methods
    - Table extraction methods
    - Event tracking for defect lineage

    Data Structure
    --------------
    Each structure is stored as a dict with the following fields:

    Core Structure Data:
      - structure: Atoms object (the atomic structure)
      - unique_id: Unique identifier string
      - creation_timestamp: datetime of creation

    Lineage Tracking (top level):
      - pristine_structure_index: Absolute index of original pristine ancestor
      - parent_index: Absolute index of immediate parent
      - generation: Distance from pristine (0, 1, 2, ...)

    Classification:
      - is_pristine: True for reference, False for defects
      - stoichiometry: Chemical formula (e.g., "Al107Mg1")

    Operation Descriptions:
      - operation: Full human-readable description
      - operations_short: Pipe-separated short form (e.g., "vacancy[5]|substitution[10->Mg]")

    History:
      - events: Complete chronological list of all defect operations

    Metadata:
      - metadata: User-provided additional data
    """

    _structures: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Basic Methods
    # ------------------------------------------------------------------
    import pandas as pd

    def to_dataframe(self) -> pd.DataFrame:
        """Convert internal list to pandas DataFrame on demand."""
        import pandas as pd

        df_data = []
        for s in self._structures:
            df_data.append(
                {
                    "structure": s["structure"],
                    "unique_id": s["unique_id"],
                    "is_pristine": s["is_pristine"],
                    "stoichiometry": s["stoichiometry"],
                    "generation": s["generation"],
                    "pristine_structure_index": s.get("pristine_structure_index", -1),
                    "parent_index": s.get("parent_index", -1),
                    "operation": s["operation"],
                    "operations_short": s.get("operations_short", ""),
                    "events": s["events"],
                    "metadata": s["metadata"],
                    "creation_timestamp": s["creation_timestamp"],
                }
            )
        return pd.DataFrame(df_data)

    def get_structure_table(self) -> pd.DataFrame:
        """Return a copy of the full structure table."""
        return self.to_dataframe()

    def get_defect_table(self) -> pd.DataFrame:
        """Return a filtered copy containing only defect rows."""
        df = self.to_dataframe()
        return df[df["is_pristine"] == False].copy()

    def get_pristine_table(self) -> pd.DataFrame:
        """Return a filtered copy containing only pristine rows."""
        df = self.to_dataframe()
        return df[df["is_pristine"] == True].copy()

    # ------------------------------------------------------------------
    # Add Structures
    # ------------------------------------------------------------------

    def add_pristine(
        self,
        atoms: Atoms,
        unique_id: str | None = None,
        metadata: dict | None = None,
        check_duplicates: bool = True,
        tolerance: float = 1e-6,
    ) -> int:
        """
        Add a pristine reference structure with optional duplicate checking.

        Parameters
        ----------
        atoms : Atoms
            Structure to add
        unique_id : str or None
            Custom unique identifier
        metadata : dict or None
            Additional metadata
        check_duplicates : bool
            If True, check if identical structure already exists
        tolerance : float
            Numerical tolerance for comparing positions and cell

        Returns
        -------
        int
            Absolute row index of the (possibly pre-existing) pristine entry
        """
        from datetime import datetime
        import numpy as np

        atoms = ensure_uids(atoms)
        uid = unique_id or f"pristine_{len(self._structures)}"

        # Check for duplicates if requested
        if check_duplicates:
            for idx, s in enumerate(self._structures):
                if s["is_pristine"]:
                    s_structure = ensure_uids(s["structure"])
                    candidate_stoich = self._get_stoichiometry(atoms)
                    if (
                        self._get_stoichiometry(s_structure) == candidate_stoich
                        and np.allclose(
                            s_structure.positions, atoms.positions, atol=tolerance
                        )
                        and np.allclose(s_structure.cell, atoms.cell, atol=tolerance)
                        and s_structure.get_chemical_symbols()
                        == atoms.get_chemical_symbols()
                    ):
                        return idx

        entry = {
            "structure": atoms.copy(),
            "unique_id": uid,
            "is_pristine": True,
            "stoichiometry": self._get_stoichiometry(atoms),
            "generation": 0,
            "pristine_structure_index": -1,
            "parent_index": -1,
            "operation": "pristine",
            "operations_short": "pristine",
            "events": [],
            "metadata": metadata or {},
            "creation_timestamp": datetime.now(),
        }
        self._structures.append(entry)
        return len(self._structures) - 1

    def add_defect(
        self,
        atoms: Atoms,
        operation: str,
        pristine_index: int,
        parent_index: int,
        events: list[dict],
        unique_id: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """
        Add a defect structure.

        Parameters
        ----------
        atoms : Atoms
            The defect structure
        operation : str
            Full human-readable operation description
        pristine_index : int
            Absolute index of original pristine ancestor
        parent_index : int
            Absolute index of immediate parent
        events : list of dict
            Complete chronological list of defect operations
        unique_id : str or None
            Custom unique identifier
        metadata : dict or None
            Additional metadata

        Returns
        -------
        int
            Absolute row index of the newly added defect entry
        """
        from datetime import datetime

        uid = unique_id or f"defect_{len(self._structures)}"
        operations_short = self._make_operations_short(events)

        entry = {
            "structure": atoms.copy(),
            "unique_id": uid,
            "is_pristine": False,
            "stoichiometry": self._get_stoichiometry(atoms),
            "generation": self._structures[parent_index]["generation"] + 1,
            "pristine_structure_index": pristine_index,
            "parent_index": parent_index,
            "operation": operation,
            "operations_short": operations_short,
            "events": events,
            "metadata": metadata or {},
            "creation_timestamp": datetime.now(),
        }
        self._structures.append(entry)
        return len(self._structures) - 1

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    @staticmethod
    def _make_operations_short(events: list[dict]) -> str:
        return make_operations_short(events)

    @staticmethod
    def _get_stoichiometry(atoms: Atoms) -> str:
        return get_stoichiometry(atoms)

    def find_structure_index(self, atoms: Atoms, tolerance: float = 1e-6) -> int | None:
        """
        Return the absolute row index of a structure, or None if absent.

        Checks for identity, then numerical equality.
        """
        import numpy as np

        atoms = ensure_uids(atoms)
        candidate_stoich = self._get_stoichiometry(atoms)

        for row_idx in range(len(self._structures)):
            stored = self._structures[row_idx]
            s_structure = ensure_uids(stored["structure"])

            if s_structure is atoms:
                return row_idx

            if (
                self._get_stoichiometry(s_structure) == candidate_stoich
                and np.allclose(s_structure.positions, atoms.positions, atol=tolerance)
                and np.allclose(s_structure.cell, atoms.cell, atol=tolerance)
                and s_structure.get_chemical_symbols() == atoms.get_chemical_symbols()
            ):
                return row_idx

        return None

    # ------------------------------------------------------------------
    # Filtering Methods
    # ------------------------------------------------------------------

    def filter_by_indices(self, indices: list[int]) -> list[dict]:
        """Get structures by absolute indices."""
        return [self._structures[i] for i in indices if i < len(self._structures)]

    def filter_by_generation(self, generation: int) -> list[dict]:
        """Get all structures at specific distance from pristine."""
        return [s for s in self._structures if s["generation"] == generation]

    def filter_by_operations_short(self, pattern: str) -> list[dict]:
        """
        Filter by operations_short field (supports wildcards).

        Examples:
          - "vacancy[5]" - exact match
          - "*vacancy[5]*" - contains vacancy[5]
          - "vacancy[*]|substitution[*]" - multiple patterns
        """
        import fnmatch

        if "*" in pattern or "?" in pattern or "[" in pattern:
            return [
                s
                for s in self._structures
                if fnmatch.fnmatch(s.get("operations_short", ""), pattern)
            ]
        else:
            return [
                s for s in self._structures if s.get("operations_short", "") == pattern
            ]

    def filter_by_operations_contains(self, operation_type: str) -> list[dict]:
        """
        Filter structures whose operations contain a specific type.

        Examples:
          - "vacancy" - all structures with vacancy
          - "substitution" - all structures with substitution
        """
        return [
            s
            for s in self._structures
            if operation_type in s.get("operations_short", "")
        ]

    def filter_by_condition(self, condition: Callable[[dict], bool]) -> list[dict]:
        """Filter by custom function."""
        return [s for s in self._structures if condition(s)]

    def filter_by_unique_id(self, unique_id: str) -> dict | None:
        """Get structure by unique ID."""
        for s in self._structures:
            if s["unique_id"] == unique_id:
                return s
        return None

    def filter_by_max_generation(self, max_generation: int) -> list[dict]:
        """
        Get all structures within N steps of pristine.

        Parameters
        ----------
        max_generation : int
            Maximum generation number (get all structures with generation <= max_generation)

        Returns
        -------
        List[dict]
            Structures with generation <= max_generation

        Example
        -------
        >>> # Get all structures up to generation 2 (pristine, first-, and second-generation defects)
        >>> gen2_and_below = container.filter_by_max_generation(2)
        """
        return [s for s in self._structures if s["generation"] <= max_generation]

    def filter_by_number_of_atoms(self, n_atoms: int) -> list[dict]:
        """
        Get structures with a specific number of atoms.

        Parameters
        ----------
        n_atoms : int
            Number of atoms to match

        Returns
        -------
        List[dict]
            Structures with exactly n_atoms atoms

        Example
        -------
        >>> # Find structures with exactly 108 atoms
        >>> structures_108 = container.filter_by_number_of_atoms(108)
        """
        return [s for s in self._structures if len(s["structure"]) == n_atoms]

    def filter_by_element_count(
        self,
        element: str,
        min_count: int | None = None,
        max_count: int | None = None,
        exact_count: int | None = None,
    ) -> list[dict]:
        """
        Get structures matching element count criteria.

        Parameters
        ----------
        element : str
            Chemical symbol (e.g., 'Al', 'Mg')
        min_count : int or None
            Minimum count of this element (inclusive)
        max_count : int or None
            Maximum count of this element (inclusive)
        exact_count : int or None
            Exact count of this element (overrides min/max if provided)

        Returns
        -------
        List[dict]
            Structures matching the element count criteria

        Example
        -------
        >>> # Get structures with exactly 1 Mg atom
        >>> single_mg = container.filter_by_element_count('Mg', exact_count=1)

        >>> # Get structures with 0 to 5 Mg atoms
        >>> few_mg = container.filter_by_element_count('Mg', min_count=0, max_count=5)

        >>> # Get structures with at least 10 Al atoms
        >>> many_al = container.filter_by_element_count('Al', min_count=10)
        """
        from collections import Counter

        matching = []
        for s in self._structures:
            syms = s["structure"].get_chemical_symbols()
            counts = Counter(syms)
            element_count = counts.get(element, 0)

            if exact_count is not None:
                if element_count == exact_count:
                    matching.append(s)
            elif (min_count is None or element_count >= min_count) and (
                max_count is None or element_count <= max_count
            ):
                matching.append(s)

        return matching

    def filter_by_stoichiometry(self, formula_pattern: str | None = None) -> list[dict]:
        """
        Get structures matching a stoichiometry pattern.

        Parameters
        ----------
        formula_pattern : str or None
            Chemical formula pattern to match. Supports wildcards (*, ?, []).
            If None, returns all structures.

        Returns
        -------
        List[dict]
            Structures matching the stoichiometry pattern

        Examples
        --------
        >>> # Exact match
        >>> exact = container.filter_by_stoichiometry('Al107Mg1')

        >>> # Get all structures with exactly 1 Mg
        >>> single_mg = container.filter_by_stoichiometry('*Mg1*')

        >>> # Get all structures containing Si
        >>> any_si = container.filter_by_stoichiometry('*Si*')
        """
        import fnmatch

        if formula_pattern is None:
            return self._structures.copy()

        matching = []
        for s in self._structures:
            stoich = s["stoichiometry"]
            if fnmatch.fnmatch(stoich, formula_pattern):
                matching.append(s)

        return matching

    def filter_by_parent(self, parent_index: int) -> list[dict]:
        """
        Get structures that have a specific parent structure.

        Parameters
        ----------
        parent_index : int
            Absolute index of the parent structure

        Returns
        -------
        List[dict]
            Structures with parent_index matching the specified index (direct children only)

        Example
        -------
        >>> # Find all defects that were created from structure at index 0
        >>> children_0 = container.filter_by_parent(0)
        """
        return [s for s in self._structures if s.get("parent_index") == parent_index]

    # ------------------------------------------------------------------
    # Selection Methods
    # ------------------------------------------------------------------

    def get_pristine_structures(self) -> list[dict]:
        """Get all pristine structures."""
        return [s for s in self._structures if s["is_pristine"]]

    def get_defect_structures(self) -> list[dict]:
        """Get all defect structures."""
        return [s for s in self._structures if not s["is_pristine"]]

    def get_structure(self, index: int) -> dict:
        """Get structure by absolute index."""
        if 0 <= index < len(self._structures):
            return self._structures[index]
        raise IndexError(f"Index {index} out of range")

    def _find_pristine_index(self, structure_idx: int) -> int:
        """Find the pristine ancestor for a given structure."""
        if self._structures[structure_idx]["is_pristine"]:
            return structure_idx
        pristine_idx = self._structures[structure_idx].get(
            "pristine_structure_index", -1
        )
        if pristine_idx == -1:
            current = structure_idx
            visited = set()
            while current != -1 and current not in visited:
                visited.add(current)
                if self._structures[current]["is_pristine"]:
                    return current
                current = self._structures[current].get("parent_index", -1)
        return pristine_idx

    def latest_pristine_index(self) -> int:
        """Absolute row index of the most recently added pristine structure."""
        pristine_rows = [i for i, s in enumerate(self._structures) if s["is_pristine"]]
        if not pristine_rows:
            raise ValueError("No pristine structure found in the container.")
        return pristine_rows[-1]

    # ------------------------------------------------------------------
    # Relative Index Resolution Methods (pyiron_nodes compatibility)
    # ------------------------------------------------------------------

    def resolve_defect_row(self, relative_index: int) -> int:
        """
        Convert a relative defect index to an absolute structure index.

        Resolves indices like 0 (first defect), 1 (second defect),
        -1 (most recent defect), -2 (second most recent), etc.

        Parameters
        ----------
        relative_index : int
            Relative defect index (0-based, negative counts from end)

        Returns
        -------
        int
            Absolute row index in the full structure table

        Raises
        ------
        IndexError
            If no defect structures exist or index out of range

        Examples
        --------
        >>> container.resolve_defect_row(0)    # First defect
        >>> container.resolve_defect_row(-1)   # Most recent defect
        >>> container.resolve_defect_row(2)    # Third defect
        """
        defect_rows = [
            i for i, s in enumerate(self._structures) if not s["is_pristine"]
        ]

        if not defect_rows:
            raise IndexError(
                "No defect structures found in the container. "
                "Add some defects first using create_vacancy_from_ids or create_vacancy_from_seed."
            )

        n_defects = len(defect_rows)

        if relative_index >= 0:
            # Positive index: 0 = first defect, 1 = second, etc.
            if relative_index >= n_defects:
                raise IndexError(
                    f"Relative defect index {relative_index} out of range. "
                    f"Container has {n_defects} defect structures (indices 0 to {n_defects-1}). "
                    f"Use a smaller index or negative indices (-{n_defects} to -1)."
                )
            return defect_rows[relative_index]
        else:
            # Negative index: -1 = most recent, -2 = second most recent, etc.
            abs_idx = n_defects + relative_index  # relative_index is negative
            if abs_idx < 0:
                raise IndexError(
                    f"Relative defect index {relative_index} out of range. "
                    f"Container has {n_defects} defect structures. "
                    f"Valid negative indices: -{n_defects} to -1."
                )
            return defect_rows[abs_idx]

    def resolve_any_row(self, relative_index: int) -> int:
        """
        Convert a relative index to an absolute structure index (any structure).

        Resolves indices like 0 (first structure), 1 (second structure),
        -1 (most recent structure), -2 (second most recent), etc.
        Works on both pristine and defect structures.

        Parameters
        ----------
        relative_index : int
            Relative index (0-based, negative counts from end)

        Returns
        -------
        int
            Absolute row index in the structure table

        Raises
        ------
        IndexError
            If container is empty or index out of range

        Examples
        --------
        >>> container.resolve_any_row(0)     # First structure (could be pristine or defect)
        >>> container.resolve_any_row(-1)    # Most recent structure
        >>> container.resolve_any_row(1)     # Second structure
        """
        n_structures = len(self._structures)

        if n_structures == 0:
            raise IndexError(
                "Container is empty. Add a structure first using add_pristine()."
            )

        if relative_index >= 0:
            # Positive index: 0 = first structure, 1 = second, etc.
            if relative_index >= n_structures:
                raise IndexError(
                    f"Relative index {relative_index} out of range. "
                    f"Container has {n_structures} structures (indices 0 to {n_structures-1}). "
                    f"Use a smaller index or negative indices (-{n_structures} to -1)."
                )
            return relative_index
        else:
            # Negative index: -1 = most recent, -2 = second most recent, etc.
            abs_idx = n_structures + relative_index  # relative_index is negative
            if abs_idx < 0:
                raise IndexError(
                    f"Relative index {relative_index} out of range. "
                    f"Container has {n_structures} structures. "
                    f"Valid negative indices: -{n_structures} to -1."
                )
            return abs_idx

    # ------------------------------------------------------------------
    # Magic Methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._structures)

    def __repr__(self) -> str:
        n_pristine = len(self.get_pristine_structures())
        n_defects = len(self.get_defect_structures())
        return f"StructureContainer({len(self)} structures: {n_pristine} pristine, {n_defects} defects)"


# ============================================================================
# Utility Functions (extracted from StructureContainer)
# ============================================================================


def _pairwise_pbc_distances(positions: list[np.ndarray], cell) -> tuple:
    """
    Compute all pairwise minimum-image distances for a list of Cartesian
    positions under periodic boundary conditions.

    Returns
    -------
    distances : dict
        ``{"0-1": float, "0-2": float, ...}``
    distance_matrix : (N, N) ndarray
    """
    import numpy as np

    inv_cell = np.linalg.inv(cell)
    n = len(positions)
    distances = {}
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            delta = positions[i] - positions[j]
            delta -= np.round(delta @ inv_cell) @ cell
            dist = np.linalg.norm(delta)
            distances[f"{i}-{j}"] = dist
            distance_matrix[i, j] = distance_matrix[j, i] = dist

    return distances, distance_matrix


@as_function_node
def GetVacancyDistances(container: StructureContainer, defect_index: int):
    """
    Get pairwise distances between vacancies in a defect structure.

    Uses positions from vacancy events (before relaxation).
    Handles periodic boundary conditions using the crystal cell.

    Parameters
    ----------
    container : StructureContainer
        The structure container
    defect_index : int
        Absolute index of the defect structure

    Returns
    -------
    dict
        Dictionary with vacancy info and distances:
        {
            'vacancies': [{'uid': int, 'position': array}, ...],
            'distances': {'0-1': float, '0-2': float, ...},
            'distance_matrix': 2D numpy array
        }

    Examples
    --------
    >>> container = AddPristine(atoms=atoms)
    >>> container = CreateDefectFromIds(
    ...     structure_container=container, defect_type="vacancy", atom_ids=[0, 5]
    ... )
    >>> result = GetVacancyDistances(container=container, defect_index=1)
    >>> print(result['distances'])
    {'0-1': 4.05}
    """
    import numpy as np

    defect = container._structures[defect_index]
    events = defect["events"]

    # Get all vacancy events
    vacancy_events = [e for e in events if e["type"] == "vacancy"]

    if len(vacancy_events) < 2:
        out = {
            "vacancies": [],
            "distances": {},
            "distance_matrix": np.array([]),
            "message": f"Need at least 2 vacancies, found {len(vacancy_events)}",
        }
    else:
        # Extract vacancy info
        vacancies = []
        for ev in vacancy_events:
            vacancies.append(
                {"uid": ev["site_uid"], "position": np.array(ev["site_pos0"])}
            )

        # Get cell for periodic boundary calculations
        pristine = container._structures[defect["pristine_structure_index"]][
            "structure"
        ]
        cell = pristine.get_cell()

        distances, distance_matrix = _pairwise_pbc_distances(
            [v["position"] for v in vacancies], cell
        )

        out = {
            "vacancies": vacancies,
            "distances": distances,
            "distance_matrix": distance_matrix,
        }
    return out


@as_function_node
def GetSubstitutionDistances(container: StructureContainer, defect_index: int):
    """
    Get pairwise distances between substituted sites in a defect structure.

    Uses positions from substitution events (before relaxation).
    Handles periodic boundary conditions using the crystal cell.

    Parameters
    ----------
    container : StructureContainer
        The structure container
    defect_index : int
        Absolute index of the defect structure

    Returns
    -------
    dict
        {
            'substitutions': [{'uid': int, 'from': str, 'to': str, 'position': array}, ...],
            'distances': {'0-1': float, '0-2': float, ...},
            'distance_matrix': 2D numpy array
        }
    """
    import numpy as np

    defect = container._structures[defect_index]
    events = defect["events"]

    sub_events = [e for e in events if e["type"] == "substitution"]

    if len(sub_events) < 2:
        out = {
            "substitutions": [],
            "distances": {},
            "distance_matrix": np.array([]),
            "message": f"Need at least 2 substitutions, found {len(sub_events)}",
        }
    else:
        substitutions = [
            {
                "uid": ev["site_uid"],
                "from": ev["from"],
                "to": ev["to"],
                "position": np.array(ev["site_pos0"]),
            }
            for ev in sub_events
        ]

        pristine = container._structures[defect["pristine_structure_index"]][
            "structure"
        ]
        cell = pristine.get_cell()

        distances, distance_matrix = _pairwise_pbc_distances(
            [s["position"] for s in substitutions], cell
        )

        out = {
            "substitutions": substitutions,
            "distances": distances,
            "distance_matrix": distance_matrix,
        }
    return out


@as_function_node
def GetSubstitutionDistancesRelaxed(atoms, events: list):
    """
    Get pairwise distances between substituted atoms in a relaxed structure.

    Uses actual post-relaxation positions. Each substituted atom is located by
    finding the nearest atom in the relaxed structure to its pre-relaxation
    position (site_pos0 from the event). This works for all phases including
    intermetallics where filtering by element is ambiguous.

    Parameters
    ----------
    atoms : ase.Atoms
        Relaxed structure (e.g., from optimize_positions_with_lammpslib)
    events : list of dict
        Event list from the structure container entry (row['events']).
        Substitution events must contain 'site_pos0'.

    Returns
    -------
    dict
        {
            'substitutions': [{'from': str, 'to': str, 'position_relaxed': array}, ...],
            'distances': {'0-1': float, ...},
            'distance_matrix': 2D numpy array
        }
    """
    import numpy as np

    sub_events = [e for e in events if e["type"] == "substitution"]

    if len(sub_events) < 2:
        out = {
            "substitutions": [],
            "distances": {},
            "distance_matrix": np.array([]),
            "message": f"Need at least 2 substitutions, found {len(sub_events)}",
        }
    else:
        cell = atoms.get_cell()
        inv_cell = np.linalg.inv(cell)

        # For each substitution, find the nearest atom in the relaxed structure
        # to the original site position. Atoms never move far enough during
        # relaxation to be closer to a different lattice site.
        relaxed_positions = []
        substitution_info = []
        for ev in sub_events:
            original_pos = np.array(ev["site_pos0"])
            diffs = atoms.positions - original_pos
            diffs -= np.round(diffs @ inv_cell) @ cell
            dists = np.linalg.norm(diffs, axis=1)
            closest_idx = np.argmin(dists)
            relaxed_positions.append(atoms.positions[closest_idx])
            substitution_info.append(
                {
                    "from": ev["from"],
                    "to": ev["to"],
                    "position_relaxed": atoms.positions[closest_idx],
                }
            )

        distances, distance_matrix = _pairwise_pbc_distances(relaxed_positions, cell)

        out = {
            "substitutions": substitution_info,
            "distances": distances,
            "distance_matrix": distance_matrix,
        }
    return out


@as_function_node
def GetInterstitialDistances(container: StructureContainer, defect_index: int):
    """
    Get pairwise distances between interstitial sites in a defect structure.

    Uses insertion positions from events (before relaxation).
    Handles periodic boundary conditions using the crystal cell.

    Parameters
    ----------
    container : StructureContainer
    defect_index : int
        Absolute index of the defect structure.

    Returns
    -------
    dict with keys 'interstitials', 'distances' {'0-1': float, ...},
    'distance_matrix'.
    """
    import numpy as np

    defect = container._structures[defect_index]
    events = defect["events"]
    int_events = [e for e in events if e["type"] == "interstitial"]

    if len(int_events) < 2:
        out = {
            "interstitials": [],
            "distances": {},
            "distance_matrix": np.array([]),
            "message": f"Need at least 2 interstitials, found {len(int_events)}",
        }
    else:
        interstitials = [
            {
                "uid": ev["atom_uid"],
                "element": ev["element"],
                "position": np.array(ev["pos0"]),
            }
            for ev in int_events
        ]

        pristine = container._structures[defect["pristine_structure_index"]][
            "structure"
        ]
        cell = pristine.get_cell()

        distances, distance_matrix = _pairwise_pbc_distances(
            [it["position"] for it in interstitials], cell
        )

        out = {
            "interstitials": interstitials,
            "distances": distances,
            "distance_matrix": distance_matrix,
        }
    return out


@as_function_node
def GetInterstitialDistancesRelaxed(atoms, events: list):
    """
    Get pairwise distances between interstitial atoms in a relaxed structure.

    Each interstitial is located via its stored atom_uid, which survives
    relaxation because optimize_positions_with_lammpslib returns a copy of
    the input structure with only positions updated (all arrays preserved).
    Falls back to nearest-neighbour from pos0 if the uid array is absent.

    Parameters
    ----------
    atoms : ase.Atoms
        Relaxed structure.
    events : list of dict
        Event list from the structure container entry (row['events']).

    Returns
    -------
    dict with keys 'interstitials', 'distances' {'0-1': float, ...},
    'distance_matrix'.
    """
    import numpy as np

    int_events = [e for e in events if e["type"] == "interstitial"]

    if len(int_events) < 2:
        out = {
            "interstitials": [],
            "distances": {},
            "distance_matrix": np.array([]),
            "message": f"Need at least 2 interstitials, found {len(int_events)}",
        }
    else:
        cell = atoms.get_cell()
        inv_cell = np.linalg.inv(cell)

        relaxed_positions = []
        interstitial_info = []
        for ev in int_events:
            uid = ev.get("atom_uid")
            idx = uid_to_index(atoms, uid) if uid is not None else None
            if idx is not None:
                pos = atoms.positions[idx]
                print(
                    f"  interstitial {ev['element']} (atom_uid={uid}): located via uid"
                )
            else:
                # fallback: nearest-neighbour from insertion position
                original_pos = np.array(ev["pos0"])
                diffs = atoms.positions - original_pos
                diffs -= np.round(diffs @ inv_cell) @ cell
                idx = np.argmin(np.linalg.norm(diffs, axis=1))
                pos = atoms.positions[idx]
                reason = (
                    "no atom_uid in event"
                    if uid is None
                    else "uid not found in atoms.arrays"
                )
                print(
                    f"  interstitial {ev['element']}: located via nearest-neighbour ({reason})"
                )
            relaxed_positions.append(pos)
            interstitial_info.append(
                {"element": ev["element"], "position_relaxed": pos}
            )

        distances, distance_matrix = _pairwise_pbc_distances(relaxed_positions, cell)

        out = {
            "interstitials": interstitial_info,
            "distances": distances,
            "distance_matrix": distance_matrix,
        }
    return out


# ============================================================================
# Standalone Wrapper Functions (GUI-Friendly)
# ============================================================================

# ----------------------------------------------------------------------
# Table/Extraction Functions
# ----------------------------------------------------------------------


@as_function_node("table")
def GetStructureTable(structure_container: StructureContainer) -> pd.DataFrame:
    """
    Return a copy of the full structure table.

    Parameters
    ----------
    structure_container : StructureContainer
        The container to extract from

    Returns
    -------
    pd.DataFrame
        Table with all structures

    Examples
    --------
    >>> container = AddPristine(atoms=atoms)
    >>> df = GetStructureTable(structure_container=container)
    """
    return structure_container.get_structure_table()


@as_function_node("table")
def GetDefectTable(structure_container: StructureContainer) -> pd.DataFrame:
    """
    Return a filtered copy containing only defect rows (is_pristine == False).

    Parameters
    ----------
    structure_container : StructureContainer
        The container to extract from

    Returns
    -------
    pd.DataFrame
        Table with only defect structures

    Examples
    --------
    >>> df = GetDefectTable(structure_container=container)
    """
    return structure_container.get_defect_table()


@as_function_node("table")
def GetPristineTable(structure_container: StructureContainer) -> pd.DataFrame:
    """
    Return a filtered copy containing only pristine rows (is_pristine == True).

    Parameters
    ----------
    structure_container : StructureContainer
        The container to extract from

    Returns
    -------
    pd.DataFrame
        Table with only pristine structures

    Examples
    --------
    >>> df = GetPristineTable(structure_container=container)
    """
    return structure_container.get_pristine_table()


# ----------------------------------------------------------------------
# Structure Utility Functions
# ----------------------------------------------------------------------


@as_function_node("stoichiometry")
def GetStoichiometry(atoms: Atoms) -> str:
    """
    Get stoichiometry string from an Atoms object.

    Parameters
    ----------
    atoms : Atoms
        Atomic structure

    Returns
    -------
    str
        Chemical formula (e.g., "Al107Mg1")

    Examples
    --------
    >>> from ase.build import bulk
    >>> atoms = bulk('Al', cubic=True)
    >>> get_stoichiometry(atoms)
    'Al4'
    """
    return get_stoichiometry(atoms)


@as_function_node("is_valid")
def ValidateStructure(atoms: Atoms, min_distance: float = 0.5) -> bool:
    """
    Validate a structure for common issues like atoms too close together.

    Parameters
    ----------
    atoms : Atoms
        The structure to validate
    min_distance : float
        Minimum allowed interatomic distance in Angstroms

    Returns
    -------
    bool
        True if structure is valid

    Raises
    ------
    ValueError
        If atoms are too close together

    Examples
    --------
    >>> validate_structure(atoms_Al)  # Returns True if valid
    >>> validate_structure(atoms_bad, min_distance=0.8)  # Check with strict cutoff
    """
    return validate_structure(atoms, min_distance)


@as_function_node("uids")
def ElementUids(atoms: Atoms, element: str, uid_key: str = UID_KEY) -> list[int]:
    """
    Get all UIDs for a given element.

    Parameters
    ----------
    atoms : Atoms
        Structure with uid tracking (added automatically if missing)
    element : str
        Chemical symbol (e.g., 'Al', 'Mg')
    uid_key : str
        Key under which uids are stored in atoms.arrays

    Returns
    -------
    List[int]
        UIDs of all atoms matching the given element

    Examples
    --------
    >>> mg_uids = element_uids(atoms, 'Mg')
    """
    return element_uids(atoms, element, uid_key)


# ----------------------------------------------------------------------
# Add Structures
# ----------------------------------------------------------------------


@as_function_node
def AddPristine(
    structure_container: StructureContainer | None = None,
    atoms: Atoms = None,
    unique_id: str | None = None,
    metadata: dict | None = None,
    check_duplicates: bool = True,
    tolerance: float = 1e-6,
) -> StructureContainer:
    """
    Add a pristine reference structure with optional duplicate checking.

    Creates a new container if structure_container is None.

    Parameters
    ----------
    structure_container : StructureContainer or None
        Container to add to (creates new if None)
    atoms : Atoms
        Structure to add
    unique_id : str or None
        Custom unique identifier
    metadata : dict or None
        Additional metadata
    check_duplicates : bool
        If True, check if identical structure already exists
    tolerance : float
        Numerical tolerance for comparing positions and cell

    Returns
    -------
    StructureContainer
        Container with the pristine structure added

    Examples
    --------
    >>> from ase.build import bulk
    >>> atoms = bulk('Al', cubic=True)
    >>> container = AddPristine(atoms=atoms, unique_id="Al_fcc")
    """
    if structure_container is None:
        structure_container = StructureContainer()
    structure_container.add_pristine(
        atoms, unique_id, metadata, check_duplicates, tolerance
    )
    return structure_container


# ----------------------------------------------------------------------
# Filter Functions
# ----------------------------------------------------------------------


@as_function_node("structures")
def FilterByIndices(
    structure_container: StructureContainer, indices: list[int]
) -> list[dict]:
    """
    Get structures by absolute indices.

    Parameters
    ----------
    structure_container : StructureContainer
        The container to filter
    indices : List[int]
        Indices to retrieve

    Returns
    -------
    List[dict]
        Structures at given indices
    """
    return structure_container.filter_by_indices(indices)


@as_function_node("structures")
def FilterByGeneration(
    structure_container: StructureContainer, generation: int
) -> list[dict]:
    """
    Get all structures at specific distance from pristine.

    Parameters
    ----------
    structure_container : StructureContainer
        The container to filter
    generation : int
        Generation number (0=pristine, 1=first defects, etc.)

    Returns
    -------
    List[dict]
        Structures at given generation

    Examples
    --------
    >>> gen1 = FilterByGeneration(structure_container=container, generation=1)
    """
    return structure_container.filter_by_generation(generation)


@as_function_node("structures")
def FilterByMaxGeneration(
    structure_container: StructureContainer, max_generation: int
) -> list[dict]:
    """
    Get all structures within N steps of pristine.

    Parameters
    ----------
    structure_container : StructureContainer
        The container to filter
    max_generation : int
        Maximum generation (get all structures with generation <= max_generation)

    Returns
    -------
    List[dict]
        Structures within max_generation

    Examples
    --------
    >>> gen2_and_below = FilterByMaxGeneration(structure_container=container, max_generation=2)
    """
    return structure_container.filter_by_max_generation(max_generation)


@as_function_node("structures")
def FilterByOperationsShort(
    structure_container: StructureContainer, pattern: str
) -> list[dict]:
    """
    Filter by operations_short field (supports wildcards).

    Parameters
    ----------
    structure_container : StructureContainer
        The container to filter
    pattern : str
        Pattern to match. Matching is exact unless ``pattern`` contains
        any of ``*``, ``?``, ``[``, in which case it's matched via
        ``fnmatch`` instead. Note that every real ``operations_short``
        value contains ``[`` (e.g. ``"vacancy[5]"``), so an exact-looking
        pattern like ``"vacancy[5]"`` is actually routed through
        ``fnmatch`` too -- and there, ``[5]`` is a character class
        matching a single ``"5"``, not the literal brackets, so it will
        *not* match the literal string ``"vacancy[5]"``. Use a trailing
        ``*`` (e.g. ``"vacancy*"``) to match bracketed values, or match
        on a pattern with no special characters (e.g. ``"pristine"``)
        for a true exact match.

    Returns
    -------
    List[dict]
        Matching structures

    Examples
    --------
    >>> exact = FilterByOperationsShort(structure_container=container, pattern="pristine")
    >>> contains = FilterByOperationsShort(structure_container=container, pattern="vacancy*")
    """
    return structure_container.filter_by_operations_short(pattern)


@as_function_node("structures")
def FilterByOperationsContains(
    structure_container: StructureContainer, operation_type: str
) -> list[dict]:
    """
    Filter structures whose operations contain a specific type.

    Parameters
    ----------
    structure_container : StructureContainer
        The container to filter
    operation_type : str
        Type of operation (e.g., "vacancy", "substitution")

    Returns
    -------
    List[dict]
        Structures containing the operation

    Examples
    --------
    >>> all_vacancies = FilterByOperationsContains(structure_container=container, operation_type="vacancy")
    """
    return structure_container.filter_by_operations_contains(operation_type)


@as_function_node("structure")
def FilterByUniqueId(
    structure_container: StructureContainer, unique_id: str
) -> dict | None:
    """
    Get structure by unique ID.

    Parameters
    ----------
    structure_container : StructureContainer
        The container to search
    unique_id : str
        Unique identifier to find

    Returns
    -------
    dict or None
        Structure dict or None if not found
    """
    return structure_container.filter_by_unique_id(unique_id)


@as_function_node("structures")
def FilterByNumberOfAtoms(
    structure_container: StructureContainer, n_atoms: int
) -> list[dict]:
    """
    Get structures with a specific number of atoms.

    Parameters
    ----------
    structure_container : StructureContainer
        The container to filter
    n_atoms : int
        Number of atoms to match

    Returns
    -------
    List[dict]
        Structures with exactly n_atoms atoms

    Examples
    --------
    >>> structures_108 = FilterByNumberOfAtoms(structure_container=container, n_atoms=108)
    """
    return structure_container.filter_by_number_of_atoms(n_atoms)


@as_function_node("structures")
def FilterByElementCount(
    structure_container: StructureContainer,
    element: str,
    min_count: int | None = None,
    max_count: int | None = None,
    exact_count: int | None = None,
) -> list[dict]:
    """
    Get structures matching element count criteria.

    Parameters
    ----------
    structure_container : StructureContainer
        The container to filter
    element : str
        Chemical symbol (e.g., 'Al', 'Mg')
    min_count : int or None
        Minimum count of this element
    max_count : int or None
        Maximum count of this element
    exact_count : int or None
        Exact count of this element (overrides min/max)

    Returns
    -------
    List[dict]
        Structures matching the element count criteria

    Examples
    --------
    >>> single_mg = FilterByElementCount(structure_container=container, element='Mg', exact_count=1)
    >>> few_mg = FilterByElementCount(structure_container=container, element='Mg', min_count=0, max_count=5)
    """
    return structure_container.filter_by_element_count(
        element, min_count, max_count, exact_count
    )


@as_function_node("structures")
def FilterByStoichiometry(
    structure_container: StructureContainer, formula_pattern: str | None = None
) -> list[dict]:
    """
    Get structures matching a stoichiometry pattern.

    Parameters
    ----------
    structure_container : StructureContainer
        The container to filter
    formula_pattern : str or None
        Chemical formula pattern to match. Supports wildcards.

    Returns
    -------
    List[dict]
        Structures matching the stoichiometry pattern

    Examples
    --------
    >>> exact = FilterByStoichiometry(structure_container=container, formula_pattern='Al107Mg1')
    >>> single_mg = FilterByStoichiometry(structure_container=container, formula_pattern='*Mg1*')
    """
    return structure_container.filter_by_stoichiometry(formula_pattern)


@as_function_node("structures")
def FilterByParent(
    structure_container: StructureContainer, parent_index: int
) -> list[dict]:
    """
    Get structures that have a specific parent structure.

    Parameters
    ----------
    structure_container : StructureContainer
        The container to filter
    parent_index : int
        Absolute index of the parent structure

    Returns
    -------
    List[dict]
        Direct children of the specified parent

    Examples
    --------
    >>> children_0 = FilterByParent(structure_container=container, parent_index=0)
    """
    return structure_container.filter_by_parent(parent_index)


# Not exposed as @as_function_node: takes a Callable, which can't cross a
# node-graph port. Plain Python helper for direct scripting use only.
def filter_by_condition(
    structure_container: StructureContainer, condition: Callable[[dict], bool]
) -> list[dict]:
    """
    Filter by custom function.

    Parameters
    ----------
    structure_container : StructureContainer
        The container to filter
    condition : Callable[[dict], bool]
        Function that takes a structure dict and returns True/False

    Returns
    -------
    List[dict]
        Structures matching the condition

    Examples
    --------
    >>> gen1_vacancies = filter_by_condition(
    ...     container,
    ...     lambda s: s['generation'] == 1 and 'vacancy' in s['operations_short']
    ... )
    """
    return structure_container.filter_by_condition(condition)


# ----------------------------------------------------------------------
# Selection Functions
# ----------------------------------------------------------------------


@as_function_node("structure")
def GetStructure(structure_container: StructureContainer, index: int) -> dict:
    """
    Get structure by absolute index.

    Parameters
    ----------
    structure_container : StructureContainer
        The container
    index : int
        Absolute index of the structure

    Returns
    -------
    dict
        Structure dictionary

    Raises
    ------
    IndexError
        If index is out of range
    """
    return structure_container.get_structure(index)


@as_function_node("index")
def FindStructureIndex(
    structure_container: StructureContainer, atoms: Atoms, tolerance: float = 1e-6
) -> int | None:
    """
    Return the absolute row index of a structure, or None if absent.

    Checks for identity, then numerical equality of stoichiometry,
    positions, cell, and chemical symbols.

    Parameters
    ----------
    structure_container : StructureContainer
        The container to search
    atoms : Atoms
        Structure to look for
    tolerance : float
        Numerical tolerance for comparing positions and cell

    Returns
    -------
    int or None
        Absolute row index of the matching structure, or None if not found
    """
    return structure_container.find_structure_index(atoms, tolerance)


@as_function_node("structures")
def GetPristineStructures(structure_container: StructureContainer) -> list[dict]:
    """
    Get all pristine structures.

    Parameters
    ----------
    structure_container : StructureContainer
        The container

    Returns
    -------
    List[dict]
        All pristine structures
    """
    return structure_container.get_pristine_structures()


@as_function_node("structures")
def GetDefectStructures(structure_container: StructureContainer) -> list[dict]:
    """
    Get all defect structures.

    Parameters
    ----------
    structure_container : StructureContainer
        The container

    Returns
    -------
    List[dict]
        All defect structures
    """
    return structure_container.get_defect_structures()


@as_function_node("index")
def LatestPristineIndex(structure_container: StructureContainer) -> int:
    """
    Absolute row index of the most recently added pristine structure.

    Parameters
    ----------
    structure_container : StructureContainer
        The container

    Returns
    -------
    int
        Index of the latest pristine structure

    Raises
    ------
    ValueError
        If no pristine structure exists
    """
    return structure_container.latest_pristine_index()


# ----------------------------------------------------------------------
# Index Resolution Functions
# ----------------------------------------------------------------------


@as_function_node("index")
def ResolveDefectRow(
    structure_container: StructureContainer, relative_index: int
) -> int:
    """
    Convert a relative defect index to an absolute structure index.

    Parameters
    ----------
    structure_container : StructureContainer
        The container
    relative_index : int
        Relative defect index (0-based, negative counts from end)

    Returns
    -------
    int
        Absolute row index

    Examples
    --------
    >>> first = ResolveDefectRow(structure_container=container, relative_index=0)  # first defect
    >>> latest = ResolveDefectRow(structure_container=container, relative_index=-1)  # most recent defect
    """
    return structure_container.resolve_defect_row(relative_index)


@as_function_node("index")
def ResolveAnyRow(structure_container: StructureContainer, relative_index: int) -> int:
    """
    Convert a relative index to an absolute structure index.

    Parameters
    ----------
    structure_container : StructureContainer
        The container
    relative_index : int
        Relative index (0-based, negative counts from end)

    Returns
    -------
    int
        Absolute row index

    Examples
    --------
    >>> latest = ResolveAnyRow(structure_container=container, relative_index=-1)
    >>> first = ResolveAnyRow(structure_container=container, relative_index=0)
    """
    return structure_container.resolve_any_row(relative_index)


# ============================================================================
# Combined Defect Creation Functions
# ============================================================================


def _resolve_parent(
    container: StructureContainer,
    parent_defect_index: int | None = None,
    input_structure: Atoms | None = None,
) -> int:
    """
    Resolve which structure to use as parent.
    Returns absolute index of the parent structure.

    Resolution modes (pyiron_nodes pattern):
    1. PRIORITY: Explicit defect index (parent_defect_index)
       - Takes precedence over input_structure
    2. Input structure object (input_structure)
       - Used if parent_defect_index is None
    3. Fallback: Latest pristine (if both None)

    Warning
    -------
    If both parent_defect_index and input_structure are provided,
    parent_defect_index takes precedence and input_structure is IGNORED.
    A clear warning is displayed in this case.
    """
    import warnings
    import numpy as np

    # Warning: Both parameters provided (pyiron_nodes behavior)
    if parent_defect_index is not None and input_structure is not None:
        warnings.warn(
            f"Both parent_defect_index ({parent_defect_index}) and input_structure provided. "
            f"parent_defect_index will be used and input_structure will be ignored. "
            f"To avoid this warning, provide only one of these parameters.",
            UserWarning,
            stacklevel=3,
        )

    # Case 1 (PRIORITY): Explicit defect index - pyiron_nodes behavior
    if parent_defect_index is not None:
        # Handle negative indices: resolve relative defect index to absolute
        if parent_defect_index < 0:
            parent_defect_index = container.resolve_defect_row(parent_defect_index)

        # Now validate the absolute index
        if not (0 <= parent_defect_index < len(container)):
            raise IndexError(
                f"Defect index {parent_defect_index} out of range. "
                f"Container has {len(container)} structures (indices 0 to {len(container)-1})."
            )
        if container._structures[parent_defect_index]["is_pristine"]:
            raise ValueError(
                f"Index {parent_defect_index} is a pristine structure. "
                f"Use input_structure parameter instead if you want to work with a pristine structure."
            )
        return parent_defect_index

    # Case 2: Input structure provided
    if input_structure is not None:
        input_structure = ensure_uids(input_structure)
        for idx, s in enumerate(container._structures):
            s_structure = ensure_uids(s["structure"])
            # Guard: skip if atom counts differ (avoids broadcast error)
            if len(s_structure) != len(input_structure):
                continue
            if (
                np.allclose(s_structure.positions, input_structure.positions)
                and np.allclose(s_structure.cell, input_structure.cell)
                and s_structure.get_chemical_symbols()
                == input_structure.get_chemical_symbols()
            ):
                return idx
        return container.add_pristine(input_structure)

    # Case 3: Default to latest pristine
    pristine_indices = [
        i for i, s in enumerate(container._structures) if s["is_pristine"]
    ]
    if not pristine_indices:
        raise ValueError(
            "No pristine structure found in the container. "
            "Add one first using container.add_pristine(atoms)."
        )
    return pristine_indices[-1]


@as_function_node
def CreateDefectFromIds(
    structure_container: StructureContainer,
    defect_type: Literal["vacancy", "substitution", "interstitial"],
    atom_ids: list[int] | None = None,
    to_element: str | None = None,
    sublattice: np.ndarray | None = None,
    site_ids: list[int] | None = None,
    element: str | None = None,
    parent_defect_index: int | None = None,
    input_structure: Atoms | None = None,
    forbid_atom_ids: list[int] | None = None,
    protect_history: bool = False,
) -> StructureContainer:
    """
    Create defects at specific, explicitly chosen sites. Merges
    ``create_vacancy_from_ids``, ``create_substitution_from_ids``, and
    ``create_interstitial_from_ids`` into a single dispatcher; the
    per-type logic is unchanged, only the entry point is shared.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify
    defect_type : {"vacancy", "substitution", "interstitial"}
        Which kind of defect to create. Determines which of the
        type-specific parameters below apply.
    atom_ids : List[int] or None
        Vacancy / substitution only, required. Specific atom indices in the
        host structure to remove or substitute.
    to_element : str or None
        Substitution only, required. Element to substitute with.
    sublattice : (N, 3) array-like or None
        Interstitial only, required. Cartesian coordinates (Å) of the
        interstitial sublattice — e.g. the output of
        ``GetVoronoiInterstitialSites``.
    site_ids : List[int] or None
        Interstitial only, required. Indices into ``sublattice`` selecting
        which sites to occupy.
    element : str or None
        Interstitial only, required. Chemical symbol of the atom to insert.
    parent_defect_index : int or None
        Index of defect structure to build on (None = use default parent)
    input_structure : Atoms or None
        Structure to use as parent (None = use container default)
    forbid_atom_ids : list of int or None
        Vacancy / substitution only. Atom IDs to exclude from selection.
    protect_history : bool
        Vacancy / substitution only. Protect atoms from previous defects.

    Returns
    -------
    StructureContainer with new defect added

    Raises
    ------
    ValueError
        If a required parameter for the selected ``defect_type`` is
        missing (``atom_ids`` for vacancy/substitution, ``to_element``
        for substitution, ``sublattice``/``site_ids``/``element`` for
        interstitial); if ``sublattice`` doesn't have shape ``(N, 3)``;
        if ``site_ids`` is empty; if no valid indices remain after
        applying ``forbid_atom_ids``; or if ``defect_type`` is not one
        of ``"vacancy"``, ``"substitution"``, ``"interstitial"``.
    IndexError
        If any ``atom_ids`` value is out of range for the host
        structure, or any ``site_ids`` value is out of range for
        ``sublattice``.
    """
    import numpy as np

    container = structure_container

    # Resolve parent structure
    parent_idx = _resolve_parent(container, parent_defect_index, input_structure)
    parent = container.get_structure(parent_idx)

    atoms = ensure_uids(parent["structure"]).copy()
    existing_events = parent["events"].copy()

    # Find pristine ancestor
    pristine_idx = container._find_pristine_index(parent_idx)

    if defect_type == "vacancy":
        if atom_ids is None:
            raise ValueError("atom_ids is required for defect_type='vacancy'")

        # Validate indices
        atoms_for_validation = ensure_uids(atoms).copy()
        validate_atoms_arrays(atoms_for_validation)
        uids = atoms_for_validation.arrays[UID_KEY].astype(int)
        syms = np.array(atoms_for_validation.get_chemical_symbols(), dtype=object)

        # Check all indices are valid
        for idx in atom_ids:
            if not (0 <= idx < len(atoms_for_validation)):
                raise IndexError(
                    f"Vacancy index {idx} is out of range for a structure with {len(atoms_for_validation)} atoms."
                )

        # Build forbid set
        forbid = set() if forbid_atom_ids is None else set(map(int, forbid_atom_ids))
        if protect_history:
            forbid |= _protected_uids_from_events(existing_events)

        # Filter out forbidden indices
        valid_indices = [idx for idx in atom_ids if int(uids[idx]) not in forbid]

        if len(valid_indices) == 0:
            raise ValueError("No valid indices after applying forbid_atom_ids filters.")

        # Record all events
        new_events = []
        for i in valid_indices:
            site_uid = int(uids[i])
            new_events.append(
                {
                    "type": "vacancy",
                    "removed_element": str(syms[i]),
                    "site_uid": site_uid,
                    "site_pos0": atoms_for_validation.positions[i].tolist(),
                    "pos_at_removal": atoms_for_validation.positions[i].tolist(),
                }
            )

        # Delete atoms (in reverse order to maintain indices)
        for i in sorted(map(int, valid_indices), reverse=True):
            del atoms_for_validation[i]

        validate_atoms_arrays(atoms_for_validation)

        # Build operation string
        if len(valid_indices) > 1:
            operation_str = f"vacancy[{len(valid_indices)}]"
        else:
            operation_str = f"vacancy[{new_events[0]['site_uid']}]"

        result_atoms = atoms_for_validation

    elif defect_type == "substitution":
        if atom_ids is None:
            raise ValueError("atom_ids is required for defect_type='substitution'")
        if to_element is None:
            raise ValueError("to_element is required for defect_type='substitution'")

        # Validate indices
        uids = atoms.arrays[UID_KEY].astype(int)
        syms = np.array(atoms.get_chemical_symbols(), dtype=object)

        # Check all indices are valid
        for idx in atom_ids:
            if not (0 <= idx < len(atoms)):
                raise IndexError(
                    f"Substitution index {idx} is out of range for a structure with {len(atoms)} atoms."
                )

        # Build forbid set
        forbid = set() if forbid_atom_ids is None else set(map(int, forbid_atom_ids))
        if protect_history:
            forbid |= _protected_uids_from_events(existing_events)

        # Filter out forbidden indices
        valid_indices = [idx for idx in atom_ids if int(uids[idx]) not in forbid]

        if len(valid_indices) == 0:
            raise ValueError("No valid indices after applying forbid_atom_ids filters.")

        # Record all events and apply substitutions
        new_events = []
        for i in valid_indices:
            atom_uid = int(uids[i])
            site_uid = atom_uid
            from_element = str(syms[i])
            site_pos0 = atoms.positions[i].tolist()

            new_events.append(
                {
                    "type": "substitution",
                    "from": from_element,
                    "to": str(to_element),
                    "atom_uid": atom_uid,
                    "site_uid": site_uid,
                    "site_pos0": site_pos0,
                    "pos_at_creation": atoms.positions[i].tolist(),
                }
            )

            atoms[i].symbol = to_element

        # Build operation string
        if len(valid_indices) > 1:
            # Use first substitution as reference
            operation_str = f"substitution[{len(valid_indices)}:{new_events[0]['from']}->{to_element}]"
        else:
            operation_str = f"substitution[{new_events[0]['from']}->{to_element}]"

        result_atoms = atoms

    elif defect_type == "interstitial":
        if sublattice is None or site_ids is None or element is None:
            raise ValueError(
                "sublattice, site_ids, and element are required for defect_type='interstitial'"
            )

        # Validate sublattice and site_ids
        sublattice_arr = np.asarray(sublattice, float)
        if sublattice_arr.ndim != 2 or sublattice_arr.shape[1] != 3:
            raise ValueError(
                f"sublattice must have shape (N, 3), got {sublattice_arr.shape}."
            )
        if len(site_ids) == 0:
            raise ValueError("site_ids is empty — provide at least one site index.")
        for sid in site_ids:
            if not (0 <= sid < len(sublattice_arr)):
                raise IndexError(
                    f"site_id {sid} is out of range for sublattice with {len(sublattice_arr)} sites."
                )

        validate_atoms_arrays(atoms)

        # Insert one atom per requested site
        new_events = []
        for sid in site_ids:
            pos = sublattice_arr[int(sid)].tolist()
            new_uid = next_uid(atoms)
            atoms = append_atom_with_uid(atoms, element, pos)
            validate_atoms_arrays(atoms)
            new_events.append(
                {
                    "type": "interstitial",
                    "element": str(element),
                    "atom_uid": int(new_uid),
                    "pos0": pos,
                    "site_label": f"site_{int(sid)}",
                }
            )

        n = len(site_ids)
        operation_str = (
            f"interstitial[{n}:{element}]" if n > 1 else f"interstitial[{element}]"
        )

        result_atoms = atoms

    else:
        raise ValueError(
            f"Unknown defect_type '{defect_type}'. "
            "Must be one of 'vacancy', 'substitution', 'interstitial'."
        )

    all_new_events = existing_events + new_events

    # Store in container
    container.add_defect(
        atoms=result_atoms,
        operation=operation_str,
        pristine_index=pristine_idx,
        parent_index=parent_idx,
        events=all_new_events,
        metadata={"parent_index": parent_idx, "pristine_index": pristine_idx},
    )

    out_container = container
    return out_container


@as_function_node
def CreateDefectBatchFromIds(
    structure_container: StructureContainer,
    defect_type: Literal["vacancy", "substitution", "interstitial"],
    target_indices: list[int],
    atom_ids: list[int] | None = None,
    to_element: str = "Mg",
    sublattice: np.ndarray | None = None,
    element: str | None = None,
    site_ids: list[int] | None = None,
    separate_structures: bool = True,
    forbid_atom_ids: list[int] | None = None,
    protect_history: bool = False,
) -> StructureContainer:
    """
    Apply specific defect sites to multiple parent structures. Merges
    ``create_vacancy_batch_from_ids``, ``create_substitution_batch_from_ids``,
    and ``create_interstitial_batch_from_ids`` into a single dispatcher around
    :func:`CreateDefectFromIds`; the per-type logic is unchanged.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify
    defect_type : {"vacancy", "substitution", "interstitial"}
        Which kind of defect to create. Determines which of the
        type-specific parameters below apply.
    target_indices : List[int]
        Absolute indices of structures to modify
    atom_ids : List[int] or None
        Vacancy / substitution only, required. Specific atom indices to
        remove or substitute for each target structure.
    to_element : str
        Substitution only. Element to substitute with.
    sublattice : (N, 3) array-like or None
        Interstitial only, required. Cartesian coordinates (Å) of the
        interstitial sublattice.
    element : str or None
        Interstitial only, required. Chemical symbol of the atom to insert.
    site_ids : list of int or None
        Interstitial only. Indices into ``sublattice`` selecting which sites
        to use. ``None`` (default) uses every site in ``sublattice``.
    separate_structures : bool (default=True)
        True: Create separate structures for each id in atom_ids/site_ids
        False: Create one structure with all defects per target
    forbid_atom_ids : list of int or None
        Vacancy / substitution only. Atom IDs to exclude from selection.
    protect_history : bool
        Vacancy / substitution only. Protect atoms from previous defects.

    Returns
    -------
    StructureContainer with all new structures added

    Raises
    ------
    ValueError
        If ``sublattice``/``element`` are missing for
        ``defect_type="interstitial"``, or any of the errors documented
        in :func:`CreateDefectFromIds` (missing required parameters,
        empty candidate pools, unknown ``defect_type``) while processing
        an individual target/id.
    IndexError
        If any ``atom_ids``/``site_ids`` value, or any entry of
        ``target_indices``, is out of range -- see
        :func:`CreateDefectFromIds`.

    Notes
    -----
    Use container methods like filter_by_generation(), filter_by_pristine_structures(),
    etc. to get the target_indices before calling this function.
    """
    import numpy as np

    container = structure_container
    rows_to_modify = target_indices

    if defect_type == "interstitial":
        if sublattice is None or element is None:
            raise ValueError(
                "sublattice and element are required for defect_type='interstitial'"
            )
        sublattice_arr = np.asarray(sublattice, float)
        effective_ids = (
            list(range(len(sublattice_arr))) if site_ids is None else list(site_ids)
        )
    else:
        sublattice_arr = None
        effective_ids = atom_ids

    if separate_structures:
        # Create separate structures for each id
        for parent_idx in rows_to_modify:
            _pd = (
                parent_idx
                if not container._structures[parent_idx]["is_pristine"]
                else None
            )
            _is = (
                container._structures[parent_idx]["structure"]
                if container._structures[parent_idx]["is_pristine"]
                else None
            )
            for single_id in effective_ids:
                container = CreateDefectFromIds._original_func(
                    structure_container=container,
                    defect_type=defect_type,
                    atom_ids=[single_id] if defect_type != "interstitial" else None,
                    to_element=to_element,
                    sublattice=sublattice_arr,
                    site_ids=[single_id] if defect_type == "interstitial" else None,
                    element=element,
                    parent_defect_index=_pd,
                    input_structure=_is,
                    forbid_atom_ids=forbid_atom_ids,
                    protect_history=protect_history,
                )
    else:
        # Apply all ids to each target structure
        for parent_idx in rows_to_modify:
            _pd = (
                parent_idx
                if not container._structures[parent_idx]["is_pristine"]
                else None
            )
            _is = (
                container._structures[parent_idx]["structure"]
                if container._structures[parent_idx]["is_pristine"]
                else None
            )
            container = CreateDefectFromIds._original_func(
                structure_container=container,
                defect_type=defect_type,
                atom_ids=effective_ids if defect_type != "interstitial" else None,
                to_element=to_element,
                sublattice=sublattice_arr,
                site_ids=effective_ids if defect_type == "interstitial" else None,
                element=element,
                parent_defect_index=_pd,
                input_structure=_is,
                forbid_atom_ids=forbid_atom_ids,
                protect_history=protect_history,
            )

    out_container = container
    return out_container


@as_function_node
def CreateDefectFromSeed(
    structure_container: StructureContainer,
    defect_type: Literal["vacancy", "substitution", "interstitial"],
    n: int = 1,
    seed: int | None = None,
    vacancy_element: str | list[str] | None = None,
    from_element: str = "Al",
    to_element: str = "Mg",
    sublattice: np.ndarray | None = None,
    element: str | None = None,
    parent_defect_index: int | None = None,
    input_structure: Atoms | None = None,
    forbid_uids: list[int] | None = None,
    protect_history: bool = False,
) -> StructureContainer:
    """
    Create random defects with a reproducible seed. Merges
    ``create_vacancy_from_seed``, ``create_substitution_from_seed``, and
    ``create_interstitial_from_seed`` into a single dispatcher; the
    per-type logic is unchanged, only the entry point is shared.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify
    defect_type : {"vacancy", "substitution", "interstitial"}
        Which kind of defect to create. Determines which of the
        type-specific parameters below apply.
    n : int
        Number of defects to create
    seed : int or None
        Random seed for reproducibility
    vacancy_element : str, list of str, or None
        Vacancy only. Element to remove (e.g., 'Al'). None = any element.
        Pass a list to specify the exact per-vacancy elements
        (e.g., ['Al', 'Mg'] removes one Al and one Mg).
        When a list is given, n must equal len(vacancy_element).
    from_element : str
        Substitution only. Element to replace.
    to_element : str
        Substitution only. Element to substitute with.
    sublattice : (N, 3) array-like or None
        Interstitial only, required. Cartesian coordinates (Å) of all
        candidate interstitial sites — e.g. the ``all_sites`` output of
        ``GetVoronoiInterstitialSites``.
    element : str or None
        Interstitial only, required. Chemical symbol of the atom to insert.
    parent_defect_index : int or None
        Index of defect structure to build on (None = use default parent)
    input_structure : Atoms or None
        Structure to use as parent (None = use container default)
    forbid_uids : list of int or None
        Vacancy / substitution only. UIDs to exclude from selection.
    protect_history : bool
        Vacancy / substitution only. Protect atoms from previous defects.

    Returns
    -------
    StructureContainer with new defect added

    Raises
    ------
    ValueError
        If a required parameter for the selected ``defect_type`` is
        missing (``sublattice``/``element`` for interstitial); if
        ``vacancy_element`` is a list whose length doesn't equal ``n``;
        if not enough candidates remain to satisfy ``n`` for the chosen
        element/defect type; if ``sublattice`` doesn't have shape
        ``(N, 3)`` or is empty; or if ``defect_type`` is not one of
        ``"vacancy"``, ``"substitution"``, ``"interstitial"``.
    """
    import numpy as np

    container = structure_container

    # Resolve parent structure
    parent_idx = _resolve_parent(container, parent_defect_index, input_structure)
    parent = container.get_structure(parent_idx)

    atoms = ensure_uids(parent["structure"]).copy()
    existing_events = parent["events"].copy()

    # Find pristine ancestor
    pristine_idx = container._find_pristine_index(parent_idx)

    if defect_type == "vacancy":
        # Set up random selection
        rng = np.random.default_rng(seed if seed is not None else 0)
        atoms_copy = ensure_uids(atoms).copy()
        validate_atoms_arrays(atoms_copy)
        uids = atoms_copy.arrays[UID_KEY].astype(int)
        syms = np.array(atoms_copy.get_chemical_symbols(), dtype=object)

        # Build forbid set
        forbid = set() if forbid_uids is None else set(map(int, forbid_uids))
        if protect_history:
            forbid |= _protected_uids_from_events(existing_events)

        # Find candidates
        if isinstance(vacancy_element, list):
            if n != len(vacancy_element):
                raise ValueError(
                    f"n={n} must equal len(vacancy_element)={len(vacancy_element)} when vacancy_element is a list"
                )
            pick_idx = []
            used = set()
            for elem in vacancy_element:
                cand = [
                    i
                    for i in range(len(atoms_copy))
                    if syms[i] == elem and int(uids[i]) not in forbid and i not in used
                ]
                if len(cand) == 0:
                    raise ValueError(f"No candidates left for element '{elem}'")
                chosen = int(rng.choice(cand))
                pick_idx.append(chosen)
                used.add(chosen)
        elif vacancy_element is None:
            cand = [i for i in range(len(atoms_copy)) if int(uids[i]) not in forbid]
            if len(cand) < n:
                raise ValueError(f"Not enough candidates: need {n}, have {len(cand)}")
            pick_idx = rng.choice(cand, size=int(n), replace=False).tolist()
        else:
            cand = [
                i
                for i in range(len(atoms_copy))
                if syms[i] == vacancy_element and int(uids[i]) not in forbid
            ]
            if len(cand) < n:
                raise ValueError(f"Not enough candidates: need {n}, have {len(cand)}")
            pick_idx = rng.choice(cand, size=int(n), replace=False).tolist()

        # Record all events
        new_events = []
        for i in pick_idx:
            site_uid = int(uids[i])
            new_events.append(
                {
                    "type": "vacancy",
                    "removed_element": str(syms[i]),
                    "site_uid": site_uid,
                    "site_pos0": atoms_copy.positions[i].tolist(),
                    "pos_at_removal": atoms_copy.positions[i].tolist(),
                }
            )

        # Delete atoms
        for i in sorted(map(int, pick_idx), reverse=True):
            del atoms_copy[i]

        validate_atoms_arrays(atoms_copy)

        # Build operation string
        if n > 1:
            operation_str = f"vacancy[{n}]"
        else:
            operation_str = f"vacancy[{new_events[0]['site_uid']}]"

        result_atoms = atoms_copy

    elif defect_type == "substitution":
        # Set up random selection
        rng = np.random.default_rng(seed if seed is not None else 0)
        uids = atoms.arrays[UID_KEY].astype(int)
        syms = np.array(atoms.get_chemical_symbols(), dtype=object)

        # Build forbid set
        forbid = set() if forbid_uids is None else set(map(int, forbid_uids))
        if protect_history:
            forbid |= _protected_uids_from_events(existing_events)

        # Find candidates
        cand = [
            i
            for i in range(len(atoms))
            if syms[i] == from_element and int(uids[i]) not in forbid
        ]

        if n > len(cand):
            raise ValueError(
                f"Not enough candidates to substitute {from_element}->{to_element}: need {n}, have {len(cand)}"
            )

        pick_idx = rng.choice(cand, size=int(n), replace=False)

        # Record events and apply substitutions
        new_events = []
        for i in pick_idx:
            atom_uid = int(uids[i])
            site_uid = atom_uid
            site_pos0 = atoms.positions[i].tolist()

            new_events.append(
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

        # Build operation string
        if n > 1:
            operation_str = f"substitution[{n}:{from_element}->{to_element}]"
        else:
            operation_str = f"substitution[{from_element}->{to_element}]"

        result_atoms = atoms

    elif defect_type == "interstitial":
        if sublattice is None or element is None:
            raise ValueError(
                "sublattice and element are required for defect_type='interstitial'"
            )

        # Validate sublattice
        sublattice_arr = np.asarray(sublattice, float)
        if sublattice_arr.ndim != 2 or sublattice_arr.shape[1] != 3:
            raise ValueError(
                f"sublattice must have shape (N, 3), got {sublattice_arr.shape}."
            )
        if len(sublattice_arr) == 0:
            raise ValueError("sublattice is empty — no candidate sites to sample from.")
        if n > len(sublattice_arr):
            raise ValueError(
                f"Requested n={n} interstitials but sublattice only has {len(sublattice_arr)} sites."
            )

        arrays_copy = ensure_uids(atoms).copy()
        validate_atoms_arrays(arrays_copy)

        # Sample without replacement
        rng = np.random.default_rng(seed if seed is not None else 0)
        picked_ids = rng.choice(len(sublattice_arr), size=int(n), replace=False)

        new_events = []
        for sid in picked_ids:
            pos = sublattice_arr[int(sid)].tolist()
            new_uid = next_uid(arrays_copy)
            arrays_copy = append_atom_with_uid(arrays_copy, element, pos)
            validate_atoms_arrays(arrays_copy)
            new_events.append(
                {
                    "type": "interstitial",
                    "element": str(element),
                    "atom_uid": int(new_uid),
                    "pos0": pos,
                    "site_label": f"site_{int(sid)}",
                }
            )

        operation_str = (
            f"interstitial[{n}:{element}]" if n > 1 else f"interstitial[{element}]"
        )

        result_atoms = arrays_copy

    else:
        raise ValueError(
            f"Unknown defect_type '{defect_type}'. "
            "Must be one of 'vacancy', 'substitution', 'interstitial'."
        )

    all_new_events = existing_events + new_events

    # Store in container
    container.add_defect(
        atoms=result_atoms,
        operation=operation_str,
        pristine_index=pristine_idx,
        parent_index=parent_idx,
        events=all_new_events,
        metadata={
            "parent_index": parent_idx,
            "pristine_index": pristine_idx,
            "seed": seed,
        },
    )

    out_container = container
    return out_container


@as_function_node
def CreateDefectBatchFromSeed(
    structure_container: StructureContainer,
    defect_type: Literal["vacancy", "substitution", "interstitial"],
    target_indices: list[int],
    n: int = 1,
    seed: int | None = None,
    vacancy_element: str | list[str] | None = None,
    from_element: str = "Al",
    to_element: str = "Mg",
    sublattice: np.ndarray | None = None,
    element: str | None = None,
    forbid_uids: list[int] | None = None,
    protect_history: bool = False,
    n_structures: int = 1,
) -> StructureContainer:
    """
    Apply random defects to multiple structures with reproducible seeds.
    Merges ``create_vacancy_batch_from_seed``, ``create_substitution_batch_from_seed``,
    and ``create_interstitial_batch_from_seed`` into a single dispatcher around
    :func:`CreateDefectFromSeed`; the per-type logic is unchanged.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify
    defect_type : {"vacancy", "substitution", "interstitial"}
        Which kind of defect to create. Determines which of the
        type-specific parameters below apply.
    target_indices : List[int]
        Absolute indices of structures to modify
    n : int
        Number of defects to create per structure.
        When vacancy_element is a list, n must equal len(vacancy_element).
    seed : int or None
        Base random seed (each structure uses different seeds)
    vacancy_element : str, list of str, or None
        Vacancy only. Element to remove (e.g., 'Al'). None = any element.
        Pass a list to fix per-vacancy elements (e.g., ['Al', 'Mg']).
    from_element : str
        Substitution only. Element to replace.
    to_element : str
        Substitution only. Element to substitute with.
    sublattice : (N, 3) array-like or None
        Interstitial only, required. Cartesian coordinates (Å) of all
        candidate interstitial sites — e.g. the ``all_sites`` output of
        ``GetVoronoiInterstitialSites``.
    element : str or None
        Interstitial only, required. Chemical symbol of the atom to insert.
    forbid_uids : list of int or None
        Vacancy / substitution only. UIDs to exclude from selection.
    protect_history : bool
        Vacancy / substitution only. Protect atoms from previous defects.
    n_structures : int
        Number of structures to create from each parent. Default=1 (backward compatible).
        If n_structures > 1, creates n_structures copies from each parent index
        with incrementing seeds: seed, seed+1, seed+2, etc.

    Returns
    -------
    StructureContainer with all new structures added

    Raises
    ------
    ValueError
        If ``sublattice``/``element`` are missing for
        ``defect_type="interstitial"``, or any of the errors documented
        in :func:`CreateDefectFromSeed` (missing required parameters,
        not enough candidates, unknown ``defect_type``) while processing
        an individual target/copy.

    Notes
    -----
    Use container methods like filter_by_generation(), filter_by_pristine_structures(),
    etc. to get the target_indices before calling this function.
    """
    import numpy as np

    container = structure_container
    rows_to_modify = target_indices

    sublattice_arr = None
    if defect_type == "interstitial":
        if sublattice is None or element is None:
            raise ValueError(
                "sublattice and element are required for defect_type='interstitial'"
            )
        sublattice_arr = np.asarray(sublattice, float)

    structure_counter = 0
    for parent_idx in rows_to_modify:
        for _copy_idx in range(n_structures):
            structure_seed = seed + structure_counter if seed is not None else None
            container = CreateDefectFromSeed._original_func(
                structure_container=container,
                defect_type=defect_type,
                n=n,
                seed=structure_seed,
                vacancy_element=vacancy_element,
                from_element=from_element,
                to_element=to_element,
                sublattice=sublattice_arr,
                element=element,
                parent_defect_index=(
                    parent_idx
                    if not container._structures[parent_idx]["is_pristine"]
                    else None
                ),
                input_structure=(
                    container._structures[parent_idx]["structure"]
                    if container._structures[parent_idx]["is_pristine"]
                    else None
                ),
                forbid_uids=forbid_uids,
                protect_history=protect_history,
            )
            structure_counter += 1

    out_container = container
    return out_container


# ============================================================================
# Voronoi Interstitial Site Finding
# ============================================================================


def _wrap_frac(frac):
    """Wrap fractional coordinates into [0, 1)."""
    import numpy as np

    return np.mod(np.asarray(frac, dtype=float), 1.0)


def _frac_equiv(a, b, tol=1e-5):
    """Check if two fractional coordinates are equivalent (mod 1)."""
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    d -= np.round(d)
    return np.linalg.norm(d) < tol


def _deduplicate_frac(frac_list, tol=1e-5):
    """Remove duplicate fractional coordinates."""
    import numpy as np

    uniq = []
    for f in frac_list:
        f = _wrap_frac(f)
        if not any(_frac_equiv(f, u, tol=tol) for u in uniq):
            uniq.append(f)
    if len(uniq) == 0:
        return np.empty((0, 3), dtype=float)
    return np.array(uniq, dtype=float)


def _extract_defect_frac_coords(defects):
    """Extract fractional coordinates from defect objects."""
    import numpy as np

    frac_list = []
    for d in defects:
        if hasattr(d, "site") and hasattr(d.site, "frac_coords"):
            frac_list.append(np.array(d.site.frac_coords, dtype=float))
        elif hasattr(d, "defect_site") and hasattr(d.defect_site, "frac_coords"):
            frac_list.append(np.array(d.defect_site.frac_coords, dtype=float))
        else:
            raise AttributeError(
                "Could not extract frac_coords from Voronoi interstitial defect object."
            )
    if len(frac_list) == 0:
        return np.empty((0, 3), dtype=float)
    return np.array(frac_list, dtype=float)


def _periodic_image_points(pos, cell, n_images: int):
    """Extend a point set with periodic images out to n_images shells."""
    import numpy as np

    offsets = np.array(
        [
            [i, j, k]
            for i in range(-n_images, n_images + 1)
            for j in range(-n_images, n_images + 1)
            for k in range(-n_images, n_images + 1)
        ],
        dtype=float,
    )
    return np.vstack([pos + off @ cell for off in offsets])  # (N*(2n+1)^3, 3)


def _filter_cluster_tile_interstitial_candidates(
    raw_candidates,
    pos,
    cell,
    r_min: float,
    cluster_tol: float,
    use_primitive: bool,
    repeat,
):
    """
    Reduce raw tessellation void-center candidates (Voronoi vertices or
    Delaunay circumcenters) to unique, non-overlapping interstitial sites.

    Keeps only candidates inside the primitive cell, drops any closer than
    ``r_min`` to a host atom, merges near-duplicates within ``cluster_tol``,
    and tiles the result across the supercell when ``use_primitive`` is set.
    """
    import numpy as np

    cell_inv = np.linalg.inv(cell)

    # --- keep only candidates inside the primitive unit cell ---
    frac = raw_candidates @ cell_inv
    inside = np.all((frac >= -1e-8) & (frac < 1 - 1e-8), axis=1)
    candidates = raw_candidates[inside]

    if len(candidates) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)

    # --- filter: too close to any host atom (minimum image convention) ---
    diff = candidates[:, None, :] - pos[None, :, :]  # (C, N, 3)
    diff_frac = diff @ cell_inv
    diff_frac -= np.round(diff_frac)
    diff_cart = diff_frac @ cell
    min_dist = np.linalg.norm(diff_cart, axis=-1).min(axis=1)  # (C,)
    candidates = candidates[min_dist >= r_min]

    if len(candidates) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)

    # --- cluster: merge candidates within cluster_tol ---
    used = np.zeros(len(candidates), dtype=bool)
    clusters = []
    for i in range(len(candidates)):
        if used[i]:
            continue
        d = np.linalg.norm(candidates - candidates[i], axis=1)
        mask = d < cluster_tol
        used[mask] = True
        clusters.append(candidates[mask].mean(axis=0))

    all_sites_prim = np.array(clusters, dtype=float)  # void centers in primitive cell
    unique_sites = all_sites_prim.copy()  # one per cluster

    if use_primitive:
        n1, n2, n3 = int(repeat[0]), int(repeat[1]), int(repeat[2])
        tiles = []
        for f in all_sites_prim:
            for i in range(n1):
                for j in range(n2):
                    for k in range(n3):
                        tiles.append(f + np.array([i, j, k], dtype=float) @ cell)
        all_sites = np.array(tiles, dtype=float)
    else:
        all_sites = all_sites_prim

    return unique_sites, all_sites


@as_function_node
def GetVoronoiInterstitialSitesPymatgen(
    atoms: Atoms,
    primitive_atoms: Atoms | None = None,
    repeat: tuple | None = None,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
    dedup_tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find Voronoi interstitial sites via pymatgen (symmetry-aware, slow).

    Uses pymatgen's ``VoronoiInterstitialGenerator`` + ``SpacegroupAnalyzer``
    to return symmetry-unique void centers expanded to all equivalent positions.
    Prefer :func:`GetVoronoiInterstitialSites` (scipy-based, no pymatgen)
    for speed; use this only when you specifically need spacegroup-reduced sites.

    The positions returned depend solely on the host geometry — not on which
    element will be inserted.  This function is the slow, SLURM-submittable
    step of the two-step workflow::

        # Step 1 — slow, submit to SLURM (memory-efficient form)
        unique_sites, all_sites = GetVoronoiInterstitialSitesPymatgen(
            atoms, primitive_atoms=prim, repeat=(3, 3, 3))

        # Step 2 — fast, element-specific
        container = CreateDefectFromIds(
            structure_container=container, defect_type="interstitial",
            sublattice=unique_sites, site_ids=[0], element='Mg')
        container = CreateDefectFromIds(
            structure_container=container, defect_type="interstitial",
            sublattice=unique_sites, site_ids=[0], element='Al')

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object (host supercell). Only used to validate consistency
        when primitive_atoms and repeat are provided; otherwise Voronoi runs
        directly on this structure.
    primitive_atoms : Atoms or None
        Primitive unit cell used to build ``atoms`` via repeat. When provided
        together with ``repeat``, Voronoi tessellation runs on this smaller
        cell (much lower memory) and the resulting sites are tiled into the
        supercell. Both ``primitive_atoms`` and ``repeat`` must be given
        together or not at all.
    repeat : tuple of int or None
        Repetition factors ``(n1, n2, n3)`` such that
        ``primitive_atoms.repeat(repeat)`` reproduces ``atoms``. Must be
        provided together with ``primitive_atoms``.
    symprec : float
        Symmetry tolerance for SpacegroupAnalyzer (default: 1e-3).
    angle_tolerance : float
        Angle tolerance for SpacegroupAnalyzer (default: 5.0).
    dedup_tol : float
        Tolerance for deduplicating symmetry-expanded fractional positions
        (default: 1e-5).

    Returns
    -------
    equivalent_sites : (N, 3) ndarray
        Symmetry-unique Voronoi interstitial sites in Cartesian coordinates (Å).
        One site per symmetry-distinct void; use for systematic exploration.
    all_sites : (M, 3) ndarray
        All symmetry-expanded positions in Cartesian coordinates (Å),
        replicated across the full supercell when ``repeat`` is given.
        Use for random sampling (more variety than equivalent_sites alone).

    Raises
    ------
    ImportError
        If ``structuretoolkit`` is not available (needed to convert the
        host structure to a pymatgen ``Structure``), or if
        ``pymatgen-analysis-defects`` is not available (needed for
        ``VoronoiInterstitialGenerator`` and ``SpacegroupAnalyzer``).
        Reported separately since either package can be missing
        independently of the other.
    ValueError
        If only one of ``primitive_atoms`` / ``repeat`` is provided.

    See Also
    --------
    CreateDefectFromIds : Insert an interstitial at a specific sublattice site.
    CreateDefectFromSeed : Randomly sample an interstitial site from the sublattice.
    """
    import numpy as np

    if not STRUCTURETOOLKIT_AVAILABLE:
        raise ImportError(
            "Voronoi interstitial site finding (pymatgen-based) requires "
            "structuretoolkit. Install it with: pip install structuretoolkit"
        )
    if not PYMATGEN_ANALYSIS_DEFECTS_AVAILABLE:
        raise ImportError(
            "Voronoi interstitial site finding (pymatgen-based) requires the "
            "pymatgen-analysis-defects package. Install it with: "
            "pip install pymatgen-analysis-defects"
        )

    if (primitive_atoms is None) != (repeat is None):
        raise ValueError(
            "primitive_atoms and repeat must be provided together or not at all."
        )

    use_primitive = primitive_atoms is not None

    # Select which structure to run Voronoi on
    work_atoms = primitive_atoms if use_primitive else atoms

    # pymatgen's API requires at least one insert species; the choice does not
    # affect the Voronoi tessellation or symmetry expansion, so we use the
    # first host element as a harmless dummy.
    _dummy = work_atoms.get_chemical_symbols()[0]

    pmg_structure = ase_to_pymatgen(work_atoms)

    # Get defect sites using VoronoiInterstitialGenerator
    gen = VoronoiInterstitialGenerator()
    defects = list(gen.get_defects(pmg_structure, {_dummy}))

    # Extract equivalent (symmetry-unique) fractional coordinates
    frac_equiv = _extract_defect_frac_coords(defects)
    if len(frac_equiv) == 0:
        equivalent_sites = np.empty((0, 3), dtype=float)
        all_sites = np.empty((0, 3), dtype=float)
    else:
        # Use SpacegroupAnalyzer to get symmetry operations
        sga = SpacegroupAnalyzer(
            pmg_structure,
            symprec=float(symprec),
            angle_tolerance=float(angle_tolerance),
        )
        symmops = sga.get_space_group_operations()

        # Expand to all symmetry-equivalent sites within the primitive cell
        frac_all_prim = []
        for f0 in frac_equiv:
            for op in symmops:
                frac_all_prim.append(_wrap_frac(op.operate(f0)))
        frac_all_prim = _deduplicate_frac(frac_all_prim, tol=dedup_tol)

        prim_lattice = pmg_structure.lattice.matrix

        # unique_sites: one representative per distinct void type (never tiled)
        equivalent_sites = frac_equiv @ prim_lattice

        if use_primitive:
            # Tile all primitive-cell void positions across the supercell by
            # adding integer-lattice-vector shifts for every (i, j, k) image.
            n1, n2, n3 = int(repeat[0]), int(repeat[1]), int(repeat[2])
            all_sites_list = []
            for f in frac_all_prim:
                cart_base = f @ prim_lattice
                for i in range(n1):
                    for j in range(n2):
                        for k in range(n3):
                            all_sites_list.append(
                                cart_base
                                + np.array([i, j, k], dtype=float) @ prim_lattice
                            )
            all_sites = np.array(all_sites_list, dtype=float)
        else:
            all_sites = frac_all_prim @ prim_lattice

    return equivalent_sites, all_sites


@as_function_node
def GetVoronoiInterstitialSites(
    atoms: Atoms,
    primitive_atoms: Atoms | None = None,
    repeat: tuple | None = None,
    r_min: float = 0.8,
    cluster_tol: float = 0.5,
    n_images: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find Voronoi interstitial sites via scipy — fast, no pymatgen dependency.

    Voronoi vertices of the atom positions (plus periodic image shells) are
    exactly the void centers of the tessellation.  No symmetry analysis is
    performed; all geometric void centers are returned directly.  The interface
    mirrors :func:`GetDelaunayInterstitialSites` and both are drop-in
    replaceable with each other::

        unique_sites, all_sites = GetVoronoiInterstitialSites(
            atoms, primitive_atoms=prim, repeat=(3, 3, 3))

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object (host supercell).  Used only for validation when
        ``primitive_atoms`` and ``repeat`` are provided; otherwise
        tessellation runs directly on this structure.
    primitive_atoms : Atoms or None
        Primitive unit cell.  When given together with ``repeat``, the
        tessellation runs on the primitive cell and results are tiled.
    repeat : tuple of int or None
        ``(n1, n2, n3)`` such that ``primitive_atoms.repeat(repeat)``
        reproduces ``atoms``.
    r_min : float
        Minimum distance (Å) a void candidate must keep from every host
        atom.  Set ≈ 0.5–1.0 Å; smaller values include tighter voids.
    cluster_tol : float
        Candidates closer than this (Å) are merged into one (centroid).
        Removes near-duplicate Voronoi vertices on or near cell boundaries.
    n_images : int
        Number of periodic image shells to include before tessellation.
        ``n_images=1`` (default) is sufficient for most structures.

    Returns
    -------
    unique_sites : (N, 3) ndarray
        One representative per cluster, in Cartesian coordinates (Å).
        Analogous to the ``equivalent_sites`` output of
        :func:`GetVoronoiInterstitialSitesPymatgen`.  Clusters are
        geometric, not symmetry-derived.
    all_sites : (M, 3) ndarray
        All deduplicated void centers in Cartesian coordinates (Å),
        replicated across the supercell when ``repeat`` is given.

    See Also
    --------
    GetVoronoiInterstitialSitesPymatgen : Symmetry-aware alternative (slower, requires pymatgen).
    GetDelaunayInterstitialSites : Equivalent via Delaunay circumcenters.
    """
    import numpy as np
    from scipy.spatial import Voronoi

    if (primitive_atoms is None) != (repeat is None):
        raise ValueError(
            "primitive_atoms and repeat must be provided together or not at all."
        )

    use_primitive = primitive_atoms is not None
    work_atoms = primitive_atoms if use_primitive else atoms

    cell = work_atoms.get_cell().array  # (3, 3)
    pos = work_atoms.get_positions()  # (N, 3)

    # --- build extended point set with periodic images ---
    pos_ext = _periodic_image_points(pos, cell, n_images)

    # --- Voronoi tessellation: vertices are the void centers ---
    vor = Voronoi(pos_ext)
    candidates_all = vor.vertices  # (V, 3)

    unique_sites, all_sites = _filter_cluster_tile_interstitial_candidates(
        candidates_all, pos, cell, r_min, cluster_tol, use_primitive, repeat
    )

    return unique_sites, all_sites


@as_function_node
def GetDelaunayInterstitialSites(
    atoms: Atoms,
    primitive_atoms: Atoms | None = None,
    repeat: tuple | None = None,
    r_min: float = 0.8,
    cluster_tol: float = 0.5,
    n_images: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find interstitial sites via Delaunay circumcenters — much faster than
    Voronoi tessellation for large or complex primitive cells.

    Circumcenters of Delaunay tetrahedra are natural void centers.  No
    pymatgen dependency; uses only scipy and numpy.  The interface mirrors
    :func:`GetVoronoiInterstitialSites` (and :func:`GetVoronoiInterstitialSitesPymatgen`)
    so all three are drop-in replaceable::

        unique_sites, all_sites = GetDelaunayInterstitialSites(
            atoms, primitive_atoms=prim, repeat=(3, 3, 3))

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object (host supercell).  Used only for validation when
        ``primitive_atoms`` and ``repeat`` are provided; otherwise
        tessellation runs directly on this structure.
    primitive_atoms : Atoms or None
        Primitive unit cell.  When given together with ``repeat``, the
        tessellation runs on the primitive cell and results are tiled.
    repeat : tuple of int or None
        ``(n1, n2, n3)`` such that ``primitive_atoms.repeat(repeat)``
        reproduces ``atoms``.
    r_min : float
        Minimum distance (Å) a void candidate must keep from every host
        atom.  Set ≈ 0.5–1.0 Å; smaller values include tighter voids.
    cluster_tol : float
        Candidates closer than this (Å) are merged into one (takes the
        centroid).  Removes near-duplicate circumcenters.
    n_images : int
        Number of periodic image shells to include before tessellation.
        ``n_images=1`` (default) is sufficient for most structures.

    Returns
    -------
    unique_sites : (N, 3) ndarray
        One representative per cluster, in Cartesian coordinates (Å).
        Analogous to the ``equivalent_sites`` output of
        :func:`GetVoronoiInterstitialSitesPymatgen`.  Note: clusters are
        geometric, not symmetry-derived; for symmetry-unique sites pass
        these through spglib separately.
    all_sites : (M, 3) ndarray
        All deduplicated void centers in Cartesian coordinates (Å),
        replicated across the supercell when ``repeat`` is given.

    See Also
    --------
    GetVoronoiInterstitialSites : Scipy-based Voronoi alternative (same speed, no pymatgen).
    """
    import numpy as np
    from scipy.spatial import Delaunay

    if (primitive_atoms is None) != (repeat is None):
        raise ValueError(
            "primitive_atoms and repeat must be provided together or not at all."
        )

    use_primitive = primitive_atoms is not None
    work_atoms = primitive_atoms if use_primitive else atoms

    cell = work_atoms.get_cell().array  # (3, 3)
    pos = work_atoms.get_positions()  # (N, 3)

    # --- build extended point set with periodic images ---
    pos_ext = _periodic_image_points(pos, cell, n_images)

    # --- Delaunay tessellation ---
    tri = Delaunay(pos_ext)

    # --- circumcenter of each tetrahedron ---
    verts = pos_ext[tri.simplices]  # (T, 4, 3)
    A = verts[:, 1:] - verts[:, :1]  # (T, 3, 3)  edge vectors from v0
    b = 0.5 * (A**2).sum(axis=-1)  # (T, 3)

    # Drop degenerate tetrahedra (coplanar vertices → singular A)
    # before the batched solve; det ≈ 0 identifies them cheaply.
    At = A.swapaxes(1, 2)  # (T, 3, 3)
    nondegenerate = np.abs(np.linalg.det(At)) > 1e-10
    At_nd = At[nondegenerate]
    b_nd = b[nondegenerate]
    v0_nd = verts[nondegenerate, 0]

    # b must be (..., m, 1) for numpy's batched solver
    x = np.linalg.solve(At_nd, b_nd[..., None])[..., 0]  # (T', 3)
    centers = x + v0_nd  # (T', 3)

    unique_sites, all_sites = _filter_cluster_tile_interstitial_candidates(
        centers, pos, cell, r_min, cluster_tol, use_primitive, repeat
    )

    return unique_sites, all_sites
