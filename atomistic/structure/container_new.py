"""
Enhanced StructureContainer Module - GUI-Friendly Version

This module provides an improved StructureContainer class for managing
pristine and defect structures with enhanced features including:
- Unambiguous operation tracking with pipe-separated short form
- Clear lineage tracking with top-level parent/pristine indices
- Duplicate checking for pristine structures
- Table extraction methods
- Structure lookup helpers
- Support for both specific and random defect creation (GUI-friendly)
- Single return pattern for all functions

Version 3.0 - June 2026 (GUI-friendly refactor)
"""

from __future__ import annotations  # Enables lazy imports for type hints

from ase import Atoms
from dataclasses import dataclass, field
from typing import List, Optional, Union, Callable

# Import for Voronoi interstitial site finding
try:
    from structuretoolkit.common import ase_to_pymatgen
    VORONOI_AVAILABLE = True
except ImportError:
    VORONOI_AVAILABLE = False


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


def append_atom_with_uid(atoms: Atoms, symbol: str, position, uid_key: str = "uid") -> Atoms:
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
        raise ValueError(f"uid array has unexpected length {u.shape[0]} for len(atoms)={len(atoms)}")
    
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
    
    _structures: List[dict] = field(default_factory=list)
    
    # ------------------------------------------------------------------
    # Basic Methods
    # ------------------------------------------------------------------
    import pandas as pd
    def to_dataframe(self) -> "pd.DataFrame":
        """Convert internal list to pandas DataFrame on demand."""
        import pandas as pd
        df_data = []
        for s in self._structures:
            df_data.append({
                'structure': s['structure'],
                'unique_id': s['unique_id'],
                'is_pristine': s['is_pristine'],
                'stoichiometry': s['stoichiometry'],
                'generation': s['generation'],
                'pristine_structure_index': s.get('pristine_structure_index', -1),
                'parent_index': s.get('parent_index', -1),
                'operation': s['operation'],
                'operations_short': s.get('operations_short', ''),
                'events': s['events'],
                'metadata': s['metadata'],
                'creation_timestamp': s['creation_timestamp'],
            })
        return pd.DataFrame(df_data)
    
    def get_structure_table(self) -> "pd.DataFrame":
        """Return a copy of the full structure table."""
        return self.to_dataframe()
    
    def get_defect_table(self) -> "pd.DataFrame":
        """Return a filtered copy containing only defect rows."""
        df = self.to_dataframe()
        return df[df['is_pristine'] == False].copy()
    
    def get_pristine_table(self) -> "pd.DataFrame":
        """Return a filtered copy containing only pristine rows."""
        df = self.to_dataframe()
        return df[df['is_pristine'] == True].copy()
    
    # ------------------------------------------------------------------
    # Add Structures
    # ------------------------------------------------------------------
    
    def add_pristine(
        self, 
        atoms: Atoms, 
        unique_id: Optional[str] = None,
        metadata: Optional[dict] = None,
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
                if s['is_pristine']:
                    s_structure = ensure_uids(s['structure'])
                    candidate_stoich = self._get_stoichiometry(atoms)
                    if (self._get_stoichiometry(s_structure) == candidate_stoich and
                        np.allclose(s_structure.positions, atoms.positions, atol=tolerance) and
                        np.allclose(s_structure.cell, atoms.cell, atol=tolerance) and
                        s_structure.get_chemical_symbols() == atoms.get_chemical_symbols()):
                        return idx
        
        entry = {
            'structure': atoms.copy(),
            'unique_id': uid,
            'is_pristine': True,
            'stoichiometry': self._get_stoichiometry(atoms),
            'generation': 0,
            'pristine_structure_index': -1,
            'parent_index': -1,
            'operation': 'pristine',
            'operations_short': 'pristine',
            'events': [],
            'metadata': metadata or {},
            'creation_timestamp': datetime.now(),
        }
        self._structures.append(entry)
        return len(self._structures) - 1
    
    def add_defect(
        self, 
        atoms: Atoms,
        operation: str,
        pristine_index: int,
        parent_index: int,
        events: List[dict],
        unique_id: Optional[str] = None,
        metadata: Optional[dict] = None,
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
            'structure': atoms.copy(),
            'unique_id': uid,
            'is_pristine': False,
            'stoichiometry': self._get_stoichiometry(atoms),
            'generation': self._structures[parent_index]['generation'] + 1,
            'pristine_structure_index': pristine_index,
            'parent_index': parent_index,
            'operation': operation,
            'operations_short': operations_short,
            'events': events,
            'metadata': metadata or {},
            'creation_timestamp': datetime.now(),
        }
        self._structures.append(entry)
        return len(self._structures) - 1
    
    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------
    
    @staticmethod
    def _make_operations_short(events: List[dict]) -> str:
        """
        Create pipe-separated short form from events.
        
        Examples:
          - "vacancy[5]"
          - "substitution[10->Mg]"
          - "vacancy[5]|substitution[10->Mg]"
          - "vacancy[5]|vacancy[10]|substitution[15->Cu]"
        
        This is unambiguous and clearly shows the sequence of operations.
        """
        if not events:
            return 'no_operations'
        
        short_ops = []
        for ev in events:
            t = ev.get('type')
            if t == 'vacancy':
                uid = ev.get('site_uid', '?')
                short_ops.append(f"vacancy[{uid}]")
            elif t == 'substitution':
                from_el = ev.get('from', '?')
                to_el = ev.get('to', '?')
                uid = ev.get('site_uid', '?')
                short_ops.append(f"substitution[{from_el}->{to_el}]")
            elif t == 'interstitial':
                el = ev.get('element', '?')
                uid = ev.get('atom_uid', '?')
                short_ops.append(f"interstitial[{el}]")
        
        return '|'.join(short_ops) if short_ops else 'no_operations'
    
    @staticmethod
    def _get_stoichiometry(atoms: Atoms) -> str:
        """Get stoichiometry string."""
        from collections import Counter
        counts = Counter(atoms.get_chemical_symbols())
        return "".join(f"{el}{counts[el]}" for el in sorted(counts))
    
    def find_structure_index(self, atoms: Atoms, tolerance: float = 1e-6) -> Optional[int]:
        """
        Return the absolute row index of a structure, or None if absent.
        
        Checks for identity, then numerical equality.
        """
        import numpy as np
        atoms = ensure_uids(atoms)
        candidate_stoich = self._get_stoichiometry(atoms)
        
        for row_idx in range(len(self._structures)):
            stored = self._structures[row_idx]
            s_structure = ensure_uids(stored['structure'])
            
            if s_structure is atoms:
                return row_idx
            
            if (self._get_stoichiometry(s_structure) == candidate_stoich and
                np.allclose(s_structure.positions, atoms.positions, atol=tolerance) and
                np.allclose(s_structure.cell, atoms.cell, atol=tolerance) and
                s_structure.get_chemical_symbols() == atoms.get_chemical_symbols()):
                return row_idx
        
        return None
    
    # ------------------------------------------------------------------
    # Filtering Methods
    # ------------------------------------------------------------------
    
    def filter_by_indices(self, indices: List[int]) -> List[dict]:
        """Get structures by absolute indices."""
        return [self._structures[i] for i in indices if i < len(self._structures)]
    
    def filter_by_generation(self, generation: int) -> List[dict]:
        """Get all structures at specific distance from pristine."""
        return [s for s in self._structures if s['generation'] == generation]
    
    def filter_by_operations_short(self, pattern: str) -> List[dict]:
        """
        Filter by operations_short field (supports wildcards).
        
        Examples:
          - "vacancy[5]" - exact match
          - "*vacancy[5]*" - contains vacancy[5]
          - "vacancy[*]|substitution[*]" - multiple patterns
        """
        import fnmatch
        if '*' in pattern or '?' in pattern or '[' in pattern:
            return [s for s in self._structures 
                   if fnmatch.fnmatch(s.get('operations_short', ''), pattern)]
        else:
            return [s for s in self._structures if s.get('operations_short', '') == pattern]
    
    def filter_by_operations_contains(self, operation_type: str) -> List[dict]:
        """
        Filter structures whose operations contain a specific type.
        
        Examples:
          - "vacancy" - all structures with vacancy
          - "substitution" - all structures with substitution
        """
        return [s for s in self._structures if operation_type in s.get('operations_short', '')]
    
    def filter_by_condition(self, condition: Callable[[dict], bool]) -> List[dict]:
        """Filter by custom function."""
        return [s for s in self._structures if condition(s)]
    
    def filter_by_unique_id(self, unique_id: str) -> Optional[dict]:
        """Get structure by unique ID."""
        for s in self._structures:
            if s['unique_id'] == unique_id:
                return s
        return None
    
    def filter_by_max_generation(self, max_generation: int) -> List[dict]:
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
        return [s for s in self._structures if s['generation'] <= max_generation]
    
    def filter_by_number_of_atoms(self, n_atoms: int) -> List[dict]:
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
        return [s for s in self._structures if len(s['structure']) == n_atoms]
    
    def filter_by_element_count(
        self, 
        element: str, 
        min_count: Optional[int] = None, 
        max_count: Optional[int] = None,
        exact_count: Optional[int] = None
    ) -> List[dict]:
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
            syms = s['structure'].get_chemical_symbols()
            counts = Counter(syms)
            element_count = counts.get(element, 0)
            
            if exact_count is not None:
                if element_count == exact_count:
                    matching.append(s)
            else:
                if (min_count is None or element_count >= min_count) and \
                   (max_count is None or element_count <= max_count):
                    matching.append(s)
        
        return matching
    
    def filter_by_stoichiometry(self, formula_pattern: Optional[str] = None) -> List[dict]:
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
            stoich = s['stoichiometry']
            if fnmatch.fnmatch(stoich, formula_pattern):
                matching.append(s)
        
        return matching
    
    def filter_by_parent(self, parent_index: int) -> List[dict]:
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
        return [s for s in self._structures if s.get('parent_index') == parent_index]
    
    # ------------------------------------------------------------------
    # Selection Methods
    # ------------------------------------------------------------------
    
    def get_pristine_structures(self) -> List[dict]:
        """Get all pristine structures."""
        return [s for s in self._structures if s['is_pristine']]
    
    def get_defect_structures(self) -> List[dict]:
        """Get all defect structures."""
        return [s for s in self._structures if not s['is_pristine']]
    
    def get_structure(self, index: int) -> dict:
        """Get structure by absolute index."""
        if 0 <= index < len(self._structures):
            return self._structures[index]
        raise IndexError(f"Index {index} out of range")
    
    def _find_pristine_index(self, structure_idx: int) -> int:
        """Find the pristine ancestor for a given structure."""
        if self._structures[structure_idx]['is_pristine']:
            return structure_idx
        pristine_idx = self._structures[structure_idx].get('pristine_structure_index', -1)
        if pristine_idx == -1:
            current = structure_idx
            visited = set()
            while current != -1 and current not in visited:
                visited.add(current)
                if self._structures[current]['is_pristine']:
                    return current
                current = self._structures[current].get('parent_index', -1)
        return pristine_idx
    
    def latest_pristine_index(self) -> int:
        """Absolute row index of the most recently added pristine structure."""
        pristine_rows = [i for i, s in enumerate(self._structures) if s['is_pristine']]
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
        defect_rows = [i for i, s in enumerate(self._structures) if not s['is_pristine']]
        
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


def get_vacancy_distances(container: StructureContainer, defect_index: int):
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
    >>> from structure_container_enhanced import StructureContainer, add_pristine, create_vacancy_from_ids
    >>> container = StructureContainer()
    >>> container = add_pristine(container, atoms)
    >>> container = create_vacancy_from_ids(container, atom_ids=[0, 5])
    >>> result = get_vacancy_distances(container, defect_index=1)
    >>> print(result['distances'])
    {'0-1': 4.05}
    """
    import numpy as np
    
    defect = container._structures[defect_index]
    events = defect['events']
    
    # Get all vacancy events
    vacancy_events = [e for e in events if e['type'] == 'vacancy']
    
    if len(vacancy_events) < 2:
        return {
            'vacancies': [],
            'distances': {},
            'distance_matrix': np.array([]),
            'message': f'Need at least 2 vacancies, found {len(vacancy_events)}'
        }
    
    # Extract vacancy info
    vacancies = []
    for ev in vacancy_events:
        vacancies.append({
            'uid': ev['site_uid'],
            'position': np.array(ev['site_pos0'])
        })
    
    # Get cell for periodic boundary calculations
    pristine = container._structures[defect['pristine_structure_index']]['structure']
    cell = pristine.get_cell()
    inv_cell = np.linalg.inv(cell)
    
    # Calculate pairwise distances with PBC
    n_vac = len(vacancies)
    distances = {}
    distance_matrix = np.zeros((n_vac, n_vac))
    
    for i in range(n_vac):
        for j in range(i+1, n_vac):
            # Calculate minimum image distance
            delta = vacancies[i]['position'] - vacancies[j]['position']
            # Apply minimum image convention
            delta -= np.round(delta @ inv_cell) @ cell
            dist = np.linalg.norm(delta)
            
            key = f"{i}-{j}"
            distances[key] = dist
            distance_matrix[i, j] = distance_matrix[j, i] = dist
    
    return {
        'vacancies': vacancies,
        'distances': distances,
        'distance_matrix': distance_matrix
    }


def get_substitution_distances(container: StructureContainer, defect_index: int):
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
    events = defect['events']

    sub_events = [e for e in events if e['type'] == 'substitution']

    if len(sub_events) < 2:
        return {
            'substitutions': [],
            'distances': {},
            'distance_matrix': np.array([]),
            'message': f'Need at least 2 substitutions, found {len(sub_events)}'
        }

    substitutions = [
        {
            'uid': ev['site_uid'],
            'from': ev['from'],
            'to': ev['to'],
            'position': np.array(ev['site_pos0']),
        }
        for ev in sub_events
    ]

    pristine = container._structures[defect['pristine_structure_index']]['structure']
    cell = pristine.get_cell()
    inv_cell = np.linalg.inv(cell)

    n_sub = len(substitutions)
    distances = {}
    distance_matrix = np.zeros((n_sub, n_sub))

    for i in range(n_sub):
        for j in range(i + 1, n_sub):
            delta = substitutions[i]['position'] - substitutions[j]['position']
            delta -= np.round(delta @ inv_cell) @ cell
            dist = np.linalg.norm(delta)
            distances[f'{i}-{j}'] = dist
            distance_matrix[i, j] = distance_matrix[j, i] = dist

    return {
        'substitutions': substitutions,
        'distances': distances,
        'distance_matrix': distance_matrix,
    }


def get_substitution_distances_relaxed(atoms, events: list):
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

    sub_events = [e for e in events if e['type'] == 'substitution']

    if len(sub_events) < 2:
        return {
            'substitutions': [],
            'distances': {},
            'distance_matrix': np.array([]),
            'message': f'Need at least 2 substitutions, found {len(sub_events)}'
        }

    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)

    # For each substitution, find the nearest atom in the relaxed structure
    # to the original site position. Atoms never move far enough during
    # relaxation to be closer to a different lattice site.
    relaxed_positions = []
    substitution_info = []
    for ev in sub_events:
        original_pos = np.array(ev['site_pos0'])
        diffs = atoms.positions - original_pos
        diffs -= np.round(diffs @ inv_cell) @ cell
        dists = np.linalg.norm(diffs, axis=1)
        closest_idx = np.argmin(dists)
        relaxed_positions.append(atoms.positions[closest_idx])
        substitution_info.append({
            'from': ev['from'],
            'to': ev['to'],
            'position_relaxed': atoms.positions[closest_idx],
        })

    n = len(relaxed_positions)
    distances = {}
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            delta = relaxed_positions[i] - relaxed_positions[j]
            delta -= np.round(delta @ inv_cell) @ cell
            dist = np.linalg.norm(delta)
            distances[f'{i}-{j}'] = dist
            distance_matrix[i, j] = distance_matrix[j, i] = dist

    return {
        'substitutions': substitution_info,
        'distances': distances,
        'distance_matrix': distance_matrix,
    }


def get_interstitial_distances(container: StructureContainer, defect_index: int):
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
    events = defect['events']
    int_events = [e for e in events if e['type'] == 'interstitial']

    if len(int_events) < 2:
        return {
            'interstitials': [],
            'distances': {},
            'distance_matrix': np.array([]),
            'message': f'Need at least 2 interstitials, found {len(int_events)}'
        }

    interstitials = [
        {'uid': ev['atom_uid'], 'element': ev['element'], 'position': np.array(ev['pos0'])}
        for ev in int_events
    ]

    pristine = container._structures[defect['pristine_structure_index']]['structure']
    cell = pristine.get_cell()
    inv_cell = np.linalg.inv(cell)

    n = len(interstitials)
    distances = {}
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            delta = interstitials[i]['position'] - interstitials[j]['position']
            delta -= np.round(delta @ inv_cell) @ cell
            dist = np.linalg.norm(delta)
            distances[f'{i}-{j}'] = dist
            distance_matrix[i, j] = distance_matrix[j, i] = dist

    return {'interstitials': interstitials, 'distances': distances, 'distance_matrix': distance_matrix}


def get_interstitial_distances_relaxed(atoms, events: list):
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

    int_events = [e for e in events if e['type'] == 'interstitial']

    if len(int_events) < 2:
        return {
            'interstitials': [],
            'distances': {},
            'distance_matrix': np.array([]),
            'message': f'Need at least 2 interstitials, found {len(int_events)}'
        }

    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)

    relaxed_positions = []
    interstitial_info = []
    for ev in int_events:
        uid = ev.get('atom_uid')
        idx = uid_to_index(atoms, uid) if uid is not None else None
        if idx is not None:
            pos = atoms.positions[idx]
            print(f"  interstitial {ev['element']} (atom_uid={uid}): located via uid")
        else:
            # fallback: nearest-neighbour from insertion position
            original_pos = np.array(ev['pos0'])
            diffs = atoms.positions - original_pos
            diffs -= np.round(diffs @ inv_cell) @ cell
            idx = np.argmin(np.linalg.norm(diffs, axis=1))
            pos = atoms.positions[idx]
            reason = 'no atom_uid in event' if uid is None else 'uid not found in atoms.arrays'
            print(f"  interstitial {ev['element']}: located via nearest-neighbour ({reason})")
        relaxed_positions.append(pos)
        interstitial_info.append({'element': ev['element'], 'position_relaxed': pos})

    n = len(relaxed_positions)
    distances = {}
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            delta = relaxed_positions[i] - relaxed_positions[j]
            delta -= np.round(delta @ inv_cell) @ cell
            dist = np.linalg.norm(delta)
            distances[f'{i}-{j}'] = dist
            distance_matrix[i, j] = distance_matrix[j, i] = dist

    return {'interstitials': interstitial_info, 'distances': distances, 'distance_matrix': distance_matrix}


def make_operations_short(events: List[dict]) -> str:
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
        return 'no_operations'
    
    short_ops = []
    for ev in events:
        t = ev.get('type')
        if t == 'vacancy':
            uid = ev.get('site_uid', '?')
            short_ops.append(f"vacancy[{uid}]")
        elif t == 'substitution':
            from_el = ev.get('from', '?')
            to_el = ev.get('to', '?')
            uid = ev.get('site_uid', '?')
            short_ops.append(f"substitution[{from_el}->{to_el}]")
        elif t == 'interstitial':
            el = ev.get('element', '?')
            uid = ev.get('atom_uid', '?')
            short_ops.append(f"interstitial[{el}]")
    
    return '|'.join(short_ops) if short_ops else 'no_operations'


# ============================================================================
# Standalone Wrapper Functions (GUI-Friendly)
# ============================================================================

# ----------------------------------------------------------------------
# Table/Extraction Functions
# ----------------------------------------------------------------------

def get_structure_table(structure_container: StructureContainer) -> "pd.DataFrame":
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
    >>> from structure_container_enhanced import StructureContainer, add_pristine, create_vacancy_from_ids
    >>> container = StructureContainer()
    >>> container = add_pristine(container, atoms)
    >>> df = get_structure_table(container)
    """
    return structure_container.get_structure_table()


def get_defect_table(structure_container: StructureContainer) -> "pd.DataFrame":
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
    >>> df = get_defect_table(container)
    """
    return structure_container.get_defect_table()


def get_pristine_table(structure_container: StructureContainer) -> "pd.DataFrame":
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
    >>> df = get_pristine_table(container)
    """
    return structure_container.get_pristine_table()


# ----------------------------------------------------------------------
# Add Structures
# ----------------------------------------------------------------------

def add_pristine(
    structure_container: Optional[StructureContainer] = None,
    atoms: Atoms = None,
    unique_id: Optional[str] = None,
    metadata: Optional[dict] = None,
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
    >>> from structure_container_enhanced import add_pristine
    >>> 
    >>> atoms = bulk('Al', cubic=True)
    >>> container = add_pristine(atoms=atoms, unique_id="Al_fcc")
    """
    if structure_container is None:
        structure_container = StructureContainer()
    structure_container.add_pristine(atoms, unique_id, metadata, check_duplicates, tolerance)
    return structure_container


# ----------------------------------------------------------------------
# Filter Functions
# ----------------------------------------------------------------------

def filter_by_indices(structure_container: StructureContainer, indices: List[int]) -> List[dict]:
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


def filter_by_generation(structure_container: StructureContainer, generation: int) -> List[dict]:
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
    >>> gen1 = filter_by_generation(container, generation=1)
    """
    return structure_container.filter_by_generation(generation)


def filter_by_max_generation(structure_container: StructureContainer, max_generation: int) -> List[dict]:
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
    >>> gen2_and_below = filter_by_max_generation(container, max_generation=2)
    """
    return structure_container.filter_by_max_generation(max_generation)


def filter_by_operations_short(structure_container: StructureContainer, pattern: str) -> List[dict]:
    """
    Filter by operations_short field (supports wildcards).
    
    Parameters
    ----------
    structure_container : StructureContainer
        The container to filter
    pattern : str
        Pattern to match (supports wildcards *, ?, [])
    
    Returns
    -------
    List[dict]
        Matching structures
    
    Examples
    --------
    >>> exact = filter_by_operations_short(container, "vacancy[5]")
    >>> contains = filter_by_operations_short(container, "*vacancy[5]*")
    """
    return structure_container.filter_by_operations_short(pattern)


def filter_by_operations_contains(structure_container: StructureContainer, operation_type: str) -> List[dict]:
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
    >>> all_vacancies = filter_by_operations_contains(container, "vacancy")
    """
    return structure_container.filter_by_operations_contains(operation_type)


def filter_by_condition(structure_container: StructureContainer, condition: Callable[[dict], bool]) -> List[dict]:
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


def filter_by_unique_id(structure_container: StructureContainer, unique_id: str) -> Optional[dict]:
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


def filter_by_number_of_atoms(structure_container: StructureContainer, n_atoms: int) -> List[dict]:
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
    >>> structures_108 = filter_by_number_of_atoms(container, 108)
    """
    return structure_container.filter_by_number_of_atoms(n_atoms)


def filter_by_element_count(
    structure_container: StructureContainer,
    element: str,
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
    exact_count: Optional[int] = None
) -> List[dict]:
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
    >>> single_mg = filter_by_element_count(container, 'Mg', exact_count=1)
    >>> few_mg = filter_by_element_count(container, 'Mg', min_count=0, max_count=5)
    """
    return structure_container.filter_by_element_count(element, min_count, max_count, exact_count)


def filter_by_stoichiometry(structure_container: StructureContainer, formula_pattern: Optional[str] = None) -> List[dict]:
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
    >>> exact = filter_by_stoichiometry(container, 'Al107Mg1')
    >>> single_mg = filter_by_stoichiometry(container, '*Mg1*')
    """
    return structure_container.filter_by_stoichiometry(formula_pattern)


def filter_by_parent(structure_container: StructureContainer, parent_index: int) -> List[dict]:
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
    >>> children_0 = filter_by_parent(container, 0)
    """
    return structure_container.filter_by_parent(parent_index)


# ----------------------------------------------------------------------
# Selection Functions
# ----------------------------------------------------------------------

def get_structure(structure_container: StructureContainer, index: int) -> dict:
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


def get_pristine_structures(structure_container: StructureContainer) -> List[dict]:
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


def get_defect_structures(structure_container: StructureContainer) -> List[dict]:
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


def latest_pristine_index(structure_container: StructureContainer) -> int:
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

def resolve_defect_row(structure_container: StructureContainer, relative_index: int) -> int:
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
    >>> first defect
    >>> first = resolve_defect_row(container, 0)
    >>> most recent defect
    >>> latest = resolve_defect_row(container, -1)
    """
    return structure_container.resolve_defect_row(relative_index)


def resolve_any_row(structure_container: StructureContainer, relative_index: int) -> int:
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
    >>> latest = resolve_any_row(container, -1)
    >>> first = resolve_any_row(container, 0)
    """
    return structure_container.resolve_any_row(relative_index)


# ============================================================================
# Helper Functions for StructureContainer
# ============================================================================

def _resolve_parent(
    container: StructureContainer,
    parent_defect_index: Optional[int] = None,
    input_structure: Optional[Atoms] = None,
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
            stacklevel=3
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
        if container._structures[parent_defect_index]['is_pristine']:
            raise ValueError(
                f"Index {parent_defect_index} is a pristine structure. "
                f"Use input_structure parameter instead if you want to work with a pristine structure."
            )
        return parent_defect_index
    
    # Case 2: Input structure provided
    if input_structure is not None:
        input_structure = ensure_uids(input_structure)
        for idx, s in enumerate(container._structures):
            s_structure = ensure_uids(s['structure'])
            # Guard: skip if atom counts differ (avoids broadcast error)
            if len(s_structure) != len(input_structure):
                continue
            if (np.allclose(s_structure.positions, input_structure.positions) and
                np.allclose(s_structure.cell, input_structure.cell) and
                s_structure.get_chemical_symbols() == input_structure.get_chemical_symbols()):
                return idx
        return container.add_pristine(input_structure)
    
    # Case 3: Default to latest pristine
    pristine_indices = [i for i, s in enumerate(container._structures) if s['is_pristine']]
    if not pristine_indices:
        raise ValueError(
            "No pristine structure found in the container. "
            "Add one first using container.add_pristine(atoms)."
        )
    return pristine_indices[-1]


# ============================================================================
# Vacancy Creation Functions
# ============================================================================

def create_vacancy_from_ids(
    structure_container: StructureContainer,
    atom_ids: List[int],
    parent_defect_index: Optional[int] = None,
    input_structure: Optional[Atoms] = None,
    forbid_atom_ids: Optional[List[int]] = None,
    protect_history: bool = False,
) -> StructureContainer:
    """
    Create vacancies at specific atom indices.
    
    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify
    atom_ids : List[int]
        Specific atom indices to remove (can determine element type from structure)
    parent_defect_index : int or None
        Index of defect structure to build on (None = use default parent)
    input_structure : Atoms or None
        Structure to use as parent (None = use container default)
    forbid_atom_ids : list of int or None
        Atom IDs to exclude from removal
    protect_history : bool
        Protect atoms from previous defects
    
    Returns
    -------
    StructureContainer with new vacancies added
    """
    import numpy as np

    container = structure_container
    
    # Resolve parent structure
    parent_idx = _resolve_parent(container, parent_defect_index, input_structure)
    parent = container.get_structure(parent_idx)
    
    atoms = ensure_uids(parent['structure']).copy()
    existing_events = parent['events'].copy()
    
    # Find pristine ancestor
    pristine_idx = container._find_pristine_index(parent_idx)
    
    # Validate indices
    atoms_for_validation = ensure_uids(atoms).copy()
    validate_atoms_arrays(atoms_for_validation)
    uids = atoms_for_validation.arrays[UID_KEY].astype(int)
    syms = np.array(atoms_for_validation.get_chemical_symbols(), dtype=object)
    
    # Check all indices are valid
    for idx in atom_ids:
        if not (0 <= idx < len(atoms_for_validation)):
            raise IndexError(f"Vacancy index {idx} is out of range for a structure with {len(atoms_for_validation)} atoms.")
    
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
        new_events.append({
            "type": "vacancy",
            "removed_element": str(syms[i]),
            "site_uid": site_uid,
            "site_pos0": atoms_for_validation.positions[i].tolist(),
            "pos_at_removal": atoms_for_validation.positions[i].tolist(),
        })
    
    # Delete atoms (in reverse order to maintain indices)
    for i in sorted(map(int, valid_indices), reverse=True):
        del atoms_for_validation[i]
    
    validate_atoms_arrays(atoms_for_validation)
    all_new_events = existing_events + new_events
    
    # Build operation string
    if len(valid_indices) > 1:
        operation_str = f"vacancy[{len(valid_indices)}]"
    else:
        operation_str = f"vacancy[{new_events[0]['site_uid']}]"
    
    # Store in container
    container.add_defect(
        atoms=atoms_for_validation,
        operation=operation_str,
        pristine_index=pristine_idx,
        parent_index=parent_idx,
        events=all_new_events,
        metadata={'parent_index': parent_idx, 'pristine_index': pristine_idx},
    )
    
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


def get_voronoi_interstitial_sites_pymatgen(
    atoms: Atoms,
    primitive_atoms: Optional[Atoms] = None,
    repeat: Optional[tuple] = None,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
    dedup_tol: float = 1e-5,
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Find Voronoi interstitial sites via pymatgen (symmetry-aware, slow).

    Uses pymatgen's ``VoronoiInterstitialGenerator`` + ``SpacegroupAnalyzer``
    to return symmetry-unique void centers expanded to all equivalent positions.
    Prefer :func:`get_voronoi_interstitial_sites` (scipy-based, no pymatgen)
    for speed; use this only when you specifically need spacegroup-reduced sites.

    The positions returned depend solely on the host geometry — not on which
    element will be inserted.  This function is the slow, SLURM-submittable
    step of the two-step workflow::

        # Step 1 — slow, submit to SLURM (memory-efficient form)
        unique_sites, all_sites = get_voronoi_interstitial_sites_pymatgen(
            atoms, primitive_atoms=prim, repeat=(3, 3, 3))

        # Step 2 — fast, element-specific
        container = create_interstitial_from_ids(
            container, positions=unique_sites, element='Mg')
        container = create_interstitial_from_ids(
            container, positions=unique_sites, element='Al')

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
        If pymatgen or structuretoolkit is not available.
    ValueError
        If only one of ``primitive_atoms`` / ``repeat`` is provided.

    See Also
    --------
    create_interstitial_from_ids : Insert an atom at a specific position.
    create_interstitial_from_seed : Randomly sample from candidate positions.
    """
    import numpy as np

    if not VORONOI_AVAILABLE:
        raise ImportError(
            "Voronoi interstitial site finding requires pymatgen and structuretoolkit. "
            "Install them with: pip install pymatgen structuretoolkit"
        )

    if (primitive_atoms is None) != (repeat is None):
        raise ValueError(
            "primitive_atoms and repeat must be provided together or not at all."
        )

    use_primitive = primitive_atoms is not None

    # Dynamically import VoronoiInterstitialGenerator to avoid hard dependency
    try:
        from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError as e:
        raise ImportError(
            f"Could not import pymatgen defect modules: {e}. "
            "Install pymatgen with: pip install pymatgen"
        )

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
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)

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
                            cart_base + np.array([i, j, k], dtype=float) @ prim_lattice
                        )
        all_sites = np.array(all_sites_list, dtype=float)
    else:
        all_sites = frac_all_prim @ prim_lattice

    return equivalent_sites, all_sites


def get_voronoi_interstitial_sites(
    atoms: Atoms,
    primitive_atoms: Optional[Atoms] = None,
    repeat: Optional[tuple] = None,
    r_min: float = 0.8,
    cluster_tol: float = 0.5,
    n_images: int = 1,
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Find Voronoi interstitial sites via scipy — fast, no pymatgen dependency.

    Voronoi vertices of the atom positions (plus periodic image shells) are
    exactly the void centers of the tessellation.  No symmetry analysis is
    performed; all geometric void centers are returned directly.  The interface
    mirrors :func:`get_delaunay_interstitial_sites` and both are drop-in
    replaceable with each other::

        unique_sites, all_sites = get_voronoi_interstitial_sites(
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
        :func:`get_voronoi_interstitial_sites_pymatgen`.  Clusters are
        geometric, not symmetry-derived.
    all_sites : (M, 3) ndarray
        All deduplicated void centers in Cartesian coordinates (Å),
        replicated across the supercell when ``repeat`` is given.

    See Also
    --------
    get_voronoi_interstitial_sites_pymatgen : Symmetry-aware alternative (slower, requires pymatgen).
    get_delaunay_interstitial_sites : Equivalent via Delaunay circumcenters.
    """
    import numpy as np
    from scipy.spatial import Voronoi

    if (primitive_atoms is None) != (repeat is None):
        raise ValueError(
            "primitive_atoms and repeat must be provided together or not at all."
        )

    use_primitive = primitive_atoms is not None
    work_atoms = primitive_atoms if use_primitive else atoms

    cell = work_atoms.get_cell().array        # (3, 3)
    cell_inv = np.linalg.inv(cell)
    pos = work_atoms.get_positions()          # (N, 3)

    # --- build extended point set with periodic images ---
    offsets = np.array(
        [[i, j, k]
         for i in range(-n_images, n_images + 1)
         for j in range(-n_images, n_images + 1)
         for k in range(-n_images, n_images + 1)],
        dtype=float,
    )
    pos_ext = np.vstack([pos + off @ cell for off in offsets])  # (N*(2n+1)^3, 3)

    # --- Voronoi tessellation: vertices are the void centers ---
    vor = Voronoi(pos_ext)
    candidates_all = vor.vertices           # (V, 3)

    # --- keep only candidates inside the primitive unit cell ---
    frac = candidates_all @ cell_inv
    inside = np.all((frac >= -1e-8) & (frac < 1 - 1e-8), axis=1)
    candidates = candidates_all[inside]

    if len(candidates) == 0:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty

    # --- filter: too close to any host atom (minimum image convention) ---
    diff = candidates[:, None, :] - pos[None, :, :]    # (C, N, 3)
    diff_frac = diff @ cell_inv
    diff_frac -= np.round(diff_frac)
    diff_cart = diff_frac @ cell
    min_dist = np.linalg.norm(diff_cart, axis=-1).min(axis=1)  # (C,)
    candidates = candidates[min_dist >= r_min]

    if len(candidates) == 0:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty

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
    unique_sites = all_sites_prim.copy()               # one per cluster

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


def get_delaunay_interstitial_sites(
    atoms: Atoms,
    primitive_atoms: Optional[Atoms] = None,
    repeat: Optional[tuple] = None,
    r_min: float = 0.8,
    cluster_tol: float = 0.5,
    n_images: int = 1,
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Find interstitial sites via Delaunay circumcenters — much faster than
    Voronoi tessellation for large or complex primitive cells.

    Circumcenters of Delaunay tetrahedra are natural void centers.  No
    pymatgen dependency; uses only scipy and numpy.  The interface mirrors
    :func:`get_voronoi_interstitial_sites` (and :func:`get_voronoi_interstitial_sites_pymatgen`)
    so all three are drop-in replaceable::

        unique_sites, all_sites = get_delaunay_interstitial_sites(
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
        :func:`get_voronoi_interstitial_sites_pymatgen`.  Note: clusters are
        geometric, not symmetry-derived; for symmetry-unique sites pass
        these through spglib separately.
    all_sites : (M, 3) ndarray
        All deduplicated void centers in Cartesian coordinates (Å),
        replicated across the supercell when ``repeat`` is given.

    See Also
    --------
    get_voronoi_interstitial_sites : Scipy-based Voronoi alternative (same speed, no pymatgen).
    """
    import numpy as np
    from scipy.spatial import Delaunay

    if (primitive_atoms is None) != (repeat is None):
        raise ValueError(
            "primitive_atoms and repeat must be provided together or not at all."
        )

    use_primitive = primitive_atoms is not None
    work_atoms = primitive_atoms if use_primitive else atoms

    cell = work_atoms.get_cell().array        # (3, 3)
    cell_inv = np.linalg.inv(cell)
    pos = work_atoms.get_positions()          # (N, 3)

    # --- build extended point set with periodic images ---
    offsets = np.array(
        [[i, j, k]
         for i in range(-n_images, n_images + 1)
         for j in range(-n_images, n_images + 1)
         for k in range(-n_images, n_images + 1)],
        dtype=float,
    )
    pos_ext = np.vstack([pos + off @ cell for off in offsets])  # (N*(2n+1)^3, 3)

    # --- Delaunay tessellation ---
    tri = Delaunay(pos_ext)

    # --- circumcenter of each tetrahedron ---
    verts = pos_ext[tri.simplices]            # (T, 4, 3)
    A = verts[:, 1:] - verts[:, :1]          # (T, 3, 3)  edge vectors from v0
    b = 0.5 * (A ** 2).sum(axis=-1)          # (T, 3)

    # Drop degenerate tetrahedra (coplanar vertices → singular A)
    # before the batched solve; det ≈ 0 identifies them cheaply.
    At = A.swapaxes(1, 2)                     # (T, 3, 3)
    nondegenerate = np.abs(np.linalg.det(At)) > 1e-10
    At_nd = At[nondegenerate]
    b_nd  = b[nondegenerate]
    v0_nd = verts[nondegenerate, 0]

    # b must be (..., m, 1) for numpy's batched solver
    x = np.linalg.solve(At_nd, b_nd[..., None])[..., 0]   # (T', 3)
    centers = x + v0_nd                       # (T', 3)

    # --- keep only candidates inside the primitive unit cell ---
    frac = centers @ cell_inv
    inside = np.all((frac >= -1e-8) & (frac < 1 - 1e-8), axis=1)
    candidates = centers[inside]

    if len(candidates) == 0:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty

    # --- filter: too close to any host atom (minimum image convention) ---
    diff = candidates[:, None, :] - pos[None, :, :]    # (C, N, 3)
    diff_frac = diff @ cell_inv
    diff_frac -= np.round(diff_frac)
    diff_cart = diff_frac @ cell
    min_dist = np.linalg.norm(diff_cart, axis=-1).min(axis=1)  # (C,)
    candidates = candidates[min_dist >= r_min]

    if len(candidates) == 0:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty

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
    unique_sites = all_sites_prim.copy()              # one per cluster

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


def create_vacancy_from_seed(
    structure_container: StructureContainer,
    n: int = 1,
    seed: Optional[int] = None,
    vacancy_element: Optional[Union[str, List[str]]] = None,
    parent_defect_index: Optional[int] = None,
    input_structure: Optional[Atoms] = None,
    forbid_uids: Optional[List[int]] = None,
    protect_history: bool = False,
) -> StructureContainer:
    """
    Create random vacancies with reproducible seed.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify
    n : int
        Number of vacancies to create
    seed : int or None
        Random seed for reproducibility
    vacancy_element : str, list of str, or None
        Element to remove (e.g., 'Al'). None = any element.
        Pass a list to specify the exact per-vacancy elements
        (e.g., ['Al', 'Mg'] removes one Al and one Mg).
        When a list is given, n must equal len(vacancy_element).
    parent_defect_index : int or None
        Index of defect structure to build on (None = use default parent)
    input_structure : Atoms or None
        Structure to use as parent (None = use container default)
    forbid_uids : list of int or None
        UIDs to exclude from removal
    protect_history : bool
        Protect atoms from previous defects
    
    Returns
    -------
    StructureContainer with new vacancy added
    """

    import numpy as np

    container = structure_container
    
    # Resolve parent structure
    parent_idx = _resolve_parent(container, parent_defect_index, input_structure)
    parent = container.get_structure(parent_idx)
    
    atoms = ensure_uids(parent['structure']).copy()
    existing_events = parent['events'].copy()
    
    # Find pristine ancestor
    pristine_idx = container._find_pristine_index(parent_idx)
    
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
                i for i in range(len(atoms_copy))
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
        cand = [i for i in range(len(atoms_copy)) if syms[i] == vacancy_element and int(uids[i]) not in forbid]
        if len(cand) < n:
            raise ValueError(f"Not enough candidates: need {n}, have {len(cand)}")
        pick_idx = rng.choice(cand, size=int(n), replace=False).tolist()
    
    # Record all events
    new_events = []
    for i in pick_idx:
        site_uid = int(uids[i])
        new_events.append({
            "type": "vacancy",
            "removed_element": str(syms[i]),
            "site_uid": site_uid,
            "site_pos0": atoms_copy.positions[i].tolist(),
            "pos_at_removal": atoms_copy.positions[i].tolist(),
        })
    
    # Delete atoms
    for i in sorted(map(int, pick_idx), reverse=True):
        del atoms_copy[i]
    
    validate_atoms_arrays(atoms_copy)
    all_new_events = existing_events + new_events
    
    # Build operation string
    if n > 1:
        operation_str = f"vacancy[{n}]"
    else:
        operation_str = f"vacancy[{new_events[0]['site_uid']}]"
    
    # Store in container
    container.add_defect(
        atoms=atoms_copy,
        operation=operation_str,
        pristine_index=pristine_idx,
        parent_index=parent_idx,
        events=all_new_events,
        metadata={'parent_index': parent_idx, 'pristine_index': pristine_idx, 'seed': seed},
    )
    
    out_container = container
    return out_container


def create_vacancy_batch_from_ids(
    structure_container: StructureContainer,
    target_indices: List[int],
    atom_ids: Optional[List[int]] = None,
    separate_structures: bool = True,
    forbid_atom_ids: Optional[List[int]] = None,
    protect_history: bool = False,
) -> StructureContainer:
    """
    Apply specific vacancy indices to multiple structures.
    
    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify
    target_indices : List[int]
        Absolute indices of structures to modify
    atom_ids : List[int]
        Specific atom indices to remove for each target structure
    separate_structures : bool (default=True)
        True: Create separate structures for each atom in atom_ids
        False: Create one structure with all vacancies per target
    forbid_atom_ids : list of int or None
        Atom IDs to exclude from removal
    protect_history : bool
        Protect atoms from previous defects
    
    Returns
    -------
    StructureContainer with all new structures added
    
    Notes
    -----
    Use container methods like filter_by_generation(), filter_by_pristine_structures(),
    etc. to get the target_indices before calling this function.
    """
    container = structure_container
    rows_to_modify = target_indices
    
    if separate_structures:
        # Create separate structures for each atom_id
        for parent_idx in rows_to_modify:
            for atom_id in atom_ids:
                container = create_vacancy_from_ids(
                    structure_container=container,
                    atom_ids=[atom_id],
                    parent_defect_index=parent_idx if not container._structures[parent_idx]['is_pristine'] else None,
                    input_structure=container._structures[parent_idx]['structure'] if container._structures[parent_idx]['is_pristine'] else None,
                    forbid_atom_ids=forbid_atom_ids,
                    protect_history=protect_history,
                )
    else:
        # Apply all atom_ids to each target structure
        for parent_idx in rows_to_modify:
            container = create_vacancy_from_ids(
                structure_container=container,
                atom_ids=atom_ids,
                parent_defect_index=parent_idx if not container._structures[parent_idx]['is_pristine'] else None,
                input_structure=container._structures[parent_idx]['structure'] if container._structures[parent_idx]['is_pristine'] else None,
                forbid_atom_ids=forbid_atom_ids,
                protect_history=protect_history,
            )
    
    out_container = container
    return out_container


def create_vacancy_batch_from_seed(
    structure_container: StructureContainer,
    target_indices: List[int],
    n: int = 1,
    seed: Optional[int] = None,
    vacancy_element: Optional[Union[str, List[str]]] = None,
    forbid_uids: Optional[List[int]] = None,
    protect_history: bool = False,
    n_structures: int = 1,
) -> StructureContainer:
    """
    Apply random vacancies to multiple structures with reproducible seeds.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify
    target_indices : List[int]
        Absolute indices of structures to modify
    n : int
        Number of vacancies to create per structure.
        When vacancy_element is a list, n must equal len(vacancy_element).
    seed : int or None
        Base random seed (each structure uses different seeds)
    vacancy_element : str, list of str, or None
        Element to remove (e.g., 'Al'). None = any element.
        Pass a list to fix per-vacancy elements (e.g., ['Al', 'Mg']).
    forbid_uids : list of int or None
        UIDs to exclude from removal
    protect_history : bool
        Protect atoms from previous defects
    n_structures : int
        Number of structures to create from each parent. Default=1 (backward compatible).
        If n_structures > 1, creates n_structures copies from each parent index
        with incrementing seeds: seed, seed+1, seed+2, etc.
    
    Returns
    -------
    StructureContainer with all new structures added
    
    Notes
    -----
    Use container methods like filter_by_generation(), filter_by_pristine_structures(),
    etc. to get the target_indices before calling this function.
    
    Examples
    --------
    >>> # Create 100 structures from pristine, each with 2 random vacancies
    >>> container = create_vacancy_batch_from_seed(
    ...     structure_container=container,
    ...     target_indices=[0],  # pristine
    ...     n=2,                 # 2 vacancies per structure
    ...     n_structures=100,    # Create 100 separate structures
    ...     seed=0
    ... )
    
    >>> # Create 50 structures from two different parents
    >>> container = create_vacancy_batch_from_seed(
    ...     structure_container=container,
    ...     target_indices=[0, 5],  # Two different structures
    ...     n=2,
    ...     n_structures=50,         # 50 from each = 100 total
    ...     seed=0
    ... )
    """
    container = structure_container
    rows_to_modify = target_indices
    
    # Apply vacancy to each selected structure
    # For each parent, create n_structures copies with incrementing seeds
    structure_counter = 0
    for parent_idx in rows_to_modify:
        for copy_idx in range(n_structures):
            structure_seed = seed + structure_counter if seed is not None else None
            container = create_vacancy_from_seed(
                structure_container=container,
                n=n,
                seed=structure_seed,
                vacancy_element=vacancy_element,
                parent_defect_index=parent_idx if not container._structures[parent_idx]['is_pristine'] else None,
                input_structure=container._structures[parent_idx]['structure'] if container._structures[parent_idx]['is_pristine'] else None,
                forbid_uids=forbid_uids,
                protect_history=protect_history,
            )
            structure_counter += 1
    
    out_container = container
    return out_container


# ============================================================================
# Substitution Creation Functions
# ============================================================================

def create_substitution_from_ids(
    structure_container: StructureContainer,
    atom_ids: List[int],
    to_element: str,
    parent_defect_index: Optional[int] = None,
    input_structure: Optional[Atoms] = None,
    forbid_atom_ids: Optional[List[int]] = None,
    protect_history: bool = False,
) -> StructureContainer:
    """
    Create substitutions at specific atom indices.
    
    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify
    atom_ids : List[int]
        Specific atom indices to substitute (can determine from_element from structure)
    to_element : str
        Element to substitute with
    parent_defect_index : int or None
        Index of defect structure to build on (None = use default parent)
    input_structure : Atoms or None
        Structure to use as parent (None = use container default)
    forbid_atom_ids : list of int or None
        Atom IDs to exclude from substitution
    protect_history : bool
        Protect atoms from previous defects
    
    Returns
    -------
    StructureContainer with new substitution added
    """
    import numpy as np

    container = structure_container
    
    # Resolve parent structure
    parent_idx = _resolve_parent(container, parent_defect_index, input_structure)
    parent = container.get_structure(parent_idx)
    
    atoms = ensure_uids(parent['structure']).copy()
    existing_events = parent['events'].copy()
    
    # Find pristine ancestor
    pristine_idx = container._find_pristine_index(parent_idx)
    
    # Validate indices
    uids = atoms.arrays[UID_KEY].astype(int)
    syms = np.array(atoms.get_chemical_symbols(), dtype=object)
    
    # Check all indices are valid
    for idx in atom_ids:
        if not (0 <= idx < len(atoms)):
            raise IndexError(f"Substitution index {idx} is out of range for a structure with {len(atoms)} atoms.")
    
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
        
        new_events.append({
            "type": "substitution",
            "from": from_element,
            "to": str(to_element),
            "atom_uid": atom_uid,
            "site_uid": site_uid,
            "site_pos0": site_pos0,
            "pos_at_creation": atoms.positions[i].tolist(),
        })
        
        atoms[i].symbol = to_element
    
    all_new_events = existing_events + new_events
    
    # Build operation string
    if len(valid_indices) > 1:
        # Use first substitution as reference
        operation_str = f"substitution[{len(valid_indices)}:{new_events[0]['from']}->{to_element}]"
    else:
        operation_str = f"substitution[{new_events[0]['from']}->{to_element}]"
    
    # Store in container
    container.add_defect(
        atoms=atoms,
        operation=operation_str,
        pristine_index=pristine_idx,
        parent_index=parent_idx,
        events=all_new_events,
        metadata={'parent_index': parent_idx, 'pristine_index': pristine_idx},
    )
    
    out_container = container
    return out_container


def create_substitution_from_seed(
    structure_container: StructureContainer,
    n: int = 1,
    seed: Optional[int] = None,
    from_element: str = "Al",
    to_element: str = "Mg",
    parent_defect_index: Optional[int] = None,
    input_structure: Optional[Atoms] = None,
    forbid_uids: Optional[List[int]] = None,
    protect_history: bool = False,
) -> StructureContainer:
    """
    Create random substitutions with reproducible seed.
    
    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify
    n : int
        Number of substitutions to create
    seed : int or None
        Random seed for reproducibility
    from_element : str
        Element to replace
    to_element : str
        Element to substitute with
    parent_defect_index : int or None
        Index of defect structure to build on (None = use default parent)
    input_structure : Atoms or None
        Structure to use as parent (None = use container default)
    forbid_uids : list of int or None
        UIDs to exclude from substitution
    protect_history : bool
        Protect atoms from previous defects
    
    Returns
    -------
    StructureContainer with new substitution added
    """
    import numpy as np

    container = structure_container
    
    # Resolve parent structure
    parent_idx = _resolve_parent(container, parent_defect_index, input_structure)
    parent = container.get_structure(parent_idx)
    
    atoms = ensure_uids(parent['structure']).copy()
    existing_events = parent['events'].copy()
    
    # Find pristine ancestor
    pristine_idx = container._find_pristine_index(parent_idx)
    
    # Set up random selection
    rng = np.random.default_rng(seed if seed is not None else 0)
    uids = atoms.arrays[UID_KEY].astype(int)
    syms = np.array(atoms.get_chemical_symbols(), dtype=object)
    
    # Build forbid set
    forbid = set() if forbid_uids is None else set(map(int, forbid_uids))
    if protect_history:
        forbid |= _protected_uids_from_events(existing_events)
    
    # Find candidates
    cand = [i for i in range(len(atoms)) if syms[i] == from_element and int(uids[i]) not in forbid]
    
    if n > len(cand):
        raise ValueError(f"Not enough candidates to substitute {from_element}->{to_element}: need {n}, have {len(cand)}")
    
    pick_idx = rng.choice(cand, size=int(n), replace=False)
    
    # Record events and apply substitutions
    new_events = []
    for i in pick_idx:
        atom_uid = int(uids[i])
        site_uid = atom_uid
        site_pos0 = atoms.positions[i].tolist()
        
        new_events.append({
            "type": "substitution",
            "from": str(from_element),
            "to": str(to_element),
            "atom_uid": atom_uid,
            "site_uid": site_uid,
            "site_pos0": site_pos0,
            "pos_at_creation": atoms.positions[i].tolist(),
        })
    
    syms[pick_idx] = to_element
    atoms.set_chemical_symbols(syms.tolist())
    
    all_new_events = existing_events + new_events
    
    # Build operation string
    if n > 1:
        operation_str = f"substitution[{n}:{from_element}->{to_element}]"
    else:
        operation_str = f"substitution[{from_element}->{to_element}]"
    
    # Store in container
    container.add_defect(
        atoms=atoms,
        operation=operation_str,
        pristine_index=pristine_idx,
        parent_index=parent_idx,
        events=all_new_events,
        metadata={'parent_index': parent_idx, 'pristine_index': pristine_idx, 'seed': seed},
    )
    
    out_container = container
    return out_container


def create_substitution_batch_from_ids(
    structure_container: StructureContainer,
    target_indices: List[int],
    atom_ids: Optional[List[int]] = None,
    to_element: str = "Mg",
    separate_structures: bool = True,
    forbid_atom_ids: Optional[List[int]] = None,
    protect_history: bool = False,
) -> StructureContainer:
    """
    Apply specific substitution indices to multiple structures.
    
    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify
    target_indices : List[int]
        Absolute indices of structures to modify
    atom_ids : List[int]
        Specific atom indices to substitute for each target structure
    to_element : str
        Element to substitute with
    separate_structures : bool (default=True)
        True: Create separate structures for each atom in atom_ids
        False: Create one structure with all substitutions per target
    forbid_atom_ids : list of int or None
        Atom IDs to exclude from substitution
    protect_history : bool
        Protect atoms from previous defects
    
    Returns
    -------
    StructureContainer with all new structures added
    
    Notes
    -----
    Use container methods like filter_by_generation(), filter_by_pristine_structures(),
    etc. to get the target_indices before calling this function.
    """
    container = structure_container
    rows_to_modify = target_indices
    
    if separate_structures:
        # Create separate structures for each atom_id
        for parent_idx in rows_to_modify:
            for atom_id in atom_ids:
                container = create_substitution_from_ids(
                    structure_container=container,
                    atom_ids=[atom_id],
                    to_element=to_element,
                    parent_defect_index=parent_idx if not container._structures[parent_idx]['is_pristine'] else None,
                    input_structure=container._structures[parent_idx]['structure'] if container._structures[parent_idx]['is_pristine'] else None,
                    forbid_atom_ids=forbid_atom_ids,
                    protect_history=protect_history,
                )
    else:
        # Apply all atom_ids to each target structure
        for parent_idx in rows_to_modify:
            container = create_substitution_from_ids(
                structure_container=container,
                atom_ids=atom_ids,
                to_element=to_element,
                parent_defect_index=parent_idx if not container._structures[parent_idx]['is_pristine'] else None,
                input_structure=container._structures[parent_idx]['structure'] if container._structures[parent_idx]['is_pristine'] else None,
                forbid_atom_ids=forbid_atom_ids,
                protect_history=protect_history,
            )
    
    out_container = container
    return out_container


def create_substitution_batch_from_seed(
    structure_container: StructureContainer,
    target_indices: List[int],
    n: int = 1,
    seed: Optional[int] = None,
    from_element: str = "Al",
    to_element: str = "Mg",
    forbid_uids: Optional[List[int]] = None,
    protect_history: bool = False,
    n_structures: int = 1,
) -> StructureContainer:
    """
    Apply random substitutions to multiple structures with reproducible seeds.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify
    target_indices : List[int]
        Absolute indices of structures to modify
    n : int
        Number of substitutions to create per structure
    seed : int or None
        Base random seed (each structure uses different seeds)
    from_element : str
        Element to replace
    to_element : str
        Element to substitute with
    forbid_uids : list of int or None
        UIDs to exclude from substitution
    protect_history : bool
        Protect atoms from previous defects
    n_structures : int
        Number of structures to create from each parent. Default=1 (backward compatible).
        If n_structures > 1, creates n_structures copies from each parent index
        with incrementing seeds: seed, seed+1, seed+2, etc.

    Returns
    -------
    StructureContainer with all new structures added

    Notes
    -----
    Use container methods like filter_by_generation(), filter_by_pristine_structures(),
    etc. to get the target_indices before calling this function.

    Examples
    --------
    >>> # Create 100 structures from pristine, each with 2 random Al→Mg substitutions
    >>> container = create_substitution_batch_from_seed(
    ...     structure_container=container,
    ...     target_indices=[0],  # pristine
    ...     n=2,                 # 2 substitutions per structure
    ...     n_structures=100,    # Create 100 separate structures
    ...     seed=0,
    ...     from_element='Al',
    ...     to_element='Mg',
    ... )
    """
    container = structure_container
    rows_to_modify = target_indices

    structure_counter = 0
    for parent_idx in rows_to_modify:
        for copy_idx in range(n_structures):
            structure_seed = seed + structure_counter if seed is not None else None
            container = create_substitution_from_seed(
                structure_container=container,
                n=n,
                seed=structure_seed,
                from_element=from_element,
                to_element=to_element,
                parent_defect_index=parent_idx if not container._structures[parent_idx]['is_pristine'] else None,
                input_structure=container._structures[parent_idx]['structure'] if container._structures[parent_idx]['is_pristine'] else None,
                forbid_uids=forbid_uids,
                protect_history=protect_history,
            )
            structure_counter += 1

    out_container = container
    return out_container


# ============================================================================
# Interstitial Creation Functions
# ============================================================================

def create_interstitial_from_ids(
    structure_container: StructureContainer,
    sublattice: "np.ndarray",
    site_ids: List[int],
    element: str,
    parent_defect_index: Optional[int] = None,
    input_structure: Optional[Atoms] = None,
) -> StructureContainer:
    """
    Create interstitials at specific sites from an interstitial sublattice.

    Mirrors ``create_vacancy_from_ids`` / ``create_substitution_from_ids``:
    ``site_ids`` are indices into ``sublattice``, exactly as ``atom_ids`` are
    indices into the host structure for the other defect types.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify.
    sublattice : (N, 3) array-like
        Cartesian coordinates (Å) of the interstitial sublattice — e.g. the
        output of ``get_voronoi_interstitial_sites`` or ``get_voronoi_interstitial_sites_pymatgen``.
    site_ids : list of int
        Indices into ``sublattice`` selecting which sites to occupy.
    element : str
        Chemical symbol of the atom to insert (e.g. ``'Mg'``).
    parent_defect_index : int or None
        Index of a defect structure to build on (None = use pristine).
    input_structure : Atoms or None
        Explicit parent structure (alternative to ``parent_defect_index``).

    Returns
    -------
    StructureContainer with the new interstitial structure added.
    """
    import numpy as np

    container = structure_container

    # Resolve parent structure
    parent_idx = _resolve_parent(container, parent_defect_index, input_structure)
    parent = container.get_structure(parent_idx)

    atoms = ensure_uids(parent['structure']).copy()
    existing_events = parent['events'].copy()

    # Find pristine ancestor
    pristine_idx = container._find_pristine_index(parent_idx)

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
        new_events.append({
            "type": "interstitial",
            "element": str(element),
            "atom_uid": int(new_uid),
            "pos0": pos,
            "site_label": f"site_{int(sid)}",
        })

    all_new_events = existing_events + new_events

    n = len(site_ids)
    operation_str = f"interstitial[{n}:{element}]" if n > 1 else f"interstitial[{element}]"

    container.add_defect(
        atoms=atoms,
        operation=operation_str,
        pristine_index=pristine_idx,
        parent_index=parent_idx,
        events=all_new_events,
        metadata={'parent_index': parent_idx, 'pristine_index': pristine_idx},
    )

    return container


def create_interstitial_from_seed(
    structure_container: StructureContainer,
    sublattice: "np.ndarray",
    element: str,
    n: int = 1,
    seed: Optional[int] = None,
    parent_defect_index: Optional[int] = None,
    input_structure: Optional[Atoms] = None,
) -> StructureContainer:
    """
    Create interstitials by randomly sampling from an interstitial sublattice.

    Mirrors ``create_vacancy_from_seed`` / ``create_substitution_from_seed``:
    ``sublattice`` defines the candidate pool; ``n`` and ``seed`` control
    random selection without replacement.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify.
    sublattice : (N, 3) array-like
        Cartesian coordinates (Å) of all candidate interstitial sites — e.g.
        the ``all_sites`` output of ``get_voronoi_interstitial_sites``.
    element : str
        Chemical symbol of the atom to insert (e.g. ``'Mg'``).
    n : int
        Number of interstitial atoms to insert (default 1).
    seed : int or None
        Random seed for reproducibility.
    parent_defect_index : int or None
        Index of a defect structure to build on (None = use pristine).
    input_structure : Atoms or None
        Explicit parent structure (alternative to ``parent_defect_index``).

    Returns
    -------
    StructureContainer with the new interstitial structure added.
    """
    import numpy as np

    container = structure_container

    # Resolve parent structure
    parent_idx = _resolve_parent(container, parent_defect_index, input_structure)
    parent = container.get_structure(parent_idx)

    atoms = ensure_uids(parent['structure']).copy()
    existing_events = parent['events'].copy()

    # Find pristine ancestor
    pristine_idx = container._find_pristine_index(parent_idx)

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
        new_events.append({
            "type": "interstitial",
            "element": str(element),
            "atom_uid": int(new_uid),
            "pos0": pos,
            "site_label": f"site_{int(sid)}",
        })

    all_new_events = existing_events + new_events
    operation_str = f"interstitial[{n}:{element}]" if n > 1 else f"interstitial[{element}]"

    container.add_defect(
        atoms=arrays_copy,
        operation=operation_str,
        pristine_index=pristine_idx,
        parent_index=parent_idx,
        events=all_new_events,
        metadata={'parent_index': parent_idx, 'pristine_index': pristine_idx, 'seed': seed},
    )

    return container


def create_interstitial_batch_from_ids(
    structure_container: StructureContainer,
    target_indices: List[int],
    sublattice: "np.ndarray",
    element: str,
    site_ids: Optional[List[int]] = None,
    separate_structures: bool = True,
) -> StructureContainer:
    """
    Apply specific interstitial sites to multiple parent structures.

    Mirrors ``create_vacancy_batch_from_ids`` / ``create_substitution_batch_from_ids``.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify.
    target_indices : List[int]
        Absolute indices of parent structures in the container.
    sublattice : (N, 3) array-like
        Cartesian coordinates (Å) of the interstitial sublattice.
    element : str
        Chemical symbol of the atom to insert.
    site_ids : list of int or None
        Indices into ``sublattice`` selecting which sites to use.
        ``None`` (default) uses every site in ``sublattice``.
    separate_structures : bool
        ``True`` (default): one new structure per site per parent.
        ``False``: one new structure per parent containing all selected sites.

    Returns
    -------
    StructureContainer with all new structures added.
    """
    import numpy as np

    container = structure_container
    sublattice_arr = np.asarray(sublattice, float)
    effective_ids = list(range(len(sublattice_arr))) if site_ids is None else list(site_ids)

    if separate_structures:
        for parent_idx in target_indices:
            _pd = parent_idx if not container._structures[parent_idx]['is_pristine'] else None
            _is = container._structures[parent_idx]['structure'] if container._structures[parent_idx]['is_pristine'] else None
            for sid in effective_ids:
                container = create_interstitial_from_ids(
                    structure_container=container,
                    sublattice=sublattice_arr,
                    site_ids=[sid],
                    element=element,
                    parent_defect_index=_pd,
                    input_structure=_is,
                )
    else:
        for parent_idx in target_indices:
            _pd = parent_idx if not container._structures[parent_idx]['is_pristine'] else None
            _is = container._structures[parent_idx]['structure'] if container._structures[parent_idx]['is_pristine'] else None
            container = create_interstitial_from_ids(
                structure_container=container,
                sublattice=sublattice_arr,
                site_ids=effective_ids,
                element=element,
                parent_defect_index=_pd,
                input_structure=_is,
            )

    return container


def create_interstitial_batch_from_seed(
    structure_container: StructureContainer,
    target_indices: List[int],
    sublattice: "np.ndarray",
    element: str,
    n: int = 1,
    seed: Optional[int] = None,
    n_structures: int = 1,
) -> StructureContainer:
    """
    Randomly sample interstitial sites for multiple parent structures.

    Mirrors ``create_vacancy_batch_from_seed`` / ``create_substitution_batch_from_seed``.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify.
    target_indices : List[int]
        Absolute indices of parent structures in the container.
    sublattice : (N, 3) array-like
        Cartesian coordinates (Å) of all candidate interstitial sites — e.g.
        the ``all_sites`` output of ``get_voronoi_interstitial_sites``.
    element : str
        Chemical symbol of the atom to insert.
    n : int
        Number of interstitial atoms per structure (default 1).
    seed : int or None
        Base random seed; increments by 1 for each new structure so every
        structure is reproducibly distinct.
    n_structures : int
        Number of structures to generate from each parent (default 1).

    Returns
    -------
    StructureContainer with all new structures added.
    """
    import numpy as np

    container = structure_container
    sublattice_arr = np.asarray(sublattice, float)

    structure_counter = 0
    for parent_idx in target_indices:
        _pd = parent_idx if not container._structures[parent_idx]['is_pristine'] else None
        _is = container._structures[parent_idx]['structure'] if container._structures[parent_idx]['is_pristine'] else None
        for _ in range(n_structures):
            structure_seed = seed + structure_counter if seed is not None else None
            container = create_interstitial_from_seed(
                structure_container=container,
                sublattice=sublattice_arr,
                element=element,
                n=n,
                seed=structure_seed,
                parent_defect_index=_pd,
                input_structure=_is,
            )
            structure_counter += 1

    return container