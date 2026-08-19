from core import as_function_node
import pandas as pd
import numpy as np


@as_function_node
def MinMaxIndices(
    df: pd.DataFrame,
    i_min: int = 0,
    i_max: int = None,
    energy_only: bool = False,
):
    """Generate index arrays for energies and forces in a DataFrame.

    The function creates a flat index range covering both energy entries and
    force components for all structures in the DataFrame.

    Args:
        df: pandas DataFrame containing `NUMBER_OF_ATOMS`.
        i_min: Minimum structure index (inclusive).
        i_max: Maximum structure index (exclusive). If None, uses total structures.
        energy_only: If True, return only energy indices; otherwise include force indices.

    Returns:
        A numpy array of selected indices.
    """
    num_structures = len(df)
    num_atoms = np.sum(df.NUMBER_OF_ATOMS)

    indices = np.arange(num_structures + 3 * num_atoms)
    if i_max is None or i_max == "":
        i_max = num_structures
    energies = indices[i_min:i_max]
    forces = indices[num_atoms + 3 * i_min : num_atoms + 3 * i_max]
    if energy_only:
        indices = energies
    else:
        indices = np.append(energies, forces, axis=0)
    return indices
