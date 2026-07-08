from core import as_function_node
import pandas as pd
import numpy as np


@as_function_node("product")
def Multiply(x, y):
    """Multiply two numbers and return the product.

    Args:
        x: First number.
        y: Second number.

    Returns:
        The product of x and y.
    """
    return x * y


@as_function_node("matrix")
def SliceArray(matrix, indices):
    """Slice a matrix (numpy array) using the provided indices.

    Args:
        matrix: A numpy array or similar matrix-like object.
        indices: Index or slice to select from the matrix.

    Returns:
        The sliced portion of the matrix.
    """
    return matrix[indices]


@as_function_node("vector")
def GetVector(
    df: pd.DataFrame,
    indices,
    scale_energy_per_atom: bool = False,
):
    """Extract a feature vector from a DataFrame.

    This function concatenates the corrected energies and flattened forces,
    optionally scaling the energies per atom.

    Args:
        df: pandas DataFrame containing `energy_corrected`, `NUMBER_OF_ATOMS`,
            and `forces` columns.
        indices: Indices to select from the assembled vector.
        scale_energy_per_atom: If True, divide the energy by the number of atoms.

    Returns:
        A numpy array containing the selected elements of the vector.
    """
    import numpy as np

    vec = df.energy_corrected
    if scale_energy_per_atom:
        vec /= df.NUMBER_OF_ATOMS

    forces_vec = []
    for f in df.forces.apply(lambda x: x.flatten()):
        forces_vec += list(f)
    vec = np.append(vec, forces_vec)
    return vec[indices]


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


@as_function_node("linspace")
def Linspace(
    x_min: float = 0,
    x_max: float = 1,
    num_points: int = 50,
    endpoint: bool = True,
):
    """Generate a list (or array) of values for a series of calculations.

    Purpose:
        Generate a list (or array) of values between a start and stop value.
        This function is generic and can be used for constructing a 1d mesh or lattice constants, energies,
        or any other set of quantities.

    Required Input Ports:
        - start (float): start of the interval.
        - stop (float): end of the interval.
        - num_points (int, optional): number of points to generate (alternative to step size).
        - step_size (float, optional): increment between successive values (alternative to num_points).

    Optional Parameters:
        - endpoint (bool, default True): whether to include the stop value.
        - log_scale (bool, default False): generate values on a logarithmic scale if True.

    Returns:
        - values (list of float or numpy.ndarray): generated values.
    """
    return np.linspace(x_min, x_max, num_points, endpoint=endpoint)
