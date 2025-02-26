from collections import defaultdict
import numpy as np
from itertools import combinations_with_replacement

RCORE = {
    # RCORE as per POTCAR * bohr to angtrom factor
    "H": 1.100000 * 0.5291773,
    "He": 1.100000 * 0.5291773,
    "Li": 2.050000 * 0.5291773,
    "Be": 1.900000 * 0.5291773,
    "B": 1.700000 * 0.5291773,
    "C": 1.500000 * 0.5291773,
    "N": 1.500000 * 0.5291773,
    "O": 1.520000 * 0.5291773,
    "F": 1.520000 * 0.5291773,
    "Ne": 1.700000 * 0.5291773,
    "Na": 2.200000 * 0.5291773,
    "Mg": 2.000000 * 0.5291773,
    "Al": 1.900000 * 0.5291773,
    "Si": 1.900000 * 0.5291773,
    "P": 1.900000 * 0.5291773,
    "S": 1.900000 * 0.5291773,
    "Cl": 1.900000 * 0.5291773,
    "Ar": 1.900000 * 0.5291773,
    "K": 2.300000 * 0.5291773,
    "Ca": 2.300000 * 0.5291773,
    "Sc": 2.500000 * 0.5291773,
    "Ti": 2.800000 * 0.5291773,
    "V": 2.700000 * 0.5291773,
    "Cr": 2.500000 * 0.5291773,
    "Mn": 2.300000 * 0.5291773,
    "Fe": 2.300000 * 0.5291773,
    "Co": 2.300000 * 0.5291773,
    "Ni": 2.300000 * 0.5291773,
    "Cu": 2.300000 * 0.5291773,
    "Zn": 2.300000 * 0.5291773,
    "Ga": 2.600000 * 0.5291773,
    "Ge": 2.300000 * 0.5291773,
    "As": 2.100000 * 0.5291773,
    "Se": 2.100000 * 0.5291773,
    "Br": 2.100000 * 0.5291773,
    "Kr": 2.300000 * 0.5291773,
    "Rb": 2.500000 * 0.5291773,
    "Sr": 2.500000 * 0.5291773,
    "Y": 2.800000 * 0.5291773,
    "Zr": 3.000000 * 0.5291773,
    "Nb": 2.400000 * 0.5291773,
    "Mo": 2.750000 * 0.5291773,
    "Tc": 2.800000 * 0.5291773,
    "Ru": 2.700000 * 0.5291773,
    "Rh": 2.700000 * 0.5291773,
    "Pd": 2.600000 * 0.5291773,
    "Ag": 2.500000 * 0.5291773,
    "Cd": 2.300000 * 0.5291773,
    "In": 3.100000 * 0.5291773,
    "Sn": 3.000000 * 0.5291773,
    "Sb": 2.300000 * 0.5291773,
    "Te": 2.300000 * 0.5291773,
    "I": 2.300000 * 0.5291773,
    "Xe": 2.500000 * 0.5291773,
    "Cs": 2.500000 * 0.5291773,
    "Ba": 2.800000 * 0.5291773,
    "La": 2.800000 * 0.5291773,
    "Hf": 3.000000 * 0.5291773,
    "Ta": 2.900000 * 0.5291773,
    "W": 2.750000 * 0.5291773,
    "Re": 2.700000 * 0.5291773,
    "Os": 2.700000 * 0.5291773,
    "Ir": 2.600000 * 0.5291773,
    "Pt": 2.600000 * 0.5291773,
    "Au": 2.500000 * 0.5291773,
    "Hg": 2.500000 * 0.5291773,
    "Tl": 3.200000 * 0.5291773,
    "Pb": 3.100000 * 0.5291773,
    "Bi": 3.000000 * 0.5291773,
}

def _element_wise_dist(structure):
    """
    Computes the minimum distance between each pair of elements in the structure using pymatgen.

    Parameters:
    structure (Structure): A pymatgen Structure object.

    Returns:
    dict: A dictionary with element pairs as keys and their minimum distance as values.
    """
    pair = defaultdict(lambda: np.inf)

    # Expand the structure to avoid PBC issues
    sr = structure  # * [2, 2, 2]  # Repeat structure in all directions
    # sr.remove_site_property("pbc")  # Turn off periodic boundary conditions if required

    # Get neighbors within a cutoff radius
    neighbors = sr.get_all_neighbors(
        r=5.0, include_index=True
    )  # Adjust the radius as needed

    # Loop through each atom and its neighbors
    for i, neighbor_list in enumerate(neighbors):
        for neighbor in neighbor_list:
            j, d = neighbor.index, neighbor.nn_distance
            ei, ej = sorted((sr[i].specie.symbol, sr[j].specie.symbol))
            pair[ei, ej] = min(d, pair[ei, ej])

    return pair

def is_valid_structure(structure, min_dist = 1.0, core_overlap_tolerance=0.2):
    # Validate both the minimum distance and RCORE-based constraints
    if get_minimum_distance(structure) < min_dist:
        return False
    return filter_distance_by_species(structure, core_overlap_tolerance=core_overlap_tolerance)
    
def get_minimum_distance(structure):
    """
    Computes the minimum interatomic distance in a structure, excluding self-site distances.
    
    Parameters:
    structure (Structure): A pymatgen Structure object.

    Returns:
    float: The minimum interatomic distance.
    """
    distance_matrix = structure.distance_matrix
    np.fill_diagonal(distance_matrix, np.inf)  # Exclude self-site distances
    return np.min(distance_matrix)


def filter_distance_by_species(structure, RCORE=RCORE, core_overlap_tolerance=0.2):
    """
    Checks if all interatomic distances in the structure are greater than the sum of 
    the core radii for the involved element pairs, allowing for a percentage overlap tolerance.

    Parameters:
    structure (Structure): A pymatgen Structure object.
    RCORE (dict): Dictionary of core radii for elements.
    core_overlap_tolerance (float): Fractional overlap tolerance allowed (e.g., 0.2 for 20%).

    Returns:
    bool: True if all interatomic distances are within the allowed overlap tolerance, False otherwise.
    """
    if len(structure) == 1:
        structure_copy = structure * [2, 2, 2]
    else:
        structure_copy = structure.copy()

    pair = _element_wise_dist(structure)
    species_list = sorted(set([site.specie.symbol for site in structure]))

    for ei, ej in combinations_with_replacement(species_list, 2):
        # Allow for overlap tolerance
        allowed_distance = (1 - core_overlap_tolerance) * (RCORE[ei] + RCORE[ej])
        if pair[ei, ej] < allowed_distance:
            return False
    return True
