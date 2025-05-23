from pyiron_workflow import Workflow
import ase.units as units

@Workflow.wrap.as_function_node("water")
def build_water(project, n_mols : int = 10):

    import numpy as np
    
    density = 1.0e-24  # g/A^3
    mol_mass_water = 18.015  # g/mol
    
    # Determining the supercell size size
    mass = mol_mass_water * n_mols / units.mol  # g
    vol_h2o = mass / density  # in A^3
    a = vol_h2o ** (1.0 / 3.0)  # A
    
    # Constructing the unitcell
    n = int(round(n_mols ** (1.0 / 3.0)))
    
    dx = 0.7
    r_O = [0, 0, 0]
    r_H1 = [dx, dx, 0]
    r_H2 = [-dx, dx, 0]
    unit_cell = (a / n) * np.eye(3)
    water = project.create_atoms(
        elements=["H", "H", "O"], positions=[r_H1, r_H2, r_O], cell=unit_cell, pbc=True
    )
    water.set_repeat([n, n, n])
    return(water)