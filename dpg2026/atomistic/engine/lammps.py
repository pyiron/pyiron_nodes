from core import as_function_node


@as_function_node
def ListPotentials(structure):
    from pyiron_lammps.potential import view_potentials, get_resource_path_from_conda

    resource_path = get_resource_path_from_conda()
    potentials = list(
        view_potentials(structure, resource_path=resource_path)["Name"].values
    )
    return potentials
