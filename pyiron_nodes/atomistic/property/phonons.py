from typing import Optional, Union

from pyiron_workflow import as_function_node
from pyiron_nodes.dev_tools import wf_data_class, parse_input_kwargs

from phonopy.api_phonopy import Phonopy


@wf_data_class(doc_func=Phonopy.generate_displacements)
class InputPhonopyGenerateSupercells:
    distance: float = 0.01
    is_plusminus: Union[str, bool] = "auto"
    is_diagonal: bool = True
    is_trigonal: bool = False
    number_of_snapshots: Optional[int] = None
    random_seed: Optional[int] = None
    temperature: Optional[float] = None
    cutoff_frequency: Optional[float] = None
    max_distance: Optional[float] = None


# @function_node()
def generate_supercells(phonopy, parameters: InputPhonopyGenerateSupercells):
    from structuretoolkit.common import phonopy_to_atoms

    phonopy.generate_displacements(**parameters)

    supercells = [phonopy_to_atoms(s) for s in phonopy.supercells_with_displacements]
    return supercells


@as_function_node("parameters")
def PhonopyParameters(
        distance: float = 0.01,
        is_plusminus: Union[str, bool] = "auto",
        is_diagonal: bool = True,
        is_trigonal: bool = False,
        number_of_snapshots: Optional[int] = None,
        random_seed: Optional[int] = None,
        temperature: Optional[float] = None,
        cutoff_frequency: Optional[float] = None,
        max_distance: Optional[float] = None,
) -> dict:
    return {
        "distance": distance,
        "is_plusminus": is_plusminus,
        "is_diagonal": is_diagonal,
        "is_trigonal": is_trigonal,
        "number_of_snapshots": number_of_snapshots,
        "random_seed": random_seed,
        "temperature": temperature,
        "cutoff_frequency": cutoff_frequency,
        "max_distance": max_distance,
    }


# The following function should be defined as a workflow macro (presently not possible)
@as_function_node()
def create_phonopy(
        structure,
        engine=None,
        executor=None,
        max_workers=1,
        parameters: Optional[InputPhonopyGenerateSupercells | dict] = None,
):
    from phonopy import Phonopy
    from structuretoolkit.common import atoms_to_phonopy
    import pyiron_workflow

    phonopy = Phonopy(unitcell=atoms_to_phonopy(structure))

    cells = generate_supercells(
        phonopy,
        parameters=parameters,
        #parameters=parse_input_kwargs(parameters, InputPhonopyGenerateSupercells),
    )

    from pyiron_nodes.atomistic.calculator.ase import static as calculator
    gs = calculator(engine=engine)
    df_new = gs.iter(structure=cells)  # , executor=executor, max_workers=max_workers)
    # print ('df: ', df_new)
    # print ('dataframe: ', df_new.out.keys())
    # return df_new
    df_new = extract_df(df_new, key='energy')
    df_new = extract_df(df_new, key='forces', col='out')
    phonopy.forces = df_new.forces

    # could be automatized (out = collect(gs, log_level))
    out = {}
    out["energies"] = df_new.energy
    out["forces"] = df_new.forces
    out["df"] = df_new

    return phonopy, out


def extract_df(df, key='energy', col=None):
    val = [i[key][-1] for i in df.out.values]
    df[key] = val
    if col is not None:
        del df[col]
    return df


@as_function_node()
def get_dynamical_matrix(phonopy, q=[0, 0, 0]):
    import numpy as np

    if phonopy.dynamical_matrix is None:
        phonopy.produce_force_constants()
        phonopy.dynamical_matrix.run(q=q)
    dynamical_matrix = np.real_if_close(phonopy.dynamical_matrix.dynamical_matrix)
    # print (dynamical_matrix)
    return dynamical_matrix


@as_function_node()
def get_eigenvalues(matrix):
    import numpy as np

    ew = np.linalg.eigvalsh(matrix)
    return ew


@as_function_node()
def check_consistency(phonopy, tolerance: float = 1e-10):
    dyn_matrix = get_dynamical_matrix(phonopy).run()
    ew = get_eigenvalues(dyn_matrix).run()

    ew_lt_zero = ew[ew < -tolerance]
    if len(ew_lt_zero) > 0:
        print(f"WARNING: {len(ew_lt_zero)} imaginary modes exist")
        has_imaginary_modes = True
    else:
        has_imaginary_modes = False
    return has_imaginary_modes


@as_function_node()
def get_total_dos(phonopy, mesh=3 * [10]):
    from pandas import DataFrame

    phonopy.produce_force_constants()
    phonopy.run_mesh(mesh=mesh)
    phonopy.run_total_dos()
    total_dos = DataFrame(phonopy.get_total_dos_dict())
    return total_dos


nodes = [
    #    generate_supercells,
    create_phonopy,
    PhonopyParameters,
    get_dynamical_matrix,
    get_eigenvalues,
    check_consistency,
    get_total_dos,
]