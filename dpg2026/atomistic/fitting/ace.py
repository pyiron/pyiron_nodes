from dataclasses import asdict, dataclass, field
from typing import Optional
from core import as_function_node
import numpy as np
import pandas as pd


@dataclass
class EmbeddingsALL:
    npot: str = "FinnisSinclairShiftedScaled"
    fs_parameters: list[int] = field(default_factory=lambda: [1, 1])
    ndensity: int = 1


@dataclass
class Embeddings:
    ALL: EmbeddingsALL = field(default_factory=EmbeddingsALL)


@dataclass
class BondsALL:
    radbase: str = "SBessel"
    radparameters: list[float] = field(default_factory=lambda: [5.25])
    rcut: float | int = 7.0
    dcut: float = 0.01


@dataclass
class Bonds:
    ALL: BondsALL = field(default_factory=BondsALL)


@dataclass
class FunctionsALL:
    nradmax_by_orders: list[int] = field(default_factory=lambda: [15, 3, 2, 1])
    lmax_by_orders: list[int] = field(default_factory=lambda: [0, 3, 2, 1])


@dataclass
class Functions:
    number_of_functions_per_element: Optional[int] = None
    ALL: FunctionsALL = field(default_factory=FunctionsALL)


@dataclass
class PotentialConfig:
    deltaSplineBins: float = 0.001
    elements: list[str] | None = None

    embeddings: Embeddings = field(default_factory=Embeddings)
    bonds: Bonds = field(default_factory=Bonds)
    functions: Functions = field(default_factory=Functions)

    def __post_init__(self):
        if not isinstance(self.embeddings, Embeddings):
            self.embeddings = Embeddings()
        if not isinstance(self.bonds, Bonds):
            self.bonds = Bonds()
        if not isinstance(self.functions, Functions):
            self.functions = Functions()

    def to_dict(self):
        def remove_none(d):
            """Recursively remove None values from dictionaries."""
            if isinstance(d, dict):
                return {k: remove_none(v) for k, v in d.items() if v is not None}
            elif isinstance(d, list):
                return [remove_none(v) for v in d if v is not None]
            else:
                return d

        return remove_none(asdict(self))
    

@as_function_node
def ParameterizePotentialConfig(
    nrad_max: tuple | list = (15, 6, 4, 1),
    l_max: tuple | list = (0, 6, 5, 1),
    number_of_functions_per_element: int = 10,
    rcut: float = 7.0,
):

    potential_config = PotentialConfig()

    potential_config.bonds.ALL.rcut = rcut
    potential_config.functions.ALL.nradmax_by_orders = list(nrad_max)
    potential_config.functions.ALL.lmax_by_orders = list(l_max)
    potential_config.functions.number_of_functions_per_element = (
        number_of_functions_per_element
    )

    return potential_config


@as_function_node
def RunLinearFit(
    potential_config,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    verbose: bool = False,
    store: bool = True,
):

    from pyace import create_multispecies_basis_config
    from pyace.linearacefit import LinearACEDataset, LinearACEFit
    from pyiron_snippets.logger import logger

    logger.setLevel(30)

    elements_set = set()
    for at in df_train["ase_atoms"]:
        elements_set.update(at.get_chemical_symbols())
    for at in df_test["ase_atoms"]:
        elements_set.update(at.get_chemical_symbols())

    elements = sorted(elements_set)
    potential_config.elements = elements
    potential_config_dict = potential_config.to_dict()

    bconf = create_multispecies_basis_config(potential_config_dict)

    train_ds = LinearACEDataset(bconf, df_train)
    train_ds.construct_design_matrix(verbose=verbose)
    if df_test.empty is False:
        test_ds = LinearACEDataset(bconf, df_test)
        test_ds.construct_design_matrix(verbose=verbose)
    else:
        test_ds = None

    linear_fit = LinearACEFit(train_dataset=train_ds)
    linear_fit.fit()

    training_dict = linear_fit.compute_errors(train_ds)
    training_e_rmse = round(training_dict["epa_rmse"] * 1000, 2)
    training_f_rmse = round(training_dict["f_comp_rmse"] * 1000, 2)
    print("====================== TRAINING INFO ======================")
    print(f"Training E RMSE: {training_e_rmse:.2f} meV/atom")
    print(f"Training F RMSE: {training_f_rmse:.2f} meV/A")

    if test_ds is not None:
        testing_dict = linear_fit.compute_errors(test_ds)
        testing_e_rmse = round(testing_dict["epa_rmse"] * 1000, 2)
        testing_f_rmse = round(testing_dict["f_comp_rmse"] * 1000, 2)
        print("======================= TESTING INFO =======================")
        print(f"Testing E RMSE: {testing_e_rmse:.2f} meV/atom")
        print(f"Testing F RMSE: {testing_f_rmse:.2f} meV/A")

    basis = linear_fit.get_bbasis()
    return basis


@as_function_node
def SavePotential(basis, filename: str = ""):
    import os

    if filename == "":
        filename = f"{'_'.join(basis.elements_name)}_linear_potential"
        folder_name = "Linear_ace_potentials"
    else:
        folder_name = os.path.dirname(filename)
        filename = os.path.basename(filename)

    folder_name = "Linear_ace_potentials"
    os.makedirs(folder_name, exist_ok=True)

    current_path = os.getcwd()
    folder_path = current_path + "/" + folder_name
    # Saving yaml and yace files
    print(
        f'Potentials "{filename}.yaml" and "{filename}.yace" are saved in "{folder_path}".'
    )

    yace_file_path = f"{folder_path}/{filename}.yace"
    basis.save(f"{folder_path}/{filename}.yaml")
    basis.to_ACECTildeBasisSet().save_yaml(yace_file_path)

    return basis, yace_file_path


@as_function_node
def PredictEnergiesAndForces(
    basis, df_train: pd.DataFrame, df_test: pd.DataFrame, store: bool = True
):

    from pyace import PyACECalculator

    data_dict = {}

    ace = PyACECalculator(basis)

    training_structures = df_train.ase_atoms

    # Reference data
    training_number_of_atoms = df_train.NUMBER_OF_ATOMS.to_numpy()
    training_energies = df_train.energy_corrected.to_numpy()

    training_epa = training_energies / training_number_of_atoms
    training_fpa = np.concatenate(df_train.forces.to_numpy()).flatten()
    data_dict["reference_training_epa"] = training_epa
    data_dict["reference_training_fpa"] = training_fpa

    # Predicted data
    training_predict = _get_predicted_energies_forces(
        ace=ace, structures=training_structures
    )
    data_dict["predicted_training_epa"] = (
        np.array(training_predict[0]) / training_number_of_atoms
    )
    data_dict["predicted_training_fpa"] = np.concatenate(training_predict[1]).flatten()

    if df_test.empty is False:

        testing_structures = df_test.ase_atoms

        # Reference data
        testing_number_of_atoms = df_test.NUMBER_OF_ATOMS.to_numpy()
        testing_energies = df_test.energy_corrected.to_numpy()

        testing_epa = testing_energies / testing_number_of_atoms
        testing_fpa = np.concatenate(df_test.forces.to_numpy()).flatten()
        data_dict["reference_testing_epa"] = testing_epa
        data_dict["reference_testing_fpa"] = testing_fpa

        # Predicted data
        testing_predict = _get_predicted_energies_forces(
            ace=ace, structures=testing_structures
        )
        data_dict["predicted_testing_epa"] = (
            np.array(testing_predict[0]) / testing_number_of_atoms
        )
        data_dict["predicted_testing_fpa"] = np.concatenate(
            testing_predict[1]
        ).flatten()

    return data_dict


def _get_predicted_energies_forces(ace, structures):
    forces = []
    energies = []

    for s in structures:
        s.calc = ace
        energies.append(s.get_potential_energy())
        forces.append(s.get_forces())
        s.calc = None
    return energies, forces


@as_function_node("design_matrix")
def DesignMatrix(
    df: pd.DataFrame,
    potential_config: PotentialConfig,
    verbose: bool = False,
    store: bool = True,
):
    """
    Constructs the design matrix for the training dataset using the provided potential configuration.
    Args:
        df_train (pd.DataFrame): The training dataset containing ASE atoms and other properties.
        potential_config (PotentialConfig): The configuration for the potential.
    Returns:
        LinearACEDataset: The constructed design matrix for the training dataset.
    """

    from pyace import create_multispecies_basis_config
    from pyace.linearacefit import LinearACEDataset
    from pyiron_snippets.logger import logger

    logger.setLevel(30)

    elements_set = set()
    for atoms in df["ase_atoms"]:
        elements_set.update(atoms.get_chemical_symbols())

    elements = sorted(elements_set)
    potential_config.elements = elements
    potential_config_dict = potential_config.to_dict()

    bconf = create_multispecies_basis_config(potential_config_dict)

    ds = LinearACEDataset(bconf, df)
    ds.construct_design_matrix(verbose=verbose)
    return ds.design_matrix
