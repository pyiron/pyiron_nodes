from core import as_function_node
from typing import Optional
import pandas as pd
import numpy as np


@as_function_node
def ReadPickledDatasetAsDataframe(
    file_path: str = "", compression: Optional[str] = None
):

    from ase.atoms import Atoms as aseAtoms

    df = pd.read_pickle(file_path, compression=compression)

    # Atoms check
    if "atoms" in df.columns:
        at = df.iloc[0]["atoms"]
        # Checking that the elements themselves have the correct atoms format
        if isinstance(at, aseAtoms):
            df.rename(columns={"atoms": "ase_atoms"}, inplace=True)
    elif "ase_atoms" not in df.columns:
        raise ValueError(
            "DataFrame should contain 'atoms' or 'ase_atoms' (ASE atoms) columns"
        )

    # NUMBER OF ATOMS check
    if "NUMBER_OF_ATOMS" not in df.columns and "number_of_atoms" in df.columns:
        df.rename(columns={"number_of_atoms": "NUMBER_OF_ATOMS"}, inplace=True)

    df["NUMBER_OF_ATOMS"] = df["NUMBER_OF_ATOMS"].astype(int)

    # energy corrected check
    if "energy_corrected" not in df.columns and "energy" in df.columns:
        df.rename(columns={"energy": "energy_corrected"}, inplace=True)

    if "pbc" not in df.columns:
        df["pbc"] = df["ase_atoms"].map(lambda atoms: np.all(atoms.pbc))

    return df


@as_function_node
def SplitTrainingAndTesting(
    data_df: pd.DataFrame, training_frac: float = 0.5, random_state: int = 42
):
    """
    Splits the filtered dataframe into training and testing sets based on a fraction of the dataset

    Args:
        data_df: A pandas.DataFrame of the filtered data DataFrame
        training_frac: A float number which dictates what is the precentage of the dataset to be used for training should be set between 0 to 1
        random_state (default = 42): Sets the random seed used to shuffle the data

    Returns:
        df_training: The training dataframe
        df_testing: The testing dataframe
    """
    if isinstance(training_frac, float):
        training_frac = np.abs(training_frac)

    if training_frac > 1:
        print("""
            Can't have the training dataset more than 100 % of the dataset
            Setting the value to 100%
            """)
        training_frac = 1
    elif training_frac == 0:
        print("Can'fit with no training dataset\nSetting the value to 1%")
        training_frac = 0.01
    df_training = data_df.sample(frac=training_frac, random_state=random_state)
    df_testing = data_df.loc[(i for i in data_df.index if i not in df_training.index)]

    return df_training, df_testing





def make_linearfit(
    workflow_name: str,
    delete_existing_savefiles=False,
    file_path: str = "mgca.pckl.tgz",
    compression: str | None = None,
    training_frac: float | int = 0.5,
    number_of_functions_per_element: int | None = 10,
    rcut: float | int = 6.0,
):

    wf = Workflow(workflow_name, delete_existing_savefiles=delete_existing_savefiles)
    if wf.has_saved_content():
        return wf

    # Workflow connections
    wf.load_dataset = ReadPickledDatasetAsDataframe(
        file_path=file_path, compression=compression
    )
    wf.split_dataset = SplitTrainingAndTesting(
        data_df=wf.load_dataset.outputs.df, training_frac=training_frac
    )
    wf.parameterize_potential = ParameterizePotentialConfig(
        number_of_functions_per_element=number_of_functions_per_element, rcut=rcut
    )
    wf.run_linear_fit = RunLinearFit(
        potential_config=wf.parameterize_potential,
        df_train=wf.split_dataset.outputs.df_training,
        df_test=wf.split_dataset.outputs.df_testing,
        verbose=False,
    )
    wf.save_potential = SavePotential(basis=wf.run_linear_fit.outputs.basis)
    wf.predict_energies_forces = PredictEnergiesAndForces(
        basis=wf.save_potential.outputs.basis,
        df_train=wf.split_dataset.outputs.df_training,
        df_test=wf.split_dataset.outputs.df_testing,
    )

    # Input mapping
    wf.inputs_map = {
        "run_linear_fit__verbose": "verbose",
        "save_potential__filename": "filename",
        "parameterize_potential__number_of_functions_per_element": "number_of_functions_per_element",
        "parameterize_potential__rcut": "rcut",
    }

    # Output maping
    wf.outputs_map = {
        "save_potential__yace_file_path": "yace_file_path",
        "predict_energies_forces__data_dict": "data_dict",
    }

    return wf
