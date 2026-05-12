from core import as_function_node
import pandas as pd
import numpy as np


def _calc_rmse(array_1, array_2, rmse_in_milli: bool = True):
    """
    Calculates the RMSE value of two arrays

    Args:
    array_1: An array or list of energy or force values
    array_2: An array or list of energy or force values

    Returns:
    rmse_in_milli: (boolean, Default = True) Set False if you want the calculated RMSE value in decimals
    rmse: The calculated RMSE value
    """
    rmse = np.sqrt(np.mean((array_1 - array_2) ** 2))
    if rmse_in_milli:
        return rmse * 1000
    else:
        return rmse


# HISTOGRAM FOR ENERGY DISTRIBUTION
@as_function_node("plot")
def PlotEnergyHistogram(df: "pd.DataFrame", bins: int = 100, log_scale: bool = True):
    """
    Plot histogram of the per-atom energies.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'energy_corrected' and 'NUMBER_OF_ATOMS' columns.
    bins : int
        Number of histogram bins.
    log_scale : bool
        Whether to use a logarithmic scale on the y-axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated histogram figure.
    """
    import matplotlib.pyplot as plt

    # Calculate energy per atom (convert to meV/atom if desired)
    df["energy_per_atom"] = df["energy_corrected"] / df["NUMBER_OF_ATOMS"]

    fig, ax = plt.subplots()
    ax.hist(df["energy_per_atom"], bins=bins, log=log_scale)

    ax.set_ylabel("Count")
    ax.set_xlabel("Energy per atom (meV/atom)")
    return fig


# HISTOGRAM FOR FORCE DISTRIBUTION
@as_function_node("plot")
def PlotForcesHistogram(df: "pd.DataFrame", bins: int = 100, log_scale: bool = True):
    """
    Plot histogram of atomic force magnitudes.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a 'forces' column with per-atom force arrays.
    bins : int
        Number of histogram bins.
    log_scale : bool
        Whether to use a logarithmic scale on the y-axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated histogram figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    array = np.concatenate(df.forces.values).flatten()

    fig, ax = plt.subplots()
    ax.hist(array, bins=bins, log=log_scale)

    ax.set_ylabel("Count")
    ax.set_xlabel(r"Force (eV/$\mathrm{\AA}$)")
    return fig


@as_function_node("plot")
def PlotEnergyFittingCurve(data_dict: dict):
    """
    Plot predicted vs reference energies for training and optional testing datasets.

    Parameters
    ----------
    data_dict : dict
        Dictionary with keys:
        - 'reference_training_epa', 'predicted_training_epa'
        - optional: 'reference_testing_epa', 'predicted_testing_epa'
        All arrays are in eV/atom.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure for display.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # Plot y=x dashed reference line
    lims = [
        data_dict["reference_training_epa"].min(),
        data_dict["reference_training_epa"].max(),
    ]
    ax.plot(lims, lims, ls="--", color="C0")

    # Optional testing set
    if "reference_testing_epa" in data_dict and "predicted_testing_epa" in data_dict:
        rmse_testing = _calc_rmse(
            data_dict["reference_testing_epa"], data_dict["predicted_testing_epa"]
        )
        ax.scatter(
            data_dict["reference_testing_epa"],
            data_dict["predicted_testing_epa"],
            color="black",
            s=30,
            marker="+",
            label=f"Testing RMSE = {rmse_testing:.2f} (meV/atom)",
        )

    # Training set
    rmse_training = _calc_rmse(
        data_dict["reference_training_epa"], data_dict["predicted_training_epa"]
    )
    ax.scatter(
        data_dict["reference_training_epa"],
        data_dict["predicted_training_epa"],
        color="C0",
        s=30,
        label=f"Training RMSE = {rmse_training:.2f} (meV/atom)",
    )

    # Labels and title
    ax.set_xlabel("DFT E (eV/atom)")
    ax.set_ylabel("Predicted E (eV/atom)")
    ax.set_title("Predicted Energy Vs Reference Energy")
    ax.legend()

    return fig


@as_function_node("plot")
def PlotForcesFittingCurve(data_dict: dict):
    """
    Plot predicted vs reference atomic forces for training and optional testing datasets.

    Parameters
    ----------
    data_dict : dict
        Dictionary with keys:
        - 'reference_training_fpa', 'predicted_training_fpa'
        - optional: 'reference_testing_fpa', 'predicted_testing_fpa'
        All arrays are in eV/Å.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure for display.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # 1:1 reference line
    lims = [
        data_dict["reference_training_fpa"].min(),
        data_dict["reference_training_fpa"].max(),
    ]
    ax.plot(lims, lims, ls="--", color="C1")

    # Optional testing set
    if "reference_testing_fpa" in data_dict and "predicted_testing_fpa" in data_dict:
        rmse_testing = _calc_rmse(
            data_dict["reference_testing_fpa"], data_dict["predicted_testing_fpa"]
        )
        ax.scatter(
            data_dict["reference_testing_fpa"],
            data_dict["predicted_testing_fpa"],
            color="black",
            s=30,
            marker="+",
            label=f"Testing RMSE = {rmse_testing:.2f}" + r" (meV/$\AA$)",
        )

    # Training set
    rmse_training = _calc_rmse(
        data_dict["reference_training_fpa"], data_dict["predicted_training_fpa"]
    )
    ax.scatter(
        data_dict["reference_training_fpa"],
        data_dict["predicted_training_fpa"],
        color="C1",
        s=30,
        label=f"Training RMSE = {rmse_training:.2f}" + r" (meV/$\AA$)",
    )

    # Labels and title
    ax.set_xlabel(r"DFT $F_i$ (eV/$\AA$)")
    ax.set_ylabel(r"Predicted $F_i$ (eV/$\AA$)")
    ax.set_title("Predicted Force Vs Reference Force")
    ax.legend()

    return fig
