from core import as_function_node


@as_function_node
def ConvertLoadToStress(df, area):
    """Convert a tensile-test load column into engineering stress.

    Args:
        df: DataFrame with a 'Load' column in kN and an
            'Extensometer elongation' column in percent.
        area: specimen cross section in mm^2.

    Returns:
        stress (numpy.ndarray): engineering stress in MPa.
        strain (numpy.ndarray): engineering strain in percent, offset-corrected
            so that the first point is zero.
    """
    kN_to_N = 1e3  # kiloNewton -> Newton
    # 1 N/mm^2 == 1 MPa, so N / mm^2 is already MPa; no further scaling needed.
    df["Stress"] = df["Load"] * kN_to_N / float(area)
    # although it says extensometer elongation, the values are in percent!
    strain = df["Extensometer elongation"].values.flatten()
    # subtract the offset from the dataset
    strain = strain - strain[0]
    stress = df["Stress"].values.flatten()
    return stress, strain


@as_function_node
def CalculateYoungsModulus(stress, strain, strain_cutoff=0.2):
    """Fit the elastic slope of a stress-strain curve up to ``strain_cutoff``.

    Args:
        stress: engineering stress in MPa.
        strain: engineering strain in percent.
        strain_cutoff: upper end of the fit range, in percent.

    Returns:
        youngs_modulus (float): Young's modulus in GPa.
    """
    import numpy as np

    percent_to_fraction = 100  # convert
    MPa_to_GPa = 1 / 1000  # convert MPa to GPa
    arg = np.argsort(np.abs(np.array(strain) - strain_cutoff))[0]
    fit = np.polyfit(strain[:arg], stress[:arg], 1)
    youngs_modulus = fit[0] * percent_to_fraction * MPa_to_GPa
    return youngs_modulus


@as_function_node("fig")
def PlotStressStrain(stress, strain, format="-"):
    """Plot a stress-strain curve (stress in MPa vs. strain in percent)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(strain, stress, format)
    ax.set_xlabel("Strain [%]")
    ax.set_ylabel("Stress [MPa]")
    return fig
