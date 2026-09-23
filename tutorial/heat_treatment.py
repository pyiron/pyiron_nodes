"""Simple heat-treatment and strengthening models used in the aiflow tutorial.

The three nodes here form a short, self-contained process chain — annealing
schedule -> grain size -> yield strength — that runs instantly and needs no
simulation engine.  They exist so that a newcomer can build a complete, useful
workflow in the visual editor within a few minutes.

The physics is textbook level and deliberately transparent: parabolic grain
growth with an Arrhenius mobility, followed by a Hall-Petch strength estimate.
The defaults are representative of a low-alloy ferritic steel, but they are not
calibrated to any particular grade — treat the numbers as illustrative.
"""

from core import as_function_node


@as_function_node
def GrainGrowth(
    temperature_K: float = 1123.0,
    time_min: float = 60.0,
    d0_um: float = 5.0,
    k0_um2_per_s: float = 1.0e11,
    Q_kJ_per_mol: float = 250.0,
):
    """Grain size after isothermal annealing, from parabolic grain growth.

    Uses the classical parabolic law ``d^2 = d0^2 + k * t`` with an Arrhenius
    mobility ``k = k0 * exp(-Q / RT)``.  Any of the inputs may be an array, in
    which case an array of grain sizes is returned — that is how this node is
    used to sweep an annealing schedule.

    Args:
        temperature_K: annealing temperature in K.
        time_min: annealing time in minutes.
        d0_um: initial (as-received) grain size in micrometres.
        k0_um2_per_s: pre-exponential growth factor in um^2/s.
        Q_kJ_per_mol: activation energy for grain-boundary migration in kJ/mol.

    Returns:
        grain_size_um (numpy.ndarray): grain size after annealing, in micrometres.
    """
    import numpy as np

    R_kJ_per_mol_K = 8.314e-3  # gas constant in kJ/(mol K)
    seconds_per_minute = 60.0

    temperature = np.asarray(temperature_K, dtype=float)
    growth_rate = k0_um2_per_s * np.exp(-Q_kJ_per_mol / (R_kJ_per_mol_K * temperature))
    grain_size_um = np.sqrt(
        d0_um**2 + growth_rate * np.asarray(time_min, dtype=float) * seconds_per_minute
    )
    return grain_size_um


@as_function_node
def HallPetch(
    grain_size_um: float = 20.0,
    sigma_0_MPa: float = 70.0,
    k_y_MPa_sqrt_um: float = 740.0,
):
    """Yield strength from grain size via the Hall-Petch relation.

    ``sigma_y = sigma_0 + k_y / sqrt(d)`` — finer grains give a stronger
    material.  Together with :func:`GrainGrowth` this says that over-annealing
    coarsens the microstructure and costs strength.

    Args:
        grain_size_um: grain size in micrometres (scalar or array).
        sigma_0_MPa: friction stress, i.e. the strength of a single crystal, in MPa.
        k_y_MPa_sqrt_um: Hall-Petch slope in MPa*sqrt(um).

    Returns:
        sigma_y_MPa (numpy.ndarray): yield strength in MPa.
    """
    import numpy as np

    grain_size = np.asarray(grain_size_um, dtype=float)
    sigma_y_MPa = sigma_0_MPa + k_y_MPa_sqrt_um / np.sqrt(grain_size)
    return sigma_y_MPa


@as_function_node("fig")
def PlotYieldStrength(
    grain_size_um=None,
    sigma_y_MPa=None,
    title: str = "Hall-Petch: strength vs. grain size",
):
    """Plot yield strength against grain size.

    Args:
        grain_size_um: grain sizes in micrometres.
        sigma_y_MPa: the corresponding yield strengths in MPa.
        title: plot title.

    Returns:
        fig (matplotlib.figure.Figure): the finished figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(grain_size_um, sigma_y_MPa, "o-")
    ax.set_xlabel("Grain size [um]")
    ax.set_ylabel("Yield strength [MPa]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return fig
