from dataclasses import replace

import landau
import numpy as np

from core import as_function_node


@as_function_node
def LinePhase(
    name: str, concentration: float, energy: float, entropy: float
) -> landau.phases.LinePhase:
    import landau

    phase = landau.phases.LinePhase(name, concentration, energy, entropy)
    return phase


@as_function_node
def TemperatureLinePhase(
    name: str,
    concentration: float,
    temperatures: np.ndarray | list[float],
    free_energies: np.ndarray | list[float],
    num_parameters: int | None = 3,
):
    phase = landau.phases.TemperatureDependentLinePhase(
        name,
        concentration,
        temperatures,
        free_energies,
        landau.interpolate.SGTE(num_parameters),
    )
    return phase


@as_function_node
def IdealSolution(
    name: str, phase1: landau.phases.Phase, phase2: landau.phases.Phase
) -> landau.phases.Phase:
    import landau

    phase = landau.phases.IdealSolution(name, phase1, phase2)
    return phase


def make_phase(dd, temperature_parameters, concentration_parameters):
    name = dd.phase.iloc[0]
    # minus 2 for terminals
    # minus 1 to be not exactly interpolating
    sub = [
        landau.phases.TemperatureDependentLinePhase(
            f"{row.phase}_{c:.03}",
            c,
            row.temperature,
            row.free_energy,
            interpolator=landau.interpolate.SGTE(temperature_parameters),
        )
        for c, row in dd.set_index("composition").iterrows()
    ]
    # only a single concentration
    if len(sub) == 1:
        return replace(sub[0], name=name)
    if concentration_parameters is not None:
        interp_params = min(len(dd) - 2 - 1, concentration_parameters)
        # terminals are present
        if len({0, 1}.intersection([s.line_concentration for s in sub])) == 2:
            if len(sub) == 2:  # only terminals are present
                return landau.phases.IdealSolution(name, *sub)
            else:
                return landau.phases.RegularSolution(name, sub, interp_params)
        else:
            return landau.phases.InterpolatingPhase(
                name, sub, interp_params, num_samples=1000
            )
    else:
        return sub


@as_function_node
def PhasesFromDataFrame(
    dataframe,
    temperature_parameters: int | None = 4,
    concentration_parameters: int | None = 1,
):
    """Convert a dataframe of free energies to list of phase objects.

    Prints the names of all found phases.

    Args:
        dataframe: should contain columns
                    `phase`: the name of the phase; rows of the same phase name
                             will be grouped in a solution phase
                    `composition`: the mole fraction of one of the constitutents
                    `temperature`: array of temperature at which free energy
                                   was sampled
                    `free_energy`: corresponding free energies
        temperature_parameters (int): how many parameters to use when
                    interpolating free energies in temperature
        concentration_parameters (int, optional): how many parameters to use
                    when interpolating free energies in concentration; if not
                    given output individual phases and change the name to
                    include the concentration

    Returns:
        list of Phase objects
        dict of Phase objects, where the dict keys are the names of the phases
    """
    phases = dataframe.groupby("phase")[dataframe.columns].apply(
        make_phase,
        include_groups=False,
        temperature_parameters=temperature_parameters,
        concentration_parameters=concentration_parameters,
    )
    phase_dict = {p.name: p for p in phases.explode()}
    print("Found phases:", *phase_dict.keys(), sep="\n")
    phase_list = list(phase_dict.values())
    return phase_list, phase_dict
