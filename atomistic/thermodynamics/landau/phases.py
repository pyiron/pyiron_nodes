from dataclasses import replace

import landau
import numpy as np

from core import as_function_node


@as_function_node("phase")
def LinePhase(
    name: str, concentration: float, energy: float, entropy: float
) -> landau.phases.LinePhase:
    import landau

    return landau.phases.LinePhase(name, concentration, energy, entropy)


@as_function_node("phase")
def TemperatureLinePhase(
    name: str,
    concentration: float,
    temperatures: np.ndarray | list[float],
    free_energies: np.ndarray | list[float],
    num_parameters: int = 3,
):
    return landau.phases.TemperatureDependentLinePhase(
        name,
        concentration,
        temperatures,
        free_energies,
        landau.interpolate.SGTE(num_parameters),
    )


@as_function_node("phase")
def IdealSolution(
    name: str, phase1: landau.phases.Phase, phase2: landau.phases.Phase
) -> landau.phases.Phase:
    import landau

    return landau.phases.IdealSolution(name, phase1, phase2)


def make_phase(dd, temperature_parameters, concentration_parameters):
    name = dd.phase.iloc[0]
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
    if concentration_parameters is None:
        # do not interpolate in concentration: keep the individual line phases,
        # whose names already carry their concentration
        return sub
    # minus 2 for terminals
    # minus 1 to be not exactly interpolating
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


@as_function_node(["phase_list", "phase_dict"])
def PhasesFromDataFrame(
    dataframe,
    temperature_parameters: int = 4,
    concentration_parameters: int | None = 1,
):
    """Convert a dataframe of free energies to list of phase objects.

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
                    when interpolating free energies in concentration; if None,
                    output the individual line phases instead, whose names
                    include their concentration

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
    # explode() flattens the concentration_parameters=None case, where make_phase
    # returns a list of line phases rather than a single solution phase
    phase_dict = {p.name: p for p in phases.explode()}
    print("Found phases:", *phase_dict.keys(), sep="\n")
    phase_list = list(phase_dict.values())
    return phase_list, phase_dict
