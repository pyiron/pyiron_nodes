from atomistics.calculators.wrapper import as_task_dict_evaluator
from pyiron_workflow import as_function_node


@as_task_dict_evaluator
def evaluate_with_lammps(structure, tasks, potential_dataframe):
    import os
    import shutil
    from pyiron_atomistics.lammps.lammps import lammps_function

    results = {}
    path_lmp_calculation = os.path.abspath("lmp_working_directory")
    if os.path.exists(path_lmp_calculation):
        shutil.rmtree(path_lmp_calculation)
    if "calc_energy" in tasks:
        shell_output, parsed_output, job_crashed = lammps_function(
            working_directory=path_lmp_calculation,
            structure=structure,
            potential=potential_dataframe,
        )
        return {"energy": parsed_output["generic"]["energy_tot"][-1]}
    else:
        raise ValueError("The LAMMPS calculator does not implement:", tasks)
    return results


@as_function_node("structure")
def get_bulk_structure(element: str):
    from ase.build import bulk

    return bulk(element, cubic=True)


@as_function_node("generated_structures")
def generate_structures(structure, vol_range: float = 0.1, num_points: int = 5):
    from atomistics.workflows.evcurve.helper import generate_structures_helper

    return generate_structures_helper(
        structure=structure,
        vol_range=vol_range,
        num_points=num_points,
    )


@as_function_node("results")
def evaluate_with_lammps_wf(task_dict: dict, potential: str):
    return evaluate_with_lammps(
        task_dict={"calc_energy": task_dict}, potential_dataframe=potential
    )


@as_function_node("results")
def analyse_structures(output_dict: dict, structure_dict: dict):
    from atomistics.workflows.evcurve.helper import analyse_structures_helper

    return analyse_structures_helper(
        output_dict=output_dict,
        structure_dict=structure_dict,
        fit_type="polynomial",
        fit_order=3,
    )


@as_function_node("plot")
def plot(fit_dict: dict):
    import matplotlib.pyplot as plt

    plt.plot(fit_dict["volume"], fit_dict["energy"])
    # plt.xlabel("Volume ($\AA^3$)")
    plt.ylabel("Energy (eV)")
    return plt.show()
