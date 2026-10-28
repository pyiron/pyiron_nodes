from pyiron_nodes.controls import IterToDataFrame
from pyiron_nodes.dataframe import GetColumnFromDataFrame
from pyiron_nodes.math_utils import Linspace
from pyiron_nodes.plotting import Plot
from core import Workflow
from core import group_node
from core import as_function_node

# ── Local node definitions ──────────────────────


@as_function_node("T_melt")
def FindMeltingTemp(temperatures: list, mean_energies: list) -> float:
    """
    Rough melting-temperature estimate from a temperature-sweep MD.

    Scans the E(T) curve for the largest dE/dT, which corresponds to the
    solid-to-liquid latent-heat discontinuity.  Returns the midpoint of the
    bracketing interval as the approximate melting temperature.
    """
    import numpy as np

    T = np.asarray(temperatures, dtype=float)
    E = np.asarray(mean_energies, dtype=float)
    dEdT = np.diff(E) / np.diff(T)
    idx = int(np.argmax(dEdT))
    T_melt = float((T[idx] + T[idx + 1]) / 2.0)
    return T_melt


# ── Group node factories ─────────────────────────────


@group_node("mean")
def GetT(
    name,
    index,
    structure,
    potential,
    calc_dataclass,
    attr,
    cubic=False,
    temperature=300,
    n_ionic_steps=10000,
    pressure=None,
    repeat_scalar=1,
    working_directory=".",
):
    from pyiron_nodes.atomistic.calculator.data import InputCalcMD
    from pyiron_nodes.atomistic.engine.lammps import (
        CreateLammpsMDInput,
        CreateLammpsStructure,
        ListPotentials,
        ParseLammpsOutput,
        RunLammpsCalculation,
    )
    from pyiron_nodes.atomistic.structure.build import Bulk
    from pyiron_nodes.atomistic.structure.transform import FixSpecies, Repeat
    from pyiron_nodes.controls import GetAttribute, pick_element
    from pyiron_nodes.math_utils import Mean
    from core import Workflow

    inner_wf = Workflow("GetT")
    inner_wf.Bulk = Bulk(name=name, cubic=cubic)
    inner_wf.md_params = InputCalcMD(
        temperature=temperature, n_ionic_steps=n_ionic_steps, pressure=pressure
    )
    inner_wf.Repeat = Repeat(structure=inner_wf.Bulk, repeat_scalar=repeat_scalar)
    inner_wf.ListPotentials = ListPotentials(structure=inner_wf.Bulk)
    inner_wf.FixSpecies = FixSpecies(structure=inner_wf.Repeat)
    inner_wf.potential = pick_element(lst=inner_wf.ListPotentials, index=index)
    inner_wf.Lammps = Lammps(
        structure=inner_wf.FixSpecies,
        potential=inner_wf.potential,
        calc_dataclass=inner_wf.md_params,
        working_directory=working_directory,
    )
    inner_wf.energies = GetAttribute(obj=inner_wf.Lammps, attr=attr)
    inner_wf.mean_energy = Mean(numbers=inner_wf.energies)
    return inner_wf.mean_energy


wf = Workflow("estimate_melt_groups")

wf.GetT = GetT(
    name="Al",
    cubic=True,
    temperature=400,
    n_ionic_steps=3000,
    pressure=500,
    repeat_scalar=4,
    index=0,
    working_directory="./melt_sweep",
    attr="energies_pot",
)

wf.temperatures = Linspace(x_min=200.0, x_max=2000.0, num_points=10)

wf.temperature_sweep = IterToDataFrame(
    node=wf.GetT,
    input_label="temperature",
    values=wf.temperatures,
    debug=False,
    executor=None,
    store=True,
)

wf.mean_energies = GetColumnFromDataFrame(df=wf.temperature_sweep, column_name="mean")

wf.T_melt = FindMeltingTemp(
    temperatures=wf.temperatures, mean_energies=wf.mean_energies
)

wf.plot = Plot(y=wf.mean_energies, x=wf.temperatures)
