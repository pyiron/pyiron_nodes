from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import Repeat
from pyiron_nodes.dpg2026.atomistic.calculator.calphy import (
    InputClass,
    SolidFreeEnergyWithTemp,
    LiquidFreeEnergyWithTemp,
    FindMeltingTemperature,
    PlotSolidLiquidFreeEnergy,
)
from pyiron_nodes.dpg2026.atomistic.engine.lammps import ListPotentials
from pyiron_nodes.dpg2026.atomistic.structure.transform import Rattle
from pyiron_nodes.controls import IterToDataFrame
from pyiron_nodes.dataframe import GetColumnFromDataFrame
from pyiron_nodes.math_utils import Linspace
from pyiron_nodes.plotting import Scatter, InputPlotOptions
from core import Workflow, group_node

_POTENTIAL = "1995--Angelo-J-E--Ni-Al-H--LAMMPS--ipr1"


@group_node("T_melt")
def MeltingAtPressure(
    structure,
    potential: str = _POTENTIAL,
    pressure: float = 0.0,
    temperature_start: float = 300.0,
    temperature_stop: float = 1200.0,
):
    """Compute the melting temperature at a given pressure using calphy."""
    inner = Workflow("MeltingAtPressure")
    inner.inp = InputClass(
        pressure=pressure,
        temperature=temperature_start,
        temperature_stop=temperature_stop,
    )
    # Solid sweep uses the pristine (unrattled) supercell.
    inner.solid_fe = SolidFreeEnergyWithTemp(
        inp=inner.inp,
        structure=structure,
        potential=potential,
    )
    # Liquid sweep starts from a rattled/disordered copy so it melts.
    inner.rattle = Rattle(structure=structure, stdev=0.5)
    inner.liquid_fe = LiquidFreeEnergyWithTemp(
        inp=inner.inp,
        structure=inner.rattle,
        potential=potential,
    )
    # inner label "T_melt_val" is intentionally distinct from the outer alias
    # "T_melt" to avoid the group_node label-collision pitfall (guide §3)
    inner.T_melt_val = FindMeltingTemperature(
        temp_solid=inner.solid_fe.outputs.temperature,
        fe_solid=inner.solid_fe.outputs.free_energy,
        temp_liquid=inner.liquid_fe.outputs.temperature,
        fe_liquid=inner.liquid_fe.outputs.free_energy,
    )
    return inner.T_melt_val


# ── Top-level P-T phase diagram workflow ──────────────────────────────────────

wf = Workflow("pt_phase_diagram")

wf.Bulk = Bulk(name="Al", cubic=True)
wf.Repeat = Repeat(structure=wf.Bulk, repeat_scalar=5)
wf.ListPotentials = ListPotentials(structure=wf.Bulk)  # informational: lists available potentials

# Pressure range in bars (0–50 000 bar ≈ 0–5 GPa); adjust num_points and x_max as needed
wf.Pressures = Linspace(x_min=0.0, x_max=50000.0, num_points=6)

# Template: structure and potential are fixed; pressure is the sweep axis.
# No store ports here — template nodes inside IterToDataFrame must omit store (guide §6)
wf.melting_template = MeltingAtPressure(
    structure=wf.Repeat,
    potential=_POTENTIAL,
    temperature_start=300.0,
    temperature_stop=1200.0,
)

# Sweep pressure → DataFrame with columns ["pressure", "T_melt"]
wf.pressure_sweep = IterToDataFrame(
    node=wf.melting_template,
    input_label="pressure",
    values=wf.Pressures.outputs.linspace,
)

# Extract the T_melt column as a list
wf.T_melt_col = GetColumnFromDataFrame(
    df=wf.pressure_sweep.outputs.df,
    column_name="T_melt",
)

wf.PlotOptions = InputPlotOptions(
    title="P-T Phase Diagram (solid-liquid boundary)",
    legend_label="T_melt (K)",
)

# P-T phase boundary scatter plot: x = pressure (bar), y = T_melt (K)
wf.pt_plot = Scatter(
    x=wf.Pressures.outputs.linspace,
    y=wf.T_melt_col.outputs.column,
    options=wf.PlotOptions,
)


# ── Reference-pressure detailed free-energy plot (P = 0) ──────────────────────
# A single-pressure solid/liquid free-energy comparison used to visually
# verify the melting-point crossing.  SolidFreeEnergyWithTemp /
# LiquidFreeEnergyWithTemp already expose a `store` port (default True), so
# these expensive single computations are hash-cached automatically.
wf.ref_inp = InputClass(pressure=0, temperature=300, temperature_stop=1200)

wf.ref_solid_fe = SolidFreeEnergyWithTemp(
    inp=wf.ref_inp,
    structure=wf.Repeat,  # unrattled crystal → solid phase
    potential=_POTENTIAL,
)

wf.ref_rattle = Rattle(structure=wf.Repeat, stdev=0.5)
wf.ref_liquid_fe = LiquidFreeEnergyWithTemp(
    inp=wf.ref_inp,
    structure=wf.ref_rattle,  # rattled start → liquid phase
    potential=_POTENTIAL,
)

wf.ref_T_melt = FindMeltingTemperature(
    temp_solid=wf.ref_solid_fe.outputs.temperature,
    fe_solid=wf.ref_solid_fe.outputs.free_energy,
    temp_liquid=wf.ref_liquid_fe.outputs.temperature,
    fe_liquid=wf.ref_liquid_fe.outputs.free_energy,
)

wf.ref_free_energy_plot = PlotSolidLiquidFreeEnergy(
    temp_solid=wf.ref_solid_fe.outputs.temperature,
    fe_solid=wf.ref_solid_fe.outputs.free_energy,
    temp_liquid=wf.ref_liquid_fe.outputs.temperature,
    fe_liquid=wf.ref_liquid_fe.outputs.free_energy,
    T_melt=wf.ref_T_melt.outputs.T_melt,
)
