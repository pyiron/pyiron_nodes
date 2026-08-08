from pyiron_nodes.controls import IterToDataFrame
from pyiron_nodes.math_utils import Linspace
from pyiron_nodes.plotting import InputPlotOptions, PlotDataFrameXY
from core import Workflow
from core import group_node

# ── Group node factories ─────────────────────────────


@group_node("T_melt")
def GetMeltingT(
    name,
    potential,
    liquid_fe__potential,
    cubic=False,
    pressure=0,
    temperature_stop=600,
    repeat_scalar=1,
    stdev=0.1,
    store=True,
    liquid_fe__store=True,
):
    from pyiron_nodes.atomistic.structure.build import Bulk
    from pyiron_nodes.atomistic.structure.transform import Repeat
    from pyiron_nodes.dpg2026.atomistic.calculator.calphy import (
        FindMeltingTemperature,
        InputClass,
        LiquidFreeEnergyWithTemp,
        SolidFreeEnergyWithTemp,
    )
    from pyiron_nodes.dpg2026.atomistic.engine.lammps import ListPotentials
    from pyiron_nodes.dpg2026.atomistic.structure.transform import Rattle
    from core import Workflow

    inner_wf = Workflow("GetMeltingT")
    inner_wf.Bulk = Bulk(name=name, cubic=cubic)
    inner_wf.InputClass = InputClass(
        pressure=pressure, temperature_stop=temperature_stop
    )
    inner_wf.Repeat = Repeat(structure=inner_wf.Bulk, repeat_scalar=repeat_scalar)
    inner_wf.ListPotentials = ListPotentials(structure=inner_wf.Bulk)
    inner_wf.rattle = Rattle(structure=inner_wf.Repeat, stdev=stdev)
    inner_wf.solid_fe = SolidFreeEnergyWithTemp(
        inp=inner_wf.InputClass,
        structure=inner_wf.Repeat,
        potential=potential,
        store=store,
    )
    inner_wf.liquid_fe = LiquidFreeEnergyWithTemp(
        inp=inner_wf.InputClass,
        structure=inner_wf.rattle,
        potential=liquid_fe__potential,
        store=liquid_fe__store,
    )
    inner_wf.T_melt_val = FindMeltingTemperature(
        temp_solid=inner_wf.solid_fe.outputs.temperature,
        fe_solid=inner_wf.solid_fe.outputs.free_energy,
        temp_liquid=inner_wf.liquid_fe.outputs.temperature,
        fe_liquid=inner_wf.liquid_fe.outputs.free_energy,
    )
    return inner_wf.T_melt_val


wf = Workflow("pt_phase_diag1")

wf.GetMeltingT = GetMeltingT(
    name="Al",
    cubic=True,
    pressure=5,
    temperature_stop=500,
    repeat_scalar=5,
    stdev=0.5,
    potential="1995--Angelo-J-E--Ni-Al-H--LAMMPS--ipr1",
    store=False,
    liquid_fe__potential="1995--Angelo-J-E--Ni-Al-H--LAMMPS--ipr1",
    liquid_fe__store=False,
)

wf.PlotOptions = InputPlotOptions(
    title="P-T Phase Diagram (solid-liquid boundary)", legend_label="T_melt (K)"
)

wf.Pressures = Linspace(x_max=50000.0, num_points=2)

wf.pressure_sweep = IterToDataFrame(
    node=wf.GetMeltingT,
    input_label="pressure",
    values=wf.Pressures,
    debug=False,
    executor=None,
    store=True,
)

wf.PlotDataFrameXY = PlotDataFrameXY(df=wf.pressure_sweep, options=wf.PlotOptions)
