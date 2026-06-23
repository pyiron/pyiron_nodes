from pyiron_nodes.dpg2026.atomistic.calculator.optimize import (
    GenericOptimizerSettings,
    Relax,
)
from pyiron_nodes.dpg2026.atomistic.engine.grace import Grace
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.dpg2026.atomistic.structure.transform import Repeat
from pyiron_nodes.atomistic.diffusion import AddInterstitialH, PlotNEBPath, RunNEB
from core import Workflow


wf = Workflow("h_diffusion_in_fcc_al")

wf.al_unit = Bulk(name="Al", crystalstructure="fcc", a=4.05, cubic=True)

wf.grace = Grace(model="GRACE-2L-OAM")

wf.opt_settings = GenericOptimizerSettings(max_steps=300, force_tolerance=0.02)

wf.al_supercell = Repeat(structure=wf.al_unit, repeat_scalar=2)

wf.h_initial = AddInterstitialH(structure=wf.al_supercell, frac_pos=[0.25, 0.0, 0.0])

wf.h_final = AddInterstitialH(structure=wf.al_supercell, frac_pos=[0.0, 0.25, 0.0])

wf.initial_relaxed = Relax(
    structure=wf.h_initial,
    engine=wf.grace,
    opt_parameters=wf.opt_settings,
    opt_mode="internal",
)
wf.initial_relaxed.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)

wf.final_relaxed = Relax(
    structure=wf.h_final,
    engine=wf.grace,
    opt_parameters=wf.opt_settings,
    opt_mode="internal",
)
wf.final_relaxed.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)

wf.neb = RunNEB(
    initial_state=wf.initial_relaxed, final_state=wf.final_relaxed, engine=wf.grace
)
wf.neb.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)

wf.neb_plot = PlotNEBPath(
    path_energies=wf.neb.outputs.path_energies,
    barrier=wf.neb.outputs.barrier,
)
