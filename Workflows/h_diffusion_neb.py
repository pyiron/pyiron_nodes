from pyiron_nodes.atomistic.diffusion import AddInterstitialH, PlotNEBPath, RunNEB
from pyiron_nodes.atomistic.structure.view import Animate
from pyiron_nodes.dpg2026.atomistic.calculator.optimize import GenericOptimizerSettings, Relax
from core import Workflow
from core import group_node

# ── Group node factories ─────────────────────────────

@group_node("structure")
def Bulk(name, cubic=False, repeat_scalar=1):
    from pyiron_nodes.atomistic.structure.build import Bulk
    from pyiron_nodes.dpg2026.atomistic.structure.transform import Repeat
    from core import Workflow
    
    inner_wf = Workflow("Bulk")
    inner_wf.al_unit = Bulk(name=name, cubic=cubic)
    inner_wf.al_supercell = Repeat(structure=inner_wf.al_unit, repeat_scalar=repeat_scalar)
    return inner_wf.al_supercell

@group_node("engine")
def LammpsEngine(structure, index):
    from pyiron_nodes.atomistic.diffusion import LammpsAseEngine
    from pyiron_nodes.atomistic.engine.lammps import ListPotentials
    from pyiron_nodes.controls import pick_element
    from core import Workflow
    
    inner_wf = Workflow("LammpsEngine")
    inner_wf.list_potentials = ListPotentials(structure=structure)
    inner_wf.potential = pick_element(lst=inner_wf.list_potentials, index=index)
    inner_wf.lammps_engine = LammpsAseEngine(potential=inner_wf.potential)
    return inner_wf.lammps_engine

wf = Workflow("h_diffusion_neb")

wf.Bulk = Bulk(name='Al', cubic=True, repeat_scalar=2)

wf.opt_settings = GenericOptimizerSettings(max_steps=300, force_tolerance=0.02)

wf.h_initial = AddInterstitialH(structure=wf.Bulk, frac_pos=[0.5, 0.0, 0.0], repeat_scalar=2)

wf.h_final = AddInterstitialH(structure=wf.Bulk, frac_pos=[0.0, 0.0, 0.5], repeat_scalar=2)

wf.LammpsEngine = LammpsEngine(structure=wf.h_initial, index=0)

wf.initial_relaxed = Relax(structure=wf.h_initial, engine=wf.LammpsEngine, opt_parameters=wf.opt_settings, opt_mode='internal', store=True)

wf.final_relaxed = Relax(structure=wf.h_final, engine=wf.LammpsEngine, opt_parameters=wf.opt_settings, opt_mode='internal', store=True)

wf.neb = RunNEB(initial_state=wf.initial_relaxed, final_state=wf.final_relaxed, engine=wf.LammpsEngine, store=True)

wf.neb_plot = PlotNEBPath(path_energies=wf.neb.outputs.path_energies, barrier=wf.neb.outputs.barrier, store=True)

wf.Animate = Animate(trajectory=wf.neb.outputs.trajectory)
