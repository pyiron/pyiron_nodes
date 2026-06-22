from pyiron_nodes.atomistic.diffusion import AddInterstitialH, PlotNEBPath, RunNEB
from pyiron_nodes.atomistic.engine.ase import EMT
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.view import Animate
from pyiron_nodes.dpg2026.atomistic.calculator.optimize import GenericOptimizerSettings, Relax
from pyiron_nodes.dpg2026.atomistic.structure.transform import Repeat
from core import Workflow
from core import group_node

wf = Workflow("h_diffusion_EMT")

wf.EMT = EMT()

wf.al_unit = Bulk(name='Al', cubic=True)

wf.opt_settings = GenericOptimizerSettings(max_steps=300, force_tolerance=0.02)

wf.al_supercell = Repeat(structure=wf.al_unit, repeat_scalar=2)

wf.h_initial = AddInterstitialH(structure=wf.al_supercell, frac_pos=[0.5, 0.0, 0.0], repeat_scalar=2)

wf.h_final = AddInterstitialH(structure=wf.al_supercell, frac_pos=[0.0, 0.0, 0.5], repeat_scalar=2)

wf.initial_relaxed = Relax(structure=wf.h_initial, engine=wf.EMT, opt_parameters=wf.opt_settings, opt_mode='internal', store=True)

wf.final_relaxed = Relax(structure=wf.h_final, engine=wf.EMT, opt_parameters=wf.opt_settings, opt_mode='internal', store=True)

wf.neb = RunNEB(initial_state=wf.initial_relaxed, final_state=wf.final_relaxed, engine=wf.EMT, n_images=11, store=True)

wf.neb_plot = PlotNEBPath(path_energies=wf.neb.outputs.path_energies, barrier=wf.neb.outputs.barrier)

wf.Animate = Animate(trajectory=wf.neb.outputs.trajectory)
