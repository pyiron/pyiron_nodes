from pyiron_nodes.atomistic.diffusion import AddInterstitialH, LammpsAseEngine, PlotNEBPath, RunNEB
from pyiron_nodes.atomistic.engine.lammps import ListPotentials
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.controls import pick_element
from pyiron_nodes.dpg2026.atomistic.calculator.optimize import GenericOptimizerSettings, Relax
from pyiron_nodes.dpg2026.atomistic.structure.transform import Repeat
from core import Workflow
from core import group_node

wf = Workflow("h_diffusion_lammps_neb2")

wf.al_unit = Bulk(name='Al', cubic=True)

wf.opt_settings = GenericOptimizerSettings(max_steps=300, force_tolerance=0.02)

wf.al_supercell = Repeat(structure=wf.al_unit, repeat_scalar=2)

wf.h_initial = AddInterstitialH(structure=wf.al_supercell, frac_pos=[0.5, 0.0, 0.0], repeat_scalar=2)

wf.h_final = AddInterstitialH(structure=wf.al_supercell, frac_pos=[0.0, 0.0, 0.5], repeat_scalar=2)

wf.list_potentials = ListPotentials(structure=wf.h_initial)

wf.potential = pick_element(lst=wf.list_potentials, index=0)

wf.lammps_engine = LammpsAseEngine(potential=wf.potential)

wf.initial_relaxed = Relax(structure=wf.h_initial, engine=wf.lammps_engine, opt_parameters=wf.opt_settings, opt_mode='internal')
wf.initial_relaxed.inputs.add("store", port_type=bool, default=False, value=True, has_explicit_default=True)

wf.final_relaxed = Relax(structure=wf.h_final, engine=wf.lammps_engine, opt_parameters=wf.opt_settings, opt_mode='internal')
wf.final_relaxed.inputs.add("store", port_type=bool, default=False, value=True, has_explicit_default=True)

wf.neb = RunNEB(initial_state=wf.initial_relaxed, final_state=wf.final_relaxed, engine=wf.lammps_engine)
wf.neb.inputs.add("store", port_type=bool, default=False, value=True, has_explicit_default=True)

wf.neb_plot = PlotNEBPath(path_energies=wf.neb.outputs.path_energies, barrier=wf.neb.outputs.barrier)
wf.neb_plot.inputs.add("store", port_type=bool, default=False, value=True, has_explicit_default=True)
