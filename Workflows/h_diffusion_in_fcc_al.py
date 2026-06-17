from pyiron_nodes.dpg2026.atomistic.calculator.optimize import GenericOptimizerSettings, Relax
from pyiron_nodes.dpg2026.atomistic.engine.grace import Grace
from pyiron_nodes.dpg2026.atomistic.structure.build import Bulk
from pyiron_nodes.dpg2026.atomistic.structure.transform import Repeat
from pyiron_nodes.plotting import Plot
from core import Workflow
from core import group_node
from core import as_function_node

# ── Local node definitions ──────────────────────

@as_function_node
def AddInterstitialH(structure: Atoms, frac_pos: list = None):
    import numpy as np
    if frac_pos is None:
        frac_pos = [0.25, 0.0, 0.0]
    new_atoms = structure.copy()
    cart_pos = np.dot(frac_pos, new_atoms.cell)
    new_atoms.append('H')
    new_atoms.positions[-1] = cart_pos
    return new_atoms

@as_function_node
def RunNEB(
    initial_state,
    final_state,
    engine,
    n_images: int = 7,
    fmax: float = 0.05,
    max_steps: int = 200,
    store: bool = False,
):
    """Run a fixed-endpoint NEB between two relaxed end states.

    Returns
    -------
    path_energies : np.ndarray
        Image energies relative to initial state (n_images + 2 values), eV.
    barrier : float
        Forward activation barrier (max of path_energies), eV.
    """
    from ase.mep.neb import SingleCalculatorNEB
    from ase.optimize import LBFGS
    from pyiron_nodes.atomistic.structure._atoms import to_ase
    import numpy as np

    initial_atoms = to_ase(initial_state.structure)
    final_atoms   = to_ase(final_state.structure)

    images = [initial_atoms.copy()] \
           + [initial_atoms.copy() for _ in range(n_images)] \
           + [final_atoms.copy()]

    calc = engine.calculator
    for img in images:
        img.calc = calc

    neb = SingleCalculatorNEB(images)
    neb.interpolate('idpp')

    opt = LBFGS(neb, logfile='/dev/null')
    opt.run(fmax=fmax, steps=max_steps)

    image_energies = [initial_state.energy] \
                   + [img.get_potential_energy() for img in images[1:-1]] \
                   + [final_state.energy]

    path_energies = np.array(image_energies) - image_energies[0]
    barrier = float(np.max(path_energies))
    return path_energies, barrier

wf = Workflow("h_diffusion_in_fcc_al")

wf.al_unit = Bulk(name='Al', crystalstructure='fcc', a=4.05, cubic=True)

wf.grace = Grace(model='GRACE-2L-OAM')

wf.opt_settings = GenericOptimizerSettings(max_steps=300, force_tolerance=0.02)

wf.al_supercell = Repeat(structure=wf.al_unit, repeat_scalar=2)

wf.h_initial = AddInterstitialH(structure=wf.al_supercell, frac_pos=[0.25, 0.0, 0.0])

wf.h_final = AddInterstitialH(structure=wf.al_supercell, frac_pos=[0.0, 0.25, 0.0])

wf.initial_relaxed = Relax(structure=wf.h_initial, engine=wf.grace, opt_parameters=wf.opt_settings, opt_mode='internal')
wf.initial_relaxed.inputs.add("store", port_type=bool, default=False, value=True, has_explicit_default=True)

wf.final_relaxed = Relax(structure=wf.h_final, engine=wf.grace, opt_parameters=wf.opt_settings, opt_mode='internal')
wf.final_relaxed.inputs.add("store", port_type=bool, default=False, value=True, has_explicit_default=True)

wf.neb = RunNEB(initial_state=wf.initial_relaxed, final_state=wf.final_relaxed, engine=wf.grace)
wf.neb.inputs.add("store", port_type=bool, default=False, value=True, has_explicit_default=True)

wf.Plot = Plot(y=wf.neb.outputs.path_energies)
