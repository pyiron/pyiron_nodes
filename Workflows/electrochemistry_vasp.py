from pyiron_nodes.atomistic.calculator.data import InputMDVASP, InputSCF
from pyiron_nodes.atomistic.engine.vasp_new import (
    CreateVaspInputResources,
    MergeVaspInput,
    RunVaspCalculation,
)
from pyiron_nodes.atomistic.structure.build import Surface
from pyiron_nodes.atomistic.structure.view import Plot3d
from pyiron_nodes.electrochemistry.add_potential.vasp import CCESetup, ParsePotential
from pyiron_nodes.electrochemistry.structure.build import add_neon_layer, add_water_film
from core import Workflow
from core import group_node

wf = Workflow("electrochemistry_vasp")

# ── build the electrochemistry cell ────────────────────────────────────────────
# orthogonal Al electrode slab with vacuum, then a water film and a Ne layer on top
wf.Surface = Surface(element="Al", size="3 4 4", vacuum=20, orthogonal=True)

wf.add_water_film = add_water_film(electrode=wf.Surface)

wf.add_neon_layer = add_neon_layer(structure=wf.add_water_film)

wf.Plot3d = Plot3d(structure=wf.add_neon_layer)

# ── VASP input: the CCE plugin runs as constant-potential MD ────────────────────
wf.scf = InputSCF(kpoints="1 1 1", smearing_type="gaussian")

wf.md = InputMDVASP(temperature=300.0, n_ionic_steps=100, time_step=1.0)

wf.calc = MergeVaspInput(scf=wf.scf, md=wf.md)

# write POSCAR / INCAR / POTCAR / KPOINTS for the full cell
wf.CreateVaspInputResources = CreateVaspInputResources(
    structure=wf.add_neon_layer,
    calc=wf.calc,
    working_directory="./electrochemistry_vasp_run",
)

# ── add the constant-potential (Ne-CCE) plugin on top of the base input ─────────
# `electrode` is the bare Al slab; `potential` is the target voltage in volts
wf.CCESetup = CCESetup(
    io_bundle=wf.CreateVaspInputResources,
    electrode=wf.Surface,
    potential=0.0,
)

wf.RunVaspCalculation = RunVaspCalculation(io_bundle=wf.CCESetup, debug=False)

# read the electrode charge (Q.dat), potential (phi.dat) and the planar-averaged
# electrostatic potential (el_pot_z.dat) written by the plugin
wf.ParsePotential = ParsePotential(
    io_bundle=wf.RunVaspCalculation.outputs.io_bundle
)
