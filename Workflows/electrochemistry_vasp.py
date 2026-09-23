from core import Workflow
from pyiron_nodes.atomistic.calculator.data import InputMDVASP, InputSCF
from pyiron_nodes.atomistic.engine.vasp_new import (
    CreateVaspInputResources,
    MergeVaspInput,
    RunVaspCalculation,
)
from pyiron_nodes.atomistic.structure.build import Surface
from pyiron_nodes.atomistic.structure.view import Plot3d
from pyiron_nodes.electrochemistry.add_potential.vasp import CCESetup, ParsePotential
from pyiron_nodes.electrochemistry.structure.build import AddNeonLayer, AddWaterFilm

wf = Workflow("electrochemistry_vasp")

# ── build the electrochemistry cell ────────────────────────────────────────────
# orthogonal Al electrode slab with vacuum, then a water film and a Ne layer on top
wf.Surface = Surface(element="Al", size="3 4 4", vacuum=20, orthogonal=True)

wf.AddWaterFilm = AddWaterFilm(electrode=wf.Surface)

wf.AddNeonLayer = AddNeonLayer(structure=wf.AddWaterFilm)

wf.Plot3d = Plot3d(structure=wf.AddNeonLayer)

# ── VASP input: the CCE plugin runs as constant-potential MD ────────────────────
wf.scf = InputSCF(kpoints="1 1 1", smearing_type="gaussian")

wf.md = InputMDVASP(temperature=300.0, n_ionic_steps=100, time_step=1.0)

wf.calc = MergeVaspInput(scf=wf.scf, md=wf.md)

# ── build the constant-potential (Ne-CCE) plugin ─────────────────────────────
# `electrode` is the bare Al slab; `potential` is the target voltage in volts.
# This only renders the plugin content and the extra INCAR/POTCAR tags — it
# does not touch disk.
wf.CCESetup = CCESetup(
    structure=wf.AddNeonLayer,
    electrode=wf.Surface,
    calc=wf.calc,
    potential=0.0,
)

# write POSCAR / INCAR / POTCAR / KPOINTS for the full cell, folding in the
# CCE plugin's extra INCAR tags and Ne ZVAL override
wf.CreateVaspInputResources = CreateVaspInputResources(
    structure=wf.CCESetup.outputs.structure,
    calc=wf.CCESetup.outputs.calc,
    plugin_data=wf.CCESetup.outputs.plugin_data,
    working_directory="./electrochemistry_vasp_run",
)

wf.RunVaspCalculation = RunVaspCalculation(
    io_bundle=wf.CreateVaspInputResources, debug=False
)

# read the electrode charge (Q.dat), potential (phi.dat) and the planar-averaged
# electrostatic potential (el_pot_z.dat) written by the plugin
wf.ParsePotential = ParsePotential(io_bundle=wf.RunVaspCalculation.outputs.io_bundle)
