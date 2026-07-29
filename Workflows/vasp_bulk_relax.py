from pyiron_nodes.atomistic.calculator.data import (
    InputMinimizationVASP,
    InputSCF,
)
from pyiron_nodes.atomistic.engine.vasp_new import (
    CreateVaspInputResources,
    MergeVaspInput,
    ParseVaspOutput,
    RunVaspCalculation,
)
from pyiron_nodes.atomistic.structure.build import Bulk
from core import Workflow
from core import group_node

wf = Workflow("vasp_bulk_relax")

# bcc Fe unit cell — the usual starting point for a bulk calculation
wf.Bulk = Bulk(name="Fe", crystalstructure="bcc", a=2.89, cubic=True)

# required SCF settings (kpoints is mandatory); a metal wants MP smearing
wf.scf = InputSCF(kpoints="6 6 6", smearing_type="methfessel-paxton")

# optional ionic relaxation — drop this port for a plain single-point (static) run
wf.minimization = InputMinimizationVASP(max_ionic_steps=50)

# combine the input pieces into one VaspInput
wf.calc = MergeVaspInput(scf=wf.scf, minimization=wf.minimization)

# write POSCAR / INCAR / POTCAR / KPOINTS into the working directory
wf.CreateVaspInputResources = CreateVaspInputResources(
    structure=wf.Bulk,
    calc=wf.calc,
    working_directory="./vasp_bulk_relax_run",
)

# run VASP (set debug=True to skip the launch and just return the working dir)
wf.RunVaspCalculation = RunVaspCalculation(
    io_bundle=wf.CreateVaspInputResources, debug=False
)

# parse the outputs; `out` is an OutputCalcMinimize (initial + final + convergence)
wf.ParseVaspOutput = ParseVaspOutput(
    io_bundle=wf.RunVaspCalculation.outputs.io_bundle
)
