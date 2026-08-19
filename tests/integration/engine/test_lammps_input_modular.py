import os
import sys
import tempfile
import unittest
from pathlib import Path

from core import Workflow

from pyiron_nodes.atomistic.engine.lammps import (
    CreateLammpsStructure,
    RunLammpsCalculation,
)
from pyiron_nodes.atomistic.engine.lammps_input import (
    AssembleLammpsInput,
    LammpsDump,
    LammpsEnsemble,
    LammpsInit,
    LammpsMinimize,
    LammpsPotential,
    LammpsThermo,
    LammpsVelocity,
)
from pyiron_nodes.atomistic.structure.build import Bulk

AL_POTENTIAL = "1999--Mishin-Y--Al--LAMMPS--ipr1"
RESOURCE_PATH = os.environ.get(
    "IPRPY_RESOURCE_PATH",
    str(Path(sys.executable).parent.parent / "share" / "iprpy"),
)


class TestModularLammpsTwoStageRun(unittest.TestCase):
    """End-to-end run of the modular section-node pipeline, mirroring
    WORKFLOWS_new/modular_example_grouped.py: a minimize stage followed by a
    short MD stage assembled onto the same io_bundle and executed in a
    single RunLammpsCalculation call. Runs locally (executor=None) rather
    than through the example's SlurmExecutor, which isn't available here.
    Does not call ParseLammpsOutput: AssembleLammpsInput never sets
    io_bundle.mode (unlike the old CreateLammpsMDInput/StaticInput/
    MinimizeInput nodes), so parsing isn't wired up for this pipeline yet.
    """

    RUN_STEPS = 20
    RESTART_FILENAME = "minimize.restart"

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        wf = Workflow("test_modular_two_stage")

        wf.Bulk = Bulk(name="Al", cubic=True)
        wf.CreateLammpsStructure = CreateLammpsStructure(
            structure=wf.Bulk,
            potential=AL_POTENTIAL,
            working_directory=os.path.join(cls._tmp.name, "modular"),
            resource_path=RESOURCE_PATH,
        )
        wf.LammpsInit = LammpsInit(structure=wf.Bulk)

        # Stage 1: minimize (zero-pressure box relax, then a static run=0
        # re-evaluation), writing a restart file at the end of the stage.
        wf.LammpsPotential = LammpsPotential(
            potential=AL_POTENTIAL, resource_path=RESOURCE_PATH
        )
        wf.LammpsDump = LammpsDump()
        wf.LammpsThermo = LammpsThermo()
        wf.LammpsMinimize = LammpsMinimize(e_tol=0.001, f_tol=0.001, pressure=0.0)
        wf.AssembleStage1 = AssembleLammpsInput(
            io_bundle=wf.CreateLammpsStructure,
            init=wf.LammpsInit,
            potential=wf.LammpsPotential.outputs.potential,
            dump=wf.LammpsDump,
            thermo=wf.LammpsThermo,
            calculation=wf.LammpsMinimize,
            run=0,
            write_restart=cls.RESTART_FILENAME,
        )

        # Stage 2: short NVT production continuing from the minimized
        # structure, appended onto the same io_bundle.
        wf.LammpsEnsemble = LammpsEnsemble(temperature=300.0)
        wf.LammpsVelocity = LammpsVelocity(temperature=300.0, time_step=1.0)
        wf.AssembleStage2 = AssembleLammpsInput(
            io_bundle=wf.AssembleStage1,
            calculation=wf.LammpsEnsemble.outputs.ensemble,
            calculation_fix_ids=wf.LammpsEnsemble.outputs.fix_ids,
            velocity=wf.LammpsVelocity,
            run=cls.RUN_STEPS,
        )

        wf.RunLammpsCalculation = RunLammpsCalculation(
            io_bundle=wf.AssembleStage2, debug=False, executor=None
        )
        wf.run()
        cls.wf = wf
        cls.working_directory = (
            wf.CreateLammpsStructure.outputs.io_bundle.value.working_directory
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_assembled_input_has_both_stages_in_order(self):
        bundle = self.wf.AssembleStage2.outputs.io_bundle.value
        s = bundle.lammps_input_string
        self.assertIn("min_style", s)
        self.assertIn(f"run {self.RUN_STEPS}", s)
        self.assertLess(s.index("min_style"), s.index(f"run {self.RUN_STEPS}"))
        # The ensemble fix from stage 2 should have been unfixed by its own
        # run, leaving nothing pending on the bundle.
        self.assertEqual(bundle.lammps_pending_fix_ids, "")

    def test_lammps_ran_successfully(self):
        # RunLammpsCalculation raises RuntimeError on a nonzero exit code,
        # so setUpClass completing at all already proves the run succeeded.
        self.assertIsNotNone(self.wf.RunLammpsCalculation.outputs.io_bundle.value)

    def test_dump_and_log_files_written(self):
        dump_path = os.path.join(self.working_directory, "dump.out")
        log_path = os.path.join(self.working_directory, "log.lammps")
        self.assertTrue(os.path.exists(dump_path))
        self.assertGreater(os.path.getsize(dump_path), 0)
        self.assertTrue(os.path.exists(log_path))
        self.assertGreater(os.path.getsize(log_path), 0)

    def test_restart_file_written(self):
        restart_path = os.path.join(self.working_directory, self.RESTART_FILENAME)
        self.assertTrue(os.path.exists(restart_path))


if __name__ == "__main__":
    unittest.main()
