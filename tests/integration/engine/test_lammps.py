import os
import sys
import tempfile
import unittest
from pathlib import Path

from core import Workflow

from pyiron_nodes.atomistic.calculator.data import InputCalcMD, InputCalcMinimize
from pyiron_nodes.atomistic.engine.lammps import (
    CreateLammpsMDInput,
    CreateLammpsMinimizeInput,
    CreateLammpsStaticInput,
    CreateLammpsStructure,
    LammpsIOBundle,
    ParseLammpsOutput,
    RunLammpsCalculation,
)
from pyiron_nodes.atomistic.structure.build import Bulk

AL_POTENTIAL = "1999--Mishin-Y--Al--LAMMPS--ipr1"
RESOURCE_PATH = os.environ.get(
    "IPRPY_RESOURCE_PATH",
    str(Path(sys.executable).parent.parent / "share" / "iprpy"),
)
N_ATOMS = 4  # FCC Al cubic unit cell


class TestLammpsStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        wf = Workflow("test_static")
        wf.Bulk = Bulk(name="Al", cubic=True)
        wf.CreateLammpsStructure = CreateLammpsStructure(
            structure=wf.Bulk,
            potential=AL_POTENTIAL,
            working_directory=os.path.join(cls._tmp.name, "static"),
            resource_path=RESOURCE_PATH,
        )
        wf.CreateLammpsStaticInput = CreateLammpsStaticInput(
            io_bundle=wf.CreateLammpsStructure
        )
        wf.RunLammpsCalculation = RunLammpsCalculation(
            io_bundle=wf.CreateLammpsStaticInput, debug=False
        )
        wf.ParseLammpsOutput = ParseLammpsOutput(
            io_bundle=wf.RunLammpsCalculation.outputs.io_bundle
        )
        wf.run()
        cls.wf = wf

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_structure(self):
        self.assertEqual(len(self.wf.Bulk.outputs.structure.value), N_ATOMS)

    def test_bundle_mode(self):
        bundle = self.wf.CreateLammpsStaticInput.outputs.io_bundle.value
        self.assertEqual(bundle.mode, "static")
        self.assertNotEqual(bundle.lammps_input_string, "")

    def test_lammps_ran(self):
        self.assertIsNotNone(self.wf.RunLammpsCalculation.outputs.io_bundle.value)

    def test_energy(self):
        out = self.wf.ParseLammpsOutput.outputs.out.value
        self.assertIsNotNone(out.energy)
        self.assertLess(out.energy, 0)

    def test_forces(self):
        out = self.wf.ParseLammpsOutput.outputs.out.value
        self.assertIsNotNone(out.force)
        self.assertEqual(out.force.shape, (N_ATOMS, 3))

    def test_structure_preserved(self):
        out = self.wf.ParseLammpsOutput.outputs.out.value
        self.assertIsNotNone(out.structure)
        self.assertEqual(len(out.structure), N_ATOMS)


class TestLammpsMinimize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        wf = Workflow("test_minimize")
        wf.Bulk = Bulk(name="Al", cubic=True)
        wf.InputCalcMinimize = InputCalcMinimize()
        wf.CreateLammpsStructure = CreateLammpsStructure(
            structure=wf.Bulk,
            potential=AL_POTENTIAL,
            working_directory=os.path.join(cls._tmp.name, "minimize"),
            resource_path=RESOURCE_PATH,
        )
        wf.CreateLammpsMinimizeInput = CreateLammpsMinimizeInput(
            io_bundle=wf.CreateLammpsStructure,
            calc_dataclass=wf.InputCalcMinimize,
        )
        wf.RunLammpsCalculation = RunLammpsCalculation(
            io_bundle=wf.CreateLammpsMinimizeInput, debug=False
        )
        wf.ParseLammpsOutput = ParseLammpsOutput(
            io_bundle=wf.RunLammpsCalculation.outputs.io_bundle
        )
        wf.run()
        cls.wf = wf

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_bundle_mode(self):
        bundle = self.wf.CreateLammpsMinimizeInput.outputs.io_bundle.value
        self.assertEqual(bundle.mode, "minimize")
        self.assertNotEqual(bundle.lammps_input_string, "")

    def test_energy_decreases(self):
        out = self.wf.ParseLammpsOutput.outputs.out.value
        self.assertIsNotNone(out.initial.energy)
        self.assertIsNotNone(out.final.energy)
        self.assertLessEqual(out.final.energy, out.initial.energy)

    def test_final_structure(self):
        out = self.wf.ParseLammpsOutput.outputs.out.value
        self.assertIsNotNone(out.final.structure)
        self.assertEqual(len(out.final.structure), N_ATOMS)

    def test_iterations(self):
        out = self.wf.ParseLammpsOutput.outputs.out.value
        self.assertGreater(out.iter_steps, 0)


class TestLammpsMD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        wf = Workflow("test_md")
        wf.Bulk = Bulk(name="Al", cubic=True)
        wf.InputCalcMD = InputCalcMD(n_ionic_steps=20, n_print=10)
        wf.CreateLammpsStructure = CreateLammpsStructure(
            structure=wf.Bulk,
            potential=AL_POTENTIAL,
            working_directory=os.path.join(cls._tmp.name, "md"),
            resource_path=RESOURCE_PATH,
        )
        wf.CreateLammpsMDInput = CreateLammpsMDInput(
            io_bundle=wf.CreateLammpsStructure,
            calc_dataclass=wf.InputCalcMD,
        )
        wf.RunLammpsCalculation = RunLammpsCalculation(
            io_bundle=wf.CreateLammpsMDInput, debug=False
        )
        wf.ParseLammpsOutput = ParseLammpsOutput(
            io_bundle=wf.RunLammpsCalculation.outputs.io_bundle
        )
        wf.run()
        cls.wf = wf

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_bundle_mode(self):
        bundle = self.wf.CreateLammpsMDInput.outputs.io_bundle.value
        self.assertEqual(bundle.mode, "md")
        self.assertNotEqual(bundle.lammps_input_string, "")

    def test_energies(self):
        out = self.wf.ParseLammpsOutput.outputs.out.value
        self.assertIsNotNone(out.energies_pot)
        self.assertGreater(len(out.energies_pot), 0)

    def test_forces_shape(self):
        out = self.wf.ParseLammpsOutput.outputs.out.value
        self.assertIsNotNone(out.forces)
        self.assertEqual(out.forces.shape[1], N_ATOMS)
        self.assertEqual(out.forces.shape[2], 3)

    def test_temperatures(self):
        out = self.wf.ParseLammpsOutput.outputs.out.value
        self.assertIsNotNone(out.temperatures)
        self.assertGreater(len(out.temperatures), 0)


class TestLammpsRestart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        restart_path = os.path.join(cls._tmp.name, "md_write", "restart.lammps")

        # Step 1: write restart file
        wf1 = Workflow("test_write_restart")
        wf1.Bulk = Bulk(name="Al", cubic=True)
        wf1.InputCalcMD = InputCalcMD(n_ionic_steps=20, n_print=10)
        wf1.CreateLammpsStructure = CreateLammpsStructure(
            structure=wf1.Bulk,
            potential=AL_POTENTIAL,
            working_directory=os.path.join(cls._tmp.name, "md_write"),
            resource_path=RESOURCE_PATH,
        )
        wf1.CreateLammpsMDInput = CreateLammpsMDInput(
            io_bundle=wf1.CreateLammpsStructure,
            calc_dataclass=wf1.InputCalcMD,
            write_restart_filename=restart_path,
        )
        wf1.RunLammpsCalculation = RunLammpsCalculation(
            io_bundle=wf1.CreateLammpsMDInput, debug=False
        )
        wf1.run()
        cls.restart_path = restart_path

        # Step 2: read restart — debug=True so we just test input generation,
        # not LAMMPS execution (which has restart/dump conflicts)
        wf2 = Workflow("test_read_restart")
        wf2.Bulk = Bulk(name="Al", cubic=True)
        wf2.InputCalcMD = InputCalcMD(n_ionic_steps=10, n_print=10)
        wf2.CreateLammpsStructure = CreateLammpsStructure(
            structure=wf2.Bulk,
            potential=AL_POTENTIAL,
            working_directory=os.path.join(cls._tmp.name, "md_read"),
            resource_path=RESOURCE_PATH,
        )
        wf2.CreateLammpsMDInput = CreateLammpsMDInput(
            io_bundle=wf2.CreateLammpsStructure,
            calc_dataclass=wf2.InputCalcMD,
            read_restart_filename=restart_path,
        )
        wf2.RunLammpsCalculation = RunLammpsCalculation(
            io_bundle=wf2.CreateLammpsMDInput, debug=True
        )
        wf2.run()
        cls.wf2 = wf2

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_restart_file_written(self):
        self.assertTrue(os.path.exists(self.restart_path))

    def test_read_restart_input_generated(self):
        bundle = self.wf2.CreateLammpsMDInput.outputs.io_bundle.value
        self.assertIn("read_restart", bundle.lammps_input_string)
        self.assertIn("reset_timestep", bundle.lammps_input_string)


class TestRunLammpsError(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_invalid_input_raises_runtime_error(self):
        bundle = LammpsIOBundle(
            structure=Bulk._original_func(name="Al", cubic=True),
            potential=AL_POTENTIAL,
            working_directory=os.path.join(self._tmp.name, "error"),
            lammps_input_string="this is not valid lammps input\n",
            lammps_structure_string="nothing\n",
        )
        with self.assertRaises(RuntimeError):
            RunLammpsCalculation._original_func(io_bundle=bundle, debug=False)


if __name__ == "__main__":
    unittest.main()
