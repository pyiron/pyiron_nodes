import os
import sys
import tempfile
import unittest
from pathlib import Path

from core import Workflow

from pyiron_nodes.atomistic.calculator.data import InputCalcMinimize
from pyiron_nodes.atomistic.engine.lammps import (
    CreateLammpsMinimizeInput,
    CreateLammpsStaticInput,
    CreateLammpsStructure,
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


if __name__ == "__main__":
    unittest.main()
