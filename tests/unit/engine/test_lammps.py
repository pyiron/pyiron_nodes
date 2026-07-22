import os
import sys
import unittest
from pathlib import Path

import tempfile

import pandas as pd
from ase.build import bulk, molecule

from pyiron_nodes.atomistic.engine.lammps import (
    LammpsIOBundle,
    ListPotentials,
    ParseLammpsOutput,
    RunLammpsCalculation,
    extract_charges_from_lammps_potential,
    write_lammps_data_full,
)

AL_POTENTIAL = "1999--Mishin-Y--Al--LAMMPS--ipr1"
RESOURCE_PATH = os.environ.get(
    "IPRPY_RESOURCE_PATH",
    str(Path(sys.executable).parent.parent / "share" / "iprpy"),
)


class TestListPotentials(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.potentials = ListPotentials._original_func(
            structure=bulk("Al", cubic=True),
            resource_path=RESOURCE_PATH,
        )

    def test_returns_list(self):
        self.assertIsInstance(self.potentials, list)
        self.assertGreater(len(self.potentials), 0)

    def test_contains_known_potential(self):
        self.assertIn(AL_POTENTIAL, self.potentials)

    def test_resource_path_autodiscover(self):
        potentials = ListPotentials._original_func(
            structure=bulk("Al", cubic=True),
            resource_path=None,
        )
        self.assertIsInstance(potentials, list)
        self.assertGreater(len(potentials), 0)


class TestExtractCharges(unittest.TestCase):
    def test_group_pattern(self):
        lines = [
            "group O type 1",
            "group H type 2",
            "set group O charge -0.830",
            "set group H charge 0.415",
        ]
        charges = extract_charges_from_lammps_potential(lines, specorder=["O", "H"])
        self.assertAlmostEqual(charges["O"], -0.830)
        self.assertAlmostEqual(charges["H"], 0.415)

    def test_set_type_pattern(self):
        lines = [
            "group O type 1",
            "group H type 2",
            "set type 1 charge -0.834",
            "set type 2 charge 0.417",
        ]
        charges = extract_charges_from_lammps_potential(lines, specorder=["O", "H"])
        self.assertAlmostEqual(charges["O"], -0.834)
        self.assertAlmostEqual(charges["H"], 0.417)

    def test_empty_lines(self):
        charges = extract_charges_from_lammps_potential([], specorder=["O", "H"])
        self.assertEqual(charges, {"O": 0.0, "H": 0.0})


class TestWriteLammpsDataFull(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        structure = molecule("H2O")
        structure.cell = [10, 10, 10]
        structure.pbc = True

        config_lines = [
            "group O type 1",
            "group H type 2",
            "set group O charge -0.830",
            "set group H charge 0.415",
        ]
        potential = pd.DataFrame({"Config": [config_lines], "Name": ["test_water"]})

        bond_dict = {
            "O": {
                "O-H": {
                    "cutoff": 1.2,
                    "max_bond_num": 2,
                    "neighbor_type": "H",
                },
                "H-O-H": {
                    "cutoff": 1.2,
                    "max_angle_num": 1,
                    "neighbor_type": "H",
                },
            }
        }

        cls.result = write_lammps_data_full(
            structure=structure,
            specorder=["O", "H"],
            bond_dict=bond_dict,
            potential=potential,
        )

    def test_returns_string(self):
        self.assertIsInstance(self.result, str)

    def test_contains_atoms_section(self):
        self.assertIn("Atoms", self.result)

    def test_contains_bonds_section(self):
        self.assertIn("Bonds", self.result)

    def test_contains_angles_section(self):
        self.assertIn("Angles", self.result)

    def test_atom_count(self):
        self.assertIn("3 atoms", self.result)


class TestParseLammpsOutputErrors(unittest.TestCase):
    def _make_bundle(self):
        return LammpsIOBundle(
            structure=bulk("Al", cubic=True),
            potential=AL_POTENTIAL,
        )

    def test_mode_none_raises(self):
        with self.assertRaises(ValueError):
            ParseLammpsOutput._original_func(io_bundle=self._make_bundle())

    def test_unknown_mode_raises(self):
        bundle = self._make_bundle()
        bundle.mode = "unknown"
        with self.assertRaises(ValueError):
            ParseLammpsOutput._original_func(io_bundle=bundle)


class TestRunLammpsCalculationDebug(unittest.TestCase):
    def test_debug_returns_working_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = LammpsIOBundle(
                structure=bulk("Al", cubic=True),
                potential=AL_POTENTIAL,
                working_directory=tmpdir,
                lammps_input_string="# test",
                lammps_structure_string="# test",
            )
            _, output = RunLammpsCalculation._original_func(
                io_bundle=bundle, debug=True
            )
            self.assertEqual(output, tmpdir)


if __name__ == "__main__":
    unittest.main()
