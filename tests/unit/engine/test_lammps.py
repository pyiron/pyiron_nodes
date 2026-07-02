import os
import sys
import unittest
from pathlib import Path

from ase.build import bulk

from pyiron_nodes.atomistic.engine.lammps import ListPotentials

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


if __name__ == "__main__":
    unittest.main()
