import os
import sys
import tempfile
import unittest
from pathlib import Path

from ase.build import bulk

from pyiron_nodes.atomistic.engine.lammps import (
    CreateLammpsStaticInput,
    CreateLammpsStructure,
    ParseLammpsOutput,
    RunLammpsCalculation,
)
from pyiron_nodes.atomistic.structure.container_new import (
    AddPristine,
    CreateDefectFromIds,
    GetVoronoiInterstitialSites,
)

AL_POTENTIAL = "1999--Mishin-Y--Al--LAMMPS--ipr1"
RESOURCE_PATH = os.environ.get(
    "IPRPY_RESOURCE_PATH",
    str(Path(sys.executable).parent.parent / "share" / "iprpy"),
)


def static_energy(atoms, working_directory, tag):
    """Run a single-point LAMMPS static calculation, mirroring
    Workflows/lammps_static_basic.py, and return the potential energy (eV)."""
    io_bundle = CreateLammpsStructure._original_func(
        structure=atoms,
        potential=AL_POTENTIAL,
        working_directory=os.path.join(working_directory, tag),
        resource_path=RESOURCE_PATH,
    )
    io_bundle = CreateLammpsStaticInput._original_func(io_bundle=io_bundle)
    io_bundle, _ = RunLammpsCalculation._original_func(io_bundle=io_bundle)
    out = ParseLammpsOutput._original_func(io_bundle=io_bundle)
    return out.energy


class TestPointDefectsMetalsWithLammps(unittest.TestCase):
    """
    Integration test combining the container_new defect-creation nodes
    (as demonstrated in Workflows/point_defects_metals.py) with a real,
    fast LAMMPS static energy calculation (as in
    Workflows/lammps_static_basic.py), rather than the slower GRACE
    ML-potential path used in Workflows/point_defect_metals_energies.py.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()

        cls.atoms = bulk("Al", cubic=True).repeat((3, 3, 3))
        container = AddPristine._original_func(atoms=cls.atoms)

        container = CreateDefectFromIds._original_func(
            structure_container=container,
            defect_type="vacancy",
            atom_ids=[0],
        )

        unique_sites, all_sites = GetVoronoiInterstitialSites._original_func(cls.atoms)
        container = CreateDefectFromIds._original_func(
            structure_container=container,
            defect_type="interstitial",
            sublattice=all_sites,
            site_ids=[0],
            element="Al",
        )

        cls.container = container

        cls.e_pristine = static_energy(
            container._structures[0]["structure"], cls._tmp.name, "pristine"
        )
        cls.e_vacancy = static_energy(
            container._structures[1]["structure"], cls._tmp.name, "vacancy"
        )
        cls.e_interstitial = static_energy(
            container._structures[2]["structure"], cls._tmp.name, "interstitial"
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_energies_are_finite_and_negative(self):
        for e in (self.e_pristine, self.e_vacancy, self.e_interstitial):
            self.assertIsNotNone(e)
            self.assertLess(e, 0)

    def test_vacancy_formation_energy_is_physically_reasonable(self):
        # Unrelaxed (static) vacancy formation energy in Al should be a
        # small positive number, roughly 1-2 eV for the relaxed case;
        # since we only ran a static (no relaxation) calculation here,
        # just check it's positive and not absurdly large.
        e_formation = self.e_vacancy - (107 / 108) * self.e_pristine
        self.assertGreater(e_formation, 0.0)
        self.assertLess(e_formation, 10.0)

    def test_interstitial_formation_energy_is_positive(self):
        e_formation = self.e_interstitial - (109 / 108) * self.e_pristine
        self.assertGreater(e_formation, 0.0)

    def test_container_lineage_unaffected_by_energy_calculation(self):
        # Running LAMMPS on the extracted `structure` objects must not have
        # mutated the container's own bookkeeping.
        self.assertEqual(len(self.container), 3)
        self.assertEqual(
            self.container._structures[1]["operations_short"], "vacancy[0]"
        )
        self.assertEqual(
            self.container._structures[2]["operations_short"], "interstitial[Al]"
        )


if __name__ == "__main__":
    unittest.main()
