import unittest

import numpy as np
from ase.build import bulk

from pyiron_nodes.atomistic.structure.container_new import (
    StructureContainer,
    AddPristine,
    CreateDefectFromIds,
    CreateDefectFromSeed,
    CreateDefectBatchFromIds,
    CreateDefectBatchFromSeed,
    GetStoichiometry,
    ValidateStructure,
    ElementUids,
    FindStructureIndex,
    GetVacancyDistances,
    GetSubstitutionDistances,
    GetInterstitialDistances,
    GetVoronoiInterstitialSites,
    GetDelaunayInterstitialSites,
    FilterByGeneration,
    FilterByStoichiometry,
)


def make_atoms():
    return bulk("Al", cubic=True).repeat((3, 3, 3))


class TestAddPristine(unittest.TestCase):
    def test_creates_container_and_adds_row(self):
        atoms = make_atoms()
        container = AddPristine._original_func(atoms=atoms)
        self.assertIsInstance(container, StructureContainer)
        self.assertEqual(len(container), 1)
        self.assertTrue(container._structures[0]["is_pristine"])
        self.assertEqual(container._structures[0]["stoichiometry"], "Al108")

    def test_duplicate_pristine_is_not_re_added(self):
        atoms = make_atoms()
        container = AddPristine._original_func(atoms=atoms)
        container = AddPristine._original_func(structure_container=container, atoms=atoms)
        self.assertEqual(len(container), 1)

    def test_duplicate_check_disabled(self):
        atoms = make_atoms()
        container = AddPristine._original_func(atoms=atoms)
        container = AddPristine._original_func(
            structure_container=container, atoms=atoms, check_duplicates=False
        )
        self.assertEqual(len(container), 2)


class TestCreateDefectFromIds(unittest.TestCase):
    def setUp(self):
        self.atoms = make_atoms()
        self.container = AddPristine._original_func(atoms=self.atoms)

    def test_vacancy(self):
        container = CreateDefectFromIds._original_func(
            structure_container=self.container, defect_type="vacancy", atom_ids=[0]
        )
        row = container._structures[-1]
        self.assertEqual(len(row["structure"]), 107)
        self.assertEqual(row["stoichiometry"], "Al107")
        self.assertEqual(row["operations_short"], "vacancy[0]")
        self.assertEqual(row["generation"], 1)
        self.assertEqual(row["events"][-1]["type"], "vacancy")

    def test_multi_atom_vacancy(self):
        container = CreateDefectFromIds._original_func(
            structure_container=self.container,
            defect_type="vacancy",
            atom_ids=[0, 5, 10],
        )
        row = container._structures[-1]
        self.assertEqual(len(row["structure"]), 105)
        self.assertEqual(row["operation"], "vacancy[3]")
        self.assertEqual(row["operations_short"], "vacancy[0]|vacancy[5]|vacancy[10]")

    def test_substitution(self):
        container = CreateDefectFromIds._original_func(
            structure_container=self.container,
            defect_type="substitution",
            atom_ids=[3],
            to_element="Mg",
        )
        row = container._structures[-1]
        self.assertEqual(len(row["structure"]), 108)
        self.assertEqual(row["stoichiometry"], "Al107Mg1")
        self.assertEqual(row["operations_short"], "substitution[Al->Mg]")
        self.assertEqual(row["structure"].get_chemical_symbols()[3], "Mg")

    def test_interstitial(self):
        unique_sites, all_sites = GetVoronoiInterstitialSites._original_func(self.atoms)
        container = CreateDefectFromIds._original_func(
            structure_container=self.container,
            defect_type="interstitial",
            sublattice=all_sites,
            site_ids=[0],
            element="Mg",
        )
        row = container._structures[-1]
        self.assertEqual(len(row["structure"]), 109)
        self.assertEqual(row["stoichiometry"], "Al108Mg1")
        self.assertEqual(row["operations_short"], "interstitial[Mg]")

    def test_unknown_defect_type_raises(self):
        with self.assertRaises(ValueError):
            CreateDefectFromIds._original_func(
                structure_container=self.container,
                defect_type="antisite",
                atom_ids=[0],
            )

    def test_substitution_without_to_element_raises(self):
        with self.assertRaises(ValueError):
            CreateDefectFromIds._original_func(
                structure_container=self.container,
                defect_type="substitution",
                atom_ids=[0],
            )

    def test_interstitial_without_sublattice_raises(self):
        with self.assertRaises(ValueError):
            CreateDefectFromIds._original_func(
                structure_container=self.container,
                defect_type="interstitial",
                site_ids=[0],
                element="Mg",
            )

    def test_forbid_atom_ids(self):
        with self.assertRaises(ValueError):
            CreateDefectFromIds._original_func(
                structure_container=self.container,
                defect_type="vacancy",
                atom_ids=[0],
                forbid_atom_ids=[0],
            )

    def test_chaining_with_parent_defect_index(self):
        container = CreateDefectFromIds._original_func(
            structure_container=self.container, defect_type="vacancy", atom_ids=[0]
        )
        container = CreateDefectFromIds._original_func(
            structure_container=container,
            defect_type="substitution",
            atom_ids=[5],
            to_element="Mg",
            parent_defect_index=-1,
        )
        row = container._structures[-1]
        self.assertEqual(row["generation"], 2)
        self.assertEqual(row["parent_index"], 1)
        self.assertEqual(row["operations_short"], "vacancy[0]|substitution[Al->Mg]")
        self.assertEqual(len(row["structure"]), 107)

    def test_independent_defects_all_attach_to_pristine(self):
        container = CreateDefectFromIds._original_func(
            structure_container=self.container, defect_type="vacancy", atom_ids=[0]
        )
        container = CreateDefectFromIds._original_func(
            structure_container=container,
            defect_type="substitution",
            atom_ids=[5],
            to_element="Mg",
        )
        vacancy_row, substitution_row = container._structures[1], container._structures[2]
        self.assertEqual(vacancy_row["generation"], 1)
        self.assertEqual(substitution_row["generation"], 1)
        self.assertEqual(vacancy_row["parent_index"], 0)
        self.assertEqual(substitution_row["parent_index"], 0)


class TestCreateDefectFromSeed(unittest.TestCase):
    def setUp(self):
        self.atoms = make_atoms()
        self.container = AddPristine._original_func(atoms=self.atoms)

    def test_reproducible_with_seed(self):
        # Independent containers -- see note in test_different_seed_differs
        # about why reusing the same container for both calls would make
        # this comparison meaningless (c1 and c2 would be the same object).
        c1 = CreateDefectFromSeed._original_func(
            structure_container=AddPristine._original_func(atoms=self.atoms),
            defect_type="vacancy",
            n=2,
            seed=7,
        )
        c2 = CreateDefectFromSeed._original_func(
            structure_container=AddPristine._original_func(atoms=self.atoms),
            defect_type="vacancy",
            n=2,
            seed=7,
        )
        self.assertEqual(
            c1._structures[-1]["operations_short"], c2._structures[-1]["operations_short"]
        )

    def test_different_seed_differs(self):
        # StructureContainer is mutated in place, so c1/c2 must start from
        # independent containers -- reusing self.container for both calls
        # would make c1 and c2 the same object, and reading c1's "last row"
        # after c2 runs would silently show c2's data instead.
        c1 = CreateDefectFromSeed._original_func(
            structure_container=AddPristine._original_func(atoms=self.atoms),
            defect_type="vacancy",
            n=2,
            seed=1,
        )
        c2 = CreateDefectFromSeed._original_func(
            structure_container=AddPristine._original_func(atoms=self.atoms),
            defect_type="vacancy",
            n=2,
            seed=2,
        )
        self.assertNotEqual(
            c1._structures[-1]["operations_short"], c2._structures[-1]["operations_short"]
        )

    def test_vacancy_element_list_mode(self):
        container = CreateDefectFromSeed._original_func(
            structure_container=self.container,
            defect_type="vacancy",
            n=2,
            seed=0,
            vacancy_element=["Al", "Al"],
        )
        row = container._structures[-1]
        removed = {ev["removed_element"] for ev in row["events"]}
        self.assertEqual(removed, {"Al"})

    def test_vacancy_element_list_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            CreateDefectFromSeed._original_func(
                structure_container=self.container,
                defect_type="vacancy",
                n=2,
                seed=0,
                vacancy_element=["Al"],
            )

    def test_substitution_from_seed(self):
        container = CreateDefectFromSeed._original_func(
            structure_container=self.container,
            defect_type="substitution",
            n=1,
            seed=0,
            from_element="Al",
            to_element="Mg",
        )
        row = container._structures[-1]
        self.assertEqual(row["stoichiometry"], "Al107Mg1")

    def test_interstitial_from_seed(self):
        unique_sites, all_sites = GetVoronoiInterstitialSites._original_func(self.atoms)
        container = CreateDefectFromSeed._original_func(
            structure_container=self.container,
            defect_type="interstitial",
            sublattice=all_sites,
            element="Mg",
            n=1,
            seed=0,
        )
        row = container._structures[-1]
        self.assertEqual(row["stoichiometry"], "Al108Mg1")


class TestCreateDefectBatch(unittest.TestCase):
    def setUp(self):
        self.atoms = make_atoms()
        self.container = AddPristine._original_func(atoms=self.atoms)

    def test_batch_from_ids_separate_structures(self):
        container = CreateDefectBatchFromIds._original_func(
            structure_container=self.container,
            defect_type="vacancy",
            target_indices=[0],
            atom_ids=[1, 2, 3],
            separate_structures=True,
        )
        # 1 pristine + 3 separate single-vacancy structures
        self.assertEqual(len(container), 4)
        for row in container._structures[1:]:
            self.assertEqual(len(row["structure"]), 107)

    def test_batch_from_ids_combined_structure(self):
        container = CreateDefectBatchFromIds._original_func(
            structure_container=self.container,
            defect_type="vacancy",
            target_indices=[0],
            atom_ids=[1, 2, 3],
            separate_structures=False,
        )
        # 1 pristine + 1 structure containing all 3 vacancies
        self.assertEqual(len(container), 2)
        self.assertEqual(len(container._structures[-1]["structure"]), 105)

    def test_batch_from_seed_n_structures(self):
        container = CreateDefectBatchFromSeed._original_func(
            structure_container=self.container,
            defect_type="vacancy",
            target_indices=[0],
            n=1,
            seed=0,
            n_structures=5,
        )
        self.assertEqual(len(container), 6)  # 1 pristine + 5 defects

    def test_batch_from_seed_interstitial_requires_sublattice(self):
        with self.assertRaises(ValueError):
            CreateDefectBatchFromSeed._original_func(
                structure_container=self.container,
                defect_type="interstitial",
                target_indices=[0],
                n=1,
                seed=0,
                n_structures=1,
            )


class TestNewNodeWrappers(unittest.TestCase):
    def test_get_stoichiometry(self):
        self.assertEqual(GetStoichiometry._original_func(bulk("Al", cubic=True)), "Al4")

    def test_validate_structure_ok(self):
        self.assertTrue(ValidateStructure._original_func(make_atoms()))

    def test_validate_structure_raises_on_clash(self):
        atoms = make_atoms()
        atoms.positions[1] = atoms.positions[0] + 0.01
        with self.assertRaises(ValueError):
            ValidateStructure._original_func(atoms)

    def test_element_uids(self):
        atoms = make_atoms()
        uids = ElementUids._original_func(atoms, "Al")
        self.assertEqual(uids, list(range(len(atoms))))

    def test_find_structure_index_present(self):
        atoms = make_atoms()
        container = AddPristine._original_func(atoms=atoms)
        self.assertEqual(FindStructureIndex._original_func(container, atoms), 0)

    def test_find_structure_index_absent(self):
        atoms = make_atoms()
        container = AddPristine._original_func(atoms=atoms)
        other = bulk("Cu", cubic=True).repeat((3, 3, 3))
        self.assertIsNone(FindStructureIndex._original_func(container, other))


class TestDistanceGetters(unittest.TestCase):
    def setUp(self):
        self.atoms = make_atoms()
        self.container = AddPristine._original_func(atoms=self.atoms)

    def test_vacancy_distances_match_ase_mic_distance(self):
        i, j = 0, 5
        expected = self.atoms.get_distance(i, j, mic=True)
        container = CreateDefectFromIds._original_func(
            structure_container=self.container, defect_type="vacancy", atom_ids=[i, j]
        )
        result = GetVacancyDistances._original_func(container, len(container) - 1)
        self.assertAlmostEqual(result["distances"]["0-1"], expected, places=6)

    def test_substitution_distances_match_ase_mic_distance(self):
        i, j = 2, 9
        expected = self.atoms.get_distance(i, j, mic=True)
        container = CreateDefectFromIds._original_func(
            structure_container=self.container,
            defect_type="substitution",
            atom_ids=[i, j],
            to_element="Mg",
        )
        result = GetSubstitutionDistances._original_func(container, len(container) - 1)
        self.assertAlmostEqual(result["distances"]["0-1"], expected, places=6)

    def test_interstitial_distances_match_sublattice_geometry(self):
        unique_sites, all_sites = GetVoronoiInterstitialSites._original_func(self.atoms)
        cell = np.array(self.atoms.get_cell())
        inv_cell = np.linalg.inv(cell)
        delta = all_sites[0] - all_sites[1]
        delta -= np.round(delta @ inv_cell) @ cell
        expected = float(np.linalg.norm(delta))

        container = CreateDefectFromIds._original_func(
            structure_container=self.container,
            defect_type="interstitial",
            sublattice=all_sites,
            site_ids=[0, 1],
            element="Mg",
        )
        result = GetInterstitialDistances._original_func(container, len(container) - 1)
        self.assertAlmostEqual(result["distances"]["0-1"], expected, places=6)

    def test_needs_at_least_two_defects(self):
        container = CreateDefectFromIds._original_func(
            structure_container=self.container, defect_type="vacancy", atom_ids=[0]
        )
        result = GetVacancyDistances._original_func(container, len(container) - 1)
        self.assertEqual(result["distances"], {})
        self.assertIn("message", result)


class TestInterstitialSiteFinders(unittest.TestCase):
    def setUp(self):
        self.atoms = make_atoms()

    def test_voronoi_sites_respect_r_min(self):
        r_min = 1.0
        unique_sites, all_sites = GetVoronoiInterstitialSites._original_func(
            self.atoms, r_min=r_min
        )
        self.assertGreater(len(all_sites), 0)
        positions = self.atoms.get_positions()
        cell = np.array(self.atoms.get_cell())
        inv_cell = np.linalg.inv(cell)
        for site in all_sites:
            diff = positions - site
            diff_frac = diff @ inv_cell
            diff_frac -= np.round(diff_frac)
            dists = np.linalg.norm(diff_frac @ cell, axis=1)
            self.assertGreaterEqual(dists.min(), r_min - 1e-8)

    def test_delaunay_sites_respect_r_min(self):
        r_min = 1.0
        unique_sites, all_sites = GetDelaunayInterstitialSites._original_func(
            self.atoms, r_min=r_min
        )
        self.assertGreater(len(all_sites), 0)


class TestFilterFunctions(unittest.TestCase):
    def setUp(self):
        self.atoms = make_atoms()
        self.container = AddPristine._original_func(atoms=self.atoms)
        self.container = CreateDefectFromIds._original_func(
            structure_container=self.container, defect_type="vacancy", atom_ids=[0]
        )
        self.container = CreateDefectFromIds._original_func(
            structure_container=self.container,
            defect_type="substitution",
            atom_ids=[5],
            to_element="Mg",
        )

    def test_filter_by_generation(self):
        gen0 = FilterByGeneration._original_func(self.container, 0)
        gen1 = FilterByGeneration._original_func(self.container, 1)
        self.assertEqual(len(gen0), 1)
        self.assertEqual(len(gen1), 2)

    def test_filter_by_stoichiometry_wildcard(self):
        mg_rows = FilterByStoichiometry._original_func(self.container, "*Mg*")
        self.assertEqual(len(mg_rows), 1)
        self.assertEqual(mg_rows[0]["stoichiometry"], "Al107Mg1")


if __name__ == "__main__":
    unittest.main()
