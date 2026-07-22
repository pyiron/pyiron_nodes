import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from ase.build import bulk

import pyiron_nodes.atomistic.structure.container_new as container_new
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
    GetSubstitutionDistancesRelaxed,
    GetInterstitialDistances,
    GetInterstitialDistancesRelaxed,
    GetVoronoiInterstitialSites,
    GetVoronoiInterstitialSitesPymatgen,
    GetDelaunayInterstitialSites,
    FilterByGeneration,
    FilterByStoichiometry,
    FilterByIndices,
    FilterByMaxGeneration,
    FilterByOperationsShort,
    FilterByOperationsContains,
    FilterByUniqueId,
    FilterByNumberOfAtoms,
    FilterByElementCount,
    FilterByParent,
    filter_by_condition,
    GetStructure,
    GetStructureTable,
    GetDefectTable,
    GetPristineTable,
    GetPristineStructures,
    GetDefectStructures,
    LatestPristineIndex,
    ResolveDefectRow,
    ResolveAnyRow,
    ensure_uids,
    next_uid,
    uid_to_index,
    validate_atoms_arrays,
    make_operations_short,
    _protected_uids_from_events,
    _resolve_parent,
    _filter_cluster_tile_interstitial_candidates,
    _deduplicate_frac,
    _extract_defect_frac_coords,
)

try:
    from pymatgen.analysis.defects.generators import (  # noqa: F401
        VoronoiInterstitialGenerator,
    )

    PYMATGEN_DEFECTS_AVAILABLE = True
except ImportError:
    PYMATGEN_DEFECTS_AVAILABLE = False


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
        container = AddPristine._original_func(
            structure_container=container, atoms=atoms
        )
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
        vacancy_row, substitution_row = (
            container._structures[1],
            container._structures[2],
        )
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
            c1._structures[-1]["operations_short"],
            c2._structures[-1]["operations_short"],
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
            c1._structures[-1]["operations_short"],
            c2._structures[-1]["operations_short"],
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


class TestRelaxedDistanceGetters(unittest.TestCase):
    def setUp(self):
        self.atoms = make_atoms()
        self.container = AddPristine._original_func(atoms=self.atoms)

    def test_substitution_distances_relaxed_matches_unrelaxed(self):
        # Feeding the (unmodified) defect structure straight back in as the
        # "relaxed" structure means the nearest-neighbour search should find
        # each substituted atom exactly at its original site, so the result
        # should match the non-relaxed getter's ase.get_distance-verified value.
        i, j = 2, 9
        expected = self.atoms.get_distance(i, j, mic=True)
        container = CreateDefectFromIds._original_func(
            structure_container=self.container,
            defect_type="substitution",
            atom_ids=[i, j],
            to_element="Mg",
        )
        row = container._structures[-1]
        result = GetSubstitutionDistancesRelaxed._original_func(
            row["structure"], row["events"]
        )
        self.assertAlmostEqual(result["distances"]["0-1"], expected, places=6)

    def test_interstitial_distances_relaxed_locates_via_uid(self):
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
        row = container._structures[-1]
        result = GetInterstitialDistancesRelaxed._original_func(
            row["structure"], row["events"]
        )
        self.assertAlmostEqual(result["distances"]["0-1"], expected, places=6)

    def test_substitution_relaxed_needs_at_least_two_events(self):
        container = CreateDefectFromIds._original_func(
            structure_container=self.container,
            defect_type="substitution",
            atom_ids=[0],
            to_element="Mg",
        )
        row = container._structures[-1]
        result = GetSubstitutionDistancesRelaxed._original_func(
            row["structure"], row["events"]
        )
        self.assertEqual(result["distances"], {})
        self.assertIn("message", result)


class TestTableAndSelectionWrappers(unittest.TestCase):
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

    def test_get_structure_table(self):
        df = GetStructureTable._original_func(self.container)
        self.assertEqual(len(df), 3)
        self.assertIn("stoichiometry", df.columns)

    def test_get_defect_table(self):
        df = GetDefectTable._original_func(self.container)
        self.assertEqual(len(df), 2)
        self.assertTrue((df["is_pristine"] == False).all())

    def test_get_pristine_table(self):
        df = GetPristineTable._original_func(self.container)
        self.assertEqual(len(df), 1)
        self.assertTrue((df["is_pristine"] == True).all())

    def test_get_pristine_structures(self):
        rows = GetPristineStructures._original_func(self.container)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["is_pristine"])

    def test_get_defect_structures(self):
        rows = GetDefectStructures._original_func(self.container)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(not r["is_pristine"] for r in rows))

    def test_latest_pristine_index(self):
        self.assertEqual(LatestPristineIndex._original_func(self.container), 0)

    def test_resolve_defect_row(self):
        self.assertEqual(ResolveDefectRow._original_func(self.container, 0), 1)
        self.assertEqual(ResolveDefectRow._original_func(self.container, -1), 2)

    def test_resolve_any_row(self):
        self.assertEqual(ResolveAnyRow._original_func(self.container, 0), 0)
        self.assertEqual(ResolveAnyRow._original_func(self.container, -1), 2)


class TestGetVoronoiInterstitialSitesPymatgen(unittest.TestCase):
    """
    pymatgen-analysis-defects is a separate package (its own GitHub repo,
    not just `pymatgen`); it's pinned in .ci_support/environment-mini.yml
    but may not be present in every environment this suite runs in (e.g. a
    local dev env), so the "dependency missing" path below is still real,
    reachable behavior worth covering rather than an environment-mini-only
    guarantee.

    Both tests deliberately use the small 4-atom conventional cell, not
    make_atoms()'s 3x3x3 supercell: SpacegroupAnalyzer's symmetry search
    (used internally by VoronoiInterstitialGenerator) scales badly with
    atom count on a non-primitive cell, and takes minutes rather than
    seconds on 108 atoms. Voronoi-void geometry doesn't depend on
    supercell size, so the unit cell is sufficient to exercise the
    function correctly and fast.
    """

    def test_raises_helpful_error_when_dependency_missing(self):
        if PYMATGEN_DEFECTS_AVAILABLE:
            self.skipTest(
                "pymatgen-analysis-defects is installed; success path covered below"
            )
        with self.assertRaises(ImportError):
            GetVoronoiInterstitialSitesPymatgen._original_func(bulk("Al", cubic=True))

    @unittest.skipUnless(
        PYMATGEN_DEFECTS_AVAILABLE, "requires pymatgen-analysis-defects"
    )
    def test_returns_sites_when_available(self):
        atoms = bulk("Al", cubic=True)
        unique_sites, all_sites = GetVoronoiInterstitialSitesPymatgen._original_func(
            atoms
        )
        self.assertGreater(len(all_sites), 0)


class TestModuleLevelHelperEdgeCases(unittest.TestCase):
    def test_next_uid_returns_zero_when_key_absent(self):
        atoms = bulk("Al", cubic=True)
        self.assertEqual(next_uid(atoms), 0)

    def test_uid_to_index_returns_none_when_key_absent(self):
        atoms = bulk("Al", cubic=True)
        self.assertIsNone(uid_to_index(atoms, 0))

    def test_validate_atoms_arrays_raises_on_length_mismatch(self):
        atoms = ensure_uids(bulk("Al", cubic=True))
        atoms.arrays["uid"] = atoms.arrays["uid"][:-1]
        with self.assertRaises(ValueError):
            validate_atoms_arrays(atoms)

    def test_make_operations_short_empty_events(self):
        self.assertEqual(make_operations_short([]), "no_operations")

    def test_protected_uids_from_events_substitution_atom_uid(self):
        events = [{"type": "substitution", "atom_uid": 5}]
        self.assertEqual(_protected_uids_from_events(events), {5})

    def test_protected_uids_from_events_substitution_site_uid_fallback(self):
        # Real substitution events always carry both atom_uid and site_uid;
        # this exercises the defensive fallback for the case they don't.
        events = [{"type": "substitution", "site_uid": 7}]
        self.assertEqual(_protected_uids_from_events(events), {7})

    def test_protected_uids_from_events_interstitial(self):
        events = [{"type": "interstitial", "atom_uid": 9}]
        self.assertEqual(_protected_uids_from_events(events), {9})

    def test_protected_uids_from_events_unknown_type_ignored(self):
        events = [{"type": "vacancy", "site_uid": 3}]
        self.assertEqual(_protected_uids_from_events(events), set())


class TestFilterAndSelectionNodeWrappers(unittest.TestCase):
    """Covers the remaining Filter*/GetStructure node wrappers and the
    underlying StructureContainer methods that TestFilterFunctions and
    TestTableAndSelectionWrappers didn't already exercise."""

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

    def test_filter_by_indices(self):
        rows = FilterByIndices._original_func(self.container, [0, 2])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["stoichiometry"], "Al107Mg1")

    def test_filter_by_max_generation(self):
        rows = FilterByMaxGeneration._original_func(self.container, 0)
        self.assertEqual(len(rows), 1)

    def test_filter_by_operations_short_exact(self):
        # A pattern with no "*"/"?"/"[" routes through the exact-match
        # branch. Note "vacancy[0]" itself would NOT work here: fnmatch
        # treats "[0]" as a character class (matches a single "0"), so any
        # operations_short value with bracket notation can only be matched
        # via the wildcard branch below, never exactly.
        rows = FilterByOperationsShort._original_func(self.container, "pristine")
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["is_pristine"])

    def test_filter_by_operations_short_wildcard(self):
        rows = FilterByOperationsShort._original_func(self.container, "vacancy*")
        self.assertEqual(len(rows), 1)

    def test_filter_by_operations_contains(self):
        rows = FilterByOperationsContains._original_func(self.container, "substitution")
        self.assertEqual(len(rows), 1)

    def test_filter_by_unique_id(self):
        uid = self.container._structures[1]["unique_id"]
        row = FilterByUniqueId._original_func(self.container, uid)
        self.assertIsNotNone(row)
        self.assertEqual(row["unique_id"], uid)

    def test_filter_by_number_of_atoms(self):
        rows = FilterByNumberOfAtoms._original_func(self.container, 107)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["operations_short"], "vacancy[0]")

    def test_filter_by_element_count_exact(self):
        rows = FilterByElementCount._original_func(self.container, "Mg", exact_count=1)
        self.assertEqual(len(rows), 1)

    def test_filter_by_element_count_range(self):
        rows = FilterByElementCount._original_func(
            self.container, "Al", min_count=100, max_count=200
        )
        self.assertEqual(len(rows), 3)

    def test_filter_by_parent(self):
        rows = FilterByParent._original_func(self.container, 0)
        self.assertEqual(len(rows), 2)

    def test_filter_by_stoichiometry_none_returns_all(self):
        rows = FilterByStoichiometry._original_func(self.container, None)
        self.assertEqual(len(rows), 3)

    def test_filter_by_condition(self):
        rows = filter_by_condition(self.container, lambda s: s["generation"] == 1)
        self.assertEqual(len(rows), 2)

    def test_get_structure(self):
        row = GetStructure._original_func(self.container, 1)
        self.assertEqual(row["operations_short"], "vacancy[0]")

    def test_get_structure_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            GetStructure._original_func(self.container, 99)

    def test_find_structure_index_identity_branch(self):
        # Passing back the exact stored object (not a numerically-equal
        # copy) hits the fast identity-check path in find_structure_index.
        stored_atoms = self.container._structures[0]["structure"]
        idx = FindStructureIndex._original_func(self.container, stored_atoms)
        self.assertEqual(idx, 0)

    def test_repr(self):
        text = repr(self.container)
        self.assertIn("3 structures", text)
        self.assertIn("1 pristine", text)
        self.assertIn("2 defects", text)

    def test_find_pristine_index_fallback_walk(self):
        # add_defect always stores a valid pristine_structure_index, so the
        # walk-up-the-parent-chain fallback only triggers if that bookkeeping
        # is ever missing/stale -- simulate that directly.
        self.container._structures[1]["pristine_structure_index"] = -1
        self.assertEqual(self.container._find_pristine_index(1), 0)


class TestResolveParentEdgeCases(unittest.TestCase):
    def setUp(self):
        self.atoms = make_atoms()
        self.container = AddPristine._original_func(atoms=self.atoms)
        self.container = CreateDefectFromIds._original_func(
            structure_container=self.container, defect_type="vacancy", atom_ids=[0]
        )

    def test_both_parent_defect_index_and_input_structure_warns(self):
        with self.assertWarns(UserWarning):
            idx = _resolve_parent(
                self.container,
                parent_defect_index=1,
                input_structure=self.atoms,
            )
        self.assertEqual(idx, 1)

    def test_parent_defect_index_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            _resolve_parent(self.container, parent_defect_index=99)

    def test_parent_defect_index_pointing_to_pristine_raises(self):
        with self.assertRaises(ValueError):
            _resolve_parent(self.container, parent_defect_index=0)

    def test_input_structure_not_found_adds_new_pristine(self):
        other = bulk("Cu", cubic=True)  # different atom count than any stored row
        idx = _resolve_parent(self.container, input_structure=other)
        self.assertEqual(idx, len(self.container) - 1)
        self.assertTrue(self.container._structures[idx]["is_pristine"])

    def test_no_pristine_structure_raises(self):
        empty = StructureContainer()
        with self.assertRaises(ValueError):
            _resolve_parent(empty)


class TestCreateDefectFromIdsMoreEdgeCases(unittest.TestCase):
    def setUp(self):
        self.atoms = make_atoms()
        self.container = AddPristine._original_func(atoms=self.atoms)

    def test_vacancy_atom_ids_none_raises(self):
        with self.assertRaises(ValueError):
            CreateDefectFromIds._original_func(
                structure_container=self.container, defect_type="vacancy"
            )

    def test_vacancy_index_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            CreateDefectFromIds._original_func(
                structure_container=self.container,
                defect_type="vacancy",
                atom_ids=[99999],
            )

    def test_vacancy_protect_history(self):
        container = CreateDefectFromIds._original_func(
            structure_container=self.container,
            defect_type="vacancy",
            atom_ids=[0],
            parent_defect_index=None,
        )
        # Chain a second vacancy on top with protect_history -- exercises
        # the forbid |= _protected_uids_from_events(...) line for vacancy.
        container = CreateDefectFromIds._original_func(
            structure_container=container,
            defect_type="vacancy",
            atom_ids=[5],
            parent_defect_index=-1,
            protect_history=True,
        )
        self.assertEqual(len(container._structures[-1]["structure"]), 106)

    def test_substitution_atom_ids_none_raises(self):
        with self.assertRaises(ValueError):
            CreateDefectFromIds._original_func(
                structure_container=self.container,
                defect_type="substitution",
                to_element="Mg",
            )

    def test_substitution_index_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            CreateDefectFromIds._original_func(
                structure_container=self.container,
                defect_type="substitution",
                atom_ids=[99999],
                to_element="Mg",
            )

    def test_substitution_protect_history(self):
        container = CreateDefectFromIds._original_func(
            structure_container=self.container,
            defect_type="substitution",
            atom_ids=[0],
            to_element="Mg",
        )
        container = CreateDefectFromIds._original_func(
            structure_container=container,
            defect_type="substitution",
            atom_ids=[5],
            to_element="Mg",
            parent_defect_index=-1,
            protect_history=True,
        )
        self.assertEqual(container._structures[-1]["generation"], 2)

    def test_substitution_no_valid_indices_after_forbid_raises(self):
        with self.assertRaises(ValueError):
            CreateDefectFromIds._original_func(
                structure_container=self.container,
                defect_type="substitution",
                atom_ids=[0],
                to_element="Mg",
                forbid_atom_ids=[0],
            )

    def test_interstitial_bad_sublattice_shape_raises(self):
        with self.assertRaises(ValueError):
            CreateDefectFromIds._original_func(
                structure_container=self.container,
                defect_type="interstitial",
                sublattice=np.zeros((3, 2)),
                site_ids=[0],
                element="Mg",
            )

    def test_interstitial_empty_site_ids_raises(self):
        unique_sites, all_sites = GetVoronoiInterstitialSites._original_func(self.atoms)
        with self.assertRaises(ValueError):
            CreateDefectFromIds._original_func(
                structure_container=self.container,
                defect_type="interstitial",
                sublattice=all_sites,
                site_ids=[],
                element="Mg",
            )

    def test_interstitial_site_id_out_of_range_raises(self):
        unique_sites, all_sites = GetVoronoiInterstitialSites._original_func(self.atoms)
        with self.assertRaises(IndexError):
            CreateDefectFromIds._original_func(
                structure_container=self.container,
                defect_type="interstitial",
                sublattice=all_sites,
                site_ids=[len(all_sites) + 100],
                element="Mg",
            )


class TestCreateDefectBatchFromIdsInterstitial(unittest.TestCase):
    def setUp(self):
        self.atoms = make_atoms()
        self.container = AddPristine._original_func(atoms=self.atoms)
        _, self.all_sites = GetVoronoiInterstitialSites._original_func(self.atoms)

    def test_interstitial_batch_success(self):
        container = CreateDefectBatchFromIds._original_func(
            structure_container=self.container,
            defect_type="interstitial",
            target_indices=[0],
            sublattice=self.all_sites,
            site_ids=[0, 1],
            element="Mg",
            separate_structures=True,
        )
        self.assertEqual(len(container), 3)  # 1 pristine + 2 separate interstitials

    def test_interstitial_batch_missing_sublattice_raises(self):
        with self.assertRaises(ValueError):
            CreateDefectBatchFromIds._original_func(
                structure_container=self.container,
                defect_type="interstitial",
                target_indices=[0],
                element="Mg",
            )


class TestCreateDefectFromSeedMoreEdgeCases(unittest.TestCase):
    def setUp(self):
        self.atoms = make_atoms()
        self.container = AddPristine._original_func(atoms=self.atoms)

    def test_unknown_defect_type_raises(self):
        with self.assertRaises(ValueError):
            CreateDefectFromSeed._original_func(
                structure_container=self.container, defect_type="antisite", n=1, seed=0
            )

    def test_vacancy_element_list_no_candidates_for_element_raises(self):
        # Container is pure Al; requesting a Mg vacancy leaves zero candidates.
        with self.assertRaises(ValueError):
            CreateDefectFromSeed._original_func(
                structure_container=self.container,
                defect_type="vacancy",
                n=2,
                seed=0,
                vacancy_element=["Al", "Mg"],
            )

    def test_vacancy_element_none_not_enough_candidates_raises(self):
        with self.assertRaises(ValueError):
            CreateDefectFromSeed._original_func(
                structure_container=self.container,
                defect_type="vacancy",
                n=1000,
                seed=0,
            )

    def test_vacancy_element_single_string(self):
        container = CreateDefectFromSeed._original_func(
            structure_container=self.container,
            defect_type="vacancy",
            n=2,
            seed=0,
            vacancy_element="Al",
        )
        self.assertEqual(len(container._structures[-1]["structure"]), 106)

    def test_vacancy_element_single_string_not_enough_candidates_raises(self):
        with self.assertRaises(ValueError):
            CreateDefectFromSeed._original_func(
                structure_container=self.container,
                defect_type="vacancy",
                n=1000,
                seed=0,
                vacancy_element="Al",
            )

    def test_substitution_protect_history_and_multi(self):
        container = CreateDefectFromSeed._original_func(
            structure_container=self.container,
            defect_type="substitution",
            n=2,
            seed=0,
            from_element="Al",
            to_element="Mg",
            protect_history=True,
        )
        self.assertEqual(container._structures[-1]["operation"], "substitution[2:Al->Mg]")

    def test_interstitial_missing_sublattice_raises(self):
        with self.assertRaises(ValueError):
            CreateDefectFromSeed._original_func(
                structure_container=self.container, defect_type="interstitial", n=1, seed=0
            )

    def test_interstitial_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            CreateDefectFromSeed._original_func(
                structure_container=self.container,
                defect_type="interstitial",
                n=1,
                seed=0,
                sublattice=np.zeros((3, 2)),
                element="Mg",
            )

    def test_interstitial_empty_sublattice_raises(self):
        with self.assertRaises(ValueError):
            CreateDefectFromSeed._original_func(
                structure_container=self.container,
                defect_type="interstitial",
                n=1,
                seed=0,
                sublattice=np.zeros((0, 3)),
                element="Mg",
            )

    def test_interstitial_n_exceeds_sublattice_raises(self):
        with self.assertRaises(ValueError):
            CreateDefectFromSeed._original_func(
                structure_container=self.container,
                defect_type="interstitial",
                n=5,
                seed=0,
                sublattice=np.zeros((2, 3)),
                element="Mg",
            )


class TestCreateDefectBatchFromSeedInterstitial(unittest.TestCase):
    def test_interstitial_batch_success(self):
        atoms = make_atoms()
        container = AddPristine._original_func(atoms=atoms)
        _, all_sites = GetVoronoiInterstitialSites._original_func(atoms)
        container = CreateDefectBatchFromSeed._original_func(
            structure_container=container,
            defect_type="interstitial",
            target_indices=[0],
            sublattice=all_sites,
            element="Mg",
            n=1,
            seed=0,
            n_structures=2,
        )
        self.assertEqual(len(container), 3)  # 1 pristine + 2 seeded interstitials


class TestDistanceGetterEdgeCases(unittest.TestCase):
    def setUp(self):
        self.atoms = make_atoms()
        self.container = AddPristine._original_func(atoms=self.atoms)

    def test_substitution_distances_needs_at_least_two(self):
        container = CreateDefectFromIds._original_func(
            structure_container=self.container,
            defect_type="substitution",
            atom_ids=[0],
            to_element="Mg",
        )
        result = GetSubstitutionDistances._original_func(container, len(container) - 1)
        self.assertEqual(result["distances"], {})
        self.assertIn("message", result)

    def test_interstitial_distances_needs_at_least_two(self):
        unique_sites, all_sites = GetVoronoiInterstitialSites._original_func(self.atoms)
        container = CreateDefectFromIds._original_func(
            structure_container=self.container,
            defect_type="interstitial",
            sublattice=all_sites,
            site_ids=[0],
            element="Mg",
        )
        result = GetInterstitialDistances._original_func(container, len(container) - 1)
        self.assertEqual(result["distances"], {})
        self.assertIn("message", result)

    def test_interstitial_distances_relaxed_needs_at_least_two(self):
        unique_sites, all_sites = GetVoronoiInterstitialSites._original_func(self.atoms)
        container = CreateDefectFromIds._original_func(
            structure_container=self.container,
            defect_type="interstitial",
            sublattice=all_sites,
            site_ids=[0],
            element="Mg",
        )
        row = container._structures[-1]
        result = GetInterstitialDistancesRelaxed._original_func(
            row["structure"], row["events"]
        )
        self.assertEqual(result["distances"], {})
        self.assertIn("message", result)

    def test_interstitial_distances_relaxed_falls_back_to_nearest_neighbour(self):
        unique_sites, all_sites = GetVoronoiInterstitialSites._original_func(self.atoms)
        container = CreateDefectFromIds._original_func(
            structure_container=self.container,
            defect_type="interstitial",
            sublattice=all_sites,
            site_ids=[0, 1],
            element="Mg",
        )
        row = container._structures[-1]
        # Strip the uid array so uid_to_index can't find the atoms by uid,
        # forcing the nearest-neighbour fallback path.
        atoms_no_uid = row["structure"].copy()
        del atoms_no_uid.arrays["uid"]
        result = GetInterstitialDistancesRelaxed._original_func(
            atoms_no_uid, row["events"]
        )
        self.assertEqual(len(result["distances"]), 1)


class TestFilterClusterTileHelperDirect(unittest.TestCase):
    def test_no_candidates_inside_cell(self):
        cell = np.eye(3) * 4.0
        raw_candidates = np.array([[100.0, 100.0, 100.0]])
        pos = np.array([[0.0, 0.0, 0.0]])
        unique_sites, all_sites = _filter_cluster_tile_interstitial_candidates(
            raw_candidates, pos, cell, 0.5, 0.5, False, None
        )
        self.assertEqual(len(unique_sites), 0)
        self.assertEqual(len(all_sites), 0)

    def test_no_candidates_after_r_min_filter(self):
        cell = np.eye(3) * 4.0
        pos = np.array([[0.0, 0.0, 0.0]])
        raw_candidates = np.array([[0.1, 0.1, 0.1]])
        unique_sites, all_sites = _filter_cluster_tile_interstitial_candidates(
            raw_candidates, pos, cell, 2.0, 0.5, False, None
        )
        self.assertEqual(len(unique_sites), 0)
        self.assertEqual(len(all_sites), 0)


class TestSiteFinderPrimitiveRepeat(unittest.TestCase):
    def test_voronoi_primitive_and_repeat_tiles(self):
        prim = bulk("Al", cubic=True)
        atoms = prim.repeat((2, 2, 2))
        unique_sites, all_sites = GetVoronoiInterstitialSites._original_func(
            atoms, primitive_atoms=prim, repeat=(2, 2, 2)
        )
        self.assertGreater(len(unique_sites), 0)
        self.assertGreater(len(all_sites), len(unique_sites))

    def test_voronoi_mismatched_primitive_repeat_raises(self):
        prim = bulk("Al", cubic=True)
        with self.assertRaises(ValueError):
            GetVoronoiInterstitialSites._original_func(make_atoms(), primitive_atoms=prim)

    def test_delaunay_primitive_and_repeat_tiles(self):
        prim = bulk("Al", cubic=True)
        atoms = prim.repeat((2, 2, 2))
        unique_sites, all_sites = GetDelaunayInterstitialSites._original_func(
            atoms, primitive_atoms=prim, repeat=(2, 2, 2)
        )
        self.assertGreater(len(unique_sites), 0)
        self.assertGreater(len(all_sites), len(unique_sites))

    def test_delaunay_mismatched_primitive_repeat_raises(self):
        with self.assertRaises(ValueError):
            GetDelaunayInterstitialSites._original_func(make_atoms(), repeat=(2, 2, 2))


class TestVoronoiAvailableFallback(unittest.TestCase):
    """
    VORONOI_AVAILABLE is set at import time based on whether
    structuretoolkit.common is importable. structuretoolkit is installed
    in this environment, so the False branch is only reachable by actually
    blocking that import -- done here in a subprocess to avoid corrupting
    the main test process's already-imported container_new module.
    """

    def test_voronoi_available_false_when_structuretoolkit_import_fails(self):
        repo_root = str(Path(__file__).resolve().parents[4])
        script = (
            "import builtins\n"
            "real_import = builtins.__import__\n"
            "def fake_import(name, *a, **k):\n"
            "    if name == 'structuretoolkit.common':\n"
            "        raise ImportError('blocked for test')\n"
            "    return real_import(name, *a, **k)\n"
            "builtins.__import__ = fake_import\n"
            "from pyiron_nodes.atomistic.structure import container_new\n"
            "assert container_new.VORONOI_AVAILABLE is False, container_new.VORONOI_AVAILABLE\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
