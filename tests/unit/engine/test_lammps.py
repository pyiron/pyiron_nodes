import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from ase.build import bulk, molecule
from ase.constraints import FixAtoms

from pyiron_nodes.atomistic.calculator.data import InputCalcMD, InputCalcMinimize
from pyiron_nodes.atomistic.engine.lammps import (
    CreateLammpsMDInput,
    CreateLammpsMinimizeInput,
    CreateLammpsStaticInput,
    CreateLammpsStructure,
    LammpsIOBundle,
    ListPotentials,
    ParseLammpsOutput,
    RunLammpsCalculation,
    extract_charges_from_lammps_potential,
    write_lammps_data_full,
)
from pyiron_nodes.electrochemistry.structure.equilibrate import WaterPotential

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


class TestLammpsDataFramePotential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        slab_potential, bond_dict = WaterPotential._original_func(quasi_2d=True)
        structure = molecule("H2O")
        structure.cell = [10, 10, 10]
        structure.pbc = True

        cls._tmp = tempfile.TemporaryDirectory()
        cls.io_bundle = CreateLammpsStructure._original_func(
            structure=structure,
            potential=slab_potential,
            working_directory=cls._tmp.name + "/water",
            bond_dict=bond_dict,
        )
        cls.io_bundle = CreateLammpsMDInput._original_func(
            io_bundle=cls.io_bundle,
            calc_dataclass=InputCalcMD._original_dataclass(),
        )
        _, cls.output = RunLammpsCalculation._original_func(
            io_bundle=cls.io_bundle, debug=True
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_units_from_dataframe(self):
        self.assertEqual(self.io_bundle.units, "real")

    def test_structure_string_generated(self):
        self.assertIn("Atoms", self.io_bundle.lammps_structure_string)

    def test_potential_string_written(self):
        self.assertIn("pair_style", self.io_bundle.lammps_potential_string)

    def test_debug_output_is_working_directory(self):
        self.assertEqual(self.output, self.io_bundle.working_directory)


class TestLammpsStringPotentialStaticAndMinimize(unittest.TestCase):
    """CreateLammpsStructure with a plain string potential exercises the
    non-'full' atom_type branch (LammpsStructure, not write_lammps_data_full),
    and CreateLammpsStaticInput/CreateLammpsMinimizeInput otherwise only get
    exercised by the integration tests, which require a real LAMMPS binary."""

    @classmethod
    def setUpClass(cls):
        cls.structure = bulk("Al", cubic=True)
        cls._tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _make_bundle(self, subdir):
        return CreateLammpsStructure._original_func(
            structure=self.structure,
            potential=AL_POTENTIAL,
            working_directory=self._tmp.name + "/" + subdir,
            resource_path=RESOURCE_PATH,
        )

    def test_structure_string_generated_for_string_potential(self):
        io_bundle = self._make_bundle("structure")
        self.assertIn("atoms", io_bundle.lammps_structure_string)

    def test_static_input_mode_and_content(self):
        io_bundle = self._make_bundle("static")
        static_bundle = CreateLammpsStaticInput._original_func(io_bundle=io_bundle)
        self.assertEqual(static_bundle.mode, "static")
        self.assertNotEqual(static_bundle.lammps_input_string, "")

    def test_static_debug_run_returns_working_directory(self):
        io_bundle = self._make_bundle("static_debug")
        static_bundle = CreateLammpsStaticInput._original_func(io_bundle=io_bundle)
        _, output = RunLammpsCalculation._original_func(
            io_bundle=static_bundle, debug=True
        )
        self.assertEqual(output, static_bundle.working_directory)

    def test_minimize_input_mode_and_content(self):
        io_bundle = self._make_bundle("minimize")
        minimize_bundle = CreateLammpsMinimizeInput._original_func(
            io_bundle=io_bundle,
            calc_dataclass=InputCalcMinimize._original_dataclass(),
        )
        self.assertEqual(minimize_bundle.mode, "minimize")
        self.assertNotEqual(minimize_bundle.lammps_input_string, "")

    def test_minimize_debug_run_returns_working_directory(self):
        io_bundle = self._make_bundle("minimize_debug")
        minimize_bundle = CreateLammpsMinimizeInput._original_func(
            io_bundle=io_bundle,
            calc_dataclass=InputCalcMinimize._original_dataclass(),
        )
        _, output = RunLammpsCalculation._original_func(
            io_bundle=minimize_bundle, debug=True
        )
        self.assertEqual(output, minimize_bundle.working_directory)


class TestElectrodeForceSamplingFrequency(unittest.TestCase):
    """The electrode-force fix used to share the dump interval.  A frame of
    trajectory is ~200 kB and a force sample is three floats, so on a long run
    that made keeping the force resolution cost a multi-GB dump.  Frozen atoms
    are what switches the force output on at all, hence the constraint."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _input_string(self, subdir, **calc_kwargs):
        structure = bulk("Al", cubic=True).repeat((2, 1, 1))
        structure.set_constraint(FixAtoms(indices=[0, 1]))
        io_bundle = CreateLammpsStructure._original_func(
            structure=structure,
            potential=AL_POTENTIAL,
            working_directory=self._tmp.name + "/" + subdir,
            resource_path=RESOURCE_PATH,
        )
        return CreateLammpsMDInput._original_func(
            io_bundle=io_bundle,
            calc_dataclass=InputCalcMD._original_dataclass(**calc_kwargs),
        ).lammps_input_string

    def test_the_force_fix_can_sample_more_often_than_the_dump(self):
        text = self._input_string("force_fast", n_print=100, n_print_force=10)

        self.assertIn("variable dumptime equal 100", text)
        self.assertIn("variable forcetime equal 10", text)
        # The two must end up on different lines of the script.
        self.assertIn("dump 1 all custom ${dumptime} dump.out", text)
        self.assertIn("ave/time 1 1 ${forcetime}", text)
        self.assertNotIn("ave/time 1 1 ${dumptime}", text)

    def test_leaving_it_unset_follows_the_dump_interval(self):
        text = self._input_string("force_default", n_print=100)

        self.assertIn("variable dumptime equal 100", text)
        self.assertIn("variable forcetime equal 100", text)

    def test_the_new_field_does_not_reach_calc_md(self):
        # calc_kwargs is splatted into calc_md, which has no such argument, so
        # it has to be popped rather than read.  A regression here is a
        # TypeError at generation time, not a wrong number in the script.
        text = self._input_string("force_popped", n_print_force=5)

        self.assertIn("variable forcetime equal 5", text)
        self.assertNotIn("n_print_force", text)


def _electrochemical_cell(piston_free=True):
    """Miniature of the electrochemistry cell: Al / H2O / Ne along z.

    The TIP3P potential from ``WaterPotential`` declares H, O, Al and Ne, so it
    is the only potential in this file that can carry an electrode plus water.
    """
    from ase import Atoms
    from ase.constraints import FixedPlane

    structure = Atoms(
        symbols=["Al", "Al", "O", "H", "H", "Ne", "Ne"],
        positions=[
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [1.0, 1.0, 5.0],
            [1.0, 1.757, 5.587],
            [1.0, 0.243, 5.587],
            [0.0, 0.0, 10.0],
            [2.0, 2.0, 10.0],
        ],
        cell=[4.0, 4.0, 30.0],
        pbc=[True, True, False],
    )
    constraints = [FixAtoms(indices=[0, 1])]
    if piston_free:
        constraints.append(FixedPlane(indices=[5, 6], direction=[1, 1, 0]))
    else:
        constraints = [FixAtoms(indices=[0, 1, 5, 6])]
    structure.set_constraint(constraints)
    return structure


def _water_cell_input(working_directory, structure=None, **md_kwargs):
    """``CreateLammpsMDInput`` on the miniature cell, as an input string."""
    potential, bond_dict = WaterPotential._original_func(metal="Al", quasi_2d=True)
    io_bundle = CreateLammpsStructure._original_func(
        structure=_electrochemical_cell() if structure is None else structure,
        potential=potential,
        working_directory=working_directory,
        bond_dict=bond_dict,
    )
    return CreateLammpsMDInput._original_func(
        io_bundle=io_bundle,
        calc_dataclass=InputCalcMD._original_dataclass(),
        **md_kwargs,
    ).lammps_input_string


def _piston_load_value(lines):
    """The z force on the `fix piston_load` line, whatever its position."""
    line = next(l for l in lines if "piston_load" in l)
    return float(line.split()[-1])


class TestPistonBarostat(unittest.TestCase):
    """The piston holds one electrode at a constant normal pressure.  The two
    fixes have to land *after* the setforce from the constraint, or the
    constraint pins the piston and the load silently does nothing."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _structure(self):
        # The cell the electrochemistry workflow builds, in miniature: an Al
        # electrode frozen outright, water in between, and a Ne electrode pinned
        # in xy but free in z so the piston can push it.
        return _electrochemical_cell()

    def _input_string(self, subdir, **md_kwargs):
        return _water_cell_input(self._tmp.name + "/" + subdir, **md_kwargs)

    def test_no_piston_species_emits_no_piston_fixes(self):
        text = self._input_string("piston_off")

        self.assertNotIn("piston_load", text)
        self.assertNotIn("piston_damp", text)

    def test_the_piston_fix_follows_the_constraint_setforce(self):
        text = self._input_string(
            "piston_order", piston_species="Ne", piston_pressure=1.0
        )

        self.assertIn("fix piston_load frozen_ne aveforce NULL NULL ", text)
        # LAMMPS applies post-force fixes in definition order, so the piston
        # fix only overrides the constraint if it comes later.
        self.assertLess(
            text.index("constraintxy setforce 0.0 0.0 NULL"),
            text.index("fix piston_load"),
        )

    def test_the_load_component_is_not_NULL(self):
        # `fix aveforce` only averages the components it is given a value for,
        # so an all-NULL fix is a silent no-op: the layer never rigidifies and
        # no load is applied, yet LAMMPS runs happily to completion.
        text = self._input_string("piston_not_null", piston_species="Ne")

        line = next(l for l in text.splitlines() if "piston_load" in l)
        self.assertNotEqual(line.split()[-1], "NULL")
        self.assertEqual(line.split()[-3:-1], ["NULL", "NULL"])

    def test_the_piston_starts_at_rest_so_the_layer_stays_planar(self):
        # The constraint only zeroes vx and vy; without this each piston atom
        # keeps its own random vz and the plane shears apart within a few ps,
        # even though `aveforce` gives them all the same force.
        text = self._input_string("piston_velocity", piston_species="Ne")

        self.assertIn("velocity frozen_ne set 0.0 0.0 0.0", text)
        self.assertLess(
            text.index("velocity constraintxy set 0.0 0.0 NULL"),
            text.index("velocity frozen_ne set 0.0 0.0 0.0"),
            "the piston reset has to override the constraint's, so it comes after",
        )

    def test_damping_is_opt_in(self):
        without = self._input_string("piston_undamped", piston_species="Ne")
        self.assertNotIn("piston_damp", without)

        with_damping = self._input_string(
            "piston_damped", piston_species="Ne", piston_damping=2.5
        )
        self.assertIn("fix piston_damp frozen_ne viscous 2.5", with_damping)

    def test_the_load_is_the_pressure_times_the_area_per_atom(self):
        from pyiron_nodes.atomistic.engine.lammps import _piston_barostat_lines

        # Hand-computed for the miniature cell: 2 Ne atoms, 4x4 Å cross-section,
        # 1 bar, metal units where force is already eV/Å.
        structure = self._structure()
        area = float(structure.cell[0, 0] * structure.cell[1, 1])
        expected_ev_per_ang = -1.0 * 1e5 * area * 1e-20 / 1.602176634e-9 / 2

        lines = _piston_barostat_lines(
            structure=structure,
            units="metal",
            piston_species="Ne",
            piston_pressure=1.0,
            piston_damping=0.0,
        )
        value = float(_piston_load_value(lines))

        self.assertAlmostEqual(value / expected_ev_per_ang, 1.0, places=9)
        self.assertLess(value, 0.0, "the load must push the piston inward")

    def test_the_load_is_converted_for_real_units(self):
        # The TIP3P water potential switches the run to `units real`, where the
        # same physical load is a different number.
        from pyiron_nodes.atomistic.engine.lammps import _piston_barostat_lines

        structure = self._structure()
        values = {}
        for units in ("metal", "real"):
            lines = _piston_barostat_lines(
                structure=structure,
                units=units,
                piston_species="Ne",
                piston_pressure=1.0,
                piston_damping=0.0,
            )
            values[units] = _piston_load_value(lines)

        from lammpsparser.units import LAMMPS_UNIT_CONVERSIONS

        # kcal/mol/Å per eV/Å
        self.assertAlmostEqual(
            values["real"] / values["metal"],
            LAMMPS_UNIT_CONVERSIONS["real"]["force"],
            places=9,
        )

    def test_the_generated_script_uses_the_runs_own_units(self):
        # The potential drives the unit style, so the number in the script has
        # to follow it rather than a hard-coded default.
        from pyiron_nodes.atomistic.engine.lammps import _piston_barostat_lines

        text = self._input_string(
            "piston_units", piston_species="Ne", piston_pressure=1.0
        )
        self.assertIn("units      real", text)

        in_script = float(
            next(l for l in text.splitlines() if "piston_load" in l).split()[-1]
        )
        expected = _piston_load_value(
            _piston_barostat_lines(
                structure=self._structure(),
                units="real",
                piston_species="Ne",
                piston_pressure=1.0,
                piston_damping=0.0,
            )
        )
        self.assertAlmostEqual(in_script / expected, 1.0, places=9)

    def test_an_absent_piston_species_is_an_error(self):
        with self.assertRaises(ValueError) as ctx:
            self._input_string("piston_missing", piston_species="Xe")
        self.assertIn("Xe", str(ctx.exception))


class TestElectrodeForceGroups(unittest.TestCase):
    """Partially constrained atoms are electrodes too — a piston is pinned in xy
    and free in z, and it is exactly the electrode the barostat is judged by."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_a_plane_constrained_species_still_gets_a_force_group(self):
        text = _water_cell_input(self._tmp.name + "/mixed")

        # Ne is pinned only in xy, Al outright; both are electrodes.
        self.assertIn("group frozen_ne id 6 7", text)
        self.assertIn("electrode_force_Ne.txt", text)
        self.assertIn("group frozen_al id 1 2", text)
        self.assertIn("electrode_force_Al.txt", text)
        # The water is what pushes on them, so it must stay out of the groups.
        self.assertIn("group mobile_el subtract all frozen_all", text)

    def test_fully_frozen_electrodes_are_unaffected(self):
        text = _water_cell_input(
            self._tmp.name + "/all_fixed",
            structure=_electrochemical_cell(piston_free=False),
        )

        self.assertIn("group frozen_ne id 6 7", text)
        self.assertIn("group frozen_al id 1 2", text)

    def test_the_group_group_compute_includes_kspace(self):
        # Without `kspace yes` the PPPM part of the force is dropped, which on a
        # charged electrode is most of the electrostatics.
        text = _water_cell_input(self._tmp.name + "/kspace")

        compute_lines = [l for l in text.splitlines() if "group/group" in l]
        self.assertTrue(compute_lines)
        for line in compute_lines:
            self.assertIn("kspace yes", line)


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
