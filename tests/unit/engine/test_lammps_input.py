import os
import sys
import tempfile
import unittest
from pathlib import Path

from ase.build import bulk
from lammpsparser.units import LAMMPS_UNIT_CONVERSIONS

from pyiron_nodes.atomistic.engine.lammps import LammpsIOBundle
from pyiron_nodes.atomistic.engine.lammps_input import (
    AssembleLammpsInput,
    LammpsDump,
    LammpsEnsemble,
    LammpsInit,
    LammpsMinimize,
    LammpsPotential,
    LammpsThermo,
    LammpsVelocity,
    SetLammpsInputString,
    WriteFile,
)
from pyiron_nodes.electrochemistry.structure.equilibrate import TIP3PSlabPotential

AL_POTENTIAL = "1999--Mishin-Y--Al--LAMMPS--ipr1"
RESOURCE_PATH = os.environ.get(
    "IPRPY_RESOURCE_PATH",
    str(Path(sys.executable).parent.parent / "share" / "iprpy"),
)


class TestLammpsInit(unittest.TestCase):
    def test_default_read_data(self):
        block = LammpsInit._original_func(structure=bulk("Al", cubic=True))
        self.assertIn("units metal", block)
        self.assertIn("dimension 3", block)
        self.assertIn("boundary p p p", block)
        self.assertIn("atom_style atomic", block)
        self.assertIn("read_data lammps.data", block)
        self.assertNotIn("read_restart", block)

    def test_read_restart(self):
        block = LammpsInit._original_func(
            structure=bulk("Al", cubic=True), read_restart="restart.lammps"
        )
        self.assertIn("read_restart restart.lammps", block)
        self.assertNotIn("read_data", block)

    def test_custom_units_and_filename(self):
        block = LammpsInit._original_func(
            structure=bulk("Al", cubic=True),
            units="real",
            atom_style="full",
            structure_filename="structure.data",
        )
        self.assertIn("units real", block)
        self.assertIn("atom_style full", block)
        self.assertIn("read_data structure.data", block)


class TestLammpsPotential(unittest.TestCase):
    def test_inline_branch_for_string_potential(self):
        section, potential_file_content = LammpsPotential._original_func(
            potential=AL_POTENTIAL, resource_path=RESOURCE_PATH
        )
        self.assertIn("pair_style", section)
        self.assertEqual(potential_file_content, "")

    def test_include_branch_for_dataframe_potential(self):
        slab_potential, _ = TIP3PSlabPotential._original_func()
        section, potential_file_content = LammpsPotential._original_func(
            potential=slab_potential
        )
        self.assertEqual(section, "include potential.inp\n")
        self.assertIn("pair_style", potential_file_content)

    def test_custom_potential_filename(self):
        slab_potential, _ = TIP3PSlabPotential._original_func()
        section, _ = LammpsPotential._original_func(
            potential=slab_potential, potential_filename="mypot.inp"
        )
        self.assertEqual(section, "include mypot.inp\n")


class TestLammpsDump(unittest.TestCase):
    def test_default(self):
        block = LammpsDump._original_func()
        self.assertIn("variable dumptime equal 100", block)
        self.assertIn("dump 1 all custom ${dumptime} dump.out", block)
        # Regression: the rendered block must end with the closing '"' of the
        # dump_modify format string intact (core's _coerce() strips leading/
        # trailing quotes from string port values otherwise).
        self.assertTrue(block.rstrip("\n").endswith('"'))

    def test_custom_n_print_and_filename(self):
        block = LammpsDump._original_func(n_print=50, filename="traj.out")
        self.assertIn("variable dumptime equal 50", block)
        self.assertIn("traj.out", block)


class TestLammpsThermo(unittest.TestCase):
    def test_default(self):
        block = LammpsThermo._original_func()
        self.assertIn("variable thermotime equal 100", block)
        self.assertIn("thermo ${thermotime}", block)

    def test_custom_n_print(self):
        block = LammpsThermo._original_func(n_print=25)
        self.assertIn("variable thermotime equal 25", block)


class TestLammpsVelocity(unittest.TestCase):
    def test_default_metal_units(self):
        conversions = LAMMPS_UNIT_CONVERSIONS["metal"]
        block = LammpsVelocity._original_func()
        expected_ts = 1.0 * conversions["time"]
        expected_temp = 2.0 * 300.0 * conversions["temperature"]
        self.assertIn(f"timestep {expected_ts}", block)
        self.assertIn(f"velocity all create {expected_temp} 42 dist gaussian", block)

    def test_custom_seed_and_temperature(self):
        conversions = LAMMPS_UNIT_CONVERSIONS["metal"]
        block = LammpsVelocity._original_func(temperature=600.0, seed=7)
        expected_temp = 2.0 * 600.0 * conversions["temperature"]
        self.assertIn(f"velocity all create {expected_temp} 7 dist gaussian", block)


class TestLammpsMinimize(unittest.TestCase):
    def test_no_pressure(self):
        block = LammpsMinimize._original_func(e_tol=0.001, f_tol=0.001)
        self.assertIn("min_style cg", block)
        self.assertIn("minimize 0.001 0.001 1000000 1000000", block)
        self.assertNotIn("box/relax", block)
        self.assertNotIn("unfix ensemble", block)

    def test_pressure_zero_still_renders_box_relax(self):
        """Regression: pressure=0.0 must not be treated as falsy/unset."""
        conversions = LAMMPS_UNIT_CONVERSIONS["metal"]
        block = LammpsMinimize._original_func(e_tol=0.001, f_tol=0.001, pressure=0.0)
        expected_relax = 0.0 * conversions["pressure"]
        self.assertIn(f"fix ensemble all box/relax iso {expected_relax}", block)
        self.assertIn("unfix ensemble", block)

    def test_pressure_nonzero(self):
        conversions = LAMMPS_UNIT_CONVERSIONS["metal"]
        block = LammpsMinimize._original_func(e_tol=0.001, f_tol=0.001, pressure=2.0)
        expected_relax = 2.0 * conversions["pressure"]
        self.assertIn(f"fix ensemble all box/relax iso {expected_relax}", block)

    def test_custom_style_and_max_iter(self):
        block = LammpsMinimize._original_func(
            e_tol=0.001, f_tol=0.001, style="fire", max_iter=500
        )
        self.assertIn("min_style fire", block)
        self.assertIn("minimize 0.001 0.001 500 500", block)


class TestLammpsEnsemble(unittest.TestCase):
    def test_nve_when_no_temperature_or_pressure(self):
        block, fix_ids = LammpsEnsemble._original_func(temperature=None, pressure=None)
        self.assertIn("fix ensemble all nve", block)
        self.assertNotIn("nvt", block)
        self.assertNotIn("npt", block)
        self.assertEqual(fix_ids, ["ensemble"])

    def test_nvt_when_temperature_only(self):
        conversions = LAMMPS_UNIT_CONVERSIONS["metal"]
        block, fix_ids = LammpsEnsemble._original_func(temperature=300.0, pressure=None)
        t = 300.0 * conversions["temperature"]
        t_damp = 100.0 * conversions["time"]
        self.assertIn(f"fix ensemble all nvt temp {t} {t} {t_damp}", block)
        self.assertEqual(fix_ids, ["ensemble"])

    def test_npt_when_temperature_and_pressure(self):
        block, fix_ids = LammpsEnsemble._original_func(temperature=300.0, pressure=0.0)
        self.assertIn("fix ensemble all npt", block)
        self.assertEqual(fix_ids, ["ensemble"])

    def test_npt_pressure_zero_is_not_treated_as_nvt(self):
        """pressure=0.0 must select the npt branch (`is not None` check), not
        fall through to nvt."""
        block, _ = LammpsEnsemble._original_func(temperature=300.0, pressure=0.0)
        self.assertNotIn("nvt", block)

    def test_nve_langevin(self):
        block, fix_ids = LammpsEnsemble._original_func(
            temperature=300.0, pressure=None, langevin=True
        )
        self.assertIn("fix ensemble all nve", block)
        self.assertIn("fix langevin all langevin", block)
        self.assertEqual(fix_ids, ["ensemble", "langevin"])

    def test_nph_langevin(self):
        block, fix_ids = LammpsEnsemble._original_func(
            temperature=300.0, pressure=0.0, langevin=True
        )
        self.assertIn("fix ensemble all nph", block)
        self.assertIn("fix langevin all langevin", block)
        self.assertEqual(fix_ids, ["ensemble", "langevin"])


class TestAssembleLammpsInputSingleStage(unittest.TestCase):
    def _make_bundle(self):
        return LammpsIOBundle(structure=bulk("Al", cubic=True), potential=AL_POTENTIAL)

    def test_section_order(self):
        bundle = self._make_bundle()
        init = LammpsInit._original_func(structure=bundle.structure)
        potential, _ = LammpsPotential._original_func(
            potential=AL_POTENTIAL, resource_path=RESOURCE_PATH
        )
        dump = LammpsDump._original_func()
        thermo = LammpsThermo._original_func()
        velocity = LammpsVelocity._original_func()
        minimize = LammpsMinimize._original_func(e_tol=0.001, f_tol=0.001)

        result = AssembleLammpsInput._original_func(
            io_bundle=bundle,
            init=init,
            potential=potential,
            dump=dump,
            thermo=thermo,
            velocity=velocity,
            calculation=minimize,
            run=0,
        )
        s = result.lammps_input_string
        markers = [
            "units metal",  # init
            "pair_style",  # potential
            "variable dumptime",  # dump
            "variable thermotime",  # thermo
            "timestep",  # velocity
            "min_style",  # calculation
            "run 0",  # run
        ]
        positions = [s.index(m) for m in markers]
        self.assertEqual(positions, sorted(positions))

    def test_omitted_sections_absent(self):
        bundle = self._make_bundle()
        result = AssembleLammpsInput._original_func(
            io_bundle=bundle,
            init=LammpsInit._original_func(structure=bundle.structure),
            run=0,
        )
        s = result.lammps_input_string
        self.assertNotIn("dumptime", s)
        self.assertNotIn("thermotime", s)
        self.assertNotIn("timestep", s)
        self.assertNotIn("min_style", s)

    def test_write_restart_independent_of_run(self):
        bundle = self._make_bundle()
        result = AssembleLammpsInput._original_func(
            io_bundle=bundle, run=None, write_restart="foo.restart"
        )
        s = result.lammps_input_string
        self.assertIn("write_restart foo.restart", s)
        self.assertNotIn("run ", s)

    def test_potential_file_content_sets_potential_string(self):
        bundle = self._make_bundle()
        slab_potential, _ = TIP3PSlabPotential._original_func()
        section, potential_file_content = LammpsPotential._original_func(
            potential=slab_potential
        )
        result = AssembleLammpsInput._original_func(
            io_bundle=bundle,
            potential=section,
            potential_file_content=potential_file_content,
            run=0,
        )
        self.assertEqual(result.lammps_potential_string, potential_file_content)
        self.assertIn("include potential.inp", result.lammps_input_string)


class TestAssembleLammpsInputMultiStage(unittest.TestCase):
    def _make_bundle(self):
        return LammpsIOBundle(structure=bulk("Al", cubic=True), potential=AL_POTENTIAL)

    def test_second_call_appends_with_blank_line_separator(self):
        bundle = self._make_bundle()
        AssembleLammpsInput._original_func(io_bundle=bundle, run=0)
        first = bundle.lammps_input_string
        AssembleLammpsInput._original_func(io_bundle=bundle, run=10)
        second = bundle.lammps_input_string
        self.assertEqual(second, first + "\n" + "run 10\n")

    def test_init_and_potential_dropped_on_later_stage(self):
        bundle = self._make_bundle()
        init = LammpsInit._original_func(structure=bundle.structure)
        AssembleLammpsInput._original_func(io_bundle=bundle, init=init, run=0)
        # Re-supplying init on a later stage should be a silent-but-noted no-op.
        AssembleLammpsInput._original_func(io_bundle=bundle, init=init, run=10)
        self.assertEqual(bundle.lammps_input_string.count("units metal"), 1)

    def test_unfix_rendered_after_run_in_same_call(self):
        bundle = self._make_bundle()
        ensemble, fix_ids = LammpsEnsemble._original_func(temperature=300.0)
        result = AssembleLammpsInput._original_func(
            io_bundle=bundle,
            calculation=ensemble,
            calculation_fix_ids=fix_ids,
            run=100,
        )
        s = result.lammps_input_string
        self.assertIn("fix ensemble all nvt", s)
        self.assertIn("run 100", s)
        self.assertIn("unfix ensemble", s)
        self.assertLess(s.index("run 100"), s.index("unfix ensemble"))
        self.assertEqual(result.lammps_pending_fix_ids, "")

    def test_unfix_rendered_after_run_arriving_in_a_later_call(self):
        bundle = self._make_bundle()
        ensemble, fix_ids = LammpsEnsemble._original_func(temperature=300.0)
        # Stage 1 sets up the fix but does not run yet.
        AssembleLammpsInput._original_func(
            io_bundle=bundle, calculation=ensemble, calculation_fix_ids=fix_ids
        )
        self.assertEqual(bundle.lammps_pending_fix_ids, "ensemble")
        # Stage 2 supplies the run that actually uses the still-pending fix.
        AssembleLammpsInput._original_func(io_bundle=bundle, run=50)
        s = bundle.lammps_input_string
        self.assertIn("run 50", s)
        self.assertIn("unfix ensemble", s)
        self.assertLess(s.index("run 50"), s.index("unfix ensemble"))
        self.assertEqual(bundle.lammps_pending_fix_ids, "")

    def test_new_calculation_with_abandoned_fix_raises(self):
        bundle = self._make_bundle()
        ensemble1, fix_ids1 = LammpsEnsemble._original_func(temperature=300.0)
        AssembleLammpsInput._original_func(
            io_bundle=bundle, calculation=ensemble1, calculation_fix_ids=fix_ids1
        )
        ensemble2, fix_ids2 = LammpsEnsemble._original_func(temperature=350.0)
        with self.assertRaises(ValueError):
            AssembleLammpsInput._original_func(
                io_bundle=bundle, calculation=ensemble2, calculation_fix_ids=fix_ids2
            )


class TestSetLammpsInputString(unittest.TestCase):
    def test_sets_string_and_potential(self):
        bundle = LammpsIOBundle(structure=bulk("Al", cubic=True), potential=AL_POTENTIAL)
        result = SetLammpsInputString._original_func(
            io_bundle=bundle,
            lammps_input_string="units metal\n",
            potential_file_content="pair_style none\n",
        )
        self.assertEqual(result.lammps_input_string, "units metal\n")
        self.assertEqual(result.lammps_potential_string, "pair_style none\n")


class TestWriteFile(unittest.TestCase):
    def test_writes_input_string_to_disk(self):
        bundle = LammpsIOBundle(
            structure=bulk("Al", cubic=True),
            potential=AL_POTENTIAL,
            lammps_input_string="units metal\n",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lmp.in")
            result_path = WriteFile._original_func(io_bundle=bundle, filename=path)
            self.assertEqual(result_path, path)
            with open(path) as f:
                self.assertEqual(f.read(), "units metal\n")


if __name__ == "__main__":
    unittest.main()
