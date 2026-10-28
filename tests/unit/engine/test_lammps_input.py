import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from ase.build import bulk
from lammpsparser.units import LAMMPS_UNIT_CONVERSIONS

from pyiron_nodes.atomistic.calculator.data import InputCalcMD, InputCalcMinimize
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
    SetLammpsCalcInput,
    SetLammpsInputString,
    WriteFile,
)

AL_POTENTIAL = "1999--Mishin-Y--Al--LAMMPS--ipr1"
RESOURCE_PATH = os.environ.get(
    "IPRPY_RESOURCE_PATH",
    str(Path(sys.executable).parent.parent / "share" / "iprpy"),
)


def _bundle(calc_input=None, **kwargs):
    """An Al bundle with the given calc dataclass attached (or none)."""
    kwargs.setdefault("potential", AL_POTENTIAL)
    kwargs.setdefault("resource_path", RESOURCE_PATH)
    return LammpsIOBundle(
        structure=bulk("Al", cubic=True), calc_input=calc_input, **kwargs
    )


def _dataframe_potential():
    """A minimal one-row potential DataFrame in the potential-table layout.

    Uses lj/cut, which needs no parameter file, so it resolves without a
    resource path and exercises the DataFrame (`include potential.inp`) branch.
    """
    return pd.DataFrame(
        {
            "Name": ["Dummy--Al"],
            "Filename": [[]],
            "Model": ["Custom"],
            "Species": [["Al"]],
            "Config": [["pair_style lj/cut 5.0\n", "pair_coeff * * 0.1 2.5\n"]],
        }
    )


def _md(**kwargs):
    return InputCalcMD._original_dataclass(**kwargs)


def _minimize(**kwargs):
    return InputCalcMinimize._original_dataclass(**kwargs)


class TestLammpsInit(unittest.TestCase):
    def test_default_read_data(self):
        block = LammpsInit._original_func(io_bundle=_bundle())
        self.assertIn("units metal", block)
        self.assertIn("dimension 3", block)
        self.assertIn("boundary p p p", block)
        self.assertIn("atom_style atomic", block)
        self.assertIn("read_data lammps.data", block)
        self.assertNotIn("read_restart", block)

    def test_read_restart(self):
        block = LammpsInit._original_func(
            io_bundle=_bundle(), read_restart="restart.lammps"
        )
        self.assertIn("read_restart restart.lammps", block)
        self.assertNotIn("read_data", block)

    def test_units_atom_style_and_filename_from_bundle(self):
        bundle = _bundle(
            units="real", atom_style="full", lammps_structure_filename="structure.data"
        )
        block = LammpsInit._original_func(io_bundle=bundle)
        self.assertIn("units real", block)
        self.assertIn("atom_style full", block)
        self.assertIn("read_data structure.data", block)

    def test_boundary_from_structure_pbc(self):
        bundle = _bundle()
        bundle.structure.pbc = [True, True, False]
        block = LammpsInit._original_func(io_bundle=bundle)
        self.assertIn("dimension 2", block)
        self.assertIn("boundary p p f", block)


class TestLammpsPotential(unittest.TestCase):
    def test_inline_branch_for_string_potential(self):
        section, potential_file_content = LammpsPotential._original_func(
            io_bundle=_bundle()
        )
        self.assertIn("pair_style", section)
        self.assertEqual(potential_file_content, "")

    def test_include_branch_for_dataframe_potential(self):
        df_potential = _dataframe_potential()
        section, potential_file_content = LammpsPotential._original_func(
            io_bundle=_bundle(potential=df_potential, resource_path=None)
        )
        self.assertEqual(section, "include potential.inp\n")
        self.assertIn("pair_style", potential_file_content)

    def test_potential_filename_from_bundle(self):
        df_potential = _dataframe_potential()
        section, _ = LammpsPotential._original_func(
            io_bundle=_bundle(
                potential=df_potential,
                resource_path=None,
                lammps_potential_filename="mypot.inp",
            )
        )
        self.assertEqual(section, "include mypot.inp\n")


class TestLammpsDump(unittest.TestCase):
    def test_default(self):
        block = LammpsDump._original_func(io_bundle=_bundle())
        self.assertIn("variable dumptime equal 100", block)
        self.assertIn("dump 1 all custom ${dumptime} dump.out", block)
        # Regression: the rendered block must end with the closing '"' of the
        # dump_modify format string intact (core's _coerce() strips leading/
        # trailing quotes from string port values otherwise).
        self.assertTrue(block.rstrip("\n").endswith('"'))

    def test_custom_n_print_and_filename(self):
        block = LammpsDump._original_func(
            io_bundle=_bundle(), n_print=50, filename="traj.out"
        )
        self.assertIn("variable dumptime equal 50", block)
        self.assertIn("traj.out", block)

    def test_n_print_inherited_from_md_or_minimize_dataclass(self):
        for calc in (_md(n_print=7), _minimize(n_print=7)):
            block = LammpsDump._original_func(io_bundle=_bundle(calc))
            self.assertIn("variable dumptime equal 7", block)


class TestLammpsThermo(unittest.TestCase):
    def test_default(self):
        block = LammpsThermo._original_func(io_bundle=_bundle())
        self.assertIn("variable thermotime equal 100", block)
        self.assertIn("thermo ${thermotime}", block)

    def test_custom_n_print(self):
        block = LammpsThermo._original_func(io_bundle=_bundle(), n_print=25)
        self.assertIn("variable thermotime equal 25", block)

    def test_port_overrides_dataclass(self):
        block = LammpsThermo._original_func(
            io_bundle=_bundle(_md(n_print=7)), n_print=25
        )
        self.assertIn("variable thermotime equal 25", block)


class TestLammpsVelocity(unittest.TestCase):
    def test_default_metal_units(self):
        conversions = LAMMPS_UNIT_CONVERSIONS["metal"]
        block = LammpsVelocity._original_func(io_bundle=_bundle())
        expected_ts = 1.0 * conversions["time"]
        expected_temp = 2.0 * 300.0 * conversions["temperature"]
        self.assertIn(f"timestep {expected_ts}", block)
        self.assertIn(f"velocity all create {expected_temp} 42 dist gaussian", block)

    def test_custom_seed_and_temperature(self):
        conversions = LAMMPS_UNIT_CONVERSIONS["metal"]
        block = LammpsVelocity._original_func(
            io_bundle=_bundle(), temperature=600.0, seed=7
        )
        expected_temp = 2.0 * 600.0 * conversions["temperature"]
        self.assertIn(f"velocity all create {expected_temp} 7 dist gaussian", block)

    def test_inherited_from_md_dataclass(self):
        conversions = LAMMPS_UNIT_CONVERSIONS["metal"]
        calc = _md(temperature=500.0, seed=3, time_step=2.0)
        block = LammpsVelocity._original_func(io_bundle=_bundle(calc))
        self.assertIn(f"timestep {2.0 * conversions['time']}", block)
        expected_temp = 2.0 * 500.0 * conversions["temperature"]
        self.assertIn(f"velocity all create {expected_temp} 3 dist gaussian", block)

    def test_initial_temperature_replaces_2x_rule(self):
        conversions = LAMMPS_UNIT_CONVERSIONS["metal"]
        calc = _md(temperature=500.0, initial_temperature=123.0)
        block = LammpsVelocity._original_func(io_bundle=_bundle(calc))
        expected_temp = 123.0 * conversions["temperature"]
        self.assertIn(f"velocity all create {expected_temp} 42", block)

    def test_units_from_bundle(self):
        conversions = LAMMPS_UNIT_CONVERSIONS["real"]
        block = LammpsVelocity._original_func(io_bundle=_bundle(units="real"))
        self.assertIn(f"timestep {1.0 * conversions['time']}", block)


class TestLammpsMinimize(unittest.TestCase):
    def test_no_pressure(self):
        block = LammpsMinimize._original_func(
            io_bundle=_bundle(), e_tol=0.001, f_tol=0.001
        )
        self.assertIn("min_style cg", block)
        self.assertIn("minimize 0.001 0.001 1000000 1000000", block)
        self.assertNotIn("box/relax", block)
        self.assertNotIn("unfix ensemble", block)

    def test_defaults_from_input_calc_minimize(self):
        defaults = _minimize()
        block = LammpsMinimize._original_func(io_bundle=_bundle())
        self.assertIn(f"min_style {defaults.style}", block)
        self.assertIn(
            f"minimize {defaults.e_tol} {defaults.f_tol} "
            f"{defaults.max_iter} {defaults.max_iter}",
            block,
        )

    def test_pressure_zero_still_renders_box_relax(self):
        """Regression: pressure=0.0 must not be treated as falsy/unset."""
        conversions = LAMMPS_UNIT_CONVERSIONS["metal"]
        block = LammpsMinimize._original_func(
            io_bundle=_bundle(), e_tol=0.001, f_tol=0.001, pressure=0.0
        )
        expected_relax = 0.0 * conversions["pressure"]
        self.assertIn(f"fix ensemble all box/relax iso {expected_relax}", block)
        self.assertIn("unfix ensemble", block)

    def test_pressure_zero_inherited_from_dataclass(self):
        block = LammpsMinimize._original_func(
            io_bundle=_bundle(_minimize(pressure=0.0))
        )
        self.assertIn("box/relax", block)

    def test_pressure_nonzero(self):
        conversions = LAMMPS_UNIT_CONVERSIONS["metal"]
        block = LammpsMinimize._original_func(
            io_bundle=_bundle(), e_tol=0.001, f_tol=0.001, pressure=2.0
        )
        expected_relax = 2.0 * conversions["pressure"]
        self.assertIn(f"fix ensemble all box/relax iso {expected_relax}", block)

    def test_custom_style_and_max_iter(self):
        block = LammpsMinimize._original_func(
            io_bundle=_bundle(), e_tol=0.001, f_tol=0.001, style="fire", max_iter=500
        )
        self.assertIn("min_style fire", block)
        self.assertIn("minimize 0.001 0.001 500 500", block)

    def test_port_overrides_dataclass(self):
        calc = _minimize(e_tol=0.5, f_tol=0.5, style="sd")
        block = LammpsMinimize._original_func(io_bundle=_bundle(calc), f_tol=0.001)
        self.assertIn("min_style sd", block)
        self.assertIn("minimize 0.5 0.001", block)

    def test_md_dataclass_pressure_does_not_leak_into_minimize(self):
        block = LammpsMinimize._original_func(io_bundle=_bundle(_md(pressure=5.0)))
        self.assertNotIn("box/relax", block)


class TestLammpsEnsemble(unittest.TestCase):
    def test_nvt_by_default(self):
        conversions = LAMMPS_UNIT_CONVERSIONS["metal"]
        block, fix_ids = LammpsEnsemble._original_func(io_bundle=_bundle())
        t = 300.0 * conversions["temperature"]
        t_damp = 100.0 * conversions["time"]
        self.assertIn(f"fix ensemble all nvt temp {t} {t} {t_damp}", block)
        self.assertEqual(fix_ids, ["ensemble"])

    def test_nve_explicit(self):
        block, fix_ids = LammpsEnsemble._original_func(
            io_bundle=_bundle(), ensemble="nve"
        )
        self.assertIn("fix ensemble all nve", block)
        self.assertNotIn("nvt", block)
        self.assertNotIn("npt", block)
        self.assertEqual(fix_ids, ["ensemble"])

    def test_nve_when_dataclass_has_no_temperature_or_pressure(self):
        block, _ = LammpsEnsemble._original_func(
            io_bundle=_bundle(_md(temperature=None, pressure=None))
        )
        self.assertIn("fix ensemble all nve", block)

    def test_npt_when_pressure_set(self):
        block, fix_ids = LammpsEnsemble._original_func(
            io_bundle=_bundle(), temperature=300.0, pressure=0.0
        )
        self.assertIn("fix ensemble all npt", block)
        self.assertEqual(fix_ids, ["ensemble"])

    def test_npt_pressure_zero_is_not_treated_as_nvt(self):
        """pressure=0.0 must select the npt branch (`is not None` check), not
        fall through to nvt."""
        block, _ = LammpsEnsemble._original_func(io_bundle=_bundle(), pressure=0.0)
        self.assertNotIn("nvt", block)

    def test_inherited_from_md_dataclass(self):
        conversions = LAMMPS_UNIT_CONVERSIONS["metal"]
        calc = _md(temperature=500.0, pressure=1.0, pressure_damping_timescale=2000.0)
        block, _ = LammpsEnsemble._original_func(io_bundle=_bundle(calc))
        t = 500.0 * conversions["temperature"]
        p = 1.0 * conversions["pressure"]
        p_damp = 2000.0 * conversions["time"]
        self.assertIn(f"npt temp {t} {t}", block)
        self.assertIn(f"iso {p} {p} {p_damp}", block)

    def test_port_overrides_dataclass(self):
        conversions = LAMMPS_UNIT_CONVERSIONS["metal"]
        block, _ = LammpsEnsemble._original_func(
            io_bundle=_bundle(_md(temperature=500.0)), temperature=300.0
        )
        t = 300.0 * conversions["temperature"]
        self.assertIn(f"nvt temp {t} {t}", block)

    def test_nvt_override_ignores_inherited_pressure(self):
        block, _ = LammpsEnsemble._original_func(
            io_bundle=_bundle(_md(pressure=1.0)), ensemble="nvt"
        )
        self.assertIn("fix ensemble all nvt", block)

    def test_npt_without_pressure_raises(self):
        with self.assertRaises(ValueError):
            LammpsEnsemble._original_func(io_bundle=_bundle(), ensemble="npt")

    def test_unknown_ensemble_raises(self):
        with self.assertRaises(ValueError):
            LammpsEnsemble._original_func(io_bundle=_bundle(), ensemble="nph")

    def test_minimize_dataclass_is_ignored(self):
        block, _ = LammpsEnsemble._original_func(
            io_bundle=_bundle(_minimize(pressure=5.0))
        )
        self.assertIn("fix ensemble all nvt", block)

    def test_nve_langevin(self):
        block, fix_ids = LammpsEnsemble._original_func(
            io_bundle=_bundle(), langevin=True
        )
        self.assertIn("fix ensemble all nve", block)
        self.assertIn("fix langevin all langevin", block)
        self.assertEqual(fix_ids, ["ensemble", "langevin"])

    def test_nph_langevin(self):
        block, fix_ids = LammpsEnsemble._original_func(
            io_bundle=_bundle(_md(pressure=0.0, langevin=True))
        )
        self.assertIn("fix ensemble all nph", block)
        self.assertIn("fix langevin all langevin", block)
        self.assertEqual(fix_ids, ["ensemble", "langevin"])


class TestSetLammpsCalcInput(unittest.TestCase):
    def test_attaches_without_mutating_input(self):
        bundle = _bundle()
        calc = _md(temperature=500.0)
        result = SetLammpsCalcInput._original_func(io_bundle=bundle, calc_input=calc)
        self.assertEqual(result.calc_input.temperature, 500.0)
        self.assertIsNone(bundle.calc_input)

    def test_later_stage_dataclass_reaches_its_sections(self):
        stage1 = AssembleLammpsInput._original_func(
            io_bundle=_bundle(_minimize()), run=0
        )
        stage2 = SetLammpsCalcInput._original_func(
            io_bundle=stage1, calc_input=_md(temperature=None, pressure=None)
        )
        block, _ = LammpsEnsemble._original_func(io_bundle=stage2)
        self.assertIn("fix ensemble all nve", block)


class TestAssembleLammpsInputSingleStage(unittest.TestCase):
    def test_section_order(self):
        bundle = _bundle()
        init = LammpsInit._original_func(io_bundle=bundle)
        potential, _ = LammpsPotential._original_func(io_bundle=bundle)
        dump = LammpsDump._original_func(io_bundle=bundle)
        thermo = LammpsThermo._original_func(io_bundle=bundle)
        velocity = LammpsVelocity._original_func(io_bundle=bundle)
        minimize = LammpsMinimize._original_func(
            io_bundle=bundle, e_tol=0.001, f_tol=0.001
        )

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
        self.assertIn("dimension 3", s)
        self.assertIn("boundary p p p", s)
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
        bundle = _bundle()
        result = AssembleLammpsInput._original_func(
            io_bundle=bundle,
            init=LammpsInit._original_func(io_bundle=bundle),
            run=0,
        )
        s = result.lammps_input_string
        self.assertNotIn("dumptime", s)
        self.assertNotIn("thermotime", s)
        self.assertNotIn("timestep", s)
        self.assertNotIn("min_style", s)

    def test_write_restart_independent_of_run(self):
        result = AssembleLammpsInput._original_func(
            io_bundle=_bundle(), run=None, write_restart="foo.restart"
        )
        s = result.lammps_input_string
        self.assertIn("write_restart foo.restart", s)
        self.assertNotIn("run ", s)

    def test_potential_file_content_sets_potential_string(self):
        df_potential = _dataframe_potential()
        bundle = _bundle(potential=df_potential, resource_path=None)
        section, potential_file_content = LammpsPotential._original_func(
            io_bundle=bundle
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
    def test_second_call_appends_with_blank_line_separator(self):
        bundle = AssembleLammpsInput._original_func(io_bundle=_bundle(), run=0)
        first = bundle.lammps_input_string
        bundle = AssembleLammpsInput._original_func(io_bundle=bundle, run=10)
        second = bundle.lammps_input_string
        self.assertEqual(second, first + "\n" + "run 10\n")

    def test_init_and_potential_dropped_on_later_stage(self):
        bundle = _bundle()
        init = LammpsInit._original_func(io_bundle=bundle)
        bundle = AssembleLammpsInput._original_func(io_bundle=bundle, init=init, run=0)
        # Re-supplying init on a later stage should be a silent-but-noted no-op.
        bundle = AssembleLammpsInput._original_func(io_bundle=bundle, init=init, run=10)
        self.assertEqual(bundle.lammps_input_string.count("units metal"), 1)

    def test_unfix_rendered_after_run_in_same_call(self):
        bundle = _bundle()
        ensemble, fix_ids = LammpsEnsemble._original_func(io_bundle=bundle)
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
        bundle = _bundle()
        ensemble, fix_ids = LammpsEnsemble._original_func(io_bundle=bundle)
        # Stage 1 sets up the fix but does not run yet.
        bundle = AssembleLammpsInput._original_func(
            io_bundle=bundle, calculation=ensemble, calculation_fix_ids=fix_ids
        )
        self.assertEqual(bundle.lammps_pending_fix_ids, "ensemble")
        # Stage 2 supplies the run that actually uses the still-pending fix.
        bundle = AssembleLammpsInput._original_func(io_bundle=bundle, run=50)
        s = bundle.lammps_input_string
        self.assertIn("run 50", s)
        self.assertIn("unfix ensemble", s)
        self.assertLess(s.index("run 50"), s.index("unfix ensemble"))
        self.assertEqual(bundle.lammps_pending_fix_ids, "")

    def test_write_restart_without_run_keeps_fix_pending(self):
        """Regression: a write_restart-only stage must not unfix an ensemble
        fix that no run has used yet, or the later run has no integrator."""
        bundle = _bundle()
        ensemble, fix_ids = LammpsEnsemble._original_func(io_bundle=bundle)
        bundle = AssembleLammpsInput._original_func(
            io_bundle=bundle,
            calculation=ensemble,
            calculation_fix_ids=fix_ids,
            write_restart="setup.restart",
        )
        self.assertIn("write_restart setup.restart", bundle.lammps_input_string)
        self.assertNotIn("unfix ensemble", bundle.lammps_input_string)
        self.assertEqual(bundle.lammps_pending_fix_ids, "ensemble")

        bundle = AssembleLammpsInput._original_func(io_bundle=bundle, run=50)
        s = bundle.lammps_input_string
        self.assertLess(s.index("write_restart setup.restart"), s.index("run 50"))
        self.assertLess(s.index("run 50"), s.index("unfix ensemble"))
        self.assertEqual(s.count("unfix ensemble"), 1)
        self.assertEqual(bundle.lammps_pending_fix_ids, "")

    def test_new_calculation_with_abandoned_fix_raises(self):
        bundle = _bundle()
        ensemble1, fix_ids1 = LammpsEnsemble._original_func(io_bundle=bundle)
        bundle = AssembleLammpsInput._original_func(
            io_bundle=bundle, calculation=ensemble1, calculation_fix_ids=fix_ids1
        )
        ensemble2, fix_ids2 = LammpsEnsemble._original_func(
            io_bundle=bundle, temperature=350.0
        )
        with self.assertRaises(ValueError):
            AssembleLammpsInput._original_func(
                io_bundle=bundle, calculation=ensemble2, calculation_fix_ids=fix_ids2
            )


class TestSetLammpsInputString(unittest.TestCase):
    def test_sets_string_and_potential(self):
        result = SetLammpsInputString._original_func(
            io_bundle=_bundle(),
            lammps_input_string="units metal\n",
            potential_file_content="pair_style none\n",
        )
        self.assertEqual(result.lammps_input_string, "units metal\n")
        self.assertEqual(result.lammps_potential_string, "pair_style none\n")


class TestWriteFile(unittest.TestCase):
    def test_writes_input_string_to_disk(self):
        bundle = _bundle(lammps_input_string="units metal\n")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lmp.in")
            result_path = WriteFile._original_func(io_bundle=bundle, filename=path)
            self.assertEqual(result_path, path)
            with open(path) as f:
                self.assertEqual(f.read(), "units metal\n")


if __name__ == "__main__":
    unittest.main()
