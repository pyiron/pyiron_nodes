"""Electrode constraints, force units, and the piston convergence check."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms, FixedPlane

from lammpsparser.compatibility.constraints import (
    _get_fixed_atom_boolean_vector,
    set_selective_dynamics,
)
from lammpsparser.units import LAMMPS_UNIT_CONVERSIONS

from pyiron_nodes.atomistic.engine.lammps import LammpsIOBundle, ParseElectrodeForce
from pyiron_nodes.electrochemistry.analysis.pressure import (
    ElectrodePressure,
    PistonPressureConvergence,
)
from pyiron_nodes.electrochemistry.structure.build import FixElectrodes

# kcal/mol/Å per eV/Å, the factor `units real` puts on every force.  Taken from
# lammpsparser rather than written out, so the test tracks the conversion the
# node actually applies.
REAL_FORCE_PER_EV = LAMMPS_UNIT_CONVERSIONS["real"]["force"]


def _cell():
    """Al / H2O / Ne along z, the shape the workflow builds."""
    return Atoms(
        symbols=["Al", "Al", "O", "H", "H", "Ne", "Ne"],
        positions=[
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [1.0, 1.0, 5.0],
            [1.0, 1.8, 5.6],
            [1.0, 0.2, 5.6],
            [0.0, 0.0, 10.0],
            [2.0, 2.0, 10.0],
        ],
        cell=[4.0, 4.0, 30.0],
        pbc=[True, True, False],
    )


class TestFixElectrodes(unittest.TestCase):
    def test_without_a_piston_everything_listed_is_frozen(self):
        result = FixElectrodes._original_func(
            structure=_cell(), fixed_species='["Al", "Ne"]'
        )

        frozen = _get_fixed_atom_boolean_vector(result)
        np.testing.assert_array_equal(frozen[[0, 1, 5, 6]], True)
        np.testing.assert_array_equal(frozen[[2, 3, 4]], False)

    def test_the_piston_species_is_pinned_in_xy_and_free_in_z(self):
        result = FixElectrodes._original_func(
            structure=_cell(), fixed_species='["Al", "Ne"]', piston_species="Ne"
        )

        frozen = _get_fixed_atom_boolean_vector(result)
        # Al: fully frozen reference electrode
        np.testing.assert_array_equal(frozen[[0, 1]], [[True] * 3] * 2)
        # Ne: the piston, free along z
        np.testing.assert_array_equal(frozen[[5, 6]], [[True, True, False]] * 2)
        # Water untouched
        np.testing.assert_array_equal(frozen[[2, 3, 4]], False)

    def test_it_becomes_the_expected_lammps_commands(self):
        result = FixElectrodes._original_func(
            structure=_cell(), fixed_species='["Al", "Ne"]', piston_species="Ne"
        )

        commands = set_selective_dynamics(structure=result, calc_md=True)

        self.assertEqual(
            commands["fix constraintxyz"], "constraintxyz setforce 0.0 0.0 0.0"
        )
        self.assertEqual(
            commands["fix constraintxy"], "constraintxy setforce 0.0 0.0 NULL"
        )
        self.assertEqual(commands["group constraintxy"], "id 6 7")

    def test_a_single_symbol_is_accepted_like_FixSpecies(self):
        result = FixElectrodes._original_func(structure=_cell(), fixed_species="Al")

        frozen = _get_fixed_atom_boolean_vector(result)
        np.testing.assert_array_equal(frozen[[0, 1]], True)
        np.testing.assert_array_equal(frozen[5:], False)

    def test_the_input_structure_is_not_modified(self):
        structure = _cell()
        FixElectrodes._original_func(structure=structure, piston_species="Ne")

        self.assertEqual(len(structure.constraints), 0)

    def test_an_absent_piston_species_is_an_error(self):
        with self.assertRaises(ValueError) as ctx:
            FixElectrodes._original_func(structure=_cell(), piston_species="Xe")
        self.assertIn("Xe", str(ctx.exception))


class TestParseElectrodeForceUnits(unittest.TestCase):
    """`fix ave/time` writes in the run's own units.  The TIP3P potential puts
    the run in `units real`, so an unconverted read is off by ~23."""

    def _bundle_with_force_file(self, directory, units, fz):
        Path(directory).mkdir(parents=True, exist_ok=True)
        Path(directory, "electrode_force_Ne.txt").write_text(
            "# step fx fy fz\n"
            + "\n".join(f"{i} 0.0 0.0 {v!r}" for i, v in enumerate(fz))
        )
        return LammpsIOBundle(
            structure=_cell(),
            potential="dummy",
            working_directory=directory,
            units=units,
        )

    def test_metal_units_pass_through_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._bundle_with_force_file(tmp, "metal", [1.0, 2.0])
            forces = ParseElectrodeForce._original_func(io_bundle=bundle)

        np.testing.assert_allclose(forces["Ne"]["fz"], [1.0, 2.0])

    def test_real_units_are_converted_to_eV_per_angstrom(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = [REAL_FORCE_PER_EV, 2 * REAL_FORCE_PER_EV]
            bundle = self._bundle_with_force_file(tmp, "real", raw)
            forces = ParseElectrodeForce._original_func(io_bundle=bundle)

        np.testing.assert_allclose(forces["Ne"]["fz"], [1.0, 2.0], rtol=1e-12)

    def test_steps_stay_integers(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._bundle_with_force_file(tmp, "real", [1.0, 2.0, 3.0])
            forces = ParseElectrodeForce._original_func(io_bundle=bundle)

        np.testing.assert_array_equal(forces["Ne"]["steps"], [0, 1, 2])

    def test_a_single_row_file_is_still_two_dimensional(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._bundle_with_force_file(tmp, "metal", [5.0])
            forces = ParseElectrodeForce._original_func(io_bundle=bundle)

        np.testing.assert_allclose(forces["Ne"]["fz"], [5.0])

    def test_no_files_gives_an_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = LammpsIOBundle(
                structure=_cell(),
                potential="dummy",
                working_directory=tmp,
                units="real",
            )
            self.assertEqual(ParseElectrodeForce._original_func(io_bundle=bundle), {})


class TestPistonPressureConvergence(unittest.TestCase):
    """At mechanical equilibrium both electrodes read the applied load.  The
    lower electrode is pushed toward -z and the upper one toward +z, so the raw
    forces have opposite signs for the same compressive pressure."""

    AREA = 16.0  # Å², the 4x4 cell above
    EV_PER_ANG3_TO_BAR = 1.602176634e11 / 1e5

    def _forces_for(self, pressure_bar, n=50):
        """Electrode forces that correspond to a given compressive pressure."""
        fz = pressure_bar / self.EV_PER_ANG3_TO_BAR * self.AREA
        return {
            "Al": {"steps": np.arange(n), "fz": np.full(n, -fz)},  # lower: -z
            "Ne": {"steps": np.arange(n), "fz": np.full(n, +fz)},  # upper: +z
        }

    def test_opposite_raw_forces_give_the_same_compressive_pressure(self):
        pressures, expected, imbalance, converged = (
            PistonPressureConvergence._original_func(
                initial_structure=_cell(),
                electrode_forces=self._forces_for(100.0),
                target_pressure=100.0,
            )
        )

        self.assertAlmostEqual(pressures["Al"]["mean"], 100.0, places=6)
        self.assertAlmostEqual(pressures["Ne"]["mean"], 100.0, places=6)
        self.assertAlmostEqual(imbalance, 0.0, places=9)
        self.assertTrue(converged)

    def test_missing_the_target_is_not_converged(self):
        _, _, _, converged = PistonPressureConvergence._original_func(
            initial_structure=_cell(),
            electrode_forces=self._forces_for(100.0),
            target_pressure=1.0,
        )

        self.assertFalse(converged)

    def test_disagreeing_electrodes_show_up_as_imbalance(self):
        forces = self._forces_for(100.0)
        forces["Ne"]["fz"] = forces["Ne"]["fz"] * 2.0

        pressures, expected, imbalance, converged = (
            PistonPressureConvergence._original_func(
                initial_structure=_cell(),
                electrode_forces=forces,
                target_pressure=100.0,
            )
        )

        self.assertAlmostEqual(pressures["Ne"]["mean"], 200.0, places=6)
        self.assertAlmostEqual(imbalance, 100.0, places=6)
        self.assertFalse(converged)

    def test_a_zero_target_does_not_make_every_run_fail(self):
        # The tolerance is relative, so it has to fall back to an absolute
        # scale when the target is 0 bar.
        _, _, _, converged = PistonPressureConvergence._original_func(
            initial_structure=_cell(),
            electrode_forces=self._forces_for(0.05),
            target_pressure=0.0,
            tolerance=0.1,
        )

        self.assertTrue(converged)

    def test_n_skip_drops_the_equilibration_frames(self):
        forces = self._forces_for(100.0, n=20)
        # First 10 frames wildly off; they must not reach the mean.
        forces["Al"]["fz"][:10] = 0.0
        forces["Ne"]["fz"][:10] = 0.0

        pressures, _, _, converged = PistonPressureConvergence._original_func(
            initial_structure=_cell(),
            electrode_forces=forces,
            target_pressure=100.0,
            n_skip=10,
        )

        self.assertAlmostEqual(pressures["Al"]["mean"], 100.0, places=6)
        self.assertTrue(converged)

    def test_charged_plates_raise_the_expected_pressure_by_the_maxwell_stress(self):
        # compute group/group only sees the electrolyte, but the piston also
        # feels the counter-electrode pulling it in.  At equilibrium the
        # electrolyte therefore has to push harder than the applied load.
        charges = {
            "metal": "Al",
            "metal_charge": 0.5,
            "neon_charge": -0.5,
            "cation": "Na",
            "cation_charge": 1.0,
            "anion": "F",
            "anion_charge": -1.0,
            "O_charge": -0.830,
            "H_charge": 0.415,
        }
        # 2 Ne atoms x 0.5 e over a 16 Å² plate
        sigma = 2 * 0.5 * 1.602176634e-19 / (16.0 * 1e-20)  # C/m²
        maxwell_bar = sigma**2 / (2 * 8.854187817e-12) / 1e5

        _, expected, _, _ = PistonPressureConvergence._original_func(
            initial_structure=_cell(),
            electrode_forces=self._forces_for(1.0),
            target_pressure=1.0,
            charges=charges,
        )

        self.assertAlmostEqual(expected, 1.0 + maxwell_bar, places=6)

    def test_without_charges_the_expected_pressure_is_just_the_load(self):
        _, expected, _, converged = PistonPressureConvergence._original_func(
            initial_structure=_cell(),
            electrode_forces=self._forces_for(100.0),
            target_pressure=100.0,
        )

        self.assertAlmostEqual(expected, 100.0, places=9)
        self.assertTrue(converged)

    def test_an_uncharged_cell_adds_nothing(self):
        charges = {
            "metal": "Al",
            "metal_charge": 0.0,
            "neon_charge": 0.0,
            "cation": "Na",
            "cation_charge": 1.0,
            "anion": "F",
            "anion_charge": -1.0,
            "O_charge": -0.830,
            "H_charge": 0.415,
        }

        _, expected, _, _ = PistonPressureConvergence._original_func(
            initial_structure=_cell(),
            electrode_forces=self._forces_for(100.0),
            target_pressure=100.0,
            charges=charges,
        )

        self.assertAlmostEqual(expected, 100.0, places=9)

    def test_an_unknown_electrode_is_an_error(self):
        with self.assertRaises(KeyError) as ctx:
            PistonPressureConvergence._original_func(
                initial_structure=_cell(),
                electrode_forces={"Al": {"fz": np.zeros(5)}},
                electrodes='["Al", "Ne"]',
            )
        self.assertIn("Ne", str(ctx.exception))

    def test_it_agrees_with_ElectrodePressure_up_to_the_sign(self):
        forces = self._forces_for(100.0)

        _, mean_pressure, _ = ElectrodePressure._original_func(
            initial_structure=_cell(),
            electrode_forces=forces,
            electrode="Ne",
        )
        pressures, _, _, _ = PistonPressureConvergence._original_func(
            initial_structure=_cell(),
            electrode_forces=forces,
            target_pressure=100.0,
        )

        # ElectrodePressure reports Pa, this node bar.
        self.assertAlmostEqual(mean_pressure / 1e5, pressures["Ne"]["mean"], places=6)


if __name__ == "__main__":
    unittest.main()
