"""Analytic checks on the dielectric and capacitance nodes.

Each test feeds a profile or trajectory whose answer is known in closed form,
so a failure points at the formula rather than at the physics of some run.
"""

import unittest

import numpy as np
from ase import Atoms

from pyiron_nodes.electrochemistry.analysis.dielectric import (
    DielectricFromPolarizationFluctuations,
    DifferentialCapacitance,
    LocalDielectricProfile,
    _charge_map,
    _free_bound_maps,
    _poisson_potential,
    _resolve_initial_step,
    EPSILON_0,
    E_CHARGE,
    K_BOLTZMANN,
    ANG_TO_M,
)
from pyiron_nodes.electrochemistry.analysis.plots import BuildDensityContext

CHARGES = {
    "metal": "Al",
    "metal_charge": 0.1,
    "neon_charge": -0.1,
    "cation": "Na",
    "cation_charge": 1.0,
    "anion": "F",
    "anion_charge": -1.0,
    "O_charge": -0.830,
    "H_charge": 0.415,
    "quasi_2d": True,
}

AREA = 100.0  # Å², a 10x10 cross-section keeps the arithmetic readable


def _structure(n_frames_cell=None):
    """A cell whose only job is to carry the xy cross-section area."""
    atoms = Atoms("Al", positions=[[0.0, 0.0, 0.0]])
    atoms.set_cell([10.0, 10.0, 60.0])
    return atoms


def _context(density_data, initial_step=0):
    return BuildDensityContext._original_func(
        density_data=density_data,
        initial_structure=_structure(),
        charges=CHARGES,
        initial_step=initial_step,
    )


class TestChargeMaps(unittest.TestCase):
    def test_the_charge_map_covers_every_species(self):
        mapping = _charge_map(CHARGES)

        self.assertEqual(set(mapping), {"Al", "Ne", "Na", "F", "O", "H"})
        self.assertEqual(mapping["Al"], 0.1)
        self.assertEqual(mapping["O"], -0.830)

    def test_water_is_the_bound_charge_and_nothing_else(self):
        free, bound = _free_bound_maps(CHARGES)

        self.assertEqual(set(bound), {"O", "H"})
        self.assertEqual(set(free), {"Al", "Ne", "Na", "F"})
        # Together they are the full map: no species may fall through the split,
        # or the total field is wrong.
        self.assertEqual({**free, **bound}, _charge_map(CHARGES))


class TestPoissonPotential(unittest.TestCase):
    def test_a_uniform_slab_gives_a_linear_field(self):
        # rho = const -> E grows linearly, Psi falls quadratically.
        n_bins, dz = 100, 0.5
        rho = np.full(n_bins, 1e-3)  # e/Å³

        e_field, psi = _poisson_potential(rho, dz)

        expected_slope = 1e-3 * E_CHARGE / ANG_TO_M**3 * dz * ANG_TO_M / EPSILON_0
        self.assertAlmostEqual(float(e_field[0] / ANG_TO_M), expected_slope, places=6)
        np.testing.assert_allclose(
            np.diff(e_field / ANG_TO_M), expected_slope, rtol=1e-10
        )
        # Psi = -int E dz, so it is monotonically decreasing for positive rho.
        self.assertTrue(np.all(np.diff(psi) < 0))

    def test_it_broadcasts_over_frames(self):
        rho = np.random.default_rng(0).normal(size=(7, 20)) * 1e-4

        e_field, psi = _poisson_potential(rho, 0.5)

        self.assertEqual(e_field.shape, (7, 20))
        self.assertEqual(psi.shape, (7, 20))
        # Each frame is integrated independently.
        single, _ = _poisson_potential(rho[3], 0.5)
        np.testing.assert_allclose(e_field[3], single)


class TestResolveInitialStep(unittest.TestCase):
    def test_an_explicit_value_wins(self):
        self.assertEqual(_resolve_initial_step({"initial_step": 10}, 3), 3)

    def test_none_falls_back_to_the_context(self):
        self.assertEqual(_resolve_initial_step({"initial_step": 10}, None), 10)

    def test_a_context_without_the_key_still_loads(self):
        # Contexts stored before initial_step existed must not break.
        self.assertEqual(_resolve_initial_step({}, None), 0)


class TestLocalDielectricProfile(unittest.TestCase):
    """A parallel-plate capacitor with a uniformly polarized slab between the
    plates has a dielectric constant that is exact, not approximate."""

    def _uniform_dielectric_data(self, eps=20.0, n_bins=200):
        """Build a density profile whose answer is ``eps`` by construction.

        A parallel-plate capacitor: +σ of free charge on the metal, −σ on the
        Ne plate, and a polarized water film between them whose bound surface
        charge screens the field down by exactly 1/eps.  The bound charge is
        laid down the way a real film produces it — oxygens (negative) drawn
        toward the positive plate, hydrogens (positive) toward the negative one
        — so every entry stays a non-negative atom count.
        """
        z = np.linspace(0.0, 40.0, n_bins)
        dz = z[1] - z[0]
        v_bin = AREA * dz

        metal = np.zeros(n_bins)
        neon = np.zeros(n_bins)
        oxygen = np.zeros(n_bins)
        hydrogen = np.zeros(n_bins)

        # Free charge: +sigma at the first bin, -sigma at the last.
        sigma_atoms = 1.0  # e per bin, in "average atom count" units
        metal[0] = sigma_atoms / CHARGES["metal_charge"]
        neon[-1] = -sigma_atoms / CHARGES["neon_charge"]

        # Bound charge: -sigma_b just inside the positive plate and +sigma_b
        # just inside the negative one, leaving sigma/eps of net charge enclosed
        # anywhere in the bulk of the film.
        bound_sigma = sigma_atoms * (1.0 - 1.0 / eps)
        oxygen[1] = -bound_sigma / CHARGES["O_charge"]
        hydrogen[-2] = bound_sigma / CHARGES["H_charge"]

        for counts in (metal, neon, oxygen, hydrogen):
            assert np.all(counts >= 0), "atom counts must stay non-negative"

        return {
            "Al": (z, metal),
            "Ne": (z, neon),
            "O": (z, oxygen),
            "H": (z, hydrogen),
            "Na": (z, np.zeros(n_bins)),
            "F": (z, np.zeros(n_bins)),
        }, v_bin

    def test_it_recovers_a_known_dielectric_constant(self):
        eps_true = 20.0
        data, _ = self._uniform_dielectric_data(eps=eps_true)

        _, _, eps_water = LocalDielectricProfile._original_func(context=_context(data))

        self.assertAlmostEqual(eps_water, eps_true, places=6)

    def test_vacuum_between_the_plates_gives_one(self):
        data, _ = self._uniform_dielectric_data(eps=1.0)

        _, eps_profile, eps_water = LocalDielectricProfile._original_func(
            context=_context(data)
        )

        self.assertAlmostEqual(eps_water, 1.0, places=6)

    def test_an_uncharged_cell_is_rejected(self):
        data, _ = self._uniform_dielectric_data()
        charges = dict(CHARGES, metal_charge=0.0, neon_charge=0.0)
        context = BuildDensityContext._original_func(
            density_data=data,
            initial_structure=_structure(),
            charges=charges,
            initial_step=0,
        )

        with self.assertRaises(ValueError) as ctx:
            LocalDielectricProfile._original_func(context=context)
        self.assertIn("charged electrode", str(ctx.exception))

    def test_poles_are_masked_rather_than_infinite(self):
        data, _ = self._uniform_dielectric_data(eps=20.0)

        _, eps_profile, _ = LocalDielectricProfile._original_func(
            context=_context(data), eps_clip=5.0
        )

        self.assertFalse(np.any(np.isinf(eps_profile)))
        finite = eps_profile[np.isfinite(eps_profile)]
        self.assertTrue(np.all(np.abs(finite) <= 5.0))


class _Trajectory:
    """Minimal stand-in for OutputCalcMD."""

    def __init__(self, species, positions, steps=None):
        self.species = list(species)
        self.positions = np.asarray(positions)
        self.steps = (
            np.arange(self.positions.shape[0]) if steps is None else np.asarray(steps)
        )


def _slab_trajectory(n_frames=400, n_water=60, seed=0, drop_amplitude=0.0):
    """Al plane at z=0, Ne plane at z=30, water sloshing in between."""
    rng = np.random.default_rng(seed)
    species = ["Al", "Al"] + ["O", "H", "H"] * n_water + ["Ne", "Ne"]
    n_atoms = len(species)

    positions = np.zeros((n_frames, n_atoms, 3))
    positions[:, 0] = [0.0, 0.0, 0.0]
    positions[:, 1] = [5.0, 5.0, 0.0]
    positions[:, -2] = [0.0, 0.0, 30.0]
    positions[:, -1] = [5.0, 5.0, 30.0]

    for mol in range(n_water):
        base = 2 + 3 * mol
        z0 = 2.0 + 26.0 * (mol + 0.5) / n_water
        jitter = rng.normal(scale=0.4, size=n_frames)
        shift = drop_amplitude * rng.normal(size=n_frames)
        positions[:, base, 2] = z0 + jitter + shift
        positions[:, base + 1, 2] = z0 + jitter + shift + 0.3
        positions[:, base + 2, 2] = z0 + jitter + shift - 0.3
        positions[:, base : base + 3, 0] = rng.uniform(0, 10, size=(n_frames, 3))
        positions[:, base : base + 3, 1] = rng.uniform(0, 10, size=(n_frames, 3))

    return _Trajectory(species, positions)


class TestDielectricFromPolarizationFluctuations(unittest.TestCase):
    def _context_for(self, trajectory):
        z = np.linspace(0.0, 30.0, 50)
        zeros = np.zeros_like(z)
        data = {
            "Al": (z, zeros),
            "Ne": (z, zeros),
            "O": (z, zeros),
            "H": (z, zeros),
            "Na": (z, zeros),
            "F": (z, zeros),
        }
        return _context(data, initial_step=0)

    def test_a_rigid_frozen_film_has_no_fluctuations_so_eps_is_one(self):
        # Every frame identical -> zero covariance -> 1/eps = 1.
        trajectory = _slab_trajectory(n_frames=20)
        trajectory.positions = np.repeat(
            trajectory.positions[:1], trajectory.positions.shape[0], axis=0
        )

        _, _, eps_water = DielectricFromPolarizationFluctuations._original_func(
            trajectory=trajectory, context=self._context_for(trajectory), n_bins=50
        )

        self.assertAlmostEqual(eps_water, 1.0, places=9)

    def test_fluctuating_water_screens(self):
        trajectory = _slab_trajectory(n_frames=400, drop_amplitude=1.0)

        _, eps_profile, eps_water = (
            DielectricFromPolarizationFluctuations._original_func(
                trajectory=trajectory,
                context=self._context_for(trajectory),
                n_bins=50,
            )
        )

        self.assertTrue(np.isfinite(eps_water))
        # A polarizable medium screens: eps > 1 somewhere in the film.
        finite = eps_profile[np.isfinite(eps_profile)]
        self.assertTrue(np.any(finite > 1.0))

    def test_a_single_frame_is_rejected(self):
        trajectory = _slab_trajectory(n_frames=1)

        with self.assertRaises(ValueError) as ctx:
            DielectricFromPolarizationFluctuations._original_func(
                trajectory=trajectory, context=self._context_for(trajectory)
            )
        self.assertIn("at least 2 frames", str(ctx.exception))

    def test_initial_step_defaults_to_the_context(self):
        trajectory = _slab_trajectory(n_frames=30, drop_amplitude=1.0)
        z = np.linspace(0.0, 30.0, 50)
        zeros = np.zeros_like(z)
        data = {k: (z, zeros) for k in ("Al", "Ne", "O", "H", "Na", "F")}

        from_context = DielectricFromPolarizationFluctuations._original_func(
            trajectory=trajectory, context=_context(data, initial_step=10), n_bins=50
        )
        explicit = DielectricFromPolarizationFluctuations._original_func(
            trajectory=trajectory,
            context=_context(data, initial_step=0),
            initial_step=10,
            n_bins=50,
        )

        np.testing.assert_allclose(from_context[1], explicit[1], equal_nan=True)


class TestDifferentialCapacitance(unittest.TestCase):
    def _context(self):
        z = np.linspace(0.0, 30.0, 50)
        zeros = np.zeros_like(z)
        data = {k: (z, zeros) for k in ("Al", "Ne", "O", "H", "Na", "F")}
        return _context(data, initial_step=0)

    def test_it_returns_the_fluctuation_formula(self):
        trajectory = _slab_trajectory(n_frames=500, drop_amplitude=1.0, seed=3)
        context = self._context()

        C_metal, C_Ne, drops = DifferentialCapacitance._original_func(
            trajectory=trajectory, context=context, n_bins=60, temperature=300.0
        )

        area_m2 = AREA * ANG_TO_M**2
        kT = K_BOLTZMANN * 300.0
        for name, C in (("Al", C_metal), ("Ne", C_Ne)):
            expected = kT / (area_m2 * np.var(drops[name])) * 100.0
            self.assertAlmostEqual(C / expected, 1.0, places=9)

    def test_the_drops_carry_the_steps_and_both_electrodes(self):
        trajectory = _slab_trajectory(n_frames=50, drop_amplitude=1.0)

        _, _, drops = DifferentialCapacitance._original_func(
            trajectory=trajectory, context=self._context(), n_bins=60
        )

        self.assertEqual(set(drops), {"steps", "Al", "Ne"})
        self.assertEqual(len(drops["Al"]), 50)
        np.testing.assert_array_equal(drops["steps"], np.arange(50))

    def test_zero_variance_is_nan_not_a_division_by_zero(self):
        trajectory = _slab_trajectory(n_frames=10)
        trajectory.positions = np.repeat(trajectory.positions[:1], 10, axis=0)

        C_metal, C_Ne, _ = DifferentialCapacitance._original_func(
            trajectory=trajectory, context=self._context(), n_bins=60
        )

        self.assertTrue(np.isnan(C_metal))
        self.assertTrue(np.isnan(C_Ne))

    def test_capacitance_is_linear_in_temperature(self):
        # C = kT / (A Var), and Var is a property of the trajectory alone, so
        # doubling T must double C exactly.
        trajectory = _slab_trajectory(n_frames=300, drop_amplitude=1.0, seed=5)
        context = self._context()

        C_300, _, _ = DifferentialCapacitance._original_func(
            trajectory=trajectory, context=context, n_bins=60, temperature=300.0
        )
        C_600, _, _ = DifferentialCapacitance._original_func(
            trajectory=trajectory, context=context, n_bins=60, temperature=600.0
        )

        self.assertAlmostEqual(C_600 / C_300, 2.0, places=9)

    def test_a_single_frame_is_rejected(self):
        trajectory = _slab_trajectory(n_frames=1)

        with self.assertRaises(ValueError) as ctx:
            DifferentialCapacitance._original_func(
                trajectory=trajectory, context=self._context()
            )
        self.assertIn("at least 2 frames", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
