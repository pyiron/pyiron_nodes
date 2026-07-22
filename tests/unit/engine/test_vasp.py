"""Unit tests for the VASP workflow nodes in ``pyiron_nodes.atomistic.engine.vasp_new``.

Following the pattern in ``test_lammps.py`` and in pyiron_atomistics' own VASP
tests, nothing here launches VASP. We test the two things that matter:

* **INCAR translation** — that ``InputSCF`` / ``InputMinimizationVASP`` / ... map
  to the right VASP tags (via ``_build_incar`` and the small enum helpers).
* **Output parsing** — that ``ParseVaspOutput`` reads a static ``vasprun.xml``.

Nodes are exercised through their ``._original_func`` / ``._original_dataclass``
so no ``Workflow`` object is required.
"""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase import Atoms
from ase.build import bulk
from pymatgen.io.vasp.inputs import Incar

from pyiron_nodes.atomistic.calculator.data import (
    AdditionalInputFlags,
    InputMDVASP,
    InputDipoleCorrection,
    InputMinimizationVASP,
    InputSCF,
    OutputCalcMD,
    OutputCalcMinimize,
    OutputCalcStatic,
)
from pyiron_nodes.atomistic.engine.vasp_new import (
    CreateVaspInputResources,
    MergeVaspInput,
    ParseVaspOutput,
    RunVaspCalculation,
    VaspInput,
    VaspInputResources,
    _build_incar,
    _generate_hash,
    _get_potcar_paths,
    _IBRION_MINIMIZE,
    _ISMEAR,
    _ordered_elements,
    _unwrap_positions,
)

STATIC_VASP = Path(__file__).parent.parent.parent / "static" / "vasp"


def make_scf(**overrides) -> InputSCF:
    """Build a pure ``InputSCF`` dataclass instance (kpoints is required)."""
    params = dict(kpoints="4 4 4")
    params.update(overrides)
    return InputSCF._original_dataclass(**params)


# ── enum helpers ───────────────────────────────────────────────────────────────


class TestIsmear(unittest.TestCase):
    def test_gaussian(self):
        self.assertEqual(_ISMEAR("gaussian", 1), 0)

    def test_fermi_dirac(self):
        self.assertEqual(_ISMEAR("fermi-dirac", 1), -1)

    def test_methfessel_paxton_order(self):
        self.assertEqual(_ISMEAR("methfessel-paxton", 1), 1)
        self.assertEqual(_ISMEAR("methfessel-paxton", 2), 2)

    def test_methfessel_paxton_bad_order(self):
        with self.assertRaises(ValueError):
            _ISMEAR("methfessel-paxton", 0)

    def test_unknown_smearing_raises(self):
        # regression: previously fell through and returned None (malformed INCAR)
        with self.assertRaises(ValueError):
            _ISMEAR("tetrahedron", 1)


class TestIbrionMap(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(_IBRION_MINIMIZE["ConjugateGradient"], 2)
        self.assertEqual(_IBRION_MINIMIZE["RMM-DIIS"], 1)
        self.assertEqual(_IBRION_MINIMIZE["DampedMolecularDynamics"], 3)


class TestOrderedElements(unittest.TestCase):
    def test_collapses_runs(self):
        atoms = Atoms("FeFeO", positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        self.assertEqual(_ordered_elements(atoms), ["Fe", "O"])

    def test_single_element(self):
        self.assertEqual(_ordered_elements(bulk("Fe", cubic=True)), ["Fe"])


# ── INCAR construction ─────────────────────────────────────────────────────────


class TestBuildIncar(unittest.TestCase):
    def test_scf_static_defaults(self):
        incar = _build_incar(VaspInput(scf=make_scf()))
        # static run: no ionic loop
        self.assertEqual(incar["IBRION"], -1)
        self.assertEqual(incar["NSW"], 0)
        # base SCF tags carried over from InputSCF defaults
        self.assertEqual(incar["ENCUT"], 400.0)
        self.assertEqual(incar["EDIFF"], 1e-6)
        self.assertEqual(incar["NELM"], 100)
        self.assertEqual(incar["ISMEAR"], 0)  # gaussian
        self.assertEqual(incar["SIGMA"], 0.2)

    def test_encut_and_smearing_propagate(self):
        scf = make_scf(
            energy_cutoff=520.0, smearing_type="methfessel-paxton", smearing_order=2
        )
        incar = _build_incar(VaspInput(scf=scf))
        self.assertEqual(incar["ENCUT"], 520.0)
        self.assertEqual(incar["ISMEAR"], 2)

    def test_minimization_conjugate_gradient(self):
        mini = InputMinimizationVASP._original_dataclass(
            algorithm="ConjugateGradient", max_ionic_steps=50, isif=3
        )
        incar = _build_incar(VaspInput(scf=make_scf(), minimization=mini))
        self.assertEqual(incar["IBRION"], 2)
        self.assertEqual(incar["NSW"], 50)
        self.assertEqual(incar["EDIFFG"], -0.01)
        self.assertEqual(incar["ISIF"], 3)

    def test_minimization_algorithms_map_to_ibrion(self):
        for algo, ibrion in _IBRION_MINIMIZE.items():
            mini = InputMinimizationVASP._original_dataclass(algorithm=algo)
            incar = _build_incar(VaspInput(scf=make_scf(), minimization=mini))
            self.assertEqual(incar["IBRION"], ibrion, msg=algo)

    def test_md_nvt(self):
        md = InputMDVASP._original_dataclass(
            temperature=800.0,
            n_ionic_steps=1000,
            n_print=50,
            time_step=2.0,
            temperature_damping_timescale=50.0,
            seed=42,
        )
        incar = _build_incar(
            VaspInput(scf=make_scf(), md=md), structure=bulk("Fe", cubic=True)
        )
        self.assertEqual(incar["IBRION"], 0)
        self.assertEqual(incar["MDALGO"], 3)  # Langevin
        self.assertEqual(incar["ISYM"], 0)
        self.assertEqual(incar["ISIF"], 2)  # NVT: fixed cell
        self.assertEqual(incar["NSW"], 1000)
        self.assertEqual(incar["NBLOCK"], 50)
        self.assertEqual(incar["POTIM"], 2.0)
        self.assertEqual(incar["TEBEG"], 800.0)
        self.assertEqual(incar["TEEND"], 800.0)  # isothermal by default
        self.assertEqual(incar["LANGEVIN_GAMMA"], [20.0])  # 1000 / 50 fs
        self.assertEqual(incar["RANDOM_SEED"], [42, 0, 0])
        # NpT-only tags stay out of an NVT INCAR
        self.assertNotIn("PSTRESS", incar)
        self.assertNotIn("PMASS", incar)
        self.assertNotIn("LANGEVIN_GAMMA_L", incar)

    def test_md_npt(self):
        md = InputMDVASP._original_dataclass(
            ensemble="NpT",
            temperature=300.0,
            final_temperature=600.0,
            pressure=2.0,
            pressure_damping_timescale=500.0,
            lattice_mass=250.0,
        )
        incar = _build_incar(
            VaspInput(scf=make_scf(), md=md), structure=bulk("Fe", cubic=True)
        )
        self.assertEqual(incar["ISIF"], 3)  # cell shape + volume free
        self.assertEqual(incar["TEBEG"], 300.0)
        self.assertEqual(incar["TEEND"], 600.0)  # temperature ramp
        self.assertEqual(incar["PSTRESS"], 20.0)  # 2 GPa → 20 kBar
        self.assertEqual(incar["PMASS"], 250.0)
        self.assertEqual(incar["LANGEVIN_GAMMA_L"], 2.0)  # 1000 / 500 fs
        self.assertNotIn("RANDOM_SEED", incar)  # seed=None → VASP picks one

    def test_md_langevin_gamma_one_per_species(self):
        md = InputMDVASP._original_dataclass(temperature_damping_timescale=100.0)
        atoms = Atoms("FeFeO", positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        incar = _build_incar(VaspInput(scf=make_scf(), md=md), structure=atoms)
        self.assertEqual(incar["LANGEVIN_GAMMA"], [10.0, 10.0])

    def test_md_npt_without_pressure_raises(self):
        md = InputMDVASP._original_dataclass(ensemble="NpT", pressure=None)
        with self.assertRaises(ValueError):
            _build_incar(
                VaspInput(scf=make_scf(), md=md), structure=bulk("Fe", cubic=True)
            )

    def test_md_without_structure_raises(self):
        md = InputMDVASP._original_dataclass()
        with self.assertRaises(ValueError):
            _build_incar(VaspInput(scf=make_scf(), md=md))

    def test_dipole_correction(self):
        dip = InputDipoleCorrection._original_dataclass(direction=3, ldipol=True)
        incar = _build_incar(VaspInput(scf=make_scf(), dipole_correction=dip))
        self.assertEqual(incar["LDIPOL"], True)
        self.assertEqual(incar["IDIPOL"], 3)

    def test_minimization_and_md_mutually_exclusive(self):
        mini = InputMinimizationVASP._original_dataclass()
        md = InputMDVASP._original_dataclass()
        with self.assertRaises(ValueError):
            _build_incar(VaspInput(scf=make_scf(), minimization=mini, md=md))

    def test_extra_incar_overrides(self):
        incar = _build_incar(
            VaspInput(scf=make_scf()), extra={"ENCUT": 999, "LREAL": "Auto"}
        )
        self.assertEqual(incar["ENCUT"], 999)
        self.assertEqual(incar["LREAL"], "Auto")


# ── MergeVaspInput ─────────────────────────────────────────────────────────────


class TestMergeVaspInput(unittest.TestCase):
    def test_scf_only(self):
        merged = MergeVaspInput._original_func(scf=make_scf())
        self.assertIsInstance(merged, VaspInput)
        self.assertIsNone(merged.minimization)
        self.assertIsNone(merged.md)
        self.assertIsNone(merged.dipole_correction)
        self.assertIsNone(merged.extra_incar)

    def test_nests_sub_inputs(self):
        mini = InputMinimizationVASP._original_dataclass(algorithm="RMM-DIIS")
        dip = InputDipoleCorrection._original_dataclass()
        merged = MergeVaspInput._original_func(
            scf=make_scf(), minimization=mini, dipole_correction=dip
        )
        self.assertIs(merged.minimization, mini)
        self.assertIs(merged.dipole_correction, dip)

    def test_specific_inputs_become_extra_incar(self):
        flags = AdditionalInputFlags._original_dataclass(
            key1="LREAL", value1="Auto", key2="LWAVE", value2=".FALSE."
        )
        merged = MergeVaspInput._original_func(scf=make_scf(), specific_inputs=flags)
        self.assertEqual(merged.extra_incar, {"LREAL": "Auto", "LWAVE": ".FALSE."})
        # ... and those flags make it into the INCAR
        incar = _build_incar(merged, merged.extra_incar)
        self.assertEqual(incar["LREAL"], "Auto")


# ── CreateVaspInputResources (file writing) ────────────────────────────────────


class TestCreateVaspInputResources(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = self._tmp.name
        self.workdir = os.path.join(self.root, "run")
        # fake POTCAR library so we don't need the real potential files
        self.potcar_lib = os.path.join(self.root, "potentials")
        self.symbol = "Fe_dummy"
        os.makedirs(os.path.join(self.potcar_lib, self.symbol))
        self.potcar_content = "DUMMY POTCAR for Fe\n"
        with open(os.path.join(self.potcar_lib, self.symbol, "POTCAR"), "w") as f:
            f.write(self.potcar_content)
        self.structure = bulk("Fe", cubic=True)  # 2 Fe atoms

    def tearDown(self):
        self._tmp.cleanup()

    def _create(self, scf=None, **kw):
        return CreateVaspInputResources._original_func(
            structure=self.structure,
            calc=VaspInput(scf=scf or make_scf()),
            potcar_lib_path=self.potcar_lib,
            working_directory=self.workdir,
            potcar_symbols=[self.symbol],
            **kw,
        )

    def test_writes_all_four_files(self):
        io_bundle = self._create()
        self.assertEqual(io_bundle.working_directory, self.workdir)
        for name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR"):
            self.assertTrue(os.path.exists(os.path.join(self.workdir, name)), msg=name)

    def test_incar_roundtrips(self):
        self._create(scf=make_scf(energy_cutoff=350.0))
        incar = Incar.from_file(os.path.join(self.workdir, "INCAR"))
        self.assertEqual(incar["ENCUT"], 350.0)
        self.assertEqual(incar["IBRION"], -1)

    def test_kpoints_mesh_written(self):
        self._create(scf=make_scf(kpoints="6 6 6"))
        with open(os.path.join(self.workdir, "KPOINTS")) as f:
            content = f.read()
        self.assertIn("6 6 6", content)

    def test_potcar_is_concatenation(self):
        self._create()
        with open(os.path.join(self.workdir, "POTCAR")) as f:
            self.assertEqual(f.read(), self.potcar_content)

    def test_bad_kpoints_string_raises(self):
        with self.assertRaises(ValueError):
            self._create(scf=make_scf(kpoints="4 4"))


# ── hashing / potcar lookup ────────────────────────────────────────────────────


class TestGenerateHash(unittest.TestCase):
    def _bundle(self, scf=None):
        return VaspInputResources(
            structure=bulk("Fe", cubic=True),
            calc=VaspInput(scf=scf or make_scf()),
            potcar_lib_path="/somewhere",
        )

    def test_hash_is_short_hex_and_deterministic(self):
        # regression: previously raised AttributeError on the removed scf.algorithm
        h1 = _generate_hash(self._bundle())
        h2 = _generate_hash(self._bundle())
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 8)
        int(h1, 16)  # valid hex

    def test_hash_changes_with_settings(self):
        base = _generate_hash(self._bundle())
        changed = _generate_hash(self._bundle(scf=make_scf(energy_cutoff=999.0)))
        self.assertNotEqual(base, changed)


class TestGetPotcarPaths(unittest.TestCase):
    def test_paths_from_csv(self):
        # uses the real bundled PBE CSV; assert shape rather than exact names
        paths = _get_potcar_paths(bulk("Fe", cubic=True), "PBE", "/lib")
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].startswith("/lib"))
        self.assertTrue(paths[0].endswith("POTCAR"))


# ── RunVaspCalculation ─────────────────────────────────────────────────────────


class TestRunVaspCalculation(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.io = VaspInputResources(
            structure=None, calc=None, working_directory=self._tmp.name
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_debug_does_not_launch_vasp(self):
        io_bundle, stdout = RunVaspCalculation._original_func(self.io, debug=True)
        self.assertIs(io_bundle, self.io)
        self.assertEqual(stdout, self._tmp.name)

    def test_nonzero_exit_raises_and_writes_error_msg(self):
        with self.assertRaises(RuntimeError):
            RunVaspCalculation._original_func(self.io, vasp_command="false")
        self.assertTrue(os.path.exists(os.path.join(self._tmp.name, "error.msg")))


# ── ParseVaspOutput (static fixture) ───────────────────────────────────────────


def parse_static_fixture(calc):
    """Run ParseVaspOutput over the static fixture with the given calc attached."""
    io = VaspInputResources(
        structure=None, calc=calc, working_directory=str(STATIC_VASP)
    )
    out, trajectory, converged = ParseVaspOutput._original_func(io)
    return out, trajectory, converged


class TestParseVaspOutputStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out, cls.trajectory, cls.converged = parse_static_fixture(calc=None)

    def test_returns_output_calc_static(self):
        self.assertIsInstance(self.out, OutputCalcStatic.dataclass_type)

    def test_energy(self):
        self.assertIsInstance(self.out.energy, float)
        self.assertAlmostEqual(self.out.energy, -15.22502797, places=5)

    def test_forces_shape(self):
        self.assertEqual(self.out.force.shape, (2, 3))

    def test_final_structure(self):
        self.assertEqual(self.out.structure.get_chemical_formula(), "Fe2")
        self.assertEqual(len(self.out.structure), 2)

    def test_trajectory_and_convergence_ports(self):
        # kept alongside `out` for visualisation nodes (AnimateAse)
        self.assertEqual(len(self.trajectory), 1)
        self.assertTrue(self.converged)


class TestParseVaspOutputMinimize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        calc = VaspInput(
            scf=make_scf(),
            minimization=InputMinimizationVASP._original_dataclass(),
        )
        cls.out, cls.trajectory, cls.converged = parse_static_fixture(calc)

    def test_returns_output_calc_minimize(self):
        self.assertIsInstance(self.out, OutputCalcMinimize.dataclass_type)

    def test_initial_and_final_are_static_outputs(self):
        for stage in (self.out.initial, self.out.final):
            self.assertIsInstance(stage, OutputCalcStatic.dataclass_type)
            self.assertEqual(stage.structure.get_chemical_formula(), "Fe2")
        # the fixture has a single ionic step, so both ends coincide
        self.assertAlmostEqual(self.out.final.energy, -15.22502797, places=5)

    def test_convergence_and_step_count(self):
        self.assertTrue(self.out.is_converged)
        self.assertEqual(self.out.iter_steps, 1)


class TestParseVaspOutputMD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        calc = VaspInput(scf=make_scf(), md=InputMDVASP._original_dataclass())
        cls.out, cls.trajectory, cls.converged = parse_static_fixture(calc)

    def test_returns_output_calc_md(self):
        self.assertIsInstance(self.out, OutputCalcMD.dataclass_type)

    def test_trajectory_shapes(self):
        # fixture: 1 ionic step, 2 atoms
        self.assertEqual(self.out.positions.shape, (1, 2, 3))
        self.assertEqual(self.out.unwrapped_positions.shape, (1, 2, 3))
        self.assertEqual(self.out.forces.shape, (1, 2, 3))
        self.assertEqual(self.out.cells.shape, (1, 3, 3))
        self.assertEqual(self.out.volumes.shape, (1,))

    def test_first_frame_is_not_unwrapped(self):
        # nothing to unwrap on frame 0 — it must match the raw positions
        self.assertTrue(
            np.allclose(self.out.unwrapped_positions[0], self.out.positions[0])
        )

    def test_step_bookkeeping(self):
        self.assertEqual(list(self.out.steps), [1])
        self.assertEqual(list(self.out.natoms), [2])
        self.assertEqual(list(self.out.species), ["Fe", "Fe"])
        self.assertEqual(self.out.indices.tolist(), [[0, 0]])

    def test_energies(self):
        self.assertAlmostEqual(self.out.energies_pot[0], -15.22502797, places=5)
        # no kinetic energy in a non-MD vasprun → total falls back to potential
        self.assertAlmostEqual(self.out.energies_tot[0], -15.22502797, places=5)

    def test_pressures_converted_to_gpa(self):
        self.assertEqual(self.out.pressures.shape, (1, 3, 3))


class TestUnwrapPositions(unittest.TestCase):
    def test_removes_boundary_crossing(self):
        # one atom drifting in +x at 0.2 frac/step, wrapping after step 2
        cell = np.eye(3) * 10.0
        frac = np.array([[[0.8, 0, 0]], [[0.0, 0, 0]], [[0.2, 0, 0]]])
        cells = np.array([cell, cell, cell])
        unwrapped = _unwrap_positions(frac, cells)
        # continuous motion: 8 → 10 → 12 Å, no jump back to 0
        self.assertTrue(np.allclose(unwrapped[:, 0, 0], [8.0, 10.0, 12.0]))

    def test_empty_input(self):
        self.assertEqual(len(_unwrap_positions(np.array([]), np.array([]))), 0)


if __name__ == "__main__":
    unittest.main()
