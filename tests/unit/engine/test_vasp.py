"""Unit tests for the VASP workflow nodes in ``pyiron_nodes.atomistic.engine.vasp_new``.

Following the pattern in ``test_lammps.py`` and in pyiron_atomistics' own VASP
tests, nothing here launches VASP. We test the two things that matter:

* **INCAR translation** — that ``InputSCF`` / ``InputMinimizationVASP`` / ... map
  to the right VASP tags (via ``_build_incar`` and the small enum helpers).
* **Output parsing** — that ``ParseVaspOutput`` turns what VASP left on disk
  into the right ports.

Output parsing is covered from three directions:

``tests/static/vasp``
    A bare ``vasprun.xml`` and nothing else — the "only the minimum survived"
    path, where stresses, magnetic moments and the volumetric grids are all
    legitimately missing.
``tests/static/vasp_full``
    A real spin-polarized bcc Fe run with OUTCAR, OSZICAR, CONTCAR and DOSCAR,
    so the OUTCAR-only quantities and the two-channel DOS are exercised against
    files VASP actually wrote.
Synthetic parse dictionaries and files
    Built in the tests themselves for the shapes no fixture here has — a
    multi-step MD trajectory, projected DOSCAR blocks, a per-ion magnetization
    table, small CHGCAR/LOCPOT grids.

Nodes are exercised through their ``._original_func`` / ``._original_dataclass``
so no ``Workflow`` object is required.
"""

import os
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase import Atoms
from ase.build import bulk
from pymatgen.io.vasp.inputs import Incar
from vaspparser.vasp.structure import read_atoms
from vaspparser.vasp.volumetric_data import VaspVolumetricData

from pyiron_nodes.atomistic.calculator.data import (
    AdditionalInputFlags,
    InputMDVASP,
    InputDipoleCorrection,
    InputMinimizationVASP,
    InputSCF,
    InputVaspDOS,
    InputVaspOutputFiles,
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
    _dos_from_electronic_structure,
    _final_magmoms,
    _generate_hash,
    _get_potcar_paths,
    _IBRION_MINIMIZE,
    _is_converged,
    _ISMEAR,
    _md_from_output,
    _ordered_elements,
    _output_parser_class,
    _parse_velocities,
    _SkippedVolumetricData,
    _static_from_output,
    _trajectory_from_output,
    _unwrap_positions,
    _volumetric_from_dict,
    read_doscar,
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

    def test_tetrahedron(self):
        # the accurate choice for a DOS; order is irrelevant for ISMEAR -5
        self.assertEqual(_ISMEAR("tetrahedron", 1), -5)

    def test_unknown_smearing_raises(self):
        # regression: previously fell through and returned None (malformed INCAR)
        with self.assertRaises(ValueError):
            _ISMEAR("marzari-vanderbilt", 1)


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
        self.assertEqual(incar["ISPIN"], 1)

    def test_spin_polarized_sets_ispin(self):
        # ISPIN 2 is what makes the magnetic_moments port available
        incar = _build_incar(VaspInput(scf=make_scf(spin_polarized=True)))
        self.assertEqual(incar["ISPIN"], 2)

    def test_output_files_left_out_writes_no_file_tags(self):
        incar = _build_incar(VaspInput(scf=make_scf()))
        for tag in ("LCHARG", "LWAVE", "LVTOT", "LVHAR"):
            self.assertNotIn(tag, incar)

    def test_output_files_map_to_tags(self):
        files = InputVaspOutputFiles._original_dataclass(
            charge_density=True,
            electrostatic_potential=True,
            hartree_potential_only=True,
            wavefunctions=False,
        )
        incar = _build_incar(VaspInput(scf=make_scf(), output_files=files))
        self.assertTrue(incar["LCHARG"])
        self.assertTrue(incar["LVTOT"])
        self.assertTrue(incar["LVHAR"])
        self.assertFalse(incar["LWAVE"])

    def test_output_files_can_switch_chgcar_off(self):
        files = InputVaspOutputFiles._original_dataclass(charge_density=False)
        incar = _build_incar(VaspInput(scf=make_scf(), output_files=files))
        self.assertFalse(incar["LCHARG"])
        self.assertNotIn("LVTOT", incar)

    def test_dos_left_out_writes_no_dos_tags(self):
        incar = _build_incar(VaspInput(scf=make_scf()))
        for tag in ("NEDOS", "LORBIT", "EMIN", "EMAX"):
            self.assertNotIn(tag, incar)

    def test_dos_defaults(self):
        incar = _build_incar(
            VaspInput(scf=make_scf(), dos=InputVaspDOS._original_dataclass())
        )
        self.assertEqual(incar["NEDOS"], 301)
        # unprojected by default: no LORBIT, and no energy window
        self.assertNotIn("LORBIT", incar)
        self.assertNotIn("EMIN", incar)
        self.assertNotIn("EMAX", incar)

    def test_dos_projected_and_window(self):
        dos = InputVaspDOS._original_dataclass(
            n_points=3001, projected=True, energy_min=-15.0, energy_max=10.0
        )
        incar = _build_incar(VaspInput(scf=make_scf(), dos=dos))
        self.assertEqual(incar["NEDOS"], 3001)
        self.assertEqual(incar["LORBIT"], 11)
        self.assertEqual(incar["EMIN"], -15.0)
        self.assertEqual(incar["EMAX"], 10.0)

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


def parse_static_fixture(calc, **kwargs):
    """Run ParseVaspOutput over the static fixture with the given calc attached.

    The fixture holds a vasprun.xml only — no OUTCAR, OSZICAR, CONTCAR, CHGCAR,
    LOCPOT or DOSCAR — so it also covers the "only the bare minimum was written"
    path through ``vaspparser``. A structure has to be supplied because there is
    no POSCAR/CONTCAR for ``parse_vasp_output`` to fall back on.
    """
    io = VaspInputResources(
        structure=bulk("Fe", "bcc", a=2.89, cubic=True),
        calc=calc,
        working_directory=str(STATIC_VASP),
    )
    return ParseVaspOutput._original_func(io, **kwargs)


def calc_ports(result):
    """Pick the (out, trajectory, converged) ports out of the node's return."""
    out, trajectory = result[0], result[1]
    converged = result[-1]
    return out, trajectory, converged


# free energy of the fixture's single ionic step, as vaspparser reports it under
# generic/energy_tot (VASP's TOTEN)
FIXTURE_ENERGY = -15.22362209


class TestParseVaspOutputStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.out,
            cls.trajectory,
            cls.last_structure,
            cls.total_energy,
            cls.magnetic_moments,
            cls.dos,
            cls.electrostatic_potential,
            cls.electron_density,
            cls.converged,
        ) = parse_static_fixture(calc=None)

    def test_returns_output_calc_static(self):
        self.assertIsInstance(self.out, OutputCalcStatic.dataclass_type)

    def test_energy(self):
        self.assertIsInstance(self.out.energy, float)
        self.assertAlmostEqual(self.out.energy, FIXTURE_ENERGY, places=5)

    def test_forces_shape(self):
        self.assertEqual(self.out.force.shape, (2, 3))

    def test_no_stress_without_outcar(self):
        # stresses are an OUTCAR-only quantity in vaspparser
        self.assertIsNone(self.out.stress)

    def test_final_structure(self):
        self.assertEqual(self.out.structure.get_chemical_formula(), "Fe2")
        self.assertEqual(len(self.out.structure), 2)

    def test_trajectory_and_convergence_ports(self):
        # kept alongside `out` for visualisation nodes (AnimateAse)
        self.assertEqual(len(self.trajectory), 1)
        self.assertTrue(self.converged)

    def test_last_structure_port(self):
        self.assertIsInstance(self.last_structure, Atoms)
        self.assertEqual(self.last_structure.get_chemical_formula(), "Fe2")
        self.assertTrue(
            np.allclose(self.last_structure.get_cell().array, np.eye(3) * 2.89)
        )

    def test_total_energy_port(self):
        self.assertIsInstance(self.total_energy, float)
        self.assertAlmostEqual(self.total_energy, FIXTURE_ENERGY, places=5)

    def test_magnetic_moments_absent_for_non_spin_polarized_run(self):
        self.assertIsNone(self.magnetic_moments)

    def test_volumetric_ports_absent_when_files_were_not_written(self):
        self.assertIsNone(self.electron_density)
        self.assertIsNone(self.electrostatic_potential)

    def test_dos_port_read_from_vasprun(self):
        # the fixture is non-magnetic, so a single spin channel
        self.assertEqual(self.dos.energies.shape, (301,))
        self.assertEqual(self.dos.total_densities.shape, (1, 301))
        self.assertEqual(self.dos.integrated_densities.shape, (1, 301))
        self.assertAlmostEqual(self.dos.efermi, 5.29132146, places=6)

    def test_dos_is_not_projected_without_lorbit(self):
        self.assertEqual(len(np.asarray(self.dos.resolved_densities)), 0)
        self.assertIsNone(self.dos.orbitals)

    def test_dos_source_doscar_without_a_doscar(self):
        result = parse_static_fixture(calc=None, dos_source="doscar")
        self.assertIsNone(result[5])

    def test_unknown_dos_source_raises(self):
        with self.assertRaises(ValueError):
            parse_static_fixture(calc=None, dos_source="eigenval")


class TestParseVaspOutputMinimize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        calc = VaspInput(
            scf=make_scf(),
            minimization=InputMinimizationVASP._original_dataclass(),
        )
        cls.out, cls.trajectory, cls.converged = calc_ports(parse_static_fixture(calc))

    def test_returns_output_calc_minimize(self):
        self.assertIsInstance(self.out, OutputCalcMinimize.dataclass_type)

    def test_initial_and_final_are_static_outputs(self):
        for stage in (self.out.initial, self.out.final):
            self.assertIsInstance(stage, OutputCalcStatic.dataclass_type)
            self.assertEqual(stage.structure.get_chemical_formula(), "Fe2")
        # the fixture has a single ionic step, so both ends coincide
        self.assertAlmostEqual(self.out.final.energy, FIXTURE_ENERGY, places=5)

    def test_convergence_and_step_count(self):
        self.assertTrue(self.out.is_converged)
        self.assertEqual(self.out.iter_steps, 1)


class TestParseVaspOutputMD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        calc = VaspInput(scf=make_scf(), md=InputMDVASP._original_dataclass())
        cls.out, cls.trajectory, cls.converged = calc_ports(parse_static_fixture(calc))

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
        self.assertEqual(list(self.out.steps), [0])
        self.assertEqual(list(self.out.natoms), [2])
        self.assertEqual(list(self.out.species), ["Fe", "Fe"])
        self.assertEqual(self.out.indices.tolist(), [[0, 0]])

    def test_energies(self):
        self.assertAlmostEqual(self.out.energies_pot[0], FIXTURE_ENERGY, places=5)
        # no kinetic energy in a non-MD vasprun → total falls back to potential
        self.assertAlmostEqual(self.out.energies_tot[0], FIXTURE_ENERGY, places=5)

    def test_outcar_only_fields_stay_empty(self):
        # pressures (from the stresses) and temperatures need an OUTCAR, which
        # the fixture does not have — the fields keep their empty default
        self.assertEqual(len(np.asarray(self.out.pressures)), 0)
        self.assertEqual(len(np.asarray(self.out.temperatures)), 0)


# ── ParseVaspOutput against a complete run ────────────────────────────────────
# ``vasp_full`` is a real spin-polarized bcc Fe calculation with a dipole
# correction: OUTCAR, OSZICAR, CONTCAR and DOSCAR alongside vasprun.xml. It
# covers everything the vasprun-only fixture cannot reach — stresses, the OUTCAR
# path through vaspparser, the CONTCAR structure and a two-channel DOS. Its
# CHGCAR and LOCPOT are left out on purpose (1.6 MB); the volumetric round trip
# is built at test time instead, further down.

STATIC_VASP_FULL = Path(__file__).parent.parent.parent / "static" / "vasp_full"

FULL_ENERGY = -15.22362202
FULL_STRESS_GPA = -28.153327  # OUTCAR reports -281.53327 kB
FULL_NELECT = 16  # 2 x Fe, 8 valence electrons each
FULL_NBANDS = 13


def parse_full_fixture(calc=None, **kwargs):
    io = VaspInputResources(
        structure=None,  # no structure needed: the fixture has a CONTCAR
        calc=calc,
        working_directory=str(STATIC_VASP_FULL),
    )
    return ParseVaspOutput._original_func(io, **kwargs)


class TestParseVaspOutputFullRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.out,
            cls.trajectory,
            cls.last_structure,
            cls.total_energy,
            cls.magnetic_moments,
            cls.dos,
            cls.electrostatic_potential,
            cls.electron_density,
            cls.converged,
        ) = parse_full_fixture()

    def test_structure_read_from_the_contcar(self):
        # structure=None, so vaspparser has to fall back to CONTCAR/POSCAR
        self.assertEqual(self.last_structure.get_chemical_formula(), "Fe2")
        self.assertTrue(
            np.allclose(self.last_structure.get_cell().array, np.eye(3) * 2.89)
        )
        self.assertTrue(
            np.allclose(self.last_structure.get_positions()[1], [1.445, 1.445, 1.445])
        )

    def test_energy_and_convergence(self):
        self.assertAlmostEqual(self.total_energy, FULL_ENERGY, places=6)
        self.assertAlmostEqual(self.out.energy, FULL_ENERGY, places=6)
        self.assertTrue(self.converged)

    def test_stress_read_from_the_outcar_in_gpa(self):
        # the one quantity vaspparser only ever gets from the OUTCAR; the
        # fixture's own "in kB" line reads -281.53327
        self.assertIsNotNone(self.out.stress)
        self.assertEqual(self.out.stress.shape, (3, 3))
        self.assertTrue(
            np.allclose(np.diag(self.out.stress), FULL_STRESS_GPA, atol=1e-3)
        )
        self.assertTrue(np.allclose(self.out.stress[0, 1], 0.0, atol=1e-6))

    def test_forces_are_zero_at_the_relaxed_geometry(self):
        self.assertEqual(self.out.force.shape, (2, 3))
        self.assertTrue(np.allclose(self.out.force, 0.0, atol=1e-6))

    def test_spin_polarized_dos_has_two_channels(self):
        self.assertEqual(self.dos.energies.shape, (301,))
        self.assertEqual(self.dos.total_densities.shape, (2, 301))
        self.assertEqual(self.dos.integrated_densities.shape, (2, 301))
        self.assertAlmostEqual(self.dos.efermi, 5.29125004, places=6)

    def test_dos_integrates_to_the_electron_count(self):
        # the physical check on any DOS: NELECT below E_F, and never more than
        # NBANDS states per spin channel over the whole window
        at_fermi = np.searchsorted(self.dos.energies, self.dos.efermi)
        self.assertAlmostEqual(
            self.dos.integrated_densities[:, at_fermi].sum(), FULL_NELECT, delta=1.0
        )
        self.assertAlmostEqual(
            self.dos.integrated_densities[:, -1].sum(), 2 * FULL_NBANDS, places=6
        )

    def test_magnetic_moments_need_lorbit_not_just_ispin(self):
        # the fixture is ISPIN 2 but LORBIT 0, so VASP printed no per-ion
        # magnetization table and the port stays empty
        self.assertIn("ISPIN = 2", (STATIC_VASP_FULL / "INCAR").read_text())
        self.assertIsNone(self.magnetic_moments)

    def test_volumetric_ports_absent_when_the_files_are(self):
        self.assertIsNone(self.electron_density)
        self.assertIsNone(self.electrostatic_potential)


class TestParseVaspOutputFullRunDoscar(unittest.TestCase):
    """`dos_source='doscar'` against a DOSCAR VASP actually wrote."""

    def test_doscar_matches_the_file_on_disk(self):
        dos = parse_full_fixture(dos_source="doscar")[5]
        raw = np.loadtxt(STATIC_VASP_FULL / "DOSCAR", skiprows=6)
        self.assertTrue(np.array_equal(dos.energies, raw[:, 0]))
        self.assertTrue(np.array_equal(dos.total_densities, raw[:, 1:3].T))
        self.assertTrue(np.array_equal(dos.integrated_densities, raw[:, 3:5].T))

    def test_header_efermi_agrees_with_vasprun(self):
        from_file = parse_full_fixture(dos_source="doscar")[5]
        from_xml = parse_full_fixture(dos_source="vasprun")[5]
        self.assertAlmostEqual(from_file.efermi, from_xml.efermi, places=6)
        self.assertTrue(np.allclose(from_file.energies, from_xml.energies, atol=1e-3))


class TestParseVaspOutputFullRunMD(unittest.TestCase):
    """The MD reshaping against a real parse dictionary rather than a built one."""

    @classmethod
    def setUpClass(cls):
        calc = VaspInput(scf=make_scf(), md=InputMDVASP._original_dataclass())
        cls.out = parse_full_fixture(calc)[0]

    def test_returns_output_calc_md(self):
        self.assertIsInstance(self.out, OutputCalcMD.dataclass_type)

    def test_temperatures_come_through_from_the_outcar(self):
        # empty for the vasprun-only fixture; here the OUTCAR supplies them
        self.assertEqual(len(np.asarray(self.out.temperatures)), 1)

    def test_pressures_come_through_in_gpa(self):
        pressures = np.asarray(self.out.pressures)
        self.assertEqual(pressures.shape, (1, 3, 3))
        self.assertTrue(
            np.allclose(
                np.diagonal(pressures, axis1=1, axis2=2), FULL_STRESS_GPA, atol=1e-3
            )
        )

    def test_volume_and_bookkeeping(self):
        self.assertTrue(np.allclose(self.out.volumes, 2.89**3))
        self.assertTrue(np.array_equal(self.out.natoms, [2]))
        self.assertEqual(list(self.out.species), ["Fe", "Fe"])

    def test_velocities_absent_for_a_run_that_wrote_none(self):
        self.assertEqual(len(np.asarray(self.out.velocities)), 0)


class TestMagneticMomentsFromOutcar(unittest.TestCase):
    """The per-ion magnetization table, end to end through vaspparser.

    No LORBIT run was available, so the reference OUTCAR is copied and a
    magnetization table in VASP's layout is spliced in ahead of the timing
    block. Everything else in the directory stays real, which is what makes this
    exercise the OUTCAR reader rather than a hand-built dictionary.
    """

    MAGNETIZATION_BLOCK = [
        " Atomic Wigner-Seitz radii\n",
        " \n",
        " magnetization (x)\n",
        " \n",
        "# of ion       s       p       d       tot\n",
        "------------------------------------------\n",
        "    1        0.000   0.000   2.100   2.200\n",
        "    2        0.000   0.000   2.200   2.300\n",
        "\n",
    ]

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.workdir = Path(self._tmp.name)
        for name in ("vasprun.xml", "OSZICAR", "CONTCAR", "POSCAR"):
            shutil.copy(STATIC_VASP_FULL / name, self.workdir / name)

        lines = (STATIC_VASP_FULL / "OUTCAR").read_text().splitlines(keepends=True)
        cut = next(i for i, line in enumerate(lines) if "General timing" in line)
        (self.workdir / "OUTCAR").write_text(
            "".join(lines[:cut] + self.MAGNETIZATION_BLOCK + lines[cut:])
        )

    def tearDown(self):
        self._tmp.cleanup()

    def parse(self):
        io = VaspInputResources(
            structure=None, calc=None, working_directory=str(self.workdir)
        )
        return ParseVaspOutput._original_func(io)

    def test_moments_are_read_per_atom(self):
        magmoms = self.parse()[4]
        self.assertIsNotNone(magmoms)
        self.assertEqual(magmoms.shape, (2,))
        self.assertTrue(np.allclose(magmoms, [2.2, 2.3]))

    def test_rest_of_the_parse_is_unaffected(self):
        result = self.parse()
        self.assertAlmostEqual(result[3], FULL_ENERGY, places=6)
        self.assertTrue(
            np.allclose(np.diag(result[0].stress), FULL_STRESS_GPA, atol=1e-3)
        )


class TestVolumetricFromDisk(unittest.TestCase):
    """CHGCAR/LOCPOT read through vaspparser, and the flags that skip them.

    The real files are 800 kB each, so small ones are generated here instead —
    which also makes the expected grid values something the test controls.
    """

    GRID = (3, 3, 3)

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.workdir = Path(self._tmp.name)
        for name in ("vasprun.xml", "OUTCAR", "OSZICAR", "CONTCAR", "POSCAR"):
            shutil.copy(STATIC_VASP_FULL / name, self.workdir / name)

        structure = read_atoms(str(STATIC_VASP_FULL / "POSCAR"))
        self.total = np.arange(np.prod(self.GRID), dtype=float).reshape(self.GRID)
        self.write_volumetric(self.workdir / "CHGCAR", structure, self.total)
        self.write_volumetric(self.workdir / "LOCPOT", structure, self.total * 2.0)

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def write_volumetric(path, structure, data):
        volumetric = VaspVolumetricData()
        volumetric.atoms = structure
        volumetric.total_data = data
        volumetric.write_vasp_volumetric(filename=str(path), normalize=False)

    def parse(self, **kwargs):
        io = VaspInputResources(
            structure=None, calc=None, working_directory=str(self.workdir)
        )
        return ParseVaspOutput._original_func(io, **kwargs)

    def test_both_grids_are_read_by_default(self):
        density, potential = self.parse()[7], self.parse()[6]
        self.assertEqual(density.total_data.shape, self.GRID)
        self.assertEqual(potential.total_data.shape, self.GRID)

    def test_charge_density_is_normalised_by_the_volume(self):
        # vaspparser reads CHGCAR with normalize=True and LOCPOT without
        density, potential = self.parse()[7], self.parse()[6]
        volume = read_atoms(str(STATIC_VASP_FULL / "POSCAR")).get_volume()
        self.assertTrue(np.allclose(density.total_data * volume, self.total, atol=1e-6))
        self.assertTrue(np.allclose(potential.total_data, self.total * 2.0, atol=1e-6))

    def test_grid_helpers_work_on_the_parsed_data(self):
        averaged = self.parse()[6].get_average_along_axis(ind=2)
        self.assertEqual(len(averaged), self.GRID[2])

    def test_electron_density_can_be_skipped(self):
        result = self.parse(parse_electron_density=False)
        self.assertIsNone(result[7])
        self.assertIsNotNone(result[6])  # LOCPOT still read

    def test_electrostatic_potential_can_be_skipped(self):
        result = self.parse(parse_electrostatic_potential=False)
        self.assertIsNone(result[6])
        self.assertIsNotNone(result[7])  # CHGCAR still read

    def test_both_can_be_skipped(self):
        result = self.parse(
            parse_electron_density=False, parse_electrostatic_potential=False
        )
        self.assertIsNone(result[6])
        self.assertIsNone(result[7])


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


# ── output helpers on synthetic vaspparser dictionaries ───────────────────────
# The static fixture is a non-magnetic single point without CHGCAR/LOCPOT, so
# the remaining branches are exercised against hand-built parse dictionaries of
# the shape ``parse_vasp_output`` returns.


def make_output_dict(dft=None, **generic):
    base = {"energy_tot": np.array([-15.0]), "dft": dft or {}}
    base.update(generic)
    return {"generic": base}


# One atom sitting still at the origin and one drifting by +2 Å per step along x
# in a 10 Å cube, so it wraps between step 0 and step 1. Cartesian positions are
# what vaspparser hands over, which makes this the fixture for everything that
# has to undo the wrapping or convert a per-step quantity.
MD_CELL = np.eye(3) * 10.0
MD_POSITIONS = np.array(
    [
        [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # wrapped: really at 10 Å
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],  # wrapped: really at 12 Å
    ]
)


def make_md_output_dict(**overrides):
    """A three-step MD parse dictionary shaped like ``parse_vasp_output`` output."""
    n_steps = len(MD_POSITIONS)
    generic = {
        "positions": MD_POSITIONS,
        "cells": np.array([MD_CELL] * n_steps),
        "forces": np.arange(n_steps * 2 * 3, dtype=float).reshape(n_steps, 2, 3),
        "volume": np.full(n_steps, 1000.0),
        "steps": np.arange(n_steps),
        "energy_pot": np.array([-15.0, -15.1, -15.2]),
        "energy_tot": np.array(
            [-14.5, -14.6, -14.7]
        ),  # includes the ions' kinetic part
        "dft": {},
    }
    generic.update(overrides)
    return {"generic": generic}


def md_trajectory():
    return [
        Atoms("Fe2", positions=frame, cell=MD_CELL, pbc=True) for frame in MD_POSITIONS
    ]


class TestFinalMagmoms(unittest.TestCase):
    def test_missing_key_gives_none(self):
        self.assertIsNone(_final_magmoms(make_output_dict()))

    def test_empty_list_gives_none(self):
        # OUTCAR present but ISPIN 1 → vaspparser leaves the list empty
        self.assertIsNone(_final_magmoms(make_output_dict(dft={"final_magmoms": []})))

    def test_collinear_takes_last_ionic_step(self):
        dft = {"final_magmoms": [[1.0, 1.0], [2.2, 2.3]]}
        magmoms = _final_magmoms(make_output_dict(dft=dft))
        self.assertTrue(np.allclose(magmoms, [2.2, 2.3]))

    def test_non_collinear_keeps_three_components(self):
        dft = {"final_magmoms": [[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]]}
        magmoms = _final_magmoms(make_output_dict(dft=dft))
        self.assertEqual(magmoms.shape, (2, 3))


class TestVolumetricFromDict(unittest.TestCase):
    def setUp(self):
        self.structure = bulk("Fe", "bcc", a=2.89, cubic=True)

    def test_missing_file_gives_none(self):
        self.assertIsNone(_volumetric_from_dict(None, self.structure))

    def test_total_data_and_atoms_are_attached(self):
        grid = np.arange(8.0).reshape(2, 2, 2)
        volumetric = _volumetric_from_dict({"total": grid}, self.structure)
        self.assertTrue(np.allclose(volumetric.total_data, grid))
        self.assertIsNone(volumetric.diff_data)
        self.assertEqual(len(volumetric.atoms), 2)
        # the grid helpers are why the object is returned rather than the array
        self.assertEqual(len(volumetric.get_average_along_axis(ind=2)), 2)

    def test_spin_difference_is_kept(self):
        grid = np.ones((2, 2, 2))
        volumetric = _volumetric_from_dict(
            {"total": grid, "diff": grid * 0.5}, self.structure
        )
        self.assertTrue(np.allclose(volumetric.diff_data, 0.5))


class TestReadDoscar(unittest.TestCase):
    """`vaspparser` has no DOSCAR parser, so `read_doscar` is ours to cover.

    The projected values encode ion/orbital/spin, so a transposed axis shows up
    as a wrong number rather than merely a wrong shape.
    """

    @staticmethod
    def write_doscar(
        path, n_ions=2, n_points=4, n_spin=2, n_orbitals=9, projected=True
    ):
        energies = np.linspace(-1.0, 3.0, n_points)
        header = f"  3.0  -1.0  {n_points}  0.5  1.0\n"
        lines = [
            f"   {n_ions}   {n_ions}   {int(projected)}   0\n",
            " volume\n",
            " temperature\n",
            "  CAR \n",
            " system\n",
            header,
        ]

        def rows(columns):
            table = np.column_stack(columns)
            return ["  " + "  ".join(f"{v:.6f}" for v in row) + "\n" for row in table]

        lines += rows(
            [energies]
            + [np.full(n_points, 10.0 + s) for s in range(n_spin)]  # DOS per spin
            + [np.full(n_points, 20.0 + s) for s in range(n_spin)]  # integrated
        )

        expected = np.zeros((n_spin, n_ions, n_orbitals, n_points))
        if projected:
            for ion in range(n_ions):
                lines.append(header)
                columns = [energies]
                for orbital in range(n_orbitals):
                    for spin in range(n_spin):  # spin varies fastest
                        value = 100 * ion + 10 * orbital + spin
                        columns.append(np.full(n_points, float(value)))
                        expected[spin, ion, orbital, :] = value
                lines += rows(columns)

        path.write_text("".join(lines))
        return energies, expected

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "DOSCAR"

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_file(self):
        self.assertIsNone(read_doscar(str(Path(self._tmp.name) / "nope")))

    def test_empty_file(self):
        self.path.write_text("")
        self.assertIsNone(read_doscar(str(self.path)))

    def test_truncated_file(self):
        self.path.write_text("only a header\n")
        self.assertIsNone(read_doscar(str(self.path)))

    def test_total_dos_non_magnetic(self):
        energies, _ = self.write_doscar(self.path, n_spin=1, projected=False)
        dos = read_doscar(str(self.path))
        self.assertTrue(np.allclose(dos.energies, energies))
        self.assertTrue(np.allclose(dos.total_densities, 10.0))
        self.assertTrue(np.allclose(dos.integrated_densities, 20.0))
        self.assertEqual(dos.efermi, 0.5)
        self.assertEqual(len(np.asarray(dos.resolved_densities)), 0)

    def test_total_dos_spin_polarized(self):
        self.write_doscar(self.path, n_spin=2, projected=False)
        dos = read_doscar(str(self.path))
        # 5 columns: E, DOS up, DOS down, integrated up, integrated down
        self.assertEqual(dos.total_densities.shape, (2, 4))
        self.assertTrue(np.allclose(dos.total_densities[0], 10.0))
        self.assertTrue(np.allclose(dos.total_densities[1], 11.0))
        self.assertTrue(np.allclose(dos.integrated_densities[1], 21.0))

    def test_projected_dos_axes(self):
        for n_spin in (1, 2):
            for n_orbitals, first in ((3, "s"), (9, "s"), (16, "s")):
                with self.subTest(n_spin=n_spin, n_orbitals=n_orbitals):
                    _, expected = self.write_doscar(
                        self.path, n_spin=n_spin, n_orbitals=n_orbitals
                    )
                    dos = read_doscar(str(self.path))
                    self.assertEqual(
                        dos.resolved_densities.shape, (n_spin, 2, n_orbitals, 4)
                    )
                    self.assertTrue(np.allclose(dos.resolved_densities, expected))
                    self.assertEqual(dos.orbitals[0], first)
                    self.assertEqual(len(dos.orbitals), n_orbitals)

    def test_orbital_names_follow_the_lorbit_scheme(self):
        self.write_doscar(self.path, n_orbitals=9)
        self.assertEqual(
            read_doscar(str(self.path)).orbitals[:4], ["s", "py", "pz", "px"]
        )
        self.write_doscar(self.path, n_orbitals=3)
        self.assertEqual(read_doscar(str(self.path)).orbitals, ["s", "p", "d"])

    def test_truncated_projected_block_keeps_the_total_dos(self):
        self.write_doscar(self.path, n_orbitals=9)
        lines = self.path.read_text().splitlines(keepends=True)
        self.path.write_text("".join(lines[:-3]))  # cut into the last ion's block
        dos = read_doscar(str(self.path))
        self.assertEqual(dos.total_densities.shape, (2, 4))
        self.assertEqual(len(np.asarray(dos.resolved_densities)), 0)


class TestTrajectoryFromOutput(unittest.TestCase):
    """The static fixture has a single ionic step, so multi-frame goes here."""

    def test_one_atoms_object_per_ionic_step(self):
        template = bulk("Fe", "bcc", a=2.89, cubic=True)
        trajectory = _trajectory_from_output(make_md_output_dict(), template)
        self.assertEqual(len(trajectory), 3)
        for frame in trajectory:
            self.assertEqual(frame.get_chemical_symbols(), ["Fe", "Fe"])

    def test_each_frame_gets_its_own_positions_and_cell(self):
        template = bulk("Fe", "bcc", a=2.89, cubic=True)
        trajectory = _trajectory_from_output(make_md_output_dict(), template)
        for frame, expected in zip(trajectory, MD_POSITIONS):
            self.assertTrue(np.allclose(frame.get_positions(), expected))
            self.assertTrue(np.allclose(frame.get_cell().array, MD_CELL))
        # the template must not be mutated by building the trajectory
        self.assertTrue(np.allclose(template.get_cell().array, np.eye(3) * 2.89))

    def test_frames_are_independent_objects(self):
        template = bulk("Fe", "bcc", a=2.89, cubic=True)
        trajectory = _trajectory_from_output(make_md_output_dict(), template)
        trajectory[0].positions[0] = [5.0, 5.0, 5.0]
        self.assertTrue(np.allclose(trajectory[1].get_positions()[0], 0.0))


class TestStaticFromOutput(unittest.TestCase):
    def test_index_picks_the_ionic_step(self):
        structure = bulk("Fe", "bcc", a=2.89, cubic=True)
        output = make_md_output_dict()
        first = _static_from_output(output, 0, structure)
        last = _static_from_output(output, -1, structure)
        self.assertAlmostEqual(first.energy, -14.5)
        self.assertAlmostEqual(last.energy, -14.7)
        self.assertTrue(np.allclose(last.force, output["generic"]["forces"][-1]))
        self.assertIs(last.structure, structure)

    def test_energy_is_a_plain_float(self):
        # downstream nodes compare and serialise this; a 0-d array would leak through
        out = _static_from_output(make_md_output_dict(), -1, None)
        self.assertIsInstance(out.energy, float)

    def test_stress_converted_from_eva3_to_gpa(self):
        # vaspparser reports stresses in eV/Å³; OutputCalcStatic.stress is GPa
        stresses = np.array([np.diag([-0.1757, -0.1757, -0.1757])] * 3)
        output = make_md_output_dict(stresses=stresses)
        out = _static_from_output(output, -1, None)
        self.assertTrue(np.allclose(np.diag(out.stress), -28.15, atol=0.01))

    def test_no_stress_without_an_outcar(self):
        self.assertIsNone(_static_from_output(make_md_output_dict(), -1, None).stress)


class TestMdFromOutput(unittest.TestCase):
    """`_md_from_output` is the most involved reshaping step, so it gets its own set."""

    def setUp(self):
        self.trajectory = md_trajectory()
        self.vasprun = str(STATIC_VASP / "vasprun.xml")

    def build(self, **overrides):
        return _md_from_output(
            make_md_output_dict(**overrides), self.trajectory, self.vasprun
        )

    def test_positions_and_cells_pass_through(self):
        out = self.build()
        self.assertTrue(np.allclose(out.positions, MD_POSITIONS))
        self.assertEqual(out.cells.shape, (3, 3, 3))
        self.assertEqual(out.forces.shape, (3, 2, 3))

    def test_unwrapping_goes_through_fractional_coordinates(self):
        # regression: vaspparser gives Cartesian positions, but unwrapping only
        # works in fractional space — a missing cell inversion silently returns
        # a trajectory that still jumps at the boundary
        out = self.build()
        drifting = out.unwrapped_positions[:, 1, 0]
        self.assertTrue(np.allclose(drifting, [8.0, 10.0, 12.0]))

    def test_static_atom_stays_put_after_unwrapping(self):
        self.assertTrue(np.allclose(self.build().unwrapped_positions[:, 0, :], 0.0))

    def test_energies_keep_the_kinetic_contribution_separate(self):
        out = self.build()
        self.assertTrue(np.allclose(out.energies_pot, [-15.0, -15.1, -15.2]))
        self.assertTrue(np.allclose(out.energies_tot, [-14.5, -14.6, -14.7]))

    def test_step_bookkeeping(self):
        out = self.build()
        self.assertTrue(np.array_equal(out.steps, [0, 1, 2]))
        self.assertTrue(np.array_equal(out.natoms, [2, 2, 2]))
        self.assertTrue(np.allclose(out.volumes, 1000.0))
        self.assertEqual(list(out.species), ["Fe", "Fe"])
        self.assertEqual(out.indices.shape, (3, 2))
        self.assertTrue(np.all(out.indices == 0))  # one species → one id

    def test_temperatures_come_from_the_outcar(self):
        self.assertEqual(len(np.asarray(self.build().temperatures)), 0)
        out = self.build(temperature=np.array([300.0, 305.0, 298.0]))
        self.assertTrue(np.allclose(out.temperatures, [300.0, 305.0, 298.0]))

    def test_pressures_converted_from_eva3_to_gpa(self):
        stresses = np.array([np.diag([0.1, 0.1, 0.1])] * 3)
        out = self.build(stresses=stresses)
        self.assertEqual(out.pressures.shape, (3, 3, 3))
        self.assertTrue(
            np.allclose(np.diagonal(out.pressures, axis1=1, axis2=2), 16.02, atol=0.01)
        )

    def test_pressures_empty_without_an_outcar(self):
        self.assertEqual(len(np.asarray(self.build().pressures)), 0)

    def test_species_ids_track_the_element_order(self):
        self.trajectory = [
            Atoms("FeO", positions=frame, cell=MD_CELL, pbc=True)
            for frame in MD_POSITIONS
        ]
        out = self.build()
        self.assertEqual(list(out.species), ["Fe", "O"])
        self.assertTrue(np.array_equal(out.indices[0], [0, 1]))


class TestParseVelocities(unittest.TestCase):
    """Velocities are read straight from vasprun.xml, bypassing vaspparser."""

    VELOCITY_XML = """<?xml version="1.0" encoding="ISO-8859-1"?>
<modeling>
 <calculation>
  <structure>
   <varray name="positions"><v> 0.0 0.0 0.0 </v><v> 0.5 0.5 0.5 </v></varray>
   <varray name="velocities"><v> 0.1 0.2 0.3 </v><v> 0.4 0.5 0.6 </v></varray>
  </structure>
 </calculation>
 <calculation>
  <structure>
   <varray name="velocities"><v> 1.1 1.2 1.3 </v><v> 1.4 1.5 1.6 </v></varray>
  </structure>
 </calculation>
</modeling>
"""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "vasprun.xml"

    def tearDown(self):
        self._tmp.cleanup()

    def test_one_frame_per_calculation(self):
        self.path.write_text(self.VELOCITY_XML)
        velocities = _parse_velocities(str(self.path))
        self.assertEqual(velocities.shape, (2, 2, 3))
        self.assertTrue(np.allclose(velocities[0][0], [0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(velocities[1][1], [1.4, 1.5, 1.6]))

    def test_static_run_has_no_velocities(self):
        # the fixture is a single point, so VASP wrote no velocity varray
        self.assertEqual(len(_parse_velocities(str(STATIC_VASP / "vasprun.xml"))), 0)

    def test_malformed_xml_is_best_effort(self):
        self.path.write_text("<modeling><calculation>")
        self.assertEqual(len(_parse_velocities(str(self.path))), 0)

    def test_missing_file_is_best_effort(self):
        # regression: an MD run parsed from an OUTCAR alone has no vasprun.xml,
        # and this used to raise FileNotFoundError out of ParseVaspOutput
        self.assertEqual(len(_parse_velocities(str(self.path / "gone"))), 0)


class TestOutputParserClass(unittest.TestCase):
    """The hook that skips huge volumetric files and hands back the parser."""

    def test_parser_instance_is_collected(self):
        collected = []
        parser = _output_parser_class(True, True, collected)()
        self.assertEqual(collected, [parser])

    def test_volumetric_readers_kept_when_requested(self):
        parser = _output_parser_class(True, True, [])()
        self.assertNotIsInstance(parser.charge_density, _SkippedVolumetricData)
        self.assertNotIsInstance(parser.electrostatic_potential, _SkippedVolumetricData)

    def test_skipped_readers_are_substituted_independently(self):
        parser = _output_parser_class(False, True, [])()
        self.assertIsInstance(parser.charge_density, _SkippedVolumetricData)
        self.assertNotIsInstance(parser.electrostatic_potential, _SkippedVolumetricData)

        parser = _output_parser_class(True, False, [])()
        self.assertNotIsInstance(parser.charge_density, _SkippedVolumetricData)
        self.assertIsInstance(parser.electrostatic_potential, _SkippedVolumetricData)

    def test_skipped_reader_never_touches_the_file(self):
        # Output.collect calls from_file unconditionally when the file exists;
        # the stand-in has to swallow that and leave total_data unset
        skipped = _SkippedVolumetricData()
        self.assertIsNone(skipped.from_file("/nonexistent/CHGCAR"))
        self.assertIsNone(skipped.total_data)


class FakeElectronicStructure:
    """Just the attributes `_dos_from_electronic_structure` reads."""

    def __init__(self, **kwargs):
        self.dos_energies = kwargs.get("dos_energies", [])
        self.dos_densities = kwargs.get("dos_densities", [])
        self.dos_idensities = kwargs.get("dos_idensities", [])
        self.efermi = kwargs.get("efermi")
        self.resolved_densities = kwargs.get("resolved_densities")
        self.orbital_dict = kwargs.get("orbital_dict")


class TestDosFromElectronicStructure(unittest.TestCase):
    def test_no_electronic_structure(self):
        self.assertIsNone(_dos_from_electronic_structure(None))

    def test_no_dos_grid(self):
        # OUTCAR-only run: an ElectronicStructure exists but carries no DOS
        self.assertIsNone(_dos_from_electronic_structure(FakeElectronicStructure()))

    def test_total_dos(self):
        dos = _dos_from_electronic_structure(
            FakeElectronicStructure(
                dos_energies=[-1.0, 0.0, 1.0],
                dos_densities=[[0.0, 2.0, 1.0]],
                dos_idensities=[[0.0, 2.0, 3.0]],
                efermi=0.5,
            )
        )
        self.assertTrue(np.allclose(dos.energies, [-1.0, 0.0, 1.0]))
        self.assertEqual(dos.total_densities.shape, (1, 3))
        self.assertEqual(dos.efermi, 0.5)
        self.assertIsNone(dos.orbitals)

    def test_projected_dos_names_orbitals_in_index_order(self):
        # orbital_dict maps name → column index and is not ordered by construction
        dos = _dos_from_electronic_structure(
            FakeElectronicStructure(
                dos_energies=[-1.0, 0.0],
                dos_densities=[[0.0, 1.0]],
                dos_idensities=[[0.0, 1.0]],
                resolved_densities=np.zeros((1, 2, 3, 2)),
                orbital_dict={"d": 2, "s": 0, "p": 1},
            )
        )
        self.assertEqual(dos.orbitals, ["s", "p", "d"])
        self.assertEqual(dos.resolved_densities.shape, (1, 2, 3, 2))

    def test_projected_dos_without_orbital_names(self):
        dos = _dos_from_electronic_structure(
            FakeElectronicStructure(
                dos_energies=[-1.0, 0.0],
                dos_densities=[[0.0, 1.0]],
                dos_idensities=[[0.0, 1.0]],
                resolved_densities=np.zeros((1, 1, 9, 2)),
            )
        )
        self.assertEqual(dos.orbitals, [])
        self.assertEqual(dos.resolved_densities.shape, (1, 1, 9, 2))


class TestParseVaspOutputDoscarSource(unittest.TestCase):
    """`dos_source='doscar'` end to end, with a DOSCAR beside the fixture."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.workdir = Path(self._tmp.name)
        shutil.copy(STATIC_VASP / "vasprun.xml", self.workdir / "vasprun.xml")
        TestReadDoscar.write_doscar(
            self.workdir / "DOSCAR", n_spin=1, n_orbitals=9, projected=True
        )
        self.io = VaspInputResources(
            structure=bulk("Fe", "bcc", a=2.89, cubic=True),
            calc=None,
            working_directory=str(self.workdir),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_doscar_source_reads_the_file(self):
        dos = ParseVaspOutput._original_func(self.io, dos_source="doscar")[5]
        self.assertEqual(dos.energies.shape, (4,))
        self.assertTrue(np.allclose(dos.total_densities, 10.0))
        self.assertEqual(dos.resolved_densities.shape, (1, 2, 9, 4))

    def test_vasprun_source_ignores_the_doscar(self):
        # same directory, other source: the 301-point grid from vasprun.xml
        dos = ParseVaspOutput._original_func(self.io, dos_source="vasprun")[5]
        self.assertEqual(dos.energies.shape, (301,))


class TestIsConverged(unittest.TestCase):
    def test_electronic_loop_that_hit_nelm_did_not_converge(self):
        calc = VaspInput(scf=make_scf(num_electronic_steps=3))
        dft = {"scf_energy_free": [[-1.0, -1.1, -1.2]]}
        self.assertFalse(_is_converged(make_output_dict(dft=dft), calc))

    def test_electronic_loop_that_stopped_early_converged(self):
        calc = VaspInput(scf=make_scf(num_electronic_steps=3))
        dft = {"scf_energy_free": [[-1.0, -1.1]]}
        self.assertTrue(_is_converged(make_output_dict(dft=dft), calc))

    def test_relaxation_that_used_every_ionic_step_did_not_converge(self):
        mini = InputMinimizationVASP._original_dataclass(max_ionic_steps=2)
        calc = VaspInput(scf=make_scf(num_electronic_steps=10), minimization=mini)
        output = make_output_dict(
            dft={"scf_energy_free": [[-1.0], [-1.0]]},
            energy_tot=np.array([-15.0, -15.1]),
        )
        self.assertFalse(_is_converged(output, calc))

    def test_md_ignores_the_ionic_step_count(self):
        # an MD run always uses all NSW steps; only the SCF loop can fail
        md = InputMDVASP._original_dataclass(n_ionic_steps=2)
        calc = VaspInput(scf=make_scf(num_electronic_steps=10), md=md)
        output = make_output_dict(
            dft={"scf_energy_free": [[-1.0], [-1.0]]},
            energy_tot=np.array([-15.0, -15.1]),
        )
        self.assertTrue(_is_converged(output, calc))

    def test_no_calc_falls_back_to_converged(self):
        self.assertTrue(_is_converged(make_output_dict(), None))


if __name__ == "__main__":
    unittest.main()
