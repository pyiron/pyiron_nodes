"""Unit tests for the electrochemistry VASP nodes in
``pyiron_nodes.electrochemistry.add_potential.vasp``.

Following the same pattern as ``test_vasp.py``, nothing here launches VASP.
The nodes are exercised through their ``._original_func`` so no ``Workflow``
object is required, and every file the nodes read (POTCAR, plugin template,
``*.dat`` outputs) is a small fixture written into a ``TemporaryDirectory``.

Covered:

* ``_modify_potcar``   — replaces the Ne ``ZVAL`` line in a POTCAR.
* ``_write_plugin_file`` — fills a plugin template and writes ``vasp_plugin.py``.
* ``CCESetup`` / ``CDCESetup`` — build the electrochemistry INCAR + plugin from
  an ``VaspInputResources`` bundle, including the structure-derived quantities.
* ``ParsePotential`` — reshapes the electrostatic-potential trace by ``NSW``.
* The shipped ``.plugin`` templates stay ``str.format``-compatible.
"""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase import Atoms
from pymatgen.io.vasp.inputs import Incar

from pyiron_nodes.atomistic.calculator.data import InputMDVASP, InputSCF
from pyiron_nodes.atomistic.engine.vasp_new import VaspInput, VaspInputResources
from pyiron_nodes.electrochemistry.add_potential.vasp import (
    CCESetup,
    CDCESetup,
    CEParameters,
    ParsePotential,
    _modify_potcar,
    _write_plugin_file,
)

# real plugin templates shipped next to the module under test
_PLUGIN_DIR = (
    Path(__file__).parent.parent.parent.parent / "electrochemistry" / "add_potential"
)

# POTCAR lines in the same shape VASP writes them; the Ne line is what the CCE
# node matches on (``.*POMASS.*ZVAL.*8\.000.*mass and valenz``).
_AU_POTCAR = "  POMASS =  196.970; ZVAL   =   11.000    mass and valenz\n"
_NE_POTCAR = "  POMASS =   20.180; ZVAL   =    8.000    mass and valenz\n"


def make_scf(**overrides) -> InputSCF:
    """Build a pure ``InputSCF`` dataclass instance (kpoints is required)."""
    params = dict(kpoints="4 4 4")
    params.update(overrides)
    return InputSCF._original_dataclass(**params)


def make_md(**overrides) -> InputMDVASP:
    params = dict(temperature=400.0, n_ionic_steps=20)
    params.update(overrides)
    return InputMDVASP._original_dataclass(**params)


# ── _modify_potcar ─────────────────────────────────────────────────────────────


class TestModifyPotcar(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.workdir = self._tmp.name
        with open(os.path.join(self.workdir, "POTCAR"), "w") as f:
            f.writelines([_AU_POTCAR, _NE_POTCAR])

    def tearDown(self):
        self._tmp.cleanup()

    def test_replaces_ne_zval(self):
        _modify_potcar(self.workdir, _NE_POTCAR, zval_ne=8.0001234)
        with open(os.path.join(self.workdir, "POTCAR")) as f:
            content = f.read()
        self.assertIn("ZVAL   =    8.0001234", content)
        # the Au line is left untouched
        self.assertIn("ZVAL   =   11.000", content)

    def test_missing_line_raises(self):
        with self.assertRaises(ValueError):
            _modify_potcar(self.workdir, "not a real POTCAR line\n", zval_ne=8.0)

    def test_missing_potcar_raises(self):
        with TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError):
                _modify_potcar(empty, _NE_POTCAR, zval_ne=8.0)


# ── _write_plugin_file ─────────────────────────────────────────────────────────


class TestWritePluginFile(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.workdir = self._tmp.name
        self.template = os.path.join(self.workdir, "template.plugin")
        with open(self.template, "w") as f:
            f.write("phi0={phi0}\ntemperature={temperature}\nnelect={nelect_neutral}\n")

    def tearDown(self):
        self._tmp.cleanup()

    def _params(self, **kw):
        base = dict(
            path_to_plugin=self.template,
            phi0=0.5,
            Q0=0.0,
            nelect_neutral=38,
            grid_position_frac=0.85,
            grid_roll_frac=0.1,
            tau=50.0,
            temperature=400.0,
            ax=10.0,
            ay=10.0,
            az=12.0,
            d_electrode=4.0,
        )
        base.update(kw)
        return CEParameters(**base)

    def test_fills_template(self):
        _write_plugin_file(self.workdir, self._params())
        with open(os.path.join(self.workdir, "vasp_plugin.py")) as f:
            content = f.read()
        self.assertIn("phi0=0.5", content)
        self.assertIn("temperature=400.0", content)
        self.assertIn("nelect=38", content)

    def test_missing_template_raises(self):
        with self.assertRaises(FileNotFoundError):
            _write_plugin_file(self.workdir, self._params(path_to_plugin="/no/such"))


# ── CCESetup / CDCESetup shared fixture ────────────────────────────────────────


class _SetupFixture(unittest.TestCase):
    """A full ``VaspInputResources`` bundle plus a fake POTCAR library.

    Structure is an orthogonal Au (electrode) + Ne (CCE gas) slab so that the
    structure-derived quantities in the setup nodes have something to chew on.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = self._tmp.name
        self.workdir = os.path.join(self.root, "run")
        os.makedirs(self.workdir)

        # fake per-element POTCAR library (read by _get_potcar_paths → ZVAL)
        self.potcar_lib = os.path.join(self.root, "potentials")
        for symbol, content in (("Au", _AU_POTCAR), ("Ne", _NE_POTCAR)):
            os.makedirs(os.path.join(self.potcar_lib, symbol))
            with open(os.path.join(self.potcar_lib, symbol, "POTCAR"), "w") as f:
                f.write(content)
        # concatenated POTCAR in the working dir (what _modify_potcar edits)
        with open(os.path.join(self.workdir, "POTCAR"), "w") as f:
            f.writelines([_AU_POTCAR, _NE_POTCAR])

        # Au at the bottom, Ne on top — orthogonal cell
        self.structure = Atoms(
            "Au2Ne2",
            positions=[[0, 0, 0], [0, 0, 2], [0, 0, 5], [0, 0, 6]],
            cell=[10.0, 10.0, 12.0],
            pbc=True,
        )
        self.electrode = Atoms("Au", positions=[[0, 0, 0]], cell=[10, 10, 12])

        self.io = VaspInputResources(
            structure=self.structure,
            calc=VaspInput(scf=make_scf(), md=make_md()),
            potcar_lib_path=self.potcar_lib,
            working_directory=self.workdir,
        )

    def tearDown(self):
        self._tmp.cleanup()


# ── CCESetup ───────────────────────────────────────────────────────────────────


class TestCCESetup(_SetupFixture):
    def _run(self, **kw):
        return CCESetup._original_func(
            io_bundle=self.io,
            electrode=self.electrode,
            path_to_plugin=str(_PLUGIN_DIR / "vasp_plugin-CCE.plugin"),
            **kw,
        )

    def test_writes_incar_and_plugin(self):
        self._run()
        self.assertTrue(os.path.exists(os.path.join(self.workdir, "INCAR")))
        self.assertTrue(os.path.exists(os.path.join(self.workdir, "vasp_plugin.py")))

    def test_incar_has_plugin_and_dipole_tags(self):
        io = self._run()
        # PLUGINS/* keys carry a slash pymatgen's Incar reader mangles, so check
        # them on the dict the node actually built
        self.assertEqual(io.extra_incar["PLUGINS/LOCAL_POTENTIAL"], "T")
        self.assertEqual(io.extra_incar["PLUGINS/OCCUPANCIES"], "T")
        # the standard dipole/NELECT tags round-trip through the written INCAR
        incar = Incar.from_file(os.path.join(self.workdir, "INCAR"))
        self.assertEqual(incar["IDIPOL"], 3)
        self.assertEqual(incar["LDIPOL"], True)
        # NELECT is neutral (2*11 + 2*8) when Q0 = 0
        self.assertAlmostEqual(incar["NELECT"], 38.0, places=6)

    def test_charge_shifts_nelect(self):
        # Q0 spread over the 2 Ne atoms: NELECT = 38 + Q0
        self._run(Q0=1.0)
        incar = Incar.from_file(os.path.join(self.workdir, "INCAR"))
        self.assertAlmostEqual(incar["NELECT"], 39.0, places=6)

    def test_ne_zval_modified_in_potcar(self):
        self._run(Q0=1.0)
        with open(os.path.join(self.workdir, "POTCAR")) as f:
            content = f.read()
        # zval_ne = 8 + Q0/n_Ne = 8 + 0.5
        self.assertIn("8.5000000", content)

    def test_requires_md(self):
        self.io.calc = VaspInput(scf=make_scf())  # no md
        with self.assertRaises(ValueError):
            self._run()

    def test_requires_ne_atoms(self):
        self.io.structure = Atoms(
            "Au2", positions=[[0, 0, 0], [0, 0, 2]], cell=[10, 10, 12], pbc=True
        )
        with self.assertRaises(ValueError):
            self._run()

    def test_non_orthogonal_cell_raises(self):
        skewed = self.structure.copy()
        cell = skewed.get_cell()
        cell[2][0] = 3.0  # tilt a3 into x
        skewed.set_cell(cell)
        self.io.structure = skewed
        with self.assertRaises(ValueError):
            self._run()

    def test_existing_output_files_warn(self):
        with open(os.path.join(self.workdir, "Q.dat"), "w") as f:
            f.write("0.0\n")
        # pre-existing outputs are flagged with a warning, not an error
        with self.assertWarns(UserWarning):
            self._run()


# ── CDCESetup ──────────────────────────────────────────────────────────────────


class TestCDCESetup(_SetupFixture):
    def _run(self, **kw):
        return CDCESetup._original_func(
            io_bundle=self.io,
            electrode=self.electrode,
            path_to_plugin=str(_PLUGIN_DIR / "vasp_plugin-CDCE_MD.plugin"),
            **kw,
        )

    def test_writes_incar_and_plugin(self):
        self._run()
        self.assertTrue(os.path.exists(os.path.join(self.workdir, "INCAR")))
        self.assertTrue(os.path.exists(os.path.join(self.workdir, "vasp_plugin.py")))

    def test_incar_has_force_stress_plugin_tag(self):
        io = self._run()
        self.assertEqual(io.extra_incar["PLUGINS/FORCE_AND_STRESS"], "T")
        self.assertEqual(io.extra_incar["PLUGINS/LOCAL_POTENTIAL"], "T")
        # LREMOVE_DRIFT is not a tag pymatgen coerces, so it stays the string "F"
        incar = Incar.from_file(os.path.join(self.workdir, "INCAR"))
        self.assertEqual(incar["LREMOVE_DRIFT"], "F")

    def test_charge_shifts_nelect(self):
        # CDCE: NELECT = nelect_neutral + round(Q0)
        self._run(Q0=2.0)
        incar = Incar.from_file(os.path.join(self.workdir, "INCAR"))
        self.assertAlmostEqual(incar["NELECT"], 40.0, places=6)

    def test_requires_md(self):
        self.io.calc = VaspInput(scf=make_scf())
        with self.assertRaises(ValueError):
            self._run()

    def test_non_orthogonal_cell_raises(self):
        skewed = self.structure.copy()
        cell = skewed.get_cell()
        cell[2][1] = 2.0  # tilt a3 into y
        skewed.set_cell(cell)
        self.io.structure = skewed
        with self.assertRaises(ValueError):
            self._run()

    def test_existing_output_files_warn(self):
        stale = os.path.join(self.workdir, "phi.dat")
        with open(stale, "w") as f:
            f.write("stale\n")
        # pre-existing outputs are flagged with a warning, not an error
        with self.assertWarns(UserWarning):
            self._run()


# ── shipped plugin templates ───────────────────────────────────────────────────


class TestPluginTemplatesFormat(unittest.TestCase):
    """The setup nodes fill the templates with ``str.format`` — guard against a
    stray unescaped brace sneaking into a shipped template."""

    def _all_params(self):
        return dict(
            phi0=0.0,
            Q0=0.0,
            nelect_neutral=38,
            grid_roll_frac=0.1,
            grid_position_frac=0.85,
            tau=50.0,
            temperature=400.0,
            ax=10.0,
            ay=10.0,
            az=12.0,
            d_electrode=4.0,
            i_Ne=1,
            n_elements=2,
            n_Ne=2,
            Q_pos="np.array([5.0, 5.0, 7.0])",
            width_wall=6.5,
            pos_right_wall=0.75,
        )

    def test_templates_are_format_compatible(self):
        for name in ("vasp_plugin-CCE.plugin", "vasp_plugin-CDCE_MD.plugin"):
            template = (_PLUGIN_DIR / name).read_text()
            # must not raise KeyError / ValueError on unescaped braces
            template.format(**self._all_params())


# ── ParsePotential ─────────────────────────────────────────────────────────────


class TestParsePotential(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.workdir = self._tmp.name
        self.nsw = 2
        self.nz = 3
        np.savetxt(os.path.join(self.workdir, "Q.dat"), np.array([0.1, 0.2]))
        np.savetxt(os.path.join(self.workdir, "phi.dat"), np.array([1.0, 1.1]))
        np.savetxt(
            os.path.join(self.workdir, "el_pot_z.dat"),
            np.arange(self.nsw * self.nz, dtype=float),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _bundle(self, extra_incar=None, md=True):
        calc = VaspInput(
            scf=make_scf(), md=make_md(n_ionic_steps=self.nsw) if md else None
        )
        return VaspInputResources(
            structure=None,
            calc=calc,
            working_directory=self.workdir,
            extra_incar=extra_incar,
        )

    def test_reshapes_by_nsw_from_md(self):
        pot2d, charge, phi = ParsePotential._original_func(self._bundle())
        self.assertEqual(pot2d.shape, (self.nsw, self.nz))
        self.assertEqual(len(charge), 2)
        self.assertEqual(len(phi), 2)

    def test_nsw_from_extra_incar_takes_precedence(self):
        # NSW in extra_incar is used even without MD settings
        pot2d, _, _ = ParsePotential._original_func(
            self._bundle(extra_incar={"NSW": self.nsw}, md=False)
        )
        self.assertEqual(pot2d.shape, (self.nsw, self.nz))

    def test_missing_file_raises(self):
        os.remove(os.path.join(self.workdir, "Q.dat"))
        with self.assertRaises(FileNotFoundError):
            ParsePotential._original_func(self._bundle())

    def test_no_nsw_anywhere_raises(self):
        with self.assertRaises(ValueError):
            ParsePotential._original_func(self._bundle(md=False))


if __name__ == "__main__":
    unittest.main()
