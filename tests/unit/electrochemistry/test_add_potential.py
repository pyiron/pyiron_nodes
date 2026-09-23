"""Unit tests for the electrochemistry VASP nodes in
``pyiron_nodes.electrochemistry.add_potential.vasp``.

Following the same pattern as ``test_vasp.py``, nothing here launches VASP.
The nodes are exercised through their ``._original_func`` so no ``Workflow``
object is required, and every file the nodes read (POTCAR, plugin template,
``*.dat`` outputs) is a small fixture written into a ``TemporaryDirectory``.

Covered:

* ``_plugin_content`` — fills a plugin template and returns its content as a
  string (no disk writes).
* ``CCESetup`` / ``CDCESetup`` — build the ``VaspPlugin`` bundle (rendered
  plugin content, extra INCAR tags, and — for CCE — the Ne ``ZVAL`` POTCAR
  override) from a bare ``structure``/``electrode``/``calc``, including the
  structure-derived quantities. Neither node touches disk anymore; that is
  ``CreateVaspInputResources``'s job now, exercised here as an integration
  check that the returned ``VaspPlugin`` is actually consumed correctly.
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
from pyiron_nodes.atomistic.engine.vasp_new import (
    CreateVaspInputResources,
    VaspInput,
    VaspInputResources,
    VaspPlugin,
)
from pyiron_nodes.electrochemistry.add_potential.vasp import (
    CCESetup,
    CDCESetup,
    CEParameters,
    ParsePotential,
    _plugin_content,
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


# ── _plugin_content ──────────────────────────────────────────────────────────


class TestPluginContent(unittest.TestCase):
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
        content = _plugin_content(self._params())
        self.assertIn("phi0=0.5", content)
        self.assertIn("temperature=400.0", content)
        self.assertIn("nelect=38", content)

    def test_missing_template_raises(self):
        with self.assertRaises(FileNotFoundError):
            _plugin_content(self._params(path_to_plugin="/no/such"))


# ── CCESetup / CDCESetup shared fixture ────────────────────────────────────────


class _SetupFixture(unittest.TestCase):
    """A bare structure/electrode/calc plus a fake per-element POTCAR library.

    Structure is an orthogonal Au (electrode) + Ne (CCE gas) slab so that the
    structure-derived quantities in the setup nodes have something to chew on.
    Neither ``CCESetup`` nor ``CDCESetup`` writes to disk anymore, so there is
    no working directory here — only the POTCAR library ``_get_potcar_paths``
    reads ZVAL values from.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = self._tmp.name

        # fake per-element POTCAR library (read by _get_potcar_paths → ZVAL)
        self.potcar_lib = os.path.join(self.root, "potentials")
        for symbol, content in (("Au", _AU_POTCAR), ("Ne", _NE_POTCAR)):
            os.makedirs(os.path.join(self.potcar_lib, symbol))
            with open(os.path.join(self.potcar_lib, symbol, "POTCAR"), "w") as f:
                f.write(content)

        # Au at the bottom, Ne on top — orthogonal cell
        self.structure = Atoms(
            "Au2Ne2",
            positions=[[0, 0, 0], [0, 0, 2], [0, 0, 5], [0, 0, 6]],
            cell=[10.0, 10.0, 12.0],
            pbc=True,
        )
        self.electrode = Atoms("Au", positions=[[0, 0, 0]], cell=[10, 10, 12])
        self.calc = VaspInput(scf=make_scf(), md=make_md())

    def tearDown(self):
        self._tmp.cleanup()


# ── CCESetup ───────────────────────────────────────────────────────────────────


class TestCCESetup(_SetupFixture):
    def _run(self, structure=None, calc=None, **kw):
        return CCESetup._original_func(
            structure=structure if structure is not None else self.structure,
            electrode=self.electrode,
            calc=calc if calc is not None else self.calc,
            potcar_lib_path=self.potcar_lib,
            path_to_plugin=str(_PLUGIN_DIR / "vasp_plugin-CCE.plugin"),
            **kw,
        )

    def test_returns_structure_calc_and_plugin_unchanged(self):
        structure, calc, plugin_data = self._run()
        self.assertIs(structure, self.structure)
        self.assertIs(calc, self.calc)
        self.assertIsInstance(plugin_data, VaspPlugin)

    def test_plugin_content_is_rendered(self):
        _, _, plugin_data = self._run(potential=0.5)
        self.assertIn("phi0 = 0.5", plugin_data.plugin_content)
        self.assertIn("temperature = 400.0", plugin_data.plugin_content)

    def test_extra_incar_has_plugin_and_nelect_tags(self):
        _, _, plugin_data = self._run()
        self.assertEqual(plugin_data.extra_incar["PLUGINS/LOCAL_POTENTIAL"], "T")
        self.assertEqual(plugin_data.extra_incar["PLUGINS/OCCUPANCIES"], "T")
        # NELECT is neutral (2*11 + 2*8) when Q0 = 0
        self.assertAlmostEqual(plugin_data.extra_incar["NELECT"], 38.0, places=6)

    def test_charge_shifts_nelect(self):
        # Q0 spread over the 2 Ne atoms: NELECT = 38 + Q0
        _, _, plugin_data = self._run(Q0=1.0)
        self.assertAlmostEqual(plugin_data.extra_incar["NELECT"], 39.0, places=6)

    def test_ne_zval_override_is_recorded(self):
        _, _, plugin_data = self._run(Q0=1.0)
        # zval_ne = 8 + Q0/n_Ne = 8 + 0.5
        (new_line,) = plugin_data.override_potcar.values()
        self.assertIn("8.5000000", new_line)

    def test_potcar_lib_path_carried_on_plugin(self):
        _, _, plugin_data = self._run()
        self.assertEqual(plugin_data.potcar_lib_path, self.potcar_lib)

    def test_requires_ne_atoms(self):
        no_ne = Atoms(
            "Au2", positions=[[0, 0, 0], [0, 0, 2]], cell=[10, 10, 12], pbc=True
        )
        with self.assertRaises(ValueError):
            self._run(structure=no_ne)

    def test_non_orthogonal_cell_raises(self):
        skewed = self.structure.copy()
        cell = skewed.get_cell()
        cell[2][0] = 3.0  # tilt a3 into x
        skewed.set_cell(cell)
        with self.assertRaises(ValueError):
            self._run(structure=skewed)

    def test_missing_md_raises(self):
        with self.assertRaises(AttributeError):
            self._run(calc=VaspInput(scf=make_scf()))  # no md


# ── CDCESetup ──────────────────────────────────────────────────────────────────


class TestCDCESetup(_SetupFixture):
    def _run(self, structure=None, calc=None, **kw):
        return CDCESetup._original_func(
            structure=structure if structure is not None else self.structure,
            electrode=self.electrode,
            calc=calc if calc is not None else self.calc,
            potcar_lib_path=self.potcar_lib,
            path_to_plugin=str(_PLUGIN_DIR / "vasp_plugin-CDCE_MD.plugin"),
            **kw,
        )

    def test_returns_structure_calc_and_plugin_unchanged(self):
        structure, calc, plugin_data = self._run()
        self.assertIs(structure, self.structure)
        self.assertIs(calc, self.calc)
        self.assertIsInstance(plugin_data, VaspPlugin)

    def test_extra_incar_has_force_stress_plugin_tag(self):
        _, _, plugin_data = self._run()
        self.assertEqual(plugin_data.extra_incar["PLUGINS/FORCE_AND_STRESS"], "T")
        self.assertEqual(plugin_data.extra_incar["PLUGINS/LOCAL_POTENTIAL"], "T")
        self.assertEqual(plugin_data.extra_incar["LREMOVE_DRIFT"], "F")

    def test_charge_shifts_nelect(self):
        # CDCE: NELECT = nelect_neutral + round(Q0)
        _, _, plugin_data = self._run(Q0=2.0)
        self.assertAlmostEqual(plugin_data.extra_incar["NELECT"], 40.0, places=6)

    def test_no_potcar_override(self):
        # unlike CCE, CDCE never touches the POTCAR
        _, _, plugin_data = self._run()
        self.assertIsNone(plugin_data.override_potcar)

    def test_non_orthogonal_cell_raises(self):
        skewed = self.structure.copy()
        cell = skewed.get_cell()
        cell[2][1] = 2.0  # tilt a3 into y
        skewed.set_cell(cell)
        with self.assertRaises(ValueError):
            self._run(structure=skewed)

    def test_missing_md_raises(self):
        with self.assertRaises(AttributeError):
            self._run(calc=VaspInput(scf=make_scf()))


# ── CCESetup / CDCESetup → CreateVaspInputResources integration ───────────────
# The setup nodes no longer write anything themselves — they hand back a
# ``VaspPlugin`` that ``CreateVaspInputResources`` is responsible for folding
# into the rendered INCAR/POTCAR content. This is the wiring the workflow
# relies on, so it is worth covering directly rather than only through each
# node in isolation.


class TestSetupIntegratesWithCreateVaspInputResources(_SetupFixture):
    def test_cce_plugin_feeds_into_create_vasp_input_resources(self):
        structure, calc, plugin_data = CCESetup._original_func(
            structure=self.structure,
            electrode=self.electrode,
            calc=self.calc,
            potcar_lib_path=self.potcar_lib,
            path_to_plugin=str(_PLUGIN_DIR / "vasp_plugin-CCE.plugin"),
            Q0=1.0,
        )
        io_bundle = CreateVaspInputResources._original_func(
            structure=structure,
            calc=calc,
            potcar_lib_path=self.potcar_lib,
            plugin_data=plugin_data,
            working_directory=os.path.join(self.root, "run"),
        )
        self.assertEqual(io_bundle.plugin_content, plugin_data.plugin_content)
        incar = Incar.from_str(io_bundle.incar_content)
        self.assertAlmostEqual(incar["NELECT"], 39.0, places=6)
        # the Ne ZVAL line was rewritten in the concatenated POTCAR content
        self.assertIn("8.5000000", io_bundle.potcar_content)
        self.assertNotIn("ZVAL   =    8.000    mass and valenz", io_bundle.potcar_content)

    def test_cdce_plugin_feeds_into_create_vasp_input_resources(self):
        structure, calc, plugin_data = CDCESetup._original_func(
            structure=self.structure,
            electrode=self.electrode,
            calc=self.calc,
            potcar_lib_path=self.potcar_lib,
            path_to_plugin=str(_PLUGIN_DIR / "vasp_plugin-CDCE_MD.plugin"),
            Q0=2.0,
        )
        io_bundle = CreateVaspInputResources._original_func(
            structure=structure,
            calc=calc,
            potcar_lib_path=self.potcar_lib,
            plugin_data=plugin_data,
            working_directory=os.path.join(self.root, "run"),
        )
        incar = Incar.from_str(io_bundle.incar_content)
        self.assertAlmostEqual(incar["NELECT"], 40.0, places=6)
        self.assertEqual(incar["LREMOVE_DRIFT"], "F")


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
