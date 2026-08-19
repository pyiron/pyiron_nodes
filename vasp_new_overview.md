# VASP Workflow Nodes — Overview and Code Explanation

## What we built

A set of pyiron workflow nodes for running VASP DFT calculations, following the same
design pattern as the existing `lammps.py`. The implementation lives in two files:

- `atomistic/engine/vasp_new.py` — the workflow nodes and their helpers
- `atomistic/calculator/data.py` — extended with the VASP input dataclasses

A user-level config file `~/.pyiron_vasp_config` stores system-specific paths so they
don't have to be hardcoded.

---

## System config: `~/.pyiron_vasp_config`

```ini
pyiron_vasp_resources = /path/to/vasp/potentials
default_POTCAR_set    = potpawPBE
default_functional    = PBE
vasp_command          = module load vasp; mpiexec -n 1 vasp_std
```

- `pyiron_vasp_resources` — base directory containing the POTCAR library
- `default_POTCAR_set` — which POTCAR set to use; `potpawPBE` → resolved to `potpaw_PBE/`
- `default_functional` — selects which CSV lookup table to use (PBE or LDA)
- `vasp_command` — the full shell command to launch VASP on your cluster

`read_potcar_config()` reads this file at module import time and stores the resolved
values in module-level constants (`_default_potcar_lib_path`, `_default_functional`,
`_default_vasp_command`). If the file is absent these fall back to empty strings /
safe defaults and the user must supply them explicitly as node parameters.

---

## The input dataclasses (`data.py`)

Each is decorated with `@as_inp_dataclass_node`, so pyiron flow renders it as an
interactive input node where every field becomes an editable parameter in the
GUI. They are split by concern rather than mapped 1:1 onto INCAR tags, and
`MergeVaspInput` layers them into one `VaspInput`:

| dataclass | required? | covers |
|---|---|---|
| `InputSCF` | yes | k-points, cutoff, smearing, electronic convergence, `spin_polarized` |
| `InputMinimizationVASP` | no | ionic relaxation — IBRION/NSW/EDIFFG/ISIF |
| `InputMDVASP` | no | Langevin MD — NVT/NpT, temperature, time step, seed |
| `InputDipoleCorrection` | no | LDIPOL/IDIPOL, e.g. for asymmetric slabs |
| `InputVaspDOS` | no | NEDOS/LORBIT/EMIN/EMAX — how finely the DOSCAR is resolved |
| `InputVaspOutputFiles` | no | LCHARG/LVTOT/LVHAR/LWAVE — which output files to write |
| `AdditionalInputFlags` | no | escape hatch for any INCAR tag not covered above |

`InputMinimizationVASP` and `InputMDVASP` are mutually exclusive; both drive the
ionic loop.

The k-point mesh lives on `InputSCF` (not on the structure node) because
sampling is a calculation setting, not a property of the structure.

---

## `VaspInputResources` — internal state carrier (`vasp_new.py`)

```python
@dataclass
class VaspInputResources:
    structure: Atoms           # ASE Atoms object
    calc: VaspInput|None       # the merged calculation settings
    potcar_lib_path: str       # path to the potpaw_PBE/ directory
    working_directory: str     # where input/output files live
    potcar_symbols: list|None  # optional override for POTCAR selection per element
    extra_incar: dict|None     # any additional INCAR tags beyond VaspInput
```

This is a plain Python `@dataclass` (not a workflow node). It acts as a data carrier
passed between nodes, exactly like `LammpsInputResources` in `lammps.py`. Each node
receives it, uses it, and passes it forward — so downstream nodes always have full
context about the calculation.

`structure` is ASE `Atoms` (not pymatgen `Structure`) so it is directly compatible
with the output of the `Bulk` node in `atomistic/structure/build.py`.

---

## The four workflow nodes

### 1. `MergeVaspInput`

```
Inputs:  scf, minimization, md, dipole_correction, dos, output_files,
         specific_inputs
Output:  calc (VaspInput)
```

Layers the optional input dataclasses on top of the mandatory `InputSCF` into a
single `VaspInput`. `minimization` and `md` are mutually exclusive — both drive
the ionic loop, and that is enforced when the INCAR is built.

---

### 2. `CreateVaspInputResources`

```
Inputs:  structure, calc, potcar_lib_path, working_directory, potcar_symbols
Output:  io_bundle (VaspInputResources)
```

Bundles the inputs into a `VaspInputResources` dataclass, then creates the
working directory (falling back to a hash of the inputs when none is given) and
writes the four VASP input files. The defaults for `potcar_lib_path` and
`vasp_command` are read from `~/.pyiron_vasp_config` at import time.

**POSCAR** — the crystal structure.
ASE `Atoms` is first converted to a pymatgen `Structure` via `AseAtomsAdaptor`,
then written using pymatgen's POSCAR writer (handles fractional coordinates,
selective dynamics, etc. correctly).

**INCAR** — the calculation parameters.
`_build_incar()` translates the `VaspInput` sub-dataclasses into a pymatgen
`Incar` dict, merges any `extra_incar` overrides, and writes it. This is where
all the VASP tags (ENCUT, EDIFF, ISMEAR, ISPIN, LCHARG, …) are set.

**POTCAR** — the pseudopotentials.
1. `_ordered_elements()` extracts the unique ordered element list from the structure
   (e.g. Fe Fe O → [Fe, O]).
2. `_get_potcar_paths()` reads the bundled CSV
   (`vasp_resources/vasp_pseudopotential_PBE_data.csv`) to find the default potential
   name for each element (e.g. Fe → "Fe", not "Fe_pv").
3. The actual POTCAR files are found at `{potcar_lib_path}/{potential_name}/POTCAR`
   and concatenated in element order using `shutil.copyfileobj`. VASP requires a
   single POTCAR containing one block per species, in the same order as POSCAR.

The CSV bundled in `vasp_resources/` is the key piece: it maps each element symbol
to its default (and available) potential variants. `potcar_symbols` can override this
per element if you need e.g. `Fe_pv` instead of `Fe`.

**KPOINTS** — the k-point sampling.
`scf.kpoints` is a string of three integers (e.g. `"4 4 4"`), written as a
Gamma-centred mesh via pymatgen's `Kpoints.gamma_automatic()`.

---

### 3. `RunVaspCalculation`

```
Inputs:  io_bundle, vasp_command, run_script_path, threads_per_core, debug
Outputs: io_bundle, stdout
```

Runs VASP as a subprocess using `subprocess.run` with `shell=True`. This is
**blocking** — `wf.run()` will not return until VASP finishes. The call inherits
the current environment (`os.environ.copy()`) and runs in `working_directory`.

`vasp_command` defaults to whatever is in `~/.pyiron_vasp_config`. On a cluster this
typically needs to load environment modules before calling `mpiexec`, e.g.:
```
module load vasp; module load intel/19.1.0 impi/2019.6; mpiexec -n 4 vasp_std
```

If VASP exits with a non-zero return code, stderr and stdout are written to
`error.msg` in the working directory and a `RuntimeError` is raised.

`debug=True` skips execution entirely and returns immediately — useful for testing
that the input files are written correctly before submitting a real calculation.

To monitor a running job, watch the OUTCAR file in a terminal:
```bash
tail -f /path/to/working_directory/OUTCAR
```

---

### 4. `ParseVaspOutput`

```
Inputs:  io_bundle, dos_source, parse_electron_density,
         parse_electrostatic_potential
Outputs: out, trajectory, last_structure, total_energy, magnetic_moments,
         dos, electrostatic_potential, electron_density, converged
```

All output parsing goes through **`vaspparser`** — the VASP counterpart of the
`lammpsparser` that `ParseLammpsOutput` uses. A single call to
`vaspparser.vasp.output.parse_vasp_output(working_directory, structure)` reads
every VASP output file that is present and returns one hierarchical dictionary:

| file | what it contributes |
|---|---|
| `vasprun.xml` | energies, forces, positions, cells, electronic structure |
| `OUTCAR` | magnetic moments, stresses, temperatures, elastic constants |
| `OSZICAR` | higher-precision SCF energies |
| `CONTCAR` | final structure at full precision |
| `DOSCAR` | density of states (read by us, not by vaspparser — see below) |
| `CHGCAR` | electron density on the FFT grid |
| `LOCPOT` | electrostatic potential on the FFT grid |
| `AECCAR0`/`AECCAR2` | Bader charges (needs the external `bader` binary) |

The node reshapes that dictionary into its ports:

- **`out`** — the calculator dataclass matching the calc that was submitted, so
  the same downstream nodes work for VASP and LAMMPS:
  `md` → `OutputCalcMD` (full ionic trajectory), `minimization` →
  `OutputCalcMinimize` (initial + final, convergence), otherwise
  `OutputCalcStatic` (single point). Stresses are converted from vaspparser's
  eV/Å³ to GPa.
- **`trajectory`** — every ionic step as an ASE `Atoms`; feeds visualisation
  nodes such as `AnimateAse`.
- **`last_structure`** — the final `Atoms`, taken from the CONTCAR when one was
  written (higher precision than the positions in `vasprun.xml`).
- **`total_energy`** — total energy of the last ionic step in eV
  (`generic/energy_tot`, i.e. VASP's TOTEN); for an MD run this includes the
  kinetic energy of the ions.
- **`magnetic_moments`** — per-atom moments of the last ionic step, shape
  `(n_atoms,)` collinear or `(n_atoms, 3)` non-collinear. They are read from the
  per-ion magnetization table in the OUTCAR, which VASP prints only when the run
  is spin-polarized (`InputSCF.spin_polarized`) **and** LORBIT is set
  (`InputVaspDOS.projected`). ISPIN 2 on its own gives a total moment but no
  per-atom breakdown, and this port stays `None`.
- **`dos`** — `OutputVaspDOS`: `energies`, `total_densities` and
  `integrated_densities`, each with a leading spin axis, plus
  `resolved_densities` `(n_spin, n_atoms, n_orbitals, n_points)` and the
  matching `orbitals` names when the projection was switched on. `energies` are
  absolute — subtract `efermi` to plot against E − E_F.
- **`electrostatic_potential`**, **`electron_density`** — `VaspVolumetricData`
  from `LOCPOT` / `CHGCAR`, or `None` when the file was not written. The grid is
  `.total_data`; `.diff_data` holds the spin difference of a spin-polarized
  CHGCAR. The object is returned rather than the bare array so the grid helpers
  stay available:
  ```python
  potential = wf.parse.outputs.electrostatic_potential.value
  potential.get_average_along_axis(ind=2)   # planar average along c → work function
  potential.write_cube_file("locpot.cube")  # for external visualisation
  ```
- **`converged`** — electronic convergence, and ionic convergence too for a
  relaxation. `vaspparser` reports no flag of its own, so the usual criterion is
  applied: a loop that stopped before hitting its own step limit (`NELM`, `NSW`)
  converged.

`parse_electron_density` and `parse_electrostatic_potential` (both default
`True`) exist because `CHGCAR` and `LOCPOT` are large and slow to read — a
`CHGCAR` scales with the FFT grid, not the number of atoms. Switching one off
leaves the file on disk untouched and the corresponding port at `None`.

### Where the DOS comes from

`vaspparser` has no DOSCAR parser. It takes the DOS from the `<dos>` block of
vasprun.xml instead, which holds the same quantity VASP writes to the DOSCAR.
`dos_source` chooses between the two:

```python
wf.parse = ParseVaspOutput(io_bundle=..., dos_source="vasprun")  # default
wf.parse = ParseVaspOutput(io_bundle=..., dos_source="doscar")   # read_doscar()
```

`"doscar"` goes through the module-level `read_doscar(filename)`, a small parser
written here because there is nothing in `vaspparser` to delegate to. It handles
ISPIN 1 and 2 and the per-ion projected blocks, and returns the same
`OutputVaspDOS` as the vasprun path, so downstream nodes cannot tell them apart.

The two sources should agree. **If they do not, treat it as a signal about the
run rather than about the parsing** — a DOS whose integral does not reach
`NELECT` at the Fermi level, or exceeds `NBANDS` at the top of the window, means
VASP produced a bad DOS. `read_doscar` exists partly to make that comparison
cheap:

```python
from pyiron_nodes.atomistic.engine.vasp_new import read_doscar

file_dos = read_doscar(f"{WORKDIR}/DOSCAR")
xml_dos = wf.parse.outputs.dos.value
np.allclose(file_dos.total_densities, xml_dos.total_densities)
```

---

## Asking VASP for the files in the first place

Two ports stay empty unless the run was set up to produce them:

**`magnetic_moments`** needs `ISPIN = 2` *and* `LORBIT`, i.e. both dataclasses:

```python
wf.scf = InputSCF(kpoints="8 8 8", spin_polarized=True)   # ISPIN 2
wf.dos = InputVaspDOS(projected=True)                     # LORBIT 11
```

**`dos`** is filled whenever VASP wrote a DOS at all, but `InputVaspDOS`
controls how usable it is — `n_points` (NEDOS) sets the energy resolution and
`projected` (LORBIT 11) adds the site- and orbital-projected channels:

```python
wf.dos = InputVaspDOS(n_points=3001, projected=True, energy_min=-15.0, energy_max=10.0)
```

For a smooth DOS, pair it with the tetrahedron method, which needs a
Gamma-centred mesh and gives no useful forces — so use it for a static run, not
a relaxation:

```python
wf.scf = InputSCF(kpoints="12 12 12", smearing_type="tetrahedron", spin_polarized=True)
```

**`electron_density` / `electrostatic_potential`** need `CHGCAR` / `LOCPOT`,
selected with `InputVaspOutputFiles` and passed to `MergeVaspInput`:

```python
wf.output_files = InputVaspOutputFiles(
    charge_density=True,             # LCHARG → CHGCAR → electron_density
    electrostatic_potential=True,    # LVTOT  → LOCPOT → electrostatic_potential
    hartree_potential_only=False,    # LVHAR  → LOCPOT without the XC part
    wavefunctions=False,             # LWAVE  → WAVECAR
)
```

Leaving `output_files` unset writes none of these tags, so VASP's own defaults
apply (`LCHARG = .TRUE.`, no LOCPOT).

---

## Complete workflow example

A spin-polarized static run that keeps both the charge density and the local
potential, so every output port is filled:

```python
from core import Workflow
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.calculator.data import (
    InputSCF,
    InputVaspDOS,
    InputVaspOutputFiles,
)
from pyiron_nodes.atomistic.engine.vasp_new import (
    MergeVaspInput,
    CreateVaspInputResources,
    RunVaspCalculation,
    ParseVaspOutput,
)

wf = Workflow("vasp_Fe")

# 1. Build structure
wf.structure = Bulk(name="Fe", crystalstructure="bcc", a=2.87, cubic=True)

# 2. SCF settings — spin_polarized=True is half of what magnetic_moments needs
wf.scf = InputSCF(kpoints="8 8 8", energy_cutoff=400.0, spin_polarized=True)

# 3. DOS resolution; projected=True is the other half (LORBIT 11)
wf.dos = InputVaspDOS(n_points=3001, projected=True)

# 4. Which output files VASP should write
wf.output_files = InputVaspOutputFiles(
    charge_density=True,           # CHGCAR → electron_density
    electrostatic_potential=True,  # LOCPOT → electrostatic_potential
)

# 5. Combine into one calc (add minimization=... or md=... for an ionic loop)
wf.calc = MergeVaspInput(
    scf=wf.scf.outputs.output,
    dos=wf.dos.outputs.output,
    output_files=wf.output_files.outputs.output,
)

# 6. Bundle inputs and write POSCAR / INCAR / POTCAR / KPOINTS
wf.resources = CreateVaspInputResources(
    structure=wf.structure.outputs.structure,
    calc=wf.calc.outputs.calc,
    working_directory="./fe_static",
)

# 7. Run VASP (blocking)
wf.run_vasp = RunVaspCalculation(
    io_bundle=wf.resources.outputs.io_bundle,
)

# 8. Parse everything back
wf.parse = ParseVaspOutput(
    io_bundle=wf.run_vasp.outputs.io_bundle,
)

wf.run()

out = wf.parse.outputs
print("Total energy:  ", out.total_energy.value, "eV")
print("Converged:     ", out.converged.value)
print("Final structure:", out.last_structure.value)
print("Trajectory:    ", len(out.trajectory.value), "ionic steps")
print("Magnetic moments:", out.magnetic_moments.value)          # (n_atoms,)
print("Electron density:", out.electron_density.value.total_data.shape)
print("Planar-averaged potential:",
      out.electrostatic_potential.value.get_average_along_axis(ind=2))
print("DOS points:      ", out.dos.value.energies.shape, "at E_F =", out.dos.value.efermi)
```

---

## Design decisions

| Decision | Reason |
|---|---|
| `vaspparser` for **all** output parsing | Same split as LAMMPS, where `ParseLammpsOutput` delegates to `lammpsparser`; one call covers vasprun.xml, OUTCAR, OSZICAR, CONTCAR, CHGCAR and LOCPOT, so pymatgen's `Vasprun` is no longer needed on the output side |
| `VaspVolumetricData` on the volumetric ports | Keeps the grid helpers (`get_average_along_axis`, `write_cube_file`) that a bare numpy array would lose; the array is still one attribute away as `.total_data` |
| CHGCAR/LOCPOT parsing is opt-out | Both files are large enough that always reading them would dominate the runtime of the node |
| `read_doscar` written by hand | The one output file `vaspparser` does not parse; it reads the DOS out of vasprun.xml instead. Having both behind `dos_source` makes the two sources directly comparable when a DOS looks wrong |
| No `pyiron_atomistics` dependency | The old `vasp.py` depended on it for output parsing; `vaspparser` provides the same parsers as a standalone package |
| No pymatgen `Potcar` class | Requires `PMG_VASP_PSP_DIR` to be configured; our approach reads files directly from the path in `~/.pyiron_vasp_config` |
| Input dataclasses live in `data.py` | Consistent with `InputCalcMD`, `InputCalcMinimize` — all calculator inputs live in one place |
| `kpoints` on `InputSCF` | K-point sampling is a calculation setting, not a structural property |
| ASE `Atoms` throughout | Compatible with `Bulk` and all other structure nodes; no conversion needed at boundaries |
| Plain variable names at `return` | pyiron infers output port labels from the AST of the return statement — attribute accesses like `result.stdout` don't produce valid labels |
