# VASP Workflow Nodes — Overview and Code Explanation

## What we built

A set of pyiron workflow nodes for running VASP DFT calculations, following the same
design pattern as the existing `lammps.py`. The implementation lives in two files:

- `atomistic/engine/vasp_new.py` — the four workflow nodes and their helpers
- `atomistic/calculator/data.py` — extended with `InputCalcDFT` (the DFT settings dataclass)

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

## `InputCalcDFT` — DFT calculation settings (`data.py`)

```python
@as_inp_dataclass_node
class InputCalcDFT:
    encut: float = 400.0       # plane-wave energy cutoff (eV)
    ediff: float = 1e-6        # electronic convergence threshold (eV)
    ediffg: float = -0.01      # ionic convergence: negative = max force (eV/Å)
    nsw: int = 0               # number of ionic steps (0 = single-point)
    ibrion: int = -1           # ionic update algorithm (-1 = none, 2 = CG)
    isif: int = 2              # which DOF to relax (2 = ions only, 3 = ions+cell)
    ismear: int = 1            # smearing scheme (1 = Methfessel-Paxton)
    sigma: float = 0.2         # smearing width (eV)
    ispin: int = 1             # spin polarization (1 = off, 2 = on)
    algo: str = "Fast"         # electronic minimizer
    prec: str = "Normal"       # precision
    ncore: int = 1             # cores per band (parallelization)
    kpoints_mesh: list = None  # k-point mesh, e.g. [4,4,4]; None → Gamma-only 1×1×1
```

Decorated with `@as_inp_dataclass_node` so pyiron flow renders it as an interactive
input node where each field becomes an editable parameter in the GUI.

It maps 1:1 to VASP INCAR tags. The `kpoints_mesh` field lives here (not on the
structure node) because it is a calculation setting, not a property of the structure.

---

## `VaspInputResources` — internal state carrier (`vasp_new.py`)

```python
@dataclass
class VaspInputResources:
    structure: Atoms           # ASE Atoms object
    calc: InputCalcDFT         # DFT settings
    working_directory: str     # where input/output files live
    functional: str            # "PBE" or "LDA"
    potcar_lib_path: str       # path to the potpaw_PBE/ directory
    potcar_symbols: list|None  # optional override for POTCAR selection per element
    extra_incar: dict|None     # any additional INCAR tags not in InputCalcDFT
```

This is a plain Python `@dataclass` (not a workflow node). It acts as a data carrier
passed between nodes, exactly like `LammpsInputResources` in `lammps.py`. Each node
receives it, uses it, and passes it forward — so downstream nodes always have full
context about the calculation.

`structure` is ASE `Atoms` (not pymatgen `Structure`) so it is directly compatible
with the output of the `Bulk` node in `atomistic/structure/build.py`.

---

## The four workflow nodes

### 1. `CreateVaspInputResources`

```
Inputs:  structure, calc, working_directory, functional, potcar_lib_path, ...
Output:  input_resources (VaspInputResources)
```

Bundles all the inputs into a `VaspInputResources` dataclass. No files are written
here — this node just collects and validates the parameters. The defaults for
`functional`, `potcar_lib_path` and `vasp_command` are read from
`~/.pyiron_vasp_config` at import time.

---

### 2. `WriteVaspInputSet`

```
Input:   input_resources (VaspInputResources)
Output:  input_resources (VaspInputResources, unchanged)
```

Creates the working directory and writes the four VASP input files:

**POSCAR** — the crystal structure.
ASE `Atoms` is first converted to a pymatgen `Structure` via `AseAtomsAdaptor`,
then written using pymatgen's POSCAR writer (handles fractional coordinates,
selective dynamics, etc. correctly).

**INCAR** — the calculation parameters.
`_build_incar()` translates `InputCalcDFT` fields into a pymatgen `Incar` dict,
merges any `extra_incar` overrides, and writes it. This is where all the VASP
tags (ENCUT, EDIFF, ISMEAR, …) are set.

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
If `calc.kpoints_mesh` is set (e.g. `[4,4,4]`), a Gamma-centred mesh is written via
pymatgen's `Kpoints.gamma_automatic()`. Otherwise a minimal 1×1×1 Gamma-only mesh
is written as plain text.

---

### 3. `RunVaspCalculation`

```
Inputs:  input_resources, vasp_command, cores, debug
Outputs: input_resources, stdout
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
Inputs:  input_resources, vasprun_filename
Outputs: out (OutputCalcStatic), trajectory (list[Atoms]), converged (bool)
```

Reads `vasprun.xml` using `ase.io.read` with `index=":"` which returns the full
ionic trajectory as a list of ASE `Atoms` objects. Each `Atoms` in the list has
energy, forces, and stress attached via an ASE calculator object.

Results are packed into `OutputCalcStatic` (from `data.py`):
- `energy` — DFT total energy of the final ionic step (eV)
- `force` — forces on all atoms in the final step (eV/Å)
- `stress` — stress tensor of the final step (eV/Å³), if available
- `structure` — the final `Atoms` object

`trajectory` gives access to all intermediate ionic steps, useful for relaxations.

---

## Complete workflow example

```python
from core import Workflow
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.calculator.data import InputCalcDFT
from pyiron_nodes.atomistic.engine.vasp_new import (
    CreateVaspInputResources,
    WriteVaspInputSet,
    RunVaspCalculation,
    ParseVaspOutput,
)

wf = Workflow("vasp_Fe")

# 1. Build structure
wf.structure = Bulk(name="Fe", crystalstructure="bcc", a=2.87, cubic=True)

# 2. DFT settings
wf.calc = InputCalcDFT()
wf.calc.inputs.encut = 400.0
wf.calc.inputs.nsw = 0          # single-point
wf.calc.inputs.ispin = 2        # spin-polarized
wf.calc.inputs.kpoints_mesh = [8, 8, 8]

# 3. Bundle inputs
wf.resources = CreateVaspInputResources(
    structure=wf.structure.outputs.structure,
    calc=wf.calc.outputs.InputCalcDFT,
    working_directory="./fe_static",
)

# 4. Write input files
wf.write = WriteVaspInputSet(
    input_resources=wf.resources.outputs.input_resources,
)

# 5. Run VASP (blocking)
wf.run_vasp = RunVaspCalculation(
    input_resources=wf.write.outputs.input_resources,
)

# 6. Parse output
wf.parse = ParseVaspOutput(
    input_resources=wf.run_vasp.outputs.input_resources,
)

wf.run()

print("Energy:", wf.parse.outputs.out.energy, "eV")
print("Converged:", wf.parse.outputs.converged)
```

---

## Design decisions

| Decision | Reason |
|---|---|
| No `pyiron_atomistics` dependency | The old `vasp.py` depended on it for output parsing; we replaced it with ASE which is already used everywhere else |
| No pymatgen `Potcar` class | Requires `PMG_VASP_PSP_DIR` to be configured; our approach reads files directly from the path in `~/.pyiron_vasp_config` |
| `InputCalcDFT` lives in `data.py` | Consistent with `InputCalcMD`, `InputCalcMinimize` — all calculator inputs live in one place |
| `kpoints_mesh` on `InputCalcDFT` | K-point sampling is a calculation setting, not a structural property |
| ASE `Atoms` throughout | Compatible with `Bulk` and all other structure nodes; no conversion needed at boundaries |
| Plain variable names at `return` | pyiron infers output port labels from the AST of the return statement — attribute accesses like `result.stdout` don't produce valid labels |
