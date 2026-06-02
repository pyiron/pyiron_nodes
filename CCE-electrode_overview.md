# VASP Electrochemistry (CCE) Workflow

## Overview

Performs electrochemical VASP molecular dynamics simulations using the
**Ne-CCE thermopotentiostat plugin**.
Maintains a constant electrode potential `phi0` during MD by dynamically
adjusting the number of electrons via Ne atoms acting as a charge reservoir.

The file are located in `electrochemistry/add_potential/`

## CCESetup

Prepares all input files needed for a CCE thermopotentiostat VASP run.

### Requires
| Input | Description |
|-------|-------------|
| `input_resources` | From `CreateVaspInputResources` |
| `electrode` | Bare electrode `Atoms` (without Ne) — used to compute `d_electrode` |
| `phi0` | Target electrode potential (V) |
| `path_to_plugin` | Path to `vasp_plugin-CCE.py` template |

### What It Does
- Writes **INCAR** (MD + dipole correction + plugin tags)
- Adjusts **Ne `ZVAL`** in POTCAR to match target charge `Q0`
- Writes **`vasp_plugin.py`** from template with all CCE parameters filled in

### Plugin File
The plugin template `vasp_plugin-CCE.py` is:
- Supplied in the `electrochemistry/add_potential/` directory
- Based on the original CCE implementation from:
  [https://github.com/eisenforschung/VASP-Python](https://github.com/eisenforschung/VASP-Python/tree/main)
- **Modified** from the original: all parameters that must be set per simulation
  (e.g. `phi0`, `Q0`, `tau`, cell dimensions) are replaced with **`{placeholder}`
  format strings** instead of hardcoded numbers, so `CCESetup` can fill them
  in automatically via Python's `str.format()`

### ⚠️ Be Careful
- **VASP ≥ 6.5 required** — earlier versions do not support Python plugins
- `ionic_update_algorithm` in `InputCalcDFT` **must** be `MolecularDynamics`
- Structure **must contain Ne atoms** — they are the CCE charge reservoir
- Cell **must be orthogonal** — plugin requires `a3 ⊥ a1, a2`
- `path_to_plugin` must point to a valid template file before running
- `working_directory` must already exist (i.e. `WriteVaspInputSet` must have run first)

---

## ParsePotential

Reads the three output files written by the CCE plugin after the VASP run: `el_pot_z.dat`, `Q.dat`, `phi.dat`

### Outputs
| Output | Shape | Description |
|--------|-------|-------------|
| `electrostatic_potential_z_2d` | `[n_steps, nz]` | Potential profile along Z per MD step |
| `charge_list` | `[n_steps]` | Electrode charge per step |
| `pot_list` | `[n_steps]` | Electrode potential per step |
