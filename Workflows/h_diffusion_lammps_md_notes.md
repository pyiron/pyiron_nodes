# H diffusion in Al via LAMMPS MD — workflow notes

Reference workflow: `h_diffusion_lammps_md.py`

---

## Pipeline overview

```
Bulk → Repeat → AddInterstitialH
                      │
              FixSpecies   ListPotentials → pick_element
                      │              │
              CreateLammpsStructure ──┘
                      │
              CreateLammpsMDInput ← InputCalcMD.outputs.output
                      │
              RunLammpsCalculation  (store=True)
                      │
              ParseLammpsOutput  →  .outputs.out  (OutputCalcMD dataclass)
                      │
              OutputCalcMD (splitter node)
                      │
          ComputeMSD → DiffusionConstant
                              │
                          msd_plot, h_pos_plot, temp_plot
```

---

## Key issues and how to fix them

### 1. Passing `@as_inp_dataclass_node` to another function node

**Bug:** `TypeError: asdict() should be called on dataclass instances`

`InputCalcMD` is decorated with `@as_inp_dataclass_node`, which makes it a
**Node subclass**. When you write:

```python
wf.lammps_input = CreateLammpsMDInput(calc_dataclass=wf.md_params)
```

the edge inference rule sees `calc_dataclass: InputCalcMD` as a Node-typed
port and routes the edge through the `"self"` handle — the receiving function
gets the live **Node object**, not the assembled dataclass. `asdict()` then
fails because a Node is not a dataclass.

**Fix:** always use `.outputs.output` to extract the pure dataclass:

```python
wf.lammps_input = CreateLammpsMDInput(
    calc_dataclass=wf.md_params.outputs.output,   # ✓
)
```

This applies to **every** `@as_inp_dataclass_node` passed to any
`@as_function_node` that calls `asdict()` or accesses fields directly.

### 2. `ParseLammpsOutput` output port name

`ParseLammpsOutput` returns the variable named `out`:

```python
out = OutputCalcMD.pure_dataclass()
...
return out          # → output port is called "out"
```

Use the explicit port reference when connecting to custom nodes:

```python
wf.msd = ComputeMSD(md_output=wf.parsed_output.outputs.out)
```

Passing the node directly (`wf.parsed_output`) also works here because it has
a single output, but the explicit reference is clearer.

### 3. `RunLammpsCalculation` output ports

The function returns a **tuple** `(io_bundle, output)`, giving two ports:

| Port | Content |
|------|---------|
| `io_bundle` | `LammpsIOBundle` — needed by `ParseLammpsOutput` |
| `output` | stdout string from LAMMPS (useful for debugging) |

```python
wf.parsed_output = ParseLammpsOutput(
    io_bundle=wf.lammps_run.outputs.io_bundle   # ✓
)
```

### 4. `FixSpecies` must receive the Al+H structure

`FixSpecies` must be applied **after** adding H, so LAMMPS gets the correct
species ordering for both elements. `ListPotentials` must also receive the
Al+H structure to filter for potentials that cover both Al and H — passing
only the pure Al supercell returns Al-only potentials.

### 5. MSD computation: use `unwrapped_positions`

`OutputCalcMD` stores both `positions` (PBC-wrapped) and
`unwrapped_positions`. For MSD you **must** use `unwrapped_positions`
(shape `(n_frames, n_atoms, 3)`) to avoid artefacts when H jumps across a
periodic boundary.

```python
unwrapped_pos = np.array(md_output.unwrapped_positions)
```

### 6. Diffusion constant from MSD

Einstein relation (3-D):

```
D = slope(MSD vs t) / 6
```

- Fit only the **linear diffusive regime** — skip the early ballistic part
  (default: first 20 % of frames via `fit_start_fraction=0.2`).
- Unit conversion: **1 Å²/ps = 1×10⁻⁸ m²/s**.
- At 800 K, 100 ps (100 000 steps, 1 fs timestep) is sufficient to see
  diffusive behaviour. At lower temperatures the required simulation time
  grows rapidly.

### 7. `frac_pos` is relative to the unit cell, not the supercell

`AddInterstitialH` computes the Cartesian position via `np.dot(frac_pos, cell)`.
The `cell` attribute of the **supercell** is `repeat_scalar` times larger than
the unit cell, so omitting the correction places H at the wrong site:

```python
# ✗ wrong — lands at 25% of the supercell edge (~3 Å for a 3×3×3 of Al)
cart_pos = np.dot([0.25, 0.0, 0.0], supercell.cell)

# ✓ correct — recover the unit cell first
unit_cell = supercell.cell / repeat_scalar
cart_pos = np.dot([0.25, 0.0, 0.0], unit_cell)   # ~1 Å, actual interstitial
```

`AddInterstitialH` accepts `repeat_scalar: int = 1` for this purpose; always
pass it to match the value used in `Repeat`:

```python
wf.al_supercell    = Repeat(structure=wf.al_bulk, repeat_scalar=3)
wf.al_h_structure  = AddInterstitialH(structure=wf.al_supercell, repeat_scalar=3)
```

### 8. Plotting: `Plot` returns a figure, not axes

`Plot` takes an optional `axis` input but returns `figure`. Chaining three
`Plot` calls onto the same axes requires a `Subplot` node to create the shared
axes first. For a multi-panel figure it is simpler to write a single custom
node:

```python
@as_function_node
def PlotHPositions(md_output, md_input, species_symbol: str = "H"):
    fig, axes = plt.subplots(3, 1, sharex=True)
    ...
    figure = fig
    return figure
```

`MultiPlot` from `pyiron_nodes.plotting` overlays multiple series on one axes
but does not support per-series colours or labels — not suitable for x/y/z
traces.

### 9. Free energy surface and migration barrier

#### Physical idea

The H trajectory samples the FCC interstitial landscape according to the Boltzmann distribution.
Binning the positions and inverting gives the 3-D free energy surface:

```
F(x,y,z) = -kT ln P(x,y,z)
```

The global minimum is the tetrahedral (T) site; the saddle between adjacent T-sites
runs through the octahedral (O) site.  The T→O energy difference is the migration
barrier, which enters the Arrhenius diffusion rate:

```
D ≈ ν* a² exp(-ΔF / kT)
```

This provides a **barrier estimate from a single MD run** without NEB or a temperature sweep.

#### Symmetry augmentation

For a short trajectory, the raw histogram is noisy.  Every sampled H position is
crystallographically equivalent to all positions obtained by applying the full
space-group symmetry of the host lattice.  Applying all *N_sym* operations (192
for FCC Al, space group Fm-3m) multiplies the effective sample count by *N_sym*
at negligible cost.

Pipeline:
1. `FoldPositionsToUnitCell` — convert unwrapped Cartesian positions to fractional
   coords in the conventional FCC unit cell (modulo 1.0 fold-back).
2. `AugmentWithSymmetry` — apply every (R, t) from `spglib.get_symmetry` (falls back
   to the 48 Oh point-group operations).  Stack to one large array.
3. `ComputeFreeEnergySurface` — 3-D histogram → Boltzmann inversion → F in eV.
4. `ExtractMigrationBarrier` — interpolate F on T→O straight-line paths using
   minimum-image convention; return the minimum peak.
5. `PlotFreeEnergySurface` — contourf slices at z ≈ 0.25, 0.50, 0.75 fractional.

#### Key implementation details

- `spglib.get_symmetry` operates in fractional space with **column** vectors:
  `x' = R @ x + t`.  For row-vector arrays `(N, 3)` the equivalent is
  `new_pos = positions @ R.T + t`.
- After applying each operation, fold back with `% 1.0` to stay in the unit cell.
- `RegularGridInterpolator` is used for sub-voxel path sampling with
  `bounds_error=False` to handle the periodic boundary cleanly.
- The barrier is the **minimum** peak over the four O-site candidates
  `(0.5,0.5,0.5)`, `(0.5,0.5,0)`, `(0.5,0,0.5)`, `(0,0.5,0.5)` — these are
  all reachable from T(0.25,0.25,0.25) without leaving `[0,1)³`.
- `n_bins=30` gives voxels of ≈ 0.135 Å for Al (a ≈ 4.05 Å), which is fine for
  a qualitative barrier estimate; use `n_bins=50` for a smoother surface.

#### Migration path plot (`PlotMigrationPath`)

**Why 3-D grid interpolation fails for the 1-D profile:**  
When a coarse 3-D histogram (n_bins=30) is Boltzmann-inverted and the resulting
`free_energy` array is interpolated along the T→O→T' line, bins in the O-site
region that have zero or very few counts are set to `np.nan` and then replaced
by a large fill value (2 × F_max).  Linear interpolation between the last populated
bin and this fill region produces a **linear rise → flat plateau → linear fall**
(trapezoid) rather than a smooth barrier.

**Fix: direct 1-D projection of the augmented positions.**

1. **T-site**: found as the mode (argmax) of the 3-D histogram of `augmented_positions`,
   not as `nanargmin(free_energy)`.  Both are equivalent in theory, but the histogram
   mode is unaffected by the `F_filled` substitution used in grid interpolation.
2. **Cylindrical tube selection**: only positions within a cylinder of radius
   `tube_fraction × d(T→O)` (default: 35%) around the T→O→T' axis are kept.
   This excludes T-site atoms that lie on parallel migration paths and would
   otherwise dilute the O-site signal.
3. **1-D histogram** with `n_bins_1d=100` along the reaction coordinate — ~14× more
   bins than along the equivalent direction in the 3-D grid — gives sufficient
   resolution to show the well/barrier shape.
4. **Gaussian smoothing** (`sigma=2` bins) suppresses counting noise without
   shifting the peak.
5. **Robust minimum-image**: `np.floor(delta + 0.5)` replaces `np.round(delta)` to
   avoid banker's rounding errors at exactly ±0.5 fractional components.

**Cosine extrapolation** for the under-sampled T'-basin:

At 800 K with a barrier of ~0.4 eV (≈ 6 kT), the probability of crossing the
saddle in a single 100 ps run is small.  The T'-basin bins remain empty and the
raw data curve simply ends at the last visited bin, showing only the rising half.

The fix: fit the sampled rising edge to

```
F(rc) = (ΔF/2) × (1 − cos(π·rc/L))
```

This function is the simplest model that satisfies:
- F(0) = 0  (T-site minimum)
- F(L) = ΔF (saddle at T→O distance L)
- F(2L) = 0 (T'-site minimum, identical to T by symmetry)
- F'(0) = F'(L) = F'(2L) = 0  (zero force at all three critical points)

`scipy.optimize.curve_fit` fits ΔF and L simultaneously to the sampled data.
The fitted barrier ΔF is typically more accurate than `max(F_sm[valid])` because
the cosine constrains the shape even when the data doesn't reach the peak.
The full double-well is then rendered as a dashed red line, with the MD data
overlaid as a solid blue line.

If the fit diverges (rare), the node falls back to `ΔF = 1.5 × F_max_raw` and
`L = d(T→O)` from the nominal O-site geometry.

Additional inputs required vs the original version:
- `augmented_positions` (from `AugmentWithSymmetry`) — needed for direct projection
- `md_input` (from `InputCalcMD`) — provides temperature for Boltzmann inversion

#### Limitations

- A **short trajectory** (< 100 ps) may not visit all equivalent T-sites and the
  barrier will be overestimated (under-sampled basin walls appear high).
- The barrier from Boltzmann inversion is a **free energy** barrier, not a 0-K
  PES barrier — it includes entropic and thermal contributions.
- For a quantitative Arrhenius analysis, run multiple temperatures and fit
  `ln D` vs `1/T` (see "Extending the workflow" below).

---

## Design decisions

| Decision | Reason |
|----------|--------|
| 3×3×3 supercell (108 Al + 1 H) | Large enough to avoid self-interaction of H with its periodic images |
| 800 K | High enough for H to diffuse on a 100 ps timescale with a typical EAM potential |
| `store=True` on `RunLammpsCalculation` | MD is expensive; caching avoids re-running when downstream nodes change |
| No `store` on template nodes | Guide requirement: template nodes passed to loops must not have `store` |
| `frac_pos=[0.25, 0.0, 0.0]` | Tetrahedral interstitial site in FCC Al; coordinates are relative to the **unit cell**, not the supercell |
| `repeat_scalar` passed to `AddInterstitialH` | Required to recover the unit cell matrix (`unit_cell = supercell_cell / repeat_scalar`) before computing the Cartesian position via `np.dot(frac_pos, unit_cell)` |
| `FoldPositionsToUnitCell` uses `al_bulk.cell` (unit cell), not supercell | Gives the correct 1×1×1 periodic unit for fold-back; supercell cell would give only 1/27 of the equivalent sites |
| `n_bins=30` in `ComputeFreeEnergySurface` | ~0.135 Å voxels in Al; good for qualitative barriers; increase to 50 for smoother surfaces if the trajectory is long |
| Minimum-image convention in `ExtractMigrationBarrier` | Required so the T→O path does not wrap around the long way when O is near a cell boundary |

---

## Extending the workflow

### Temperature-dependent diffusion (Arrhenius)

Wrap the MD + diffusion analysis in a group node and sweep over temperatures
with `IterToDataFrame`:

```python
@group_node("diffusion_constant")
def DiffusionAtTemperature(temperature: float = 800.0, ...):
    inner = Workflow("DiffusionAtTemperature")
    inner.md_params = InputCalcMD(temperature=temperature, ...)
    ...
    return inner.diffusion   # single output

wf.temperatures = Linspace(start=500.0, stop=1200.0, num=8)
wf.template = DiffusionAtTemperature(...)
wf.sweep = IterToDataFrame(
    node=wf.template,
    input_label="temperature",
    values=wf.temperatures.outputs.linspace,
)
```

Fit `ln(D)` vs `1/T` to extract the activation energy.

### Multiple H atoms

Change `AddInterstitialH` to insert N atoms at distinct interstitial sites and
average the MSD over all of them in `ComputeMSD` (already handled — it averages
over all atoms matching `species_symbol`).

### Using a different potential

Change the `index` in `pick_element` to select a different potential from the
iprpy database, or pass a potential name string directly to
`CreateLammpsStructure`. Check available potentials by running
`ListPotentials(structure=wf.al_h_structure)` standalone first.

### Longer / higher-resolution trajectories

Adjust `InputCalcMD`:
- `n_ionic_steps`: total MD steps
- `n_print`: output frequency (more frames = smoother MSD but larger dump file)
- `time_step`: 0.5 fs for high-T runs where H moves fast

### Velocity autocorrelation function (VACF)

An alternative route to D that converges faster than MSD for short trajectories:

```
D = (1/3) ∫₀^∞ <v(0)·v(t)> dt
```

`OutputCalcMD.velocities` (shape `(n_frames, n_atoms, 3)`) provides the
per-atom velocities needed.
