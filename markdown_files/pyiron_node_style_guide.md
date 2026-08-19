# pyiron_nodes — node style, inferred from review decisions

Context for anyone (human or agent) writing a new function node in `pyiron_nodes`.

Every rule below is **derived from an actual keep/replace decision** made while
resolving the 12 Tier B duplicate groups on 2026-08-19, plus the Tier A commit
`48831d6`. Where two copies of a node differed, the surviving copy is the evidence.
The "why" column names the node that settled it, so you can go read the real diff.

---

## 1. The shape of a node

```python
from core import as_function_node          # module top: the decorator, cheap deps only

@as_function_node
def NodeName(arg: type = default) -> ReturnType:
    """One-line summary.

    Longer explanation if the node has a non-obvious contract.

    Args:
        arg: what it is, units, valid range.

    Returns:
        what comes out, and under what name.
    """
    import heavy_library                    # heavy imports inside the body

    result = do_the_work(arg)
    return result                           # output port is named "result"
```

Decorator inventory in the tree today: `@as_function_node` (427), `@group_node` (28),
`@as_inp_dataclass_node` (26), `@as_out_dataclass_node` (18), `@as_macro_node` (3).

---

## 2. Output ports

A node's output port names are its public API — renaming one breaks every saved graph
wired to it. Two spellings produce them:

```python
@as_function_node("phase_data")     @as_function_node
def X(...):                         def X(...):
    df = compute()                      phase_data = compute()
    return df                           return phase_data
# port: "phase_data"                # port: "phase_data"
```

**Preference: the bare decorator with a descriptively named result variable.** Chosen
in `List5` (`list` → `list_out`), `IdealSolution`, `LinePhase`, `PlotMuPhaseDiagram`
(`plot` → `fig`), and `SaveStructures`. The tree is split 236 bare / 189 explicit, so
both are idiomatic — but when the two forms were the *only* difference, the named
variable won every time.

Use the explicit label when the natural variable name would be a poor port name
(`df`, `out`, `tmp`) and you don't want to rename the variable — that is why
`CalcPhaseDiagram` kept `@as_function_node("phase_data")` over a body that returned a
variable literally called `df`.

Multiple outputs: name each returned variable, or list the labels.

```python
@as_function_node                              # ports: phase_list, phase_dict
def PhasesFromDataFrame(dataframe):
    ...
    phase_dict = {p.name: p for p in phases.explode()}
    phase_list = list(phase_dict.values())
    return phase_list, phase_dict
```

---

## 3. Return values

| Rule | Why |
|---|---|
| **Return the thing you produced, even for side-effect nodes.** | `SaveStructures` — the version that wrote the pickle *and* `return df` beat the one that wrote and returned nothing. A node with no output port cannot be wired to anything. |
| **The return annotation must be true.** | `TransitionTemperature` — the rejected copy was annotated `-> float` but returned a matplotlib `Figure`. This was the single strongest reason to reject a version. |
| **A compute node returns its number; a plot node returns its figure.** Don't merge the two roles. | Same decision. If you need both, that is two nodes, or an explicit two-port return. |

---

## 4. Imports

**Heavy imports go inside the function body.** 256 of 446 decorated nodes do this, and
`.ruff.toml` explicitly ignores `PLC0415` ("import should be at top-level") to permit
it. Keep only the decorator and cheap typing/dataclass imports at module level.

```python
@as_function_node
def PlotEnergy(df):
    import matplotlib.pyplot as plt      # ✔ inside
    import seaborn as sns
```

**No unused imports.** `CalcPhaseDiagram` was decided partly on the rejected copy
carrying `import matplotlib.pyplot as plt` in a node that never plots. Ruff's `F401`
catches these; don't leave them for the linter to find.

---

## 5. Docstrings

**Every node gets one, and it must describe the signature that is actually there.**

- `MinMaxIndices` was *replaced* purely to gain a docstring — the incoming version had
  no other advantage.
- `IterToDataFrame` was *kept* over a fork whose docstring had been rewritten into a
  spec-sheet listing input ports (`parameters`, `calculation_parameters`), output ports
  (`loop_results`) and settings (`parallel_execution`, `max_concurrent_jobs`) that
  **did not exist in either implementation**. A confidently wrong docstring is worse
  than none.

Style is Google-ish — `Args:` / `Returns:` blocks, one entry per parameter:

```python
    """Generate index arrays for energies and forces in a DataFrame.

    The function creates a flat index range covering both energy entries and
    force components for all structures in the DataFrame.

    Args:
        df: pandas DataFrame containing `NUMBER_OF_ATOMS`.
        i_min: Minimum structure index (inclusive).
        i_max: Maximum structure index (exclusive). If None, uses total structures.
        energy_only: If True, return only energy indices; otherwise include force indices.

    Returns:
        A numpy array of selected indices.
    """
```

Document optional-with-meaning parameters, including what `None` *does*:

```python
        concentration_parameters (int, optional): how many parameters to use
                    when interpolating free energies in concentration; if not
                    given output individual phases and change the name to
                    include the concentration
```

---

## 6. Signatures

- **Type-hint every parameter, with a default where one makes sense.**
- **Use `X | None` when `None` is a real mode**, not just a sentinel —
  `PhasesFromDataFrame` takes `concentration_parameters: int | None = 1`, where `None`
  selects a genuinely different code path.
- **Don't drop configurability to simplify.** `TransitionTemperature` kept its `dmu`
  and `plot` parameters against a fork that had neither.
- **Adding an optional parameter is the preferred way to reconcile two versions.**
  `PlotConcPhaseDiagram` gained `concavity: float | None = None`, passed through as
  `alpha=concavity or 0.1`, preserving default behaviour exactly.
- `store` is injected by the decorator — you do not declare it, but callers may pass it.

---

## 7. Control flow and formatting

- **Explicit `if`/`else` over a dense ternary.** `MinMaxIndices` took
  ```python
  if energy_only:
      indices = energies
  else:
      indices = np.append(energies, forces, axis=0)
  ```
  over `indices = energies if energy_only else np.append(...)`.
- **Multi-line strings as triple-quoted blocks**, not backslash continuations —
  `SplitTrainingAndTesting` was replaced for exactly this.
- **`print()` for user-facing messages, not `IPython.display.display()`.**
  `PhasesFromDataFrame` swapped `display("Found phases:", *names)` for
  `print("Found phases:", *names, sep="\n")`; nodes should not assume a notebook.
- Formatting is enforced by **ruff** (`E,F,UP,B,SIM,I,C4,ERA,PL`) then **black**, via
  `.pre-commit-config.yaml`. Don't hand-format; let the hooks run.

---

## 8. Plotting nodes

The convention that survived review (`PlotConcPhaseDiagram`, and the module it lives
in, where every plot node follows it):

```python
@as_function_node
def PlotSomething(data, option: bool = True):
    """One line about what is plotted."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()                       # own your axes
    sns.scatterplot(data=body, x="x", y="y", ax=ax)  # thread ax= everywhere
    ax.set_xlabel("X [unit]")                      # ax.set_*, not plt.*
    ax.set_ylabel("Y [unit]")
    return fig                                     # return it; no plt.show()
```

The rejected fork of `PlotConcPhaseDiagram` created `fig, ax` and then never used
`ax` — every seaborn call and every label went through global pyplot state. It works
until something else touches the current axes. **Always thread `ax=`.**

Axis labels carry units in brackets: `"Temperature [K]"`, `"Free Energy [eV/atom]"`.

---

## 9. Helpers

Private helpers are plain functions — snake_case, no decorator, no `@as_function_node`:

```python
def make_phase(dd, temperature_parameters, concentration_parameters):
    ...
```

They live next to the node that uses them and **move with it**. When
`PhasesFromDataFrame` was replaced, its `make_phase` helper had to be ported in the
same edit — the feature (`concentration_parameters=None`) was split across both.
A node is not self-contained just because the audit lists it alone.

---

## 10. Naming

| Thing | Convention | Example |
|---|---|---|
| Node | `PascalCase`, verb-or-noun phrase | `SplitTrainingAndTesting`, `PlotMuPhaseDiagram` |
| Private helper | `snake_case`, leading `_` if module-internal | `_iterate_node`, `make_phase` |
| Output variable | descriptive; it becomes the port name | `phase_list`, `list_out`, `indices` |
| Module | `snake_case`, grouped by domain | `atomistic/thermodynamics/landau/plot.py` |

Put the node where it belongs by *domain*, not by convenience: `MinMaxIndices` reads
`df.NUMBER_OF_ATOMS` and computes ACE energy/force index ranges, so it stays in
`ml_potentials/fitting/linearfit.py` — the fork had filed it under "basic maths".

---

## 11. Worked example

A node written to every rule above.

```python
# atomistic/property/thermal.py
from core import as_function_node


@as_function_node
def HeatCapacityFromEnergies(
    temperatures: list[float],
    energies: list[float],
    n_atoms: int = 1,
    smoothing: int | None = None,
) -> float:
    """Compute the heat capacity from a sampled energy-temperature curve.

    Differentiates E(T) numerically and divides by the number of atoms, so the
    result is per-atom regardless of supercell size.

    Args:
        temperatures: temperature samples in K, ascending.
        energies: total energies in eV, one per temperature.
        n_atoms: number of atoms in the cell the energies were computed for.
        smoothing (int, optional): window length for a moving average applied
            before differentiating; if not given the raw curve is used, which
            is noisier but introduces no lag.

    Returns:
        Heat capacity in eV/K/atom, sampled at the midpoints of `temperatures`.
    """
    import numpy as np

    t = np.asarray(temperatures, dtype=float)
    e = np.asarray(energies, dtype=float)

    if smoothing is not None:
        window = np.ones(smoothing) / smoothing
        e = np.convolve(e, window, mode="same")

    heat_capacity = np.gradient(e, t) / n_atoms
    return heat_capacity
```

What to notice:

- bare decorator; the port is `heat_capacity`, readable in the node picker
- `numpy` imported inside the body, decorator imported at module top
- `smoothing: int | None` where `None` is a documented mode, not a placeholder
- explicit `if`, no ternary
- annotation matches what is returned
- docstring documents every argument, with units

And the matching plot node, if one is needed — a **separate** node, not a `plot=True`
flag bolted onto the one above:

```python
@as_function_node
def PlotHeatCapacity(temperatures: list[float], heat_capacity: list[float]):
    """Plot heat capacity against temperature."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(temperatures, heat_capacity)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Heat Capacity [eV/K/atom]")
    return fig
```

---

## 12. Quick checklist

- [ ] `@as_function_node`, bare, with a descriptively named returned variable
- [ ] Return annotation matches the actual return
- [ ] Side-effect nodes still return their product
- [ ] Heavy imports inside the body; nothing unused anywhere
- [ ] Docstring with `Args:`/`Returns:`, describing the real signature
- [ ] Type hints and sensible defaults; `X | None` only where `None` means something
- [ ] Explicit `if`/`else`; triple-quoted multi-line strings; `print`, not `display`
- [ ] Plot nodes: own `fig, ax`, thread `ax=`, label with units, `return fig`, no `plt.show()`
- [ ] Helpers are plain snake_case functions, kept beside their node
- [ ] Filed in the module matching its domain
- [ ] Ruff + black clean (pre-commit)
