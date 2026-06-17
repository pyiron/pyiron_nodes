"""
pyiron_nodes/atomistic/structure/defect_phases.py
──────────────────────────────────────────────────
Nodes for computing defect formation energies from a structure/energy
DataFrame.

The ``structure`` column of the input DataFrame may contain either plain
ASE :class:`~ase.atoms.Atoms` objects **or** picklable
:class:`~pyiron_nodes.atomistic.structure._atoms.OutputAtoms` dataclass
instances.  All public functions accept both transparently via the
:func:`~pyiron_nodes.atomistic.structure._atoms.to_ase` helper.

Typical workflow
----------------
::

    wf.AddElementCountColumns      = AddElementCountColumns(df=...)
    wf.AddDefectConcentrationColumns = AddDefectConcentrationColumns(
        df=wf.AddElementCountColumns.outputs.df)
    wf.AddDefectFormationEnergyColumn = AddDefectFormationEnergyColumn(
        df=wf.AddDefectConcentrationColumns.outputs.df,
        chemical_potentials={"Al": -3.74, "Mg": -1.59})
"""

from __future__ import annotations

from core import as_function_node
import numpy as np
import pandas as pd

from pyiron_nodes.atomistic.structure._atoms import (
    OutputAtoms,
    to_ase,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_element_counts(structure) -> dict[str, int]:
    """
    Return a ``{symbol: count}`` dict for *structure*.

    Parameters
    ----------
    structure : Atoms or OutputAtoms
        Accepts both raw ASE :class:`~ase.atoms.Atoms` and picklable
        :class:`OutputAtoms` instances.

    Returns
    -------
    dict[str, int]
        E.g. ``{"Al": 107, "Mg": 1}``.
    """
    from collections import Counter

    # OutputAtoms stores symbols directly as a list — no ASE needed.
    # Fall back to to_ase() for any other type.
    if isinstance(structure, OutputAtoms):
        return dict(Counter(structure.symbols))

    atoms = to_ase(structure)
    return dict(Counter(atoms.get_chemical_symbols()))


def _all_elements(structure_series: pd.Series) -> list[str]:
    """
    Collect every element symbol present across all structures in
    *structure_series*, sorted alphabetically.

    Parameters
    ----------
    structure_series : pd.Series
        Series of :class:`~ase.atoms.Atoms` or :class:`OutputAtoms`
        objects.

    Returns
    -------
    list[str]
        Sorted list of unique element symbols,
        e.g. ``["Al", "Mg", "Si"]``.
    """
    elements: set[str] = set()
    for structure in structure_series:
        elements.update(_get_element_counts(structure).keys())
    return sorted(elements)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@as_function_node
def AddElementCountColumns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extend a structure/energy DataFrame with one ``n_<element>`` column
    per chemical species found across all structures.

    Each new column gives the number of atoms of that element in the
    corresponding row.  Missing elements are filled with ``0``.

    The input DataFrame must contain at least:

    * ``structure`` – :class:`~ase.atoms.Atoms` or
      :class:`OutputAtoms` objects
    * ``energy``    – total energy (float, in eV)

    Parameters
    ----------
    df : pd.DataFrame
        Input frame with ``structure`` and ``energy`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with additional ``n_<element>`` integer columns,
        e.g. ``n_Al``, ``n_Mg``.

    Raises
    ------
    ValueError
        If *df* is missing the ``structure`` column.

    Examples
    --------
    >>> result = AddElementCountColumns(df=df)
    >>> print(result[["energy", "n_Al"]])
       energy  n_Al
    0  -100.0   108
    1   -99.1   107
    """
    if "structure" not in df.columns:
        raise ValueError(
            "Input DataFrame must contain a 'structure' column of "
            "ASE Atoms or OutputAtoms objects."
        )

    elements = _all_elements(df["structure"])

    result = df.copy()
    for el in elements:
        result[f"n_{el}"] = result["structure"].apply(
            lambda s, e=el: _get_element_counts(s).get(e, 0)
        )
        result[f"n_{el}"] = result[f"n_{el}"].astype(int)

    df = result
    return df


@as_function_node
def AddDefectConcentrationColumns(
    df: pd.DataFrame,
    pristine_row: int = 0,
) -> pd.DataFrame:
    """
    Extend the element-count DataFrame with ``delta_n_<element>`` columns
    giving the *change* in atom count relative to a reference (pristine)
    row.

    This is the :math:`\\Delta n_i` term needed for the defect formation
    energy:

    .. math::

        E^f = E_{\\text{defect}} - E_{\\text{pristine}}
              - \\sum_i \\Delta n_i \\, \\mu_i

    The input DataFrame must already contain ``n_<element>`` columns
    (call :func:`AddElementCountColumns` first).

    Parameters
    ----------
    df : pd.DataFrame
        Frame with ``structure``, ``energy``, and ``n_<element>`` columns.
    pristine_row : int, optional
        *Relative* row index of the pristine (reference) structure within
        *df*.  Follows standard Python list semantics: ``0`` = first row,
        ``-1`` = last row.  Default ``0``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with additional ``delta_n_<element>`` integer columns.

    Raises
    ------
    ValueError
        If no ``n_<element>`` columns are found (call
        :func:`AddElementCountColumns` first).
    IndexError
        If *pristine_row* is out of range.

    Examples
    --------
    >>> result = AddDefectConcentrationColumns(df=df, pristine_row=0)
    >>> print(result[["energy", "n_Al", "delta_n_Al"]])
       energy  n_Al  delta_n_Al
    0  -100.0   108           0
    1   -99.1   107          -1
    """
    n_columns = [c for c in df.columns if c.startswith("n_")]
    if not n_columns:
        raise ValueError(
            "No 'n_<element>' columns found. "
            "Call AddElementCountColumns before "
            "AddDefectConcentrationColumns."
        )

    all_rows = list(df.index)
    if not all_rows:
        raise IndexError("DataFrame is empty.")
    try:
        ref_abs = all_rows[pristine_row]
    except IndexError:
        raise IndexError(
            f"pristine_row={pristine_row} is out of range for a "
            f"DataFrame with {len(all_rows)} row(s)."
        )

    result = df.copy()
    for col in n_columns:
        el = col[len("n_") :]
        ref_count = int(df.loc[ref_abs, col])
        result[f"delta_n_{el}"] = (result[col] - ref_count).astype(int)

    df = result
    return df


@as_function_node
def ComputeChemicalPotentials(
    df: pd.DataFrame,
    pristine_row: int = 0,
    mu_reference: float | list | np.ndarray | None = None,
    mu_reference_element: str | None = None,
) -> dict[str, float | np.ndarray]:
    """
    Derive chemical potentials from the DFT energies stored in *df*,
    optionally overriding or supplying one potential as a scalar or array.

    For a **unary** reference structure (only one element present) the
    chemical potential of that element is simply the energy per atom:

    .. math::

        \\mu_A = \\frac{E_\\text{pristine}}{n_A}

    For the **second** chemical potential (or any element whose unary
    reference is not present in *df*), the value can be supplied via
    *mu_reference* and *mu_reference_element*.  This value may be a
    plain ``float`` **or** an array-like (e.g. a :class:`numpy.ndarray`
    of values sampled along a chemical-potential range), enabling
    phase-diagram sweeps without looping in user code.

    When *mu_reference* is an array the returned dictionary is in
    **array mode**: all values are :class:`numpy.ndarray` of the same
    shape, with unary potentials broadcast to that shape.

    The function scans *every* row of *df* and, for each row that
    contains atoms of **exactly one** element, records

    .. math::

        \\mu_\\text{element} = \\frac{E_\\text{row}}{n_\\text{element}}

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that must contain:

        * ``energy``          – total DFT energy (eV)
        * ``n_<element>``     – atom counts (produced by
          :func:`AddElementCountColumns`)

        An optional ``name`` column is used only for warning messages.
    pristine_row : int, optional
        *Relative* row index of the pristine reference structure,
        following Python list semantics (``0`` = first, ``-1`` = last).
        Default ``0``.
    mu_reference : float or list or numpy.ndarray or None, optional
        Chemical potential for the element specified by
        *mu_reference_element*.  May be:

        * ``None``  – no externally supplied potential; the element must
          have a unary row in *df* or it will be absent from the output.
        * ``float`` – single scalar value (eV).
        * array-like – sequence of values (eV); all output potentials
          are broadcast to the same shape, enabling sweeps over the
          chemical-potential range, e.g.::

              mu_reference = np.linspace(-2.0, -1.0, 100)

    mu_reference_element : str or None, optional
        Element symbol whose chemical potential is supplied via
        *mu_reference* (e.g. ``"Mg"``).  Must be provided when
        *mu_reference* is not ``None``; ignored otherwise.

    Returns
    -------
    dict[str, float | numpy.ndarray]
        Mapping of element symbol → chemical potential (eV).

        * **Scalar mode** (``mu_reference`` is ``None`` or a plain
          ``float``): values are ``float``.
        * **Array mode** (``mu_reference`` is array-like): values are
          :class:`numpy.ndarray` of the same shape as *mu_reference*,
          with unary potentials broadcast accordingly.

        If a unary reference for a given element is found more than once
        in *df* the **last** occurrence wins (a ``UserWarning`` is
        emitted).

    Raises
    ------
    ValueError
        If *df* is missing the ``energy`` column or all ``n_<element>``
        columns.
    ValueError
        If *mu_reference* is provided but *mu_reference_element* is
        ``None`` or not a string.
    ValueError
        If *mu_reference_element* is provided but not found among the
        elements present in *df*.
    IndexError
        If *pristine_row* is out of range.

    Warns
    -----
    UserWarning
        When the same element's chemical potential (from a unary row) is
        overwritten by a later unary row.
    UserWarning
        When no unary row is found for an element and no external value
        was supplied via *mu_reference*, meaning that element's potential
        will be absent from the output.
    UserWarning
        When *mu_reference_element* already has a unary row in *df*;
        the externally supplied value takes precedence.

    Examples
    --------
    Pure-Al pristine row → scalar μ_Al; scalar μ_Mg supplied externally:

    >>> mu = ComputeChemicalPotentials(
    ...     df=df,
    ...     mu_reference=-1.59,
    ...     mu_reference_element="Mg",
    ... )
    >>> mu
    {'Al': -3.74, 'Mg': -1.59}

    Array μ_Mg for a chemical-potential sweep; μ_Al broadcast to same
    shape:

    >>> mu = ComputeChemicalPotentials(
    ...     df=df,
    ...     mu_reference=np.linspace(-2.0, -1.0, 100),
    ...     mu_reference_element="Mg",
    ... )
    >>> mu["Al"].shape, mu["Mg"].shape
    (100,), (100,)
    """
    import warnings

    # ------------------------------------------------------------------
    # 1.  Validate columns
    # ------------------------------------------------------------------
    if "energy" not in df.columns:
        raise ValueError("Input DataFrame must contain an 'energy' column.")

    n_columns = [c for c in df.columns if c.startswith("n_")]
    if not n_columns:
        raise ValueError(
            "No 'n_<element>' columns found. "
            "Call AddElementCountColumns before "
            "ComputeChemicalPotentials."
        )

    # ------------------------------------------------------------------
    # 2.  Validate mu_reference / mu_reference_element pair
    # ------------------------------------------------------------------
    if mu_reference is not None:
        if not isinstance(mu_reference_element, str) or not mu_reference_element:
            raise ValueError(
                "mu_reference_element must be a non-empty string when "
                "mu_reference is provided."
            )

    elements = [c[len("n_") :] for c in n_columns]

    if mu_reference_element is not None and mu_reference is not None:
        if mu_reference_element not in elements:
            raise ValueError(
                f"mu_reference_element='{mu_reference_element}' is not "
                f"present in the DataFrame columns. "
                f"Available elements: {elements}."
            )

    # ------------------------------------------------------------------
    # 3.  Resolve pristine row
    # ------------------------------------------------------------------
    all_rows = list(df.index)
    if not all_rows:
        raise IndexError("DataFrame is empty.")

    try:
        ref_abs = all_rows[pristine_row]
    except IndexError:
        raise IndexError(
            f"pristine_row={pristine_row} is out of range for a "
            f"DataFrame with {len(all_rows)} row(s)."
        )

    # ------------------------------------------------------------------
    # 4.  Determine scalar vs. array mode from mu_reference
    # ------------------------------------------------------------------
    array_mode = mu_reference is not None and np.ndim(mu_reference) > 0

    if array_mode:
        mu_ref_array = np.asarray(mu_reference, dtype=float)
    else:
        mu_ref_array = None

    # ------------------------------------------------------------------
    # 5.  Scan every row; record μ for unary rows
    # ------------------------------------------------------------------
    # Store scalar potentials first; broadcast to arrays at the end.
    scalar_potentials: dict[str, float] = {}

    for abs_idx in all_rows:
        counts = {el: int(df.loc[abs_idx, f"n_{el}"]) for el in elements}
        present = {el: n for el, n in counts.items() if n > 0}

        if len(present) == 0:
            continue

        energy = float(df.loc[abs_idx, "energy"])

        if len(present) == 1:
            element, n_atoms = next(iter(present.items()))

            # External value takes precedence over unary rows for the
            # reference element.
            if element == mu_reference_element and mu_reference is not None:
                warnings.warn(
                    f"A unary row for '{element}' (index {abs_idx}) "
                    "was found in df, but the externally supplied "
                    "mu_reference takes precedence.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            if element in scalar_potentials:
                row_label = (
                    str(df.loc[abs_idx, "name"])
                    if "name" in df.columns
                    else f"row_{abs_idx}"
                )
                warnings.warn(
                    f"Chemical potential for '{element}' is being "
                    f"overwritten by {row_label} "
                    f"(index {abs_idx}). "
                    "Keep only one unary reference per element, or "
                    "ensure the desired reference appears last.",
                    UserWarning,
                    stacklevel=2,
                )

            scalar_potentials[element] = energy / n_atoms

    # ------------------------------------------------------------------
    # 6.  Inject externally supplied potential
    # ------------------------------------------------------------------
    if mu_reference is not None:
        if array_mode:
            # Stored separately; will be added during broadcast step.
            pass
        else:
            scalar_potentials[mu_reference_element] = float(mu_reference)

    # ------------------------------------------------------------------
    # 7.  Warn about elements with no potential at all
    # ------------------------------------------------------------------
    for el in elements:
        has_scalar = el in scalar_potentials
        is_ref_el = el == mu_reference_element and mu_reference is not None
        if not has_scalar and not is_ref_el:
            warnings.warn(
                f"No unary reference row found for element '{el}' and "
                "no external value was supplied via mu_reference. "
                "Its chemical potential will be absent from the output. "
                "Add a row containing only that element to *df*, or "
                "supply its chemical potential via mu_reference.",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # 8.  Build final output (broadcast scalars to arrays if needed)
    # ------------------------------------------------------------------
    chemical_potentials: dict[str, float | np.ndarray] = {}

    if array_mode:
        # Every scalar potential is broadcast to the shape of mu_ref_array
        for el, val in scalar_potentials.items():
            chemical_potentials[el] = np.full(mu_ref_array.shape, val, dtype=float)
        # Add the externally supplied array potential
        chemical_potentials[mu_reference_element] = mu_ref_array.copy()
    else:
        chemical_potentials = dict(scalar_potentials)

    return chemical_potentials


@as_function_node
def ComputeDefectFormationEnergy(
    df: pd.DataFrame,
    chemical_potentials: dict[str, float | list | np.ndarray],
    pristine_row: int = 0,
) -> dict[str, float | np.ndarray]:
    """
    Compute defect formation energies for every row in *df* and return
    them together with the chemical-potential x-axis values.

    The formation energy is defined as:

    .. math::

        E^f_\\text{defect} = E_\\text{defect} - E_\\text{pristine}
                             - \\sum_i \\Delta n_i \\, \\mu_i

    The returned dictionary always contains the special key
    ``"mu_values"`` holding the x-axis data for plotting.  The x-axis
    is taken from the **first array-valued** entry in
    *chemical_potentials* (i.e. the varying / second-phase potential).
    Scalar potentials that are constant across all calculations are
    intentionally excluded from the x-axis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least an ``energy`` column and either
        ``delta_n_<element>`` columns or ``n_<element>`` columns.
    chemical_potentials : dict[str, float | list | numpy.ndarray]
        Mapping of element symbol to its chemical potential.  Values may
        be plain floats **or** array-like objects of identical length,
        e.g.::

            # μ_Al is constant (pristine / elemental reference),
            # μ_Mg varies → used as x-axis
            {"Al": np.full(50, -3.74), "Mg": np.linspace(-2, -1, 50)}

        Only elements present as ``delta_n_*`` (or ``n_*``) columns in
        *df* are used; extra keys are silently ignored.  Elements present
        in *df* but absent from *chemical_potentials* default to ``0``.
    pristine_row : int, optional
        Row index (Python list semantics) of the pristine reference
        structure, used only when ``delta_n_*`` columns are absent.
        Default ``0``.

    Returns
    -------
    dict[str, float | numpy.ndarray]
        Mapping of defect name → formation energy, **plus** the special
        key ``"mu_values"`` → x-axis chemical-potential array.

        ``"mu_values"`` is the **first array-valued** entry in
        *chemical_potentials* whose values are **not all equal**
        (i.e. the genuinely varying potential).  If all potentials are
        scalar, ``"mu_values"`` is ``0.0``.

    Raises
    ------
    ValueError
        If *df* contains neither ``delta_n_*`` nor ``n_*`` columns.
    ValueError
        If *df* is missing the ``energy`` column.
    IndexError
        If *pristine_row* is out of range (fallback path only).

    Examples
    --------
    >>> mu_Mg = np.linspace(-2.0, -1.0, 10)
    >>> result = ComputeDefectFormationEnergy(
    ...     df=df,
    ...     chemical_potentials={
    ...         "Al": np.full(10, -3.74),   # constant  → NOT x-axis
    ...         "Mg": mu_Mg,                # varying   → x-axis
    ...     },
    ... )
    >>> result["mu_values"] is mu_Mg
    True
    """
    # ------------------------------------------------------------------
    # 1. Validate required columns
    # ------------------------------------------------------------------
    if "energy" not in df.columns:
        raise ValueError("Input DataFrame must contain an 'energy' column.")

    delta_columns = [c for c in df.columns if c.startswith("delta_n_")]
    n_columns = [c for c in df.columns if c.startswith("n_")]

    if not delta_columns and not n_columns:
        raise ValueError(
            "No 'delta_n_<element>' or 'n_<element>' columns found. "
            "Call AddElementCountColumns and "
            "AddDefectConcentrationColumns before "
            "ComputeDefectFormationEnergy."
        )

    # ------------------------------------------------------------------
    # 2. Build / retrieve delta_n values
    # ------------------------------------------------------------------
    all_rows = list(df.index)

    try:
        ref_abs = all_rows[pristine_row]
    except IndexError:
        raise IndexError(
            f"pristine_row={pristine_row} is out of range for a "
            f"DataFrame with {len(all_rows)} row(s)."
        )

    if delta_columns:
        delta_df = df[delta_columns].copy()
        delta_df.columns = [c[len("delta_n_") :] for c in delta_columns]
    else:
        delta_df = pd.DataFrame(index=df.index)
        for col in n_columns:
            el = col[len("n_") :]
            ref_count = int(df.loc[ref_abs, col])
            delta_df[el] = (df[col] - ref_count).astype(int)

    # ------------------------------------------------------------------
    # 3. Normalise chemical potentials; identify the varying x-axis
    #
    #    A potential qualifies as the x-axis when it is array-valued AND
    #    its values are not all identical (i.e. it genuinely varies).
    #    The first such entry in iteration order wins.
    # ------------------------------------------------------------------
    mu: dict[str, float | np.ndarray] = {}
    mu_values: float | np.ndarray = 0.0
    array_mode = False

    for element, value in chemical_potentials.items():
        if np.ndim(value) == 0:
            # Plain scalar – constant potential (e.g. pristine reference)
            mu[element] = float(value)
        else:
            arr = np.asarray(value, dtype=float)
            mu[element] = arr
            # Only treat as x-axis if it actually varies
            if not array_mode and not np.all(arr == arr[0]):
                mu_values = arr
                array_mode = True

    # ------------------------------------------------------------------
    # 4. Reference (pristine) energy
    # ------------------------------------------------------------------
    e_pristine = float(df.loc[ref_abs, "energy"])

    # ------------------------------------------------------------------
    # 5. Compute formation energies row by row
    # ------------------------------------------------------------------
    def _row_label(abs_idx: int) -> str:
        if "name" in df.columns:
            name = df.loc[abs_idx, "name"]
            if pd.notna(name) and str(name).strip():
                return str(name)
        if abs_idx == ref_abs:
            return "pristine"
        return f"row_{abs_idx}"

    result: dict[str, float | np.ndarray] = {"mu_values": mu_values}

    for abs_idx in all_rows:
        e_defect = float(df.loc[abs_idx, "energy"])
        delta_e = e_defect - e_pristine

        if array_mode:
            correction: float | np.ndarray = np.zeros_like(mu_values, dtype=float)
        else:
            correction = 0.0

        for element in delta_df.columns:
            dn = int(delta_df.loc[abs_idx, element])
            mu_i = mu.get(element, 0.0)
            correction = correction + dn * mu_i  # type: ignore[operator]

        result[_row_label(abs_idx)] = delta_e - correction

    return result


@as_function_node
def PlotDefectFormationEnergy(
    formation_energies: dict[str, float | np.ndarray],
    mu_label: str = "μ (eV)",
    ef_label: str = "Formation energy (eV)",
    title: str = "Defect formation energy diagram",
    exclude_keys: list[str] | None = None,
) -> object:
    """
    Plot defect formation energies as a function of chemical potential.

    Reads the ``"mu_values"`` key from *formation_energies* as the
    x-axis and plots one line per defect species.  The pristine curve
    (always zero) is drawn as a dashed reference line.

    Parameters
    ----------
    formation_energies : dict[str, float | numpy.ndarray]
        Output of :func:`ComputeDefectFormationEnergy`.  Must contain
        the special key ``"mu_values"``.  All other keys are treated as
        defect labels; their values must be broadcastable to the shape
        of ``"mu_values"``.
    mu_label : str, optional
        Label for the x-axis.  Default ``"μ (eV)"``.
    ef_label : str, optional
        Label for the y-axis.  Default ``"Formation energy (eV)"``.
    title : str, optional
        Plot title.  Default ``"Defect formation energy diagram"``.
    exclude_keys : list[str] or None, optional
        Additional dictionary keys to suppress from the plot (on top of
        the always-excluded ``"mu_values"``).  Useful for hiding the
        pristine reference line, e.g. ``exclude_keys=["pristine"]``.
        Default ``None``.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure object (can be saved or displayed by the
        caller).

    Raises
    ------
    KeyError
        If ``"mu_values"`` is absent from *formation_energies*.
    ValueError
        If *formation_energies* contains no plottable defect entries
        after excluding reserved keys.

    Examples
    --------
    >>> fig = PlotDefectFormationEnergy(
    ...     formation_energies=result,
    ...     mu_label="μ_Mg (eV)",
    ...     title="Al-Mg defect phase diagram",
    ... )
    >>> fig.savefig("phase_diagram.pdf")
    """
    import matplotlib.pyplot as plt

    if "mu_values" not in formation_energies:
        raise KeyError(
            "'mu_values' key not found in formation_energies. "
            "Make sure to use the output of ComputeDefectFormationEnergy."
        )

    # Keys that are never plotted as individual defect lines
    reserved = {"mu_values"}
    if exclude_keys:
        reserved.update(exclude_keys)

    mu_values = formation_energies["mu_values"]
    # Ensure mu_values is array-like for uniform handling
    mu_arr = np.atleast_1d(np.asarray(mu_values, dtype=float))

    defect_keys = [k for k in formation_energies if k not in reserved]
    if not defect_keys:
        raise ValueError(
            "No plottable defect entries found in formation_energies "
            f"after excluding reserved keys {reserved}."
        )

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    for key in defect_keys:
        ef = formation_energies[key]
        ef_arr = np.atleast_1d(np.asarray(ef, dtype=float))

        line_kwargs: dict = dict(label=key, linewidth=2)

        # Pristine is always zero – draw as a thin dashed reference
        if key == "pristine":
            line_kwargs.update(
                linestyle="--",
                linewidth=1,
                color="black",
                alpha=0.5,
            )

        ax.plot(mu_arr, ef_arr, **line_kwargs)

    # Zero-line for visual reference
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")

    ax.set_xlabel(mu_label, fontsize=13)
    ax.set_ylabel(ef_label, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig
