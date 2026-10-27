from core import group_node, as_function_node
import numpy as np


@as_function_node
def ExtractStackingFault(
    gsfe_values: list,
    usf_search_fraction: float = 1.0,
    isf_search_lo: float = 0.2,
    isf_search_hi: float = 0.8,
):
    """Extract intrinsic (ISF) and unstable (USF) stacking fault energies from a
    generalized stacking fault energy (GSFE) curve.

    A GSFE curve is computed by sweeping the rigid shift of the upper half of a
    slab along a slip direction and recording the relaxed energy cost per unit area.
    Two special points are read from it:

    **USF (unstable stacking fault energy)**: the maximum of the GSFE curve.
    It is the energy barrier a partial dislocation must overcome to glide, and it
    controls the Peierls stress and ease of plastic deformation.  For FCC metals
    on the (111) plane along [112], the USF typically occurs near shift = 1/6.

    **ISF (intrinsic stacking fault energy)**: the local minimum at an
    intermediate shift where the faulted stacking sequence (e.g. ABCABC →
    ABCAB|ABC for FCC) is metastable.  The ISF controls the separation between
    partial dislocations and therefore the width of stacking fault ribbons.
    For FCC metals on (111)[112], the ISF is near shift = 1/3.

    The search window parameters let you adapt to different crystal symmetries
    and slip directions.  The defaults are tuned for FCC (111)[112]:

    - USF is the global maximum (``usf_search_fraction=1.0``).
    - ISF is the minimum between 20% and 80% of the shift range, which captures
      the local minimum at shift = 1/3 while excluding the endpoints
      (both zero by construction for a periodic shift).

    Parameters
    ----------
    gsfe_values : list
        Ordered sequence of GSFE energies in mJ/m², from shift = 0 to shift = 1.
        Typically obtained with ``IterToDataFrame`` over a ``LayerShift`` scan
        followed by ``GetColumnFromDataFrame``.  Must contain at least 3 points.
    usf_search_fraction : float
        USF is the maximum in ``gsfe_values[:int(usf_search_fraction * n)]``.
        Default ``1.0`` (global max).  Set to ``0.5`` to restrict the search to
        the first half of the curve if the curve is periodic and you want only
        the first barrier.
    isf_search_lo, isf_search_hi : float
        The ISF search window as fractions of the curve length.  ISF is the
        minimum in ``gsfe_values[lo_idx:hi_idx]`` where the indices are
        ``int(isf_search_lo * (n-1))`` and ``int(isf_search_hi * (n-1)) + 1``.
        The default ``[0.2, 0.8]`` window captures the ISF at shift = 1/3
        while excluding endpoints.

    Returns
    -------
    isf_energy_mJ_per_m2 : float
        Intrinsic stacking fault energy in mJ/m².
    usf_energy_mJ_per_m2 : float
        Unstable stacking fault energy in mJ/m².

    Notes
    -----
    For an FCC metal, an expected ISF order of magnitude is 10–300 mJ/m² and
    USF is typically 2–5× higher.  If ``isf_energy_mJ_per_m2`` comes out near
    zero or equal to ``usf_energy_mJ_per_m2``, check the shift range and
    direction: a shift along [110] instead of [112] may not produce a local
    minimum at an intermediate position, and the search window bounds should
    be tightened accordingly.
    """
    arr = np.array(gsfe_values, dtype=float)
    n = len(arr)
    if n < 3:
        raise ValueError(
            f"gsfe_values must have at least 3 points to locate ISF and USF; "
            f"got {n}.  Run the GSFE scan with more shift points "
            f"(Linspace num_points >= 3)."
        )

    # USF: maximum in the first usf_search_fraction of the curve
    usf_end = max(1, int(round(usf_search_fraction * (n - 1))) + 1)
    usf_energy_mJ_per_m2 = float(arr[:usf_end].max())

    # ISF: minimum in the interior window, excluding endpoints (which are 0 by construction)
    isf_lo = max(1, int(round(isf_search_lo * (n - 1))))
    isf_hi = min(n - 1, int(round(isf_search_hi * (n - 1))) + 1)
    if isf_lo >= isf_hi:
        raise ValueError(
            f"ISF search window is empty (lo={isf_lo}, hi={isf_hi} for n={n}). "
            "Widen isf_search_lo / isf_search_hi or increase the number of GSFE points."
        )
    isf_energy_mJ_per_m2 = float(arr[isf_lo:isf_hi].min())

    return isf_energy_mJ_per_m2, usf_energy_mJ_per_m2


@group_node("defect_energythe ")
def GetStackingFaultEnergy(element: str, engine):
    from pyiron_nodes.atomistic.structure.build import Surface
    from pyiron_nodes.atomistic.calculator.ase import StaticEnergy
    from pyiron_nodes.atomistic.structure.calc import DefectEnergyPerArea
    from pyiron_nodes.atomistic.structure.transform import LayerShift
    from pyiron_nodes.atomistic.structure.view import PlotCNA
    from core import Workflow

    wf = Workflow("GetStackingFaultEnergy")
    wf.Surface = Surface(element=element, size="1 2 10", vacuum=20, orthogonal=True)
    wf.CreateDefect = LayerShift(wf.Surface, shift_y=-1 / 3)
    wf.ReferenceEnergy = StaticEnergy(structure=wf.Surface, engine=engine)
    wf.PlotCNA = PlotCNA(structure=wf.CreateDefect)
    wf.DefectEnergy = StaticEnergy(structure=wf.CreateDefect, engine=engine)
    wf.DefectEnergyPerArea = DefectEnergyPerArea(
        energy_defect=wf.DefectEnergy,
        energy_ref=wf.ReferenceEnergy,
        structure=wf.Surface,
    )

    return wf.DefectEnergyPerArea.outputs.delta_erg
