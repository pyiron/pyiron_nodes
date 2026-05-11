from core import as_function_node, as_out_dataclass_node
from ase.build import bulk
from typing import Optional, Literal
from ase.atoms import Atoms


@as_function_node("structure")
def Bulk(
    name: str,
    crystalstructure: Optional[
        Literal["fcc", "bcc", "hcp", "diamond", "rocksalt"]
    ] = None,
    a: Optional[float] = None,
    c: Optional[float] = None,
    u: Optional[float] = None,
    orthorhombic: bool = False,
    cubic: bool = False,
) -> Atoms:
    """
    Create a bulk crystal structure.

    **Scientific purpose**
    Generate a pristine bulk unit cell for a given element and crystal lattice
    (e.g. fcc Al, bcc Fe, hcp Mg). This is the typical starting point for defect
    studies, molecular‑dynamics simulations, or high‑throughput materials screening.

    **Required inputs**
    - ``name``: Chemical symbol of the element (e.g. ``"Al"``).
    - ``crystalstructure``: One of ``"fcc"``, ``"bcc"``, ``"hcp"``, ``"diamond"``,
      ``"rocksalt"``; if omitted the default of the underlying factory is used.
    - ``a``, ``c``, ``c_over_a``, ``u``: Optional lattice parameters or internal
      coordinates.
    - ``orthorhombic`` / ``cubic``: Force the cell shape to be orthorhombic or cubic.

    **Typical use‑cases**
    * Building a bulk material before inserting vacancies, interstitials, or
      surfaces.
    * Preparing input structures for relaxation or MD runs.
    * Generating a library of bulk cells for high‑throughput workflows.

    Returns
    -------
    Node returns an Atoms object from ``pyiron_atomistics._StructureFactory().bulk`` compatible
    with ASE/pyiron.
    """
    return bulk(
        name=name,
        crystalstructure=crystalstructure,
        a=a,
        c=c,
        u=u,
        orthorhombic=orthorhombic,
        cubic=cubic,
    )
