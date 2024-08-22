from pyiron_workflow import as_function_node
from pyiron_workflow import as_macro_node

from typing import Optional


# from pyiron_workflow.workflow import Workflow


@as_function_node("structure")
def bulk(
    name: str,
    crystalstructure: Optional[str] = None,
    a: Optional[float | int] = None,
    c: Optional[float | int] = None,
    c_over_a: Optional[float] | int = None,
    u: Optional[float | int] = None,
    orthorhombic: bool = False,
    cubic: bool = False,
):
    from pyiron_atomistics import _StructureFactory

    return _StructureFactory().bulk(
        name,
        crystalstructure,
        a,
        c,
        c_over_a,
        u,
        orthorhombic,
        cubic,
    )


@as_macro_node("structure")
def cubic_bulk_cell(
    wf, element: str, cell_size: int = 1, vacancy_index: int | None = None
):
    from pyiron_nodes.atomistic.structure.transform import (
        create_vacancy,
        repeat,
    )

    # print ('create bulk')
    wf.bulk = bulk(name=element, cubic=True)
    wf.cell = repeat(structure=wf.bulk, repeat_scalar=cell_size)

    wf.structure = create_vacancy(structure=wf.cell, index=vacancy_index)
    return wf.structure  # .outputs.structure


nodes = [bulk, cubic_bulk_cell]
