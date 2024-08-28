from __future__ import annotations

from pyiron_workflow import as_function_node, as_macro_node


@as_function_node("structure")
def bulk(
    name,
    crystalstructure=None,
    a=None,
    c=None,
    covera=None,
    u=None,
    orthorhombic=False,
    cubic=False,
):
    from pyiron_atomistics import _StructureFactory

    return _StructureFactory().bulk(
        name,
        crystalstructure,
        a,
        c,
        covera,
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
