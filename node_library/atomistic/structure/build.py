from pyiron_workflow import as_function_node
from pyiron_workflow import as_macro_node


# from pyiron_workflow.workflow import Workflow


@as_function_node("structure")
def bulk(
        name='Al',
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
        wf, element: str = 'Al', cell_size: int = 3, vacancy_index: int | None = None
):
    from node_library.atomistic.structure.transform import (
        create_vacancy,
        repeat,
    )

    # print ('create bulk')
    wf.bulk = bulk(name=element, cubic=True)
    wf.cell = repeat(structure=wf.bulk, repeat_scalar=cell_size)

    wf.structure = create_vacancy(structure=wf.cell, index=vacancy_index)
    return wf.structure  # .outputs.structure


nodes = [bulk, cubic_bulk_cell]
