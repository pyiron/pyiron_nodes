from ase.atoms import Atoms
from core import as_function_node


@as_function_node("plot") #, use_cache=False)
def PlotSPG(structures: list[Atoms]):
    """Plot a histogram of space groups in input list."""
    import matplotlib.pyplot as plt
    from structuretoolkit.analyse import get_symmetry

    spacegroups = []
    for structure in structures:
        spacegroups.append(get_symmetry(structure).info["number"])
    plt.hist(spacegroups)
    return plt.show()
