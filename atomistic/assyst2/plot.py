from typing import Iterable

from ase import Atoms

from core import as_function_node


@as_function_node("plot")
def PlotConcentration(structures: list[Atoms]):
    """
    Plot histograms of the concentrations in each structure.

    Args:
        structures (list of Atoms): structures to take concentrations of
    """
    from collections import Counter

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    symbols = [Counter(s.symbols) for s in structures]
    elements = sorted(set.union(*(set(s) for s in symbols)))

    df = pd.DataFrame([{e: c[e] / sum(c.values()) for e in elements} for c in symbols])

    sns.histplot(
        data=df.melt(var_name="element", value_name="concentration"),
        x="concentration",
        hue="element",
        multiple="dodge",
    )

    return plt.show()
