import pandas as pd
from core import as_function_node


@as_function_node("plot")
def PlotEVcurve(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(df["volume"], df["energy"])
    ax.set_ylabel("Count")
    ax.set_xlabel("Energy per atom (meV/atom)")
    return fig
