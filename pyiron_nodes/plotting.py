"""
For graphical representations of data.
"""

from __future__ import annotations

from typing import Optional, Literal

import numpy as np
import pandas as pd

from core import as_function_node, as_inp_dataclass_node

@as_inp_dataclass_node
class InputPlotOptions:
    title: Optional[str] = ""
    color: Literal[
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "k",
        "w",
        "blue",
        "green",
        "red",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "white",
    ] = "b"
    symbol: Literal[
        ".",
        ",",
        "o",
        "v",
        "^",
        "<",
        ">",
        "1",
        "2",
        "3",
        "4",
        "s",
        "p",
        "*",
        "h",
        "H",
        "+",
        "x",
        "X",
        "D",
        "d",
        "|",
        "_",
    ] = "o"
    legend_label: Optional[str] = ""
    log_x: bool = False
    log_y: bool = False 


@as_function_node("fig")
def PlotDataFrame(
    df: pd.DataFrame,
    x: Optional[list | np.ndarray] = None,
    options: InputPlotOptions = InputPlotOptions().run(),
):
    """
    Plot a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to plot.
    x : list or np.ndarray, optional
        Columns to use for the x‑axis.
    options : InputPlotOptions, optional
        Plot configuration (title, colour, symbol, legend, log‑scales).
    """
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    df.plot(x=x, ax=ax)
    # Apply color and symbol from options
    for line in ax.get_lines():
        line.set_color(options.color)
        line.set_marker(options.symbol)
    # Apply options
    if options.title:
        ax.set_title(options.title)
    if options.legend_label:
        for line in ax.get_lines():
            line.set_label(options.legend_label)
        ax.legend()
    if options.log_x:
        ax.set_xscale("log")
    if options.log_y:
        ax.set_yscale("log")
    return fig


@as_function_node("fig")
def MergePlots(fig1, fig2):
    """
    Merge two Matplotlib Figure objects by overlaying their plot data onto a single figure.
    Supports line plots and scatter plots from the first Axes of each input figure.
    Returns a new Figure containing the combined data.
    """
    from matplotlib import pyplot as plt
    import matplotlib.collections as mcoll

    # Create a new figure and axis.
    fig, ax = plt.subplots()

    def _copy_axes(source_ax):
        # Copy line plots
        for line in source_ax.get_lines():
            x = line.get_xdata()
            y = line.get_ydata()
            label = line.get_label()
            ax.plot(x, y, label=label, color=line.get_color(), marker=line.get_marker())
        # Copy scatter plots (PathCollection)
        for col in source_ax.collections:
            if isinstance(col, mcoll.PathCollection):
                offsets = col.get_offsets()
                if offsets.size == 0:
                    continue
                x = offsets[:, 0]
                y = offsets[:, 1]
                label = col.get_label()
                ax.scatter(x, y, label=label, color=col.get_facecolor()[0])

    merged_handles = []
    merged_labels = []

    if fig1 is not None:
        _copy_axes(fig1.axes[0])
        h, l = fig1.axes[0].get_legend_handles_labels()
        merged_handles.extend(h)
        merged_labels.extend(l)

    if fig2 is not None:
        _copy_axes(fig2.axes[0])
        h, l = fig2.axes[0].get_legend_handles_labels()
        merged_handles.extend(h)
        merged_labels.extend(l)

    if merged_handles:
        ax.legend(handles=merged_handles, labels=merged_labels)
    else:
        ax.legend()
    return fig


@as_function_node("fig")
def PlotDataFrameXY(
    df: pd.DataFrame,
    x: Optional[list | np.ndarray] = None,
    options: InputPlotOptions = InputPlotOptions().run(),
):
    """
    Plot a DataFrame with X‑ and Y‑axes.

    Parameters
    ----------
    df : pd.DataFrame
    x : optional column(s) for the x‑axis
    options : InputPlotOptions, optional
    """
    from matplotlib import pyplot as plt

    # Default labels in case not deduced
    x_label = "x label not defined"
    y_label = "y label not defined"

    # Check if dataframe has only two columns and x parameter is not provided.
    if df.shape[1] == 2 and x is None:
        columns = df.columns
        x = columns[0]  # First column for x-axis.
        y = columns[1]  # Second column for y-axis.
        x_label, y_label = x, y
    else:
        y = None
        if isinstance(x, str):
            x_label = x

    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    df.plot(x=x, y=y, ax=ax)
    # Apply color and symbol from options
    for line in ax.get_lines():
        line.set_color(options.color)
        line.set_marker(options.symbol)
    # Apply options
    if options.title:
        ax.set_title(options.title)
    if options.legend_label:
        for line in ax.get_lines():
            line.set_label(options.legend_label)
        ax.legend()
    if options.log_x:
        ax.set_xscale("log")
    if options.log_y:
        ax.set_yscale("log")
    return fig


@as_function_node("fig")
def Scatter(
    x: Optional[list | np.ndarray] = None,
    y: Optional[list | np.ndarray] = None,
    options: InputPlotOptions = InputPlotOptions().run(),
):
    """
    Simple scatter plot.
    """
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(x, y, color=options.color, marker=options.symbol, label=options.legend_label if options.legend_label else None)
    # Apply options
    if options.title:
        ax.set_title(options.title)
    if options.legend_label:
        # Legend will be created automatically from the scatter label
        pass
    if options.log_x:
        ax.set_xscale("log")
    if options.log_y:
        ax.set_yscale("log")
    return fig


@as_function_node("fig")
def LinearFittingCurve(
    x: Optional[list | np.ndarray] = None,
    y: Optional[list | np.ndarray] = None,
    options: InputPlotOptions = InputPlotOptions().run(),
):
    """
    Plot data together with a linear fit and an ideal line.
    """
    from matplotlib import pyplot as plt
    import numpy as np

    rms = np.sqrt(np.var(x - y))
    correlation_coefficient = np.corrcoef(x, y)[0, 1]
    print(f"Correlation Coefficient: {correlation_coefficient}")
    print(f"RMS: {rms}")

    x_ideal = np.linspace(min(x), max(x), 100)
    y_ideal = np.poly1d(np.polyfit(x, y, 1))(x_ideal)

    fig, ax = plt.subplots()
    ax.plot(x_ideal, x_ideal, "--", label="Ideal")
    ax.plot(x_ideal, y_ideal, label="Fitted")
    ax.scatter(x, y, color=options.color, marker=options.symbol)

    # Apply options
    if options.title:
        ax.set_title(options.title)
    if options.legend_label:
        ax.legend([options.legend_label])
    if options.log_x:
        ax.set_xscale("log")
    if options.log_y:
        ax.set_yscale("log")
    ax.legend()
    return fig


@as_function_node("fig")
def ShowArray(
    mat: Optional[np.ndarray],
    aspect_ratio: float = None,
    options: InputPlotOptions = InputPlotOptions().run(),
):
    """
    Display an array as an image.
    """
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    if aspect_ratio is not None:
        ax.imshow(mat, aspect=aspect_ratio)
    else:
        ax.imshow(mat)

    # Apply options
    if options.title:
        ax.set_title(options.title)
    if options.legend_label:
        ax.legend([options.legend_label])
    if options.log_x:
        ax.set_xscale("log")
    if options.log_y:
        ax.set_yscale("log")
    return fig


@as_function_node("fig")
def Histogram(
    x: Optional[list | np.ndarray],
    bins: int = 50,
    options: InputPlotOptions = InputPlotOptions().run(),
):
    """
    Plot a histogram.
    """
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.hist(x, bins=bins, color=options.color)

    # Apply options
    if options.title:
        ax.set_title(options.title)
    if options.legend_label:
        ax.legend([options.legend_label])
    if options.log_x:
        ax.set_xscale("log")
    if options.log_y:
        ax.set_yscale("log")
    return fig


@as_function_node("figure")
def Plot(
    y: Optional[list | np.ndarray | pd.core.series.Series],
    x: Optional[list | np.ndarray | pd.core.series.Series] = None,
    axis: Optional[object] = None,
    options: InputPlotOptions = InputPlotOptions().run(),
):
    """
    General plot function that respects InputPlotOptions.
    """
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    print("plotting: ", len(np.shape(y)))
    if len(np.shape(y)) == 1:
        # If x is not provided, generate a default sequence
        x = np.arange(len(y)) if x is None else x
    else:
        x = None

    if axis is None:
        fig, ax = plt.subplots()
    else:
        # assume axis is an Axes object passed in
        ax = axis
        fig = ax.figure

    # Plot data
    if x is None:
        ax.plot(y, color=options.color, marker=options.symbol, label=options.legend_label)
    else:
        ax.plot(x, y, color=options.color, marker=options.symbol, label=options.legend_label)

    # Log scales if needed
    if options.log_x:
        ax.set_xscale("log")
    if options.log_y:
        ax.set_yscale("log")

    # Set title if provided
    if options.title:
        ax.set_title(options.title)

    # Add legend if label provided
    if options.legend_label:
        ax.legend()

    return fig


@as_function_node("figure")
def MultiPlot(
    y: Optional[list | np.ndarray | pd.core.series.Series],
    x: Optional[list | np.ndarray | pd.core.series.Series] = None,
    axis: Optional[object] = None,
    options: InputPlotOptions = InputPlotOptions().run(),
):
    """
    Plot multiple series respecting InputPlotOptions.
    """
    import numpy as np
    from matplotlib import pyplot as plt

    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis
        fig = ax.figure

    # Handle multi-series plotting
    if isinstance(y, (list, tuple)) and not isinstance(y, np.ndarray):
        for i, yy in enumerate(y):
            xx = (
                np.arange(len(yy))
                if x is None
                else (x[i] if isinstance(x, (list, tuple)) else x)
            )
            ax.plot(
                xx,
                yy,
                color=options.color,
                marker=options.symbol,
                label=options.legend_label,
            )
    else:
        xx = np.arange(len(y)) if x is None else x
        ax.plot(xx, y, color=options.color, marker=options.symbol, label=options.legend_label)

    # Log scales if needed
    if options.log_x:
        ax.set_xscale("log")
    if options.log_y:
        ax.set_yscale("log")

    # Set title if provided
    if options.title:
        ax.set_title(options.title)

    # Add legend if label provided
    if options.legend_label:
        ax.legend()

    return fig


@as_function_node("axes")
def Subplot(
    nrows: Optional[int] = 1,
    ncols: Optional[int] = 1,
    sharex: Optional[bool] = False,
    sharey: Optional[bool] = False,
):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey)
    # Return both figure and axes so caller can decide which to use
    return axes


@as_function_node("axis")
def Title(axis: Optional[object] = None, title: Optional[str] = ""):
    from matplotlib import pyplot as plt

    if axis is None:
        # if None, create a new figure+axes for standalone title
        fig, ax = plt.subplots()
    else:
        ax = axis
    ax.set_title(title)
    return ax


@as_function_node
def AnalyseConvergencePlot(
    y: "np.ndarray",
    x: Optional["np.ndarray"] = None,
    i_start: int = 0,
    smoothen: bool = False,
    window_size: int = 10,
    mode: Literal[
        "reflect",
        "mirror",
        "constant",
        "nearest",
        "wrap",
        "grid-constant",
    ] = "nearest",
    ref_to_min: bool = False,
):
    """
    Plot the convergence of a scalar series ``y`` (e.g. energy, force norm)
    as a function of an optional abscissa ``x`` (iteration number, time,
    etc.).  The function is intended to be used as a pyiron node, therefore
    it returns a single Matplotlib ``Figure`` object.

    Parameters
    ----------
    y : np.ndarray
        1‑D array containing the quantity to be analysed.
    x : np.ndarray | None, optional
        Optional abscissa.  If ``None`` a simple integer index
        ``np.arange(len(y))`` is used.
    i_start : int, default ``0``
        Index from which the plot should start – useful to discard the
        initial equilibration part of a trajectory.
    smoothen : bool, default ``False``
        If ``True`` a simple moving‑average (window = 5) is applied to ``y``
        before plotting.  This helps visualise noisy data.
    ref_to_min : bool, default ``False``
        If ``True`` the plotted values are shifted by the minimum value in the
        displayed range (i.e. ``y - min(y)``).  This is handy when one wants to
        see how quickly the series approaches its lowest value.

    Returns
    -------
    Figure
        Matplotlib figure containing the convergence plot.
    """
    # ------------------------------------------------------------------
    # Imports – kept inside the node so that the node can be imported
    # without pulling heavy optional dependencies.
    # ------------------------------------------------------------------
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    # ------------------------------------------------------------------
    # Convert inputs to NumPy arrays and handle the optional x‑axis.
    # ------------------------------------------------------------------
    y = np.asarray(y).ravel()
    if x is None:
        x = np.arange(y.size)
    else:
        x = np.asarray(x).ravel()

    # ------------------------------------------------------------------
    # Slice the data from the requested start index.
    # ------------------------------------------------------------------
    y_plot = y[i_start:]
    x_plot = x[i_start:]

    # ------------------------------------------------------------------
    # Optional smoothing (simple moving average with a fixed window).
    # ------------------------------------------------------------------
    if smoothen:
        y_plot = uniform_filter1d(y_plot, size=window_size, mode=mode)

    # ------------------------------------------------------------------
    # Optional reference to the minimum value.
    # ------------------------------------------------------------------
    if ref_to_min:
        min_val = np.min(y_plot)
        y_plot = y_plot - min_val

    # ------------------------------------------------------------------
    # Build the figure.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(x_plot, y_plot, marker="o", linestyle="-", color="tab:blue")
    ax.set_xlabel("Iteration" if x is None else "x")
    ylabel = "y"
    if ref_to_min:
        ylabel += " – min"
    ax.set_ylabel(ylabel)
    ax.set_title("Convergence")
    ax.grid(True, which="both", ls=":", alpha=0.7)

    # Highlight the final value with a horizontal line.
    ax.axhline(
        y_plot[-1], color="tab:red", ls="--", lw=1, label=f"final = {y_plot[-1]:.3e}"
    )
    ax.legend(fontsize="small", loc="best")

    return fig
