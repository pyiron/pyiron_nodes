import numpy as np
from copy import copy
import pandas as pd
from typing import Any, Union
from concurrent.futures import as_completed
from core import as_function_node, Node


def _iterate_node(
    node,
    input_label: str,
    values,
    copy_results=True,
    collect_input=False,
    debug=False,
    executor=None,
):
    """Iterate over a sequence of ``values`` and execute the given ``node`` for each value.

    The function supports both sequential execution (when ``executor`` is ``None``) and
    parallel execution (when an ``executor`` implementing the ``concurrent.futures``
    interface is provided). It can optionally copy the results to avoid side‑effects and
    collect the input values that were processed.

    Parameters
    ----------
    node : Node
        The node instance whose ``run`` method will be called for each input.
    input_label : str
        The name of the attribute on ``node.inputs`` that receives each ``value``.
    values : iterable
        An iterable of values to be fed into the node.
    copy_results : bool, optional
        If ``True`` (default), a shallow copy of each node output is stored.
    collect_input : bool, optional
        If ``True``, the function also returns the list of input values.
    debug : bool, optional
        When ``True``, prints debugging information for each iteration.
    executor : concurrent.futures.Executor, optional
        Executor for parallel execution. If ``None``, runs sequentially.

    Returns
    -------
    list or tuple
        ``out_lst`` – list of node outputs; if ``collect_input`` is ``True`` also
        returns ``inp_lst`` – the list of input values.
    """
    out_lst = []
    inp_lst = [] if collect_input else None

    if executor is None:
        # Sequential execution
        for value in values:
            node.inputs.__setattr__(input_label, value)
            # TODO: provide more elaborate options
            try:
                out = node.run()
            except Exception as e:
                print("execution error: ", e)
                continue
            if copy_results:
                out = copy(out)
            out_lst.append(out)
            if collect_input:
                inp_lst.append(value)
            if debug:
                print(f"iterating over {input_label} = {value}, out={out}")
                print("out list: ", [id(o) for o in out_lst])
    else:
        # Parallel execution
        futures = {
            executor.submit(node, **{input_label: value}): (idx, value)
            for idx, value in enumerate(values)
        }
        # Placeholder, to restore original order after as_completed
        results = [None] * len(values)
        for future in as_completed(futures):
            idx, val = futures[future]
            out = future.result()
            if copy_results:
                out = copy(out)
            results[idx] = out
            if debug:
                print(f"Parallel iter: {input_label}={val}, out={out}")
        out_lst = results
        if collect_input:
            inp_lst = list(values)

    return (out_lst, inp_lst) if collect_input else out_lst


# --- Node iteration to DataFrame ---
@as_function_node
def IterToDataFrame(
    node: Node,
    input_label: str,
    values: Union[list, np.ndarray],
    debug: bool = False,
    executor=None,
    store: bool = False,
) -> pd.DataFrame:
    """
    Node Name: ParameterLoop
    Purpose: Iterate over a generic list of parameter values, invoking the provided ``node`` for each entry and collecting results.

    Required Input Ports:
    - parameters (list[Any]) – the list of values to iterate over (type depends on the node's expected input)
    - calculation_parameters (dict, optional) – additional parameters shared across all runs, if applicable

    Output Ports:
    - loop_results (list[dict]) – each entry contains the iterated parameter value and the node's output(s)

    Important Parameters or Settings:
    - parallel_execution (bool, default=False) – whether to submit jobs concurrently (if the backend supports it)
    - max_concurrent_jobs (int, default=4) – limit for parallel submissions

    Parameters
    ----------
    node : Node
        The node instance to be executed for each input.
    input_label : str
        The name of the input attribute on ``node`` that receives each value.
    values : Union[list, np.ndarray]
        Collection of input values to iterate over.
    debug : bool, default False
        Enable verbose output for debugging each iteration.
    executor : concurrent.futures.Executor, optional
        Executor for parallel execution; if ``None`` the function runs sequentially.
    store : bool, default False
        Internal flag used by the ``as_function_node`` decorator for caching.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the collected inputs (or ``input_{input_label}`` if name clash) and the node's outputs. If the node returns a dataclass, each field occupies its own column.
    """
    from dataclasses import fields, is_dataclass

    # ------------------------------------------------------------------
    # 1️⃣ Run the node over all values
    # ------------------------------------------------------------------
    out_lst, inp_lst = _iterate_node(
        node,
        input_label,
        values,
        copy_results=True,
        collect_input=True,
        debug=debug,
        executor=executor,
    )

    # ------------------------------------------------------------------
    # 2️⃣ Prepare a dict that will be fed to pd.DataFrame
    # ------------------------------------------------------------------
    data_dict: dict[str, List[Any]] = {}

    # ------------------------------------------------------------------
    # 2.1 Input column – avoid name clash with node outputs
    # ------------------------------------------------------------------
    output_labels = list(node.outputs.keys())
    if input_label in output_labels:
        data_dict[f"input_{input_label}"] = inp_lst
    else:
        data_dict[input_label] = inp_lst

    # ------------------------------------------------------------------
    # 3️⃣ Analyse the first output to decide how to unpack the rest
    # ------------------------------------------------------------------
    first_out = out_lst[0] if out_lst else None

    # Helper: is the result a dataclass instance?
    def _is_dataclass_instance(obj: Any) -> bool:
        return is_dataclass(obj) and not isinstance(obj, type)

    # ------------------------------------------------------------------
    # 3.1 Tuple / list / np.ndarray output (multiple scalar outputs)
    # ------------------------------------------------------------------
    multi_output = isinstance(first_out, (tuple, list, np.ndarray)) and len(
        first_out
    ) == len(output_labels)

    # ------------------------------------------------------------------
    # 3.2 Dataclass output – each field becomes a column
    # ------------------------------------------------------------------
    if _is_dataclass_instance(first_out):
        # Extract field names once – they will be the column names
        dc_fields = [f.name for f in fields(first_out)]

        # Initialise a list for each field
        for f_name in dc_fields:
            data_dict[f_name] = []

        # Fill the column lists
        for out in out_lst:
            # Defensive: if a particular iteration returned something else,
            # fall back to NaN for all fields.
            if _is_dataclass_instance(out):
                for f_name in dc_fields:
                    data_dict[f_name].append(getattr(out, f_name))
            else:
                for f_name in dc_fields:
                    data_dict[f_name].append(np.nan)

    # ------------------------------------------------------------------
    # 3.3 Regular scalar / single‑value output
    # ------------------------------------------------------------------
    elif multi_output:
        # Node returns a sequence that matches the declared output labels
        for idx, label in enumerate(output_labels):
            data_dict[label] = [out[idx] for out in out_lst]

    else:
        # Node returns a single scalar (or a single object) per iteration
        if len(output_labels) == 1:
            # Simple case – one declared output
            data_dict[output_labels[0]] = out_lst
        else:
            # Ambiguous case – more declared outputs than we can unpack.
            # We store the whole object under each label (the original
            # behaviour) – this mirrors the previous implementation.
            for label in output_labels:
                data_dict[label] = out_lst

    # ------------------------------------------------------------------
    # 4️⃣ Build the DataFrame (fallback to raw dict on error)
    # ------------------------------------------------------------------
    try:
        df = pd.DataFrame(data_dict)
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        df = pd.DataFrame.from_dict(data_dict, orient="columns")

    return df
