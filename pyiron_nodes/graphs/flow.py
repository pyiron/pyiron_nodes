import pathlib
from typing import Optional

from core import as_function_node
from core.graph import Graph, Node


@as_function_node
def LoadGraph(path: str):
    """
    Load a workflow graph from *path*.

    If *path* is not an absolute path, it is interpreted as relative to the default
    workflow storage location.

    Parameters
    ----------
    path : str
        Path to the saved graph (JSON, YAML, …).

    Returns
    -------
    pyiron_core.pyiron_workflow.api.graph.Graph
        The deserialized workflow graph.
    """
    from core.graph import Graph
    from core.config import paths

    # 1️⃣  Is the supplied path already absolute?
    if not pathlib.Path(path).is_absolute():
        path = paths.WORKFLOW_STORAGE / path

    # 4️⃣  Load and return the graph
    graph = Graph().load(path)
    return graph


@as_function_node
def SaveGraph(graph: Graph, path: str):
    from core.config import paths
    from os.path import isfile

    # 1️⃣  Is the supplied path already absolute?
    if not pathlib.Path(path).is_absolute():
        path = paths.WORKFLOW_STORAGE / path

    # 4️⃣  Save and return the graph
    graph.save(path)
    status = isfile(path)
    return status


@as_function_node
def Display(graph: Graph):
    from pyironflow.gui_utilities import GuiGraph

    plot = GuiGraph(graph)
    return plot


@as_function_node
def DisplayNodes(graph: Graph):
    nodes = graph.nodes
    return nodes


@as_function_node
def DisplayEdges(graph: Graph):
    edges = graph.edges
    return edges


# @as_function_node
# def DisplayNodeData(graph: Graph):
#     from pyironflow.gui_utilities import display_gui_data

#     data = display_gui_data(graph)
#     return data


# @as_function_node
# def DisplayNodeStyle(graph: Graph):
#     from pyiron_core.pyiron_workflow.api.gui import display_gui_style

#     style = display_gui_style(graph)
#     return style


@as_function_node
def NodesToGui(graph: Graph, remove_none: Optional[bool] = False):
    from pyironflow.gui_utilities import _nodes_to_gui

    nodes = _nodes_to_gui(graph, remove_none=False)
    return nodes


@as_function_node
def EdgesToGui(graph: Graph):
    from pyironflow.gui_utilities import _edges_to_gui

    edges = _edges_to_gui(graph, remove_none=False)
    return edges


@as_function_node(["GraphNode", "Node"])
def ExtractNode(node_label: str, graph: Graph):
    node = graph.nodes[node_label]
    return node, node.node


@as_function_node
def NodeInput(node: Node):
    inputs = node.node.inputs
    return inputs


@as_function_node
def DisplayGraphAsJson(graph: Graph):
    import json

    from IPython.display import JSON

    graph_json = JSON(json.dumps(graph.__getstate__(), indent=2), exanded=True)

    return graph_json


# @as_function_node
# def OptimizeGraphConnections(graph: Graph):
#     raise NotImplementedError(
#         "pyiron_core.pyiron_workflow.graph.base._optimize_graph_connections did not exist at time of refactoring"
#     )


# @as_function_node
# def MarkNodeAsExpanded(graph: Graph, node_label: str, expanded: Optional[bool] = True):
#     from pyiron_core.pyiron_workflow.api.gui import (
#         _mark_node_as_collapsed,
#         _mark_node_as_expanded,
#     )

#     if expanded:
#         graph = _mark_node_as_expanded(graph, node_label)
#     else:
#         graph = _mark_node_as_collapsed(graph, node_label)

#     return graph


# @as_function_node
# def GetGraphFromMacro(macro_node):
#     raise NotImplementedError(
#         "pyiron_core.pyiron_workflow.graph.base._get_graph_from_macro did not exist at time of refactoring"
#     )


# @as_function_node
# def GetActiveNodes(graph: Graph):
#     from pyiron_core.pyiron_workflow.api.gui import _get_active_nodes

#     nodes = _get_active_nodes(graph)
#     return nodes


@as_function_node
def ExpandNode(graph: Graph, node_label: str):
    from core.graph_utils import expand_node

    expanded_graph = graph.copy()
    expand_node(expanded_graph.copy(), node_label)
    return expanded_graph


@as_function_node
def AnalyzeParents(graph: Graph):
    from pyironflow.gui_utilities import _nodes_to_gui
    import pandas as pd

    # print(f"graph.nodes keys: {list(graph.nodes.keys())}")
    # for label, node in graph.nodes.items():
    #     print(f"  node '{label}': parent={getattr(node, 'parent', None)!r}, "
    #         f"expanded={getattr(node, 'expanded', 'N/A')}")

    # # Now check _nodes_to_gui output
    rows = _nodes_to_gui(graph)
    df = pd.DataFrame(rows)[["id", "parentId", "extent"]]
    return df


@as_function_node
def GetCodeFromGraph(
    graph: Graph,
):
    from core.graph_to_workflow import graph_to_workflow_code

    code = graph_to_workflow_code(graph, workflow_name=graph.label)

    return code


@as_function_node
def GetFunctionFromNode(node: Node):
    import inspect

    code = f"# {node._module_path}\n{node._source}"
    return code


@as_function_node
def DisplayCode(code):
    from pygments import highlight
    from pygments.formatters import TerminalFormatter
    from pygments.lexers import Python2Lexer

    print(highlight(code, Python2Lexer(), TerminalFormatter()))

    end = None
    return end


@as_function_node
def ConvertMacroToWorkflow(macro_node):
    kwargs = {}
    for inp in macro_node.inputs.data["label"]:
        inp_port_label = f"inp_port_{inp}"
        kwargs[inp] = inp_port_label

    out = macro_node._func(**kwargs)

    workflow = out._workflow
    return workflow


# @as_function_node
# def GetUpdatedGraph(full_graph, level: Optional[int] = 0):
#     from pyiron_core.pyiron_workflow.api.graph import get_updated_graph

#     graph = get_updated_graph(full_graph, level=level)
#     return graph


@as_function_node
def TopologicalSort(graph: Graph):
    from core.graph import topological_sort

    graph = topological_sort(graph)
    return graph


@as_function_node
def RemoveNode(graph: Graph, node_label: str):
    from core.graph import remove_node

    graph = remove_node(graph, node_label)
    return graph
