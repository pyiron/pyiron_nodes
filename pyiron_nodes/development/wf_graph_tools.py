"""
# wf_graph_tools Module
This module provides a collection of tools to convert and manipulate workflow graphs. 
A workflow graph is represented by a set of nodes and edges, and this module offers 
various functions to transform these graphs into different representations, such as 
executable workflow code.

Additionally, this module includes helper tools to assist with tasks such as sorting 
the order of nodes in the graph, making it easier to work with and analyze workflow graphs.
The tools provided in this module are designed to be flexible and reusable, allowing users 
to easily integrate them into their own workflow processing pipelines.

# Key Features
Conversion of workflow graphs into various representations, including executable code
Helper tools for sorting and manipulating node order in the graph
Flexible and reusable design for easy integration into existing workflows
Usage

To use this module, simply import it and access the various functions and tools provided. For example:
    import wf_graph_tools as gt

# Create a workflow graph
    graph = gt.WorkflowGraph()
# format
# graph.nodes = [(node_label, import_lib_path), ...]
# graph.edges = [(node_label_i/handle, node_label_j/handle), ...]


# Convert a workflow to execteable code
    from pyiron_workflow import Workflow
    import pyiron_nodes

    wf = Workflow("compute_elastic_constants")

    wf.engine = pyiron_nodes.atomistic.engine.ase.M3GNet()
    wf.input_elastic = pyiron_nodes.atomistic.property.elastic.InputElasticTensor()
    wf.bulk = pyiron_nodes.atomistic.structure.build.Bulk(name=Pb, cubic=True)
    wf.elastic = pyiron_nodes.atomistic.property.elastic.ElasticConstants(
        engine=wf.engine, structure=wf.bulk, parameters=wf.input_elastic
    )

    code = gt.get_code_from_wf(wf)
    print(code)

# Convert a workflow to a graph
    graph = gt.get_graph_from_wf(wf)


# Sort the nodes in the graph
    sorted_graph = gt.topological_sort(graph)
"""

import dataclasses
from collections import defaultdict

from pyironflow.reactflow import get_edges as _get_edges
from pyironflow.reactflow import get_nodes as _get_nodes
from pyironflow.reactflow import get_node_from_path

from pyiron_workflow import Workflow


__author__ = "Joerg Neugebauer"
__copyright__ = (
    "Copyright 2024, Max-Planck-Institut for Sustainable Materials GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.1"
__maintainer__ = ""
__email__ = ""
__status__ = "development"
__date__ = "Nov 20, 2024"


@dataclasses.dataclass
class WorkflowGraph:
    nodes: list = dataclasses.field(default_factory=list)
    edges: list = dataclasses.field(default_factory=list)
    label: str = "my_workflow_name"


def _filter_and_flatten_nested_dict_keys(data, keys_to_keep):
    def filter_and_flatten_dict(d, keys):
        result = {}
        for key in keys:
            if "/" in key:
                top_key, nested_key = key.split("/", 1)
                if top_key in d and isinstance(d[top_key], dict):
                    result[f"{top_key}__{nested_key}"] = d[top_key].get(nested_key)
            else:
                if key in d:
                    result[key] = d[key]
        return result

    return [filter_and_flatten_dict(item, keys_to_keep) for item in data]


def _rename_keys(dict_list, key_mapping):
    """
    Rename keys in a list of dictionaries.

    Args:
    dict_list (list): A list of dictionaries to modify.
    key_mapping (dict): A dictionary mapping old keys to new keys.

    Returns:
    list: A new list of dictionaries with renamed keys.
    """
    result = []
    for d in dict_list:
        new_dict = {}
        for old_key, value in d.items():
            new_key = key_mapping.get(old_key, old_key)
            new_dict[new_key] = value
        result.append(new_dict)
    return result


def _different_indices(list1, list2):
    return [i for i in range(len(list1)) if list1[i] != list2[i]]


def _nodes_from_dict(nodes_dict):
    return [(node["label"], node["import_path"]) for node in nodes_dict]


def _edges_from_dict(edges_dict):
    edges = []
    for edge in edges_dict:
        source = edge["source"]
        source_handle = edge["sourceHandle"]
        if source.startswith("var_"):
            if isinstance(source_handle, str):
                source_handle = f"__str_{source_handle}"
        edges.append(
            (
                f'{edge["source"]}/{source_handle}',
                f'{edge["target"]}/{edge["targetHandle"]}',
            )
        )

    return edges


def get_nodes_from_wf(wf, keys_to_keep=["data/label", "data/import_path"]):
    key_mapping = {"data__label": "label", "data__import_path": "import_path"}
    nodes_dict = _filter_and_flatten_nested_dict_keys(_get_nodes(wf), keys_to_keep)
    nodes_dict = _rename_keys(nodes_dict, key_mapping)
    return nodes_dict


def _convert_to_integer_representation(graph: WorkflowGraph):
    # Create a dictionary mapping node labels to indices
    node_to_index = {node[0]: index for index, node in enumerate(graph.nodes)}

    # Convert edge list to integer representation
    integer_edges = [
        (node_to_index[edge[0].split("/")[0]], node_to_index[edge[1].split("/")[0]])
        for edge in graph.edges
    ]

    return integer_edges


def topological_sort(graph: WorkflowGraph):
    # Kahn's algorithm for topological sorting
    # Create a graph and in-degree count for each node

    sort_graph = defaultdict(list)
    edges = _convert_to_integer_representation(graph)
    nodes = range(len(graph.nodes))

    in_degree = {node: 0 for node in nodes}

    # Build the graph and count in-degrees
    for edge in edges:
        n_i, n_j = edge
        sort_graph[n_i].append(n_j)
        in_degree[n_j] += 1

    # Initialize queue with nodes having 0 in-degree
    queue = [node for node in nodes if in_degree[node] == 0]

    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)

        # Reduce in-degree of adjacent nodes
        for neighbor in sort_graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check if there's a cycle
    if len(result) != len(nodes):
        return None  # Graph has a cycle

    sorted_nodes = [graph.nodes[i] for i in result]
    sorted_graph = WorkflowGraph(
        nodes=sorted_nodes, edges=graph.edges, label=graph.label
    )

    return sorted_graph


def get_graph_from_wf(wf: Workflow) -> WorkflowGraph:
    # get edges between nodes
    keys_to_keep = ["target", "targetHandle", "source", "sourceHandle"]
    edges = _filter_and_flatten_nested_dict_keys(_get_edges(wf), keys_to_keep)

    # add edges for non-default inputs
    nodes = get_nodes_from_wf(
        wf,
        keys_to_keep=[
            "data/label",
            "data/import_path",
            "data/target_values",
            "data/target_labels",
        ],
    )
    nodes_non_default_inp_param = []
    for node in nodes:
        label = node["label"]
        import_path = node["import_path"]
        node_obj = get_node_from_path(import_path)()
        changed_args = _different_indices(
            node_obj.inputs.to_list(), node["data__target_values"]
        )
        for i in changed_args:
            value = node["data__target_values"][i]
            handle = node["data__target_labels"][i]
            if value not in ("NonPrimitive", "NotData"):
                inp_node_label = f"var_{label}__{handle}"
                edge = dict(
                    target=label,
                    targetHandle=handle,
                    source=inp_node_label,
                    sourceHandle=value,
                )

                edges.append(edge)
                inp_node = dict(label=inp_node_label, data__import_path=value)
                nodes_non_default_inp_param.append(inp_node)

        nodes = get_nodes_from_wf(
            wf,
            keys_to_keep=["data/label", "data/import_path"],
        )

    key_mapping = {"data__label": "label", "data__import_path": "import_path"}
    nodes = _rename_keys(nodes_non_default_inp_param + nodes, key_mapping=key_mapping)

    graph = WorkflowGraph(
        nodes=_nodes_from_dict(nodes), edges=_edges_from_dict(edges), label=wf.label
    )
    sorted_graph = topological_sort(graph)

    return sorted_graph


def get_code_from_graph(
    graph: WorkflowGraph,
    workflow_lib="pyiron_workflow",
    pyiron_nodes_lib="pyiron_nodes",
):
    """
    Generate Python source code from workflow graph.

    Args:
        label (str): The label to use in the generated code.
        module_a (str): The name of the module to import from.
        module_b (str): The name of the module to import.

    Returns:
        str: The generated Python source code.
    """
    import black

    code = f"""
from {workflow_lib} import Workflow
import {pyiron_nodes_lib}

wf = Workflow('{graph.label}')

"""

    # Add nodes to Workflow
    for node in graph.nodes:
        label, import_path = node

        if not label.startswith("var_"):
            code += f"""wf.{label} = {import_path}("""

            # Add edges
            first_edge = True
            for edge in graph.edges:
                edge_source, edge_target = edge
                target, target_handle = edge_target.split("/")
                source, source_handle = edge_source.split("/")
                if target == label:
                    if first_edge:
                        first_edge = False
                    else:
                        code += """, """
                    if source.startswith("var_"):
                        if source_handle.startswith("__str_"):
                            code += f"""{target_handle}='{source_handle[6:]}'"""
                        else:
                            code += f"""{target_handle}={source_handle}"""
                    else:
                        code += f"""{target_handle}=wf.{source}"""
            code += f""") \n"""
            # code += '\n' + 'print(wf.run()) \n'

    formatted_code = black.format_str(code, mode=black.FileMode())

    return formatted_code


def get_code_from_wf(wf: Workflow):
    """Generate Python source code from pyiron_workflow"""

    graph = get_graph_from_wf(wf)

    code = get_code_from_graph(graph)

    return code
