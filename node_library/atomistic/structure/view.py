from pyiron_workflow import as_function_node


@as_function_node("plot")
def plot3d(structure):
    return structure.plot3d()

nodes = [plot3d]
