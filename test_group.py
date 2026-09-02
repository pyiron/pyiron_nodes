from core import Workflow
from core.groups import group_node
from core import as_function_node

# ── Local node definitions ──────────────────────


@as_function_node
def report(report_label: str, value: float):
    """Print a labelled result."""
    msg = f"  [report] {report_label}: {value:.4f}"
    print(msg)
    return msg


@as_function_node
def load_data(path: str):
    """Simulate loading a list of numbers from a file."""
    data = [1.0, 4.0, 9.0, 16.0, 25.0]
    print(f"  [load_data] loaded {len(data)} values from '{path}'")
    return data


@as_function_node
def compute_sqrt(data: list):
    """Take the square root of every element."""
    import math

    result = [math.sqrt(x) for x in data]
    print(f"  [compute_sqrt] {data} → {result}")
    return result


@as_function_node
def compute_mean(data: list):
    """Compute the arithmetic mean."""
    mean = sum(data) / len(data)
    print(f"  [compute_mean] mean = {mean}")
    return mean


# ── Group node factories ─────────────────────────────


@group_node("mean")
def preprocessing(path):
    inner_wf = Workflow("preprocessing")
    inner_wf.load = load_data(path=path)
    inner_wf.sqrt = compute_sqrt(data=inner_wf.load.outputs.data)
    inner_wf.mean = compute_mean(data=inner_wf.sqrt.outputs.result)
    return inner_wf.mean


wf = Workflow("ml_pipeline")

wf.preprocessing = preprocessing(path="data.csv")

wf.report = report(report_label="result", value=wf.preprocessing.outputs.mean)
