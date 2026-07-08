from core import as_function_node


@as_function_node("Executor")
def SingleNodeExecutor(max_workers: int = 1):
    from executorlib import SingleNodeExecutor as Executor

    return Executor(max_workers=int(max_workers))


@as_function_node("Executor")
def FluxClusterExecutor(cache_directory: str = "./cache"):
    from executorlib import FluxClusterExecutor as Executor

    return Executor(cache_directory=cache_directory)
