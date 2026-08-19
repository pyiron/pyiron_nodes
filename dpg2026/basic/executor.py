from core import as_function_node


@as_function_node("Executor")
def FluxClusterExecutor(cache_directory: str = "./cache"):
    from executorlib import FluxClusterExecutor as Executor

    return Executor(cache_directory=cache_directory)
