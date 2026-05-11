from dataclasses import field
from typing import Optional, Callable, Literal
from core import as_function_node, as_inp_dataclass_node


@as_function_node("Executor")
def SingleNodeExecutor(max_workers: int = 1):
    from executorlib import SingleNodeExecutor as Executor

    return Executor(max_workers=int(max_workers))


@as_function_node("Executor")
def ThreadPoolExecutor(max_workers: int = 1):
    from concurrent.futures import ThreadPoolExecutor as Executor

    return Executor(max_workers=max_workers)


@as_function_node("Executor")
def ProcessPoolExecutor(max_workers: int = 1):
    from concurrent.futures import ProcessPoolExecutor as Executor

    return Executor(max_workers=max_workers)


# SLURM version - same interface, just swap the executor
DEFAULT_SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --output=time.out
#SBATCH --job-name={{job_name}}
#SBATCH --chdir={{working_directory}}
#SBATCH --get-user-env=L
#SBATCH --partition={{partition}}
{%- if run_time_max %}
#SBATCH --time={{ [1, run_time_max // 60]|max }}
{%- endif %}
{%- if dependency %}
#SBATCH --dependency=afterok:{{ dependency | join(',') }}
{%- endif %}
{%- if memory_max %}
#SBATCH --mem={{memory_max}}G
{%- endif %}
#SBATCH --cpus-per-task={{cores}}

{{command}}
"""


@as_inp_dataclass_node
class SlurmAdvancedSettings:
    # Resource settings
    threads_per_core: int = 1
    gpus_per_core: int = 0
    cwd: Optional[str] = None
    openmpi_oversubscribe: bool = False
    slurm_cmd_args: list = field(default_factory=list)
    submission_template: str = DEFAULT_SLURM_TEMPLATE
    # Executor settings
    pysqa_config_directory: Optional[str] = None
    hostname_localhost: Optional[bool] = None
    block_allocation: bool = False
    init_function: Optional[Callable] = None
    disable_dependencies: bool = False
    refresh_rate: float = 0.01
    plot_dependency_graph: bool = False
    plot_dependency_graph_filename: Optional[str] = None


@as_function_node("Executor")
def SlurmExecutor(
    partition: str = Literal["normal"],
    run_time_max: int = 180,  # in seconds
    memory_max: int = None,  # in GB
    cache_directory: str = "./cache",
    advanced: SlurmAdvancedSettings = None,
):
    from executorlib import SlurmClusterExecutor

    if advanced is None:
        advanced = SlurmAdvancedSettings().run()

    resource_dict = {
        # "cores": cores,
        "threads_per_core": advanced.threads_per_core,
        "gpus_per_core": advanced.gpus_per_core,
        "submission_template": advanced.submission_template,
        "partition": partition,
        "run_time_max": run_time_max,
    }
    if advanced.cwd is not None:
        resource_dict["cwd"] = advanced.cwd
    if memory_max is not None:
        resource_dict["memory_max"] = memory_max
    if advanced.openmpi_oversubscribe:
        resource_dict["openmpi_oversubscribe"] = advanced.openmpi_oversubscribe
    if advanced.slurm_cmd_args:
        resource_dict["slurm_cmd_args"] = advanced.slurm_cmd_args

    return SlurmClusterExecutor(
        cache_directory=cache_directory,
        resource_dict=resource_dict,
        pysqa_config_directory=advanced.pysqa_config_directory,
        hostname_localhost=advanced.hostname_localhost,
        block_allocation=advanced.block_allocation,
        init_function=advanced.init_function,
        disable_dependencies=advanced.disable_dependencies,
        refresh_rate=advanced.refresh_rate,
        plot_dependency_graph=advanced.plot_dependency_graph,
        plot_dependency_graph_filename=advanced.plot_dependency_graph_filename,
    )


@as_function_node("Executor")
def ThreadPoolExecutorNode(max_workers: int = 4):
    """
    Create a ThreadPoolExecutor as a workflow node.

    This node can be used to configure parallel execution at the workflow level.
    When assigned to `wf.executor`, it will be automatically detected and used
    by `wf.run()` for parallel execution.

    Parameters
    ----------
    max_workers : int
        Maximum number of workers in the pool (default: 4)

    Returns
    -------
    concurrent.futures.ThreadPoolExecutor
        Configured executor instance

    Examples
    --------
    >>> wf = Workflow("my_workflow")
    >>> wf.executor = ThreadPoolExecutorNode(max_workers=8)
    >>> results = wf.run(verbose=True)  # Uses 8 workers automatically
    """
    from concurrent.futures import ThreadPoolExecutor as Executor

    return Executor(max_workers=max_workers)


@as_function_node
def fcc_metals():
    """
    List of metals with FCC structure
    """
    list_of_metals = ["Cu", "Ag", "Au", "Pt", "Ni", "Pd", "Rh", "Ir"]
    return list_of_metals


@as_function_node("Executor")
def SubgraphExecutorNode(
    executor_type: str = "database",
    working_directory: str = None,
    data_directory: str = None,
    **kwargs
):
    """
    Create a SubgraphExecutor for remote execution.

    This executor enables running subgraphs on remote machines or HPC queues
    using executorlib backends. Compatible with database queues, file systems,
    and HPC schedulers.

    Parameters
    ----------
    executor_type : str
        Type of executorlib backend:
        - "database": Database-backed queue (default, good for multi-machine)
        - "singlenode": Local execution
        - "filesystem": File-based queue (good for NFS/shared storage)
        - "workqueue": Work Queue system
        - "hsqs": HPC Simple Queue System (for SLURM/PBS)
    working_directory : str, optional
        Working directory for temporary files
    data_directory : str, optional
        Directory for shared storage (results across workers)
    **kwargs
        Additional executorlib configuration (e.g., database credentials)

    Returns
    -------
    SubgraphExecutor
        Configured executor instance

    Examples
    --------
    >>> # Database queue (multi-machine sharing)
    >>> wf.executor = SubgraphExecutorNode(executor_type="database")
    >>> wf.calc = ExpensiveCalculation(remote_execute=True)
    >>> results = wf.run()

    >>> # File system queue (NFS/shared storage)
    >>> wf.executor = SubgraphExecutorNode(
    ...     executor_type="filesystem",
    ...     data_directory="/shared/storage"
    ... )

    >>> # HPC queue (via SLURM)
    >>> wf.executor = SubgraphExecutorNode(executor_type="hsqs")
    >>>
    >>> # On execution machines, start workers:
    >>> # pyiron-worker --executor-type hsqs --data-dir /shared/storage
    """
    from core.subgraph_executor import SubgraphExecutor

    # Import path configuration
    try:
        from core.config import paths

        if working_directory is None:
            working_directory = str(paths.WORKING_DIRECTORY)
        if data_directory is None:
            data_directory = str(paths.DATA_STORAGE)
    except (ImportError, AttributeError):
        # Fallback to defaults
        pass

    executor = SubgraphExecutor(
        executor_type=executor_type,
        working_directory=working_directory,
        data_directory=data_directory,
        **kwargs
    )

    return executor


@as_function_node("Executor")
def ProcessPoolExecutorNode(max_workers: int = 4):
    """
    Create a ProcessPoolExecutor for remote workflow execution.

    This executor serializes the entire workflow graph to JSON and executes
    it on remote processes, bypassing the pickling limitations of standard
    ProcessPoolExecutor. This provides true process-based parallelism with
    full CPU utilization.

    Use this for CPU-intensive workflows where you want to leverage multiple
    cores with process isolation. The entire graph is executed on a remote
    worker process, avoiding pickling issues.

    Parameters
    ----------
    max_workers : int
        Number of worker processes (default: 4)

    Returns
    -------
    RemoteGraphExecutor
        Configured executor instance

    Examples
    --------
    >>> wf = Workflow("my_workflow")
    >>>
    >>> # Set executor
    >>> wf.executor = ProcessPoolExecutorNode(max_workers=4)
    >>>
    >>> # Add workflow nodes
    >>> wf.calc = ExpensiveCalculation(...)
    >>> wf.process = AnotherCalculation(...)
    >>>
    >>> # Execute - entire graph runs on remote process
    >>> results = wf.run(verbose=True)

    Note
    ----
    - Entire graph serialized as JSON (no pickling of nodes/closures)
    - True process isolation for CPU-bound workloads
    - Compatible with standard executor interface
    - For distributed multi-machine execution, use SubgraphExecutorNode instead
    """
    from core.remote_graph_executor import RemoteGraphExecutor

    return RemoteGraphExecutor(max_workers=max_workers)
