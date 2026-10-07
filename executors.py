from dataclasses import field
from typing import Optional, Callable, Literal
from core import as_function_node, as_inp_dataclass_node


@as_function_node("Executor")
def SingleNodeExecutor(max_workers: int = 1):
    from executorlib import SingleNodeExecutor as Executor

    return Executor(max_workers=int(max_workers))


@as_function_node("Executor")
def FluxClusterExecutor(cache_directory: str = "./cache", non_modal: bool = False):
    from executorlib import FluxClusterExecutor as Executor

    cache_directory = _resolve_cache_directory(cache_directory)
    executor = Executor(cache_directory=cache_directory, wait=not non_modal)
    _make_stoppable(executor, cache_directory, None, "flux")
    if non_modal:
        _make_non_modal(executor)
    return executor


def _make_non_modal(executor):
    """Tell aiflow not to wait for work submitted to *executor*.

    The run submits the job, reports it and ends; the result is collected by a
    later run or from the Database panel, both of which find it by the node
    hash.  Registered from here rather than passed through ``core`` for the
    same reason as ``_make_stoppable``: it is a property of the executor the
    user configured on the canvas.
    """
    from core import mark_non_modal

    mark_non_modal(executor)


def _resolve_cache_directory(cache_directory):
    """Anchor a relative *cache_directory* under the configured data storage.

    The default ``./cache`` is relative to the process working directory, which
    is wherever the notebook happened to be launched from.  A queued job writes
    its result into that directory and the client is the only thing that knows
    where it was — so a kernel restarted from a different directory cannot find
    the result it is still waiting for, and resubmits.  Anchoring the path makes
    the cache the fixed, findable thing it has to be for recovery to work.

    Shared with the recovery path, which has to anchor the same way when it
    reads the directory back out of the database (see
    :func:`core.remote_cache.resolve_cache_directory`).
    """
    from core.remote_cache import resolve_cache_directory

    return resolve_cache_directory(cache_directory)


def _make_stoppable(executor, cache_directory, config_directory, backend):
    """Teach aiflow's Stop button how to cancel jobs on a cluster executor.

    A queued or running job is not a child process, so there is nothing to
    signal — it has to be cancelled through the scheduler.  ``executorlib``
    already knows how: ``terminate_tasks_in_cache`` reads the queue id each task
    recorded in the cache directory and deletes the job through pysqa
    (``scancel`` for SLURM).  Registering it here rather than in ``core`` keeps
    every scheduler import on this side of the boundary.

    Note that this cancels **every** job in *cache_directory*, not only the node
    being stopped: executorlib exposes no future-to-queue-id mapping.  In
    practice the cache directory belongs to one executor in one workflow, which
    is the run being stopped anyway.

    Failing to register is not fatal — Stop then reports that the node cannot be
    interrupted, which is what happened before this existed.
    """
    try:
        from core import attach_stopper
        from executorlib.task_scheduler.file.spawner_pysqa import (
            terminate_tasks_in_cache,
        )
    except ImportError:
        return

    # executorlib renamed this parameter from ``config_directory`` to
    # ``pysqa_config_directory`` (it is the latter in 1.9.x, the former in the
    # 1.8.1 that environment.yml still pins).  Passing the wrong one raises
    # TypeError at the moment Stop is pressed — the one moment the user cannot
    # afford it — so bind the name the installed version actually accepts.
    import inspect

    try:
        accepted = inspect.signature(terminate_tasks_in_cache).parameters
    except (TypeError, ValueError):       # pragma: no cover - C or wrapped impl
        accepted = {}
    config_kwarg = (
        "pysqa_config_directory" if "pysqa_config_directory" in accepted
        else "config_directory"
    )

    attach_stopper(
        executor,
        lambda: terminate_tasks_in_cache(
            cache_directory=cache_directory,
            backend=backend,
            **{config_kwarg: config_directory},
        ),
    )


@as_function_node("Executor")
def ThreadPoolExecutor(max_workers: int = 1):
    from concurrent.futures import ThreadPoolExecutor as Executor

    return Executor(max_workers=max_workers)


@as_function_node("Executor")
def ProcessPoolExecutor(max_workers: int = 1):
    from concurrent.futures import ProcessPoolExecutor as Executor

    return Executor(max_workers=max_workers)


@as_function_node("Executor")
def ForkExecutor(poll_interval: float = 0.1):
    """Make one node stoppable, by running it in a process that can be killed.

    Usually there is no need for this node: tick **Run isolated (stoppable)** in
    a node's ``Ports ▾`` menu and it gets the same treatment with nothing to
    wire.  Use this when you want to set ``poll_interval`` explicitly.

    Wire this into the ``executor`` port of a node that does long, uninterruptible
    work in the interpreter — ``RunASEMD``, a ``calphy`` free-energy leg, a long
    relaxation.  Pressing Stop cancels those only *between* nodes otherwise,
    because Python cannot interrupt a call in progress; with this executor the
    node itself is killed and reported as Cancelled.

    Unlike ``ProcessPoolExecutor`` this forks, so unpicklable inputs — a LAMMPS
    potential, a ``calphy`` job — do not have to be sent to the worker, and
    nothing is reloaded per call.  The node's *return value* still has to be
    picklable, which arrays and dataclasses are.  POSIX only.

    Not usable with a TensorFlow-backed calculator such as ``GRACE``: a forked
    child deadlocks the first time it touches TensorFlow, so this executor
    refuses to run rather than hang.  Those nodes are stoppable in-process
    instead — ``Minimize`` checks for a Stop between optimiser steps.

    Parameters
    ----------
    poll_interval : float, optional
        How often, in seconds, to check whether the run has been stopped.
    """
    from core import ForkExecutor as Executor

    return Executor(poll_interval=poll_interval)


# SLURM version - same interface, just swap the executor
#: Submission script for the embedded (non-pysqa-configured) Slurm path.
#:
#: Note the absence of ``--get-user-env``: Slurm only accepts it without an
#: argument on some builds, and ``--get-user-env=L`` makes ``sbatch`` exit 255
#: with "option '--get-user-env' doesn't allow an argument".  That rejection is
#: invisible from the workflow — executorlib raises it on a background thread,
#: so the future never completes and the run simply hangs with an empty queue.
#: It is not needed either: ``sbatch`` defaults to ``--export=ALL``, so the job
#: already inherits the submitting process's environment, which is exactly the
#: one holding this checkout's ``PYTHONPATH`` and the agent CLI on ``PATH``.
#: Re-running the login scripts on top of that could only lose those.
DEFAULT_SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --output=time.out
#SBATCH --job-name={{job_name}}
#SBATCH --chdir={{working_directory}}
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
    # Resource settings — only the ones not exposed on SlurmExecutor itself.
    # threads_per_core and pysqa_config_directory are now on SlurmExecutor so
    # typical jobs don't need this node at all.
    gpus_per_core: int = 0
    cwd: Optional[str] = None
    openmpi_oversubscribe: bool = False
    slurm_cmd_args: list = field(default_factory=list)
    submission_template: str = DEFAULT_SLURM_TEMPLATE
    # Executor settings
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
    threads_per_core: int = 1,
    pysqa_config_directory: Optional[str] = None,
    cache_directory: str = "./cache",
    job_name: str = "",
    non_modal: bool = False,
    advanced: SlurmAdvancedSettings = None,
):
    """Run nodes as SLURM jobs.

    ``threads_per_core`` is the effective parallelism of each task: executorlib
    multiplies it by the slot count before writing the batch script (so
    ``threads_per_core=4`` yields ``--cpus-per-task=4``).

    ``pysqa_config_directory`` points to a site's pysqa queue configuration.
    When set, the embedded submission template is automatically cleared so that
    pysqa's own templates are used; to keep the built-in template alongside pysqa,
    pass a custom ``SlurmAdvancedSettings(submission_template=...)`` via
    ``advanced``.

    With ``non_modal`` the workflow does not wait: each node wired to this
    executor is submitted, reported, and left in the queue, and the run ends.
    That is what lets a ``for`` loop submit a parametric study in one go
    instead of one job at a time.  Collect the results by running the workflow
    again once the jobs are done, or from the Database panel — both find them
    by the node's hash, which is also the cache key of the job.
    """
    from executorlib import SlurmClusterExecutor

    if advanced is None:
        advanced = SlurmAdvancedSettings().run()

    # Anchor the cache so a restarted kernel can still find jobs submitted from
    # a different working directory — see _resolve_cache_directory.
    cache_directory = _resolve_cache_directory(cache_directory)

    resource_dict = {
        # "cores": cores,
        "threads_per_core": threads_per_core,
        "gpus_per_core": advanced.gpus_per_core,
        "submission_template": advanced.submission_template,
        "partition": partition,
        "run_time_max": run_time_max,
    }
    # An explicit job_name overrides executorlib's default (the cache-directory
    # basename, which produces "cache" for every job and makes squeue unreadable).
    if job_name:
        resource_dict["job_name"] = job_name
    # When using pysqa, the site's own template is authoritative — the embedded
    # Python-side template should step aside unless the user has already set a
    # custom one via advanced.submission_template.
    if pysqa_config_directory and resource_dict["submission_template"] == DEFAULT_SLURM_TEMPLATE:
        resource_dict["submission_template"] = ""
    if advanced.cwd is not None:
        resource_dict["cwd"] = advanced.cwd
    if memory_max is not None:
        resource_dict["memory_max"] = memory_max
    if advanced.openmpi_oversubscribe:
        resource_dict["openmpi_oversubscribe"] = advanced.openmpi_oversubscribe
    if advanced.slurm_cmd_args:
        resource_dict["slurm_cmd_args"] = advanced.slurm_cmd_args

    # advanced.pysqa_config_directory is kept as a fallback so that any
    # existing workflow that passed a directory via advanced continues to work.
    _pysqa_dir = pysqa_config_directory or getattr(advanced, "pysqa_config_directory", None)

    executor = SlurmClusterExecutor(
        cache_directory=cache_directory,
        resource_dict=resource_dict,
        pysqa_config_directory=_pysqa_dir,
        hostname_localhost=advanced.hostname_localhost,
        block_allocation=advanced.block_allocation,
        init_function=advanced.init_function,
        disable_dependencies=advanced.disable_dependencies,
        refresh_rate=advanced.refresh_rate,
        plot_dependency_graph=advanced.plot_dependency_graph,
        plot_dependency_graph_filename=advanced.plot_dependency_graph_filename,
        # executorlib's own flag: do not wait for outstanding tasks when this
        # executor is shut down.  Walking away from a queued job is the whole
        # point of non_modal, and the job itself does not care — it writes its
        # result into cache_directory either way.
        wait=not non_modal,
    )
    # Pressing Stop cancels the jobs in `cache_directory` — see _make_stoppable
    # for why that is the whole directory rather than just this node's job.
    _make_stoppable(executor, cache_directory, _pysqa_dir, "slurm")
    if non_modal:
        _make_non_modal(executor)
    return executor


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
