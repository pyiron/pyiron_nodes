"""
Agentic workflow-generation benchmark — the repeating canvas
=============================================================
The entry point for production runs.  ``RunBenchmarkSuite`` handles
``repeats ≥ 1`` and every configuration flag is a port.

Executor
--------
The ``slurm`` node is a ``SlurmExecutor`` wired to ``benchmark.task_executor``.
For cluster runs the benchmark submits every ``(task, arm, repetition)`` as its
own batch job, so a twenty-task × two-arm × five-repetition run finishes in the
wall time of its slowest single task.

To run in-process with threads instead, disconnect the ``task_executor`` wire or
delete the ``slurm`` node.

What to do with it
------------------
1. Pick ``tier`` and ``limit``.  Start with ``limit=1`` — a single generation
   turn costs real money.
2. Set ``workdir`` to a dated directory.  The run goes into
   ``<workdir>/<model>_<hash>/rep_NNN/<arm>/<task>``.
3. ``arms`` switches a whole branch off: ``"aiflow"`` or ``"scratch"`` alone
   runs half the graph and spends half the money.
4. ``repeats`` adds independent repetitions and Wilson confidence intervals.
5. Adjust the ``slurm`` node for your partition and resource limits.  Set
   ``job_name`` to something meaningful — it appears in ``squeue``.
6. Run.  ``BenchmarkReport`` prints the statistics; ``PlotBenchmark`` returns
   five figures on separate output-port tabs.
7. For a written summary, drop a ``SummarizeBenchmark`` node on the canvas and
   wire the report, the frame and the figure ports into it.  It is deliberately
   not wired here: it calls the agent, so it would spend money on every run.

.. warning::
   This executes LLM-generated code.  It runs in a child process with a hard
   timeout, which contains hangs and crashes — it is **not** a security sandbox.
"""

from pyiron_nodes.benchmark import BenchmarkReport, PlotBenchmark, _BenchmarkSuiteCore
from pyiron_nodes.executors import SlurmExecutor
from core import Workflow

wf = Workflow("workflow_agent_benchmark_repeating")
wf.storage_enabled = True

wf.slurm = SlurmExecutor(
    partition="s.cmfe",
    run_time_max=86400,
    memory_max=16,
    pysqa_config_directory="/cmmc/u/system/SLES12/soft/pyiron/dev/pyiron-resources-cmmc/queues",
    job_name="bench",
)

wf.suite = _BenchmarkSuiteCore(
    limit=1, exec_timeout_s=1000, max_budget_usd=10.0, task_executor=wf.slurm
)

wf.report = BenchmarkReport(
    df=wf.suite.outputs.df, max_repairs=wf.suite.outputs.repair_budget
)

wf.figure = PlotBenchmark(
    df=wf.suite.outputs.df, max_repairs=wf.suite.outputs.repair_budget
)
