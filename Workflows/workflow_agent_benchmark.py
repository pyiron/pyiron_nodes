"""
Agentic workflow-generation benchmark — aiflow vs. from scratch
===============================================================
Takes a list of plain-English tasks and has **Claude Code** solve each one
twice: once as an aiflow workflow, and once as a plain-Python program with the
framework taken away.  Both arms then climb the same five-tier validation
ladder, get the same repair budget and the same model, so the difference between
the two columns is the effect of aiflow itself.

==== ============ ======================================= =============================
tier name         aiflow arm                              scratch arm
==== ============ ======================================= =============================
1    syntax       ``ast.parse(source)``                   identical
2    constructs   module-level ``wf`` is a ``Workflow``   ``main`` callable, no framework
3    executes     ``wf.run()`` completes                  ``main()`` returns a ``dict``
4    plausible    named node output within range          named dict key within range
5    roundtrip    ``Workflow.load(wf.save())`` agrees     fresh-process re-run agrees
==== ============ ======================================= =============================

The generating agent runs headless (``claude -p``) with Read/Grep/Glob/Write but
**no Bash**, so it cannot run what it writes.  Every repair is therefore driven
by this workflow and is explicitly counted — the ``repair_cycles`` column means
what it says.

The loop lives inside the higher-order ``WorkflowAgent`` node, so the outer
graph stays a DAG and the whole task suite can be swept with
``IterToDataFrame`` — optionally in parallel through a thread pool.  One sweep
per arm, stacked row-wise by ``StackResults``.

Every node is defined in ``pyiron_nodes.benchmark``; this module is the canvas
wiring of them, and re-exports the names so older imports keep working.

Three ways to run it
--------------------
**One node.**  ``RunBenchmarkSuite`` does the whole thing — task suite, library
mirror, freeze manifest, both arms, repetitions, ``results.csv``, aggregation —
with every option of the ``workflow_bench --run`` CLI exposed as a port:

.. code-block:: python

    from pyiron_nodes.benchmark import RunBenchmarkSuite

    bench = RunBenchmarkSuite(
        tier="atomistic_hard",
        arms="both",            # or just "aiflow" / "scratch"
        model="opus",
        max_repairs=3,
        repeats=5,              # error bars; one repetition has none
        workdir="bench_runs/hard_opus_2026-10-02",
        effort="high",
        run_variant=True,
        max_workers=3,
    )
    bench.run()
    df = bench.outputs.df.value

**The canvas.**  The assembled ``wf`` below shows the same pipeline node by
node, which is the point when you want to re-run only the report, swap the task
source or watch the sweep progress:

1. Pick a ``tier`` on ``TaskSuite`` (dropdown), or wire ``TaskList`` instead and
   type your own tasks.
2. Set ``model``, ``arms``, ``max_repairs``, the timeouts, ``effort`` and the
   optional turns on the single ``BenchSettings`` node — both agents and the
   report are wired from it, so the two arms cannot drift apart.  ``arms``
   selects which arm runs at all.
3. Leave ``allow_reference_workflows=False`` on ``NodeLibraryMirror`` unless you
   deliberately want to measure the retrieval-assisted case.
4. Set ``max_workers`` on the thread pool for parallel generation.
5. Run.  ``BenchmarkReport`` prints the statistics; ``PlotBenchmark`` returns
   five figures, one per output-port tab: ``figure`` (ladder pass-rates, the
   success-vs-repair-budget curve with its confidence interval, and the arm
   comparison), ``failures``, ``cost``, ``reliability`` and ``reuse``.  For a
   written summary, add a ``SummarizeBenchmark`` node — it is not wired here
   because it calls the agent and would spend money on every run.
   The report's third output, ``tasks``, is the per-task table: view that port
   and every row offers two links — the workflow the agent wrote (opens as a
   canvas tab) and ``reasoning.md``, the readable transcript of how it got
   there.  ``results.csv`` keeps the same paths for later.

**One group.**  ``BenchmarkSuite`` is that same canvas folded into a single
``@group_node``: every option above is a boundary port, and expanding the node
puts all twelve inner nodes back on the canvas, so nothing is given up for the
tidier picture.  ``Workflows/workflow_agent_benchmark_compact`` wires it to the
report and the plot — three nodes for the whole experiment.  Unlike
``RunBenchmarkSuite`` it has no ``repeats``: one group is one repetition.

All three paths drive the same ``WorkflowAgent``, so they produce the same
columns; repetitions and their per-cell confidence intervals are only available
from ``RunBenchmarkSuite`` (or afterwards from ``AggregateRepeats``).

Generated code is written to ``<workdir>/<arm>/<task-slug>/{workflow,solution}.py``
and kept for inspection.  When a follow-up variant runs it edits that same file,
so the graded version is first snapshotted alongside it as ``*_primary.py``.
That directory is the audit trail for one row: a snapshot and the error text per
attempt, the raw ``chat_*.jsonl`` per agent turn (``save_chat=True``), and the
``reasoning.md`` rendered from them.  Its path is in the ``task_dir`` column.

.. warning::
   This executes LLM-generated code.  It runs in a child process with a hard
   timeout, which contains hangs and crashes — it is **not** a security sandbox.
"""

from core import Workflow
from pyiron_nodes.controls import IterToDataFrame
from pyiron_nodes.executors import ThreadPoolExecutorNode

# The nodes themselves live in ``pyiron_nodes.benchmark``; this module is one
# wiring of them.  The names not used by the graph below are re-exported for
# the callers and notebooks that still import them from here.
from pyiron_nodes.benchmark import (
    AggregateRepeats,
    ArmTasks,
    BenchmarkReport,
    BenchmarkSuite,
    BenchSettings,
    BenchWorkDir,
    GenerateWorkflow,
    NodeLibraryMirror,
    OptimizeWorkflow,
    PlotBenchmark,
    RepairWorkflow,
    RunBenchmarkSuite,
    StackResults,
    SummarizeBenchmark,
    TaskList,
    TaskSuite,
    ValidateWorkflow,
    VaryWorkflow,
    WorkflowAgent,
    warn_if_the_harness_never_ran,
)

__all__ = [
    "wf",
    "AggregateRepeats",
    "ArmTasks",
    "BenchmarkReport",
    "BenchmarkSuite",
    "BenchSettings",
    "BenchWorkDir",
    "GenerateWorkflow",
    "NodeLibraryMirror",
    "OptimizeWorkflow",
    "PlotBenchmark",
    "RepairWorkflow",
    "RunBenchmarkSuite",
    "StackResults",
    "SummarizeBenchmark",
    "TaskList",
    "TaskSuite",
    "ValidateWorkflow",
    "VaryWorkflow",
    "WorkflowAgent",
    "warn_if_the_harness_never_ran",
]

# ── Workflow ────────────────────────────────────────────────────────────────

wf = Workflow("workflow_agent_benchmark")
wf.storage_enabled = True

wf.mirror = NodeLibraryMirror()

wf.pool = ThreadPoolExecutorNode(max_workers=3)

wf.settings = BenchSettings()

wf.task_suite = TaskSuite(tier="generic")

wf.workdir = BenchWorkDir()

wf.tasks_aiflow = ArmTasks(tasks=wf.task_suite, arms=wf.settings.outputs.arms)

wf.tasks_scratch = ArmTasks(
    tasks=wf.task_suite, arm="scratch", arms=wf.settings.outputs.arms
)

wf.agent_aiflow = WorkflowAgent(
    model=wf.settings.outputs.model,
    max_repairs=wf.settings.outputs.max_repairs,
    workdir=wf.workdir,
    exec_timeout_s=wf.settings.outputs.exec_timeout_s,
    max_budget_usd=wf.settings.outputs.max_budget_usd,
    agent_timeout_s=wf.settings.outputs.agent_timeout_s,
    lib_dir=wf.mirror,
    run_variant=wf.settings.outputs.run_variant,
    run_optimize=wf.settings.outputs.run_optimize,
    effort=wf.settings.outputs.effort,
    save_chat=wf.settings.outputs.save_chat,
    verbose=wf.settings.outputs.verbose,
)

wf.agent_scratch = WorkflowAgent(
    model=wf.settings.outputs.model,
    max_repairs=wf.settings.outputs.max_repairs,
    workdir=wf.workdir,
    exec_timeout_s=wf.settings.outputs.exec_timeout_s,
    max_budget_usd=wf.settings.outputs.max_budget_usd,
    agent_timeout_s=wf.settings.outputs.agent_timeout_s,
    arm="scratch",
    lib_dir=wf.mirror,
    run_variant=wf.settings.outputs.run_variant,
    run_optimize=wf.settings.outputs.run_optimize,
    effort=wf.settings.outputs.effort,
    save_chat=wf.settings.outputs.save_chat,
    verbose=wf.settings.outputs.verbose,
)

wf.bench_aiflow = IterToDataFrame(
    node=wf.agent_aiflow,
    input_label="task",
    values=wf.tasks_aiflow,
    debug=False,
    executor=wf.pool,
    store=True,
)

wf.bench_scratch = IterToDataFrame(
    node=wf.agent_scratch,
    input_label="task",
    values=wf.tasks_scratch,
    debug=False,
    executor=wf.pool,
    store=True,
)

wf.results = StackResults(
    df_aiflow=wf.bench_aiflow, df_scratch=wf.bench_scratch, workdir=wf.workdir
)

wf.report = BenchmarkReport(df=wf.results, max_repairs=wf.settings.outputs.max_repairs)

wf.figure = PlotBenchmark(df=wf.results, max_repairs=wf.settings.outputs.max_repairs)


if __name__ == "__main__":
    wf.run()
