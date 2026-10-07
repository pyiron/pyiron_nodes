"""
Agentic workflow-generation benchmark — the compact canvas
==========================================================
Two entry points in one file, sharing the same ``report`` and ``figure`` nodes:

``wf`` (``RunBenchmarkSuite``)
    A function node that handles ``repeats ≥ 1``.  It cannot be expanded, but
    every flag is a port — including ``repeats`` — and statistical error bars
    appear when ``repeats > 1``.  This is the default view the GUI opens.

``wf_group`` (``BenchmarkSuite``)
    The expandable group: the twelve nodes of the full pipeline are inside it.
    Expand the node on the canvas to see them, rewire one, or re-run a single
    arm.  **No** ``repeats`` — one group run is one repetition.

Both expose ``repair_budget`` so the report's repair-budget curve is drawn to
the same scale as the run, without typing the number twice.

What to do with it
------------------
1. Pick ``tier`` and ``limit``.  Start with ``limit=1`` — a single generation
   turn costs real money, and a full tier costs it once per task per arm.
2. Set ``workdir`` (on ``wf``) or ``path`` (on ``wf_group``) to a dated
   directory.  ``wf`` then writes to
   ``<workdir>/<model>_<hash>/rep_NNN/<arm>/<task>``, so two configurations
   never share a directory, and a rerun of the *same* configuration reuses
   what it already finished instead of paying for it twice — set
   ``resume=False`` to force a fresh measurement.
3. ``arms`` switches a whole branch off: ``"aiflow"`` or ``"scratch"`` alone
   runs half the graph and spends half the money.
4. ``repeats`` (only on ``wf``) adds independent repetitions and Wilson
   confidence intervals.  A single repetition cannot separate a real difference
   from noise; three or more give a meaningful interval.  Raising it later is
   cheap: ``repeats=1`` followed by ``repeats=5`` only pays for 2-5.
5. ``verbose=True`` narrates every agent turn, tagged by arm and task.  Without
   it a task prints nothing for minutes, and a run that never reached the agent
   looks merely fast.
6. Run.  ``BenchmarkReport`` prints the statistics; ``PlotBenchmark`` returns
   five figures, one per output-port tab: ``figure`` (the headline),
   ``failures`` (where tasks died and who is to blame), ``cost``,
   ``reliability`` (per-task pass rate with confidence intervals) and ``reuse``.
7. For a written summary, drop a ``SummarizeBenchmark`` node on the canvas and
   wire the report, the frame and the figure ports into it.  It is deliberately
   not wired here: it calls the agent, so it would spend money on every run.
8. Wire a ``SlurmExecutor`` node to ``task_executor`` to submit every
   ``(task, arm, repetition)`` as its own batch job — a long tier then
   finishes in the wall time of its slowest single task.  For thread
   execution (the default) leave ``task_executor`` unwired.

When to use the third entry point
----------------------------------
The hand-wired graph at the bottom of ``workflow_agent_benchmark.py`` is the
same pipeline with every node already on the canvas, which is what you want
when the plumbing itself is the thing you are editing.

.. warning::
   This executes LLM-generated code.  It runs in a child process with a hard
   timeout, which contains hangs and crashes — it is **not** a security sandbox.
"""

from core import Workflow

from pyiron_nodes.benchmark import (
    BenchmarkReport,
    BenchmarkSuite,
    PlotBenchmark,
    RunBenchmarkSuite,
)

# ── Entry point 1: function node with repeats (primary — default GUI view) ────

wf = Workflow("workflow_agent_benchmark_repeating")
wf.storage_enabled = True

wf.benchmark = RunBenchmarkSuite(
    tier="generic",
    limit=1,
    arms="both",
    model="sonnet",
    max_repairs=3,
    repeats=1,  # increase for error bars; each extra rep reruns all tasks
    workdir="bench_runs",
    max_workers=3,
    verbose=True,
)

wf.report = BenchmarkReport(
    df=wf.benchmark.outputs.df,
    max_repairs=wf.benchmark.outputs.repair_budget,
)

wf.figure = PlotBenchmark(
    df=wf.benchmark.outputs.df,
    max_repairs=wf.benchmark.outputs.repair_budget,
)


# ── Entry point 2: expandable group (no repeats) ─────────────────────────────

wf_group = Workflow("workflow_agent_benchmark_compact")
wf_group.storage_enabled = True

# `benchmark`, not `suite`: expanding a group inlines its twelve inner nodes
# into this graph, and a label shared with one of them would collapse the two
# into a single node whose edges form a cycle.
wf_group.benchmark = BenchmarkSuite(
    tier="generic",
    limit=1,
    arms="both",
    model="sonnet",
    max_repairs=3,
    path="bench_runs",
    max_workers=3,
    verbose=True,
)

wf_group.report = BenchmarkReport(
    df=wf_group.benchmark.outputs.df,
    max_repairs=wf_group.benchmark.outputs.repair_budget,
)

wf_group.figure = PlotBenchmark(
    df=wf_group.benchmark.outputs.df,
    max_repairs=wf_group.benchmark.outputs.repair_budget,
)


if __name__ == "__main__":
    wf.run()
