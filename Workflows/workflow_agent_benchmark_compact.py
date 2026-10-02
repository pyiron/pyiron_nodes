"""
Agentic workflow-generation benchmark — the compact canvas
==========================================================
The same experiment as ``workflow_agent_benchmark``, drawn as three nodes
instead of fourteen.  ``BenchmarkSuite`` is a ``@group_node``: the twelve nodes
of the full pipeline — settings, task suite, library mirror, work directory,
thread pool, the two arm gates, the two agents, the two sweeps and the stack —
are all still there, inside it.  Expand the node on the canvas to see them,
rewire one, or re-run a single arm.

So this is not a simplification of the benchmark, only of the picture of it.
Every option is a boundary port on the group, and the comparison is still made
by one ``BenchSettings`` feeding both arms, which is what stops them drifting
apart.

What to do with it
------------------
1. Pick ``tier`` and ``limit`` on the group.  Start with ``limit=1`` — a single
   generation turn costs real money, and a full tier costs it once per task
   per arm.
2. Set ``path`` to a dated directory, or a rerun overwrites the last one.
3. ``arms`` switches a whole branch off: ``"aiflow"`` or ``"scratch"`` alone
   runs half the graph and spends half the money.
4. ``verbose=True`` narrates every agent turn, tagged by arm and task.  Without
   it a task prints nothing for minutes, and a run that never reached the agent
   looks merely fast.
5. Run.  ``BenchmarkReport`` prints the statistics, ``PlotBenchmark`` draws the
   ladder pass-rates and the arm comparison.

``repair_budget`` comes back out of the group so the report's
success-vs-repair-budget curve is drawn to the scale that was actually used,
rather than to a number typed in twice.

When to use the other two entry points
--------------------------------------
``RunBenchmarkSuite`` (one function node, in the sibling module) adds
repetitions and Wilson confidence intervals; one ``BenchmarkSuite`` group is
one repetition.  The hand-wired graph at the bottom of the sibling module is
the same pipeline with every node already on the canvas, which is what you
want when the plumbing itself is the thing you are editing.

.. warning::
   This executes LLM-generated code.  It runs in a child process with a hard
   timeout, which contains hangs and crashes — it is **not** a security sandbox.
"""

from core import Workflow

from pyiron_nodes.Workflows.workflow_agent_benchmark import (
    BenchmarkReport,
    BenchmarkSuite,
    PlotBenchmark,
)

# ── Workflow ────────────────────────────────────────────────────────────────

wf = Workflow("workflow_agent_benchmark_compact")
wf.storage_enabled = True

# `benchmark`, not `suite`: expanding a group inlines its twelve inner nodes
# into this graph, and a label shared with one of them would collapse the two
# into a single node whose edges form a cycle.
wf.benchmark = BenchmarkSuite(
    tier="generic",
    limit=1,
    arms="both",
    model="sonnet",
    max_repairs=3,
    path="bench_runs",
    max_workers=3,
    verbose=True,
)

wf.report = BenchmarkReport(
    df=wf.benchmark.outputs.df, max_repairs=wf.benchmark.outputs.repair_budget
)

wf.figure = PlotBenchmark(
    df=wf.benchmark.outputs.df, max_repairs=wf.benchmark.outputs.repair_budget
)


if __name__ == "__main__":
    wf.run()
