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

Two ways to run it
------------------
**One node.**  ``RunBenchmarkSuite`` does the whole thing — task suite, library
mirror, freeze manifest, both arms, repetitions, ``results.csv``, aggregation —
with every option of the ``workflow_bench --run`` CLI exposed as a port:

.. code-block:: python

    from pyiron_nodes.Workflows.workflow_agent_benchmark import RunBenchmarkSuite

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
5. Run.  ``BenchmarkReport`` prints the statistics; ``PlotBenchmark`` draws the
   ladder pass-rates, the success-vs-repair-budget curve and the arm comparison.

Both paths drive the same ``WorkflowAgent``, so they produce the same columns;
repetitions and their per-cell confidence intervals are only available from
``RunBenchmarkSuite`` (or afterwards from ``AggregateRepeats``).

Generated code is written to ``<workdir>/<arm>/<task-slug>/{workflow,solution}.py``
and kept for inspection.  When a follow-up variant runs it edits that same file,
so the graded version is first snapshotted alongside it as ``*_primary.py``.

.. warning::
   This executes LLM-generated code.  It runs in a child process with a hard
   timeout, which contains hangs and crashes — it is **not** a security sandbox.
"""

from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import Workflow, as_function_node
from pyiron_nodes.controls import IterToDataFrame
from pyiron_nodes.executors import ThreadPoolExecutorNode

from pyiron_ai.workflow_bench import (
    AGENT_BUDGET_USD,
    AGENT_TIMEOUT_S,
    ARM_ARTIFACT,
    ARMS,
    OPTIMIZE_PROMPT,
    PYIRON_NODES_ROOT,
    WORKFLOW_OPT_GUIDE,
    BenchOutcome,
    agent_env,
    build_node_library_mirror,
    expect_of,
    generation_prompt,
    get_tasks,
    progress_listener,
    repair_prompt,
    run_claude_code,
    slugify,
    spec_of,
    tier_of,
    update_freeze_manifest_model_id,
    validate_workflow_file,
    write_freeze_manifest,
    write_node_index,
)
from pyiron_ai import workflow_bench

# ── Local node definitions ──────────────────────


@as_function_node("tasks")
def TaskSuite(
    tier: Optional[
        Literal["all", "generic", "pyiron_nodes", "atomistic", "atomistic_hard"]
    ] = "all",
    limit: int = 0,
):
    """Return the curated benchmark tasks for one difficulty tier.

    The suite lives in ``pyiron_ai.workflow_bench.TASK_SUITE`` and is graded by
    how much of aiflow a task forces the agent to understand:

    ``generic``
        plain node composition — arrays, maths, plotting.
    ``pyiron_nodes``
        higher-order composition — sweeps with ``IterToDataFrame``, group
        nodes, parallel executors.
    ``atomistic``
        the materials node stack — structures, ASE/EMT energies, surfaces.
    ``atomistic_hard``
        real simulation protocols — H diffusion barriers by NEB, the melting
        point of Al, a GaN(0001) surface phase diagram.  These are range-checked
        against physically sensible values by the ``plausible`` tier, and they
        carry their own longer per-task timeouts.

    Parameters
    ----------
    tier : str
        Which tier to run (dropdown in the GUI); ``"all"`` concatenates them.
    limit : int
        Truncate to the first *n* tasks; ``0`` means no limit.  Useful for a
        cheap smoke run before committing to a full benchmark.

    Returns
    -------
    tasks : list[str]
        The task prompts, ready to wire into ``IterToDataFrame.values``.
    """
    return get_tasks(tier=tier, limit=limit)


@as_function_node("lib_dir")
def NodeLibraryMirror(allow_reference_workflows: bool = False):
    """Build the node library the agent is allowed to read.

    ``pyiron_nodes/Workflows/`` holds finished reference workflows for several
    benchmark tasks — a GaN surface phase diagram, an Al melting point, an H NEB
    barrier.  Exposing them would turn the aiflow arm into a retrieval test, so
    by default the agent gets a mirror of the library with that directory
    removed.

    Setting ``allow_reference_workflows=True`` points it at the real repository
    instead.  That is a legitimate second condition to measure — how much does
    having a worked example help? — but it is not the number to compare against
    the from-scratch arm.

    Parameters
    ----------
    allow_reference_workflows : bool
        ``False`` (default) mirrors the library without ``Workflows/``.

    Returns
    -------
    lib_dir : str
        Directory to hand the agent via ``--add-dir``.
    """
    return str(build_node_library_mirror(allow_reference_workflows))


@as_function_node("workdir")
def BenchWorkDir(path: str = "bench_runs"):
    """Single source of truth for the output directory.

    Wire ``wf.workdir.path`` to change where all arms write their artifacts
    and where ``results.csv`` lands — without having to update three separate
    nodes.  Use a dated subdirectory (e.g. ``bench_runs/generic_2026-08-28``)
    to avoid overwriting a previous run.
    """
    return path


@as_function_node(
    [
        "model",
        "arms",
        "max_repairs",
        "exec_timeout_s",
        "max_budget_usd",
        "agent_timeout_s",
        "effort",
        "run_variant",
        "run_optimize",
        "save_chat",
    ]
)
def BenchSettings(
    model: Optional[Literal["sonnet", "opus", "haiku"]] = "sonnet",
    model_other: str = "",
    arms: Optional[Literal["aiflow", "scratch", "both"]] = "both",
    max_repairs: int = 3,
    exec_timeout_s: int = 300,
    max_budget_usd: float = AGENT_BUDGET_USD,
    agent_timeout_s: int = AGENT_TIMEOUT_S,
    effort: Optional[Literal["default", "low", "medium", "high", "max"]] = "default",
    run_variant: bool = True,
    run_optimize: bool = False,
    save_chat: bool = True,
):
    """Single source of truth for everything the two arms must share.

    The comparison this benchmark makes is only a comparison if the two
    ``WorkflowAgent`` instances differ in ``arm`` and in nothing else.  Setting
    the model, the repair budget or the timeouts on each node separately makes
    that an assumption about the canvas; wiring both from here makes it a
    property of the graph.

    Parameters
    ----------
    model : str
        Claude Code model alias (dropdown).
    model_other : str
        Free-text model id, used instead of *model* when non-empty — the CLI's
        ``--model`` takes any alias the installed ``claude`` accepts, and the
        dropdown cannot list future ones.
    arms : str
        Which arm(s) to run: ``"aiflow"``, ``"scratch"`` or ``"both"``.  Wire
        into ``ArmTasks.arms`` to switch a whole branch of the graph off; a
        deselected arm sweeps an empty task list and contributes no rows.
    max_repairs : int
        Repair budget per task.  ``0`` measures raw first-attempt quality.
    exec_timeout_s : int
        Hard limit per validation run.  Tasks that declare a longer timeout of
        their own (the ``atomistic_hard`` tier does) override this upwards.
    max_budget_usd : float
        Spend cap per agent turn.
    agent_timeout_s : int
        Wall-clock ceiling per agent turn.
    effort : str
        Reasoning effort forwarded to ``claude --effort``; ``"default"`` omits
        the flag and lets the CLI choose.
    run_variant : bool
        Ask for the declared follow-up edit after a task passes.
    run_optimize : bool
        After a task passes, spend one turn applying
        ``workflow_optimization_guide.md`` and re-validate (aiflow arm only).
    save_chat : bool
        Keep the per-turn chat logs (``chat_gen.jsonl`` and friends).

    Returns
    -------
    model, arms, max_repairs, exec_timeout_s, max_budget_usd, agent_timeout_s, effort, run_variant, run_optimize, save_chat
        One port per setting, ready to wire into both ``WorkflowAgent`` nodes.
    """
    # Warn rather than raise: this node only configures the run, and a caller
    # who has stubbed the agent out legitimately needs no CLI.  The real run
    # still fails every task, but at least it says so before the first one.
    if workflow_bench.claude_bin() is None:
        print(
            "WARNING: `claude` CLI not found on PATH, in "
            f"${workflow_bench.CLAUDE_BIN_ENV}, or in the usual install "
            "locations. Unless the agent is stubbed out, every task will come "
            "back `no_file` and the report will measure this harness, not "
            f"aiflow. Set os.environ['{workflow_bench.CLAUDE_BIN_ENV}'].",
            flush=True,
        )

    resolved_model = model_other.strip() or model
    # "default" is the GUI's way of saying "omit --effort"; the agent nodes
    # spell that as the empty string.
    effort_flag = "" if effort in (None, "", "default") else effort
    return (
        resolved_model,
        arms,
        max_repairs,
        exec_timeout_s,
        max_budget_usd,
        agent_timeout_s,
        effort_flag,
        run_variant,
        run_optimize,
        save_chat,
    )


@as_function_node("tasks")
def ArmTasks(
    tasks: list = None,
    arm: Optional[Literal["aiflow", "scratch"]] = "aiflow",
    arms: Optional[Literal["aiflow", "scratch", "both"]] = "both",
):
    """Gate a task list on whether *arm* is one of the selected *arms*.

    Lets the single ``BenchSettings.arms`` dropdown decide which branches of the
    graph actually spend money: the deselected arm gets an empty list, its sweep
    produces an empty frame and ``StackResults`` drops it.  Without this both
    sweeps always run, and the only way to measure one arm is to delete nodes.

    Parameters
    ----------
    tasks : list
        The full task list — wire ``TaskSuite`` or ``TaskList``.
    arm : str
        Which arm this instance guards; set it to match the ``WorkflowAgent``
        it feeds.
    arms : str
        The selection — wire ``BenchSettings.arms``.

    Returns
    -------
    tasks : list
        *tasks* when this arm is selected, otherwise ``[]``.
    """
    if not tasks:
        return []
    return list(tasks) if arms in ("both", arm) else []


@as_function_node("tasks")
def TaskList(
    task_1: str = "Sample sin(x) at 50 points between 0 and 2*pi and plot it.",
    task_2: str = "",
    task_3: str = "",
    task_4: str = "",
    task_5: str = "",
):
    """Collect up to five hand-written tasks into a list.

    An ad-hoc alternative to ``TaskSuite``: each ``task_n`` is a GUI-settable
    string port and empty strings are dropped, so leaving the later ports blank
    simply runs fewer tasks.  Wire ``tasks`` into ``IterToDataFrame.values``.

    Returns
    -------
    tasks : list[str]
        Non-empty task strings, in order.
    """
    return [t for t in [task_1, task_2, task_3, task_4, task_5] if t.strip()]



def _progress(verbose: bool, arm: str, label: str):
    """A reporter for ``progress_listener``, or ``None`` to stay quiet.

    ``None`` is the point: it keeps ``run_claude_code`` on its non-streaming
    path, so a quiet run behaves exactly as it did before.  A benchmark task
    otherwise prints nothing for minutes, which is what let a run that never
    reached the agent at all look merely fast.
    """
    if not verbose:
        return None
    tag = f"[{arm}/{slugify(str(label), 24)}]"

    def report(line: str) -> None:
        print(f"{tag} {line}", flush=True)

    return report


def warn_if_the_harness_never_ran(df) -> str:
    """Shout when every row failed before the agent started, and return the text.

    A benchmark that could not start the agent still produces a perfectly
    formatted report full of zeros, which reads as "aiflow scored 0 %" when it
    actually means "this harness is broken".  The distinction is the
    ``agent_error`` column, so it is checked here rather than left to the eye.
    """
    if df is None or not len(df) or "final_stage" not in df:
        return ""
    if not (df["final_stage"] == "no_file").all():
        return ""
    reasons = sorted({str(e).strip() for e in df.get("agent_error", []) if str(e).strip()})
    if not reasons:
        return ""
    banner = "\n".join(
        [
            "!" * 78,
            "  NOT A RESULT: every task failed before the agent ran.",
            f"  {len(df)} row(s), all `no_file`, blamed on the harness.",
            "",
            *(f"    {r}" for r in reasons),
            "",
            "  Fix the above and re-run; the 0 % below measures this harness,",
            "  not aiflow.",
            "!" * 78,
        ]
    )
    print(banner, flush=True)
    return banner


# ── The three steps of the agentic loop, each an ordinary node ──────────────


@as_function_node(
    [
        "path",
        "session_id",
        "gen_seconds",
        "cost_usd",
        "num_turns",
        "tokens_in",
        "tokens_cache_created",
        "tokens_cache_read",
        "tokens_out",
        "agent_error",
        "timed_out",
        "model_id",
    ]
)
def GenerateWorkflow(
    task: str = "",
    model: str = "sonnet",
    workdir: str = "bench_runs/task",
    exec_timeout_s: int = 300,
    max_budget_usd: float = AGENT_BUDGET_USD,
    agent_timeout_s: int = AGENT_TIMEOUT_S,
    arm: Optional[Literal["aiflow", "scratch"]] = "aiflow",
    lib_dir: str = "",
    effort: str = "",
    chat_log_path: str = "",
    verbose: bool = False,
):
    """Have Claude Code write one solution for *task*, in the given *arm*.

    Runs a headless ``claude -p`` session in *workdir* with Read/Grep/Glob/Write
    and no Bash.

    In the ``aiflow`` arm the agent is told to read ``docs/llm_workflow_guide.md``
    first and to search the node library for an existing node before writing a
    new one, and must leave a fully wired module-level ``wf`` in ``workflow.py``
    without running it.

    In the ``scratch`` arm the framework is taken away: it may read the installed
    third-party packages (ASE, LAMMPS, phonopy, …) but not ``core`` or
    ``pyiron_nodes``, and must write a ``solution.py`` exposing
    ``main() -> dict``.  Both arms get the same model, tools and budget, which
    is what makes the two columns comparable.

    In the aiflow arm the agent's working directory is seeded with
    ``NODE_INDEX.md``, a generated one-line-per-node catalogue of *lib_dir*, and
    the prompt tells it to read that instead of grepping the library.  Without it
    the search alone consumed the whole wall clock on the hard tasks.

    Parameters
    ----------
    arm : str
        ``"aiflow"`` or ``"scratch"`` (dropdown in the GUI).
    lib_dir : str
        Node library to expose in the aiflow arm — wire ``NodeLibraryMirror``
        here.  Empty falls back to the real ``pyiron_nodes`` repository, which
        includes the reference workflows.  Ignored by the scratch arm.
    agent_timeout_s : int
        Wall-clock ceiling for the generation turn.  A turn that hits it is
        retried once from scratch: a timeout is a harness event, not a verdict on
        the agent, and must not be scored as a failed attempt.
    max_budget_usd : float
        Spend cap for one turn.

    Returns
    -------
    path : str
        Absolute path of the file the agent was asked to write.  It may not
        exist if the agent failed — the caller checks.
    session_id : str
        Claude Code session id; pass it to ``RepairWorkflow`` so the repair turn
        keeps the generation context.
    gen_seconds : float
        Wall-clock seconds for the generation turn, both attempts included.
    cost_usd : float
        Cost of this turn.
    num_turns : int
        Agent turns consumed.
    agent_error : str
        Non-empty when the CLI itself failed (timeout, budget cap, auth).
    timed_out : bool
        The generation hit ``agent_timeout_s`` — on both attempts, if it was
        retried.  Kept apart from ``agent_error`` so a run out of wall clock is
        never counted as a workflow the agent got wrong.
    """
    from pathlib import Path

    workdir = Path(workdir).resolve()
    target = workdir / ARM_ARTIFACT[arm]
    if arm == "aiflow":
        write_node_index(lib_dir or PYIRON_NODES_ROOT, workdir)

    prompt = generation_prompt(task, exec_timeout_s, arm=arm, lib_dir=lib_dir or None)

    def attempt():
        return run_claude_code(
            prompt,
            workdir=workdir,
            model=model,
            max_budget_usd=max_budget_usd,
            timeout_s=agent_timeout_s,
            arm=arm,
            lib_dir=lib_dir or None,
            effort=effort or None,
            chat_log_path=chat_log_path or None,
            env=agent_env(),
            on_event=progress_listener(_progress(verbose, arm, task)),
        )

    def hit_wall(r):
        return r.is_error and "timed out" in (r.error or "")

    run = attempt()
    seconds, cost, turns = run.seconds, run.cost_usd, run.num_turns
    tokens_in = run.tokens_in
    tokens_cache_created = run.tokens_cache_created
    tokens_cache_read = run.tokens_cache_read
    tokens_out = run.tokens_out
    if hit_wall(run) and not target.is_file():
        run = attempt()
        seconds += run.seconds
        cost += run.cost_usd
        turns += run.num_turns
        tokens_in += run.tokens_in
        tokens_cache_created += run.tokens_cache_created
        tokens_cache_read += run.tokens_cache_read
        tokens_out += run.tokens_out

    return (
        str(target),
        run.session_id,
        round(seconds, 1),
        cost,
        turns,
        tokens_in,
        tokens_cache_created,
        tokens_cache_read,
        tokens_out,
        run.error if run.is_error else "",
        hit_wall(run),
        run.model_id,
    )


@as_function_node(
    [
        "stage",
        "syntax_ok",
        "constructs_ok",
        "executes_ok",
        "plausible_ok",
        "roundtrip_ok",
        "error",
        "n_nodes",
        "n_edges",
        "measured_json",
        "c1_violations",
        "c2_violations",
    ]
)
def ValidateWorkflow(
    path: str = "",
    timeout_s: int = 300,
    arm: Optional[Literal["aiflow", "scratch"]] = "aiflow",
    task: str = "",
    expect_json: str = "",
):
    """Run the five-tier validation ladder on a generated file.

    Tier 1 parses the source; tiers 2-5 run in an isolated child process with a
    hard timeout, so an artifact that hangs or crashes the interpreter cannot
    take the benchmark down with it.

    ==== ================================== ==================================
    Tier aiflow arm                         scratch arm
    ==== ================================== ==================================
    1    ``ast.parse(source)``              identical
    2    module leaves a wired ``wf``       ``main`` callable, no framework
    3    ``wf.run()`` completes             ``main()`` returns a ``dict``
    4    named node output within range     named dict key within range
    5    save / load / re-run agrees        fresh-process re-run agrees
    ==== ================================== ==================================

    Parameters
    ----------
    task : str
        The task prompt.  Used only to look up the acceptance ranges for tier 4;
        tasks without ranges pass that tier automatically.
    expect_json : str
        Explicit ``{name: [low, high]}`` ranges as JSON, overriding the lookup by
        *task*.  Used when re-validating a variant, whose plausible range is not
        the original's — copper does not melt at aluminium's temperature.

    Returns
    -------
    stage : str
        ``"complete"`` or ``"<tier>_fail"`` — the first tier that failed.
    syntax_ok, constructs_ok, executes_ok, plausible_ok, roundtrip_ok : bool
        Per-tier verdicts.
    error : str
        Truncated traceback or message from the failing tier; ``""`` on success.
    n_nodes, n_edges : int
        Size of the graph (aiflow) or the number of top-level definitions
        (scratch).
    measured_json : str
        The named results actually extracted, as JSON — the physical answer the
        run produced, for the record.
    """
    import json

    expect = json.loads(expect_json) if expect_json else expect_of(task)
    v = validate_workflow_file(path, timeout_s=timeout_s, arm=arm, expect=expect)
    return (
        v.stage,
        v.syntax_ok,
        v.constructs_ok,
        v.executes_ok,
        v.plausible_ok,
        v.roundtrip_ok,
        v.error,
        v.n_nodes,
        v.n_edges,
        json.dumps(v.measured),
        v.c1_violations,
        v.c2_violations,
    )


@as_function_node(
    [
        "session_id",
        "repair_seconds",
        "cost_usd",
        "num_turns",
        "tokens_in",
        "tokens_cache_created",
        "tokens_cache_read",
        "tokens_out",
        "agent_error",
    ]
)
def RepairWorkflow(
    path: str = "",
    session_id: str = "",
    stage: str = "executes",
    error: str = "",
    model: str = "sonnet",
    max_budget_usd: float = AGENT_BUDGET_USD,
    agent_timeout_s: int = AGENT_TIMEOUT_S,
    arm: Optional[Literal["aiflow", "scratch"]] = "aiflow",
    lib_dir: str = "",
    effort: str = "",
    chat_log_path: str = "",
    verbose: bool = False,
):
    """Hand a validation failure back to the agent that wrote the code.

    Resumes the generation session (``claude -p --resume``) so the agent still
    remembers its own design decisions, tells it which tier failed and why, and
    lets it edit its file in place.

    Returns
    -------
    session_id : str
        Session id to use for the next repair turn.
    repair_seconds : float
        Wall-clock seconds for this repair turn.
    cost_usd : float
        Cost of this turn.
    num_turns : int
        Agent turns consumed.
    agent_error : str
        Non-empty when the CLI itself failed.
    """
    from pathlib import Path

    run = run_claude_code(
        repair_prompt(stage, error, arm=arm, lib_dir=lib_dir or None),
        workdir=Path(path).parent,
        model=model,
        resume=session_id or None,
        max_budget_usd=max_budget_usd,
        timeout_s=agent_timeout_s,
        arm=arm,
        lib_dir=lib_dir or None,
        effort=effort or None,
        chat_log_path=chat_log_path or None,
        env=agent_env(),
        on_event=progress_listener(_progress(verbose, arm, path)),
    )
    return (
        run.session_id or session_id,
        run.seconds,
        run.cost_usd,
        run.num_turns,
        run.tokens_in,
        run.tokens_cache_created,
        run.tokens_cache_read,
        run.tokens_out,
        run.error if run.is_error else "",
    )


@as_function_node(
    [
        "session_id",
        "variant_seconds",
        "cost_usd",
        "num_turns",
        "tokens_in",
        "tokens_cache_created",
        "tokens_cache_read",
        "tokens_out",
        "diff_loc",
        "agent_error",
    ]
)
def VaryWorkflow(
    path: str = "",
    session_id: str = "",
    variant: str = "",
    expect_json: str = "",
    model: str = "sonnet",
    max_budget_usd: float = AGENT_BUDGET_USD,
    agent_timeout_s: int = AGENT_TIMEOUT_S,
    arm: Optional[Literal["aiflow", "scratch"]] = "aiflow",
    effort: str = "",
    chat_log_path: str = "",
    verbose: bool = False,
):
    """Ask the agent for one follow-up change to a solution that already works.

    "Now do the same for copper."  A benchmark that only scores the first working
    answer measures the wrong thing: research is a sequence of edits, and the
    claim aiflow makes is about the second question, not the first.  This node
    times that second question, identically in both arms, and reports the size of
    the diff the agent had to make.

    Returns
    -------
    diff_loc : int
        Lines added plus removed.  A rewired graph should need fewer than a
        rewritten script — and if it does not, that is the honest result.
    """
    import json
    from pathlib import Path

    target = Path(path)
    before = target.read_text(encoding="utf-8", errors="replace")

    run = run_claude_code(
        workflow_bench.variant_prompt(
            variant, arm=arm, expect=json.loads(expect_json) if expect_json else None
        ),
        workdir=target.parent,
        model=model,
        resume=session_id or None,
        max_budget_usd=max_budget_usd,
        timeout_s=agent_timeout_s,
        arm=arm,
        effort=effort or None,
        chat_log_path=chat_log_path or None,
        env=agent_env(),
        on_event=progress_listener(_progress(verbose, arm, path)),
    )
    after = (
        target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
    )
    return (
        run.session_id or session_id,
        run.seconds,
        run.cost_usd,
        run.num_turns,
        run.tokens_in,
        run.tokens_cache_created,
        run.tokens_cache_read,
        run.tokens_out,
        workflow_bench.count_edit(before, after),
        run.error if run.is_error else "",
    )


@as_function_node(
    [
        "stage",
        "constructs_ok",
        "executes_ok",
        "roundtrip_ok",
        "error",
        "n_nodes",
        "n_edges",
        "c1_violations",
        "c2_violations",
        "diff_loc",
        "cost_usd",
        "seconds",
        "agent_error",
    ]
)
def OptimizeWorkflow(
    path: str = "",
    session_id: str = "",
    model: str = "sonnet",
    max_budget_usd: float = AGENT_BUDGET_USD,
    agent_timeout_s: int = AGENT_TIMEOUT_S,
    arm: Optional[Literal["aiflow", "scratch"]] = "aiflow",
    lib_dir: str = "",
    effort: str = "",
    chat_log_path: str = "",
    verbose: bool = False,
):
    """Apply workflow_optimization_guide.md in one LLM turn, then re-validate.

    Sends a single repair-style prompt to the agent that wrote the workflow,
    instructing it to read the optimization guide and apply the C1/C2 structural
    rules and Strategies A-D.  Re-runs the full validation ladder on the edited
    file and returns the before/after graph quality metrics.

    Only called from ``WorkflowAgent`` when ``run_optimize=True`` and the task
    already reached ``complete`` — it improves a working workflow, not a broken one.

    Returns
    -------
    stage : str
        Validation stage after optimization (should be ``"complete"`` if the
        optimization preserved the workflow semantics).
    c1_violations, c2_violations : int
        Graph-quality violation counts *after* optimization.
    diff_loc : int
        Lines added plus removed by the optimization turn.
    cost_usd, seconds : float
        Cost and wall-clock for this turn.
    agent_error : str
        Non-empty when the CLI itself failed.
    """
    import time
    from pathlib import Path

    target = Path(path)
    before = (
        target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
    )
    prompt = OPTIMIZE_PROMPT.format(guide_opt=WORKFLOW_OPT_GUIDE)

    t0 = time.time()
    run = run_claude_code(
        prompt,
        workdir=target.parent,
        model=model,
        resume=session_id or None,
        max_budget_usd=max_budget_usd,
        timeout_s=agent_timeout_s,
        arm=arm,
        lib_dir=lib_dir or None,
        effort=effort or None,
        chat_log_path=chat_log_path or None,
        env=agent_env(),
        on_event=progress_listener(_progress(verbose, arm, path)),
    )
    after = (
        target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
    )
    diff = workflow_bench.count_edit(before, after)

    checker = ValidateWorkflow(
        path=path,
        timeout_s=600,
        arm=arm,
        task="",
    )
    checker.run()

    return (
        checker.outputs.stage.value,
        checker.outputs.constructs_ok.value,
        checker.outputs.executes_ok.value,
        checker.outputs.roundtrip_ok.value,
        checker.outputs.error.value,
        checker.outputs.n_nodes.value,
        checker.outputs.n_edges.value,
        checker.outputs.c1_violations.value,
        checker.outputs.c2_violations.value,
        diff,
        run.cost_usd,
        round(time.time() - t0, 1),
        run.error if run.is_error else "",
    )


@as_function_node("result")
def WorkflowAgent(
    task: str = "",
    model: Optional[Literal["sonnet", "opus", "haiku"]] = "sonnet",
    max_repairs: int = 3,
    workdir: str = "bench_runs",
    exec_timeout_s: int = 300,
    max_budget_usd: float = AGENT_BUDGET_USD,
    agent_timeout_s: int = AGENT_TIMEOUT_S,
    arm: Optional[Literal["aiflow", "scratch"]] = "aiflow",
    lib_dir: str = "",
    run_variant: bool = True,
    run_optimize: bool = False,
    verbose: bool = True,
    effort: str = "",
    save_chat: bool = True,
):
    """Generate → validate → repair one task in one arm, and measure every step.

    A **higher-order node**: the whole agentic loop is driven inside this one
    function by running ``GenerateWorkflow``, ``ValidateWorkflow`` and
    ``RepairWorkflow`` with ``.run()``.  Keeping the loop internal means the
    outer workflow is still a DAG, so ``WorkflowAgent`` can be dropped straight
    into ``IterToDataFrame`` and swept over a whole task suite — the pattern the
    rest of aiflow uses for parameter sweeps.

    Use two instances, one per ``arm``, to get the aiflow-vs-scratch comparison;
    everything else about them should be identical, or the comparison is not one.

    This node deliberately has **no** ``store`` port: it is used as a template
    inside ``IterToDataFrame``, where per-iteration caching would be wrong.

    Parameters
    ----------
    task : str
        The plain-English task — the field swept when batching.
    model : str
        Claude Code model alias (dropdown in the GUI).  Sweep this port instead
        of ``task`` to compare models on a fixed task.
    max_repairs : int
        Repair budget.  ``0`` measures raw first-attempt quality.
    workdir : str
        Parent directory for the per-task scratch directories.  The arm is added
        as a subdirectory, so the two arms never overwrite each other.
    exec_timeout_s : int
        Hard limit for each validation run.  Tasks that declare their own longer
        timeout (the ``atomistic_hard`` tier does) override this upwards.
    max_budget_usd : float
        Spend cap per agent turn.
    agent_timeout_s : int
        Wall-clock ceiling per agent turn.  Shared by both arms — an asymmetric
        ceiling would measure the harness rather than the framework.
    arm : str
        ``"aiflow"`` writes a workflow; ``"scratch"`` writes plain Python with
        the framework forbidden.
    lib_dir : str
        Node library to expose in the aiflow arm — wire ``NodeLibraryMirror``.
    run_variant : bool
        After a task passes, ask for the follow-up edit its ``TaskSpec`` declares
        and re-run the ladder.  One extra turn per solved task; switch it off for
        a cheap pass-rate-only run.
    verbose : bool
        Print per-attempt progress.

    Returns
    -------
    result : BenchOutcome
        A dataclass with every measured field.  ``IterToDataFrame`` expands it
        into one DataFrame column per field.
    """
    import json
    import shutil
    import time
    from pathlib import Path

    t0 = time.time()
    scratch = Path(workdir).resolve() / arm / slugify(task)
    scratch.mkdir(parents=True, exist_ok=True)

    spec = spec_of(task)
    # A hard task states how long its physics legitimately takes; honour that
    # rather than failing it on a timeout meant for two-second toy graphs.
    timeout_s = max(exec_timeout_s, spec.timeout_s if spec else 0)

    out = BenchOutcome(
        task=task,
        tier=tier_of(task),
        model=model,
        arm=arm,
        scored=bool(expect_of(task)),
    )

    def say(msg):
        if verbose:
            print(f"[{arm}/{slugify(task, 24)}] {msg}", flush=True)

    def _save_snapshot(path, n):
        src = Path(path)
        if src.is_file():
            shutil.copy2(src, scratch / f"{src.stem}_attempt_{n}.py")

    def _save_error(error, n):
        (scratch / f"error_attempt_{n}.txt").write_text(error or "")

    # ── generate ────────────────────────────────────────────────────────────
    say(f"generating with {model} …")
    gen = GenerateWorkflow(
        task=task,
        model=model,
        workdir=str(scratch),
        exec_timeout_s=timeout_s,
        max_budget_usd=max_budget_usd,
        agent_timeout_s=agent_timeout_s,
        arm=arm,
        lib_dir=lib_dir,
        effort=effort,
        chat_log_path=str(scratch / "chat_gen.jsonl") if save_chat else "",
        verbose=verbose,
    )
    gen.run()
    path = gen.outputs.path.value
    session_id = gen.outputs.session_id.value
    out.workflow_path = path
    out.gen_seconds = gen.outputs.gen_seconds.value
    out.gen_cost_usd = gen.outputs.cost_usd.value
    out.gen_tokens_in = gen.outputs.tokens_in.value
    out.gen_tokens_cache_created = gen.outputs.tokens_cache_created.value
    out.gen_tokens_cache_read = gen.outputs.tokens_cache_read.value
    out.gen_tokens_out = gen.outputs.tokens_out.value
    out.cost_usd += gen.outputs.cost_usd.value
    out.num_turns += gen.outputs.num_turns.value
    out.total_tokens_in += gen.outputs.tokens_in.value
    out.total_tokens_cache_created += gen.outputs.tokens_cache_created.value
    out.total_tokens_cache_read += gen.outputs.tokens_cache_read.value
    out.total_tokens_out += gen.outputs.tokens_out.value
    out.agent_error = gen.outputs.agent_error.value
    out.timed_out = gen.outputs.timed_out.value
    out.model_id_actual = gen.outputs.model_id.value

    if not Path(path).is_file():
        out.stage_first = out.final_stage = "no_file"
        out.first_error = out.last_error = (
            out.agent_error or f"the agent produced no {Path(path).name}"
        )
        # A wall-clock timeout — or a CLI that never started at all — says
        # nothing about the framework under test.  Blaming a misconfigured
        # harness on the agent is how a broken run comes to look like a 0 %
        # pass rate instead of like a broken run.
        out.blamed_on = "harness" if (out.timed_out or out.agent_error) else "agent"
        out.total_seconds = round(time.time() - t0, 1)
        say(f"no file produced ({out.blamed_on}: {out.last_error[:120]})")
        return out

    # ── validate, then repair until it passes or the budget runs out ────────
    checker = ValidateWorkflow(path=path, timeout_s=timeout_s, arm=arm, task=task)
    checker.run()
    stage = checker.outputs.stage.value
    error = checker.outputs.error.value
    say(f"attempt 0 → {stage}")

    out.stage_first = stage
    out.first_error = error
    out.syntax_ok_first = checker.outputs.syntax_ok.value
    out.constructs_ok_first = checker.outputs.constructs_ok.value
    out.executes_ok_first = checker.outputs.executes_ok.value
    out.plausible_ok_first = checker.outputs.plausible_ok.value
    out.roundtrip_ok_first = checker.outputs.roundtrip_ok.value

    # Snapshot attempt 0 (the first generated version)
    _save_snapshot(path, 0)
    _save_error(error, 0)

    repair_details = []

    while stage != "complete" and out.repair_cycles < max_repairs:
        failed_tier = stage.replace("_fail", "")
        say(f"repair {out.repair_cycles + 1} (failed at {failed_tier}) …")
        fixer = RepairWorkflow(
            path=path,
            session_id=session_id,
            stage=failed_tier,
            error=error,
            model=model,
            max_budget_usd=max_budget_usd,
            agent_timeout_s=agent_timeout_s,
            arm=arm,
            lib_dir=lib_dir,
            effort=effort,
            chat_log_path=(
                str(scratch / f"chat_repair_{out.repair_cycles + 1}.jsonl")
                if save_chat
                else ""
            ),
            verbose=verbose,
        )
        fixer.run()
        session_id = fixer.outputs.session_id.value
        out.repair_cycles += 1
        repair_cost = fixer.outputs.cost_usd.value
        repair_tok_in = fixer.outputs.tokens_in.value
        repair_tok_cache_created = fixer.outputs.tokens_cache_created.value
        repair_tok_cache_read = fixer.outputs.tokens_cache_read.value
        repair_tok_out = fixer.outputs.tokens_out.value
        out.repair_seconds += fixer.outputs.repair_seconds.value
        out.cost_usd += repair_cost
        out.num_turns += fixer.outputs.num_turns.value
        out.total_tokens_in += repair_tok_in
        out.total_tokens_cache_created += repair_tok_cache_created
        out.total_tokens_cache_read += repair_tok_cache_read
        out.total_tokens_out += repair_tok_out
        if not out.agent_error:
            out.agent_error = fixer.outputs.agent_error.value

        checker = ValidateWorkflow(path=path, timeout_s=timeout_s, arm=arm, task=task)
        checker.run()
        prev_stage = stage
        stage = checker.outputs.stage.value
        error = checker.outputs.error.value
        say(f"attempt {out.repair_cycles} → {stage}")

        # Snapshot this attempt
        _save_snapshot(path, out.repair_cycles)
        _save_error(error, out.repair_cycles)

        repair_details.append(
            {
                "cycle": out.repair_cycles,
                "stage_before": prev_stage,
                "stage_after": stage,
                "seconds": round(fixer.outputs.repair_seconds.value, 1),
                "cost_usd": round(repair_cost, 4),
                "tokens_in": repair_tok_in,
                "tokens_cache_created": repair_tok_cache_created,
                "tokens_cache_read": repair_tok_cache_read,
                "tokens_out": repair_tok_out,
                "snapshot_file": f"{Path(path).stem}_attempt_{out.repair_cycles}.py",
                "error_file": f"error_attempt_{out.repair_cycles}.txt",
            }
        )

    out.repair_details_json = json.dumps(repair_details)

    out.final_stage = stage
    out.last_error = error
    out.syntax_ok = checker.outputs.syntax_ok.value
    out.constructs_ok = checker.outputs.constructs_ok.value
    out.executes_ok = checker.outputs.executes_ok.value
    out.plausible_ok = checker.outputs.plausible_ok.value
    out.roundtrip_ok = checker.outputs.roundtrip_ok.value
    out.n_nodes = checker.outputs.n_nodes.value
    out.n_edges = checker.outputs.n_edges.value
    out.measured_json = checker.outputs.measured_json.value
    out.c1_violations = checker.outputs.c1_violations.value
    out.c2_violations = checker.outputs.c2_violations.value
    out.hit_repair_cap = stage != "complete" and out.repair_cycles >= max_repairs
    # Composition is an aiflow-arm question: the scratch arm has no graph at all,
    # so leave it False there rather than recording a meaningless verdict.
    out.composed = arm == "aiflow" and out.n_nodes >= 3 and out.n_edges >= 2
    out.blamed_on = workflow_bench.blame_error(error, path)

    source = Path(path).read_text(encoding="utf-8", errors="replace")
    out.loc = workflow_bench.count_loc(source)
    # Node reuse is an aiflow-arm question by construction: the scratch arm is
    # forbidden from importing the library at all, so leave it at 0/0 rather
    # than recording a meaningless zero-reuse "result".
    if arm == "aiflow":
        out.n_reused_nodes, out.n_new_nodes = workflow_bench.count_node_reuse(source)
    # ── optional optimization turn: apply C1/C2 rules to a complete workflow ─
    if run_optimize and stage == "complete" and arm == "aiflow":
        out.optimize_attempted = True
        # Snapshot the validated workflow before the optimization edits it.
        shutil.copy2(path, Path(path).with_name(f"{Path(path).stem}_primary.py"))
        say("optimizing graph structure …")
        optimizer = OptimizeWorkflow(
            path=path,
            session_id=session_id,
            model=model,
            max_budget_usd=max_budget_usd,
            agent_timeout_s=agent_timeout_s,
            arm=arm,
            lib_dir=lib_dir,
            effort=effort,
            chat_log_path=str(scratch / "chat_optimize.jsonl") if save_chat else "",
            verbose=verbose,
        )
        optimizer.run()
        out.optimize_stage = optimizer.outputs.stage.value
        out.optimize_ok = out.optimize_stage == "complete"
        out.optimize_error = optimizer.outputs.error.value
        out.optimize_diff_loc = optimizer.outputs.diff_loc.value
        out.optimize_cost_usd = optimizer.outputs.cost_usd.value
        out.optimize_seconds = optimizer.outputs.seconds.value
        out.c1_violations_after = optimizer.outputs.c1_violations.value
        out.c2_violations_after = optimizer.outputs.c2_violations.value
        out.cost_usd += optimizer.outputs.cost_usd.value
        if out.optimize_ok:
            shutil.copy2(path, Path(path).with_name(f"{Path(path).stem}_optimized.py"))
        say(
            f"optimize → {out.optimize_stage} (diff {out.optimize_diff_loc} LOC, "
            f"C1: {out.c1_violations}→{out.c1_violations_after}, "
            f"C2: {out.c2_violations}→{out.c2_violations_after})"
        )
    # ── the follow-up edit: what does it cost to change your mind? ───────────
    variant, variant_expect = workflow_bench.variant_of(task)
    if run_variant and variant and stage == "complete":
        out.variant = variant
        out.variant_attempted = True
        # The variant edits `path` in place, so without this snapshot the
        # version the ladder actually scored is gone once the run ends — the
        # measured columns would describe a file no longer on disk.
        shutil.copy2(path, Path(path).with_name(f"{Path(path).stem}_primary.py"))
        say("applying the follow-up variant …")
        varier = VaryWorkflow(
            path=path,
            session_id=session_id,
            variant=variant,
            expect_json=json.dumps(variant_expect),
            model=model,
            max_budget_usd=max_budget_usd,
            agent_timeout_s=agent_timeout_s,
            arm=arm,
            effort=effort,
            chat_log_path=str(scratch / "chat_variant.jsonl") if save_chat else "",
            verbose=verbose,
        )
        varier.run()
        out.variant_seconds = round(varier.outputs.variant_seconds.value, 1)
        out.variant_cost_usd = round(varier.outputs.cost_usd.value, 4)
        out.variant_diff_loc = varier.outputs.diff_loc.value
        out.num_turns += varier.outputs.num_turns.value
        out.total_tokens_in += varier.outputs.tokens_in.value
        out.total_tokens_cache_created += varier.outputs.tokens_cache_created.value
        out.total_tokens_cache_read += varier.outputs.tokens_cache_read.value
        out.total_tokens_out += varier.outputs.tokens_out.value

        # Judged on the variant's own ranges; an empty dict means the ladder
        # stops at executes/roundtrip rather than at a range known to be wrong.
        rechecker = ValidateWorkflow(
            path=path,
            timeout_s=timeout_s,
            arm=arm,
            task=task,
            expect_json=json.dumps(variant_expect),
        )
        rechecker.run()
        out.variant_stage = rechecker.outputs.stage.value
        out.variant_error = rechecker.outputs.error.value
        out.variant_ok = out.variant_stage == "complete"
        say(f"variant → {out.variant_stage} ({out.variant_diff_loc} lines changed)")

    out.repair_seconds = round(out.repair_seconds, 1)
    out.cost_usd = round(out.cost_usd + out.variant_cost_usd, 4)
    out.total_seconds = round(time.time() - t0, 1)
    return out


@as_function_node("df")
def StackResults(
    df_aiflow: pd.DataFrame = None,
    df_scratch: pd.DataFrame = None,
    workdir: str = "bench_runs",
):
    """Stack the two arms' result frames row-wise into one table.

    ``IterToDataFrame`` sweeps a single input label, so each arm needs its own
    sweep node; this puts the two back together.  It is a concatenation, not a
    join — the ``arm`` column is what distinguishes the rows, and every
    downstream report groups on it.  Either input may be ``None``, which makes
    single-arm runs work unchanged.

    The stacked frame is also written to ``<workdir>/results.csv``, as the CLI
    path already does.  A GUI run costs real money and hours of wall clock; up to
    now its numbers lived only inside the widget and died with the kernel.

    Parameters
    ----------
    workdir : str
        Where to write ``results.csv``; wire the same value the ``WorkflowAgent``
        nodes use.  Empty disables the write.

    Returns
    -------
    df : pd.DataFrame
        All rows from both arms, index reset.
    """
    from pathlib import Path

    frames = [f for f in (df_aiflow, df_scratch) if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    warn_if_the_harness_never_ran(df)

    if workdir:
        target = Path(workdir).resolve()
        target.mkdir(parents=True, exist_ok=True)
        df.to_csv(target / "results.csv", index=False)
        print(f"results written to {target / 'results.csv'}")
    return df


@as_function_node(["summary", "stats"])
def BenchmarkReport(df: pd.DataFrame = None, max_repairs: int = 3):
    """Aggregate the per-task results into the benchmark's headline metrics.

    Reports, overall and per tier: the pass rate at each tier of the validation
    ladder (first attempt and after repair), the success-vs-repair-budget curve
    ``P(executes | <= k repairs)``, the repair-cycle distribution, the share of
    graph nodes reused from ``pyiron_nodes`` rather than newly written, and the
    total wall-clock and cost.

    Returns
    -------
    summary : str
        Human-readable report (also printed).
    stats : pd.DataFrame
        One row per group (``overall`` plus one per tier).
    """
    stats = workflow_bench.summarize(df, max_repairs=max_repairs)
    summary = workflow_bench.format_summary(stats, df)
    print(summary)
    return summary, stats


@as_function_node("figure")
def PlotBenchmark(df: pd.DataFrame = None, max_repairs: int = 3):
    """Plot the ladder pass rates, the repair curve and the arm comparison.

    Left panel: percentage of tasks reaching each tier of the ladder, first
    attempt versus after repair.  Middle panel: cumulative execution success as
    a function of the repair budget — the shape reported in the paper.  Right
    panel: aiflow versus from-scratch per ladder tier, which is the controlled
    comparison; it is drawn only when the frame actually holds both arms.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    stats = workflow_bench.summarize(df, max_repairs=max_repairs)
    overall = stats[stats["group"] == "overall"].iloc[0]
    stages = list(workflow_bench.STAGES)
    arm_rows = stats[stats["group"].str.startswith("arm=")]
    n_panels = 3 if len(arm_rows) > 1 else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4))
    ax_l, ax_m = axes[0], axes[1]

    x = np.arange(len(stages))
    first = [overall.get(f"{s}_first_pct", np.nan) for s in stages]
    final = [overall.get(f"{s}_final_pct", np.nan) for s in stages]
    ax_l.bar(x - 0.2, first, width=0.4, label="first attempt")
    ax_l.bar(x + 0.2, final, width=0.4, label=f"after <= {max_repairs} repairs")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(stages, rotation=20, ha="right")
    ax_l.set_ylim(0, 105)
    ax_l.set_ylabel("tasks passing (%)")
    ax_l.set_title("Validation ladder")
    ax_l.legend(frameon=False, fontsize=8)

    ks = list(range(max_repairs + 1))
    curve = [overall.get(f"executes_le_{k}_repairs_pct", np.nan) for k in ks]
    ax_m.plot(ks, curve, marker="o")
    ax_m.set_xticks(ks)
    ax_m.set_xlabel("repair cycles allowed")
    ax_m.set_ylabel("solutions that execute (%)")
    ax_m.set_ylim(0, 105)
    ax_m.set_title("Success vs. repair budget")
    ax_m.grid(alpha=0.3)

    if n_panels == 3:
        ax_r = axes[2]
        width = 0.8 / len(arm_rows)
        for i, (_, row) in enumerate(arm_rows.iterrows()):
            offset = (i - (len(arm_rows) - 1) / 2) * width
            ax_r.bar(
                x + offset,
                [row.get(f"{s}_final_pct", np.nan) for s in stages],
                width=width,
                label=f"{str(row['group']).split('=', 1)[1]} (n={int(row['n_tasks'])})",
            )
        ax_r.set_xticks(x)
        ax_r.set_xticklabels(stages, rotation=20, ha="right")
        ax_r.set_ylim(0, 105)
        ax_r.set_ylabel("tasks passing (%)")
        ax_r.set_title(f"aiflow vs. from scratch (<= {max_repairs} repairs)")
        ax_r.legend(frameon=False, fontsize=8)

    n = int(overall["n_tasks"])
    models = (
        ", ".join(sorted(set(df["model"]))) if df is not None and "model" in df else ""
    )
    fig.suptitle(f"Agentic workflow generation — {n} runs, {models}", fontsize=10)
    fig.tight_layout()
    return fig


# ── The whole suite in one node ─────────────────────────────────────────────


@as_function_node(["df", "stats", "summary", "run_dir"])
def RunBenchmarkSuite(
    tier: Optional[
        Literal["all", "generic", "pyiron_nodes", "atomistic", "atomistic_hard"]
    ] = "generic",
    limit: int = 0,
    tasks: list = None,
    arms: Optional[Literal["aiflow", "scratch", "both"]] = "both",
    model: Optional[Literal["sonnet", "opus", "haiku"]] = "sonnet",
    model_other: str = "",
    max_repairs: int = 3,
    repeats: int = 1,
    workdir: str = "bench_runs",
    exec_timeout_s: int = 300,
    max_budget_usd: float = AGENT_BUDGET_USD,
    agent_timeout_s: int = AGENT_TIMEOUT_S,
    effort: Optional[Literal["default", "low", "medium", "high", "max"]] = "default",
    allow_reference_workflows: bool = False,
    lib_dir: str = "",
    run_variant: bool = True,
    run_optimize: bool = False,
    save_chat: bool = True,
    max_workers: int = 3,
    verbose: bool = True,
):
    """Build and run the whole benchmark — every arm, every repetition — at once.

    The canvas graph below is the same benchmark laid out node by node, which is
    the better way to *look* at it.  This node is the better way to *run* it:
    repetitions and their confidence intervals cannot be expressed as a static
    DAG, and every CLI flag is a port here.

    Examples
    --------
    >>> node = RunBenchmarkSuite(tier="generic", limit=3, repeats=5)
    >>> node.run()
    >>> print(node.outputs.summary.value)

    Parameters
    ----------
    tier, limit :
        Which curated tasks to run; ignored when *tasks* is given.
    tasks : list[str]
        Run these prompts instead of the curated suite.
    arms : str
        ``"both"`` is the controlled comparison; a single arm is a cheaper probe.
    model, model_other :
        The dropdown, or any identifier typed into *model_other*, which wins.
    repeats : int
        Independent repetitions.  More than one adds per-cell Wilson intervals
        via :func:`pyiron_ai.bench_stats.aggregate_repeats`; the LLM is
        stochastic, so a single repetition cannot separate a real difference
        between the arms from noise.
    max_workers : int
        Tasks generated in parallel.  Each one is a separate agent, so this
        multiplies the spend rate, not the total.

    Returns
    -------
    df, stats, summary, run_dir
        The per-task rows, the statistics table, the printable report, and the
        directory holding ``results.csv`` and the freeze manifest.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import pandas as pd

    from pyiron_nodes.controls import IterToDataFrame

    task_list = [
        t
        for t in (list(tasks) if tasks else get_tasks(tier=tier, limit=limit))
        if str(t).strip()
    ]
    if not task_list:
        raise ValueError(
            "no tasks to run — check `tier`/`limit`, or wire a non-empty `tasks` list"
        )

    # Fail in a second rather than producing a tidy report full of zeros: every
    # task would come back `no_file` without the CLI, which looks like a result.
    claude = workflow_bench.claude_bin()
    if claude is None:
        raise RuntimeError(
            "`claude` CLI not found, so no task could reach the agent — refusing "
            "to start, because the run would report 0 % for the harness rather "
            "than for aiflow. Looked on PATH, in "
            f"${workflow_bench.CLAUDE_BIN_ENV}, and in "
            f"{', '.join(workflow_bench.CLAUDE_BIN_GLOBS)}. Set "
            f"os.environ['{workflow_bench.CLAUDE_BIN_ENV}'] to the binary."
        )

    arm_list = list(ARMS) if arms == "both" else [arms]
    resolved_model = model_other.strip() or model
    effort_flag = "" if effort in (None, "", "default") else effort
    library = lib_dir or str(build_node_library_mirror(allow_reference_workflows))
    root = Path(workdir).resolve()
    n_reps = max(1, int(repeats))
    workers = max(1, int(max_workers))

    print(
        f"running {len(task_list)} task(s) × {len(arm_list)} arm(s) × {n_reps} rep(s) "
        f"with model={resolved_model} into {root}\n"
        f"agent CLI: {claude}\n"
        f"node library visible to the agent: {library}\n",
        flush=True,
    )

    frames = []
    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for rep in range(1, n_reps + 1):
            rep_dir = root / f"rep_{rep:03d}" if n_reps > 1 else root
            rep_dir.mkdir(parents=True, exist_ok=True)
            write_freeze_manifest(
                rep_dir,
                model=resolved_model,
                tier="custom" if tasks else tier,
                arms=arm_list,
                max_repairs=max_repairs,
                exec_timeout=exec_timeout_s,
                effort=effort_flag or None,
            )
            rep_frames = []
            for arm in arm_list:
                template = WorkflowAgent(
                    model=resolved_model,
                    max_repairs=max_repairs,
                    workdir=str(rep_dir),
                    exec_timeout_s=exec_timeout_s,
                    max_budget_usd=max_budget_usd,
                    agent_timeout_s=agent_timeout_s,
                    arm=arm,
                    lib_dir=library,
                    run_variant=run_variant,
                    run_optimize=run_optimize,
                    verbose=verbose,
                    effort=effort_flag,
                    save_chat=save_chat,
                )
                sweep = IterToDataFrame(
                    node=template,
                    input_label="task",
                    values=task_list,
                    executor=pool,
                    debug=False,
                    store=False,
                )
                sweep.run()
                part = sweep.outputs.df.value
                if part is not None and len(part):
                    if n_reps > 1:
                        # The directory name, which is what aggregate_repeats
                        # puts in this column when it reads the run back.
                        part = part.assign(rep=rep_dir.name)
                    rep_frames.append(part)

            # BenchOutcome calls it model_id_actual: the identifier the API
            # reported, as opposed to the alias ("sonnet") that was requested.
            ids = [
                m
                for f in rep_frames
                for m in (f["model_id_actual"] if "model_id_actual" in f else [])
                if str(m).strip() and str(m) != "nan"
            ]
            if ids:
                update_freeze_manifest_model_id(rep_dir, str(ids[0]))

            # One results.csv per repetition, which is the layout
            # ``aggregate_repeats`` reads back; the combined file goes to the
            # root below.
            if rep_frames:
                rep_df = pd.concat(rep_frames, ignore_index=True)
                rep_df.to_csv(rep_dir / "results.csv", index=False)
                frames.append(rep_df)
    finally:
        pool.shutdown(wait=True)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "results.csv", index=False)
    warn_if_the_harness_never_ran(df)

    stats = workflow_bench.summarize(df, max_repairs=max_repairs)
    summary = workflow_bench.format_summary(stats, df)
    print(f"\n{summary}", flush=True)
    print(
        f"\n{len(df)} row(s) in {time.time() - t0:.0f} s → {root / 'results.csv'}",
        flush=True,
    )

    if n_reps > 1:
        from pyiron_ai.bench_stats import aggregate_repeats

        stats = aggregate_repeats(root)

    return df, stats, summary, str(root)


@as_function_node(["stats", "df"])
def AggregateRepeats(run_dir: str = "bench_runs"):
    """Rebuild the per-cell statistics from a finished multi-repetition run.

    Lets a run that was interrupted — or one aggregated with different
    assumptions — be re-reduced without paying for the agent again.

    Returns
    -------
    stats, df
        The per-cell table (mean and Wilson interval per group) and the raw rows
        from every repetition, with a ``rep`` column.
    """
    from pathlib import Path

    import pandas as pd

    from pyiron_ai.bench_stats import aggregate_repeats

    root = Path(run_dir).resolve()
    stats = aggregate_repeats(root)
    raw = root / "results_aggregated.csv"
    df = pd.read_csv(raw) if raw.is_file() else pd.DataFrame()
    return stats, df


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
