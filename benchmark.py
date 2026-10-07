"""
Nodes for the agentic workflow-generation benchmark
===================================================
The node library for the aiflow-vs-from-scratch benchmark: every node the
experiment is built from, independent of any particular wiring of them.

``Workflows/workflow_agent_benchmark`` assembles these into the canvas graph,
and ``Workflows/workflow_agent_benchmark_compact`` into the two-node version.
Import from here when writing a graph of your own.

Entry points
------------
``RunSingleBenchmark``
    One (model, task, arm, rep) cell — sets up all directories, builds the
    library mirror, and runs ``WorkflowAgent``.  For notebooks and one-off
    re-runs.
``RunBenchmarkSuite``
    The whole benchmark as one function node, including ``repeats`` and the
    confidence intervals that need them.  Cannot be expanded.
``BenchmarkSuite``
    The same pipeline as an expandable ``@group_node`` — twelve nodes inside,
    every option a boundary port.  One run is one repetition.
``AggregateRepeats``
    Re-reduce a finished multi-repetition run without paying the agent again.
``BenchmarkReport`` / ``PlotBenchmark``
    The statistics table, and five figures — the headline plus failure anatomy,
    cost, per-task reliability and node reuse, one per output port.
``SingleBenchmarkReport``
    Same outputs as ``BenchmarkReport`` for one ``BenchOutcome``; wire it
    directly to ``RunSingleBenchmark``.
``SummarizeBenchmark``
    Asks the agent to write the run up as markdown with the figures inlined.
    Unwired by default: it costs a turn every time it runs.

Building blocks
---------------
``TaskSuite``, ``TaskList``, ``ArmTasks``
    Where the tasks come from, and which arm sees them.
``BenchSettings``, ``NodeLibraryMirror``, ``BenchWorkDir``
    The settings both arms must share, the library the agent may read, and
    where everything is written.
``WorkflowAgent``
    Generate → validate → repair for one task in one arm; the higher-order
    node the sweeps are built on.
``GenerateWorkflow``, ``ValidateWorkflow``, ``RepairWorkflow``, ``VaryWorkflow``, ``OptimizeWorkflow``
    The individual agent turns, usable on their own.
``StackResults``
    The two arms' frames concatenated, and ``results.csv`` written.

.. warning::
   These nodes execute LLM-generated code.  It runs in a child process with a
   hard timeout, which contains hangs and crashes — it is **not** a security
   sandbox.
"""

from typing import Literal, Optional

import pandas as pd

from core import Workflow, as_function_node, group_node
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
        "verbose",
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
    verbose: bool = False,
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
    verbose : bool
        Narrate each agent turn, tagged by arm and task.  A long task otherwise
        prints nothing for minutes, which is what let a run that never reached
        the agent at all look merely fast.

    Returns
    -------
    model, arms, max_repairs, exec_timeout_s, max_budget_usd, agent_timeout_s, effort, run_variant, run_optimize, save_chat, verbose
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
        verbose,
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
    reasons = sorted(
        {str(e).strip() for e in df.get("agent_error", []) if str(e).strip()}
    )
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
            # Both arms, so a score cannot depend on the operator's CLAUDE.md.
            bare=True,
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
        bare=True,
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
        bare=True,
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
        bare=True,
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
    run_tag: str = "",
    resume: bool = True,
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
    run_tag : str
        Identifier of the configuration this task belongs to, stamped into the
        result and checked before any result is reused.  Left empty, it is
        derived from the settings above — supply it only to keep a batch of
        tasks tagged identically.
    resume : bool
        Reuse a finished result already sitting in the task directory instead of
        paying for the agent again.  Switch it off to force a fresh measurement
        of the same configuration.

    Returns
    -------
    result : BenchOutcome
        A dataclass with every measured field.  ``IterToDataFrame`` expands it
        into one DataFrame column per field.
    """
    import datetime
    import json
    import shutil
    import time
    from pathlib import Path

    t0 = time.time()
    tag = run_tag or workflow_bench.run_config_tag(
        model=model,
        effort=effort,
        max_repairs=max_repairs,
        exec_timeout_s=exec_timeout_s,
        agent_timeout_s=agent_timeout_s,
        max_budget_usd=max_budget_usd,
        run_variant=run_variant,
        run_optimize=run_optimize,
    )
    scratch = Path(workdir).resolve() / arm / workflow_bench.task_dirname(task)
    scratch.mkdir(parents=True, exist_ok=True)

    def say(msg):
        if verbose:
            print(f"[{arm}/{slugify(task, 24)}] {msg}", flush=True)

    # A finished result is a measurement, not a cache of convenience: rerunning
    # it would cost an agent turn and return a *different* number, so a resumed
    # run that silently re-measured would quietly change the results it was
    # asked to extend.  The stored tag is what keeps this honest — a record left
    # by a different model or repair budget is not reused.
    if resume:
        cached = workflow_bench.load_outcome(scratch, task=task, arm=arm, run_tag=tag)
        if cached is not None:
            say(f"reusing finished result ({cached.final_stage}) from {scratch}")
            return cached

    # No reusable record, so whatever is in here is the wreckage of a run that
    # did not finish.  It has to go before the agent sees it: an agent that
    # finds a half-written artifact reads it as its own work, reports that the
    # file "already exists and is correct", and the harness grades the leftover
    # from a previous run instead of anything this one produced.
    stale = scratch / workflow_bench.ARM_ARTIFACT.get(arm, "workflow.py")
    if stale.is_file():
        say(f"discarding {stale.name} left by an unfinished run")
        stale.unlink()

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
        task_dir=str(scratch),
        run_tag=tag,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        exec_timeout_s=timeout_s,
    )

    def _finish(outcome):
        """Stamp the shared trailing fields, render the transcript, record it."""
        outcome.total_seconds = round(time.time() - t0, 1)
        # Written last, so one file covers the generation turn and every repair
        # it took; the results table links to it as the "reasoning" column.
        outcome.reasoning_path = workflow_bench.write_reasoning_digest(
            scratch, f"{arm}: {task}"
        )
        # The record a later run resumes from.  Harness failures write nothing,
        # so a run that never reached the agent is retried rather than frozen.
        workflow_bench.save_outcome(scratch, outcome)
        return outcome

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
    out.session_id = session_id
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
        say(f"no file produced ({out.blamed_on}: {out.last_error[:120]})")
        # The transcript matters most in exactly this case: there is no artifact
        # to read, so what the agent did instead is the only evidence there is.
        return _finish(out)

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
        out.session_id = session_id
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
    # ── everything below here edits `path` in place ──────────────────────────
    # Both the optimization turn and the follow-up variant rewrite the agent's
    # file.  Snapshot it once, now, and point the row at the snapshot: the
    # columns above describe *this* version, and linking them to a file that
    # two later agent turns rewrote is how a reader ends up reading the variant
    # and believing it is the graded solution.  Taking the copy inside each
    # branch, as this used to, also let the variant's copy overwrite the
    # optimizer's — so with both enabled no snapshot of the scored file
    # survived at all.
    def _snapshot(tag):
        dest = Path(path).with_name(f"{Path(path).stem}_{tag}.py")
        shutil.copy2(path, dest)
        return str(dest)

    variant, variant_expect = workflow_bench.variant_of(task)
    if stage == "complete" and (
        (run_optimize and arm == "aiflow") or (run_variant and variant)
    ):
        out.workflow_path = _snapshot("primary")

    # ── optional optimization turn: apply C1/C2 rules to a complete workflow ─
    if run_optimize and stage == "complete" and arm == "aiflow":
        out.optimize_attempted = True
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
            out.optimized_path = _snapshot("optimized")
        say(
            f"optimize → {out.optimize_stage} (diff {out.optimize_diff_loc} LOC, "
            f"C1: {out.c1_violations}→{out.c1_violations_after}, "
            f"C2: {out.c2_violations}→{out.c2_violations_after})"
        )
    # ── the follow-up edit: what does it cost to change your mind? ───────────
    if run_variant and variant and stage == "complete":
        out.variant = variant
        out.variant_attempted = True
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
        # The variant's own physical result.  Kept separate from the primary's:
        # a variant that was asked to change the material and returns the same
        # number has not done the edit, and only these two columns side by side
        # can show that.
        out.variant_measured_json = rechecker.outputs.measured_json.value
        out.variant_path = _snapshot("variant")
        say(f"variant → {out.variant_stage} ({out.variant_diff_loc} lines changed)")

    out.repair_seconds = round(out.repair_seconds, 1)
    out.cost_usd = round(out.cost_usd + out.variant_cost_usd, 4)
    return _finish(out)


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
    # One column per physical quantity, here rather than in the report, so the
    # numbers reach `results.csv` too — a result that only exists inside a JSON
    # string in a DataFrame cannot be analysed after the run.
    df = workflow_bench.expand_measured(df)
    warn_if_the_harness_never_ran(df)

    if workdir:
        target = Path(workdir).resolve()
        target.mkdir(parents=True, exist_ok=True)
        df.to_csv(target / "results.csv", index=False)
        print(f"results written to {target / 'results.csv'}")
    return df


@as_function_node(["summary", "stats", "tasks"])
def BenchmarkReport(df: pd.DataFrame = None, max_repairs: int = 3):
    """Aggregate the per-task results into the benchmark's headline metrics.

    Reports, overall and per tier: the pass rate at each tier of the validation
    ladder (first attempt and after repair), the success-vs-repair-budget curve
    ``P(executes | <= k repairs)``, the repair-cycle distribution, the share of
    graph nodes reused from ``pyiron_nodes`` rather than newly written, the
    token budget (split by fresh input, cache writes, cache reads and output),
    and the total wall-clock and cost — including cost per *completed* task,
    which is the only cost number that compares two arms fairly.

    Returns
    -------
    summary : str
        Human-readable report (also printed).
    stats : pd.DataFrame
        One row per group (``overall`` plus one per tier, and per arm when the
        frame holds both).
    tasks : pd.DataFrame
        One row per task, with the paths that make every number checkable: the
        artifact the agent wrote and its rendered reasoning.  Viewing this port
        in the GUI renders both as clickable links — the workflow opens as a
        canvas tab, the transcript in the file view.
    """
    max_repairs = workflow_bench.resolve_max_repairs(max_repairs, df)
    stats = workflow_bench.summarize(df, max_repairs=max_repairs)
    summary = workflow_bench.format_summary(stats, df)
    tasks = workflow_bench.task_table(df)
    print(summary)
    return summary, stats, tasks


@as_function_node(["summary", "stats", "tasks"])
def SingleBenchmarkReport(result=None, max_repairs: int = 3):
    """Report for one benchmark cell — same outputs as ``BenchmarkReport``.

    Wraps the single ``BenchOutcome`` from ``RunSingleBenchmark`` into a
    one-row DataFrame and runs the same summary, statistics table, and task
    table as ``BenchmarkReport``.  Output ports are identical so both nodes
    can be swapped in any downstream wiring.

    Parameters
    ----------
    result : BenchOutcome
        The outcome returned by ``RunSingleBenchmark``.
    max_repairs : int
        Repair budget used for the success-vs-repair-budget curve.  Should
        match the value given to ``RunSingleBenchmark``.

    Returns
    -------
    summary : str
        Human-readable report (also printed).
    stats : pd.DataFrame
        One row (``overall``) with the same columns as ``BenchmarkReport``.
    tasks : pd.DataFrame
        One row for this task with the artifact and reasoning paths.
    """
    from dataclasses import asdict

    if result is None:
        empty = pd.DataFrame()
        msg = "no result"
        print(msg)
        return msg, empty, empty

    row = asdict(result) if hasattr(result, "__dataclass_fields__") else dict(vars(result))
    df = pd.DataFrame([row])
    max_repairs = workflow_bench.resolve_max_repairs(max_repairs, df)
    stats = workflow_bench.summarize(df, max_repairs=max_repairs)
    summary = workflow_bench.format_summary(stats, df)
    tasks = workflow_bench.task_table(df)
    print(summary)
    return summary, stats, tasks


# ── Figure styling ──────────────────────────────────────────────────────────
#
# The palette is computed, not chosen.  Validated against the light chart
# surface with the dataviz validator:
#
#     node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" \
#         --mode light --pairs all            -> every check passes
#
# The slot *ordering* is the colour-blindness safety mechanism rather than a
# cosmetic choice, so substituting or re-ordering a hue means re-running the
# validator — never eyeballing whether two colours "look different enough".

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK_MUTED = "#898781"
INK_SECONDARY = "#52514e"

#: Identity of the two arms.  Keyed by arm name and never by position, so
#: dropping one arm from a run cannot repaint the other — a reader who learned
#: "aiflow is blue" keeps being right.
ARM_COLORS = {"aiflow": "#2a78d6", "scratch": "#eb6834"}

#: Who a failure is blamed on — identity, so three categorical slots.  Slot 3
#: measures 2.74:1 against the surface, under the 3:1 bar, so anything drawn in
#: it carries a direct value label instead of relying on the colour alone.
BLAME_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
BLAME_KINDS = ("agent", "library", "harness")

#: One hue light→dark for the ladder, because its stages are *ordered*: giving
#: them categorical hues would say they are unrelated categories.  Starts at the
#: lightest step that still clears 2:1 against the surface.
LADDER_RAMP = (
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#184f95",
)

#: Where a task finally stopped, in ladder order, so the ramp runs
#: failure → success and the darkest segment is the one that worked.
FINAL_STAGES = (
    "no_file",
    "syntax",
    "constructs",
    "executes",
    "plausible",
    "roundtrip",
    "complete",
)


def _new_figure(*args, **kwargs):
    """A Figure that pyplot does not hold a reference to.

    ``plt.subplots`` registers every figure in a global list that is only
    emptied by ``plt.close``, and a node that *returns* its figures can never
    close them.  Five per run then accumulate until matplotlib warns about a
    leak — which is exactly what it would be.  Constructing the Figure directly
    keeps it out of that registry; it renders and saves identically.
    """
    from matplotlib.figure import Figure

    return Figure(*args, **kwargs)


def _style(ax, title="", xlabel="", ylabel="", grid_axis="y"):
    """Recessive hairline chrome — solid, one shade off the surface, never dashed.

    Dashing a gridline makes it read as a threshold or a projection when it is
    only a grid, so the rule here is a solid hairline and no top/right spine.
    """
    if title:
        ax.set_title(title, fontsize=10, color=INK_SECONDARY)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK_SECONDARY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    return ax


def _unavailable(ax, why: str):
    """Say why a panel is empty rather than drawing an empty one.

    An axes with no marks reads as a measured zero, and an output port left at
    ``None`` renders as the literal word in the GUI and reads as a bug.  A
    sentence naming the missing ingredient is better than either.
    """
    ax.text(
        0.5,
        0.5,
        why,
        ha="center",
        va="center",
        fontsize=9,
        color=INK_MUTED,
        transform=ax.transAxes,
        wrap=True,
    )
    ax.set_axis_off()
    return ax


def _arm_color(arm) -> str:
    return ARM_COLORS.get(str(arm), LADDER_RAMP[4])


def _arms_of(df) -> list:
    """The arms present, in a fixed order, so colours never shuffle."""
    if df is None or not len(df) or "arm" not in getattr(df, "columns", ()):
        return []
    present = set(df["arm"].dropna().astype(str))
    known = [a for a in ARM_COLORS if a in present]
    return known + sorted(present - set(known))


def _overall(stats):
    """The ``overall`` row, or ``None`` when there is nothing to summarise."""
    if stats is None or not len(stats) or "group" not in getattr(stats, "columns", ()):
        return None
    rows = stats[stats["group"] == "overall"]
    return rows.iloc[0] if len(rows) else None


def _label_bars(ax, bars, values, fmt="{:.0f}", horizontal=False, suffix=""):
    """Direct-label a bar series at its end, outside the mark.

    Selective by construction: this is used on the short series where the value
    *is* the message, never on a per-point basis across a dense chart.
    """
    import math

    for bar, value in zip(bars, values):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        text = fmt.format(value) + suffix
        if horizontal:
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {text}",
                va="center",
                ha="left",
                fontsize=8,
                color=INK_SECONDARY,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text,
                va="bottom",
                ha="center",
                fontsize=8,
                color=INK_SECONDARY,
            )


def _label_segment(ax, x, y, width, value, on_dark: bool, min_width=7.0):
    """Label a stacked segment only when the text fits inside it with padding.

    A label clipped by its own segment is worse than no label: the value stays
    in the legend and the table either way, but a cropped number is read as a
    different number.  The ink is chosen from the *segment's* darkness rather
    than from its value — the two are unrelated, and picking by value puts grey
    text on a dark fill whenever a small share lands late in the ramp.
    """
    if value < min_width:
        return
    ax.text(
        x + width / 2,
        y,
        f"{value:.0f}%",
        ha="center",
        va="center",
        fontsize=8,
        color=SURFACE if on_dark else INK_SECONDARY,
    )


def _headroom(ax, factor=1.22):
    """Leave room above the tallest bar so the legend never sits on the data."""
    top = max((p.get_height() for p in ax.patches), default=0)
    if top > 0:
        ax.set_ylim(0, top * factor)


@as_function_node(["figure", "failures", "cost", "reliability", "reuse"])
def PlotBenchmark(df: pd.DataFrame = None, max_repairs: int = 3):
    """Draw the benchmark as five figures, one per question a reader asks.

    The ports are separate figures rather than panels of one, because the GUI
    gives every output port its own tab: five tabs are readable where fifteen
    panels on one canvas are not.

    ``figure``
        The headline, unchanged in shape: ladder pass rates first-attempt vs.
        after repair, the success-vs-repair-budget curve — now with its 95 %
        Wilson interval — and the arm comparison when both arms are present.
    ``failures``
        Where tasks died on the ladder, and who is answerable: agent, library
        or harness.  A run that never reached the agent looks nothing like a
        run the agent failed, and this is the panel that shows the difference.
    ``cost``
        What it cost per task and per *solved* task, cost against wall clock,
        and how the repair budget was actually spent.  An arm that is cheap per
        task and solves nothing is not the cheap arm.
    ``reliability``
        Per-task pass rate with Wilson intervals across repetitions, hardest
        first — which separates a consistently hard task from a flaky one.
        Needs ``repeats > 1`` to say anything; it says so when it cannot.
    ``reuse``
        The arm × tier pass-rate grid, and how much of each solution was
        assembled from ``pyiron_nodes`` rather than written fresh — the claim
        aiflow exists to make.

    Every port always returns a Figure.  When the data cannot support a panel
    it carries a sentence naming what is missing, because an empty axes reads
    as a measured zero.

    Returns
    -------
    figure, failures, cost, reliability, reuse : matplotlib.figure.Figure
    """
    max_repairs = workflow_bench.resolve_max_repairs(max_repairs, df)
    stats = workflow_bench.summarize(df, max_repairs=max_repairs)
    overall = _overall(stats)
    arms = _arms_of(df)

    headline = _plot_headline(df, stats, overall, arms, max_repairs)
    failures = _plot_failures(df, stats, arms)
    cost = _plot_cost(df, stats, arms, max_repairs)
    reliability = _plot_reliability(df, arms)
    reuse = _plot_reuse(df, stats, arms)

    for fig in (headline, failures, cost, reliability, reuse):
        fig.patch.set_facecolor(SURFACE)
        for ax in fig.axes:
            ax.set_facecolor(SURFACE)
    return headline, failures, cost, reliability, reuse


def _plot_headline(df, stats, overall, arms, max_repairs):
    """Ladder, repair curve and arm comparison — the three-number summary.

    The panel count is load-bearing: two panels for one arm, three for two.
    """
    import numpy as np

    stages = list(workflow_bench.STAGES)
    arm_rows = (
        stats[stats["group"].str.startswith("arm=")]
        if stats is not None and len(stats) and "group" in stats.columns
        else stats.iloc[0:0] if stats is not None else None
    )
    n_panels = 3 if arm_rows is not None and len(arm_rows) > 1 else 2

    fig = _new_figure(figsize=(5.5 * n_panels, 4))
    axes = fig.subplots(1, n_panels)
    ax_l, ax_m = axes[0], axes[1]

    if overall is None:
        for ax in axes:
            _unavailable(ax, "no results yet — run the benchmark first")
        fig.tight_layout()
        return fig

    # Before → after on the same measure, so one hue in two shades rather than
    # two categorical hues, which would imply two unrelated things.
    x = np.arange(len(stages))
    first = [overall.get(f"{s}_first_pct", np.nan) for s in stages]
    final = [overall.get(f"{s}_final_pct", np.nan) for s in stages]
    ax_l.bar(x - 0.2, first, width=0.4, color=LADDER_RAMP[1], label="first attempt")
    ax_l.bar(
        x + 0.2,
        final,
        width=0.4,
        color=LADDER_RAMP[5],
        label=f"after <= {max_repairs} repairs",
    )
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(stages, rotation=20, ha="right")
    ax_l.set_ylim(0, 105)
    _style(ax_l, "Validation ladder", ylabel="tasks passing (%)")
    ax_l.legend(frameon=False, fontsize=8)

    # The interval is the point of repeating a run, so the curve carries it.
    _repair_curve_panel(ax_m, df, arms, max_repairs, overall)

    if n_panels == 3:
        ax_r = axes[2]
        width = 0.8 / len(arm_rows)
        for i, (_, row) in enumerate(arm_rows.iterrows()):
            arm = str(row["group"]).split("=", 1)[1]
            ax_r.bar(
                x + (i - (len(arm_rows) - 1) / 2) * width,
                [row.get(f"{s}_final_pct", np.nan) for s in stages],
                width=width,
                color=_arm_color(arm),
                label=f"{arm} (n={int(row['n_tasks'])})",
            )
        ax_r.set_xticks(x)
        ax_r.set_xticklabels(stages, rotation=20, ha="right")
        ax_r.set_ylim(0, 105)
        _style(ax_r, f"aiflow vs. from scratch (<= {max_repairs} repairs)",
               ylabel="tasks passing (%)")
        ax_r.legend(frameon=False, fontsize=8)

    n = int(overall["n_tasks"])
    models = (
        ", ".join(sorted(set(df["model"]))) if df is not None and "model" in df else ""
    )
    fig.suptitle(f"Agentic workflow generation — {n} runs, {models}", fontsize=10,
                 color=INK_SECONDARY)
    fig.tight_layout()
    return fig


def _repair_curve_panel(ax, df, arms, max_repairs, overall):
    """P(executes | <= k repairs), with the interval that says whether it means anything.

    Drawn from :func:`pyiron_ai.bench_stats.repair_curve`, which already returns
    the Wilson bounds — recomputing the rate here would be a second opinion on a
    number the harness already has.
    """
    import numpy as np

    from pyiron_ai.bench_stats import repair_curve

    ks = list(range(max_repairs + 1))
    series = [(a, repair_curve(df, arm=a, max_k=max_repairs)) for a in arms] or [
        (None, repair_curve(df, max_k=max_repairs))
    ]
    drew = False
    for arm, curve in series:
        if curve is None or not len(curve):
            continue
        drew = True
        color = _arm_color(arm) if arm else LADDER_RAMP[4]
        rate = curve["rate"] * 100
        ax.plot(curve["k"], rate, marker="o", markersize=5, linewidth=2,
                color=color, label=str(arm) if arm else None,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.fill_between(curve["k"], curve["lo95"] * 100, curve["hi95"] * 100,
                        color=color, alpha=0.15, linewidth=0)

    if not drew:
        # The frame predates the columns the curve needs; fall back to the
        # point estimates rather than leaving the panel blank.
        ax.plot(ks, [overall.get(f"executes_le_{k}_repairs_pct", np.nan) for k in ks],
                marker="o", color=LADDER_RAMP[4], linewidth=2)

    ax.set_xticks(ks)
    ax.set_ylim(0, 105)
    _style(ax, "Success vs. repair budget", xlabel="repair cycles allowed",
           ylabel="solutions that execute (%)")
    if len(arms) > 1:
        ax.legend(frameon=False, fontsize=8)


def _plot_failures(df, stats, arms):
    """Where tasks died, and who is answerable for it.

    The second panel is the one that keeps a broken afternoon from being
    reported as a capability result: a pass rate that silently absorbs harness
    failures is misleading in whichever direction the breakage happened to fall.
    """
    import numpy as np

    fig = _new_figure(figsize=(13, 4.2))
    ax_l, ax_r = fig.subplots(1, 2)

    if df is None or not len(df) or "final_stage" not in getattr(df, "columns", ()):
        _unavailable(ax_l, "no results yet — run the benchmark first")
        _unavailable(ax_r, "no results yet — run the benchmark first")
        fig.tight_layout()
        return fig

    rows = arms or ["all tasks"]
    y = np.arange(len(rows))

    # Part-to-whole across an *ordered* set of outcomes, so one hue light→dark
    # and the darkest segment is the one that finished.
    for i, arm in enumerate(rows):
        sub = df if arm == "all tasks" else df[df["arm"].astype(str) == arm]
        total = len(sub)
        if not total:
            continue
        left = 0.0
        for step, (stage, color) in enumerate(zip(FINAL_STAGES, LADDER_RAMP)):
            share = 100.0 * (sub["final_stage"] == stage).sum() / total
            if share <= 0:
                continue
            ax_l.barh(
                i,
                share,
                left=left,
                height=0.42,
                color=color,
                # A 2px surface-coloured gap, not an outline drawn to separate.
                edgecolor=SURFACE,
                linewidth=2,
                label=stage if i == 0 else None,
            )
            _label_segment(ax_l, left, i, share, share, on_dark=step >= 3)
            left += share

    ax_l.set_yticks(y)
    ax_l.set_yticklabels(rows)
    ax_l.set_xlim(0, 100)
    ax_l.set_ylim(-0.6, len(rows) - 0.4)
    _style(ax_l, "Where tasks stopped on the ladder", xlabel="tasks (%)",
           grid_axis="x")
    handles, labels = ax_l.get_legend_handles_labels()
    if handles:
        ax_l.legend(handles, labels, frameon=False, fontsize=7, ncol=4,
                    loc="upper center", bbox_to_anchor=(0.5, -0.22))

    # Who is to blame — identity, so categorical slots, every bar labelled
    # because slot 3 sits under the 3:1 contrast bar.
    if "blamed_on" not in df.columns:
        _unavailable(ax_r, "this run recorded no blame attribution")
    else:
        width = 0.8 / max(len(BLAME_KINDS), 1)
        x = np.arange(len(rows))
        for j, (kind, color) in enumerate(zip(BLAME_KINDS, BLAME_COLORS)):
            counts = [
                int(
                    (
                        (df if a == "all tasks" else df[df["arm"].astype(str) == a])[
                            "blamed_on"
                        ]
                        == kind
                    ).sum()
                )
                for a in rows
            ]
            bars = ax_r.bar(
                x + (j - (len(BLAME_KINDS) - 1) / 2) * width,
                counts,
                width=width,
                color=color,
                label=kind,
            )
            _label_bars(ax_r, bars, counts)
        ax_r.set_xticks(x)
        ax_r.set_xticklabels(rows)
        _style(ax_r, "Failures by who is answerable", ylabel="tasks")
        _headroom(ax_r)
        ax_r.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)

        notes = []
        if "timed_out" in df.columns:
            notes.append(f"{int(df['timed_out'].sum())} timed out")
        if "hit_repair_cap" in df.columns:
            notes.append(f"{int(df['hit_repair_cap'].sum())} exhausted the repair budget")
        if notes:
            ax_r.text(0.5, -0.22, " · ".join(notes), transform=ax_r.transAxes,
                      ha="center", fontsize=8, color=INK_MUTED)

    fig.tight_layout()
    return fig


def _plot_cost(df, stats, arms, max_repairs):
    """What the run cost, and what a usable answer cost.

    Cost per task and cost per *solved* task share one axis because they share
    one unit; putting a rate on a second y-axis beside them would invent a
    correlation that is not in the data.
    """
    import numpy as np

    fig = _new_figure(figsize=(16, 4.2))
    ax_l, ax_m, ax_r = fig.subplots(1, 3)

    if df is None or not len(df) or "cost_usd" not in getattr(df, "columns", ()):
        for ax in (ax_l, ax_m, ax_r):
            _unavailable(ax, "this run recorded no cost")
        fig.tight_layout()
        return fig

    rows = arms or ["all tasks"]

    # Panel 1 — mean cost per task against cost per solved task.
    x = np.arange(len(rows))
    per_task, per_solved = [], []
    for arm in rows:
        sub = df if arm == "all tasks" else df[df["arm"].astype(str) == arm]
        per_task.append(float(sub["cost_usd"].mean()) if len(sub) else float("nan"))
        solved = int((sub["final_stage"] == "complete").sum()) if "final_stage" in sub else 0
        per_solved.append(
            float(sub["cost_usd"].sum()) / solved if solved else float("nan")
        )
    b1 = ax_l.bar(x - 0.2, per_task, width=0.4, color=LADDER_RAMP[1],
                  label="per task attempted")
    b2 = ax_l.bar(x + 0.2, per_solved, width=0.4, color=LADDER_RAMP[5],
                  label="per task solved")
    _label_bars(ax_l, b1, per_task, fmt="${:.2f}")
    _label_bars(ax_l, b2, per_solved, fmt="${:.2f}")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(rows)
    _style(ax_l, "Cost of an attempt vs. cost of an answer", ylabel="USD")
    _headroom(ax_l, 1.45)
    ax_l.legend(frameon=False, fontsize=8, loc="upper left")
    if any(np.isnan(per_solved)):
        ax_l.text(0.5, -0.2, "a missing bar means the arm solved nothing",
                  transform=ax_l.transAxes, ha="center", fontsize=8, color=INK_MUTED)

    # Panel 2 — one point per task; two series at most, so colour is safe here.
    if "total_seconds" not in df.columns:
        _unavailable(ax_m, "this run recorded no wall-clock time")
    else:
        for arm in rows:
            sub = df if arm == "all tasks" else df[df["arm"].astype(str) == arm]
            ax_m.scatter(sub["total_seconds"], sub["cost_usd"], s=60,
                         color=_arm_color(arm), label=str(arm),
                         edgecolor=SURFACE, linewidth=2, zorder=3)
        _style(ax_m, "Cost against wall clock, per task",
               xlabel="seconds", ylabel="USD")
        if len(rows) > 1:
            ax_m.legend(frameon=False, fontsize=8)

    # Panel 3 — emphasis, not two categorical hues: solved is the point and
    # unsolved is the context it has to be read against.
    if "repair_cycles" not in df.columns:
        _unavailable(ax_r, "this run recorded no repair cycles")
    else:
        ks = list(range(int(max_repairs) + 1))
        solved_mask = (
            df["final_stage"] == "complete"
            if "final_stage" in df.columns
            else df.get("executes_ok", pd.Series(False, index=df.index))
        )
        solved = [int(((df["repair_cycles"] == k) & solved_mask).sum()) for k in ks]
        unsolved = [int(((df["repair_cycles"] == k) & ~solved_mask).sum()) for k in ks]
        ax_r.bar(ks, solved, width=0.6, color=LADDER_RAMP[4], label="solved",
                 edgecolor=SURFACE, linewidth=2)
        ax_r.bar(ks, unsolved, width=0.6, bottom=solved, color=GRID,
                 label="unsolved", edgecolor=SURFACE, linewidth=2)
        ax_r.set_xticks(ks)
        _style(ax_r, "How the repair budget was spent",
               xlabel="repair cycles used", ylabel="tasks")
        top = max((s + u for s, u in zip(solved, unsolved)), default=0)
        if top:
            ax_r.set_ylim(0, top * 1.25)
        ax_r.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)

    fig.tight_layout()
    return fig


def _plot_reliability(df, arms):
    """Per-task pass rate with the interval that says whether to believe it.

    Sorted hardest first, because the useful question is which tasks to look at
    next, and a task that fails every repetition is a different problem from one
    that fails half of them.
    """
    import numpy as np

    from pyiron_ai.bench_stats import wilson_ci

    fig = _new_figure(figsize=(11, 5.5))
    ax = fig.subplots()

    if df is None or not len(df) or "task" not in getattr(df, "columns", ()):
        _unavailable(ax, "no results yet — run the benchmark first")
        fig.tight_layout()
        return fig
    if "final_stage" not in df.columns:
        _unavailable(ax, "this run recorded no per-task outcome")
        fig.tight_layout()
        return fig

    rows = arms or ["all tasks"]
    cells = {}
    for arm in rows:
        sub = df if arm == "all tasks" else df[df["arm"].astype(str) == arm]
        for task, grp in sub.groupby("task"):
            n = len(grp)
            k = int((grp["final_stage"] == "complete").sum())
            lo, hi = wilson_ci(k, n)
            cells[(str(task), arm)] = (k / n if n else float("nan"), lo, hi, n)

    if not cells:
        _unavailable(ax, "no tasks to report")
        fig.tight_layout()
        return fig

    tasks = sorted(
        {t for t, _ in cells},
        key=lambda t: np.nanmean([cells[(t, a)][0] for a in rows if (t, a) in cells]),
    )
    capped = len(tasks) > 25
    tasks = tasks[:25]

    y = np.arange(len(tasks))
    height = 0.8 / len(rows)
    for i, arm in enumerate(rows):
        offset = (i - (len(rows) - 1) / 2) * height
        rates = [100 * cells[(t, arm)][0] if (t, arm) in cells else np.nan for t in tasks]
        lo = [100 * (cells[(t, arm)][0] - cells[(t, arm)][1]) if (t, arm) in cells else 0
              for t in tasks]
        hi = [100 * (cells[(t, arm)][2] - cells[(t, arm)][0]) if (t, arm) in cells else 0
              for t in tasks]
        ax.barh(y + offset, rates, height=height, color=_arm_color(arm),
                label=str(arm), edgecolor=SURFACE, linewidth=2)
        ax.errorbar(rates, y + offset, xerr=[lo, hi], fmt="none",
                    ecolor=INK_SECONDARY, elinewidth=1, capsize=3, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([t[:52] + ("…" if len(t) > 52 else "") for t in tasks],
                       fontsize=8)
    ax.set_xlim(0, 105)
    _style(ax, "Pass rate per task, hardest first", xlabel="repetitions reaching complete (%)",
           grid_axis="x")
    if len(rows) > 1:
        # Outside the axes: with one row per task there is no empty corner left
        # for it to sit in without covering a bar.
        ax.legend(frameon=False, fontsize=8, loc="lower right",
                  bbox_to_anchor=(1.0, 1.01), ncol=len(rows))

    reps = max(n for *_, n in cells.values())
    note = (
        f"{reps} repetition(s) per task — the bars are a single run, so the "
        "intervals span almost everything; set repeats > 1 to narrow them"
        if reps < 2
        else f"95 % Wilson intervals over {reps} repetitions"
    )
    if capped:
        note += "  ·  showing the 25 hardest tasks"
    ax.text(0.5, -0.13, note, transform=ax.transAxes, ha="center", fontsize=8,
            color=INK_MUTED)
    fig.tight_layout()
    return fig


def _plot_reuse(df, stats, arms):
    """The arm × tier grid, and how much of each answer was already in the library.

    Node reuse is the claim aiflow exists to make, so it is reported next to the
    grid that says whether the reuse bought a higher pass rate.
    """
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    fig = _new_figure(figsize=(13, 4.4))
    ax_l, ax_r = fig.subplots(1, 2)

    if df is None or not len(df):
        _unavailable(ax_l, "no results yet — run the benchmark first")
        _unavailable(ax_r, "no results yet — run the benchmark first")
        fig.tight_layout()
        return fig

    rows = arms or ["all tasks"]
    tiers = sorted(df["tier"].dropna().astype(str).unique()) if "tier" in df else []

    # Magnitude on a grid: one hue light→dark, with the value in every cell so
    # the chart carries its own table and needs no colour bar.
    if not tiers or "final_stage" not in df.columns:
        _unavailable(ax_l, "this run recorded no tiers to compare")
    else:
        cmap = LinearSegmentedColormap.from_list("aiflow_blue", LADDER_RAMP)
        grid = np.full((len(rows), len(tiers)), np.nan)
        for i, arm in enumerate(rows):
            sub = df if arm == "all tasks" else df[df["arm"].astype(str) == arm]
            for j, tier in enumerate(tiers):
                cell = sub[sub["tier"].astype(str) == tier]
                if len(cell):
                    grid[i, j] = 100.0 * (cell["final_stage"] == "complete").mean()
        # pcolormesh rather than imshow: it takes a 2px surface gap between
        # cells, which keeps a small grid from reading as two solid slabs.
        ax_l.pcolormesh(
            np.arange(len(tiers) + 1),
            np.arange(len(rows) + 1),
            grid,
            cmap=cmap,
            vmin=0,
            vmax=100,
            edgecolors=SURFACE,
            linewidth=2,
        )
        for i in range(len(rows)):
            for j in range(len(tiers)):
                if np.isnan(grid[i, j]):
                    continue
                ax_l.text(j + 0.5, i + 0.5, f"{grid[i, j]:.0f}%", ha="center",
                          va="center", fontsize=9,
                          color=SURFACE if grid[i, j] > 55 else INK_SECONDARY)
        ax_l.set_xticks(np.arange(len(tiers)) + 0.5)
        ax_l.set_xticklabels(tiers, rotation=20, ha="right")
        ax_l.set_yticks(np.arange(len(rows)) + 0.5)
        ax_l.set_yticklabels(rows)
        # Row 0 on top, matching the panel beside it.
        ax_l.set_ylim(len(rows), 0)
        _style(ax_l, "Tasks reaching complete, by arm and tier", grid_axis=None)

    if {"n_reused_nodes", "n_new_nodes"} - set(df.columns):
        _unavailable(ax_r, "this run recorded no node provenance")
    else:
        y = np.arange(len(rows))
        for i, arm in enumerate(rows):
            sub = df if arm == "all tasks" else df[df["arm"].astype(str) == arm]
            reused = float(sub["n_reused_nodes"].sum())
            new = float(sub["n_new_nodes"].sum())
            total = reused + new
            if not total:
                continue
            pct = 100.0 * reused / total
            ax_r.barh(i, pct, height=0.42, color=LADDER_RAMP[4],
                      edgecolor=SURFACE, linewidth=2,
                      label="reused from pyiron_nodes" if i == 0 else None)
            ax_r.barh(i, 100 - pct, left=pct, height=0.42, color=GRID,
                      edgecolor=SURFACE, linewidth=2,
                      label="written fresh" if i == 0 else None)
            ax_r.text(101, i, f"{pct:.0f}%  ({int(reused)}/{int(total)})",
                      va="center", fontsize=8, color=INK_SECONDARY)
        ax_r.set_yticks(y)
        ax_r.set_yticklabels(rows)
        ax_r.set_xlim(0, 100)
        # Top-down, matching the heatmap beside it: the same two arms listed in
        # opposite orders on one figure is a misreading waiting to happen.
        ax_r.set_ylim(len(rows) - 0.4, -0.6)
        _style(ax_r, "Where the graph nodes came from", xlabel="nodes (%)",
               grid_axis="x")
        handles, labels = ax_r.get_legend_handles_labels()
        if handles:
            ax_r.legend(handles, labels, frameon=False, fontsize=8,
                        loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)

    fig.tight_layout()
    return fig


# ── Turning a finished run into something a person reads ────────────────────

#: Marker the summary agent places in its text, and the node replaces with the
#: rendered image.  The indirection is the point: the model decides *where* each
#: figure belongs in the narrative without ever handling a base64 blob.
FIGURE_MARKER = "{{figure:%s}}"

#: One line per figure, told to the agent so it can choose placements sensibly.
#: Keys are also the port names of :func:`PlotBenchmark`, in the order an
#: unplaced figure is appended.
FIGURE_CAPTIONS = {
    "figure": "the headline — ladder pass rates, the repair-budget curve and "
              "the arm comparison",
    "failures": "where tasks stopped on the ladder, and who each failure is "
                "blamed on",
    "cost": "cost per attempt and per solved task, cost against wall clock, "
            "and how the repair budget was spent",
    "reliability": "per-task pass rate with 95 % Wilson intervals, hardest "
                   "task first",
    "reuse": "pass rate by arm and tier, and how many graph nodes were reused "
             "from the library rather than written fresh",
}


def _figure_png(fig, dpi: int = 100) -> bytes:
    """Render a Figure to PNG bytes without disturbing it.

    ``bbox_inches="tight"`` matches what the GUI does when it captures a Figure
    port, so what lands in the report is what the canvas would have shown.
    """
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    return buf.getvalue()


def _note(title: str, *lines: str) -> str:
    """A markdown note explaining why there is no report.

    Starts with a ``# `` heading because the GUI decides whether a string is
    markdown by looking at its first characters; without one this renders as
    preformatted text with its quotes showing.
    """
    return "\n\n".join((f"# {title}", *lines)) + "\n"


@as_function_node(["report", "path"])
def SummarizeBenchmark(
    summary: str = "",
    stats: pd.DataFrame = None,
    df: pd.DataFrame = None,
    figure=None,
    failures=None,
    cost=None,
    reliability=None,
    reuse=None,
    run_dir: str = "",
    model: Optional[Literal["sonnet", "opus", "haiku"]] = "sonnet",
    effort: str = "",
    max_words: int = 700,
    max_budget_usd: float = 1.0,
    agent_timeout_s: int = 300,
    verbose: bool = True,
):
    """Ask Claude to turn a finished benchmark into a report, figures included.

    Wire ``summary``/``stats`` from :func:`BenchmarkReport`, ``df`` and
    ``run_dir`` from :func:`RunBenchmarkSuite`, and the five figure ports
    straight across from :func:`PlotBenchmark`.

    The agent is given **numbers only** — the formatted summary, the per-group
    statistics and the provenance record.  It never sees the images; it places
    them with markers that this node substitutes afterwards.  That keeps the
    turn cheap and its output reproducible, and it means a figure can never be
    described wrongly because the model misread a chart.

    This node spends money every time it runs, which is why no canvas wires it
    by default.  It refuses to spend any when there is nothing to summarise or
    no CLI to ask — a report invented over an empty frame is worse than none.

    Parameters
    ----------
    max_words : int
        Length budget handed to the model.  A benchmark summary that runs long
        stops being read.
    max_budget_usd, agent_timeout_s : float, int
        Deliberately far below the generation defaults: this is one short
        writing turn, not an agentic loop.

    Returns
    -------
    report : str
        Markdown, with each figure base64-inlined so one Output tab shows the
        prose and the charts together.
    path : str
        The self-contained ``summary.html`` written next to the results, or an
        empty string when no ``run_dir`` was given.
    """
    import base64
    import tempfile
    from pathlib import Path

    figures = {
        name: fig
        for name, fig in (
            ("figure", figure),
            ("failures", failures),
            ("cost", cost),
            ("reliability", reliability),
            ("reuse", reuse),
        )
        if fig is not None
    }

    def say(msg):
        if verbose:
            print(f"[summary] {msg}", flush=True)

    have_rows = df is not None and len(df)
    have_stats = stats is not None and len(stats)
    if not have_rows and not have_stats and not str(summary).strip():
        return _note(
            "Nothing to summarise",
            "No results were wired in, so there is nothing for the agent to "
            "read and no reason to pay for a turn.",
            "Wire `summary` and `stats` from `BenchmarkReport`, and `df` from "
            "`RunBenchmarkSuite`.",
        ), ""

    claude = workflow_bench.claude_bin()
    if claude is None:
        return _note(
            "No `claude` CLI, so no summary",
            "The agent could not be reached, so nothing was written and nothing "
            "was spent. Reporting a benchmark without the write-up is better "
            "than reporting a write-up nobody produced.",
            f"Looked on PATH, in ${workflow_bench.CLAUDE_BIN_ENV}, and in "
            f"{', '.join(workflow_bench.CLAUDE_BIN_GLOBS)}.",
        ), ""

    with tempfile.TemporaryDirectory(prefix="bench_summary_") as tmp:
        work = Path(tmp)
        text = str(summary).strip() or workflow_bench.format_summary(stats, df)
        (work / "summary.txt").write_text(text)
        (work / "stats.md").write_text(
            stats.to_markdown(index=False) if have_stats else "no per-group statistics"
        )
        (work / "provenance.txt").write_text(
            "\n".join(workflow_bench.format_provenance(df)) if have_rows else "unrecorded"
        )

        if figures:
            listing = "\n".join(
                f"- `{FIGURE_MARKER % name}` — {FIGURE_CAPTIONS[name]}"
                for name in figures
            )
            figure_rule = workflow_bench.SUMMARY_FIGURE_RULE.format(figures=listing)
        else:
            figure_rule = workflow_bench.SUMMARY_NO_FIGURE_RULE

        prompt = workflow_bench.SUMMARY_PROMPT.format(
            stages=" → ".join(workflow_bench.STAGES),
            max_words=int(max_words),
            figure_rule=figure_rule,
        )

        say(f"asking {model} for a write-up of {len(df) if have_rows else 0} row(s)")
        # The bare name, as every other agent-calling node here uses: the
        # documented way to stub this out is to patch `run_claude_code` on *this
        # module*, and a qualified `workflow_bench.run_claude_code` would sail
        # straight past that stub and spend real money in a test.
        run = run_claude_code(
            prompt,
            workdir=str(work),
            model=model,
            max_budget_usd=max_budget_usd,
            timeout_s=agent_timeout_s,
            effort=effort or None,
            env=agent_env(),
            bare=True,
        )

        written = work / "SUMMARY.md"
        if not written.is_file():
            return _note(
                "The agent wrote no summary",
                f"`{model}` was asked for `SUMMARY.md` and produced nothing.",
                f"Reported error: {run.error or 'none'}",
                f"Its reply was: {(run.text or '').strip()[:400] or '(empty)'}",
            ), ""
        body = written.read_text()

    say(f"{len(body.split())} words, ${run.cost_usd:.3f}, {run.seconds:.0f}s")

    # The same text composed twice against two sets of images: base64 for the
    # panel, which cannot load a file path, and relative links for the copy on
    # disk, which stays diffable and legible that way.
    inline = {
        name: f"![{FIGURE_CAPTIONS[name]}]"
              f"(data:image/png;base64,{base64.b64encode(_figure_png(fig)).decode()})"
        for name, fig in figures.items()
    }
    linked = {
        name: f"![{FIGURE_CAPTIONS[name]}](figures/{name}.png)" for name in figures
    }
    report = _compose_report(body, inline)
    path = _write_summary_files(run_dir, report, _compose_report(body, linked),
                                figures, say)
    return report, path


def _compose_report(raw: str, images: dict) -> str:
    """Substitute the markers the model placed, and append the ones it did not.

    A figure that was computed and then silently dropped is one the reader is
    entitled to assume does not exist, so an unplaced figure goes to the end
    rather than nowhere.
    """
    out, placed = raw, set()
    for name, image in images.items():
        marker = FIGURE_MARKER % name
        if marker in out:
            out = out.replace(marker, image)
            placed.add(name)

    missing = [n for n in images if n not in placed]
    if missing:
        out += "\n\n## Figures\n\n" + "\n\n".join(
            f"**{n}** — {FIGURE_CAPTIONS[n]}\n\n{images[n]}" for n in missing
        )
    # The GUI decides whether a string is markdown from its first characters.
    if not out.lstrip().startswith("#"):
        out = f"# Benchmark summary\n\n{out}"
    return out


def _write_summary_files(run_dir, body, markdown_body, figures, say) -> str:
    """Write the portable copies beside the results; never fail the node for it.

    Two forms, because they are read in different places: ``summary.html`` is
    self-contained and opens anywhere, while ``summary.md`` with sibling PNGs is
    what a repository or a paper draft wants.
    """
    import html as html_mod
    from pathlib import Path

    if not str(run_dir).strip():
        return ""
    try:
        root = Path(run_dir).resolve()
        if figures:
            figure_dir = root / "figures"
            figure_dir.mkdir(parents=True, exist_ok=True)
            for name, fig in figures.items():
                (figure_dir / f"{name}.png").write_bytes(_figure_png(fig, dpi=150))
        root.mkdir(parents=True, exist_ok=True)
        (root / "summary.md").write_text(markdown_body)

        try:
            import markdown as markdown_mod

            rendered = markdown_mod.markdown(body, extensions=["tables", "fenced_code"])
        except Exception:
            # No converter available: the text is still readable, which matters
            # more than it being pretty.
            rendered = f"<pre>{html_mod.escape(body)}</pre>"
        (root / "summary.html").write_text(
            "<!doctype html><meta charset='utf-8'>"
            "<title>Benchmark summary</title>"
            "<style>body{max-width:52rem;margin:3rem auto;padding:0 1rem;"
            "font:16px/1.6 system-ui,-apple-system,'Segoe UI',sans-serif;"
            "color:#0b0b0b;background:#fcfcfb}"
            "img{max-width:100%;height:auto;display:block;margin:1.5rem 0}"
            "table{border-collapse:collapse}td,th{border:1px solid #e1e0d9;"
            "padding:.3rem .6rem;text-align:left}"
            "code{background:#f0efec;padding:.1rem .3rem;border-radius:3px}"
            "</style>\n" + rendered
        )
        say(f"wrote {root / 'summary.html'}")
        return str(root / "summary.html")
    except OSError as exc:
        # The report itself is already in hand; losing the copy on disk must not
        # cost the caller the thing they asked for.
        say(f"could not write the summary files: {exc}")
        return ""


# ── The whole suite in one node ─────────────────────────────────────────────


@as_function_node(["df", "stats", "summary", "run_dir", "repair_budget"])
def _BenchmarkSuiteCore(
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
    resume: bool = True,
    task_executor=None,
):
    """Implementation node — use ``RunBenchmarkSuite`` on the canvas.

    This private node holds the loop and I/O logic that cannot be expressed as a
    static DAG.  ``RunBenchmarkSuite`` wraps it in a ``@group_node`` so it is
    expandable on the canvas.

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
        Independent repetitions.  More than one adds per-cell Wilson intervals.
    max_workers : int
        Cells launched in parallel.  Bounds the whole run, not one repetition:
        every ``(task, arm, repetition)`` is submitted at once.
    task_executor : concurrent.futures.Executor, optional
        A ready-made executor.  ``None`` (the default) uses a
        ``ThreadPoolExecutor(max_workers=max_workers)`` in-process.  Wire a
        ``SlurmExecutor`` node here (via the ``RunBenchmarkSuite`` group's
        ``task_executor`` port) for cluster execution.  The node that created
        the executor owns it; this node will not shut it down.
    workdir : str
        Parent directory.  The run goes into ``<workdir>/<model>_<hash>/rep_NNN``.
    resume : bool
        Reuse tasks that already finished there instead of paying for them again.
    """
    import time
    from concurrent.futures import as_completed
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
    n_reps = max(1, int(repeats))
    workers = max(1, int(max_workers))

    # One directory per configuration under `workdir`, so a Sonnet run and an
    # Opus run of the same suite cannot land on top of each other — and, with
    # `resume`, cannot be mistaken for each other's finished work.
    tag = workflow_bench.run_config_tag(
        model=resolved_model,
        effort=effort_flag,
        max_repairs=max_repairs,
        exec_timeout_s=exec_timeout_s,
        agent_timeout_s=agent_timeout_s,
        max_budget_usd=max_budget_usd,
        run_variant=run_variant,
        run_optimize=run_optimize,
        allow_reference_workflows=allow_reference_workflows,
    )
    root = Path(workdir).resolve() / tag
    root.mkdir(parents=True, exist_ok=True)

    # Build on shared NFS (inside workdir), not node-local /tmp, so every
    # Slurm compute node that picks up a WorkflowAgent task can reach the
    # library files the agent needs to read and the NODE_INDEX to index.
    library = lib_dir or str(build_node_library_mirror(allow_reference_workflows, dest=root))

    print(
        f"running {len(task_list)} task(s) × {len(arm_list)} arm(s) × {n_reps} rep(s) "
        f"with model={resolved_model} into {root}\n"
        f"agent CLI: {claude}\n"
        f"node library visible to the agent: {library}\n"
        f"{'reusing' if resume else 'ignoring'} results already finished there\n",
        flush=True,
    )

    frames = []
    t0 = time.time()

    # Every repetition's directory and manifest exists before anything is
    # submitted, because the cells now run concurrently and two threads racing
    # to write one manifest would interleave.
    rep_dirs = {}
    for rep in range(1, n_reps + 1):
        # Always ``rep_NNN``, even for a single repetition.  A first run with
        # repeats=1 that wrote straight into the root could not be extended by a
        # later repeats=5: it would look for rep_001, find nothing, and pay for
        # all five instead of the four that are new.
        rep_dir = root / f"rep_{rep:03d}"
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
        rep_dirs[rep] = rep_dir

    from concurrent.futures import ThreadPoolExecutor as _ThreadPool

    if task_executor is None:
        pool = _ThreadPool(max_workers=workers)
        owned = True
    else:
        pool = task_executor
        owned = False

    def run_cell(rep, arm):
        """One arm of one repetition: a sweep over every task."""
        template = WorkflowAgent(
            model=resolved_model,
            max_repairs=max_repairs,
            workdir=str(rep_dirs[rep]),
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
            run_tag=tag,
            resume=resume,
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
        if part is None or not len(part):
            return rep, None
        # The directory name, which is what aggregate_repeats puts in this
        # column when it reads the run back — so the column means the same
        # thing whether the frame came from here or from re-reading the run.
        return rep, part.assign(rep=rep_dirs[rep].name)

    # The cells are launched together rather than one repetition at a time, so
    # `pool` sees every (task, arm, repetition) at once.  Running them in series
    # would cap a Slurm run at one repetition's worth of jobs in the queue and
    # leave the cluster idle between repetitions; `pool` is still what bounds
    # the real concurrency.  This outer pool only holds blocking calls, so it is
    # sized to the number of cells rather than to `max_workers`.
    cells = [(rep, arm) for rep in rep_dirs for arm in arm_list]
    per_rep = {rep: [] for rep in rep_dirs}
    outer = _ThreadPool(max_workers=max(1, len(cells)))
    try:
        futures = [outer.submit(run_cell, rep, arm) for rep, arm in cells]
        for future in as_completed(futures):
            rep, part = future.result()
            if part is not None:
                per_rep[rep].append(part)
    finally:
        outer.shutdown(wait=True)
        if owned:
            pool.shutdown(wait=True)

    for rep, rep_frames in sorted(per_rep.items()):
        if not rep_frames:
            continue
        rep_dir = rep_dirs[rep]
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
        # ``aggregate_repeats`` reads back; the combined file goes to the root.
        rep_df = workflow_bench.expand_measured(
            pd.concat(rep_frames, ignore_index=True)
        )
        rep_df.to_csv(rep_dir / "results.csv", index=False)
        frames.append(rep_df)

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

    repair_budget = int(max_repairs)
    return df, stats, summary, str(root), repair_budget


@as_function_node("result")
def RunSingleBenchmark(
    task: str = "",
    tier: Optional[
        Literal["generic", "pyiron_nodes", "atomistic", "atomistic_hard"]
    ] = "generic",
    task_index: int = 0,
    model: Optional[Literal["sonnet", "opus", "haiku"]] = "sonnet",
    model_other: str = "",
    arm: Optional[Literal["aiflow", "scratch"]] = "aiflow",
    rep: int = 1,
    workdir: str = "bench_runs",
    max_repairs: int = 3,
    exec_timeout_s: int = 300,
    max_budget_usd: float = AGENT_BUDGET_USD,
    agent_timeout_s: int = AGENT_TIMEOUT_S,
    effort: Optional[Literal["default", "low", "medium", "high", "max"]] = "default",
    allow_reference_workflows: bool = False,
    lib_dir: str = "",
    run_variant: bool = True,
    run_optimize: bool = False,
    save_chat: bool = True,
    verbose: bool = True,
    resume: bool = True,
    store: bool = False,
):
    """Run one (model, task, arm, rep) benchmark cell end-to-end.

    Sets up the standard directory structure
    (``<workdir>/<config_tag>/rep_NNN/<arm>/<task_slug>/``), builds the
    node-library mirror, writes the freeze manifest, and delegates to
    ``WorkflowAgent`` for the actual agent turn.  Use it to run or re-run a
    specific cell from a notebook without the full suite machinery.

    For a full sweep over tasks, arms, and repetitions use
    ``RunBenchmarkSuite`` instead.

    Parameters
    ----------
    task : str
        The plain-English task prompt.  If empty, ``tier`` and ``task_index``
        select a task from the curated suite.
    tier : str
        Which curated task tier to draw from when ``task`` is empty.
    task_index : int
        Zero-based index into the task list of ``tier`` (used only when
        ``task`` is empty).
    model : str
        Claude Code model alias (dropdown in the GUI).
    model_other : str
        Free-text model id; overrides ``model`` when non-empty — the GUI
        dropdown cannot list future or fine-tuned model ids.
    arm : str
        ``"aiflow"`` writes a ``Workflow``; ``"scratch"`` writes plain Python.
    rep : int
        1-based repetition index; the artifact lands under ``rep_{rep:03d}/``.
    workdir : str
        Root output directory.  The config-tag subdirectory and the rep
        directory are added automatically.
    max_repairs : int
        Repair budget; ``0`` measures raw first-attempt quality.
    exec_timeout_s : int
        Hard limit per validation run; hard tasks may raise this automatically.
    max_budget_usd : float
        Spend cap per agent turn.
    agent_timeout_s : int
        Wall-clock ceiling per agent turn.
    effort : str
        Reasoning effort forwarded to ``claude --effort``; ``"default"`` omits
        the flag.
    allow_reference_workflows : bool
        Expose ``pyiron_nodes/Workflows/`` to the agent.  Useful as a second
        experimental condition but breaks the aiflow-vs-scratch comparison.
    lib_dir : str
        Pre-built node-library directory; leave empty to build it automatically.
    run_variant : bool
        Run the declared follow-up edit after a task passes.
    run_optimize : bool
        Spend one extra turn applying the optimization guide (aiflow arm only).
    save_chat : bool
        Write per-turn chat logs alongside the artifact.
    verbose : bool
        Print per-attempt progress.
    resume : bool
        Reuse a finished result from the task directory rather than re-running.

    Returns
    -------
    result : BenchOutcome
        All measured fields for this cell.
    """
    from pathlib import Path

    resolved_model = model_other.strip() or model
    effort_flag = "" if effort in (None, "", "default") else effort

    # Resolve the task text when not supplied directly.
    actual_task = task.strip()
    if not actual_task:
        suite = get_tasks(tier=tier, limit=0)
        if not suite:
            raise ValueError(f"Task tier {tier!r} is empty.")
        if task_index < 0 or task_index >= len(suite):
            raise IndexError(
                f"task_index={task_index} out of range for tier {tier!r} "
                f"(0..{len(suite) - 1})."
            )
        actual_task = suite[task_index]

    # Build the standard directory path that _BenchmarkSuiteCore would produce.
    tag = workflow_bench.run_config_tag(
        model=resolved_model,
        effort=effort_flag,
        max_repairs=max_repairs,
        exec_timeout_s=exec_timeout_s,
        agent_timeout_s=agent_timeout_s,
        max_budget_usd=max_budget_usd,
        run_variant=run_variant,
        run_optimize=run_optimize,
        allow_reference_workflows=allow_reference_workflows,
    )
    root = Path(workdir).resolve() / tag
    rep_dir = root / f"rep_{max(1, int(rep)):03d}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    write_freeze_manifest(
        rep_dir,
        model=resolved_model,
        tier="custom" if task.strip() else tier,
        arms=[arm],
        max_repairs=max_repairs,
        exec_timeout=exec_timeout_s,
        effort=effort_flag or None,
    )

    library = lib_dir or str(
        build_node_library_mirror(allow_reference_workflows, dest=root)
    )

    agent = WorkflowAgent(
        task=actual_task,
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
        resume=resume,
    )
    agent.run()
    result = agent.outputs.result.value
    return result


@group_node("df", "stats", "summary", "run_dir", "repair_budget")
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
    resume: bool = True,
    task_executor=None,
):
    """Run the whole benchmark — every arm, every repetition — at once.

    Wire a ``SlurmExecutor`` node to ``task_executor`` for cluster execution;
    leave it unwired to run agents in-process with a thread pool.

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
        between the arms from noise.  Raising it later is cheap: existing
        repetitions are reused.
    max_workers : int
        Cells launched in parallel.  Bounds the whole run, not one repetition:
        every ``(task, arm, repetition)`` is submitted at once.
    task_executor : concurrent.futures.Executor, optional
        Wire a ``SlurmExecutor`` node here to submit each task as its own batch
        job, so a twenty-task × two-arm × five-repetition run finishes in the
        wall time of its slowest single task.  ``None`` (the default) uses a
        ``ThreadPoolExecutor(max_workers=max_workers)`` in-process.  The executor
        node owns the executor; this node will not shut it down.
    workdir : str
        Parent directory.  The run goes into
        ``<workdir>/<model>_<hash>/rep_NNN/<arm>/<task>`` so two configurations
        never share a directory, and a rerun of the *same* configuration reuses
        what it already finished instead of paying for it twice.
    resume : bool
        Reuse tasks that already finished in that directory instead of paying
        for them again.  This is what makes ``repeats`` extendable.

    Returns
    -------
    df, stats, summary, run_dir, repair_budget
        The per-task rows, the statistics table, the printable report, the
        directory holding ``results.csv`` and the freeze manifest, and the
        ``max_repairs`` that was actually used — wire this to
        ``BenchmarkReport.max_repairs`` so the report's repair-budget curve
        is drawn to the same scale without typing the number twice.
    """
    wf = Workflow("RunBenchmarkSuite")
    wf.suite = _BenchmarkSuiteCore(
        tier=tier, limit=limit, tasks=tasks, arms=arms,
        model=model, model_other=model_other, max_repairs=max_repairs,
        repeats=repeats, workdir=workdir, exec_timeout_s=exec_timeout_s,
        max_budget_usd=max_budget_usd, agent_timeout_s=agent_timeout_s,
        effort=effort, allow_reference_workflows=allow_reference_workflows,
        lib_dir=lib_dir, run_variant=run_variant, run_optimize=run_optimize,
        save_chat=save_chat, max_workers=max_workers, verbose=verbose,
        resume=resume, task_executor=task_executor,
    )
    return (
        wf.suite.outputs.df,
        wf.suite.outputs.stats,
        wf.suite.outputs.summary,
        wf.suite.outputs.run_dir,
        wf.suite.outputs.repair_budget,
    )


@as_function_node(["stats", "df"])
def AggregateRepeats(run_dir: str = "bench_runs"):
    """Rebuild the per-cell statistics from a finished multi-repetition run.

    Lets a run that was interrupted — or one aggregated with different
    assumptions — be re-reduced without paying for the agent again.

    Parameters
    ----------
    run_dir : str
        The directory holding the ``rep_NNN/`` subdirectories — that is
        ``RunBenchmarkSuite``'s ``run_dir`` output, which is one level *below*
        its ``workdir``.  Pointing this at the work dir instead finds no
        repetitions, because each configuration keeps its own subdirectory
        there and mixing two of them would average different measurements
        together.

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


# ── The whole sweep as one expandable node ──────────────────────────────────


@group_node("df", "repair_budget")
def BenchmarkSuite(
    tier: Optional[
        Literal["all", "generic", "pyiron_nodes", "atomistic", "atomistic_hard"]
    ] = "generic",
    limit: int = 0,
    allow_reference_workflows: bool = False,
    path: str = "bench_runs",
    max_workers: int = 3,
    model: Optional[Literal["sonnet", "opus", "haiku"]] = "sonnet",
    model_other: str = "",
    arms: Optional[Literal["aiflow", "scratch", "both"]] = "both",
    max_repairs: int = 3,
    exec_timeout_s: int = 300,
    max_budget_usd: float = 4.0,
    agent_timeout_s: int = 1800,
    effort: Optional[Literal["default", "low", "medium", "high", "max"]] = "default",
    run_variant: bool = True,
    run_optimize: bool = False,
    save_chat: bool = True,
    verbose: bool = False,
):
    """Both arms, every task, one node — with the graph still inside it.

    The same pipeline the canvas at the bottom of this module spells out by
    hand, collapsed to a single node whose every option is a boundary port.
    Unlike ``RunBenchmarkSuite``, which does the sweep inside one Python
    function, this is a real subgraph: expand it and the twelve nodes are
    there to inspect, rewire and re-run individually.  Use it when you want the
    canvas to show the experiment rather than the plumbing, and still be able
    to open the plumbing.

    Repetitions and their confidence intervals are not available here — one
    group is one repetition.  Use ``RunBenchmarkSuite(repeats=n)`` for error
    bars, or ``AggregateRepeats`` over a directory of finished runs.

    Parameters
    ----------
    tier, limit
        Which curated tasks to run — see ``TaskSuite``.
    allow_reference_workflows
        ``True`` points the aiflow arm at the real repository, reference
        solutions and all.  See ``NodeLibraryMirror``; leave it off unless you
        are deliberately measuring the retrieval-assisted case.
    path
        Output directory.  Use a dated one, or a rerun overwrites the last.
    max_workers
        Threads for the two sweeps.
    model, model_other, arms, max_repairs, exec_timeout_s, max_budget_usd, agent_timeout_s, effort, run_variant, run_optimize, save_chat, verbose
        Forwarded to ``BenchSettings``, which feeds both arms from one place so
        they cannot drift apart.  ``arms`` switches a whole branch off.

    Returns
    -------
    df : pandas.DataFrame
        One row per task per arm — wire into ``BenchmarkReport``.
    repair_budget : int
        The ``max_repairs`` that was actually used, so the report's
        repair-budget curve is drawn to the same scale without being told
        twice.  Named apart from the input port of the same meaning so the
        canvas cannot confuse the two sides of the node.
    """
    wf = Workflow("BenchmarkSuite")

    wf.settings = BenchSettings(
        model=model,
        model_other=model_other,
        arms=arms,
        max_repairs=max_repairs,
        exec_timeout_s=exec_timeout_s,
        max_budget_usd=max_budget_usd,
        agent_timeout_s=agent_timeout_s,
        effort=effort,
        run_variant=run_variant,
        run_optimize=run_optimize,
        save_chat=save_chat,
        verbose=verbose,
    )
    wf.mirror = NodeLibraryMirror(allow_reference_workflows=allow_reference_workflows)
    # Inner labels must differ from every label on the canvas outside the
    # group: expanding inlines them into the parent graph, and a collision
    # silently drops the inner node and re-points its edges at the group node
    # itself — which then feeds its own children, i.e. a cycle.  Hence
    # `task_suite` and not `suite`.
    wf.task_suite = TaskSuite(tier=tier, limit=limit)
    wf.dir = BenchWorkDir(path=path)
    wf.pool = ThreadPoolExecutorNode(max_workers=max_workers)

    wf.tasks_aiflow = ArmTasks(
        tasks=wf.task_suite, arm="aiflow", arms=wf.settings.outputs.arms
    )
    wf.tasks_scratch = ArmTasks(
        tasks=wf.task_suite, arm="scratch", arms=wf.settings.outputs.arms
    )

    wf.agent_aiflow = WorkflowAgent(
        model=wf.settings.outputs.model,
        max_repairs=wf.settings.outputs.max_repairs,
        workdir=wf.dir,
        exec_timeout_s=wf.settings.outputs.exec_timeout_s,
        max_budget_usd=wf.settings.outputs.max_budget_usd,
        agent_timeout_s=wf.settings.outputs.agent_timeout_s,
        arm="aiflow",
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
        workdir=wf.dir,
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

    wf.stack = StackResults(
        df_aiflow=wf.bench_aiflow, df_scratch=wf.bench_scratch, workdir=wf.dir
    )
    return wf.stack, wf.settings.outputs.max_repairs


__all__ = [
    # Entry points
    "RunBenchmarkSuite",
    "BenchmarkSuite",
    "AggregateRepeats",
    "BenchmarkReport",
    "PlotBenchmark",
    "SummarizeBenchmark",
    # Task sources and gating
    "TaskSuite",
    "TaskList",
    "ArmTasks",
    # Shared configuration
    "BenchSettings",
    "NodeLibraryMirror",
    "BenchWorkDir",
    # The agentic loop
    "WorkflowAgent",
    "GenerateWorkflow",
    "ValidateWorkflow",
    "RepairWorkflow",
    "VaryWorkflow",
    "OptimizeWorkflow",
    # Result assembly
    "StackResults",
    "warn_if_the_harness_never_ran",
]
