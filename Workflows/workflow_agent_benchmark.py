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

Open in PyironFlow
------------------
1. Pick a ``tier`` on the ``TaskSuite`` node (dropdown), or wire ``TaskList``
   instead and type your own tasks.
2. Pick the ``model`` on both ``WorkflowAgent`` nodes (dropdown) and the repair
   budget ``max_repairs``.
3. Leave ``allow_reference_workflows=False`` on ``NodeLibraryMirror`` unless you
   deliberately want to measure the retrieval-assisted case.
4. Set ``max_workers`` on the thread pool for parallel generation.
5. Run.  ``BenchmarkReport`` prints the statistics; ``PlotBenchmark`` draws the
   ladder pass-rates, the success-vs-repair-budget curve and the arm comparison.

Generated code is written to ``<workdir>/<arm>/<task-slug>/{workflow,solution}.py``
and kept for inspection.  When a follow-up variant runs it edits that same file,
so the graded version is first snapshotted alongside it as ``*_primary.py``.

.. warning::
   This executes LLM-generated code.  It runs in a child process with a hard
   timeout, which contains hangs and crashes — it is **not** a security sandbox.
"""

from typing import Literal, Optional

import pandas as pd

from core import Workflow, as_function_node
from pyiron_nodes.controls import IterToDataFrame
from pyiron_nodes.executors import ThreadPoolExecutorNode

from pyiron_ai.workflow_bench import (
    AGENT_BUDGET_USD,
    AGENT_TIMEOUT_S,
    ARM_ARTIFACT,
    PYIRON_NODES_ROOT,
    BenchOutcome,
    build_node_library_mirror,
    expect_of,
    generation_prompt,
    get_tasks,
    repair_prompt,
    run_claude_code,
    slugify,
    spec_of,
    tier_of,
    validate_workflow_file,
    write_node_index,
)
from pyiron_ai import workflow_bench

# ── Task sources ────────────────────────────────────────────────────────────


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


# ── Higher-order agent node: the loop lives here, the outer graph stays a DAG ─


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
    verbose: bool = True,
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

    if not Path(path).is_file():
        out.stage_first = out.final_stage = "no_file"
        out.first_error = out.last_error = (
            out.agent_error or f"the agent produced no {Path(path).name}"
        )
        # A wall-clock timeout says nothing about the framework under test.
        out.blamed_on = "harness" if out.timed_out else "agent"
        out.total_seconds = round(time.time() - t0, 1)
        say("no file produced")
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


# ── Reporting ───────────────────────────────────────────────────────────────


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


@as_function_node("workdir")
def BenchWorkDir(path: str = "bench_runs"):
    """Single source of truth for the output directory.

    Wire ``wf.workdir.path`` to change where all arms write their artifacts
    and where ``results.csv`` lands — without having to update three separate
    nodes.  Use a dated subdirectory (e.g. ``bench_runs/generic_2026-08-28``)
    to avoid overwriting a previous run.
    """
    return path


# ── Workflow ────────────────────────────────────────────────────────────────

wf = Workflow("workflow_agent_benchmark")

wf.task_suite = TaskSuite(tier="generic", limit=0)

wf.mirror = NodeLibraryMirror(allow_reference_workflows=False)

wf.pool = ThreadPoolExecutorNode(max_workers=3)

# Change wf.workdir.path once to redirect all output (both arms + results.csv).
wf.workdir = BenchWorkDir(path="bench_runs")

# One agent template per arm — identical in every respect but `arm`, which is
# the whole point of the comparison.
wf.agent_aiflow = WorkflowAgent(
    task="",
    model="sonnet",
    max_repairs=3,
    workdir=wf.workdir,
    exec_timeout_s=300,
    arm="aiflow",
    lib_dir=wf.mirror,
)

wf.agent_scratch = WorkflowAgent(
    task="",
    model="sonnet",
    max_repairs=3,
    workdir=wf.workdir,
    exec_timeout_s=300,
    arm="scratch",
    lib_dir=wf.mirror,
)

wf.bench_aiflow = IterToDataFrame(
    node=wf.agent_aiflow,
    input_label="task",
    values=wf.task_suite,
    executor=wf.pool.outputs.Executor,
    debug=False,
    store=False,
)

wf.bench_scratch = IterToDataFrame(
    node=wf.agent_scratch,
    input_label="task",
    values=wf.task_suite,
    executor=wf.pool.outputs.Executor,
    debug=False,
    store=False,
)

wf.results = StackResults(
    df_aiflow=wf.bench_aiflow,
    df_scratch=wf.bench_scratch,
    workdir=wf.workdir,
)

wf.report = BenchmarkReport(df=wf.results, max_repairs=3)

wf.figure = PlotBenchmark(df=wf.results, max_repairs=3)


if __name__ == "__main__":
    wf.run()
