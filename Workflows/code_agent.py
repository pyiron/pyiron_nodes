from dataclasses import dataclass
from typing import Optional, Literal

from pyiron_nodes.executors import ThreadPoolExecutorNode
from pyiron_nodes.controls import IterToDataFrame
from core import Workflow
from core import as_function_node


# ── LLM config dataclass (plain data, not a node) ───────────────────────────

@dataclass
class LLMConfig:
    """Bundle backend identifier and model name for routing calls."""
    backend: str
    model_name: str


_DEFAULT_CONFIG = LLMConfig(backend="ollama", model_name="llama3.1")

OPENAI_API_BASE = "https://chat-ai.academiccloud.de/v1"


def call_llm(prompt: str, max_tokens: int, cfg: LLMConfig = None) -> str:
    """Dispatch one prompt to the chosen backend and return the raw response text.

    Supported backends: ``"ollama"`` (local server), ``"openai_academic"``
    (OpenAI-compatible endpoint), and ``"claude"`` (Anthropic Foundry). All
    heavy imports live inside the branches so *loading* this workflow requires
    no LLM libraries — only *running* it does.
    """
    c = cfg if cfg is not None else _DEFAULT_CONFIG

    if c.backend == "ollama":
        from pyiron_ai.node_store import _ollama_generate
        return _ollama_generate(model=c.model_name, prompt=prompt, max_tokens=max_tokens)

    if c.backend == "openai_academic":
        import keyring
        from pyiron_ai.node_store import _openai_generate
        content, _ = _openai_generate(
            model=c.model_name, prompt=prompt, max_tokens=max_tokens,
            api_key=keyring.get_password("openai", "api_key"),
            api_base=OPENAI_API_BASE,
        )
        return content

    if c.backend == "claude":
        import os
        from anthropic import AnthropicFoundry
        client = AnthropicFoundry(
            api_key=os.environ["ANTHROPIC_FOUNDRY_API_KEY"],
            resource=os.environ["ANTHROPIC_FOUNDRY_RESOURCE"],
        )
        response = client.messages.create(
            model=c.model_name, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    raise ValueError(f"Unknown backend: {c.backend!r}")


# ── Local node definitions ──────────────────────────────────────────────────


@as_function_node("models")
def ListModels(
    backend: Optional[Literal["openai_academic", "claude", "ollama"]] = "ollama",
    index: Optional[int] = 0,
):
    """Query available models for *backend* and return one as an ``LLMConfig``.

    The ``backend`` input renders as a **dropdown** in the GUI; ``index``
    selects which model to use (0 = first installed model). Wire the
    ``models`` output port into ``GenerateCode.model`` or ``CodeAgent.model``
    to route all LLM calls through this single configuration node.

    Parameters
    ----------
    backend : str
        Which LLM backend to use (dropdown in the GUI).
    index : int or None
        ``None`` → return a list of ``LLMConfig`` objects, one per available
        model, ready to sweep an agent over with ``IterToDataFrame``;
        ``0, 1, …`` → return the single ``LLMConfig(backend, model_name[index])``.

    Returns
    -------
    models : LLMConfig or list[LLMConfig]
        A single ``LLMConfig`` when *index* is an integer, or a list of
        ``LLMConfig`` objects (one per model) when ``index is None`` — each ready
        to wire into an agent node's ``model`` port.
    """
    if backend == "ollama":
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
        models_list = [m["name"] for m in r.json().get("models", [])]

    elif backend == "openai_academic":
        import keyring
        from pyiron_ai.node_store import _list_available_models
        models_list = _list_available_models(
            api_key=keyring.get_password("openai", "api_key"),
            api_base=OPENAI_API_BASE,
        )

    elif backend == "claude":
        import os
        _KNOWN = ["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"]
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_FOUNDRY_API_KEY"])
            models_list = [m.id for m in client.models.list()]
        except Exception:
            models_list = _KNOWN

    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    if index is None:
        return [LLMConfig(backend=backend, model_name=name) for name in models_list]
    return LLMConfig(backend=backend, model_name=models_list[index])


@as_function_node(["code", "elapsed", "model_name"])
def GenerateCode(task: str = "", model: LLMConfig = None, max_tokens: int = 2000):
    """Ask the LLM to write Python code that accomplishes a plain-English *task*.

    The first step of the agentic loop. Prompts the backend for a single
    fenced ``python`` block and extracts the code from the reply.

    Parameters
    ----------
    task : str
        Natural-language description of what the code should do.
    model : LLMConfig
        Backend/model config produced by ``ListModels``.
    max_tokens : int
        Generation budget for the reply.

    Returns
    -------
    code : str
        The extracted Python source (may or may not run — checked by ``TryCode``).
    elapsed : float
        Wall-clock seconds the LLM call took.
    model_name : str
        The model that was actually used.
    """
    import time
    import textwrap
    from pyiron_ai.node_store import extract_code

    cfg = model if model is not None else _DEFAULT_CONFIG
    prompt = textwrap.dedent(f"""
        Write Python code to accomplish the following task.
        Return ONLY a fenced ```python code block — no explanation.

        Task: {task}
    """).strip()
    t0 = time.time()
    code = extract_code(call_llm(prompt, max_tokens, cfg=cfg))
    return code, round(time.time() - t0, 1), cfg.model_name


@as_function_node(["success", "output", "error"])
def TryCode(code: str = ""):
    """Execute a Python code string in a fresh namespace, capturing the outcome.

    The "observe" step of the loop. Runs *code* with ``exec`` in an isolated
    namespace, redirecting stdout and catching any exception so a failure never
    crashes the workflow.

    Parameters
    ----------
    code : str
        Python source to execute.

    Returns
    -------
    success : bool
        ``True`` if the code ran without raising.
    output : str
        Whatever the code printed to stdout.
    error : str or None
        ``"ExceptionType: message"`` if it raised, else ``None``.
    """
    import io
    import contextlib

    buf = io.StringIO()
    error = None
    with contextlib.redirect_stdout(buf):
        try:
            exec(code, {})
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
    return (error is None), buf.getvalue(), error


@as_function_node("fixed_code")
def FixCode(
    code: str = "",
    error: str = "",
    task: str = "",
    model: LLMConfig = None,
    max_tokens: int = 2000,
):
    """Ask the LLM to repair *code* given the *error* it raised.

    The "repair" step. Sends the original task description, the failing code,
    and the error message back to the model and asks for a corrected fenced
    ``python`` block.

    Returns
    -------
    fixed_code : str
        The repaired Python source, to be re-tested by ``TryCode``.
    """
    import textwrap
    from pyiron_ai.node_store import extract_code

    prompt = textwrap.dedent(f"""
        The following Python code was written to accomplish this task:
        "{task}"

        It raised an error:
        {error}

        Here is the code:
        ```python
        {code}
        ```

        Fix the code so it runs without errors and fulfils the task.
        Return ONLY the corrected Python code block.
    """).strip()
    return extract_code(call_llm(prompt, max_tokens, cfg=model if model is not None else _DEFAULT_CONFIG))


@as_function_node("tasks")
def TaskList(
    task_1: str = "print the first 10 Fibonacci numbers",
    task_2: str = "compute the sum of squares of all even numbers from 1 to 20 and print it",
    task_3: str = "print all prime numbers below 30",
    task_4: str = "",
    task_5: str = "",
):
    """Collect up to five plain-English coding tasks into a list for batch sweeping.

    Each ``task_n`` input is a GUI-settable string port. Empty strings are
    dropped automatically, so leaving ``task_4`` and ``task_5`` blank runs only
    the first three tasks. Wire the ``tasks`` output into the ``values`` port of
    ``IterToDataFrame`` to sweep ``CodeAgent`` over the full list.

    Returns
    -------
    tasks : list[str]
        Non-empty task strings in order.
    """
    return [t for t in [task_1, task_2, task_3, task_4, task_5] if t.strip()]


@as_function_node(["success", "code", "final_output", "attempts_taken", "elapsed", "model_name"])
def CodeAgent(
    task: str = "print the first 10 Fibonacci numbers",
    model: LLMConfig = None,
    max_attempts: int = 4,
    max_tokens: int = 2000,
):
    """Self-correcting agent: generate → try → fix, looping until the code runs.

    A **higher-order node**: the full generate-execute-repair loop lives inside
    this single function, driving ``GenerateCode`` / ``TryCode`` / ``FixCode``
    via ``.run()``. Because the loop is internal, the outer workflow graph stays
    a valid DAG and ``CodeAgent`` itself can be used as a template in
    ``IterToDataFrame`` to batch an entire task suite.

    Parameters
    ----------
    task : str
        The programming task to solve — the field swept when batching.
    model : LLMConfig
        Backend/model config produced by ``ListModels`` (falls back to
        ``_DEFAULT_CONFIG`` when ``None``).
    max_attempts : int
        Maximum number of generate-and-repair cycles.
    max_tokens : int
        Token budget per LLM call.

    Returns
    -------
    success : bool
        ``True`` if the final code ran without raising.
    code : str
        The final code produced.
    final_output : str
        The stdout of the last execution attempt.
    attempts_taken : int
        Number of execution attempts (1 = solved on the first try).
    elapsed : float
        Total wall-clock time in seconds.
    model_name : str
        The model that was used.
    """
    import time
    t0 = time.time()

    gen = GenerateCode(task=task, model=model, max_tokens=max_tokens)
    gen.run()
    code = gen.outputs.code.value
    model_name = gen.outputs.model_name.value

    output = ""
    for attempt in range(1, max_attempts + 1):
        runner = TryCode(code=code)
        runner.run()
        success = runner.outputs.success.value
        output = runner.outputs.output.value
        error = runner.outputs.error.value

        if success:
            break

        fixer = FixCode(code=code, error=error, task=task, model=model, max_tokens=max_tokens)
        fixer.run()
        code = fixer.outputs.fixed_code.value

    return success, code, output, attempt, round(time.time() - t0, 1), model_name


# ── Workflow ────────────────────────────────────────────────────────────────
#
# Open in PyironFlow:
#   1. Set `backend` / `index` on ListModels (dropdown + spinner).
#   2. Edit task strings on the TaskList node (up to 5).
#   3. Adjust `max_workers` on the ThreadPool node for parallel execution.
#   4. Run — each task is solved by a self-correcting agent; results collected
#      into a DataFrame with columns: task, success, code, final_output,
#      attempts_taken, elapsed, model_name.

wf = Workflow("code_agent")

wf.model = ListModels(backend="ollama", index=0)

wf.tasks = TaskList(
    task_1="print the first 10 Fibonacci numbers",
    task_2="compute the sum of squares of all even numbers from 1 to 20 and print it",
    task_3="print all prime numbers below 30",
)

wf.pool = ThreadPoolExecutorNode(max_workers=3)

wf.agent = CodeAgent(
    task="",
    model=wf.model.outputs.models,
    max_attempts=4,
)

wf.sweep = IterToDataFrame(
    node=wf.agent,
    input_label="task",
    values=wf.tasks,
    executor=wf.pool.outputs.Executor,
    debug=False,
    store=False,
)
