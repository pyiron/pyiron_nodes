"""
Application 2 — Self-correcting agentic code writer with aiflow
==============================================================
Backs the "self-correcting agentic code writer" application in
``docs/paper_draft.md``. A condensed, importable version of
``notebooks/agentic_code_writer.ipynb``.

The agentic *generate -> try -> fix* loop is expressed with the SAME primitives
as a numerical workflow: each LLM call is an ``@as_function_node``; the loop is a
higher-order ``CodeAgent`` node; ``IterToDataFrame`` batches a task suite.

Backends: Claude, OpenAI-compatible, or local Ollama (see the notebook for
configuration). Because a live LLM is not available in headless CI, this module
also exposes ``illustrative_success_rates()`` which returns representative
numbers for Fig. 4. Those numbers are PLACEHOLDERS pending a real benchmark run
and are flagged as such in the figure.
"""

import sys

sys.path.insert(0, "pyiron_core/src")
sys.path.insert(0, "/Users/jorgneugebauer/git_libs/pyiron_nodes")

import io
import contextlib
import textwrap

import pandas as pd

from core import as_function_node, Workflow, Node


# --------------------------------------------------------------------------- #
#  Building-block nodes (LLM calls and code execution are ordinary nodes).      #
#  `call_llm` is injected so the module imports without any backend present.    #
# --------------------------------------------------------------------------- #

@as_function_node("code")
def GenerateCode(task: str, call_llm=None, max_tokens: int = 2000):
    """Ask the LLM to write Python code for *task*."""
    from pyiron_ai.node_store import extract_code

    prompt = textwrap.dedent(
        f"""
        Write Python code to accomplish the following task.
        Return ONLY a fenced ```python code block — no explanation.

        Task: {task}
        """
    ).strip()
    code = extract_code(call_llm(prompt, max_tokens))
    return code


@as_function_node(["success", "output", "error"])
def TryCode(code: str):
    """Execute *code* in a fresh namespace; capture stdout and any exception."""
    buf = io.StringIO()
    error = None
    with contextlib.redirect_stdout(buf):
        try:
            exec(code, {})
        except Exception as e:  # noqa: BLE001 — deliberately broad for the sandbox
            error = f"{type(e).__name__}: {e}"
    return (error is None), buf.getvalue(), error


@as_function_node("code")
def FixCode(task: str, code: str, error: str, call_llm=None, max_tokens: int = 2000):
    """Ask the LLM to repair *code* given the *error* it produced."""
    from pyiron_ai.node_store import extract_code

    prompt = textwrap.dedent(
        f"""
        The following Python code for the task below raised an error.
        Return ONLY a corrected fenced ```python code block.

        Task: {task}
        Code:
        {code}
        Error: {error}
        """
    ).strip()
    fixed = extract_code(call_llm(prompt, max_tokens))
    return fixed


# --------------------------------------------------------------------------- #
#  Higher-order agent node: drives generate -> try -> fix until success.        #
#  The loop lives INSIDE one node; the outer graph stays a DAG.                 #
# --------------------------------------------------------------------------- #

@as_function_node(["success", "attempts", "code", "output"])
def CodeAgent(task: str, call_llm=None, max_repairs: int = 3):
    """Self-correcting loop built from the building-block nodes above."""
    code = GenerateCode(task=task, call_llm=call_llm).pull()
    attempts = 1
    success, output, error = TryCode(code=code).pull()
    while not success and attempts <= max_repairs:
        code = FixCode(task=task, code=code, error=error, call_llm=call_llm).pull()
        success, output, error = TryCode(code=code).pull()
        attempts += 1
    return success, attempts, code, output


def run_agent(task: str, call_llm, max_repairs: int = 3):
    """Wire and run a single-task agent workflow."""
    wf = Workflow("code_agent")
    wf.agent = CodeAgent(task=task, call_llm=call_llm, max_repairs=max_repairs)
    return wf.run()


# --------------------------------------------------------------------------- #
#  Figure data.                                                                 #
# --------------------------------------------------------------------------- #

def illustrative_success_rates():
    """Representative success rate (%) by model and number of repair cycles.

    PLACEHOLDER data for Fig. 4 pending a real benchmarked run against a live
    backend. Shape reflects the expected trend: larger models start higher and
    every repair cycle recovers additional tasks with diminishing returns.
    """
    data = {
        "repairs": [0, 1, 2, 3],
        "Claude Opus 4.8": [78, 92, 97, 98],
        "GPT-class": [64, 83, 91, 94],
        "Llama 3.1 8B (local)": [41, 60, 71, 76],
    }
    return pd.DataFrame(data), True  # (df, is_placeholder)


if __name__ == "__main__":
    df, placeholder = illustrative_success_rates()
    tag = "  [ILLUSTRATIVE — placeholder]" if placeholder else ""
    print(f"=== Agentic code writer: success rate by repair cycle ==={tag}")
    print(df.to_string(index=False))
    print(
        "\nTo run for real, provide a `call_llm(prompt, max_tokens)` callable "
        "(see notebooks/agentic_code_writer.ipynb) and call run_agent(task, call_llm)."
    )
