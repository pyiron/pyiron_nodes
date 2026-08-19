# Duplicate-node review procedure

How we resolve duplicated `@as_*_node` definitions in `pyiron_nodes`, one node at a
time, with a human making every keep/replace/delete call.

Used for Tier A (2026-08-19, commit `48831d6`) and Tier B (12 groups) of the
*pyiron_nodes Duplicate Audit*. Follow this to run Tier C and Tier D, or to re-run
the whole thing after the tree drifts again.

---

## 1. Terminology

Fix these words before starting; the whole procedure depends on them being unambiguous.

| Term | Meaning |
|---|---|
| **A** | The *first* location listed in the audit for that node — in practice the structured/canonical tree (`atomistic/…`, `controls.py`, `utilities.py`). |
| **B** | The *second* location — in practice the `dpg2026/` fork or the flat sibling module. |
| **"keep A, delete B"** | A is untouched. B's definition is removed. |
| **"replace A with B, delete B"** | B's *body* is written into A's *location*. Then B is removed. The canonical import path never changes; only the implementation does. |
| **"replace B with A, delete A"** | Mirror image: B's location becomes canonical. |
| **"keep both"** | Skip; the duplicate stands. |

The critical one is **replace A with B, delete B** — it means *the code from B lives at
A's path*. It does **not** mean "delete A". Nothing that imports A ever breaks.

### Re-export policy

We do **not** add `from … import X` shims in the emptied module. Established by the
Tier A commit (`dpg2026/atomistic/fitting/dataset.py` kept a dangling `Optional`
import and no shim) and followed throughout Tier B. If the emptied module ends up
with no nodes at all, it becomes a candidate for deletion in the cleanup phase (§6).

---

## 2. One node at a time — the loop

For each group in the tier, in the audit's order:

### 2.1 Locate both definitions (line numbers in the audit are stale)

```bash
grep -rn "def <NodeName>(" --include=*.py .
```

The audit's line numbers were taken at commit `523329f`; every commit since shifts
them. Always re-grep.

### 2.2 Extract and diff the two bodies by AST, not by eye

Decorators must be included in the slice — the decorator carries the output-port
names, which is where much of the real drift hides.

```python
import ast, difflib
def grab(path, name):
    src = open(path).read().splitlines(keepends=True)
    for n in ast.parse("".join(src)).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            s = min([d.lineno for d in n.decorator_list] + [n.lineno])
            return s, src[s-1:n.end_lineno]

a_s, a = grab("<A path>", "<NodeName>")
b_s, b = grab("<B path>", "<NodeName>")
print("".join(difflib.unified_diff(a, b, fromfile="A", tofile="B", lineterm="\n")))
```

### 2.3 Gather the three facts that decide it

**Usage — who actually imports each side.** Search the whole development tree, not
just the repo, and include notebooks:

```bash
cd /cmmc/u/gvolpe/developement
grep -rn "<NodeName>" --include=*.py --include=*.ipynb . | grep -v "def <NodeName>"
```

> ⚠️ `cd` persists between shell calls. After grepping the parent directory, `cd`
> back into `pyiron_nodes/` before the next `git`/`python` command, or it will run
> against the wrong repo.

**History — which copy is maintained.** Both copies usually entered the repo in the
same bulk migration (`f98b7cb`, 2026-05-11, "migrate to pyiron_aiflow"), so "which
came first" is *not* answerable and should not be claimed. What *is* answerable is
what happened after:

```bash
git log --format="%h|%ad|%an|%s" --date=short -- <A path> | head -5
git log --format="%h|%ad|%an|%s" --date=short -- <B path> | head -5
```

A copy with post-migration commits is the maintained line; a copy with only the
directory-move commit is a frozen snapshot. (This is what settled `IterToDataFrame`.)

**Ports — what the node exposes downstream.** Two spellings produce a port name:

```python
@as_function_node("phase_data")     # explicit label
def X(...):
    df = ...
    return df                       # port is "phase_data"

@as_function_node                   # inferred from the returned variable
def X(...):
    phase_data = ...
    return phase_data               # port is also "phase_data"
```

Two copies can therefore look different and be identical, *or* look similar and
silently rename a port. Always state the resulting port name for both sides.

### 2.4 Present it in chat

One screen, in this order:

1. **Header** — `**A —** [file.py:LINE](path#LLINE)` and the same for B, with the
   similarity score from the audit.
2. **The diff**, trimmed to what matters. If the node is short, show the full body of
   at least one side.
3. **Classification of each difference**: cosmetic (formatting, import order,
   docstring wording) vs. behavioural (signature, return value, port name, control
   flow). Say plainly when a "0.99 similarity" difference is purely cosmetic, and
   when a "0.94" hides a changed return type.
4. **Usage** — call sites for each side, or "no call sites for either".
5. **History** — maintained vs. frozen, if it distinguishes them.
6. **Anything that must move with the node** — private helpers the node calls
   (`make_phase` for `PhasesFromDataFrame`), which the audit never lists because they
   are not decorated.
7. **Any bug spotted in both copies** — flag it, but keep it out of the decision.

Then ask, with 3–4 concrete options. Template:

- `Replace A with B, delete B` / `Keep A, delete B` — the two default poles
- A **merge** option whenever one side has a feature the other lacks
  (e.g. "Keep A + add `concavity`", "Keep A + take B's docstring")
- `Keep both, do nothing`

Mark one `(Recommended)` and put it first. Recommend on evidence — maintained line,
truthful annotations, no unused imports, more capable output — not on which path
looks tidier.

### 2.5 Apply

- **keep A** → delete B's definition only.
- **replace A with B** → edit A's body to match B *exactly*, then delete B. Verify
  with an AST diff that A and B are identical **before** removing B:

  ```python
  d = "".join(difflib.unified_diff(grab(A, name)[1], grab(B, name)[1], lineterm="\n"))
  print("IDENTICAL" if not d else d)
  ```

  This caught nothing silently wrong in Tier B, which is the point — it is cheap and
  it converts "I think I copied it right" into a fact.

- Deleting a node from the middle of a file: prefer an exact-match `Edit` that
  includes the following `def` line as an anchor. For long nodes, slice by AST line
  span instead, and re-check the surrounding blank lines afterwards.

### 2.6 Verify, every time

```bash
python -c "import ast;[ast.parse(open(f).read()) for f in ['<A>','<B>']];print('parse OK')"
grep -rn "def <NodeName>(" --include=*.py .        # must print exactly one line
```

---

## 3. Recording the decisions

Keep a running table; it is the commit message and the answer to "why is this gone?"
six months from now.

| # | Node | Decision | Canonical location |
|---|---|---|---|
| 1 | `SplitTrainingAndTesting` | replace A with B, delete B | `linearfit.py:255` |
| … | | | |

Note explicitly any **port renames** the decision caused — they are the only part of
this work that can break a graph a user saved earlier.

---

## 4. What *not* to do

- Don't batch. Twelve nodes reviewed one at a time surfaced three genuine behavioural
  divergences that a bulk "delete the fork" would have thrown away.
- Don't trust the similarity score. 0.99 was cosmetic; 0.90 hid a `float` vs
  `Figure` return; 0.86 hid two different plotting engines.
- Don't claim which copy is older unless git actually shows it.
- Don't silently fix unrelated bugs found along the way — surface them and let the
  human schedule them (see `PlotMuPhaseDiagram`'s inverted `border` check).

---

## 5. Tier-wide finish: workflow compatibility

After the last node, validate every consumer. Two static checks, no execution needed:

**Imports** — for each `from pyiron_nodes.… import X` in `Workflows/*.py`, resolve the
module to a file and confirm `X` is defined at module level.

**Wiring** — for each `wf.x = SomeNode(...)`, check every keyword argument against the
node's real signature, and every `wf.x.outputs.<label>` against the node's real output
labels (decorator labels, else the returned variable names).

A ready-made implementation lives at `scratchpad/wfcheck.py`. Known false positives it
already accounts for, and which any reimplementation must too:

- **`store=`** is injected by the node decorator, not declared in signatures — ignore it.
- **Module-level defs shadow nothing, but function-scoped imports shadow nothing
  either.** Only top-level `ImportFrom` statements bind names for the module body;
  a node defined locally with `@group_node` wins over a same-named import that appears
  inside another function. Getting this wrong reports a phantom `Bulk` mismatch.

Then fix what the checks find — including breakage that predates this work
(`add_water_film` → `AddWaterFilm` in the two electrochemistry workflows was a
rename never propagated).

---

## 6. Tier-wide finish: cleanup

Removing nodes empties modules. For each file touched:

1. **Modules with no nodes left** — check for importers repo-wide *and* in notebooks,
   check that the package `__init__.py` doesn't pull them in (in `dpg2026/` they are
   all empty), check whether any surviving private helper is duplicated in the live
   module. If dead on all counts, `git rm -f` it.
2. **Modules that keep content** — remove imports that no longer have a user.
3. **Helpers left calling deleted nodes** — a module whose only remaining function
   references names that no longer exist is a latent `NameError`, not dead weight.
   `dpg2026/atomistic/fitting/dataset.py` was found this way: its `make_linearfit` was
   byte-identical to the live one and every node it called had been removed.

Verify nothing references a deleted module:

```bash
grep -rn "basic\.list\|basic\.math\|…" --include=*.py --include=*.ipynb \
     /cmmc/u/gvolpe/developement
```

---

## 7. Session checklist

- [ ] Re-grep line numbers; audit numbers are stale
- [ ] AST-diff both bodies, decorators included
- [ ] Usage across repo **and** notebooks
- [ ] Git history: maintained vs frozen
- [ ] Port names for both sides stated
- [ ] Helpers that must move with the node identified
- [ ] Present → ask → apply, one node at a time
- [ ] AST-diff A against B before deleting B (on "replace")
- [ ] Parse check + single-definition check after each node
- [ ] Decision table kept
- [ ] Workflow import + wiring check at the end
- [ ] Emptied modules and dangling imports cleaned
- [ ] Port renames called out explicitly
