# Codemap Atlas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a re-runnable, visual-spatial "codebase atlas" — four composable skills that emit SHA-keyed JSON snapshots + self-contained HTML showing the structural code map, a "you are here" project-state narrative, and a diff vs the last run / a given commit.

**Architecture:** Deterministic Python scripts (stdlib `ast` + `git`) extract *facts* into versioned JSON; Claude skills synthesize *understanding* (narrative, change summary); an HTML template fuses both into pannable Mermaid diagrams (rendered client-side, no graphviz). Every run writes a snapshot keyed to the git SHA, so "diff vs last run" and "diff vs commit X" are one operation.

> **Superseded 2026-07-01 (as-built layout differs).** Two things changed after this plan was written:
> (1) the renderer emits **self-contained inline SVG**, not client-side Mermaid (the CDN dependency broke
> offline); (2) the engine + skills were moved out of the repo to a shared home — engine at
> `~/.claude/codemap/scripts/`, skills at `~/.claude/skills/codemap-*`, each repo keeping only
> `docs/codemap/codemap.config.yaml` + generated artifacts. Task steps below that write to
> `docs/codemap/scripts/...` or `tests/codemap/...` describe the original build layout. Current
> architecture: `docs/codemap/README.md`.

**Tech Stack:** Python 3.11 stdlib only for extraction (`ast`, `subprocess`+git, `json`, `pathlib`); `PyYAML` (already in env) for config; Mermaid.js via CDN for client-side rendering; pytest for tests.

**Design spec:** `docs/superpowers/specs/2026-06-29-codemap-atlas-design.md` — read it first.

## Global Constraints

- **No external binaries beyond `git`.** No `dot`/graphviz, no mermaid-cli, no pyreverse. Extraction is stdlib `ast`; rendering is client-side Mermaid.js. (Verified 2026-06-29: env has none of those tools.)
- **Python env:** `/mnt/weka/adovlatyan/.conda/envs/dinov3_env_210clone/bin/python` (torch 2.10). Scripts must run under plain CPython 3.11 — no torch import.
- **JSON contracts are `schema_version: 1`** exactly as defined in design spec §8. Do not silently change field names — the differ depends on them.
- **All artifacts under `docs/codemap/`**, git-tracked, snapshots keyed by short git SHA, pruned to `snapshots.keep_last` (default 10).
- **Scripts emit facts; skills emit prose.** `extract_*.py` never writes narrative; the narrative/`summary` fields are filled by the skill layer only.
- **Repo search:** use `fff` MCP tools, not raw grep/find (per repo CLAUDE.md).
- **Validate against fixture before real repo** (repo harness doctrine): every extractor is TDD'd on a tiny fixture package with hand-known edges before being pointed at `dinov3/`.

---

## File Structure

```
docs/codemap/
  codemap.config.yaml            # Task 1 — per-repo template (dinov3-tuned)
  scripts/
    cm_common.py                 # Task 2 — config load, snapshot paths, git helpers, schema version
    extract_structure.py         # Tasks 3-5 — ast module-overview + lens call-graphs + mermaid
    extract_state_facts.py       # Task 6 — git/phase-doc/memory/job facts
    diff_snapshots.py            # Task 7 — structural + state delta
    render_html.py               # Task 8 — fragment + atlas templating
  snapshots/                     # generated
tests/codemap/
  fixtures/sample_pkg/           # Task 3 — tiny package with known call/import edges
  test_cm_common.py              # Task 2
  test_extract_structure.py      # Tasks 3-5
  test_extract_state_facts.py    # Task 6
  test_diff_snapshots.py         # Task 7
  test_render_html.py            # Task 8
skills (user skills dir, e.g. ~/.claude/skills/ or a plugin):
  codemap-structure/SKILL.md     # Task 9
  codemap-state/SKILL.md         # Task 10
  codemap-diff/SKILL.md          # Task 11
  codemap-atlas/SKILL.md         # Task 12
```

Run tests with:
```bash
PY=/mnt/weka/adovlatyan/.conda/envs/dinov3_env_210clone/bin/python
cd /home/adovlatyan/dinov3-performance-optimizations/dinov3
$PY -m pytest tests/codemap/ -v
```

---

### Task 1: Config template + directory scaffold

**Files:**
- Create: `docs/codemap/codemap.config.yaml`
- Create: `docs/codemap/.gitignore` (ignore nothing of substance; keep snapshots tracked — see below)

**Interfaces:**
- Produces: the YAML config consumed by every script and skill. Schema is design spec §7.

- [ ] **Step 1: Write the config** (copy design spec §7 verbatim into `docs/codemap/codemap.config.yaml`). It must contain `project`, `structure.lenses` (the 4 dinov3 lenses), `state`, `snapshots` keys exactly as in the spec.

- [ ] **Step 2: Create `docs/codemap/.gitignore`** with a single comment line so the dir is tracked but no rule excludes snapshots:
```
# Snapshots ARE tracked on purpose (git history = diff substrate). Nothing ignored here.
```

- [ ] **Step 3: Verify YAML parses**
Run: `$PY -c "import yaml,pathlib; print(list(yaml.safe_load(pathlib.Path('docs/codemap/codemap.config.yaml').read_text())))"`
Expected: `['project', 'structure', 'state', 'snapshots']`

- [ ] **Step 4: Commit**
```bash
git add docs/codemap/codemap.config.yaml docs/codemap/.gitignore
git commit -m "feat(codemap): config template + scaffold"
```

---

### Task 2: `cm_common.py` — shared helpers

**Files:**
- Create: `docs/codemap/scripts/cm_common.py`
- Test: `tests/codemap/test_cm_common.py`

**Interfaces:**
- Produces:
  - `SCHEMA_VERSION = 1`
  - `load_config(repo_root: Path) -> dict`
  - `git_sha(repo_root: Path) -> str` (short sha, `+ "-dirty"` if working tree dirty)
  - `snapshot_dir(repo_root: Path, cfg: dict, sha: str) -> Path` (creates it)
  - `write_json(path: Path, obj: dict) -> None` (sorted keys, indent 2, trailing newline)
  - `prune_snapshots(repo_root: Path, cfg: dict) -> list[str]` (keep newest `keep_last` by mtime; returns removed shas)
  - `update_latest_symlink(snap_root: Path, sha: str) -> None`

- [ ] **Step 1: Write the failing test** `tests/codemap/test_cm_common.py`:
```python
import json, subprocess
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/codemap/scripts"))
import cm_common as cm

def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

def make_repo(tmp_path):
    repo = tmp_path / "r"; repo.mkdir()
    _git(repo, "init", "-q"); _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text("x")
    _git(repo, "add", "."); _git(repo, "commit", "-qm", "init")
    return repo

def test_schema_version_is_one():
    assert cm.SCHEMA_VERSION == 1

def test_git_sha_clean_and_dirty(tmp_path):
    repo = make_repo(tmp_path)
    sha = cm.git_sha(repo)
    assert sha and "-dirty" not in sha
    (repo / "f.txt").write_text("y")
    assert cm.git_sha(repo).endswith("-dirty")

def test_write_json_roundtrip(tmp_path):
    p = tmp_path / "a.json"
    cm.write_json(p, {"b": 1, "a": 2})
    text = p.read_text()
    assert text.endswith("\n")
    assert list(json.loads(text)) == ["a", "b"]  # sorted keys

def test_prune_keeps_newest(tmp_path):
    repo = make_repo(tmp_path)
    snap = repo / "docs/codemap/snapshots"; snap.mkdir(parents=True)
    import os, time
    for i, name in enumerate(["aaa", "bbb", "ccc"]):
        d = snap / name; d.mkdir()
        os.utime(d, (1000 + i, 1000 + i))
    cfg = {"snapshots": {"dir": "docs/codemap/snapshots", "keep_last": 2}}
    removed = cm.prune_snapshots(repo, cfg)
    assert removed == ["aaa"]
    assert {p.name for p in snap.iterdir()} == {"bbb", "ccc"}
```

- [ ] **Step 2: Run test to verify it fails**
Run: `$PY -m pytest tests/codemap/test_cm_common.py -v`
Expected: FAIL (ModuleNotFoundError: cm_common)

- [ ] **Step 3: Implement `docs/codemap/scripts/cm_common.py`:**
```python
"""Shared helpers for the codemap toolchain. Stdlib + PyYAML only."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import yaml

SCHEMA_VERSION = 1

def load_config(repo_root: Path) -> dict:
    p = Path(repo_root) / "docs/codemap/codemap.config.yaml"
    if not p.exists():
        raise FileNotFoundError(f"missing codemap config: {p}")
    return yaml.safe_load(p.read_text())

def git_sha(repo_root: Path) -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root,
                         check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root,
                           check=True, capture_output=True, text=True).stdout.strip()
    return f"{sha}-dirty" if dirty else sha

def snapshot_dir(repo_root: Path, cfg: dict, sha: str) -> Path:
    d = Path(repo_root) / cfg["snapshots"]["dir"] / sha
    d.mkdir(parents=True, exist_ok=True)
    return d

def write_json(path: Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

def prune_snapshots(repo_root: Path, cfg: dict) -> list[str]:
    snap_root = Path(repo_root) / cfg["snapshots"]["dir"]
    keep = cfg["snapshots"].get("keep_last", 10)
    dirs = [p for p in snap_root.iterdir() if p.is_dir() and not p.is_symlink()]
    dirs.sort(key=lambda p: p.stat().st_mtime)
    removed = []
    for p in dirs[:-keep] if keep < len(dirs) else []:
        for child in sorted(p.rglob("*"), reverse=True):
            child.unlink() if child.is_file() else child.rmdir()
        p.rmdir(); removed.append(p.name)
    return removed

def update_latest_symlink(snap_root: Path, sha: str) -> None:
    link = Path(snap_root) / "latest"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(sha)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `$PY -m pytest tests/codemap/test_cm_common.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**
```bash
git add docs/codemap/scripts/cm_common.py tests/codemap/test_cm_common.py
git commit -m "feat(codemap): shared helpers (config, git sha, snapshot prune)"
```

---

### Task 3: `extract_structure.py` — module-overview graph (TDD on fixture)

**Files:**
- Create: `docs/codemap/scripts/extract_structure.py`
- Create fixture: `tests/codemap/fixtures/sample_pkg/__init__.py`, `mod_a.py`, `mod_b.py`
- Test: `tests/codemap/test_extract_structure.py`

**Interfaces:**
- Produces (module level, consumed by later tasks):
  - `build_module_overview(package_root: Path) -> dict` returns `{"nodes":[{"id","label","loc"}], "edges":[{"src","dst","kind":"import","weight"}]}` where ids are dotted module names *within the package only*.
  - `module_name(package_root: Path, file: Path) -> str` (dotted name relative to package parent).

- [ ] **Step 1: Create the fixture package.**
`tests/codemap/fixtures/sample_pkg/__init__.py`: (empty)
`tests/codemap/fixtures/sample_pkg/mod_a.py`:
```python
from sample_pkg import mod_b

def alpha(x):
    return mod_b.beta(x) + 1

def gamma():
    return alpha(0)
```
`tests/codemap/fixtures/sample_pkg/mod_b.py`:
```python
def beta(y):
    return y * 2
```

- [ ] **Step 2: Write the failing test** (append to `tests/codemap/test_extract_structure.py`):
```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/codemap/scripts"))
import extract_structure as es

FIX = Path(__file__).resolve().parent / "fixtures/sample_pkg"

def test_module_overview_has_intra_package_import_edge():
    ov = es.build_module_overview(FIX)
    ids = {n["id"] for n in ov["nodes"]}
    assert "sample_pkg.mod_a" in ids and "sample_pkg.mod_b" in ids
    edges = {(e["src"], e["dst"]) for e in ov["edges"]}
    assert ("sample_pkg.mod_a", "sample_pkg.mod_b") in edges

def test_module_overview_loc_counted():
    ov = es.build_module_overview(FIX)
    node = next(n for n in ov["nodes"] if n["id"] == "sample_pkg.mod_b")
    assert node["loc"] >= 2
```

- [ ] **Step 3: Run test to verify it fails**
Run: `$PY -m pytest tests/codemap/test_extract_structure.py -v`
Expected: FAIL (ModuleNotFoundError / AttributeError)

- [ ] **Step 4: Implement the module-overview part of `extract_structure.py`:**
```python
"""Structural extraction via stdlib ast. No external binaries. Approximate name-based
call resolution (orientation aid, not a sound static analysis)."""
from __future__ import annotations
import ast
from pathlib import Path

def _iter_py(package_root: Path):
    for p in sorted(Path(package_root).rglob("*.py")):
        if "__pycache__" not in p.parts:
            yield p

def module_name(package_root: Path, file: Path) -> str:
    package_root = Path(package_root); file = Path(file)
    rel = file.relative_to(package_root.parent).with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    return ".".join(parts)

def build_module_overview(package_root: Path) -> dict:
    package_root = Path(package_root)
    pkg_top = package_root.name
    modules = {module_name(package_root, f): f for f in _iter_py(package_root)}
    nodes, edges = [], {}
    for modname, f in modules.items():
        src = f.read_text()
        nodes.append({"id": modname, "label": modname.split(".")[-1],
                      "loc": len(src.splitlines())})
        tree = ast.parse(src)
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                targets = [node.module]
            for t in targets:
                if not t.startswith(pkg_top):
                    continue
                # resolve to nearest known module id
                dst = t if t in modules else None
                if dst is None:
                    dst = next((m for m in modules if t.startswith(m + ".") or m.startswith(t + ".")), None)
                if dst and dst != modname:
                    edges[(modname, dst)] = edges.get((modname, dst), 0) + 1
    return {"nodes": sorted(nodes, key=lambda n: n["id"]),
            "edges": [{"src": s, "dst": d, "kind": "import", "weight": w}
                      for (s, d), w in sorted(edges.items())]}
```

- [ ] **Step 5: Run test to verify it passes**
Run: `$PY -m pytest tests/codemap/test_extract_structure.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**
```bash
git add docs/codemap/scripts/extract_structure.py tests/codemap/test_extract_structure.py tests/codemap/fixtures
git commit -m "feat(codemap): module-overview import graph via ast"
```

---

### Task 4: `extract_structure.py` — lens call-graphs

**Files:**
- Modify: `docs/codemap/scripts/extract_structure.py`
- Modify: `tests/codemap/test_extract_structure.py`

**Interfaces:**
- Consumes: `module_name`, `_iter_py` from Task 3.
- Produces:
  - `build_symbol_table(package_root: Path) -> dict[str, dict]` mapping a *simple* callable name (function/method/class) to `{"qualname","file","line","kind"}`. On name collisions, keep a list under `"_collisions"`.
  - `build_lens(package_root, entrypoints: list[str], max_depth: int) -> dict` returns `{"name"?, "entrypoints", "nodes":[{"id","file","line","kind"}], "edges":[{"src","dst","kind":"call"}], "unresolved_entrypoints":[...]}`. Entry format `"relpath.py:qualname"` or `"relpath.py"` (whole-module = all its top defs as entrypoints).

  Resolution is **name-based and best-effort**: a `Call` to `foo(...)` or `x.foo(...)` resolves to symbol-table entry `foo` if unique; ambiguous/missing calls are dropped (recorded only as edges when resolved). BFS to `max_depth`.

- [ ] **Step 1: Write the failing test** (append):
```python
def test_lens_follows_calls_to_depth():
    lens = es.build_lens(FIX, ["mod_a.py:gamma"], max_depth=3)
    node_ids = {n["id"] for n in lens["nodes"]}
    # gamma -> alpha -> beta
    assert {"gamma", "alpha", "beta"} <= node_ids
    edges = {(e["src"], e["dst"]) for e in lens["edges"]}
    assert ("gamma", "alpha") in edges
    assert ("alpha", "beta") in edges

def test_lens_records_unresolved_entrypoint():
    lens = es.build_lens(FIX, ["mod_a.py:does_not_exist"], max_depth=2)
    assert "mod_a.py:does_not_exist" in lens["unresolved_entrypoints"]
```

- [ ] **Step 2: Run test to verify it fails**
Run: `$PY -m pytest tests/codemap/test_extract_structure.py -k lens -v`
Expected: FAIL (no build_lens)

- [ ] **Step 3: Implement (append to `extract_structure.py`):**
```python
def _defs_in_module(tree, modfile):
    """Yield (simple_name, qualname, lineno, kind, node) for top-level & class methods."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node.name, node.lineno, "function", node
        elif isinstance(node, ast.ClassDef):
            yield node.name, node.name, node.lineno, "class", node
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield sub.name, f"{node.name}.{sub.name}", sub.lineno, "method", sub

def build_symbol_table(package_root: Path) -> dict:
    table, collisions = {}, {}
    for f in _iter_py(package_root):
        tree = ast.parse(f.read_text())
        for simple, qual, line, kind, node in _defs_in_module(tree, f):
            entry = {"qualname": qual, "file": str(f), "line": line, "kind": kind, "_node": node}
            if simple in table:
                collisions.setdefault(simple, [table[simple]]).append(entry)
            else:
                table[simple] = entry
    table["_collisions"] = collisions
    return table

def _called_names(node) -> list[str]:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                out.append(f.id)
            elif isinstance(f, ast.Attribute):
                out.append(f.attr)
    return out

def _resolve_entrypoint(package_root, table, ep):
    if ":" in ep:
        _, qual = ep.split(":", 1)
        simple = qual.split(".")[-1]
    else:
        simple = None  # whole-module handled by caller
    if simple and simple in table and simple not in table.get("_collisions", {}):
        return simple
    return None

def build_lens(package_root, entrypoints, max_depth=3) -> dict:
    table = build_symbol_table(Path(package_root))
    collisions = table.get("_collisions", {})
    start, unresolved = [], []
    for ep in entrypoints:
        r = _resolve_entrypoint(package_root, table, ep)
        if r:
            start.append(r)
        else:
            unresolved.append(ep)
    nodes, edges, seen = {}, set(), set()
    frontier = [(s, 0) for s in start]
    for s in start:
        e = table[s]; nodes[s] = {"id": s, "file": e["file"], "line": e["line"], "kind": e["kind"]}
    while frontier:
        name, depth = frontier.pop(0)
        if name in seen or depth >= max_depth:
            continue
        seen.add(name)
        entry = table.get(name)
        if not entry:
            continue
        for callee in _called_names(entry["_node"]):
            if callee in table and callee not in collisions and callee != name:
                ce = table[callee]
                nodes.setdefault(callee, {"id": callee, "file": ce["file"],
                                          "line": ce["line"], "kind": ce["kind"]})
                edges.add((name, callee))
                frontier.append((callee, depth + 1))
    return {"entrypoints": entrypoints,
            "nodes": sorted(nodes.values(), key=lambda n: n["id"]),
            "edges": [{"src": s, "dst": d, "kind": "call"} for s, d in sorted(edges)],
            "unresolved_entrypoints": unresolved}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `$PY -m pytest tests/codemap/test_extract_structure.py -k lens -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**
```bash
git add docs/codemap/scripts/extract_structure.py tests/codemap/test_extract_structure.py
git commit -m "feat(codemap): lens call-graph extraction (name-based BFS)"
```

---

### Task 5: `extract_structure.py` — mermaid + CLI + snapshot write

**Files:**
- Modify: `docs/codemap/scripts/extract_structure.py`
- Modify: `tests/codemap/test_extract_structure.py`

**Interfaces:**
- Consumes: `build_module_overview`, `build_lens` (Tasks 3-4), `cm_common`.
- Produces:
  - `to_mermaid(graph: dict, *, directed=True) -> str` — `graph TD` text; node ids sanitized, labels quoted.
  - `build_structure(repo_root: Path, cfg: dict) -> dict` — the full `structure.json` object (spec §8) incl. `module_overview`, `lenses` (each with config `name`/`description`), and a `mermaid` dict keyed by `module_overview` + each lens name.
  - CLI `python extract_structure.py [--repo .]` writes `structure.json` into the current snapshot dir and prints its path.

- [ ] **Step 1: Write the failing test** (append):
```python
def test_to_mermaid_renders_edges():
    g = {"nodes": [{"id": "alpha"}, {"id": "beta"}], "edges": [{"src": "alpha", "dst": "beta"}]}
    m = es.to_mermaid(g)
    assert m.startswith("graph TD")
    assert "alpha" in m and "beta" in m and "-->" in m

def test_build_structure_shape(tmp_path):
    # minimal config pointing at the fixture package
    cfg = {"project": {"package_root": str(FIX)},
           "structure": {"module_overview": True, "max_call_depth": 3,
                         "lenses": [{"name": "g", "entrypoints": ["mod_a.py:gamma"]}]}}
    out = es.build_structure(FIX.parents[0], cfg)  # repo_root arg unused for package path here
    assert out["schema_version"] == 1
    assert "module_overview" in out and out["lenses"][0]["name"] == "g"
    assert "g" in out["mermaid"] and "module_overview" in out["mermaid"]
```

- [ ] **Step 2: Run test to verify it fails**
Run: `$PY -m pytest tests/codemap/test_extract_structure.py -k "mermaid or build_structure" -v`
Expected: FAIL

- [ ] **Step 3: Implement (append to `extract_structure.py`):**
```python
import re, sys, datetime
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cm_common as cm

def _safe(node_id: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", node_id)

def to_mermaid(graph: dict, *, directed=True) -> str:
    lines = ["graph TD"]
    for n in graph["nodes"]:
        nid = _safe(n["id"]); label = n.get("label", n["id"])
        lines.append(f'  {nid}["{label}"]')
    arrow = "-->" if directed else "---"
    for e in graph["edges"]:
        lines.append(f'  {_safe(e["src"])} {arrow} {_safe(e["dst"])}')
    return "\n".join(lines)

def build_structure(repo_root: Path, cfg: dict) -> dict:
    pkg = Path(cfg["project"]["package_root"])
    if not pkg.is_absolute():
        pkg = Path(repo_root) / pkg
    sc = cfg["structure"]
    overview = build_module_overview(pkg) if sc.get("module_overview", True) else {"nodes": [], "edges": []}
    lenses, mermaid = [], {}
    if overview["nodes"]:
        mermaid["module_overview"] = to_mermaid(overview)
    for lc in sc.get("lenses", []):
        lens = build_lens(pkg, lc["entrypoints"], sc.get("max_call_depth", 3))
        lens["name"] = lc["name"]; lens["description"] = lc.get("description", "")
        lenses.append(lens)
        mermaid[lc["name"]] = to_mermaid(lens)
    return {"schema_version": cm.SCHEMA_VERSION,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_sha": cm.git_sha(repo_root) if (Path(repo_root) / ".git").exists() else "nogit",
            "package_root": str(pkg), "module_overview": overview,
            "lenses": lenses, "mermaid": mermaid}

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()
    cfg = cm.load_config(repo)
    out = build_structure(repo, cfg)
    sha = cm.git_sha(repo)
    d = cm.snapshot_dir(repo, cfg, sha)
    cm.write_json(d / "structure.json", out)
    print(d / "structure.json")

if __name__ == "__main__":
    main()
```
Note: drop the private `_node` AST objects before serializing — they are not JSON-serializable. Adjust `build_lens`/`build_symbol_table` callers in `build_structure` so emitted `nodes` never include `_node` (they already don't — `build_lens` copies only id/file/line/kind). Confirm `build_structure` output has no `_node` keys.

- [ ] **Step 4: Run test to verify it passes**
Run: `$PY -m pytest tests/codemap/test_extract_structure.py -v`
Expected: PASS (all structure tests)

- [ ] **Step 5: JSON-serializability guard test** (append + run):
```python
import json
def test_structure_json_serializable():
    cfg = {"project": {"package_root": str(FIX)},
           "structure": {"module_overview": True, "lenses": [{"name": "g", "entrypoints": ["mod_a.py:gamma"]}]}}
    json.dumps(es.build_structure(FIX.parents[0], cfg))  # must not raise
```
Run: `$PY -m pytest tests/codemap/test_extract_structure.py -k serializable -v` → PASS

- [ ] **Step 6: Commit**
```bash
git add docs/codemap/scripts/extract_structure.py tests/codemap/test_extract_structure.py
git commit -m "feat(codemap): mermaid emit + structure.json CLI"
```

---

### Task 6: `extract_state_facts.py` — git/phase/memory/job facts

**Files:**
- Create: `docs/codemap/scripts/extract_state_facts.py`
- Test: `tests/codemap/test_extract_state_facts.py`

**Interfaces:**
- Produces:
  - `vs_compare(repo_root, branch) -> dict` → `{"branch","ahead","behind","changed_files":[{"path","added","deleted"}]}` (uses `git rev-list --left-right --count` and `git diff --numstat`). If `branch` missing, `ahead/behind=None`, changed_files=[].
  - `recent_commits(repo_root, n=10) -> list` → `[{"sha","subject","date"}]`.
  - `phase_doc_headings(path) -> list[str]` → markdown `#`/`##`/`###` headings.
  - `build_state_facts(repo_root, cfg) -> dict` → full `state.facts.json` (spec §8), tolerating missing memory_dir/phase_docs/job glob.
  - CLI writes `state.facts.json` into the snapshot dir.

- [ ] **Step 1: Write failing test** `tests/codemap/test_extract_state_facts.py`:
```python
import subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/codemap/scripts"))
import extract_state_facts as sf

def _git(repo, *a): subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)

def make_repo(tmp_path):
    repo = tmp_path / "r"; repo.mkdir()
    _git(repo, "init", "-q", "-b", "master"); _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    (repo / "a.py").write_text("x=1\n"); _git(repo, "add", "."); _git(repo, "commit", "-qm", "init")
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "a.py").write_text("x=1\ny=2\n"); _git(repo, "add", "."); _git(repo, "commit", "-qm", "feat: add y")
    return repo

def test_vs_compare_counts_ahead(tmp_path):
    repo = make_repo(tmp_path)
    out = sf.vs_compare(repo, "master")
    assert out["ahead"] == 1 and out["behind"] == 0
    assert any(c["path"] == "a.py" for c in out["changed_files"])

def test_recent_commits(tmp_path):
    repo = make_repo(tmp_path)
    cs = sf.recent_commits(repo, n=5)
    assert cs[0]["subject"].startswith("feat: add y")

def test_phase_headings(tmp_path):
    p = tmp_path / "phase.md"; p.write_text("# A\nblah\n## B.1\n### deep\n")
    assert sf.phase_doc_headings(p) == ["# A", "## B.1", "### deep"]

def test_build_state_facts_tolerates_missing(tmp_path):
    repo = make_repo(tmp_path)
    cfg = {"state": {"compare_branch": "master", "phase_docs": ["nope.md"],
                     "memory_dir": "/does/not/exist"}, "project": {}}
    out = sf.build_state_facts(repo, cfg)
    assert out["schema_version"] == 1 and out["branch"] == "feature"
    assert out["memory_index"] == [] and out["phase_docs"] == []
```

- [ ] **Step 2: Run → FAIL.** `$PY -m pytest tests/codemap/test_extract_state_facts.py -v`

- [ ] **Step 3: Implement `extract_state_facts.py`:**
```python
"""Project-state FACTS only (no narrative). Stdlib + git."""
from __future__ import annotations
import subprocess, glob, datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cm_common as cm

def _run(repo, *a):
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout

def _branch_exists(repo, b):
    return subprocess.run(["git", "rev-parse", "--verify", "-q", b], cwd=repo,
                          capture_output=True).returncode == 0

def vs_compare(repo_root, branch) -> dict:
    if not branch or not _branch_exists(repo_root, branch):
        return {"branch": branch, "ahead": None, "behind": None, "changed_files": []}
    counts = _run(repo_root, "rev-list", "--left-right", "--count", f"{branch}...HEAD").split()
    behind, ahead = (int(counts[0]), int(counts[1])) if len(counts) == 2 else (None, None)
    changed = []
    for line in _run(repo_root, "diff", "--numstat", f"{branch}...HEAD").splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            add, dele, path = parts
            changed.append({"path": path, "added": None if add == "-" else int(add),
                            "deleted": None if dele == "-" else int(dele)})
    return {"branch": branch, "ahead": ahead, "behind": behind, "changed_files": changed}

def recent_commits(repo_root, n=10) -> list:
    fmt = "%h\x1f%s\x1f%cI"
    out = _run(repo_root, "log", f"-{n}", f"--pretty={fmt}")
    res = []
    for line in out.splitlines():
        sha, subj, date = line.split("\x1f")
        res.append({"sha": sha, "subject": subj, "date": date})
    return res

def phase_doc_headings(path) -> list:
    p = Path(path)
    if not p.exists():
        return []
    return [ln.rstrip() for ln in p.read_text().splitlines() if ln.lstrip().startswith("#")]

def _memory_index(memory_dir) -> list:
    d = Path(memory_dir) if memory_dir else None
    if not d or not d.exists():
        return []
    out = []
    for f in sorted(d.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        desc = ""
        for ln in f.read_text().splitlines():
            if ln.startswith("description:"):
                desc = ln.split(":", 1)[1].strip(); break
        out.append({"name": f.stem, "desc": desc})
    return out

def _recent_jobs(job_glob) -> list:
    if not job_glob:
        return []
    files = sorted(glob.glob(job_glob, recursive=True), key=lambda f: Path(f).stat().st_mtime if Path(f).exists() else 0)[-10:]
    return [{"log": f} for f in files]  # jobid/state enrichment is the skill's job

def build_state_facts(repo_root, cfg) -> dict:
    st = cfg.get("state", {})
    branch = _run(repo_root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    phase = []
    for pd in st.get("phase_docs", []):
        p = Path(repo_root) / pd
        if p.exists():
            phase.append({"path": pd, "headings": phase_doc_headings(p)})
    return {"schema_version": cm.SCHEMA_VERSION,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_sha": cm.git_sha(repo_root), "branch": branch,
            "vs_compare_branch": vs_compare(repo_root, st.get("compare_branch")),
            "recent_commits": recent_commits(repo_root, 10),
            "phase_docs": phase, "memory_index": _memory_index(st.get("memory_dir")),
            "recent_jobs": _recent_jobs(st.get("job_log_glob"))}

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); args = ap.parse_args(argv)
    repo = Path(args.repo).resolve(); cfg = cm.load_config(repo)
    out = build_state_facts(repo, cfg)
    d = cm.snapshot_dir(repo, cfg, cm.git_sha(repo))
    cm.write_json(d / "state.facts.json", out); print(d / "state.facts.json")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run → PASS.** `$PY -m pytest tests/codemap/test_extract_state_facts.py -v`

- [ ] **Step 5: Commit**
```bash
git add docs/codemap/scripts/extract_state_facts.py tests/codemap/test_extract_state_facts.py
git commit -m "feat(codemap): state-facts extractor (git/phase/memory/jobs)"
```

---

### Task 7: `diff_snapshots.py` — structural + state delta

**Files:**
- Create: `docs/codemap/scripts/diff_snapshots.py`
- Test: `tests/codemap/test_diff_snapshots.py`

**Interfaces:**
- Consumes: `structure.json` + `state.facts.json` shapes (Tasks 5-6).
- Produces:
  - `diff_structure(old: dict, new: dict) -> dict` → `{"added_nodes","removed_nodes","added_edges","removed_edges","lenses_changed"}`. Nodes compared by `id` within each lens + module_overview; a lens is "changed" if its node/edge sets differ.
  - `diff_state(old: dict, new: dict) -> dict` → `{"new_commits":[sha...], "changed_files_since":[path...]}`.
  - `build_change(old_dir: Path, new_dir: Path) -> dict` → full `change.json` (spec §8) with empty `summary` (skill fills it).
  - CLI `python diff_snapshots.py --new <dir> [--old <dir>]`; default old = previous snapshot by mtime.

- [ ] **Step 1: Write failing test** `tests/codemap/test_diff_snapshots.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/codemap/scripts"))
import diff_snapshots as ds

def test_diff_structure_detects_added_edge():
    old = {"module_overview": {"nodes": [{"id": "a"}], "edges": []},
           "lenses": [{"name": "L", "nodes": [{"id": "a"}], "edges": []}]}
    new = {"module_overview": {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"src": "a", "dst": "b"}]},
           "lenses": [{"name": "L", "nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"src": "a", "dst": "b"}]}]}
    d = ds.diff_structure(old, new)
    assert "b" in d["added_nodes"] and ["a", "b"] in [list(e) for e in d["added_edges"]]
    assert "L" in d["lenses_changed"]

def test_diff_state_new_commits():
    old = {"recent_commits": [{"sha": "111", "subject": "x"}],
           "vs_compare_branch": {"changed_files": [{"path": "a.py"}]}}
    new = {"recent_commits": [{"sha": "222", "subject": "y"}, {"sha": "111", "subject": "x"}],
           "vs_compare_branch": {"changed_files": [{"path": "a.py"}, {"path": "b.py"}]}}
    d = ds.diff_state(old, new)
    assert d["new_commits"] == ["222"]
    assert "b.py" in d["changed_files_since"]
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `diff_snapshots.py`:**
```python
"""Pure JSON diff over two snapshot dirs. Skill fills the human summary."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cm_common as cm

def _node_ids(graph): return {n["id"] for n in graph.get("nodes", [])}
def _edge_set(graph): return {(e["src"], e["dst"]) for e in graph.get("edges", [])}

def diff_structure(old, new) -> dict:
    o_nodes = _node_ids(old.get("module_overview", {})); n_nodes = _node_ids(new.get("module_overview", {}))
    o_edges = _edge_set(old.get("module_overview", {})); n_edges = _edge_set(new.get("module_overview", {}))
    old_lenses = {l["name"]: l for l in old.get("lenses", [])}
    new_lenses = {l["name"]: l for l in new.get("lenses", [])}
    for name, l in new_lenses.items():
        n_nodes |= _node_ids(l); n_edges |= _edge_set(l)
    for name, l in old_lenses.items():
        o_nodes |= _node_ids(l); o_edges |= _edge_set(l)
    changed = []
    for name in sorted(set(old_lenses) | set(new_lenses)):
        ol = old_lenses.get(name, {}); nl = new_lenses.get(name, {})
        if _node_ids(ol) != _node_ids(nl) or _edge_set(ol) != _edge_set(nl):
            changed.append(name)
    return {"added_nodes": sorted(n_nodes - o_nodes), "removed_nodes": sorted(o_nodes - n_nodes),
            "added_edges": sorted(list(e) for e in (n_edges - o_edges)),
            "removed_edges": sorted(list(e) for e in (o_edges - n_edges)),
            "lenses_changed": changed}

def diff_state(old, new) -> dict:
    old_shas = {c["sha"] for c in old.get("recent_commits", [])}
    new_commits = [c["sha"] for c in new.get("recent_commits", []) if c["sha"] not in old_shas]
    old_files = {f["path"] for f in old.get("vs_compare_branch", {}).get("changed_files", [])}
    new_files = {f["path"] for f in new.get("vs_compare_branch", {}).get("changed_files", [])}
    return {"new_commits": new_commits, "changed_files_since": sorted(new_files - old_files)}

def _load(d, name):
    p = Path(d) / name
    return json.loads(p.read_text()) if p.exists() else {}

def build_change(old_dir, new_dir) -> dict:
    o_struct, n_struct = _load(old_dir, "structure.json"), _load(new_dir, "structure.json")
    o_state, n_state = _load(old_dir, "state.facts.json"), _load(new_dir, "state.facts.json")
    return {"schema_version": cm.SCHEMA_VERSION,
            "from_sha": o_struct.get("git_sha") or (Path(old_dir).name if old_dir else None),
            "to_sha": n_struct.get("git_sha") or Path(new_dir).name,
            "structure_delta": diff_structure(o_struct, n_struct),
            "state_delta": diff_state(o_state, n_state), "summary": ""}

def _previous_dir(snap_root: Path, new_dir: Path):
    dirs = [p for p in snap_root.iterdir() if p.is_dir() and not p.is_symlink() and p != new_dir]
    dirs.sort(key=lambda p: p.stat().st_mtime)
    return dirs[-1] if dirs else None

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); ap.add_argument("--new", required=True); ap.add_argument("--old", default=None)
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve(); cfg = cm.load_config(repo)
    new_dir = Path(args.new); snap_root = Path(repo) / cfg["snapshots"]["dir"]
    old_dir = Path(args.old) if args.old else _previous_dir(snap_root, new_dir)
    out = build_change(old_dir, new_dir) if old_dir else {
        "schema_version": cm.SCHEMA_VERSION, "from_sha": None, "to_sha": new_dir.name,
        "structure_delta": {}, "state_delta": {}, "summary": "first snapshot — no baseline"}
    cm.write_json(new_dir / "change.json", out); print(new_dir / "change.json")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run → PASS.** `$PY -m pytest tests/codemap/test_diff_snapshots.py -v`

- [ ] **Step 5: Commit**
```bash
git add docs/codemap/scripts/diff_snapshots.py tests/codemap/test_diff_snapshots.py
git commit -m "feat(codemap): snapshot differ (structural + state delta)"
```

---

### Task 8: `render_html.py` — fragments + fused atlas

**Files:**
- Create: `docs/codemap/scripts/render_html.py`
- Test: `tests/codemap/test_render_html.py`

**Interfaces:**
- Consumes: `structure.json`, `state.json`, `change.json`.
- Produces:
  - `mermaid_block(mermaid_text: str) -> str` → `<pre class="mermaid">…</pre>`.
  - `structure_fragment(structure: dict) -> str` — one collapsible section per lens + the module overview, each a mermaid block + an unresolved-entrypoint warning chip if present.
  - `state_fragment(state: dict) -> str` — renders the skill's `narrative` markdown-ish into HTML plus a compact facts table.
  - `change_fragment(change: dict) -> str` — the skill's `summary` + added/removed lists.
  - `render_atlas(structure, state, change) -> str` — full self-contained HTML doc: `<head>` loads Mermaid.js from CDN + `mermaid.initialize`, a top "you are here" banner (state narrative), then structure + change sections.
  - CLI: `--atlas <snapshot_dir>` writes `atlas.html` to `docs/codemap/atlas.html`; `--fragments <snapshot_dir>` writes the three `*.fragment.html`.

- [ ] **Step 1: Write failing test** `tests/codemap/test_render_html.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/codemap/scripts"))
import render_html as rh

def test_mermaid_block_wraps():
    assert 'class="mermaid"' in rh.mermaid_block("graph TD\n a-->b")

def test_structure_fragment_shows_unresolved():
    s = {"mermaid": {"module_overview": "graph TD", "L": "graph TD"},
         "lenses": [{"name": "L", "description": "d", "unresolved_entrypoints": ["x.py:gone"]}]}
    html = rh.structure_fragment(s)
    assert "gone" in html and "mermaid" in html

def test_render_atlas_is_selfcontained():
    s = {"mermaid": {"module_overview": "graph TD"}, "lenses": []}
    st = {"narrative": "You are **here**.", "branch": "feature", "vs_compare_branch": {"ahead": 1, "behind": 0}}
    ch = {"summary": "added a lens", "structure_delta": {"added_nodes": ["b"]}, "state_delta": {"new_commits": ["222"]}}
    html = rh.render_atlas(s, st, ch)
    assert "mermaid" in html.lower() and "You are" in html and "<html" in html.lower()
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `render_html.py`** (keep CSS minimal; Mermaid from CDN — note this HTML needs network to render diagrams, acceptable since opened on the user's Mac):
```python
"""Render snapshot JSON into self-contained HTML (client-side Mermaid)."""
from __future__ import annotations
import html, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cm_common as cm

CDN = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"

def mermaid_block(mermaid_text: str) -> str:
    return f'<pre class="mermaid">\n{html.escape(mermaid_text)}\n</pre>'

def _chip(text, kind="warn"):
    return f'<span class="chip {kind}">{html.escape(text)}</span>'

def structure_fragment(structure: dict) -> str:
    out = ["<section><h2>Structural map</h2>"]
    mer = structure.get("mermaid", {})
    if "module_overview" in mer:
        out.append("<details open><summary>Module overview</summary>")
        out.append(mermaid_block(mer["module_overview"]) + "</details>")
    for lens in structure.get("lenses", []):
        name = lens["name"]
        out.append(f"<details open><summary>{html.escape(name)} — {html.escape(lens.get('description',''))}</summary>")
        for ep in lens.get("unresolved_entrypoints", []):
            out.append(_chip(f"unresolved: {ep}"))
        if name in mer:
            out.append(mermaid_block(mer[name]))
        out.append("</details>")
    out.append("</section>")
    return "\n".join(out)

def _md_lite(text: str) -> str:
    t = html.escape(text or "")
    import re
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    return "<br>".join(t.splitlines())

def state_fragment(state: dict) -> str:
    vs = state.get("vs_compare_branch", {})
    facts = (f'branch <code>{html.escape(str(state.get("branch")))}</code> · '
             f'ahead {vs.get("ahead")} / behind {vs.get("behind")}')
    return (f'<section class="youarehere"><h2>You are here</h2>'
            f'<p class="facts">{facts}</p><div>{_md_lite(state.get("narrative",""))}</div></section>')

def change_fragment(change: dict) -> str:
    sd = change.get("structure_delta", {}); std = change.get("state_delta", {})
    parts = [f'<section><h2>Since last snapshot</h2><p>{_md_lite(change.get("summary",""))}</p><ul>']
    parts.append(f'<li>+nodes: {html.escape(str(sd.get("added_nodes", [])))}</li>')
    parts.append(f'<li>-nodes: {html.escape(str(sd.get("removed_nodes", [])))}</li>')
    parts.append(f'<li>new commits: {html.escape(str(std.get("new_commits", [])))}</li>')
    parts.append("</ul></section>")
    return "\n".join(parts)

_CSS = """body{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
h1{border-bottom:2px solid #444}.youarehere{background:#f5f8ff;border-left:4px solid #3b6;padding:1rem;border-radius:6px}
.chip{display:inline-block;background:#fde;color:#900;border-radius:10px;padding:1px 8px;margin:2px;font-size:12px}
details{margin:.5rem 0;border:1px solid #eee;border-radius:6px;padding:.5rem}summary{cursor:pointer;font-weight:600}
.facts{color:#555;font-size:13px}code{background:#eef;padding:0 4px;border-radius:3px}"""

def render_atlas(structure: dict, state: dict, change: dict) -> str:
    body = state_fragment(state) + structure_fragment(structure)
    if change:
        body += change_fragment(change)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Codemap Atlas</title><style>{_CSS}</style>
<script type="module">import mermaid from "{CDN}";mermaid.initialize({{startOnLoad:true,securityLevel:'loose'}});</script>
</head><body><h1>Codemap Atlas</h1>{body}</body></html>"""

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); ap.add_argument("--atlas", default=None)
    ap.add_argument("--fragments", default=None); args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()
    def _load(d, n):
        p = Path(d) / n; return json.loads(p.read_text()) if p.exists() else {}
    if args.fragments:
        d = Path(args.fragments)
        (d / "structure.fragment.html").write_text(structure_fragment(_load(d, "structure.json")))
        (d / "state.fragment.html").write_text(state_fragment(_load(d, "state.json")))
        (d / "change.fragment.html").write_text(change_fragment(_load(d, "change.json")))
        print(d)
    if args.atlas:
        d = Path(args.atlas)
        html_doc = render_atlas(_load(d, "structure.json"), _load(d, "state.json"), _load(d, "change.json"))
        out = repo / "docs/codemap/atlas.html"; out.write_text(html_doc); print(out)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run → PASS.** `$PY -m pytest tests/codemap/test_render_html.py -v`

- [ ] **Step 5: Full suite green.** `$PY -m pytest tests/codemap/ -v` → all PASS

- [ ] **Step 6: Commit**
```bash
git add docs/codemap/scripts/render_html.py tests/codemap/test_render_html.py
git commit -m "feat(codemap): HTML fragment + atlas renderer (client-side mermaid)"
```

---

### Task 9: `codemap-structure` skill

**Files:**
- Create: `<skills-dir>/codemap-structure/SKILL.md`

(`<skills-dir>` = where the user's other custom skills live; confirm at execution time — likely `~/.claude/skills/` or a plugin under `~/.claude/plugins/`. The four skills are repo-agnostic; only the config is repo-specific.)

**Interfaces:**
- Consumes: `extract_structure.py`, `render_html.py`.
- Produces: a `structure.json` + `structure.fragment.html` in the current snapshot dir; a short chat summary.

- [ ] **Step 1: Write `SKILL.md`:**
```markdown
---
name: codemap-structure
description: Generate the structural code map (module overview + curated lens call-graphs) for the current repo, writing a SHA-keyed snapshot and an HTML fragment. Use when the user wants to see/refresh how files and functions link, or as part of building the codemap atlas. Reads docs/codemap/codemap.config.yaml.
---

# codemap-structure

Generate the structural map from `docs/codemap/codemap.config.yaml`.

## Steps
1. Resolve the python: read `project.python` from the config (fallback: `python3`).
2. Run the extractor:
   `<python> docs/codemap/scripts/extract_structure.py --repo .`
   It writes `structure.json` into `docs/codemap/snapshots/<sha>/` and prints the path.
3. Render the fragment:
   `<python> docs/codemap/scripts/render_html.py --repo . --fragments docs/codemap/snapshots/<sha>`
4. Read `structure.json`. If `unresolved_entrypoints` is non-empty for any lens, tell the user
   plainly — that means a config entrypoint was renamed/moved (the map has drifted), and offer to
   fix the config lens.
5. Report: how many lenses, node/edge counts per lens, and the snapshot path. Do NOT invent
   structure the JSON does not contain.

## Notes
- This skill emits FACTS only (no narrative). Pure mechanical run + sanity check.
- If the config is missing, offer to scaffold one from docs/superpowers/specs §7.
```

- [ ] **Step 2: Manual verify** — invoke the skill on this repo; confirm a `structure.json` appears under `docs/codemap/snapshots/<sha>/` with the 4 dinov3 lenses and non-empty mermaid. Eyeball one lens for sanity (e.g. `loss-forward` should include `forward_backward`).

- [ ] **Step 3: Commit**
```bash
git add <skills-dir>/codemap-structure/SKILL.md
git commit -m "feat(codemap): codemap-structure skill"
```

---

### Task 10: `codemap-state` skill (the narrative synthesizer)

**Files:**
- Create: `<skills-dir>/codemap-state/SKILL.md`

**Interfaces:**
- Consumes: `extract_state_facts.py`. Produces `state.facts.json` (script) then `state.json` (facts + `narrative`) + `state.fragment.html`.

- [ ] **Step 1: Write `SKILL.md`:**
```markdown
---
name: codemap-state
description: Generate the project-state "you are here" map for the current repo — a narrative synthesized from git, phase docs, and memory that re-seats the mental model (current phase, what just shipped, what we're trying now and why, branch-vs-master, live vs closed experiments). Use when the user feels lost / wants to re-orient, or as part of the codemap atlas. Reads docs/codemap/codemap.config.yaml.
---

# codemap-state

## Steps
1. Run the facts extractor:
   `<python> docs/codemap/scripts/extract_state_facts.py --repo .`
   → writes `state.facts.json` into the snapshot dir.
2. Read `state.facts.json`. Then READ the bodies it points to (the phase_docs paths, and the
   referenced memory files under memory_dir whose desc looks current). Do not rely on headings alone.
3. **Write the narrative** — a few tight paragraphs aimed at someone re-orienting after time away:
   - Where we are: current phase + one-line project framing.
   - What just shipped (last few commits, in plain language — not raw subjects).
   - What we're trying NOW and WHY (pull from the current_goal_anchor / latest phase section).
   - How this branch differs from compare_branch (use ahead/behind + changed_files).
   - Which experiments are live vs closed (from memory/phase docs).
   Be concrete and honest; cite job ids / file paths. This is the anti-cognitive-surrender core —
   write it so the user rebuilds the map in their head, not just skims links.
4. Write `state.json` = the facts object plus a top-level `"narrative"` string (your markdown).
   Use: read state.facts.json, add the key, write back with the same path.
5. Render: `<python> docs/codemap/scripts/render_html.py --repo . --fragments docs/codemap/snapshots/<sha>`
6. Show the user the narrative in chat too.

## Guardrails
- Verify before claiming (repo CLAUDE.md): if you state a job passed/failed or a file changed,
  it must be grounded in the facts JSON or a file you actually read.
- Narrative only — never edit code or configs from this skill.
```

- [ ] **Step 2: Manual verify** — run on this repo; confirm `state.json` has a `narrative` that correctly names the current phase (Phase 7.8) and branch (`exp/ddp-fullgraph-head-static-shapes`) and references real recent commits.

- [ ] **Step 3: Commit**
```bash
git add <skills-dir>/codemap-state/SKILL.md
git commit -m "feat(codemap): codemap-state narrative skill"
```

---

### Task 11: `codemap-diff` skill

**Files:**
- Create: `<skills-dir>/codemap-diff/SKILL.md`

**Interfaces:**
- Consumes: `diff_snapshots.py`. Produces `change.json` (delta + skill `summary`) + `change.fragment.html`.

- [ ] **Step 1: Write `SKILL.md`:**
```markdown
---
name: codemap-diff
description: Show what changed in the codebase map since the last codemap snapshot or since a given commit — structural deltas (added/removed call edges, changed lenses) plus state deltas (new commits, newly changed files). Use when the user asks "what changed since last time / since commit X" in terms of the codemap. Reads docs/codemap/codemap.config.yaml.
---

# codemap-diff

## Steps
1. Identify the current snapshot dir (latest under docs/codemap/snapshots, or the one just produced).
   The new snapshot must already have structure.json + state.facts.json (run codemap-structure /
   codemap-state first, or codemap-atlas which does both).
2. Run the differ:
   `<python> docs/codemap/scripts/diff_snapshots.py --repo . --new docs/codemap/snapshots/<new_sha>`
   (add `--old docs/codemap/snapshots/<sha>` to diff against a specific commit's snapshot.)
   → writes change.json with an empty `summary`.
3. Read change.json. **Write the `summary`**: 2-5 sentences in plain language — what structurally
   moved (which lenses, which call edges appeared/vanished) and what that implies about the work
   (tie to the new_commits). If `lenses_changed` includes a hot path, say which experiment/commit
   likely caused it.
4. Write the summary back into change.json (read, set "summary", write).
5. Render the fragment and report the diff to the user.

## Notes
- First run has no baseline — say so, don't fabricate a diff.
- The structural diff is name-based and approximate (orientation aid); don't over-claim a removed
  edge means dead code — say "the map no longer follows X→Y", and suggest verifying.
```

- [ ] **Step 2: Manual verify** — make a snapshot, add a trivial commit + a new lens edge, snapshot again, run diff; confirm `change.json` reports the new commit and the changed lens.

- [ ] **Step 3: Commit**
```bash
git add <skills-dir>/codemap-diff/SKILL.md
git commit -m "feat(codemap): codemap-diff skill"
```

---

### Task 12: `codemap-atlas` skill (orchestrator) + prune/symlink

**Files:**
- Create: `<skills-dir>/codemap-atlas/SKILL.md`

**Interfaces:**
- Consumes: all three skills + `render_html.py --atlas`, `cm_common.prune_snapshots`, `update_latest_symlink`.
- Produces: `docs/codemap/atlas.html`, pruned snapshots, updated `latest` symlink.

- [ ] **Step 1: Write `SKILL.md`:**
```markdown
---
name: codemap-atlas
description: Build the full codebase ATLAS in one shot — runs the structural map, the "you are here" state narrative, and the diff-since-last-run, then fuses them into a single self-contained HTML page (docs/codemap/atlas.html) you open in a browser. Use when the user wants the whole visual orientation refreshed, e.g. after landing changes or starting a session. Reads docs/codemap/codemap.config.yaml.
---

# codemap-atlas

One-command orientation. Orchestrates the other three codemap skills, then fuses.

## Steps
1. Run codemap-structure (produces structure.json + fragment in the current snapshot dir).
2. Run codemap-state (produces state.json with narrative + fragment). This includes the
   narrative-writing step — do it properly, that's the core value.
3. Run codemap-diff against the previous snapshot (produces change.json with summary). If no
   previous snapshot exists, note "first atlas — no diff".
4. Fuse: `<python> docs/codemap/scripts/render_html.py --repo . --atlas docs/codemap/snapshots/<sha>`
   → writes docs/codemap/atlas.html.
5. Prune + symlink: run a tiny python snippet using cm_common:
   `<python> -c "import sys; sys.path.insert(0,'docs/codemap/scripts'); import cm_common as cm, yaml, pathlib;
   repo=pathlib.Path('.').resolve(); cfg=cm.load_config(repo);
   print('removed', cm.prune_snapshots(repo,cfg));
   cm.update_latest_symlink(repo/cfg['snapshots']['dir'], cm.git_sha(repo))"`
6. Tell the user: atlas written to docs/codemap/atlas.html (open in a browser on the Mac), plus a
   2-3 line spoken summary of the "you are here" narrative and the biggest change since last time.
7. Offer to commit the snapshot + atlas (don't auto-commit unless the user asked).

## Notes
- This is the skill to reach for first when re-orienting. The other three are for targeted refreshes.
```

- [ ] **Step 2: End-to-end manual verify on dinov3:**
  - Invoke `codemap-atlas`.
  - Confirm `docs/codemap/atlas.html` exists and is self-contained (opens, diagrams render in a browser).
  - Confirm the snapshot dir has all of: structure.json, state.facts.json, state.json, change.json, *.fragment.html.
  - Confirm `snapshots/latest` symlink points at the current sha.
  - Sanity-read the atlas: the four lenses render, the "you are here" narrative correctly states Phase 7.8 + the branch, and the diff panel is sane.

- [ ] **Step 3: Commit**
```bash
git add <skills-dir>/codemap-atlas/SKILL.md
git commit -m "feat(codemap): codemap-atlas orchestrator skill"
```

- [ ] **Step 4: First real atlas snapshot (optional, ask user):**
```bash
git add docs/codemap/atlas.html docs/codemap/snapshots
git commit -m "chore(codemap): first atlas snapshot for dinov3 Phase 7.8"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** every design-spec section maps to a task — D1 snapshots (Tasks 2,5,6,12), D2 four skills (Tasks 9-12), D3 diff-over-snapshots (Task 7), D4 curated lenses (Tasks 1,4), D5 in-repo git-tracked + prune (Tasks 1,2,12), D6 narrative (Task 10), D7 hybrid scripts-vs-skill (scripts Tasks 2-8, skills 9-12), D8 mermaid client-side (Tasks 5,8). Config §7 → Task 1. JSON contracts §8 → Tasks 5,6,7. Error handling §10 → unresolved_entrypoints (Tasks 4,9), missing-baseline (Task 7,11), tolerant facts (Task 6). Testing §11 → fixture + per-script tests. Reuse §12 → repo-agnostic skills + config (Tasks 9-12).

**Placeholder scan:** no TBD/TODO; every code step has complete code; `<skills-dir>` and `<sha>`/`<python>` are intentional runtime-resolved values, called out explicitly, not placeholders for logic.

**Type consistency:** `build_module_overview`/`build_lens`/`build_structure`/`to_mermaid` signatures match across Tasks 3-5 and their use in 8; `structure.json`/`state.facts.json`/`change.json` field names match between producers (5,6,7) and the differ (7) and renderer (8). `cm_common` names (`git_sha`, `snapshot_dir`, `write_json`, `prune_snapshots`, `update_latest_symlink`) used consistently.

**Known approximation (intentional, documented):** call resolution is name-based/best-effort — surfaced to the user in the diff skill (Task 11) and design spec §9. Acceptable for an orientation aid; not a soundness-critical analysis.
