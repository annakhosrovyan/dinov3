"""Structural extraction via stdlib ast. No external binaries. Approximate name-based
call resolution (orientation aid, not a sound static analysis)."""
from __future__ import annotations
import ast
from pathlib import Path
import re, sys, datetime
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cm_common as cm

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
                # For "from X import Y", try both X and X.Y
                targets = [node.module]
                for alias in node.names:
                    targets.append(f"{node.module}.{alias.name}")
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

def _resolve_entrypoint(package_root, table, ep) -> list[str]:
    """Return a list of start **qualnames** for the entrypoint.

    Returns length-1 list for 'relpath.py:Qualname' form, length-N for whole-module form,
    empty list if unresolved.

    Disambiguation for 'relpath.py:Qualname':
    - Collect all candidates for the simple name (primary table entry + collision list).
    - If the relpath is provided, filter by file suffix match AND qualname equality.
      If exactly one candidate matches, use it.
    - Otherwise fall back: if simple name is unique (not collided), use it.
    - Else unresolved.
    """
    collisions = table.get("_collisions", {})
    if ":" in ep:
        relpath, qual = ep.split(":", 1)
        simple = qual.split(".")[-1]
        # Collect ALL candidates.  Note: collisions[simple] already contains the
        # first entry (table[simple]) as its first element, so we must not double-count.
        candidates: list
        if simple in collisions:
            candidates = collisions[simple]   # all candidates, including the first one
        elif simple in table:
            candidates = [table[simple]]
        else:
            candidates = []
        # Try file+qualname disambiguation using component-boundary suffix match
        relpath_norm = relpath.replace("\\", "/")
        matched = [
            c for c in candidates
            if ("/" + c["file"].replace("\\", "/")).endswith("/" + relpath_norm)
            and c["qualname"] == qual
        ]
        if len(matched) == 1:
            return [matched[0]["qualname"]]
        # Fallback: unique simple name (not collided)
        if simple in table and simple not in collisions:
            return [table[simple]["qualname"]]
        return []
    # Whole-module form: 'relpath.py' — expand to all top-level defs in that file.
    # Normalize separators and do a component-boundary suffix match.
    relpath = ep.replace("\\", "/")
    results = []
    for simple, entry in table.items():
        if simple.startswith("_"):
            continue  # skip _collisions and any private sentinel keys
        if simple in collisions:
            continue
        # Top-level only: no "." in qualname
        if "." in entry.get("qualname", simple):
            continue
        filepath = entry.get("file", "").replace("\\", "/")
        if ("/" + filepath).endswith("/" + relpath):
            results.append(entry["qualname"])
    return results

def build_lens(package_root, entrypoints, max_depth=3) -> dict:
    table = build_symbol_table(Path(package_root))
    collisions = table.get("_collisions", {})
    # Build qualname → entry reverse lookup (covers both unique entries and collisions).
    qname_table: dict = {}
    for k, v in table.items():
        if k == "_collisions":
            continue
        qname_table[v["qualname"]] = v
    for entries in collisions.values():
        for e in entries:
            qname_table[e["qualname"]] = e
    start, unresolved = [], []
    for ep in entrypoints:
        qnames = _resolve_entrypoint(package_root, table, ep)
        if qnames:
            start.extend(qnames)
        else:
            unresolved.append(ep)
    nodes, edges, seen = {}, set(), set()
    frontier = [(q, 0) for q in start]
    for q in start:
        e = qname_table.get(q)
        if e:
            nodes[q] = {"id": q, "file": e["file"], "line": e["line"], "kind": e["kind"]}
    while frontier:
        qname, depth = frontier.pop(0)
        if qname in seen or depth >= max_depth:
            continue
        seen.add(qname)
        entry = qname_table.get(qname)
        if not entry:
            continue
        for callee in _called_names(entry["_node"]):
            # Callee resolution is best-effort by simple name: skip collisions.
            if callee in table and callee not in collisions:
                ce = table[callee]
                callee_qname = ce["qualname"]
                if callee_qname == qname:
                    continue  # skip self-calls
                if callee_qname not in nodes:
                    nodes[callee_qname] = {"id": callee_qname, "file": ce["file"],
                                           "line": ce["line"], "kind": ce["kind"]}
                edges.add((qname, callee_qname))
                frontier.append((callee_qname, depth + 1))
    return {"entrypoints": entrypoints,
            "nodes": sorted(nodes.values(), key=lambda n: n["id"]),
            "edges": [{"src": s, "dst": d, "kind": "call"} for s, d in sorted(edges)],
            "unresolved_entrypoints": unresolved}

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
