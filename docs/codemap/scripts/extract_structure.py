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
