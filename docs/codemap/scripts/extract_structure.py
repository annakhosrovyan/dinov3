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
