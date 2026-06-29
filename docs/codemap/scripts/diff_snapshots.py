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
