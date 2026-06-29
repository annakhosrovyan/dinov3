from pathlib import Path
import sys
import json
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

def test_structure_json_serializable():
    cfg = {"project": {"package_root": str(FIX)},
           "structure": {"module_overview": True, "lenses": [{"name": "g", "entrypoints": ["mod_a.py:gamma"]}]}}
    json.dumps(es.build_structure(FIX.parents[0], cfg))  # must not raise

def test_lens_whole_module_entrypoint():
    """A colon-less entrypoint 'relpath.py' should seed ALL top-level defs as BFS starts."""
    lens = es.build_lens(FIX, ["mod_a.py"], max_depth=3)
    # mod_a.py is not unresolved — it expands to alpha + gamma
    assert "mod_a.py" not in lens["unresolved_entrypoints"]
    node_ids = {n["id"] for n in lens["nodes"]}
    # alpha and gamma are top-level defs in mod_a; beta is reached via call following
    assert "alpha" in node_ids
    assert "gamma" in node_ids
    assert "beta" in node_ids
    edges = {(e["src"], e["dst"]) for e in lens["edges"]}
    assert ("gamma", "alpha") in edges
    assert ("alpha", "beta") in edges


def test_lens_disambiguates_collided_method():
    """A 'relpath.py:Class.method' entrypoint with a name collision resolves unambiguously."""
    # mod_c.py has Engine.run and Decoy.run — 'run' is a collided simple name.
    lens = es.build_lens(FIX, ["mod_c.py:Engine.run"], max_depth=3)
    # Must resolve — not land in unresolved_entrypoints
    assert "mod_c.py:Engine.run" not in lens["unresolved_entrypoints"]
    node_ids = {n["id"] for n in lens["nodes"]}
    # Engine.run is the seed; Engine.helper is reached via self.helper() call
    assert "Engine.run" in node_ids
    assert "Engine.helper" in node_ids
    # Decoy.run was NOT selected — should not appear
    assert "Decoy.run" not in node_ids
    edges = {(e["src"], e["dst"]) for e in lens["edges"]}
    # Engine.run calls self.helper() → Engine.helper
    assert ("Engine.run", "Engine.helper") in edges


def test_lens_expands_through_underscore_function():
    """BFS must expand underscore-prefixed functions as callees, not just seeds.

    epsilon -> _delta -> beta.
    The bug skipped _delta in qname_table, so beta was never reached.
    """
    lens = es.build_lens(FIX, ["mod_b.py:epsilon"], max_depth=3)
    # entrypoint must resolve
    assert "mod_b.py:epsilon" not in lens["unresolved_entrypoints"]
    node_ids = {n["id"] for n in lens["nodes"]}
    # All three nodes must appear
    assert "epsilon" in node_ids, f"epsilon missing from {node_ids}"
    assert "_delta" in node_ids, f"_delta missing from {node_ids}"
    assert "beta" in node_ids, f"beta missing — _delta subtree not expanded: {node_ids}"
    edges = {(e["src"], e["dst"]) for e in lens["edges"]}
    assert ("epsilon", "_delta") in edges, f"epsilon->_delta edge missing: {edges}"
    assert ("_delta", "beta") in edges, f"_delta->beta edge missing (bug): {edges}"
