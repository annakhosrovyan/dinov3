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
