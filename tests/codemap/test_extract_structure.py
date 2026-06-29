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
