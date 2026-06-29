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
