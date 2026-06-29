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
