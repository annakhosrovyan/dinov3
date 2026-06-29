import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/codemap/scripts"))
import render_html as rh


def test_svg_graph_renders_nodes_and_edges():
    g = {"nodes": [{"id": "alpha", "kind": "function", "file": "m.py"},
                   {"id": "beta", "kind": "function", "file": "m.py"}],
         "edges": [{"src": "alpha", "dst": "beta", "kind": "call"}]}
    svg = rh.svg_graph(g, "L")
    assert "<svg" in svg and "alpha" in svg and "beta" in svg
    assert 'class="edge"' in svg and 'data-src="alpha"' in svg      # real edge drawn
    assert "<pre" not in svg and "mermaid" not in svg.lower()        # no mermaid text


def test_long_label_is_not_truncated():
    long_id = "SSLMetaArch.get_student_output_with_a_really_long_method_name"
    g = {"nodes": [{"id": long_id, "kind": "method", "file": "m.py"}], "edges": []}
    svg = rh.svg_graph(g, "L")
    # the full identifier survives (possibly across two wrapped tspans), no ellipsis
    assert "..." not in svg
    assert all(part in svg for part in long_id.split("."))


def test_nodes_coloured_by_group():
    # two classes -> two distinct stroke colours (smart colouring)
    g = {"nodes": [{"id": "Foo.a", "kind": "method", "file": "m.py"},
                   {"id": "Bar.b", "kind": "method", "file": "m.py"}], "edges": []}
    svg = rh.svg_graph(g, "L")
    strokes = set(rh._color(k)[1] for k in ("Foo", "Bar"))
    assert all(s in svg for s in strokes)
    assert 'class="legend"' in svg          # legend maps colour -> group


def test_structure_fragment_shows_unresolved():
    s = {"module_overview": {"nodes": [], "edges": []},
         "lenses": [{"name": "L", "description": "d", "nodes": [], "edges": [],
                     "unresolved_entrypoints": ["x.py:gone"]}]}
    result = rh.structure_fragment(s)
    assert "gone" in result and 'class="chip' in result  # unresolved entrypoint surfaced as a chip


def test_render_markdown_handles_headings_bold_code_lists():
    md = "## Section\nSome **bold** and `code`.\n\n- one\n- two"
    html = rh._render_markdown(md)
    assert "<h4>Section</h4>" in html         # ## -> h4 (section already owns h2/h3 chain)
    assert "<strong>bold</strong>" in html
    assert "<code>code</code>" in html
    assert "<li>one</li>" in html and "<li>two</li>" in html


def test_render_atlas_is_selfcontained_no_cdn():
    s = {"module_overview": {"nodes": [{"id": "m", "label": "m", "loc": 3}], "edges": []},
         "lenses": [{"name": "loss", "description": "d", "nodes": [{"id": "fwd", "kind": "function", "file": "a.py"}],
                     "edges": [], "unresolved_entrypoints": []}]}
    st = {"narrative": "## You Are Here\nWe are **here** on `branch`.", "branch": "feature",
          "vs_compare_branch": {"ahead": 1, "behind": 0}}
    ch = {"summary": "first snapshot — no baseline", "structure_delta": {"added_nodes": ["b"]},
          "state_delta": {"new_commits": ["222"]}}
    doc = rh.render_atlas(s, st, ch)
    assert "<!doctype html>" in doc.lower() and "<svg" in doc
    assert "We are" in doc and "<strong>here</strong>" in doc
    # the whole point: nothing fetched from the network
    assert "cdn" not in doc.lower()
    assert "jsdelivr" not in doc.lower()
    assert "mermaid" not in doc.lower()
    assert "<script src" not in doc.lower()      # only an inline <script>, no external src
