"""Render snapshot JSON into a fully self-contained HTML atlas.

No external assets: graphs are drawn as inline SVG computed here (a small layered
left-to-right layout), so the page renders offline with no CDN / JS dependency.
A tiny inline script adds optional hover-highlighting; the diagrams are fully
readable without it.
"""
from __future__ import annotations
import html, json, re, hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

_esc = html.escape

# ----------------------------------------------------------------------------- colour
# Light fill + darker stroke/text per group. Groups = a class (qualname prefix)
# or a module/file, so a class's methods share one colour.
_PALETTE = [
    ("#e8f0fe", "#1a73e8"), ("#e6f4ea", "#188038"), ("#fce8e6", "#c5221f"),
    ("#fef7e0", "#b06000"), ("#f3e8fd", "#8430ce"), ("#e4f7fb", "#00796b"),
    ("#fde7f3", "#b80672"), ("#eaeef2", "#455a64"), ("#fff0e0", "#9a5b00"),
    ("#e3f2e1", "#2e7d32"), ("#e8eaf6", "#3949ab"), ("#fbe9e7", "#bf360c"),
]


def _color(group: str):
    idx = int(hashlib.md5(group.encode()).hexdigest(), 16) % len(_PALETTE)
    return _PALETTE[idx]


def _group_key(node: dict) -> str:
    nid = node["id"]
    if node.get("kind"):  # lens node: class methods share the class; else file stem
        if "." in nid:
            return nid.split(".")[0]
        f = node.get("file", "")
        return Path(f).stem if f else nid
    # module-overview node: group by parent package
    return nid.rsplit(".", 1)[0] if "." in nid else nid


# ----------------------------------------------------------------------------- sizing
_CHAR = 7.0     # px per char for the bold 13px name
_PADX = 26
_MINW = 120
_MAXW = 300


def _wrap2(s: str):
    """Split a too-long label into 2 lines, preferring a dotted boundary near the middle."""
    mid = len(s) // 2
    dots = [i for i, c in enumerate(s) if c == "."]
    if dots:
        b = min(dots, key=lambda i: abs(i - mid))
        return s[: b + 1], s[b + 1 :]
    return s[:mid], s[mid:]


def _box_size(label: str):
    w1 = len(label) * _CHAR + _PADX
    if w1 <= _MAXW:
        return max(_MINW, int(w1)), 46, [label]
    l1, l2 = _wrap2(label)
    w = max(len(l1), len(l2)) * _CHAR + _PADX
    return max(_MINW, min(_MAXW, int(w))), 62, [l1, l2]


# ----------------------------------------------------------------------------- layout
def _layout(graph: dict):
    """Layered left-to-right layout. Returns (nodes, edges, width, height, groups)."""
    nodes = {n["id"]: dict(n) for n in graph.get("nodes", [])}
    edges = [(e["src"], e["dst"]) for e in graph.get("edges", [])
             if e["src"] in nodes and e["dst"] in nodes and e["src"] != e["dst"]]
    if not nodes:
        return {}, [], 40, 40, []

    # longest-path layering (Bellman-Ford style relaxation, cycle-capped)
    level = {nid: 0 for nid in nodes}
    for _ in range(len(nodes)):
        changed = False
        for s, d in edges:
            if level[d] < level[s] + 1:
                level[d] = level[s] + 1
                changed = True
        if not changed:
            break

    PAD, COL_GAP, ROW_GAP = 18, 64, 16
    cols: dict[int, list[str]] = {}
    for nid in sorted(nodes):                       # stable order within a column
        cols.setdefault(level[nid], []).append(nid)

    for nid, n in nodes.items():
        w, h, lines = _box_size(n["id"])
        n["w"], n["h"], n["lines"] = w, h, lines
        n["group"] = _group_key(n)

    # per-column x (column as wide as its widest node), per-column stacked y
    x = PAD
    total_h = 0
    for k in sorted(cols):
        colw = max(nodes[nid]["w"] for nid in cols[k])
        y = PAD
        for nid in cols[k]:
            nodes[nid]["x"] = x
            nodes[nid]["y"] = y
            nodes[nid]["w"] = colw
            y += nodes[nid]["h"] + ROW_GAP
        total_h = max(total_h, y)
        x += colw + COL_GAP
    width = x - COL_GAP + PAD
    height = total_h + PAD
    groups = sorted({n["group"] for n in nodes.values()})
    return nodes, edges, width, height, groups


def _edge_path(s: dict, d: dict) -> str:
    sx, sy = s["x"] + s["w"], s["y"] + s["h"] / 2
    dx, dy = d["x"], d["y"] + d["h"] / 2
    cx = max(24, (dx - sx) * 0.5)
    return f"M{sx:.0f},{sy:.0f} C{sx + cx:.0f},{sy:.0f} {dx - cx:.0f},{dy:.0f} {dx:.0f},{dy:.0f}"


def svg_graph(graph: dict, gid: str) -> str:
    """One inline-SVG graph plus a small colour legend. Self-contained (no JS required)."""
    nodes, edges, width, height, groups = _layout(graph)
    if not nodes:
        return '<p class="empty">(no nodes)</p>'

    parts = [
        f'<svg class="cmgraph" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" preserveAspectRatio="xMinYMin meet">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" '
        'orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#9aa0a6"/></marker></defs>',
    ]
    # edges first (under nodes), coloured by source group
    for s, d in edges:
        sn, dn = nodes[s], nodes[d]
        _, stroke = _color(sn["group"])
        parts.append(
            f'<path class="edge" data-src="{_esc(s)}" data-dst="{_esc(d)}" '
            f'd="{_edge_path(sn, dn)}" fill="none" stroke="{stroke}" stroke-width="1.4" '
            f'stroke-opacity="0.4" marker-end="url(#arrow)"/>'
        )
    # nodes
    for nid in sorted(nodes):
        n = nodes[nid]
        fill, stroke = _color(n["group"])
        x, y, w, h, lines = n["x"], n["y"], n["w"], n["h"], n["lines"]
        kind = n.get("kind") or (f'{n["loc"]} loc' if "loc" in n else "")
        parts.append(
            f'<g class="node" data-node="{_esc(nid)}"><title>{_esc(nid)}</title>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>'
        )
        if len(lines) == 1:
            parts.append(f'<text class="nm" x="{x + 12}" y="{y + 20}" fill="{stroke}">{_esc(lines[0])}</text>')
            ky = y + 36
        else:
            parts.append(f'<text class="nm" x="{x + 12}" y="{y + 18}" fill="{stroke}">{_esc(lines[0])}</text>')
            parts.append(f'<text class="nm" x="{x + 12}" y="{y + 34}" fill="{stroke}">{_esc(lines[1])}</text>')
            ky = y + 52
        if kind:
            parts.append(f'<text class="kd" x="{x + 12}" y="{ky}">{_esc(kind)}</text>')
        parts.append("</g>")
    parts.append("</svg>")

    legend = ['<div class="legend">']
    for g in groups:
        _, stroke = _color(g)
        legend.append(f'<span class="lg"><i style="background:{stroke}"></i>{_esc(g)}</span>')
    legend.append("</div>")
    return f'<div class="graphwrap" id="{_esc(gid)}">{"".join(parts)}{"".join(legend)}</div>'


def _chip(text, kind="warn"):
    return f'<span class="chip {kind}">{_esc(text)}</span>'


def structure_fragment(structure: dict) -> str:
    out = ['<section><h2>Structural map</h2>']
    mo = structure.get("module_overview")
    if mo and mo.get("nodes"):
        out.append('<details><summary>Module overview</summary>')
        out.append(svg_graph(mo, "module_overview") + "</details>")
    for lens in structure.get("lenses", []):
        name = lens["name"]
        out.append(f'<details open><summary>{_esc(name)} — {_esc(lens.get("description", ""))}</summary>')
        for ep in lens.get("unresolved_entrypoints", []):
            out.append(_chip(f"unresolved: {ep}"))
        out.append(svg_graph(lens, name))
        out.append("</details>")
    out.append("</section>")
    return "\n".join(out)


# ----------------------------------------------------------------------------- markdown
_INLINE = (
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"`([^`]+?)`"), r"<code>\1</code>"),
)


def _inline_md(text: str) -> str:
    t = _esc(text)
    for pat, rep in _INLINE:
        t = pat.sub(rep, t)
    return t


def _render_markdown(md: str) -> str:
    """Render a useful markdown subset: #/##/### headings, bold, `code`, - bullets, paragraphs."""
    out, para, bullets = [], [], []

    def flush_para():
        if para:
            out.append("<p>" + "<br>".join(_inline_md(l) for l in para) + "</p>")
            para.clear()

    def flush_bullets():
        if bullets:
            out.append("<ul>" + "".join(f"<li>{_inline_md(b)}</li>" for b in bullets) + "</ul>")
            bullets.clear()

    for raw in (md or "").splitlines():
        s = raw.strip()
        if not s:
            flush_para(); flush_bullets(); continue
        m = re.match(r"(#{1,6})\s+(.*)", s)
        if m:
            flush_para(); flush_bullets()
            lvl = min(len(m.group(1)) + 2, 5)        # narrative # -> h3 (section owns h2)
            out.append(f"<h{lvl}>{_inline_md(m.group(2))}</h{lvl}>")
            continue
        if re.match(r"[-*]\s+", s):
            flush_para()
            bullets.append(re.sub(r"^[-*]\s+", "", s))
            continue
        flush_bullets()
        para.append(s)
    flush_para(); flush_bullets()
    return "\n".join(out)


def state_fragment(state: dict) -> str:
    vs = state.get("vs_compare_branch", {})
    facts = (f'branch <code>{_esc(str(state.get("branch")))}</code> · '
             f'ahead {_esc(str(vs.get("ahead", "?")))} / behind {_esc(str(vs.get("behind", "?")))}')
    return (f'<section class="youarehere"><h2>You are here</h2>'
            f'<p class="facts">{facts}</p>'
            f'<div class="narrative">{_render_markdown(state.get("narrative", ""))}</div></section>')


def change_fragment(change: dict) -> str:
    sd = change.get("structure_delta", {})
    std = change.get("state_delta", {})
    parts = [f'<section><h2>Since last snapshot</h2>'
             f'<div class="narrative">{_render_markdown(change.get("summary", ""))}</div><ul>']
    parts.append(f'<li>+nodes: {_esc(str(sd.get("added_nodes", [])))}</li>')
    parts.append(f'<li>-nodes: {_esc(str(sd.get("removed_nodes", [])))}</li>')
    parts.append(f'<li>new commits: {_esc(str(std.get("new_commits", [])))}</li>')
    parts.append("</ul></section>")
    return "\n".join(parts)


_CSS = """
:root{--ink:#1a1a1a;--mut:#5f6368}
*{box-sizing:border-box}
body{font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;max-width:1180px;margin:2rem auto;padding:0 1rem;color:var(--ink)}
h1{border-bottom:2px solid #444;padding-bottom:.3rem}
h2{margin:.2rem 0}
.youarehere{background:#f5f8ff;border-left:4px solid #34a853;padding:.6rem 1.2rem;border-radius:8px;margin:1rem 0}
.narrative h3{font-size:1.05rem;margin:1rem 0 .3rem}
.narrative h4{font-size:.95rem;margin:.8rem 0 .2rem;color:#333}
.narrative h5{font-size:.9rem;margin:.7rem 0 .2rem;color:#444}
.narrative p{margin:.5rem 0}
.narrative ul{margin:.3rem 0 .6rem 1.1rem;padding:0}
.narrative li{margin:.15rem 0}
.facts{color:var(--mut);font-size:13px;margin:.2rem 0 .6rem}
code{background:#eef1f7;padding:0 4px;border-radius:3px;font-size:.92em}
.chip{display:inline-block;background:#fde;color:#900;border-radius:10px;padding:1px 9px;margin:2px 4px;font-size:12px}
details{margin:.6rem 0;border:1px solid #e6e6e6;border-radius:8px;padding:.5rem .8rem;background:#fff}
summary{cursor:pointer;font-weight:600;padding:.2rem 0}
.graphwrap{overflow-x:auto;padding:.4rem 0}
svg.cmgraph{max-width:none}
svg.cmgraph .nm{font:600 13px system-ui,sans-serif}
svg.cmgraph .kd{font:10px system-ui,sans-serif;fill:#80868b}
svg.cmgraph.dim .edge{stroke-opacity:.06}
svg.cmgraph.dim .node{opacity:.28}
svg.cmgraph .edge.hl{stroke-opacity:.95;stroke-width:2.4}
svg.cmgraph .node.hl{opacity:1}
.legend{display:flex;flex-wrap:wrap;gap:.4rem .9rem;margin:.5rem 0 .2rem;font-size:12px;color:var(--mut)}
.legend .lg{display:inline-flex;align-items:center;gap:.3rem}
.legend i{width:11px;height:11px;border-radius:3px;display:inline-block}
.empty{color:var(--mut);font-style:italic}
"""

_JS = """
for(const svg of document.querySelectorAll('svg.cmgraph')){
  const edges=[...svg.querySelectorAll('.edge')];
  svg.querySelectorAll('.node').forEach(node=>{
    const id=node.getAttribute('data-node');
    node.addEventListener('mouseenter',()=>{
      svg.classList.add('dim');
      const keep=new Set([id]);
      edges.forEach(e=>{const s=e.getAttribute('data-src'),d=e.getAttribute('data-dst');
        if(s===id||d===id){e.classList.add('hl');keep.add(s);keep.add(d);}});
      svg.querySelectorAll('.node').forEach(n=>{if(keep.has(n.getAttribute('data-node')))n.classList.add('hl');});
    });
    node.addEventListener('mouseleave',()=>{
      svg.classList.remove('dim');
      svg.querySelectorAll('.hl').forEach(el=>el.classList.remove('hl'));
    });
  });
}
"""


def render_atlas(structure: dict, state: dict, change: dict) -> str:
    body = state_fragment(state) + structure_fragment(structure)
    if change:
        body += change_fragment(change)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Codemap Atlas</title><style>' + _CSS + "</style></head><body>"
        "<h1>Codemap Atlas</h1>" + body +
        "<script>" + _JS + "</script></body></html>"
    )


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--atlas", default=None)
    ap.add_argument("--fragments", default=None)
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()

    def _load(d, n):
        p = Path(d) / n
        return json.loads(p.read_text()) if p.exists() else {}

    if args.fragments:
        d = Path(args.fragments)
        (d / "structure.fragment.html").write_text(structure_fragment(_load(d, "structure.json")))
        (d / "state.fragment.html").write_text(state_fragment(_load(d, "state.json")))
        (d / "change.fragment.html").write_text(change_fragment(_load(d, "change.json")))
        print(d)
    if args.atlas:
        d = Path(args.atlas)
        html_doc = render_atlas(_load(d, "structure.json"), _load(d, "state.json"), _load(d, "change.json"))
        out = repo / "docs/codemap/atlas.html"
        out.write_text(html_doc)
        print(out)


if __name__ == "__main__":
    main()
