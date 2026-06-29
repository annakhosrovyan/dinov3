"""Render snapshot JSON into self-contained HTML (client-side Mermaid)."""
from __future__ import annotations
import html, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cm_common as cm

CDN = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"

def mermaid_block(mermaid_text: str) -> str:
    return f'<pre class="mermaid">\n{html.escape(mermaid_text)}\n</pre>'

def _chip(text, kind="warn"):
    return f'<span class="chip {kind}">{html.escape(text)}</span>'

def structure_fragment(structure: dict) -> str:
    out = ["<section><h2>Structural map</h2>"]
    mer = structure.get("mermaid", {})
    if "module_overview" in mer:
        out.append("<details open><summary>Module overview</summary>")
        out.append(mermaid_block(mer["module_overview"]) + "</details>")
    for lens in structure.get("lenses", []):
        name = lens["name"]
        out.append(f"<details open><summary>{html.escape(name)} — {html.escape(lens.get('description',''))}</summary>")
        for ep in lens.get("unresolved_entrypoints", []):
            out.append(_chip(f"unresolved: {ep}"))
        if name in mer:
            out.append(mermaid_block(mer[name]))
        out.append("</details>")
    out.append("</section>")
    return "\n".join(out)

def _md_lite(text: str) -> str:
    t = html.escape(text or "")
    import re
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    return "<br>".join(t.splitlines())

def state_fragment(state: dict) -> str:
    vs = state.get("vs_compare_branch", {})
    facts = (f'branch <code>{html.escape(str(state.get("branch")))}</code> · '
             f'ahead {vs.get("ahead")} / behind {vs.get("behind")}')
    return (f'<section class="youarehere"><h2>You are here</h2>'
            f'<p class="facts">{facts}</p><div>{_md_lite(state.get("narrative",""))}</div></section>')

def change_fragment(change: dict) -> str:
    sd = change.get("structure_delta", {}); std = change.get("state_delta", {})
    parts = [f'<section><h2>Since last snapshot</h2><p>{_md_lite(change.get("summary",""))}</p><ul>']
    parts.append(f'<li>+nodes: {html.escape(str(sd.get("added_nodes", [])))}</li>')
    parts.append(f'<li>-nodes: {html.escape(str(sd.get("removed_nodes", [])))}</li>')
    parts.append(f'<li>new commits: {html.escape(str(std.get("new_commits", [])))}</li>')
    parts.append("</ul></section>")
    return "\n".join(parts)

_CSS = """body{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
h1{border-bottom:2px solid #444}.youarehere{background:#f5f8ff;border-left:4px solid #3b6;padding:1rem;border-radius:6px}
.chip{display:inline-block;background:#fde;color:#900;border-radius:10px;padding:1px 8px;margin:2px;font-size:12px}
details{margin:.5rem 0;border:1px solid #eee;border-radius:6px;padding:.5rem}summary{cursor:pointer;font-weight:600}
.facts{color:#555;font-size:13px}code{background:#eef;padding:0 4px;border-radius:3px}"""

def render_atlas(structure: dict, state: dict, change: dict) -> str:
    body = state_fragment(state) + structure_fragment(structure)
    if change:
        body += change_fragment(change)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Codemap Atlas</title><style>{_CSS}</style>
<script type="module">import mermaid from "{CDN}";mermaid.initialize({{startOnLoad:true,securityLevel:'loose'}});</script>
</head><body><h1>Codemap Atlas</h1>{body}</body></html>"""

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); ap.add_argument("--atlas", default=None)
    ap.add_argument("--fragments", default=None); args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()
    def _load(d, n):
        p = Path(d) / n; return json.loads(p.read_text()) if p.exists() else {}
    if args.fragments:
        d = Path(args.fragments)
        (d / "structure.fragment.html").write_text(structure_fragment(_load(d, "structure.json")))
        (d / "state.fragment.html").write_text(state_fragment(_load(d, "state.json")))
        (d / "change.fragment.html").write_text(change_fragment(_load(d, "change.json")))
        print(d)
    if args.atlas:
        d = Path(args.atlas)
        html_doc = render_atlas(_load(d, "structure.json"), _load(d, "state.json"), _load(d, "change.json"))
        out = repo / "docs/codemap/atlas.html"; out.write_text(html_doc); print(out)

if __name__ == "__main__":
    main()
