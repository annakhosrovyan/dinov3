# codemap (this repo's instance)

This directory is a **codemap instance**, not the tool. The engine and the four skills live
globally, outside this repo:

- **Engine (shared):** `~/.claude/codemap/scripts/` (stdlib `ast` + `git`; the abstract base).
- **Skills (global):** `~/.claude/skills/codemap-{structure,state,diff,atlas}` — invoke with
  `/codemap-atlas` (full refresh) or the targeted ones.
- **Config template:** `~/.claude/codemap/codemap.config.template.yaml`.

What lives HERE (the per-repo "subclass"):

- `codemap.config.yaml` — this repo's package_root, lenses, state paths. The only hand-edited file.
- `snapshots/` — SHA-keyed fact snapshots (the diff substrate; tracked on purpose).
- `atlas.html` — the generated self-contained page. Open it in a browser to re-orient.

To refresh: run `/codemap-atlas` from the repo root. To set this up in another repo: create
`docs/codemap/codemap.config.yaml` from the template (the skill scaffolds it) and run `/codemap-atlas`
there — no per-repo install.
