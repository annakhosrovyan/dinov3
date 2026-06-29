---
name: codemap-structure
description: Generate the structural code map (module overview + curated lens call-graphs) for the current repo, writing a SHA-keyed snapshot and an HTML fragment. Use when the user wants to see/refresh how files and functions link, or as part of building the codemap atlas. Reads docs/codemap/codemap.config.yaml.
---

# codemap-structure

Generate the structural map from `docs/codemap/codemap.config.yaml`.

## Steps
1. Resolve the python: read `project.python` from the config (fallback: `python3`).
2. Run the extractor:
   `<python> docs/codemap/scripts/extract_structure.py --repo .`
   It writes `structure.json` into `docs/codemap/snapshots/<sha>/` and prints the path.
3. Render the fragment:
   `<python> docs/codemap/scripts/render_html.py --repo . --fragments docs/codemap/snapshots/<sha>`
4. Read `structure.json`. If `unresolved_entrypoints` is non-empty for any lens, tell the user
   plainly — that means a config entrypoint was renamed/moved (the map has drifted), and offer to
   fix the config lens.
5. Report: how many lenses, node/edge counts per lens, and the snapshot path. Do NOT invent
   structure the JSON does not contain.

## Notes
- This skill emits FACTS only (no narrative). Pure mechanical run + sanity check.
- If the config is missing, offer to scaffold one from docs/superpowers/specs §7.
