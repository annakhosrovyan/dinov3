---
name: codemap-atlas
description: Build the full codebase ATLAS in one shot — runs the structural map, the "you are here" state narrative, and the diff-since-last-run, then fuses them into a single self-contained HTML page (docs/codemap/atlas.html) you open in a browser. Use when the user wants the whole visual orientation refreshed, e.g. after landing changes or starting a session. Reads docs/codemap/codemap.config.yaml.
---

# codemap-atlas

One-command orientation. Orchestrates the other three codemap skills, then fuses.

## Steps
1. Run codemap-structure (produces structure.json + fragment in the current snapshot dir).
2. Run codemap-state (produces state.json with narrative + fragment). This includes the
   narrative-writing step — do it properly, that's the core value.
3. Run codemap-diff against the previous snapshot (produces change.json with summary). If no
   previous snapshot exists, note "first atlas — no diff".
4. Fuse: `<python> docs/codemap/scripts/render_html.py --repo . --atlas docs/codemap/snapshots/<sha>`
   → writes docs/codemap/atlas.html.
5. Prune + symlink: run a tiny python snippet using cm_common:
   `<python> -c "import sys; sys.path.insert(0,'docs/codemap/scripts'); import cm_common as cm, pathlib;
   repo=pathlib.Path('.').resolve(); cfg=cm.load_config(repo);
   print('removed', cm.prune_snapshots(repo,cfg));
   cm.update_latest_symlink(repo/cfg['snapshots']['dir'], cm.git_sha(repo))"`
6. Tell the user: atlas written to docs/codemap/atlas.html (open in a browser on the Mac), plus a
   2-3 line spoken summary of the "you are here" narrative and the biggest change since last time.
7. Offer to commit the snapshot + atlas (don't auto-commit unless the user asked). Suggested commit
   command: `git add docs/codemap/atlas.html docs/codemap/snapshots && git commit -m "chore(codemap): update atlas <sha>"`.
   Staging the whole `docs/codemap/snapshots` path ensures pruned-away snapshot dirs are recorded as deletions, keeping the tree clean so the next run's SHA isn't marked `-dirty`.

## Notes
- This is the skill to reach for first when re-orienting. The other three are for targeted refreshes.
