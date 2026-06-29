---
name: codemap-diff
description: Show what changed in the codebase map since the last codemap snapshot or since a given commit — structural deltas (added/removed call edges, changed lenses) plus state deltas (new commits, newly changed files). Use when the user asks "what changed since last time / since commit X" in terms of the codemap. Reads docs/codemap/codemap.config.yaml.
---

# codemap-diff

## Steps
1. Identify the current snapshot dir (latest under docs/codemap/snapshots, or the one just produced).
   The new snapshot must already have structure.json + state.facts.json (run codemap-structure /
   codemap-state first, or codemap-atlas which does both).
2. Run the differ:
   `<python> docs/codemap/scripts/diff_snapshots.py --repo . --new docs/codemap/snapshots/<new_sha>`
   (add `--old docs/codemap/snapshots/<sha>` to diff against a specific commit's snapshot.)
   → writes change.json with an empty `summary`.
3. Read change.json. **Write the `summary`**: 2-5 sentences in plain language — what structurally
   moved (which lenses, which call edges appeared/vanished) and what that implies about the work
   (tie to the new_commits). If `lenses_changed` includes a hot path, say which experiment/commit
   likely caused it.
4. Write the summary back into change.json (read, set "summary", write).
5. Render the fragment and report the diff to the user.

## Notes
- First run has no baseline — say so, don't fabricate a diff.
- The structural diff is name-based and approximate (orientation aid); don't over-claim a removed
  edge means dead code — say "the map no longer follows X→Y", and suggest verifying.
- Reliable diffs require **clean-commit** snapshots: a `<sha>-dirty` snapshot captures the working-tree state at run time and is OVERWRITTEN if you re-run at the same HEAD with different uncommitted changes. To compare two points in history, snapshot at clean commits.
