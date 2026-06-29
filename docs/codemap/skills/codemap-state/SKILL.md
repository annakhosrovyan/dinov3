---
name: codemap-state
description: Generate the project-state "you are here" map for the current repo — a narrative synthesized from git, phase docs, and memory that re-seats the mental model (current phase, what just shipped, what we're trying now and why, branch-vs-master, live vs closed experiments). Use when the user feels lost / wants to re-orient, or as part of the codemap atlas. Reads docs/codemap/codemap.config.yaml.
---

# codemap-state

## Steps
1. Run the facts extractor:
   `<python> docs/codemap/scripts/extract_state_facts.py --repo .`
   → writes `state.facts.json` into the snapshot dir.
2. Read `state.facts.json`. Then READ the bodies it points to (the phase_docs paths, and the
   referenced memory files under memory_dir whose desc looks current). Do not rely on headings alone.
3. **Write the narrative** — a few tight paragraphs aimed at someone re-orienting after time away:
   - Where we are: current phase + one-line project framing.
   - What just shipped (last few commits, in plain language — not raw subjects).
   - What we're trying NOW and WHY (pull from the current_goal_anchor / latest phase section).
   - How this branch differs from compare_branch (use ahead/behind + changed_files).
   - Which experiments are live vs closed (from memory/phase docs).
   Be concrete and honest; cite job ids / file paths. This is the anti-cognitive-surrender core —
   write it so the user rebuilds the map in their head, not just skims links.
4. Write `state.json` = the facts object plus a top-level `"narrative"` string (your markdown).
   Use: read state.facts.json, add the key, write back with the same path.
5. Render: `<python> docs/codemap/scripts/render_html.py --repo . --fragments docs/codemap/snapshots/<sha>`
6. Show the user the narrative in chat too.

## Guardrails
- Verify before claiming (repo CLAUDE.md): if you state a job passed/failed or a file changed,
  it must be grounded in the facts JSON or a file you actually read.
- Narrative only — never edit code or configs from this skill.
