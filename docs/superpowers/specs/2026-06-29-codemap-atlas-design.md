# Codemap Atlas — Design Spec

**Date:** 2026-06-29
**Author:** Aram + Claude (brainstorming session)
**Status:** Approved architecture; implementation plan to follow
**Repo at design time:** `dinov3-performance-optimizations/dinov3`, branch `exp/ddp-fullgraph-head-static-shapes`, SHA `1b78e8f`

> **Superseded 2026-07-01 (as-built layout differs).** This spec describes the original repo-embedded
> layout. As shipped, the engine and skills do NOT live in this repo: the engine (stdlib scripts) is at
> `~/.claude/codemap/scripts/` and the four skills are real files at `~/.claude/skills/codemap-*`. Each
> repo keeps only `docs/codemap/codemap.config.yaml` + generated `snapshots/`/`atlas.html`. The renderer
> also emits **self-contained inline SVG**, not client-side Mermaid. So references below to
> `docs/codemap/scripts/...` and `tests/codemap/...` are the design-time paths, not the current ones.
> Current architecture: `docs/codemap/README.md`.

---

## 1. Problem

The user (ML researcher, this dinov3 perf-optimization fork) is hitting two named failure modes from
Addy Osmani's essays:

- **Cognitive surrender** — passively accepting AI-driven changes without holding the model in your head.
- **Comprehension debt** — the gap between "code that exists / changes that landed" and "what I actually
  understand," compounding over time.

Concretely: hard to maintain a mental map of (a) the dinov3 codebase structure (never read E2E, complex
domain), (b) what we're currently working on, (c) recent work, (d) what we're trying next. The user wants
**visual-spatial** tools that externalize this map, are **re-runnable on demand** as the code/experiments
evolve, support a **diff vs last run / vs git commits**, and are **reusable across codebases** via a
per-repo template.

## 2. Goals / Non-goals

**Goals**
- Externalize both the *structural* map (how files/functions link) and the *project-state* map ("you are here").
- Visual-spatial output (rendered diagrams + readable narrative) in one place.
- Re-runnable on demand; cheap to re-run as work evolves.
- "What changed since last run / since commit X" as a first-class, reliable feature.
- Reusable: a generic skill set + a per-repo config that carries codebase/branch/phase specifics.
- Directly fight cognitive surrender: the state map *re-seats the mental model*, it doesn't just dump links.

**Non-goals (YAGNI for v1)**
- Whole-repo automatic call graphs (unreadable hairball for a codebase this size).
- A live/served web app or daemon. Output is static self-contained HTML opened in a browser.
- Replacing the existing `docs/html/` hand-authored docs or the memory/KB systems — this *complements* them.
- Self-grading quizzes (deferred; see §10 future work).

## 3. Locked decisions (from brainstorming)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Snapshot-based architecture.** Every generator run writes a timestamped, SHA-keyed `*.json` snapshot + an HTML fragment. | Makes "diff vs last run" and "diff vs commit X" the *same* operation with a different baseline — free, reliable. |
| D2 | **4 composable skills**, all runnable standalone: `codemap-structure`, `codemap-state`, `codemap-diff`, `codemap-atlas` (orchestrator). | Modular > monolith; matches user's instinct. |
| D3 | **Change-localized map = diff over snapshots**, not an independent generator. | The two generators already emit JSON; the differ is pure JSON comparison, no duplicated parsing. |
| D4 | **Curated hot-paths, layered** structural map: a coarse module-interaction overview + drill-down call graphs only for declared entry points. | Readable & intentional; the curated entry-point list IS the reusable per-repo template. |
| D5 | **In-repo, git-tracked snapshots** under `docs/codemap/`, keyed to git SHA, pruned to last ~10. | Git history becomes the diff substrate; small JSON; portable to a fresh clone / labmate. |
| D6 | **Narrative "you are here" explainer** for the state map (paragraphs synthesized from git + memory + phase docs), not just a pointer dashboard. | This is the anti-cognitive-surrender core. |
| D7 | **Hybrid build:** deterministic scripts extract *facts* (AST/pyreverse, git); the *skill* synthesizes *understanding* (narrative, change summary); an HTML template fuses script-JSON + Claude-prose. | Scripts can't understand; Claude can't be deterministic. Each does only what it's good at. |
| D8 | **Rendering substrate: Mermaid text embedded in self-contained HTML**, rendered client-side via mermaid.js (CDN or vendored). | Env has `pyreverse` (emits Mermaid/DOT text) but **no `dot`/graphviz binary and no mermaid-cli**, so server-side raster rendering is impossible. Client-side mermaid gives pan/zoom/clickable diagrams and reuses the existing `docs/html/` pattern. |

## 4. Environment facts (verified 2026-06-29)

- Canonical env: `/mnt/weka/adovlatyan/.conda/envs/dinov3_env_210clone` (torch 2.10).
- `pyreverse` available (ships with pylint, which is installed). Can emit `mermaid`/`dot`/`puml` text via `-o mmd`.
- **No** `dot` binary on PATH, **no** `mmdc` (mermaid-cli), **no** `pydeps`/`code2flow`/`pyan3`.
- Existing visual surface: `docs/html/claude/*.html` (self-contained HTML docs).
- `mcp__claude_ai_Mermaid_Chart__validate_and_render_mermaid_diagram` MCP tool exists (optional validation of generated mermaid during dev; not a runtime dep).

## 5. System architecture

```
                 docs/codemap/codemap.config.yaml   ← per-repo template (the only codebase-specific file)
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                        
  codemap-structure     codemap-state                 
   (generator)           (generator)                  
   scripts: AST/         scripts: git +               
   pyreverse →           memory/phase-doc             
   structure.json        scan → state.facts.json      
   + mermaid             SKILL: narrative synthesis    
   → structure.          → state.json (+narrative)    
     fragment.html       + state.fragment.html        
        │                     │                        
        └─────────┬───────────┘                        
                  ▼                                     
            codemap-diff  (consumer)                    
            diff(now snapshot, baseline snapshot)       
            scripts: JSON delta; SKILL: change summary  
            → change.json + change.fragment.html        
                  │                                     
                  ▼                                     
            codemap-atlas (composer / orchestrator)     
            runs the 3 (or reuses latest), stitches     
            fragments → docs/codemap/atlas.html         
```

- All four skills are standalone. `codemap-atlas` is the one-command path.
- Every skill reads `codemap.config.yaml`. Swapping that file retargets the whole system to a new repo.

## 6. Artifact layout (in repo, git-tracked)

```
docs/codemap/
  codemap.config.yaml                # per-repo template (hand-edited)
  atlas.html                         # latest fused atlas (overwritten each atlas run)
  snapshots/
    <git-sha>/                       # one dir per snapshotted commit (prune to last ~10)
      structure.json
      state.facts.json               # script-emitted facts
      state.json                     # state.facts + skill narrative
      change.json                    # present if a diff was run for this snapshot
      structure.fragment.html
      state.fragment.html
      change.fragment.html
    latest -> <git-sha>/             # symlink convenience pointer
  scripts/                           # deterministic harness (lives here or under repo scripts/)
    extract_structure.py
    extract_state_facts.py
    diff_snapshots.py
    render_html.py                   # fragment + atlas templating
```

(Pruning keeps the most recent ~`snapshots.keep_last` SHA dirs; `latest` symlink updated each run.)

## 7. Config schema — `docs/codemap/codemap.config.yaml`

The reusable template. dinov3-tuned example:

```yaml
project:
  name: dinov3-satellite
  one_liner: "Satellite-specialized DINOv3 SSL ViT; perf-optimization fork (DDP/FSDP2, cudagraphs)"
  package_root: dinov3
  python: /mnt/weka/adovlatyan/.conda/envs/dinov3_env_210clone/bin/python

structure:
  module_overview: true          # render coarse package-level import/interaction graph
  max_call_depth: 3              # how deep to follow calls from each entrypoint
  lenses:                        # curated hot-paths = the readable, intentional drill-downs
    - name: train-loop
      description: "Core training iteration"
      entrypoints: ["dinov3/train/train.py:do_train"]
    - name: loss-forward
      description: "Student/teacher forward + 4 loss terms + backward"
      entrypoints: ["dinov3/train/ssl_meta_arch.py:SSLMetaArch.forward_backward"]
    - name: distributed-wrap
      description: "DDP/FSDP2 wrap, torch.compile, activation checkpointing"
      entrypoints: ["dinov3/fsdp/ac_compile_parallelize.py"]
    - name: data-pipeline
      description: "Loader + multi-crop collate"
      entrypoints:
        - "dinov3/data/loaders.py:make_data_loader"
        - "dinov3/data/collate.py:collate_data_and_cast"

state:
  compare_branch: master
  phase_docs: ["docs/phase7_perf_plan.md"]
  memory_dir: "/home/adovlatyan/.claude/projects/-home-adovlatyan-dinov3-performance-optimizations-dinov3/memory"
  current_goal_anchor: "docs/phase7_perf_plan.md#7.8"   # optional; where 'what we're trying now' lives
  job_log_glob: "/mnt/weka/adovlatyan/logs/**"          # optional, surfaces recent/open jobs

snapshots:
  dir: docs/codemap/snapshots
  keep_last: 10
```

## 8. JSON contracts (schema_version: 1)

**structure.json** (script-emitted, deterministic):
```json
{
  "schema_version": 1, "generated_at": "ISO8601", "git_sha": "…",
  "package_root": "dinov3",
  "module_overview": {
    "nodes": [{"id": "dinov3.train.train", "label": "train", "loc": 1234}],
    "edges": [{"src": "dinov3.train.train", "dst": "dinov3.train.ssl_meta_arch", "kind": "import", "weight": 3}]
  },
  "lenses": [
    {"name": "train-loop", "entrypoints": ["…"],
     "nodes": [{"id": "qualname", "file": "…", "line": 42, "kind": "function|method|class"}],
     "edges": [{"src": "qualname", "dst": "qualname", "kind": "call"}]}
  ],
  "mermaid": {"module_overview": "graph TD …", "train-loop": "graph TD …"}
}
```

**state.facts.json** (script-emitted, deterministic):
```json
{
  "schema_version": 1, "generated_at": "ISO8601", "git_sha": "…", "branch": "…",
  "vs_compare_branch": {"branch": "master", "ahead": 12, "behind": 0,
    "changed_files": [{"path": "dinov3/train/ssl_meta_arch.py", "added": 40, "deleted": 5}]},
  "recent_commits": [{"sha": "…", "subject": "…", "date": "…"}],
  "phase_docs": [{"path": "docs/phase7_perf_plan.md", "headings": ["7.8 …"]}],
  "memory_index": [{"name": "project_phase7_comms_residual", "desc": "…"}],
  "recent_jobs": [{"jobid": "80711", "state": "PENDING|RUNNING|…", "name": "…"}]
}
```

**state.json** = `state.facts.json` + `"narrative": "<markdown the SKILL writes>"`.

**change.json** (diff of two snapshots; script delta + skill summary):
```json
{
  "schema_version": 1, "from_sha": "…", "to_sha": "…",
  "structure_delta": {"added_nodes": [], "removed_nodes": [], "added_edges": [], "removed_edges": [],
                      "lenses_changed": ["loss-forward"]},
  "state_delta": {"new_commits": [], "changed_files_since": []},
  "summary": "<markdown the SKILL writes: what changed and why it matters>"
}
```

## 9. Per-skill responsibilities

- **codemap-structure**: run `extract_structure.py` (AST walk for the lenses + pyreverse/import scan for the
  module overview) → `structure.json` (incl. mermaid strings) → `render_html.py` → `structure.fragment.html`.
  Pure mechanical; the skill mostly invokes the script and sanity-checks output.
- **codemap-state**: run `extract_state_facts.py` (git ahead/behind + changed files vs `compare_branch`,
  recent commits, phase-doc headings, memory index, optional job states) → `state.facts.json`. **Then the skill
  reads the facts + the referenced phase-doc/memory bodies and *writes the narrative*** ("you are here": phase,
  just-shipped, trying-now-and-why, branch-vs-master, live vs closed experiments) → `state.json` +
  `state.fragment.html`.
- **codemap-diff**: pick baseline (default: previous snapshot dir; or `--since <sha>`), run `diff_snapshots.py`
  to compute structural + state deltas → `change.json` delta; **skill writes the human `summary`** →
  `change.fragment.html`.
- **codemap-atlas**: ensure fresh snapshot (run the 3, or `--reuse-latest`), then `render_html.py --atlas`
  stitches the fragments + a top "you are here" banner → `docs/codemap/atlas.html`. Update `snapshots/latest`
  symlink; prune to `keep_last`.

## 10. Error handling & edge cases

- **Missing config** → skill emits a clear "no `docs/codemap/codemap.config.yaml`; run init / copy the template" message; offers to scaffold one.
- **Entrypoint not found** (renamed/moved during refactor) → script records it under `"unresolved_entrypoints"` in `structure.json` and continues; HTML surfaces it as a warning chip (this is itself a useful "the map drifted" signal).
- **No previous snapshot** (first run) → `codemap-diff` reports "baseline = none (first snapshot)"; atlas omits the change panel.
- **Detached/dirty tree** → snapshot dir keyed by `HEAD` short SHA + a `-dirty` suffix if working tree is dirty; recorded in JSON.
- **mermaid too large** (a lens explodes) → cap nodes per diagram (config `max_call_depth` + a hard node cap); overflow noted in JSON and HTML.
- **memory_dir absent** (running on a different machine / repo) → skip memory_index gracefully.

## 11. Testing / validation strategy

- **Scripts validated against known ground truth first** (per repo harness doctrine): a tiny fixture package
  with hand-known call/import edges; assert `extract_structure.py` recovers them exactly.
- **JSON schema stability**: a schema check (jsonschema or hand-rolled) so the differ never breaks on drift.
- **Diff correctness**: snapshot a fixture, mutate it (add a call, a commit), assert `change.json` reports
  exactly the introduced delta.
- **Mermaid validity**: optionally pipe generated mermaid through the Mermaid MCP validator in dev.
- **End-to-end smoke on dinov3**: run `codemap-atlas`, open `atlas.html`, eyeball the four lenses + the
  "you are here" narrative for the current Phase 7.8 state.

## 12. Reuse on another codebase

1. Copy `docs/codemap/` scripts + skills (skills are repo-agnostic).
2. Write a new `codemap.config.yaml`: project name/one-liner, `package_root`, `python`, the curated `lenses`
   (entrypoints that matter for *that* repo), `compare_branch`, `phase_docs`, `memory_dir`.
3. Run `codemap-atlas`. The generic skills + scripts adapt to the new config.

## 13. Future work (explicitly deferred)

- Self-check questions in the state map (the third state-depth option).
- Auto-suggesting lenses from git churn / centrality instead of hand-curation.
- A `--serve` mode if static HTML proves limiting.
- Cross-snapshot timeline view (more than pairwise diff).

---

### Resume instructions (for a fresh session)

Read this spec, then the implementation plan at
`docs/superpowers/plans/2026-06-29-codemap-atlas-plan.md` (created next). Execute phases in order; each phase
has its own verification. The four skills live under the user's skills dir; the deterministic scripts live
under `docs/codemap/scripts/`. Start by scaffolding `extract_structure.py` against a fixture (Phase 1).
