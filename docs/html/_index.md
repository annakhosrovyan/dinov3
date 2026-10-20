# docs/html/

HTML artifacts from the perf optimization project. Files live in three top-level folders by
**generator** (`claude/`, `codex/`, `explainers/`); within this index they are grouped by **phase**
and tagged with a **Status** so superseded pages are obvious at a glance.

- **Current** — reflects the live state of the project.
- **Superseded** — accurate when written, but its thesis/numbers were later replaced. Kept for the record.
- **Historical** — an earlier phase, retained as background. Not wrong, just not the active work.

Start here: [`claude/index.html`](claude/index.html) is the landing page (newest phase first).

## Phase 8 — the optimization ladder (active)

| File | What it covers | Status |
|------|----------------|--------|
| explainers/phase8-optimization-ladder-2026-07-10.html | **Phase 8 roadmap explainer** — the ranked 9-candidate ladder, the two-stage framing (clean arrival skew, *then* tune comms), and the sequencing diagram showing that #1 root IRQ steering **gates** the comms cluster (#3-screen/#5/#6) while #7/#3-dump/#2 run rootless today. Includes killed levers, the four framing corrections, and the `in_order=False` rejection. Mirrors `docs/phase8_optimization_candidates.md`. | **Current** |

## Phase 7 — flat-MFU residual (the comms thesis)

| File | What it covers | Status |
|------|----------------|--------|
| claude/phase7_summary.html | **Summary dashboard** — residual is communication/sync, not compute (job 75082); ladder 7.0–7.7. Mirrors `docs/phase7_perf_plan.md`. | **Current** |
| claude/phase7_recent_runs.html | **Recent runs (2026-06-26)** — §7.7 bs128/160/192 differential (two-mechanism finding: compute U-shape + growing residual) + §7.3 metric-gate result (small lever, ~−3.5% GPU-side NCCL, wait relocated not eliminated). Jobs 80545/80546/80547 vs baseline 75082. | **Current** |
| explainers/torch-compile-dispatch-phase7-2026-06-11.html | "What actually compiles in Phase 7?" — torch.compile dispatch map. This is the *compute* thesis (Part II in the plan); superseded by 7.1 as the residual driver but still accurate as background. | Superseded |
| codex/ | Codex-produced HTML pages (phase 5 status/experiments/mental-model, upstream PR findings) — model-comparison counterpart to claude/ | Historical |

## Phase 6 — DDP + cudagraphs

| File | What it covers | Status |
|------|----------------|--------|
| claude/phase6_ledger.html | DDP + cudagraphs experiment scoreboard. | Historical |

## Phase 5 — FSDP2 / torch.compile screening

| File | What it covers | Status |
|------|----------------|--------|
| claude/status.html | Phase 5 status — decision board and experiment table. | Historical |
| claude/concepts.html | Glossary, hypotheses, and mental models. | Historical |
| claude/mfu_reference.html | MFU reference (FLOP formula, ceilings). | Historical |
| codex/phase5_current.html | Phase 5 current state (Codex view). | Historical |
| codex/phase5_experiments.html | Phase 5 experiments (Codex view). | Historical |
| codex/phase5_mental_model.html | Phase 5 mental model (Codex view). | Historical |
| codex/phase5_archive.html | Phase 5 archive (Codex view). | Historical |
| explainers/torch-compile-dinov3-phase5-2026-05-21.html | torch.compile + CUDA graphs learning guide. | Historical |
| explainers/dinov3-training-pipeline-ibot-compile-explainer-2026-05-22.html | iBOT + torch.compile pipeline explainer. | Historical |

## Methodology (cross-phase)

| File | What it covers | Status |
|------|----------------|--------|
| explainers/automated-paper-ingestion-pipeline-2026-07-13.html | **Automated paper ingestion proposals** — Zotero-first, thin scheduled digest, and self-hosted arxiv-sanity-lite options; includes the recommended four-week pilot and ResearchRabbit handoff. Mirrors `docs/research/automated-paper-ingestion-pipeline-plan-2026-07-13.md`. | **Current** |
| explainers/autoresearch-loops-mfu-2026-07-03.html | **Autoresearch loops synthesis** — how open-ended agent loops can feasibly be (verifier exactness × eval cost), evidence from AVO/AlphaEvolve/Sakana/METR/Ralph/Paradigma, three-tier map for our lab, and the T2 MFU screening-loop blueprint. | **Current** |

## Upstream PR findings (cross-phase)

| File | What it covers | Status |
|------|----------------|--------|
| claude/upstream_pr_findings_v2.html | PR #324 and PR #340 corrections (v2 — current). | Current |
| claude/upstream_pr_findings.html | Upstream PR findings (v1). | Superseded |
| codex/upstream_pr_findings.html | Upstream PR findings (Codex view). | Superseded |

## Supporting files

| File | What it covers |
|------|----------------|
| claude/index.html | Landing page / navigation for the `claude/` views. |
| codex/style.css | Shared stylesheet for the `codex/` pages. |

## Conventions

- **`claude/`** — Claude-generated summaries and status dashboards. New phase summaries land here
  (e.g. `phase7_summary.html`), styled to match `status.html` / `phase6_ledger.html`.
- **`codex/`** — Codex-generated summaries (Phase 5).
- **`explainers/`** — standalone *educational* HTML docs (not status/model-comparison artifacts),
  dated in the filename.
- When a page is superseded, mark it **Superseded** here rather than deleting it — the prior reasoning
  is the record of why the current direction is right.
