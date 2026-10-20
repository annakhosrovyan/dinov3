# DINOv3 Perf Docs Index

Project docs for YerevaNN DINOv3 performance optimization work.

## Root files

| File | What it covers | Status |
|------|---------------|--------|
| yerevann-dinov3-perf-optimizations-project-dossier.md | Full project dossier — goals, phases, context | Current reference |
| branch-history.md | Branch chain history — single source of truth | Live, updated each phase |
| phase6_perf_plan.md | Phase 6 plan: DDP + cudagraphs champion, bs validation, soak | Closed |
| phase7_perf_plan.md | Phase 7: compile dispatch, nsys residual attribution, straggler 7.8/7.9 | **Closed (7.9 verdict 2026-07-06)** |
| phase8_optimization_candidates.md | Phase 8 candidate ladder (GPT-5.6-sol review, Codex-gated) + decision log | **Active** |
| phase7_harness_audit.md | Harness audit: F1–F5 fixes, v2 nsys reports, retractions | Reference |
| phase7_harness_audit_handoff.md | Agent handoff for the harness audit | Handoff, done |
| phase7_harness_fix_handoff.md | Agent handoff for harness fixes F1–F5 | Handoff, done |
| phase7_compile_dispatch_agent_handoff.md | Agent handoff for compile-dispatch investigation | Handoff, done |
| ddp_checkpoint_cleanup_results_2026-06-05.md | DDP checkpoint load/write/resume smoke results | Reference |
| handoff.md | General session handoff notes | Scratch |
| handoff_ddp_checkpoint_cleanup.md | Handoff for DDP checkpoint cleanup | Handoff, done |
| [research/](research/_index.md) | Educational explainers, upstream findings, setup reviews, handoff context | Subfolder |
| [html/](html/_index.md) | HTML artifacts: model-comparison experiment (claude/ vs codex/) + explainers/ | Subfolder |
| codemap/ | Codemap-atlas generated code maps | Subfolder |
| codex/ | Codex-session artifacts and reviews | Subfolder |
| gpt-5-pro-content/ | Context packs and prompts for GPT-5 Pro research sessions | Subfolder |
| superpowers/ | Superpowers skill working notes | Subfolder |
| [archive/](archive/_index.md) | Phases 1–5 material — closed plans, experiment logs, outdated trackers | Subfolder |

## Navigation notes

- Current phase: `phase8_optimization_candidates.md` (candidate ladder + decision log)
- Completed phase plans: `phase7_perf_plan.md`, `phase6_perf_plan.md`
- For project history and branch context: `branch-history.md`
- For synthesized knowledge: `~/knowledge-base/wiki/index.md` or `/kb-search`
