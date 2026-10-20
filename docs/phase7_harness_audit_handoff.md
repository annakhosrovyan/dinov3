# Handoff: Phase 7 profiling-harness audit

**Date**: 2026-07-02 · **Branch**: `exp/ddp-fullgraph-head-static-shapes` · **Repo**: `~/dinov3-performance-optimizations/dinov3`

## Mission

Audit the Phase 7 nsys profiling harness for bugs. Every Phase 7 conclusion (comms residual,
batch-scaling U-shape, metric-gate ≈0, and now the rank-7 straggler verdict) rests on numbers this
harness produces. Before we act on the straggler finding, we need to know the harness is sound.

**Files under audit** (both untracked, working tree only):
- `scripts/profiling/phase7_nsys_report.py` — the analyzer (sqlite → per-iter report)
- `scripts/profiling/phase7_nsys_ddp_capture.sh` — the capture sbatch script

**This is read-only analysis + a written report. Do not queue GPU jobs. Fixes to the scripts are
in scope only after findings are confirmed against data.**

## Ground-truth data available (no jobs needed)

| Trace | Config | Path |
|---|---|---|
| 80711 | bs192, w24, PERRANK_DIAG, Sinkhorn NVTX | `/mnt/weka/adovlatyan/nsys_profiles/phase7/80711/` (`.sqlite`, `.nsys-rep`, `.phase7.json`, `.phase7.md`) |
| 80545/80546/80547 | bs128/160/192 sweep (80547 = metrics_sync_period=10) | `/mnt/weka/adovlatyan/nsys_profiles/phase7/<jobid>/` |
| 75082 | original bs128 residual-attribution capture | `/mnt/weka/adovlatyan/nsys_profiles/phase7/75082/` |

Slurm logs: `/mnt/weka/adovlatyan/logs/p7-nsys-<jobid>.{out,err}`. The 80711 `.err` contains
`[RANKDATA]`/`[RANKTOPO]` lines — these are **independent of the harness** (raw stderr from
`dinov3/logging/helpers.py` / `train.py`), useful as cross-checks.

## Specific anomalies to explain (from the 80711 report)

Each is either a real effect or a harness bug — classify which:

1. **Heavy tails**: iter wall mean 617 ms vs p50 472 ms; `losses_backward` mean 234 ms vs p50
   89 ms. Real skew (rank-7 straggler stalls bunching into some iters) or iteration-boundary
   misattribution in the segmenter?
2. **"dropped 2 out-of-band segment(s)"** — the outlier filter (`0 < wall ≤ max(10 s, 6×median)`).
   What did it drop in 80711, and does the rule ever drop legitimate straggler-inflated iters
   (which would bias the very skew we're measuring)?
3. **Exposed-vs-hidden NCCL math** — union 554 ms, exposed 530 ms, "sum/union=1.0×". Verify the
   interval-union and compute-overlap subtraction logic against a hand query.
4. **NVTX partition** — do top-level ranges + `untracked` sum to wall per iter? Any
   double-counting between top-level and nested (`·`) ranges? Is `allreduce_metrics` 44.6%
   arithmetically consistent with `forward_backward` 47.3% + others + untracked 5.9%?
5. **CUDA-graph attribution** — capture uses `--cuda-graph-trace=node`. Are kernels inside graph
   replays attributed to the right iteration and NVTX range, or smeared?
6. **Iteration segmentation** — how are iteration boundaries detected? Confirm robustness at the
   capture-window edges (delay=720 s, duration=60 s ⇒ partial first/last iters).
7. **Rank↔device mapping** in per-rank tables (compute / all-busy / NCCL-exposed). Rank 7 shows
   all-busy 474 ms vs ~857 ms for others — consistent with "last arriver spins least", but verify
   the mapping isn't permuted.
8. **H2D DMA section** — bytes/iter 334 MB, bandwidth 25.5 GB/s, "0% hidden". Sanity-check the
   memcpy classification and the exposed computation (25.5 GB/s is near PCIe gen5 x16 practical
   ceiling — plausible but verify it isn't NVLink or pageable traffic misclassified).
9. **Capture script** — env propagation to all 8 ranks (`METRICS_SYNC_PERIOD`,
   `DINOV3_PERRANK_DIAG`), NVTX enabled via `--profiling`, delay/duration window vs actual
   startup time (did the window land in steady state for all past captures, not just 80711?).

## Method (suggested)

1. Read both scripts end-to-end first; map the data flow sqlite → tables.
2. Re-derive 3–4 headline numbers of the 80711 report **independently** with direct
   `sqlite3` queries against `ddp-cg-bs192-w24pf8-80711.sqlite` (nsys export schema:
   `CUPTI_ACTIVITY_KIND_KERNEL`, `CUPTI_ACTIVITY_KIND_MEMCPY`, `NVTX_EVENTS`,
   `CUPTI_ACTIVITY_KIND_RUNTIME`, StringIds join). Compare.
3. Check invariants per iteration: Σ(top-level NVTX) + untracked ≈ wall; GPU busy + gap ≈ wall;
   NCCL exposed ≤ NCCL union ≤ wall.
4. Classify every finding: **BUG** (wrong number) / **MISLEADING** (correct number, wrong
   implied interpretation) / **OK**. For each BUG, state which past Phase 7 conclusions it
   touches (see `docs/phase7_perf_plan.md` ladder, lines ~25–40, and §7.8).
5. Write the audit report to `docs/phase7_harness_audit.md` (findings table + evidence:
   queries run, numbers compared). Machine-readable summary block at top.

## Constraints (from CLAUDE.md — binding)

- Verify claims in actual data/code before writing them; cite paths, line numbers, and query
  output in the report.
- **Codex-gate the audit conclusions before finalizing the report**: `codex exec --model gpt-5.5
  -c model_reasoning_effort=high --sandbox read-only` with the draft findings; verify any
  flagged mechanism error yourself in code/data before accepting.
- Use `fff` MCP tools / LSP for repo searches, not raw grep/find (shell grep on Weka
  logs/sqlite output is fine — those aren't git-indexed).
- Writing style rules in `~/CLAUDE.md` apply to the report.
- Do not modify anything under `/mnt/weka/adovlatyan/nsys_profiles/` (evidence). Scratch work
  goes in the session scratchpad or `/mnt/weka/adovlatyan/nsys_stats/`.

## What hangs on this audit

- If the NVTX/segmentation math is wrong, the 32.6%→44.6% `allreduce_metrics` narrative and the
  batch-residual growth numbers (415→445→506 ms) need restating.
- The **rank-7 straggler finding is only partially harness-dependent**: `[RANKDATA]`
  (data_time 0.37–0.42 s on rank 7 vs ~0.3 ms on ranks 0–6, iters 690/700/710) comes straight
  from stderr and stands on its own. The harness supplies the corroborating per-rank
  NCCL-exposed inversion. State clearly in the report which conclusions survive even if bugs
  are found.

## Definition of done

`docs/phase7_harness_audit.md` exists with: findings table (BUG/MISLEADING/OK per item above),
independent re-derivations for the headline numbers, Codex review outcome, and a short list of
script fixes (if any) with proposed patches described but not necessarily applied.
