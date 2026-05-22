# DINOv3 Perf-Work Branch History

Single source of truth for the perf-optimization branch chain. Update this file whenever
a new phase branch is cut or a phase is closed.

Last updated: 2026-05-22

## Exit strategy

We are in heavy dev mode and **cannot merge to `master` directly** for the foreseeable future.
Phase branches form a linear forward chain. The eventual exit, once the DDP + fullgraph
work is wrapped, is:

1. Branch from `master` directly.
2. Cherry-pick the load-bearing commits (MFU infra, DDP wrapping, whatever Phase 6 produces)
   onto that branch.
3. Open a PR to `master`.

The phase chain below is *dev history*, not what gets merged. Only the final curated PR has
to be merge-clean.

## Branch chain (linear forward; each branch is a descendant of the one above)

| Branch | Role | Head commit | State |
|---|---|---|---|
| `master` | Upstream / eventual PR target | `3e0693a` "optimize satellite data pipeline" | Frozen wrt our work |
| `mfu-tracking-baseline` | MFU instrumentation foundation (`dinov3/utils/mfu.py`, CUDA-event timing in `train.py`, `tests/test_mfu.py`) | `2423391` "preparing planning docs for phase 2" | Frozen |
| `perf-ddp-vs-fsdp` | DDP + FSDP2 wrapping in `ac_compile_parallelize.py`, worst-case memprofile infra | `a281fc8` "Improve worst-case memprofile scripts" | Frozen (3 commits ahead of origin's last sync, all carried forward) |
| `perf-fsdp2-pipeline` | Phase 5 — FSDP2 wrap-up: AC matrix, compile_mode learnings, max-autotune closeout, NCCL nsys, all Phase 5 experiment scripts/docs/HTML ledgers | `045aa22` "Phase 5 closeout, Phase 6 plan opening" | **ARCHIVED 2026-05-21.** No new commits. |
| `perf-ddp-fullgraph` ← **HEAD** | Phase 6 — DDP + `torch.compile(fullgraph=True)` + CUDA graphs (active branch) | `045aa22` (no commits yet, identical to `perf-fsdp2-pipeline`) | Active |

All branches pushed to `origin` (annakhosrovyan/dinov3). Branch and commit metadata is
authoritative in git itself; this table is the human-readable index.

## Phase ledger

### Phase 2 — MFU tracking baseline (closed)

- Branch: `mfu-tracking-baseline`
- Established: MFU instrumentation (`dinov3/utils/mfu.py`), CUDA-event step timing
  (`dinov3/train/train.py:526,558,608–614`), test anchors (`tests/test_mfu.py`).
- Convention locked in: `compute_dino_flops_per_image()` returns MACs; `compute_mfu()`
  multiplies by 2 and uses `H100_BF16_TFLOPS = 989.0` dense.
- See `docs/dinov3-mfu-tracking-initial-brief-03-27-26.md`, `docs/mfu-results-2026-03-30.md`.

### Phase 4 — DDP vs FSDP2 + compile-mode screening (closed)

- Branch: `perf-ddp-vs-fsdp`
- Established: DDP wrapping in `ac_compile_parallelize.py:281–322` (single all-reduce path,
  `gradient_as_bucket_view=True`, `static_graph=True`).
- Closed: `max-autotune-no-cudagraphs` incompatible with iBOT dynamic mask shapes
  (`learnings/compile_modes.md`). `compile_mode: null` is the only viable path under
  the current iBOT stochastic-mask design.
- Initial screening (100-iter, archived) suggested DDP+ES bs=256 ~23.9 % MFU,
  FSDP2 bs=256 ~23.5 %. These figures are now treated as a measurement-convention
  caveat (see Phase 5 §6.1 / Phase 6 §6.1).

### Phase 5 — FSDP2 + Data/Compute Pipeline (closed 2026-05-21)

- Branch: `perf-fsdp2-pipeline`
- Doc: `docs/phase5_perf_plan.md` (status header at top has the closeout)
- Active baseline at close: **FSDP2 ZeRO-3 bs=96** (production-safe, ~12 % MFU steady).
- Key results:
  - nsys traces (jobs 39140 / 39141): NCCL AllGather 69.5 % kernel time, NCCL↔compute
    overlap 11–16 %. Communication is the dominant bottleneck under FSDP2.
  - bs=128 FSDP2 OOM in real long training (researcher report 2026-05-12) — not caught
    by worst-case memprofile script.
  - Selective and full activation checkpointing both verified safe at bs=96 but never
    promoted to `run.sh`.
  - DDP bs=96 calibration (**job 48312**, gpu03): iter 400–999 mean **1,387 img/s,
    7.94 % MFU, step_ms 545 ms, peak alloc 25.8 GB**. This is the current verified
    DDP baseline that Phase 6 must beat.
- Closed by Armen Aghajanyan's input (2026-05-21): FSDP2 is overkill for ViT-B at
  ~85 M params; drop FSDP, use DDP, push throughput via `torch.compile(fullgraph=True)`
  + CUDA graphs.
- Cancelled at close: wrap-up 2×2 grid jobs 51069 / 51165 / 51166 (FSDP2 reshT × bs
  grid). No longer load-bearing.

### Phase 6 — DDP + torch.compile(fullgraph=True) (active, opened 2026-05-21)

- Branch: `perf-ddp-fullgraph`
- Doc: `docs/phase6_perf_plan.md`
- Anchor target to beat: **1,387 img/s, 7.94 % MFU at bs=96** (Phase 5 close, job 48312).
- Aspirational target: lab-wide internal **30 % MFU** (~5.2 k img/s at bs=96, 8×H100).
- Central hypothesis (H6): enabling `fullgraph=True` + `triton.cudagraphs=true` on DDP,
  with the iBOT dynamic-shape blocker resolved or worked around, closes a substantial
  fraction of the 7.94 % → 30 % gap.
- Five-rung experiment ladder (see Phase 6 plan §6):
  - 6.1 Diagnostic (TORCH_LOGS recompile/graph-break audit on current path)
  - 6.2 Backbone-only CUDA graphs (cheap, no code change)
  - 6.3 Static-shape pilot for iBOT (Option B padding by default; A fixed-K and
    C bucketing as alternatives)
  - 6.4 Combined end-to-end fullgraph (the H6 test)
  - 6.5 bs=128 headroom on the fullgraph path (gated on 6.4)
- Open design decisions blocking start of work (see Phase 6 plan §10): shape-fix
  strategy A/B/C, ordering of diagnostic vs experiment-first.

## Key commits to remember (cherry-pick candidates for the eventual master PR)

These are the commits whose contents have to land on the future master PR. Listed in
likely cherry-pick order. Update as Phase 6 produces more.

| Commit | Branch | What it adds |
|---|---|---|
| (range from `mfu-tracking-baseline`) | mfu-tracking-baseline | MFU instrumentation: `dinov3/utils/mfu.py`, `tests/test_mfu.py`, CUDA-event step timing in `dinov3/train/train.py` |
| (range from `perf-ddp-vs-fsdp`) | perf-ddp-vs-fsdp | DDP wrapping path in `dinov3/fsdp/ac_compile_parallelize.py` (`_ac_compile_parallelize_ddp`), `train.distributed_strategy` config |
| `045aa22` | perf-fsdp2-pipeline | Phase 5 closeout + Phase 6 plan + experiment-recipe scripts. Most of this commit is *docs/scripts archive* — only the `CLAUDE.md` perf-priors refresh and `learnings/compile_modes.md` update are runtime-relevant. The HTML ledgers, gpt5-pro context bundles, and Phase 5-specific sbatch scripts are dev history that does NOT need to land on master. |
| (TBD Phase 6 commits) | perf-ddp-fullgraph | Will likely include: iBOT static-shape implementation in `dinov3/data/collate.py` + `dinov3/loss/ibot_patch_loss.py`, a `train.static_ibot_shapes` config flag, possibly a `wrap_compile_block` extension to take fullgraph on heads too, and the final `run.sh` recipe change to enable the winning config. |

## Outstanding cross-phase questions

These are not phase-specific; they cut across the chain and inform future PR scope.

1. **The archived 100-iter DDP convention question** (Phase 5 left open, Phase 6 §6.1
   diagnostic addresses): archived job 9631 family reported ~4 k img/s at bs=128 under
   the same MFU formula. Either (a) short-soak overestimate, or (b) codebase regression
   since April. Not yet resolved.
2. **bs=128 OOM under DDP**: Phase 5's OOM was under FSDP2. DDP memory math is different
   (no resharding, full params per rank). bs=96 sits at 25.8 GB peak; bs=128 linear estimate
   is ~34 GB, still fits 80 GB H100. Untested as of phase open.
3. **Promotion to `run.sh`**: no Phase 5 finding has been promoted to the production
   recipe. Phase 6 will produce one or more candidates; promotion gated on a long-soak
   convergence validation by the researcher.

## How to update this file

- When a new phase branch is cut: add a row to the branch chain table, add a new phase
  section to the ledger, and update the "active" marker.
- When a phase closes: change "Active" → "ARCHIVED [date]" in the branch chain table,
  add the closeout summary to that phase's ledger entry.
- When a load-bearing commit lands: add it to the cherry-pick-candidate table.
- Bump the "Last updated" date at the top.

`origin/<branch>` is the authoritative state for any branch; this file is a human-readable
index, not a substitute for `git log`.
