# Phase 8 — Optimization Candidate Ladder

**Date**: 2026-07-10
**Source**: GPT-5.6-sol review at xhigh reasoning (Codex CLI 0.144.1, read-only, repo-grounded — it read
`train.py`, `ssl_meta_arch.py`, `ac_compile_parallelize.py`, `loaders.py`, `phase7_perf_plan.md` and cited
line numbers). Every mechanism claim below was re-verified in the working tree before being recorded here.
**Input brief**: compressed Phase 6/7 state summary (champion config, residual attribution post harness-audit,
7.9 straggler verdict, deferred/rejected levers).

## Framing (accepted)

Phase 8 is a **two-stage program: remove host/storage arrival skew first, then optimize actual DDP
communication.** The "exposed NCCL ≈ 56% of p50 wall" figure is largely *where late-rank waiting appears*
(NCCL LL kernels spin-wait), not 56% transferable communication. Comms-side experiments (buckets,
`CUDA_DEVICE_MAX_CONNECTIONS`, NVLS) are uninterpretable until rank-arrival skew is cleaned — a late rank
makes every collective look slow.

## Ranked candidates

Gain estimates are against steady-state **wall** time (never CUDA-event MFU).

| # | Lever | Mechanism / expected gain | Effort | First cheap probe | Status |
|---|---|---|---|---|---|
| 1 | **Root IRQ steering** — storage-NIC (mlx5) IRQs → Slurm-reserved cores 48-55/104-111 + irqbalance ban, gpu01–08 | Removes NET_RX softirq tax from job cpusets (7.9 root cause). **8–20% wall**; ~25% upside suggested (not proven) by 345 vs 472 ms clean-iter p50 at bs192 | Medium ops: root, audit doc + reversal plan (perf-sysctl pattern) | Steer only storage-NIC queues; verify `/proc/interrupts` + softirq counters move; matched same-node w16 no-slice A/B at bs160/192; judge raw wall + `[RANKDATA]` tails | **Awaiting user go** (shared infra) |
| 2 | DataLoader worker-runtime hygiene: `cv2.setNumThreads(1)` in workers, prefetch-factor screen, lazy per-worker HDF5 reopen (`hdf5_dataset.py:41` opens at construction) | Reduces per-worker CPU waste under a taxed/shared CPU budget. Part of Codex's 3–12% bundle (bulk of that bundle was `in_order=False`, now rejected — expect the low end) | Low–medium | One arm each on matched node, w16/pf8 baseline | Open |
| ~~2a~~ | ~~`in_order=False` on DataLoader~~ | ~~Slow worker can't block ready batches from peers~~ | — | — | **REJECTED 2026-07-10 (user decision)** — see Decision log below |
| 3 | Measure real DDP bucket overlap | Backbone + heads are separate DDP instances, default 25 MiB buckets (`ac_compile_parallelize.py:347` — no `bucket_cap_mb`). cudagraphs is **per-block** capture, so hooks can fire between block backwards — overlap is measurable, not assumed. **0–8%** (zero possible) | Low–medium; exact-gradient path | Dump `_get_ddp_logging_data()` per wrapper at steady state; then screen 8/16/25/50 MiB caps. Also count buffer broadcasts; test `broadcast_buffers=False` after asserting buffer equality (RoPE periods etc. look immutable) | Open |
| 4 | Remove both per-iter host syncs together | 7.3 metric gating alone was flat because waiting relocated to `step_end_event.synchronize()` (`train.py:868`). Untested combo: `metrics_sync_period>1` **plus** event-ring timing read every N iters → host can launch ahead. **0–4%** | Medium (NaN detection becomes delayed unless redesigned) | Period-10 both; compare CPU launch-ahead / GPU gaps in nsys; flat result closes it | Open |
| 5 | Re-test `CUDA_DEVICE_MAX_CONNECTIONS=1` vs 8 on the cleaned recipe | "Validated optimal" (2116 vs 1953) was one run per arm, w24, before loader/IRQ contamination was known. **0–5%** | Very low | Two same-node replicates per setting at bs192, w16 (ideally post-IRQ-fix); rank by raw wall | Open — after #1 or at least on w16 |
| 6 | Validate NVLS; do not broadly force NCCL env vars | NVLS is the only SHARP path relevant on a single NVSwitch node. Unset `NCCL_NVLS_ENABLE` = automatic, not disabled — so first *verify* what NCCL actually selects. **0–5%** after skew cleanup | Low | `NCCL_DEBUG=INFO` (algorithm/NVLS selection, NCCL version); `all_reduce_perf` at bucket-sized payloads, default vs forced NVLS | Open — after #1 |
| 7 | Fused AdamW | `build_optimizer()` (`train.py:110`) uses foreach default; fused shrinks the ~10 ms optimizer phase. **0.3–1.5%** | Very low | One `fused=True` arm; check first-step state, checkpoint resume, reference losses. EMA already `_foreach_`-fused (~0.7 ms) — leave it | Open |
| 8 | Teacher/student forward stream overlap | Independent until loss construction (`ssl_meta_arch.py:416`), but both are large ViT workloads on the same SMs. **0–5%, likely flat/negative** | High (cuda-graph trees, allocator, collective ordering) | Single-GPU microbench: teacher on side stream, join before losses, exclude Sinkhorn collectives; continue only if ≥5% | Exploratory only |
| 9 | H2D copy-stream double-buffer | Exposed H2D bounded at 5.7–12.3 ms/iter → **0.5–2.2%** | High vs payoff | Re-measure exposed H2D after IRQ cleanup; build only if >2% wall | Stays deferred |

## Killed levers (frozen-convergence constraint) — verified in code

- **bf16 gradient-compression DDP hooks**: dead. The DDP path casts params to bf16 *before* wrapping
  (`ac_compile_parallelize.py:336-343`), so gradients already reduce in bf16; a hook saves zero bytes.
- **fp32-reduce tuning**: `reduce_dtype` is FSDP2-policy-only (`ac_compile_parallelize.py:269`); the DDP
  path never uses it. Implementing fp32 DDP reduction would double traffic.
- **PowerSGD**: approximate gradients + error-feedback state; conflicts with convergence-parity requirement.

## Corrections to our own framing (verified, adopt in future docs)

1. **`train.cudagraphs=true` is per-block CUDA-graph capture, not full-step.** `wrap_compile_block()` does
   `module.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": True})` per transformer
   block (`ac_compile_parallelize.py:71`). Do not claim "all gradients appear at replay end".
2. **56% exposed NCCL ≠ 56% comms opportunity** — it is where straggler wait shows up (spin-wait in LL
   kernels; lands in whichever collective the late rank hits).
3. **Logged `mfu` in `training_metrics.json` is CUDA-event-based** (`step_start_event`→`step_end_event`).
   "Wall MFU" is a separate offline recomputation.
4. **7.9 diagnosed the straggler; it is not remediated** until IRQ steering runs. The bs192 w16 −17% is
   cross-node; only the bs160 −11%/−6% is node-controlled.

## Decision log

- **2026-07-10 — `in_order=False` REJECTED (Aram).** Data ordering for transformer pre-training is part of
  the training recipe, not a free knob: reproducibility and the data order/mix matter for convergence
  (the same reason pre-/mid-/post-training curricula exist). Out-of-order batch consumption would also
  break the `sampler_advance = iterations × batch` resume contract. Throughput is not worth either cost.
  Standing rule: **no optimization that changes which samples arrive in which order.**
- 2026-07-10 — rest of the ladder accepted as reasonable by Aram. IRQ steering (#1) still awaiting explicit go.

## Execution sketch

1. User decision on #1 (IRQ steering). If go: prepare audit doc + reversal procedure, steer, A/B same-node.
2. In parallel, rootless one-day screens: #7 fused AdamW, #3 bucket-logging dump, #2 worker hygiene arms.
3. Post-IRQ-fix: #5 MAX_CONNECTIONS re-test, #3 bucket-cap screen, #6 NVLS validation; re-decide bs160 vs bs192.
