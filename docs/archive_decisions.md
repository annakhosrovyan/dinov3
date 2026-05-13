# Performance Work Archive — Decisions, Errata, and Phase History

Compressed record of the early performance optimization work on the `mfu-tracking-baseline` and
`perf-ddp-vs-fsdp` branches. The current authoritative state lives in `CLAUDE.md`, `learnings/`,
and `docs/perf_experiment_log.md`. This file is append-only historical reference.

---

## MFU Implementation — Errata (important for reading old logs)

Two convention errors were made and corrected during the initial MFU implementation:

**Error 1 — MAC vs hardware FLOP (corrected 2026-03-31, commit `6c943fc`)**
`compute_dino_flops_per_image()` returns MACs (1 MAC = 1 multiply-add, matching fvcore/DINOv2).
The denominator must be in hardware FLOPs (1 multiply-add = 2 FLOPs). A 2× conversion factor
was missing. Effect: all pre-fix MFU logs are 2× too low.

**Error 2 — H100 TFLOPS denominator (corrected 2026-04-01, commit `cdea2dc`)**
NVIDIA's published 1979 TFLOPS assumes 2:4 structured sparsity. Dense BF16 matmuls (all
standard transformer training) peak at **989 TFLOPS**. The initial code used 1979. Effect: all
logs between the two fixes are still 2× too low vs the correct denominator.

**Corrected baseline numbers** (2-GPU job 6585, 8-GPU job 6770, bs=64):
- 2-GPU synthetic: ~9.9% hardware MFU (dense)
- 8-GPU real data: ~11.3% MFU overall, ~12–13.4% steady-state

Any run logged before 2026-04-01 shows MFU values at ¼ of the corrected hardware-dense MFU.

---

## Profiling Infrastructure — What Was Built (Step 2, 2026-04-02)

Files added on branch `mfu-tracking-baseline` relative to master:

| File | Role |
|------|------|
| `dinov3/utils/profiling.py` | NVTX wrapper, PyTorch profiler builder, memory stats, run metadata |
| `dinov3/train/train.py` | Profiler setup, NVTX ranges, extended memory metrics, `--profiling` flag |
| `dinov3/train/ssl_meta_arch.py` | `set_nvtx()`, 5 inner-phase NVTX ranges |
| `scripts/profiling_run.sh` | 8-GPU 15-iter profiling script with trace export |

NVTX hierarchy (used for Nsight Systems analysis):
```
forward_backward
  ├── H2D_transfer
  ├── teacher_fwd
  ├── student_fwd
  ├── gram_fwd          (conditional)
  └── losses_backward
grad_clip
allreduce_metrics
optimizer_step
ema_update
schedule_update
```

Profiling is zero-cost when not active (`--profiling` flag). Default training path unchanged.
Trace artifacts from job 7553: `/mnt/weka/adovlatyan/profiler_traces/2026-04-02/7553/`.

---

## Training Setup State as of 2026-04-03

Captured here because several items were "not yet done" at the time and are now resolved:

| Item | State at 2026-04-03 | Resolved? |
|------|---------------------|-----------|
| `sharded_eval_checkpoint` | `false` in default config, not set in run.sh | **Yes** — added to `run.sh` after soak tests confirmed it avoids `full_tensor()` materialization at eval |
| Async eval | Not implemented | **Deprioritized** — inline eval with sharded checkpoint is sufficient |
| `compile_mode` screening | Only default `null` mode tested | **Closed** — `max-autotune-no-cudagraphs` tested and failed due to iBOT dynamic mask shapes; `null` is the only viable mode |
| `expandable_segments` on FSDP2 | Not tested | **Closed** — tested, hurts ZeRO-3 FSDP2 by 7–12%; re-test warranted with no-release mode |

---

## Performance Optimization Phase Summary

Work completed in order across `mfu-tracking-baseline` → `perf-ddp-vs-fsdp`:

| Phase | Action | Outcome |
|-------|--------|---------|
| 1 | MFU tracking implementation | Done. Corrected conventions; 8-GPU baseline ~11.3% MFU |
| 2 | Profiling infrastructure | Done. NVTX + PyTorch profiler. NCCL at 47% of CUDA time in ZeRO-3 FSDP2 |
| 3 | DDP vs FSDP2 initial screening | DDP eliminates NCCL all_gather overhead; bs=256 vs bs=64 was the dominant driver |
| 4 | Batch-size scaling | bs=64→256 captured most throughput gain |
| 5 | expandable_segments + DDP | Fixed allocator fragmentation stalls at bs=256; required for DDP at large bs |
| 6 | compile_mode screening | max-autotune-no-cudagraphs rejected; null is the only viable mode |
| 7 | Worst-case memory profiling | DDP+ES bs=256 = 67.2 GB worst-case; FSDP2 bs=128 = 36.3 GB |
| 8 | 500-iter soak test | DDP+ES bs=256 confirmed: 23.9% MFU, zero memory creep, zero alloc_retries |
| 9 | DDP vs FSDP2 revalidation | Tim Darcet confirmed both are fine; 0.4 pp gap is from ZeRO-3 overhead |
| 10 | FSDP2 long soak (job 29703) | bs=128 ZeRO-3: zero fragmentation at 1300+ iters; colleague OOM was config-specific |
| 11 | FSDP2 no-release screening (jobs 31102/31103) | ~23.2% MFU at bs=256 — no gain over ZeRO-3 (23.5%) at matched bs; memory virtually identical (~64 vs ~66 GB at bs=256). Closed. |

Current production config: `run.sh` — DDP+ES bs=256, 23.9% MFU (soak-tested 500 iters, 67.2 GB).

**Memory note**: DDP+ES bs=256 uses 67.2 GB — tight, ~13 GB headroom. FSDP2 ZeRO-3 bs=256 uses
~65.6 GB — equally tight. FSDP2's real memory advantage is at bs=128 (36.3 GB, 43 GB headroom),
which makes it the safer choice for full training runs and the only viable option for multi-node.
At bs=256, FSDP2 and DDP are memory-equivalent. The MFU comparison at bs=256 (FSDP2 23.5% vs
DDP+ES 23.9%) is also only from 100-iter screening runs — not soak-tested.

FSDP2 ZeRO-3 bs=128 is the correct production reference for FSDP2 (not bs=256).
FSDP2 no-release: closed — same memory as ZeRO-3 at matched bs, no throughput benefit.

---

## Operating Principles That Remain Valid

From the original execution plan:

1. Trace the exposed step time before changing anything.
2. Classify the limiter (compute, memory bandwidth, NCCL, data IO, Python overhead).
3. Estimate the step-time share.
4. Change only the knobs that attack the exposed bottleneck.
5. Non-recipe-changing work should not displace recipe-level experiments (crop count, etc.).
