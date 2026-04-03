# Performance Bottleneck Ledger

This file tracks the current causal model of step time.

Use it to answer:

1. What is currently exposed on the critical path?
2. What type of bottleneck is it?
3. Which knobs can plausibly move it?
4. Which tempting ideas should be deferred?

## Current Summary

**Current dominant bottleneck**: Memory allocator fragmentation was masking the true DDP bs=256
ceiling. After adding `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the DDP bs=256 bimodal
instability is eliminated and DDP becomes the dominant strategy at all tested batch sizes.

Last updated from:

- run: expandable_segments screening (jobs 9650–9657)
- date: 2026-04-03
- artifacts: `/mnt/weka/adovlatyan/logs/ddp-es-9650.out` through `fsdp2-es-9657.out`

Current decision:

- **DDP bs=256 + expandable_segments is the new recommended operating point** — 24.5% MFU, 4229 img/s, 66 GB
- **DDP bs=128 + expandable_segments** is the best memory-efficient option — 22.9% MFU, 3993 img/s, 34 GB
- FSDP2 is now *beaten* at all batch sizes. With expandable_segments it degrades an additional 7–12%.
- CUDA graphs are NOT viable (multi-crop tensor reuse breaks CUDAGraph trees)
- Activation checkpointing is an enabler for larger batches but loses throughput at equal batch
  and did not produce a net win in the tested FSDP2 regime
- bs=320 OOMs for both DDP and FSDP2 even with expandable_segments — hard memory ceiling at bs=256

## Profiler Baseline (Job 7553)

Config: 8×H100, bs=64/GPU, torch.compile=True, FSDP2 SHARD_GRAD_OP, real Weka data, ViT-B.
Profiler window: 3 active steps (iters 5-7), post-compile warmup.
Per-step CUDA time: ~706ms (rank 0), ~647ms (rank 7). Step wall time: ~213-247ms.

**Note on CUDA time vs wall time**: CUDA time sums all kernel durations including overlapped ops.
Wall time is lower because NCCL comms overlap with compute. The profiler's "Self CUDA %" reflects
kernel time on the GPU, not necessarily exposed/serial time. Actual exposed NCCL cost is lower
than the raw 47% because FSDP2 prefetches all_gather during compute. Still, NCCL is clearly dominant.

### Time Breakdown (rank 0, 3 steps averaged)

| Category | Self CUDA ms | % of CUDA | Notes |
|---|---|---|---|
| NCCL all_gather (FSDP unshard) | 203 | 28.7% | 129 calls — per-block forward+backward |
| NCCL reduce_scatter (FSDP grad) | 86 | 12.2% | 45 calls — gradient scatter |
| NCCL all_reduce (loss/metrics) | 42 | 5.9% | 48 calls — includes async loss centers |
| **Total NCCL** | **331** | **46.8%** | |
| aten::mm (matmul fwd) | 78 | 11.0% | 480 calls — QKV, proj, FFN |
| sm90 gemm kernel | 51 | 7.3% | BF16 tensor core matmuls |
| aten::addmm (linear layers) | 32 | 4.6% | 198 calls — heads, misc |
| flash_attention_backward | 28 | 4.0% | 72 calls — SDPA backward |
| Triton fused layernorm+cat | 30 | 4.2% | 72 calls — compiled kernels |
| AdamW optimizer step | 38 | 5.4% | Per-step optimizer |
| **Total compute** | **~257** | **~36%** | |
| Other (overhead, scheduling) | ~118 | ~17% | |

## Step 5 Screening Summary (2026-04-02)

### Results Table

| Run | Job ID | bs/GPU | cudagraphs | ckpt | MFU % | img/s | step_ms | max_mem GB | Output / Log Path | Status |
|-----|--------|--------|-----------|------|-------|-------|---------|-----------|-------------------|--------|
| Baseline | 7553 | 64 | false | false | 13.8 | 2404 | 213 | 18.0 | `/mnt/weka/adovlatyan/output_profile_7553`; traces: `/mnt/weka/adovlatyan/profiler_traces/2026-04-02/7553/` | OK |
| B | 7563 | 128 | false | false | 20.9 | 3647 | 279 | 33.0 | `/mnt/weka/adovlatyan/output_screen_bs128_cgfalse_ckptfalse_7563` | OK |
| C | 7564 | 64 | true | false | — | — | — | — | N/A (failed before output dir) | FAILED (cudagraph tensor overwrite) |
| D | 7565 | 128 | true | false | — | — | — | — | N/A (failed before output dir) | FAILED (same) |
| **E** | **7566** | **256** | **false** | **false** | **23.5** | **4106** | **498** | **65.6** | `/mnt/weka/adovlatyan/output_screen_bs256_cgfalse_ckptfalse_7566` | **BEST throughput** |
| F | 7567 | 384 | false | false | — | — | — | — | N/A (OOM on iter 0) | OOM (78 GB) |
| G | 7573 | 320 | false | false | — | — | — | — | N/A (OOM on iter 0) | OOM (77 GB) |
| H | 7574 | 256 | false | true | 21.4 | 3734 | 548 | 37.3 | `/mnt/weka/adovlatyan/output_screen_bs256_cgfalse_ckpttrue_7574` | -9% vs E |
| I | 7593 | 384 | false | true | 8.1 | 1428 | 2074 | 55.8 | `/mnt/weka/adovlatyan/output_screen_bs384_cgfalse_ckpttrue_7593` | -65% vs E |
| J | 7594 | 512 | false | true | — | — | — | — | N/A (OOM) | OOM (75 GB even w/ ckpt) |

### Key Findings

1. **Batch scaling is the dominant knob**: 64→256 = +70% MFU, +71% throughput
2. **Diminishing returns**: 64→128 gained +51% MFU; 128→256 gained only +12% more
3. **CUDA graphs incompatible**: multi-crop architecture uses `forward_features_list` which
   processes global and local crops sequentially — CUDAGraph tree can't handle the tensor reuse
4. **Checkpointing loses throughput**: At equal batch size (256), checkpointing costs ~10% throughput
   (-9% img/s) but saves ~28 GB memory. Not worth it unless batch can go much higher to compensate.
5. **OOM boundary**: bs=256 (65.6 GB) fits; bs=320 (77 GB) OOMs. Max without checkpointing is ~280.
6. **Bimodal MFU pattern**: Both bs=256 runs show ~11% MFU for iters 10-50, then ~23% for 60-99.
   Likely cause: torch.compile reoptimization or CUDA allocator defragmentation on large allocations.

### What Emerged as Promising

1. **DDP switch** (highest value) — eliminates FSDP2 all_gather/reduce_scatter overhead
2. **Batch size = 256** — optimal under current FSDP2, near memory ceiling

### What Was Deprioritized

1. **CUDA graphs** — not viable without architectural changes to multi-crop forward pass
2. **Activation checkpointing** — loses throughput at equal batch; only useful as enabler for
   much larger batches under DDP where memory ceiling is different
3. **Batch sizes beyond 256** — OOM without checkpointing; with checkpointing the net gain is negative

## Active Ledger

```text
Name: FSDP2 communication overhead (all_gather + reduce_scatter)
Evidence: Job 7553 profiler — 289ms NCCL (all_gather + reduce_scatter) per step = 41% of CUDA time.
  ViT-B is 172 MB in BF16 — fits trivially on 80 GB H100. FSDP2 shards a model that does
  not need sharding, paying communication cost with zero memory benefit.
Approx Exposed Share: ~40-50% of step time (partially overlapped with compute, but still dominant)
Bound Type: Communication-bound (NCCL)
Candidate Knobs:
  1. Switch to DDP (eliminates all_gather + reduce_scatter; retains one all_reduce ~42ms)
  2. FSDP2 FULL_SHARD → SHARD_GRAD_OP (already using SHARD_GRAD_OP, so no further reduction here)
  3. Overlap improvements (FSDP2 prefetch tuning) — limited upside, doesn't fix the root cause
Prerequisites: DDP wrapper code (already supported by PyTorch, needs integration in ac_compile_parallelize.py)
Why It Might Matter: This is the single largest cost category. Removing it would ~1.6× throughput.
Why It Might Not Matter: If compute time grows (larger batch, checkpointing), comms become a smaller fraction.
  But at current config, this is clearly the dominant bottleneck.
Decision: HIGH PRIORITY — benchmark DDP vs FSDP2 as the next optimization.
```

```text
Name: Matmul compute (aten::mm + sm90 gemm + aten::addmm)
Evidence: ~161ms combined = ~23% of CUDA time. These are the actual ViT forward/backward matmuls.
Approx Exposed Share: ~23% of CUDA time (this IS the useful compute)
Bound Type: Compute-bound (already using BF16 tensor cores, TF32, flash attention)
Candidate Knobs:
  1. Batch size increase (amortizes fixed per-step overhead, increases arithmetic intensity)
  2. Sequence packing (eliminates padding waste — but ViT has no padding)
  3. FP8 (already has config support, but not yet validated)
Prerequisites: Resolve communication bottleneck first — matmul efficiency gains are second-order
Why It Might Matter: After fixing comms, this becomes the dominant cost (as it should be).
Why It Might Not Matter: Matmuls are already running on tensor cores with BF16 — not much room for kernel-level optimization.
Decision: MONITOR — becomes the target after NCCL overhead is removed.
```

```text
Name: Optimizer step (AdamW)
Evidence: 38ms = 5.4% of CUDA time per step (rank 0), 27ms (rank 7).
Approx Exposed Share: ~5%
Bound Type: Memory-bandwidth-bound (element-wise ops over all parameters)
Candidate Knobs:
  1. Fused multi-tensor optimizer (already enabled: cfg.optim.multi_tensor_optim=true)
  2. Reduce parameter count (not applicable — ViT-B is the target architecture)
Prerequisites: None
Why It Might Matter: After removing NCCL overhead, this becomes a ~10% share.
Why It Might Not Matter: 38ms is already fast for full ViT-B parameter update. Multi-tensor is enabled.
Decision: LOW PRIORITY — already optimized.
```

```text
Name: Flash attention backward
Evidence: ~31ms = 4% of CUDA time (72 calls across student/teacher forward+backward).
Approx Exposed Share: ~4%
Bound Type: Compute/memory-bandwidth-bound (FA2 is already in use via SDPA)
Candidate Knobs: None practical — already using flash_attention via SDPA.
Prerequisites: None
Why It Might Matter: Small but real cost.
Why It Might Not Matter: Already using the best available kernel (FA2 on H100).
Decision: NO ACTION — already using flash attention.
```

```text
Name: Activation checkpointing (currently disabled)
Evidence: Not a bottleneck — it's a potential enabler. Max allocated=16.8 GB out of 80 GB.
  Currently using only ~21% of GPU memory. Enabling checkpointing would free memory
  for larger batch sizes.
Approx Exposed Share: N/A (enabler, not a cost)
Bound Type: N/A
Candidate Knobs:
  1. Enable train.checkpointing=true — selective AC per transformer block
  2. Enable train.checkpointing_full=true — full recompute
Prerequisites: Only valuable if memory is the constraint on batch scaling
Why It Might Matter: If batch_size_per_gpu=64 is not the optimal, checkpointing enables 128+.
Why It Might Not Matter: At 16.8 GB / 80 GB, there's plenty of headroom even without checkpointing.
Decision: DEFER — only explore if batch scaling hits OOM.
```

## Deferred / Low-Value Work

1. **CUDA graph capture**: TESTED AND FAILED. Multi-crop architecture (forward_features_list) processes
   global and local crops sequentially, causing tensor reuse that breaks CUDAGraph trees.
   Would require `torch.compiler.cudagraph_mark_step_begin()` calls or restructuring the forward pass.
2. **Data pipeline optimization**: data_time=1.1-1.5s (wall clock per print interval, not per iter). Data loading is not on the critical path with 20 workers + prefetch_factor=8.
3. **Sequence packing**: ViT patches are fixed-size, no padding waste. Not applicable.
4. **Mixed precision tuning**: Already BF16 params + FP32 reduction. FP8 is a future option but not the current bottleneck.
5. **Activation checkpointing**: TESTED. Saves ~28 GB at bs=256 but costs ~10% throughput.
   Only worth revisiting if DDP frees enough memory to push batch sizes much higher (e.g. bs=512+).

## Step 6 Branch Decision (2026-04-02)

**Decision: Split to a new branch for DDP vs FSDP2 work.**

Rationale per plan Step 6:
- The next move changes distributed strategy (FSDP2 → DDP)
- This is risky enough that it may need to be abandoned cleanly
- The `mfu-tracking-baseline` branch should be committed as-is with profiling infra + screening results

New branch name: `perf-ddp-vs-fsdp`

DDP benchmark results (2026-04-03):

| Strategy | Job ID | bs/GPU | MFU % | img/s | max_mem GB | Output / Log Path |
|----------|--------|--------|-------|-------|-----------|-------------------|
| FSDP2 | 7553 / 7563 / 7566 | 64 / 128 / 256 | 13.8 / 20.9 / 23.5 | 2404 / 3647 / 4106 | 18.0 / 33.0 / 65.6 | See Step 5 table above |
| DDP | 9630 | 64 | 18.1 | 3169 | 17.7 | `/mnt/weka/adovlatyan/output_ddp_bs64_ckptfalse_9630`; SLURM: `/mnt/weka/adovlatyan/logs/ddp-9630.out` |
| DDP | 9631 | 128 | 23.1 | 4042 | 34.0 | `/mnt/weka/adovlatyan/output_ddp_bs128_ckptfalse_9631`; SLURM: `/mnt/weka/adovlatyan/logs/ddp-9631.out` |
| DDP | 9632 | 256 | 12-18 | 2113-3157 | 66.5 | `/mnt/weka/adovlatyan/output_ddp_bs256_ckptfalse_9632`; SLURM: `/mnt/weka/adovlatyan/logs/ddp-9632.out` |

Key insight: DDP wins at bs≤128 by eliminating per-block all_gather/reduce_scatter.
But at bs=256, FSDP2's overlapped per-block communication actually outperforms DDP's
single end-of-backward all-reduce.

**Recommended config (before expandable_segments)**: DDP bs=128 (23.1% MFU, 34 GB)

---

## expandable_segments Screening Results (2026-04-03)

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was tested across all prior configs.

### expandable_segments Full Comparison Table

| Strategy | bs/GPU | ES | MFU % | img/s | step_ms | max_mem GB | Δ vs no-ES |
|----------|--------|----|-------|-------|---------|-----------|-----------|
| DDP | 64 | no | 18.1 | 3169 | 162 | 17.7 | — |
| DDP+ES | 64 | yes | 17.5 | 3061 | 157–186 | 17.6 | -3% |
| DDP | 128 | no | 23.1 | 4042 | 253 | 34.0 | — |
| DDP+ES | 128 | yes | 22.9 | 3993 | 249–253 | 33.9 | -1% |
| DDP | 256 | no | 12–18 (bimodal) | 2113–3157 | 586–933 | 66.5 | — |
| **DDP+ES** | **256** | **yes** | **24.5** | **4229** | **475–478** | **66.5** | **FIXED** |
| FSDP2 | 64 | no | 13.8 | 2404 | 213 | 18.0 | — |
| FSDP2+ES | 64 | yes | 12.1 | 2110 | 235–246 | 16.7 | -12% |
| FSDP2 | 128 | no | 20.9 | 3647 | 279 | 33.0 | — |
| FSDP2+ES | 128 | yes | 18.4 | 3213 | 315–330 | 33.0 | -12% |
| FSDP2 | 256 | no | 23.5 | 4106 | 498 | 65.6 | — |
| FSDP2+ES | 256 | yes | 21.8 | 3809 | 521–532 | 65.6 | -7% |
| DDP+ES | 320 | yes | OOM | — | — | >80 GB | — |
| FSDP2+ES | 320 | yes | OOM | — | — | >80 GB | — |

### Key Findings

1. **DDP bs=256 + ES eliminates bimodal instability** — the periodic MFU collapse (12–18% bimodal)
   was caused by CUDA allocator defragmentation. `expandable_segments` allows the allocator to grow
   existing segments rather than reallocate, preventing the defrag pauses. Result: stable 24.5% MFU.

2. **`expandable_segments` is harmful for FSDP2** across all batch sizes (7–12% regression).
   FSDP2's fine-grained per-block alloc/free pattern apparently conflicts with the segment growth
   heuristic — likely causes over-reservation that wastes bandwidth.

3. **bs=320 OOMs regardless of ES** — expandable_segments is a runtime allocation strategy, not
   a compile-time optimization. torch.compile's peak memory during iter 0 exceeds 80 GB at bs=320.
   The hard ceiling is confirmed at bs=256 (≈66 GB peak for DDP).

### Updated Recommended Configs

| Use case | Config | MFU | img/s | Memory |
|----------|--------|-----|-------|--------|
| **Best throughput** | DDP bs=256 + ES | **24.5%** | **4229** | 66 GB |
| **Memory-efficient** | DDP bs=128 + ES | 22.9% | 3993 | 34 GB |
| **Reference (old)** | FSDP2 bs=256 | 23.5% | 4106 | 66 GB |

**Rule**: Always set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when using DDP.
Never use it with FSDP2.
