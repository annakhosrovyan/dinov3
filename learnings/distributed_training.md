# Distributed Training Learnings — AI Systems Perf Eng Ch04

Notes distilled from `~/knowledge-base/ai_systems_perf_engineering/ai_systems_perf_eng_ch04.md`.
Tailored to DINOv3 8×H100 DDP/FSDP2 experiments.

---

## Core Principle: Overlap Communication with Computation

The goal is `time_iter ≈ max(time_compute, time_comm)`, not `time_compute + time_comm`.

| Approach | Backward | Comm | Total | Overlap |
|----------|----------|------|-------|---------|
| Manual all-reduce (no overlap) | 10 ms | 12 ms | 22 ms | 0% |
| DDP (overlap) | 10 ms | 12 ms | ~12 ms | ~50%+ |

The repo supports both DDP and FSDP2. The current production script uses DDP+expandable-segments;
the FSDP2 path uses composable FSDP2 (`fully_shard` from `torch.distributed._composable.fsdp`),
not the legacy `FSDP` class with `ShardingStrategy`.

For FSDP2 with `FULL_SHARD` / ZeRO-3 (`reshard_after_forward=True` per block), AllGather can overlap
with forward compute between blocks. Verify with Nsight Systems timeline: compute kernels and NCCL
ops should interleave, not run sequentially.

---

## NCCL Communication Algorithms (Auto-Selected)

| Algorithm | Best For |
|-----------|---------|
| **Ring** | Large messages (bandwidth-dominated, 2× data/n_gpus per link) |
| **Tree** | Small messages (latency-dominated, O(log N) steps) |
| **NVLSTree** | NVLink domains (tree + NVLink SHARP offload) |
| **PAT** | Large data + many GPUs (ring throughput + tree latency) |

NCCL auto-selects; override with `NCCL_ALGO=NVLSTree,PAT` only if benchmarked benefit confirmed.

---

## Critical Environment Variables

```bash
# Debug (verbose, not production):
export NCCL_DEBUG=INFO               # See which network path NCCL uses
# Must see "NET/IB" → InfiniBand/NVLink path; "NET/Socket" = TCP fallback = bad

# Performance:
export NCCL_ASYNC_ERROR_HANDLING=1   # Graceful rank failure handling
export NCCL_SOCKET_IFNAME=ib0        # Force InfiniBand interface (if multiple NICs)
export NCCL_IGNORE_CPU_AFFINITY=1    # Let NCCL place threads optimally

# Multi-NIC (if applicable):
export NCCL_NSOCKS_PERTHREAD=2       # 2 NICs → 2 sockets per thread
export NCCL_SOCKET_NTHREADS=2        # Product must not exceed 64

# SHARP (in-network aggregation, if available):
export NCCL_SHARP_DISABLE=0          # Keep enabled; gives 2-5× speedup at 32+ nodes
```

**Never leave these in production:**
```bash
export NCCL_P2P_DISABLE=1            # Kills intranode GPU-direct P2P
export NCCL_SHM_DISABLE=1            # Increases latency 10-100×
```

---

## Pitfalls & Fixes

**Pitfall: Creating NCCL communicators every iteration**
- Cost: 48 ms overhead per init (real measured: total iter = 48.5 ms vs 0.5 ms without)
- Fix: `dist.init_process_group()` once at program start, not inside the training loop

**Pitfall: Gloo instead of NCCL backend**
- Cost: 2 GB/s vs 100 GB/s (50× slower)
- Fix: Always `dist.init_process_group(backend="nccl", ...)`

**Pitfall: `.item()` or `torch.cuda.synchronize()` inside training loop**
- `.item()` forces GPU→CPU transfer → full GPU sync → kills overlap
- Fix: Log loss asynchronously; sync only once per N iterations

**Pitfall: Mismatched NCCL versions**
- Symptom: Hangs or silent fallback to slower implementation
- Fix: `torch.cuda.nccl.version()` to check; pin NCCL version in environment

---

## DDP Bucket Size Tuning

```python
model = DistributedDataParallel(model, bucket_cap_mb=50)  # default 25 MB
# Larger buckets → fewer NCCL calls → less overhead, higher latency for last bucket
# Smaller buckets → more overlap (start all-reduce sooner), more NCCL overhead
# 50 MB is a good starting point for large models
```

**Note:** DINOv3 supports both DDP and FSDP2. DDP bucket tuning only applies to the DDP path.
FSDP2 has its own prefetch/overlap behavior around shard materialization.

---

## FSDP2 with torch.compile

```python
# DINOv3 actual API: composable FSDP2 (per-block, reshard_after_forward=True = FULL_SHARD / ZeRO-3)
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
fsdp_config = {"mesh": device_mesh, "mp_policy": mp_policy}
for block in model.blocks:
    fully_shard(block, **fsdp_config, reshard_after_forward=True)  # ZeRO-3 per block
fully_shard(model, **fsdp_config, reshard_after_forward=True)

# Legacy FSDP1 API (NOT what DINOv3 uses, shown for reference):
# fsdp_model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, ...)
```

Graph breaks at FSDP block boundaries are **expected and correct behavior** — one block's params materialized at a time via AllGather.

---

## Straggler Detection

```python
dist.monitored_barrier(timeout=datetime.timedelta(seconds=30))
# Raises error on the rank that doesn't arrive within 30s
# Use for debugging rank imbalance; remove for production
```

---

## Measuring Communication vs Compute Split

```bash
# Reduce batch size by 50% and measure NIC throughput (GB/s):
# - If NIC throughput drops to ~60% of original → compute is bottleneck (GPUs starving network)
# - If NIC throughput stays near original → network is bottleneck

# Monitor NVLink utilization:
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv -l 1
```

---

## FSDP2 (ZeRO-3) vs DDP: communication volume difference (2026-04-01)

| Strategy | Communication per iteration | Relative overhead |
|----------|---------------------------|-------------------|
| DDP | 2× model params (1× all-reduce = reduce-scatter + all-gather) | baseline |
| ZeRO-1/2 | 2× model params | same as DDP |
| **FSDP2 / ZeRO-3** | **3× model params** (2× all-gather in fwd+bwd + 1× reduce-scatter) | **+50% vs DDP** |

For ViT-B: 86M params × 2 bytes (BF16) = 172 MB. DDP needs 344 MB per iter; FSDP2 needs 516 MB per iter.

**Critical question**: ViT-B fits entirely in one 80GB H100 (172 MB << 80 GB). FSDP is designed for models that don't fit. Using FSDP2 on ViT-B adds 50% communication overhead for modest memory savings (~1.3 GB/GPU from sharded params+grads+optimizer).

From the book: *"If a model fits on a single GPU: DDP is preferred over ZeRO"*

**Actual measured delta**: at matched bs=256, FSDP2 (23.5% MFU) vs DDP+ES (23.9% MFU) — 0.4 pp
gap. The communication overhead is not the dominant bottleneck at this scale. On NVLink, 516 MB per
step at 450 GB/s = ~1.1 ms, which is <0.5% of a 220 ms step. The overhead shows up as graph-break
launch cost and allocation churn, not raw bandwidth.

**Conclusion (2026-04-25)**: At matched bs=256, DDP and FSDP2 are essentially tied (0.4 pp gap
from ZeRO-3 overhead). The DDP vs FSDP2 question is closed — both are fine per Tim Darcet
directly. `run.sh` uses DDP+ES because it is production-validated, not because DDP is
architecturally superior. See the "Authoritative Conclusion" section at the end of this file.

---

## Decision Implication for DINOv3

At 11% MFU (8×H100, bs=64), if Nsight Systems shows:
- **NCCL and compute interleaved** → overlap is working, bottleneck is elsewhere
- **NCCL blocking compute** → FSDP prefetch config or graph break issue
- **Any rank significantly slower** → dataset I/O imbalance or masking variance across ranks

---

## `expandable_segments` in this repo (experimental results 2026-04-03)

**Result**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a targeted win for the compiled
DDP bs=256 path in this repo, but it should not be treated as a general rule for all training
strategies or all memory failures.

### What PyTorch means by `expandable_segments`

PyTorch documents `expandable_segments` as useful when allocation sizes change frequently. The
common example is changing batch size, but that is not the right mental model for this repo.

Our outer training config is mostly fixed:
- `crops.global_crops_size=224`
- `crops.local_crops_size=96`
- `crops.local_crops_number=8`

The relevant variability comes from inside each training step. The student path always feeds both
global and local crops through the backbone together in
`dinov3/train/ssl_meta_arch.py:get_student_output()`, and the backbone iterates those two
resolutions through every block in `dinov3/models/vision_transformer.py:forward_features_list()`.
That means the allocator still sees a mix of tensor sizes and lifetimes within one step even when
the user-facing batch size is constant.

### What the logs actually show

DDP bs=256 without ES showed bimodal MFU: alternating roughly `~11%` and `~18%` phases across the
run. With ES enabled, the bimodal pattern disappeared and the run stabilized at `24.5%` MFU and
`4229 img/s`.

These screening runs were short (`train.OFFICIAL_EPOCH_LENGTH=100`, `optim.epochs=1` in
`scripts/screening_ddp_expandseg.sh`), so they prove the short-horizon throughput behavior, not
the absence of any long-horizon memory creep.

This is the important repo-specific interpretation:
- ES helped because the DDP bs=256 compiled multi-crop path was fragmentation-prone.
- ES removed allocator defrag pauses and the resulting throughput collapse.
- ES did **not** create more true capacity.

The strongest evidence that this is fragmentation relief rather than a new hard-memory capability
is that bs=320 still OOMs even with ES. In this repo, ES fixes stalls and allocator pathology at
bs=256; it does not move the real memory ceiling.

The remaining uncertainty is whether a much longer run could still accumulate enough allocator
fragmentation or retained state to fail later. The current evidence argues against an immediate
capacity problem at bs=256, but it does not fully rule out a late OOM without a longer soak test.

### Working causal model

Without ES, PyTorch's CUDA allocator reserves fixed-size memory segments. In the DDP bs=256 case,
the compiled multi-crop execution pattern plus large backward/all-reduce allocations appears to
trigger periodic allocator defragmentation. Those pauses show up as the low-MFU phase.

With ES, the allocator can grow segments in place instead of repeatedly churning the segment pool.
That fits this specific DDP workload well enough to eliminate the bimodal behavior.

This explanation is still an inference from the experiment log, not a memory-snapshot proof. A
stricter confirmation pass would compare `memory_reserved`, `memory_summary()`, or
`memory_snapshot()` for DDP bs=256 with and without ES.

### FSDP2 + ES regression

FSDP2 also has a variable allocation pattern, but the results go the other direction here:
`expandable_segments` hurt FSDP2 by roughly `7-12%` across the tested batch sizes.

The practical lesson is not "ES is bad." It is:
- allocator tuning is strategy-specific
- DDP and FSDP2 interact with the allocator differently
- a memory tweak that helps one path can hurt the other

### Config guidance for experiments and real training

Use ES selectively based on strategy and objective:

```python
# DDP experiments or real DDP training at the high-batch operating point:
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Default FSDP2 training path in this repo:
# leave PYTORCH_CUDA_ALLOC_CONF unset unless new evidence says otherwise
```

In Slurm scripts:
```bash
# DDP screening or real DDP runs at bs=256:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default FSDP2 config / real training:
# do not set PYTORCH_CUDA_ALLOC_CONF just because it helped DDP
```

For real training, treat this as a current best operating hypothesis rather than a final guarantee.
The next confidence step is a longer bs=256 DDP+ES soak run that logs
`torch.cuda.max_memory_reserved()`, `torch.cuda.memory_reserved()`, and step time over thousands of
iterations, especially across checkpoint and eval boundaries.

### DDP bs=256+ES soak test confirms production safety (2026-04-07, job 16718)

500-iter soak test with 5 eval cycles and 2 checkpoint cycles. [MEMPROFILE] at every phase:

| Phase | max_reserved_mb | alloc_retries |
|-------|----------------|---------------|
| compile_warmup_iter0 | 67,156 MB | 0 |
| steady_state (iter 10) | 67,260 MB | 0 |
| pre_eval (×5) | 67,260 MB | 0 |
| eval_complete (×5) | 67,260 MB | 0 |
| checkpoint_complete (×2) | 67,260 MB | 0 |

**max_reserved_mb is perfectly flat across all 17 MEMPROFILE events. Zero memory creep.**

Steady-state MFU (iters 50–490): **23.92% average**, range 23.4–24.1%. No bimodal behavior.
Confirmed: DDP bs=256+ES is **production-safe** for long runs.

### Worst-case memory profile (2026-04-07, memprofile scripts)

| Config | Worst-case reserved | Headroom | alloc_retries |
|--------|--------------------|---------:|---------------|
| DDP bs=256 + ES | 67,260 MB | ~12.8 GB | 0 |
| FSDP2 bs=128 (no ES) | 36,340 MB | ~43 GB | 0 |

Eval phases add no reserved memory (allocator reuses already-reserved pages).
Checkpoint adds ~1.9 GB for FSDP2 (DCP write + optimizer extraction); zero additive for DDP.

### Final rankings (8×H100, ViT-B, torch.compile, real data)

| Config | MFU | img/s | Mem | Notes |
|--------|-----|-------|-----|-------|
| DDP bs=256 + ES | **23.9%** | **4180** | 67.3 GB | Production-confirmed (500-iter soak) |
| **FSDP2 bs=256** | **23.5%** | **4106** | **66 GB** | **0.4 pp behind DDP+ES at matched bs** |
| DDP bs=128 + ES | ~23.1% | ~4050 | 33.9 GB | Conservative fallback |
| DDP bs=128 | 23.1% | 4042 | 34 GB | Without ES |
| DDP bs=64 + ES | 17.5% | 3061 | 18 GB | No benefit from ES at bs=64 |
| FSDP2 bs=256 + ES | 21.8% | 3809 | 66 GB | ES actively hurts FSDP2 |

> **Key revision (2026-04-25)**: FSDP2 bs=256 is only 0.4 pp behind DDP+ES at the same batch size.
> The "2× MFU" narrative compared FSDP2 bs=64 (original) vs DDP+ES bs=256 — the difference is
> almost entirely from batch-size scaling. DDP and FSDP2 are both fine at this scale (confirmed
> by Tim Darcet). `run.sh` uses DDP+ES as the production-validated config. The DDP vs FSDP2
> question is closed — see the "Authoritative Conclusion" section at the end of this file.

Note: earlier short-run screening reported 24.5% for DDP bs=256+ES; the soak test average of
23.92% is the more reliable number — it averages across thermal ramp and minor step-time variance.

---

## FSDP2 Allocator Fragmentation Study (2026-04-07/08)

### Hypothesis

Our 70-iter worst-case profiling correctly measures the single-step peak (FSDP2 bs=128 = 36.3 GB).
A colleague hit OOM at FSDP2 bs=128 without ES in a real training run. The hypothesis: FSDP2's
~24 alloc/free cycles per step (12 blocks × forward+backward, each all_gather ~112 MB) fragment
the caching allocator over thousands of steps, growing `reserved_mb` beyond 80 GB despite a
single-step peak of only 36 GB.

### Experiment (jobs 16750/16751): 500-iter fragmentation tracking

```bash
sbatch scripts/memfrag_fsdp2.sh 128 false 500 10   # FSDP2 no ES — expect fragmentation
sbatch scripts/memfrag_fsdp2.sh 128 true  500 10   # FSDP2 + ES  — control
```

Results — `[MEMFRAG]` logged every 10 iters, rank=0:

| Metric | FSDP2 bs=128 no ES | FSDP2 bs=128 + ES |
|--------|-------------------|-------------------|
| `alloc_retries` (iter 9→499) | **0 throughout** | 0 throughout |
| `inactive_split_mb` range | **290–294 MB (flat)** | 0 MB (flat) |
| `fragmentation_ratio` | **0.008 flat** | 0.000 flat |
| `current_reserved_mb` | 34,464 MB | 34,050 MB |

### Result: hypothesis is a **null result**

FSDP2's per-block allocations are highly uniform in size. The caching allocator reuses
identically-sized freed blocks efficiently — zero `alloc_retries` at any point across
500 iterations. The 290 MB `inactive_split_mb` in no-ES is a static compile-time artifact
(blocks split during warmup), not a growing problem.

The colleague's OOM is NOT explained by allocator fragmentation within 500 iters.
Remaining hypotheses: (a) Gram loss enabled — adds a third teacher forward, ~172 MB extra,
(b) pretrained weight load temporarily double-buffers model at init (our runs use `pretrained_weights=""`),
(c) shared GPU resources, (d) fragmentation at >3750 iters — 500-iter trend is flat but
we cannot exclude accumulation starting later.

### What ES actually contributes

- `inactive_split_mb = 0` (vs 290 MB) — cleaner allocator state, but the 290 MB is harmless
- `current_reserved_mb` 414 MB lower — slightly tighter packing
- Zero `alloc_retries` in both cases — the benefit is insurance, not a fix for an observed problem

For FSDP2 single-node single-run, **ES provides allocator cleanliness at 7–12% throughput cost,
but is not currently needed to prevent OOM** at bs=128, at least through 500 iters.
Use DDP+ES instead — it gains throughput AND keeps the allocator clean.

### Diagnostic tooling (for future investigations)

```python
# Key memory_stats() fields:
stats = torch.cuda.memory_stats()
stats["num_alloc_retries"]                  # cumulative retries — growing = bad
stats["inactive_split_bytes.all.current"]   # bytes in unusable fragments
stats["reserved_bytes.all.current"]         # total reserved (active + fragmented + free)
stats["allocated_bytes.all.current"]        # currently in use
```

To run: `DINOV3_MEMORY_PROFILE=1 DINOV3_MEMORY_PROFILE_PERIOD=10 sbatch ...`
Parse: `grep '\[MEMFRAG\]' <log> | grep 'rank=0'`

### Short version

FSDP2 without ES does NOT fragment meaningfully at ViT-B bs=128 over 500 iterations.
The OOM that prompted this investigation is explained by something else (different config,
shared GPU, or very long run fragmentation beyond 3750 iters). Job 29703 (5000-iter long soak)
is running to test beyond the 500-iter window.

---

## Authoritative Conclusion: DDP vs FSDP2 for ViT-B on single node (2026-04-25)

Tim Darcet (DINOv2/v3 co-author) on this exact question, April 2026:

> *"I have not used DDP once since 2022 haha*  
> *In fsdp you have the option to not release shards, in which case it's basically equivalent*  
> *to DDP, with the gather moved before the fwd instead of after the bwd*  
> *I think both are fine if both fit"*

### Key insight

> **FSDP2 with `reshard_after_forward=False` is DDP-like in communication volume, but with
> FSDP2's sharded gradient/optimizer-state machinery.**

FSDP2 is not a single mode. It has two important operating points:

| FSDP2 mode | `reshard_after_forward` | Communication | Memory behavior | Best use |
|-----------|------------------------|-------------|--------|---------------|
| No-release / DDP-like | `False` | DDP-like volume; all-gather before forward, reduce-scatter after backward | unsharded params stay resident after forward; gradients/optimizer state remain sharded | default FSDP2 path when the model fits |
| ZeRO-3 / full-shard | `True` | extra per-layer all-gather in backward | lower parameter residency, more communication | use when memory/headroom matters |

Our 0.4 pp MFU gap (FSDP2 23.5% vs DDP+ES 23.9%) comes specifically from `reshard_after_forward=True`
(per-block all-gather overhead). Switching to `reshard_after_forward=False` should close most or all
of that gap while staying in the FSDP2 API. Darcet's workflow points in this direction.

This changes the prior:
- DDP remains a valid simple baseline.
- FSDP2 no-release is likely the better long-run default if it matches DDP+ES throughput, because it
  keeps sharded optimizer/gradient state and stays inside the FSDP2 runtime/compiler ecosystem.
- FSDP2 ZeRO-3 remains the mode to test when extra batch size, sequence length, or model scale needs
  lower parameter residency.

### Practical guidance for this codebase

1. **`run.sh` DDP+ES is production-validated and correct today.**
2. **Add a FSDP2 no-release config before changing production.** The current repo hard-codes
   `reshard_after_forward=True`, so `train.distributed_strategy=fsdp2` today still means ZeRO-3.
3. **If FSDP2 no-release ties DDP+ES, prefer FSDP2 for future work.** That makes later PyTorch
   compiler/runtime experiments easier without giving up the current throughput.
4. **Do not restart DDP-vs-ZeRO-3 bake-offs.** The old debate is closed; the useful next comparison
   is DDP+ES vs FSDP2 no-release.
5. **Next non-distributed priority remains local crop count (8->4)** because it has larger potential
   upside than sub-1 pp runtime choices.

### Automatic overlap and bucketing implications

The PyTorch Inductor overlap/bucketing work is relevant to FSDP2, but the FSDP2 mode matters:

- `reshard_after_forward=False`: fewer all-gather opportunities in backward because unsharded params
  are kept after forward. This is the best candidate for DDP-like production throughput.
- `reshard_after_forward=True`: more per-layer collectives to schedule and bucket. This is the more
  interesting candidate for automatic overlap/bucketing research.

For a newer-PyTorch experiment, test both:

1. FSDP2 no-release (`reshard_after_forward=False`) as the production-style FSDP2 baseline.
2. FSDP2 ZeRO-3 (`reshard_after_forward=True`) with Inductor distributed overlap/bucketing enabled.

The win condition for the overlap/bucketing branch should be a clear gain over FSDP2 no-release and
DDP+ES after warmup, not just parity with the current ZeRO-3 baseline.
