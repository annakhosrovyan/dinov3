# Distributed Training Learnings — AI Systems Perf Eng Ch04

Notes distilled from `~/knowledge-base/ai_systems_perf_engineering/ai_systems_perf_eng_ch04.md`.
Tailored to DINOv3 8×H100 / FSDP2 setup.

---

## Core Principle: Overlap Communication with Computation

The goal is `time_iter ≈ max(time_compute, time_comm)`, not `time_compute + time_comm`.

| Approach | Backward | Comm | Total | Overlap |
|----------|----------|------|-------|---------|
| Manual all-reduce (no overlap) | 10 ms | 12 ms | 22 ms | 0% |
| DDP (overlap) | 10 ms | 12 ms | ~12 ms | ~50%+ |

**DINOv3 uses FSDP2 (`SHARD_GRAD_OP`)** — should overlap AllGather with forward compute between blocks. Verify with Nsight Systems timeline (compute kernels and NCCL ops should interleave, not run sequentially).

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

**Note:** DINOv3 uses FSDP2, not DDP. FSDP2 has its own prefetch/overlap tuning via `BackwardPrefetch.BACKWARD_PRE`.

---

## FSDP2 with torch.compile

```python
# Compile first, then wrap FSDP at transformer block granularity:
model = torch.compile(model, mode="max-autotune")
fsdp_model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy(...),
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,  # what DINOv3 uses
    use_orig_params=True,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,   # DINOv3 config: fp32 reduction
        buffer_dtype=torch.bfloat16,
    ),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # overlaps AllGather with backward
)
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

**Critical question**: ViT-B fits entirely in one 80GB H100 (172 MB << 80 GB). FSDP is designed for models that don't fit. Using FSDP2 on ViT-B adds 50% communication overhead for zero memory benefit.

From the book: *"If a model fits on a single GPU: DDP is preferred over ZeRO"*

On a single NVLink node, the 172 MB extra communication is fast (~microseconds on NVLink) so this may not be the dominant bottleneck — but it's unnecessary overhead. **Test DDP vs FSDP2 experimentally** to measure the actual delta.

**Why non-trivial**: FSDP2 is the "modern default" in many PyTorch training setups. It's the right tool for large models (>10B params) but actively harmful for small models that fit in GPU memory.

**Decision implication**: Benchmark DDP vs FSDP2 on single node. If step time drops significantly, switch run.sh to DDP. ViT-B (86M) has no reason to use FSDP2 on 80GB H100s.

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

### Final rankings (8×H100, ViT-B, torch.compile, real data)

| Config | MFU | img/s | Mem | Notes |
|--------|-----|-------|-----|-------|
| DDP bs=256 + ES | **24.5%** | **4229** | 66 GB | New best — stable, no bimodal |
| DDP bs=128 + ES | 22.9% | 3993 | 34 GB | Best memory-efficient option |
| FSDP2 bs=256 | 23.5% | 4106 | 66 GB | Old best — beaten by DDP+ES |
| DDP bs=128 | 23.1% | 4042 | 34 GB | Without ES — still good |
| DDP bs=64 + ES | 17.5% | 3061 | 18 GB | No benefit from ES at bs=64 |
| FSDP2 bs=256 + ES | 21.8% | 3809 | 66 GB | ES actively hurts FSDP2 |
