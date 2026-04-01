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

## Decision Implication for DINOv3

At 11% MFU (8×H100, bs=64), if Nsight Systems shows:
- **NCCL and compute interleaved** → overlap is working, bottleneck is elsewhere
- **NCCL blocking compute** → FSDP prefetch config or graph break issue
- **Any rank significantly slower** → dataset I/O imbalance or masking variance across ranks
