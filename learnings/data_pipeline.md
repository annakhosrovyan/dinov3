# Data Pipeline Learnings — AI Systems Perf Eng Ch05

Notes distilled from `~/knowledge-base/ai_systems_perf_engineering/ai_systems_perf_eng_ch05.md`.
Tailored to DINOv3 satellite data pipeline (HDF5, Weka FS, multi-worker DataLoader).

---

## Storage Bandwidth Requirements

- **Per-GPU minimum**: ~200 MB/s to keep GPU fed
- **8× H100**: 8 × 200 MB/s = **1.6 GB/s aggregate** needed
- DINOv3 uses Weka FS (`/mnt/weka/akhosrovyan/re-id/pretraining/`) — high-throughput parallel FS, should be fine

If step-time increases when switching from synthetic data to real data → data pipeline is a bottleneck.

---

## Diagnosing I/O Bottleneck

```python
# Method 1: Time the loader alone (no GPU compute)
for batch in loader:
    pass  # Just measure fetch time
# If this << iteration time → data pipeline is NOT the bottleneck

# Method 2: Run with num_workers=0 to isolate pure Python overhead
# Then increase workers to see how much of gap is parallelism

# Method 3: Nsight Systems — GPU idle gaps at iteration start = data stall
```

**Bottleneck symptoms:**
| Symptom | Root Cause |
|---------|-----------|
| GPU idle during iteration start (nsys timeline) | Data loading too slow |
| CPU at 100% during training | DataLoader workers saturated |
| `next(iter)` time >> step compute time | Prefetch not keeping up |

---

## DataLoader Config (what DINOv3 run.sh uses vs what's possible)

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=20,          # run.sh override (default lower in config)
    pin_memory=True,         # always — DMA transfer instead of pageable memcpy
    persistent_workers=True, # run.sh — avoids worker respawn between epochs
    prefetch_factor=8,       # run.sh override (default 4)
)
```

**Tuning guidance:**
- `num_workers`: Start at `2 × num_gpus` (16 for 8×H100), go up to 20–32 if CPU has headroom
- `pin_memory=True`: Almost always beneficial — enables GPU DMA for H2D copies
- `prefetch_factor=8`: 8 batches pre-loaded per worker — good for bursty I/O (HDF5 random access)
- `persistent_workers=True`: Critical if dataset iteration time > worker respawn time

---

## H2D Overlap via Async Streams

```python
copy_stream = torch.cuda.Stream()
compute_stream = torch.cuda.current_stream()
transfer_done = torch.cuda.Event()

# Preload first batch
with torch.cuda.stream(copy_stream):
    next_inputs = first_batch.to(device, non_blocking=True)
    transfer_done.record(stream=copy_stream)

for batch in loader:
    # Wait for H2D
    compute_stream.wait_event(transfer_done)
    inputs = next_inputs

    # Kick off next H2D in background
    with torch.cuda.stream(copy_stream):
        next_inputs = next_batch.to(device, non_blocking=True)
        transfer_done.record(stream=copy_stream)

    # Compute — overlaps with next H2D
    outputs = model(inputs)
```

**Verify with Nsight Systems**: H2D copy and compute kernels should overlap in timeline.

---

## File Format Matters

- **Good**: HDF5 (like `/mnt/weka/adovlatyan/synthetic_intelinair.h5`), Arrow, Parquet, WebDataset (tar shards) — large contiguous reads
- **Bad**: Millions of individual files — random reads, high metadata overhead

DINOv3 uses HDF5 for at least some datasets (`HDF5Dataset`). Sequential reads within HDF5 are fast; random access can be slower — use large `prefetch_factor` to amortize latency.

---

## GPUDirect Storage (GDS) — For Extreme Cases

- GPU reads directly from NVMe/Weka without CPU bounce buffer
- **Throughput improvement**: ~20% (8.0 → 9.6 GB/s measured in book benchmark)
- **CPU savings**: Frees CPU for preprocessing
- **Requires**: CUDA toolkit + filesystem support (WekaFS is GDS-compatible)
- **API**: `cuFile` library — not worth implementing unless data loading confirmed bottleneck

**Current setup**: Weka FS supports GDS. Not needed until profiling shows I/O bound.

---

## DALI (GPU Data Loading) — For Image-Heavy Workloads

- Offloads image decode + augmentation to GPU
- Especially valuable for JPEG decoding (frees 6–8 CPU cores)
- DINOv3 does its own augmentation in Python/CPU — DALI could speed this up if CPU-bound
- Only worth implementing if profiling shows CPU augmentation is the bottleneck

---

## Continuous Profiling Approach (from Ch05)

1. Baseline: measure throughput (samples/sec) on single GPU
2. Scale 1→8→multinode; check scaling efficiency at each step
3. Profile for bottleneck at each scale with nsys
4. One change at a time; remeasure
5. **Target**: GPU utilization > 80%, GPU idle time < 10%

---

## Decision Implication for DINOv3

DINOv3 already has `num_workers=20`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=8` in `run.sh`. This is a well-configured baseline.

**If profiling shows data pipeline is NOT the bottleneck** (likely): focus on kernel-level optimizations (batch size, mixed precision, torch.compile).

**If profiling shows GPU idle at iter start** (possible with real Weka data vs synthetic): try increasing `num_workers` to 32, or check Weka throughput with `iostat` from the compute node.
