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

## Measured num_workers benchmarks (from ml-engineering book, A100 setup) (2026-04-01)

```
num_workers=0:  ~5.4s per iteration  ← blocking, terrible
num_workers=1:  ~3.9s per iteration
num_workers=2:  ~1.3s per iteration  ← big jump
num_workers=3:  ~0.8s per iteration  ← best in benchmark
num_workers=4:  ~0.9s per iteration  ← diminishing / overhead
```

Diminishing returns after 2–3 workers for typical workloads. DINOv3 uses `num_workers=20` (for 8 GPUs = 20 workers total for the node's DataLoader). That's 2.5 workers/GPU — within the range, may be over-provisioned.

**pin_memory + non_blocking measured gains:**
```
pin_memory=True,  non_blocking=True:  0.459s  ← best
pin_memory=True,  non_blocking=False: 0.522s
pin_memory=False, non_blocking=True:  0.658s  
pin_memory=False, non_blocking=False: 0.646s
```
→ `pin_memory=True + non_blocking=True` = **14% faster H2D** vs default. DINOv3 already uses `pin_memory=True`. The `non_blocking=True` is applied in `collate_data_and_cast()`.

**Why non-trivial**: The effect is measured, not assumed. Also: `pin_memory` uses special OS-pinned memory — if system RAM is tight, this can cause CPU-side OOM. Monitor with `free -h`.

**Decision implication**: DINOv3's DataLoader config is already well-tuned. Don't add workers speculatively — the gain plateaus fast and excess workers add memory overhead.

---

## Decision Implication for DINOv3

DINOv3 already has `num_workers=20`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=8` in `run.sh`. This was a reasonable starting baseline.

**Phase 6.B.1 update (2026-05-26)**: the 20/8 config was empirically over-provisioned for Weka+HDF5 data. Phase 6.B.1 (job 58188) measured `data_time ≈ 0.002–0.004s` vs `step_time ≈ 0.44–0.50s` — the loader was idle 99% of the time. More importantly, 20 workers × pf=8 × 8 ranks created ~307 GB host-RAM pressure that directly caused the bs≥192 OOM in Phase 6.A.6 (Slurm cgroup exhausted). The config has since been reduced to `num_workers=12, prefetch_factor=4`.

**If profiling shows data pipeline is NOT the bottleneck** (likely on Weka): focus on kernel-level optimizations. Use the `data_time / step_time` ratio below to size the loader correctly.

**If profiling shows GPU idle at iter start**: increase `num_workers` first, then `prefetch_factor`. Check Weka throughput with `iostat` from the compute node.

---

## Balanced DataLoader Sizing: The Throughput-Critical Constraint (2026-05-26)

*Derived from Phase 6.B.1 (job 58188, DDP+cudagraphs bs=128 soak on real Weka data).*

### The constraint formula

The loader must produce batches at least as fast as the GPU consumes them:

```
worker_rate ≥ GPU consumption rate
num_workers / per_batch_latency_s ≥ 1 / step_time_s
→ num_workers_needed ≥ per_batch_latency_s / step_time_s
```

At `step_time ≈ 0.5s` (DDP+cudagraphs bs=128): if a single worker produces one batch in ~0.5–1.0s (typical for 10-crop PIL augmentation on Weka), the theoretical minimum is **1–2 workers**; everything above absorbs timing variance only.

**Per-batch latency** is not directly logged — proxy it from a loader-only timing loop (`for batch in loader: pass`) or infer from `data_time` at `num_workers=1`. It includes: file read (Weka seek + HDF5 decode), all augmentations (Albumentations multi-crop), mask generation in `collate_data_and_cast()`, and crop stacking.

### Primary diagnostic: `data_time / step_time`

`data_time` is logged per-iteration by `MetricLogger` (`dinov3/logging/helpers.py:65-133`). Check it after the compile warmup clears (iter > ~30).

| data_time / step_time | Interpretation                 | Action                                         |
|-----------------------|-------------------------------|------------------------------------------------|
| < 5%                  | Loader massively over-provisioned | Can reduce num_workers and/or prefetch_factor |
| 5–15%                 | Well-matched, healthy          | Hold current settings                          |
| 15–40%                | Approaching saturation         | Don't reduce; monitor jitter                   |
| > 40%                 | Loader is the bottleneck       | Increase num_workers first, then prefetch_factor |

Phase 6.B.1 measured < 1% → the 20/8 config had 10–50× more loader capacity than needed.

### Memory math: batches in flight across the node

With DDP, each rank runs its own DataLoader. Total batches held in pinned host memory:

```
batches_in_flight_per_rank = num_workers × prefetch_factor
node_total_batches = batches_in_flight_per_rank × num_ranks
```

For DINOv3 DDP bs=128 (10 crops, ~240 MB per batch in pinned host RAM):

| Config                     | Batches/rank | Node batches (8 ranks) | Approx host RAM |
|----------------------------|--------------|------------------------|-----------------|
| 20 × 8 (original run.sh)   | 160          | 1280                   | ~307 GB         |
| 12 × 4 (Phase 6.B.1)       | 48           | 384                    | ~92 GB          |
| 8 × 2 (aggressive)         | 16           | 128                    | ~30 GB          |
| 4 × 2 (conservative)       | 8            | 64                     | ~15 GB          |
| 2 × 2 (theoretical floor)  | 4            | 32                     | ~8 GB           |

**The 20×8 config was the root cause of bs≥192 host-RAM OOM** (Phase 6.A.6). The Slurm cgroup memory limit was hit by DataLoader prefetch alone, not VRAM.

`persistent_workers=True` adds a fixed baseline of ~500 MB–1 GB per worker (Python interpreter + HDF5 mmap + decoded buffers) on top of the prefetch queue. Going from 20→12 workers saves ~4–15 GB from this baseline alone.

### Prefetch factor minimum: pf ≥ 2

`non_blocking=True` in `.to(device)` only creates actual async H2D overlap when the next batch is already in **pinned host memory** when the GPU reaches it. With `pf=1`, no batch is ready at that moment → the H2D copy becomes synchronous and blocks the next step.

**`prefetch_factor=2` is the hard minimum** for `non_blocking=True` to be effective. `pf=4` provides a safety margin against augmentation variance spikes. Never go below 2.

### What to use

| Scenario                                | num_workers | prefetch_factor | Notes                              |
|-----------------------------------------|--------------|-----------------|------------------------------------|
| DDP bs=128, normal training             | 8            | 2               | ~30 GB host RAM; validated safe    |
| DDP bs=128, tight Slurm cgroup          | 4            | 2               | ~15 GB; wide margin                |
| FSDP2 bs=96 (run.sh)                   | 12           | 4               | Reduced from 20/8; Phase 6.B.1     |
| Any config, profiling soak              | 4            | 2               | Minimize loader noise in profiles  |

**Verification workflow**: after submitting a job, check the `data:` column in training logs at iters 30–100 (past compile warmup). Compute `data_time / step_time`. If < 5%, you can reduce further. If > 15%, increase workers.

### CORRECTION (2026-05-27): cgroup memory ≠ the host-RAM table above

The "Approx host RAM" estimates above are **batch-buffer arithmetic only** and do NOT match
what the Slurm cgroup actually reports. Measured fact from three bs=128 + AC=full soaks
(jobs 58188 / 59273 / 59274): cgroup `memory.current` peaks at **exactly the `--mem` limit
every time** (300/300, 430/430, 540/540 GB). Reason: `memory.current` counts **reclaimable
file-backed page cache** (HDF5/tile reads off Weka), which expands to fill whatever limit you
give it and is reclaimed under pressure instead of OOM-killing. So:

- **Do not read cgroup `memory.current` as "the job needs this much RAM."** It's an upper
  bound set by your own `--mem`, not a measurement. To find the true (anonymous) floor, log
  `memory.stat` (anon vs file split) or step `--mem` down until OOM.
- **`sacct MaxRSS` is useless here** — it sums per-task RSS and double-counts COW fork pages
  across 100+ workers (reported 883 GB / 988 GB against 430/540 GB limits).
- OOM happens on **anonymous** memory, and notably **during torch.compile warmup before iter
  0** (job 57799: 20w/8pf OOM'd at `--mem=380G`). 16w configs did not OOM at 430–540G.

### CORRECTION: prefetch depth, not worker count, hid the loader (16w sweep)

Earlier note claimed "worker count is the binding throughput knob." The 16w/4pf vs 16w/8pf
runs (same workers) show the opposite at this scale: only **pf=8** drove `data_time` to ~0
(p50 ≈ 2 ms); pf=4 at 16 workers stayed loader-bound. At 16 workers, a 4-deep queue drains
between steps; an 8-deep queue absorbs the tail. **When the GPU step is long (AC=full ≈
430 ms GPU event), deepen prefetch before adding workers.**

Also: with the loader fully hidden (data_time≈0), wall iter (~900 ms) was still ~2× the
GPU-event step (~430 ms). That residual is CPU-side per-iter overhead (collate/cast, H2D
launch, EMA, GC, cudagraph replay) — the next lever after the loader is non-binding.
MetricLogger `images_per_sec` (= 1024 / GPU-step-ms) hides it and overstates delivered
throughput ~2×; always cross-check against the wall `time:` column.
