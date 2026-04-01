# Kernel Optimization Learnings — AI Systems Perf Eng Ch08 & Ch09

Notes distilled from:
- `~/knowledge-base/ai_systems_perf_engineering/ai_systems_perf_eng_ch08.md` (occupancy, warp efficiency, ILP)
- `~/knowledge-base/ai_systems_perf_engineering/ai_systems_perf_eng_ch09.md` (arithmetic intensity, kernel fusion, mixed precision, Tensor Cores)

---

## Four Performance Regimes (Diagnostic Framework — Ch08)

| Regime | Symptom | Fix |
|--------|---------|-----|
| **Underutilizing** | Low occupancy, low FLOPS, low BW, idle gaps | Increase parallelism, fix launch config |
| **Latency bound** | Low BW well below peak, high `Stall: Exec Dependency` | ILP, loop unrolling, prefetch |
| **Memory bound** | BW near peak, FLOPS low (ALUs idle waiting for data) | Increase arithmetic intensity, kernel fusion, tiling |
| **Compute bound** | FLOPS near peak, BW low | Overlap communication, data pipeline |

**For DINOv3 at 11% MFU**: likely **memory bound** (FFN GEMMs dominate, attention memory-intensive, BF16 already enabled). Profiling will confirm.

---

## Key Nsight Compute Warp Stall Reasons

| Stall Reason | Meaning | Fix |
|---|---|---|
| `Long Scoreboard` | Waiting on DRAM loads | Memory-bound; increase arithmetic intensity |
| `Short Scoreboard` | Shared memory latency | Reduce shared mem usage or use padding |
| `Exec Dependency` | Latency-bound; instructions depend on previous results | Increase ILP |
| `Math Pipe Throttle` / `Compute Unit Busy` | Compute-bound | Focus on communication/data overlap |
| `Not Selected` / `Idle` | Insufficient parallelism / low occupancy | Increase block count, fix launch config |

---

## Occupancy — Key Facts for H100

- **Max**: 64 warps per SM = 2048 threads per SM
- Occupancy is a **means to hide latency**, not a goal itself — diminishing returns after 50–75%
- **Limiters**: registers per thread, shared memory per block, or thread count

**Auto-tune block size at runtime:**
```cuda
int minGridSize, bestBlockSize;
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize, &bestBlockSize, myKernel, 0, 0);
myKernel<<<minGridSize, bestBlockSize>>>(...);
```

**Force compiler to cap registers per thread** (frees warps):
```cuda
__global__ __launch_bounds__(256, 8)  // max 256 threads/block, min 8 blocks/SM
void myKernel(...) { ... }
// 8 blocks × 256 threads = 2048 threads/SM = 100% occupancy IF register budget fits
```
Trade-off: reduces per-thread register budget → less unrolling/ILP → slightly lower throughput per warp.

**PyTorch angle**: `torch.compile(mode="max-autotune")` selects optimal tile sizes → better occupancy automatically.

---

## Instruction-Level Parallelism (ILP) — Critical for Latency Hiding

**Key insight from book (Table 8-4):**
| ILP | Threads/SM needed for 100% utilization | Occupancy % needed |
|-----|----------------------------------------|-------------------|
| 1 (no ILP) | 1,536 (48 warps) | 75% |
| 2 | 1,024 (32 warps) | 50% |
| 4 | 512 (16 warps) | 25% |

→ With 4-way ILP, you only need 25% occupancy to saturate the GPU. ILP is more powerful than chasing occupancy.

**How to get ILP: loop unrolling**
```cuda
// BEFORE: Sequential dependency
for (int i = 0; i < N; ++i) {
    sum += a[i] * b[i];  // must wait for load each iteration
}

// AFTER: 2-way ILP (expose independent operations)
for (int i = 0; i + 1 < N; i += 2) {
    float a0 = a[i], b0 = b[i];       // load pair 0
    float a1 = a[i+1], b1 = b[i+1];  // load pair 1 (independent, in-flight)
    sum += a0 * b0 + a1 * b1;         // both mults can proceed in parallel
}
```

**Warning**: Too much unrolling → register spillage → worse. Sweet spot: 2–4 way ILP.

**In PyTorch**: `torch.compile` does this automatically for elementwise ops. For custom CUDA kernels, use `#pragma unroll 4`.

---

## Arithmetic Intensity — The Core Metric (Ch09)

**Formula**: Arithmetic Intensity = FLOPS / Bytes moved from DRAM

| Intensity | Regime | H100 Boundary |
|-----------|--------|---------------|
| < 0.43 FLOPS/byte | Memory-bound | Below roofline |
| > 0.43 FLOPS/byte | Compute-bound (approaching ALU peak) | Above roofline |

H100 HBM bandwidth: ~3.35 TB/s. Peak FP32: ~1.45 PFLOPS. Crossover: 1450/3350 ≈ **0.43 FLOPS/byte**.

**Why DINOv3 FFN matmuls are memory-bound without batching**: At bs=1 per layer, intensity is low. Larger batches amortize memory reads over more FLOPs → intensity increases.

---

## Kernel Fusion — Biggest Win for Memory-Bound Code

**Without fusion** (two kernels):
```cuda
y = sin(x);   // Read x, write y to HBM (8 bytes)
z = sqrt(y);  // Read y, write z to HBM (8 bytes)
// Intensity: 2 FLOPS / 12 bytes = 0.167
```

**With fusion** (one kernel):
```cuda
z[i] = sqrt(sin(x[i]));
// Intensity: 2 FLOPS / 8 bytes = 0.25 (+50%)
// Plus: eliminates one kernel launch overhead
```

**Expected impact**: 2–8× speedup on memory-bound kernels.

**PyTorch gets this automatically** via `torch.compile` for adjacent elementwise ops. Manual fusion needed only for domain-specific ops (custom attention variants, satellite-specific preprocessing).

---

## Mixed Precision — Tensor Core Utilization

| Format | Tensor Core Speed | Notes |
|--------|------------------|-------|
| FP32 | Baseline | No Tensor Core benefit |
| TF32 | ~3× FP32 | Same exponent range, lower mantissa. Enable with `torch.set_float32_matmul_precision('high')` |
| **BF16** | **~4× FP32** | **DINOv3 already uses this.** Preferred over FP16 (no loss scaling needed) |
| FP16 | ~4× FP32 | Needs loss scaling for training stability |
| FP8 | ~8× FP32 | Requires Transformer Engine; calibration needed |

**DINOv3 is already on BF16 params + FP32 reduce** (`compute_precision` config). This is correct.

**Ensure TF32 is also enabled** for FP32 fallback operations:
```python
torch.set_float32_matmul_precision('high')  # already in train.py:44
torch.backends.cudnn.allow_tf32 = True      # already in train.py:44
```

---

## FlashAttention / SDPA — Already Using (confirmed)

`torch.nn.functional.scaled_dot_product_attention(q, k, v)` in PyTorch ≥ 2.0 auto-dispatches to FlashAttention 2 for BF16/FP16 on CUDA with no explicit mask.

**Why this matters for arithmetic intensity**: FA2 fuses QK^T + softmax + output projection into one kernel, eliminating intermediate HBM writes → much higher arithmetic intensity than naive attention.

**Decision**: Do NOT add `flash-attn` package — already running FA2. FA3 only if sequences >> 1K tokens.

---

## Batch Size Effect on MFU

At bs=64/GPU (current): `max_memory_allocated=16.4GB` out of 80GB H100 → **~20% VRAM utilization**.

Increasing batch size directly improves arithmetic intensity of FFN GEMMs:
- More computations per memory read → higher intensity → closer to compute-bound
- Expected: bs=128 → ~+15% MFU; bs=256 → ~+25% MFU (diminishing returns after ~70% VRAM utilization)

**Priority**: Test bs=128 → bs=256 before any kernel rewriting.

---

## Activation Checkpointing Trade-off

```python
# Recompute activations during backward instead of storing them
output = torch.utils.checkpoint.checkpoint(transformer_block, input, use_reentrant=False)
# Memory saved: O(batch × seq_len × hidden) per block
# Compute cost: ~25-30% extra in backward pass
```

**When to use**: To fit larger batch sizes when VRAM is the constraint. Current DINOv3 has ~64GB headroom → not needed yet. Enable if bs=256+ causes OOM.

---

## CUTLASS vs Hand-Tuned Kernels

For any custom CUDA kernel involving GEMMs: use CUTLASS templates, not hand-tuned code.

| Metric | Hand-tuned | CUTLASS |
|--------|-----------|---------|
| Tensor Core utilization | 98% | 98% |
| Dev effort | Very high | Low |

CUTLASS automatically handles: tile sizing for TMEM (256 KB/SM on Blackwell), TMA prefetch, double buffering, FP8/BF16/TF32 dispatch.

---

## Structured 2:4 Sparsity (Future Option)

- Prune 50% of weights in 2:4 pattern → Sparse Tensor Cores skip zeros → ~2× throughput on GEMMs
- **Training only**: Sparse forward pass; gradients do not benefit
- **Practical**: Inference or fine-tuning after dense pretraining
- Not relevant for current DINOv3 pretraining, but relevant for downstream inference deployment

---

## Optimization Priority for DINOv3 (11% → 30% MFU)

Given current setup (BF16 + torch.compile already enabled):

1. **Increase batch size**: bs 64 → 128 → 256 (20% VRAM → 40% → 80%) — biggest bang for effort
2. **Profile first**: nsys + ncu to confirm memory-bound regime before optimizing
3. **Check torch.compile graph breaks**: TORCH_LOGS="+dynamo" — any unexpected breaks = free wins
4. **Verify FA2 is active**: Should be automatic with SDPA + BF16 — confirm in ncu (fused attention kernel)
5. **Communication overlap**: Nsight Systems timeline — FSDP AllGather should interleave with compute
6. **Batched multi-crop** (experimental): Process all crops jointly instead of sequential global+local — reduces kernel launch overhead, better GPU utilization
