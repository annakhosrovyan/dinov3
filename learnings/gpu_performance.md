# GPU Performance Learnings — DINOv3

Non-trivial insights discovered during optimization work. Format: insight + why non-trivial + decision implication.

---

## FFN dominates attention ~15:1 at ViT-B sequence lengths (2026-03-31)

At N=197 tokens (ViT-B global crop, 196 patches + 1 CLS):
- FFN cost per layer: N × 4D² = 197 × 4 × 768² ≈ **464M ops**
- Attention cost per layer: N² × D = 197² × 768 ≈ **30M ops**
- Ratio: ~15:1 FFN-to-attention

The crossover where attention becomes dominant is N ≈ 4D. For ViT-B (D=768): **N ≈ 3072 tokens**.

**Why non-trivial**: Common assumption is "attention is the bottleneck." That's only true at long sequences (e.g., 1K+ tokens for video, long documents). At the short sequences typical of image ViTs, FFN dominates heavily.

**Decision implication**: FlashAttention 3 is low priority for this workload. Optimize FFN kernels (batched multi-crop, larger batch size for better matmul efficiency) first. FA3 becomes relevant if we ever process sequences ≥ 1K tokens.

---

## F.scaled_dot_product_attention auto-dispatches to FlashAttention 2 in PyTorch 2.6 (2026-03-31)

`attention.py:116` uses `torch.nn.functional.scaled_dot_product_attention(q, k, v)`. In PyTorch ≥ 2.0, this automatically uses FA2 kernels on CUDA devices when inputs are BF16/FP16 and there is no explicit attention mask.

**Why non-trivial**: FA2 integration is not prominently documented. Easy to assume standard SDPA is a naive O(N²) kernel and add a separate FA dependency unnecessarily.

**Decision implication**: Do not add `flash-attn` package just to "enable FlashAttention" — it's already running. FA3 (which IS different) requires `flash-attn>=2.7` + explicit API call. For N=197, not worth it yet.

---

## torch.cuda.memory_allocated() ≠ VRAM utilization (2026-03-31)

The `mem: 386 (max mem: 16786)` in training logs (`helpers.py:113–114`) reports `torch.cuda.memory_allocated()` — bytes held by **live PyTorch tensors only**. The CUDA caching allocator also reserves larger blocks not counted here (`torch.cuda.memory_reserved()` is higher).

At bs=64/GPU, max_allocated=16.4GB out of 80GB H100 VRAM → ~20% utilization. There is ~64GB headroom.

**Why non-trivial**: Easy to misread 16GB as "we're using 16GB of VRAM" and think the GPU is close to full. It's the opposite — we're heavily under-utilizing.

**Decision implication**: Batch size can likely be increased 4× (to bs=256) without OOM. Test bs=128 first. Larger batches = better matmul efficiency = higher MFU.

---

## GPU thermal drift: ~40% step-time increase over 5min at sustained 8×H100 full load (2026-03-31)

Job 6770 (gpu03, 8×H100, ViT-B, bs=64): step_time_ms increased from ~220ms (iters 100–140) to ~320ms (iters 220–300) over ~3 minutes of sustained full-GPU compute. That's a ~45% slowdown purely from thermals. This pulled the overall MFU from ~3.3% (early steady-state) down to ~2.82% (run average).

**Why non-trivial**: Benchmark numbers look different depending on when you sample them. The "first clean window" post-compile is optimistic; the thermal steady state (after ~5 min) is the representative number.

**Decision implication**: For MFU benchmarks, either (a) run long enough to reach thermal steady state and report that, or (b) be explicit about whether numbers are "early steady state" vs "thermal steady state." The gap is ~40% — not negligible. Also: don't compare benchmarks between short runs and long runs directly.

---

## H100 BF16 MAMF = 794.5 TFLOPS, not 989 (2026-04-01)

MAMF (Maximum Achievable Matmul FLOPS) is the peak throughput actually measurable on real hardware at the best matrix shapes. H100 SXM BF16 MAMF benchmarks (from ml-engineering book):
- Theoretical peak: 989 TFLOPS
- MAMF achieved: **794.5 TFLOPS** (80.3% efficiency at ideal shapes)
- Best shape tested: 2048×2048×13312

At 11% MFU (109 TFLOPS achieved): **13.7% of MAMF** — that's the true compute efficiency gap to close.

**Why non-trivial**: We use 989 TFLOPS as the denominator in `compute_mfu()`. That's correct for apples-to-apples MFU comparison (it's the convention). But when reasoning about how much headroom exists, the realistic ceiling is 794.5, not 989. "30% MFU" = 297 TFLOPS, which is 37% of MAMF — very achievable.

**Decision implication**: When thinking "how far are we from the hardware limit" use MAMF (794.5) not theoretical (989). The gap is 109 → 794.5 = 7.3× — but reaching 37% of MAMF (30% MFU target) is a 2.7× improvement from current.

---

## Python GC causes multi-GPU stragglers at scale (2026-04-01)

On 8+ GPUs, automatic Python garbage collection runs at different times on different ranks. Each rank pauses briefly → rank 0 may finish backward while rank 7 is GCing → straggler-induced idle time. Shows as periodic dips in MFU or step-time spikes.

```python
import gc
gc.disable()   # At trainer start
# Then manually in training loop:
if iteration % 100 == 0:
    gc.collect()
```

**Why non-trivial**: Not documented in PyTorch training guides. Visible only in long runs with many ranks. GC pauses are small but desynchronized across ranks, causing the fast ranks to wait.

**Decision implication**: Add `gc.disable()` to DINOv3 training startup. Add periodic `gc.collect()` every 100 iters.

---

## Tensor dimension alignment: seq_len and hidden_dim matter; batch size doesn't (2026-04-01)

For Tensor Core efficiency (tile and wave quantization effects):
- **Matters greatly**: sequence length, hidden dimension — must be multiples of 64–128 for good alignment
- **Minimal impact**: batch size — "usually has little to no impact" on matmul efficiency
- ViT-B: hidden_dim=768 (divisible by 64 ✓), seq_len=197 — 197 is **not** a multiple of 64 (196 patches + 1 CLS)

197 / 64 = 3.08 (not aligned). The 3 patches "waste" doesn't matter at these small sequence lengths — but if we ever pad or change sequence structure, keep it at multiples of 64.

**Why non-trivial**: Common advice is "align your batch size to powers of 2." This is mostly wrong for modern matmuls. Sequence length and hidden dim are what matter for tensor core tile alignment.

**Decision implication**: Don't stress over batch size alignment. If sequence length ever changes (e.g., different patch size or crop size), target multiples of 64.
