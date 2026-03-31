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
