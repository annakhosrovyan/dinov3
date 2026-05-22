## Context for Armen — NCCL Bottlenecks in DINOv3

**What we're training:**
Satellite-specialized DINOv3 (Meta's latest DINO variant) on a YSU/YerevanNN cluster. 8×H100 NVLink, single node. Model is ViT-B (86M params, 12 blocks, dim=768), 5-channel satellite input (Sentinel-1/2, NAIP), DINO + iBOT + KoLeo objectives with 2 global crops (seq=197) + 8 local crops (seq=37) per step.

**Distributed strategy:** FSDP2 ZeRO-3 (`reshard_after_forward=True`). Block-level wrapping — each of 12 ViT blocks is an independent FSDP unit. BF16 params, FP32 gradient reduction.

**Measured bottleneck (nsys, job 39141, 2026-05-08):**
- **NCCL = 80.4% of all GPU kernel time**
- `ncclDevKernel_AllGather_RING_LL` alone = **69.5%** of all kernel time (36,161 ms vs 3,547 ms matmul — NCCL is 10× more than compute)
- Compute kernels: only 6.8% matmul, 1.6% FlashAttention
- **NCCL ↔ compute overlap: 11%** — 89% of NCCL time serializes against compute
- Root cause: ZeRO-3 reshard triggers 1 AllGather per block per forward + 1 per block per backward = 24 AllGathers/step, each ~12.7 ms avg, against a ViT-B GEMM that's too small to hide a 12 ms collective

**Current throughput:** ~7–8% MFU hardware dense (989 TFLOPS H100 BF16 dense). Lab target is 30%.

**What we've already tried:**
- `NCCL_PROTO=LL128`: +12% in a 300-iter test, but a 1000-iter repeat on the same node got only +6-7% with high CV. Not shipped — cause unclear (cluster background load vs long-run effects).
- `TORCH_NCCL_AVOID_RECORD_STREAMS=1`: +5.7% in isolation, stacks poorly with LL128 (−6% combined)
- `NCCL_NVLS_ENABLE=1`: −8.7% (autotuner doesn't choose NVLS anyway for our message sizes)
- `NCCL_BUFFSIZE=16M`: −20.6%, avoid
- `NCCL_NTHREADS=256`: −10% (halving threads from default 512)
- `reshard_after_forward=False` (ZeRO-2-equivalent): removes the forward AllGather but adds ~10 GB memory pressure at bs=96. Matrix shows no throughput advantage in our current measurement window.

**Key constraint you should know about:**
iBOT uses stochastic per-image mask ratios (random mask ratio per image per batch, non-deterministic token counts). This means `torch.compile` can only run in `mode=None` (no specialization). `max-autotune` and CUDA graphs are fundamentally incompatible — we confirmed this. So we cannot use CUDA graph capture to amortize kernel launch overhead.

**Questions we have:**
1. At 8×H100 on a single node with NVLink, 12-block FSDP2 ZeRO-3 giving 11% NCCL-compute overlap — is the wrapping granularity the main problem? Would coarser wrapping (e.g., 4 blocks per FSDP unit = 3 units instead of 12) significantly change the overlap picture, or is the ViT-B compute density just too low to hide even a coarser AllGather?
2. Gradient accumulation (accumulate N micro-steps before AllReduce) — would that help here? We currently do one NCCL sync per step. With FSDP2 ZeRO-3, does grad accum meaningfully reduce the AllGather count (the forward AllGathers still happen every micro-step)?
3. LL128 gave +12% in a short soak but wouldn't reliably reproduce over 1000 iters. What's your read on why LL128 might be unstable on longer runs — allocator pressure? NCCL buffer contention? Thermal/clock interaction?
4. Is there a path to ≥20% MFU at 8×H100 for a ViT-B in this regime without going to more nodes, or is this a wall — too little compute per collective?

---

**The card in `status.html`** now has: job ID, date, node, script invocation, all key training config flags (FSDP2, compile, AC, precision, dataset, reshard mode), nsys capture parameters (delay=360s/duration=120s and why), and the full bottleneck breakdown from the summary — so it's self-contained for a quick briefing or referencing mid-Q&A.