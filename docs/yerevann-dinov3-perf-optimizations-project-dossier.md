# YerevaNN DINOv3 Performance Optimizations — Project Dossier

**Branch**: `perf-fsdp2-pipeline`  
**Opened**: 2026-05-07  
**Updated**: 2026-05-20  
**Cluster**: BCM11 — 8× DGX-style nodes, each with 8× H100 80GB SXM5  
**Model**: ViT-B (85M params), 5-channel satellite imagery, DINOv3 DINO+iBOT objectives  

---

## Current Status (authoritative as of 2026-05-20)

| Item | State |
|------|-------|
| **Production default** | FSDP2 ZeRO-3, `bs=96`, selective AC, `reshT` — `1,351 img/s`, `7.73% MFU`, `14.3 GB peak` |
| **Memory fallback** | `bs=96 + full AC` — saves 5.4 GB, `<1.5%` throughput cost (`1,317 img/s`, `8.9 GB`) |
| **NCCL LL128 knob** | +12.1% in Stage A (300 iter, CV 4.9%). Stage C 1000-iter rerun didn't replicate cleanly. **Not shipped.** |
| **bs=128** | **REOPENED** — `bs=128 reshF sel AC` (job 47554, gpu05, 1000 iter): mean `1,341` / median `1,312 img/s`, `7.67% MFU`. Cross-node vs gpu07 anchor. Needs paired same-node A/B + OOM-history clarification. |
| **Next experiment** | Paired same-node A/B: `bs=128 reshT` vs `reshF` (1000 iter, eval+ckpt off). Resolve original OOM (reshT or reshF?). LL128 stacking is after. |
| **Current MFU vs target** | ~7–8% (8-GPU) vs **30% lab target** |

---

## Phase 5 Progress Gates (5/8 complete)

- [x] nsys traces captured — jobs 39140 (bs=128) and 39141 (bs=96)
- [x] Bottleneck identified: NCCL at 70–80% kernel time, 11–16% overlap
- [x] AC verified at bs=96 — both selective and full modes stable
- [x] Short soak with eval + checkpoint pressure (forced at iter 400/800)
- [x] Full bs×AC matrix measured — all 6 cells (reshT row)
- [ ] Choose production candidate + longer validation run
- [ ] Phase 5 row added to `archive_decisions.md`
- [ ] `run.sh` updated with rollback note

---

## Key Empirical Results

### Batch × AC Matrix (primary — all reshT, gpu07, iter 400–999)

| Config | img/s | MFU | Peak Mem | Job | Notes |
|--------|-------|-----|----------|-----|-------|
| **bs=96 sel AC ★** | **1,351** | **7.73%** | 14.3 GB | 45280 | Default |
| bs=96 full AC 💾 | 1,317 | 7.54% | 8.9 GB | 45281 | Memory fallback |
| bs=96 no AC | 1,279 | 7.32% | 24.9 GB | 45367 | Dominated by sel AC |
| bs=128 sel AC | 1,276 | 7.30% | 18.9 GB | 45366 | Clean rerun; no advantage over bs=96 |
| bs=128 full AC | 1,297 | 7.42% | 11.7 GB | 45364 | gpu03 cross-node |
| bs=128 no AC | 1,267 | 7.25% | 33.0 GB | 45282 | OOM risk unresolved |

**Key surprise**: Selective AC is *faster* than no-AC at bs=96 (+5.6%). Keeping large activation tensors live creates HBM cache pressure and allocator overhead that hurts more than the recompute cost. This is regime-specific — H100 BF16 at this batch size is not compute-bound enough for the textbook tradeoff.

### reshard × AC Matrix (bs=96, cross-node comparison)

| Config | img/s | MFU | Peak Mem | Job | Node |
|--------|-------|-----|----------|-----|------|
| reshT sel AC ★ | 1,351 | 7.73% | 14.3 GB | 45280 | gpu07 |
| reshT no AC | 1,279 | 7.32% | 24.9 GB | 45367 | gpu07 |
| **reshF no AC** | **1,143** | **6.54%** | 25.1 GB | 45368 | gpu07 — **worst cell** |
| reshF sel AC | 1,305 | 7.47% | 14.5 GB | 45369 | gpu03 (x-node) |
| reshF full AC | 1,326 | 7.59% | 9.1 GB | 45370 | gpu05 (x-node) |

`reshF` without AC is the worst cell — worse than `reshT`. With AC, `reshF` is roughly tied with `reshT`, because FSDP2's prefetch already hides most of the `reshT` all-gather cost.

### Missing-Cell Probe — bs=128 × reshF × sel AC (2026-05-20)

**Job 47554** · gpu05 · 1000 iters · no NCCL knobs

- Mean: `1,341 img/s`, Median: `1,312 img/s`, MFU: `7.67%`, CV: `10.6%`, Peak mem: `18.6 GB`
- `[100,299]` ≈ `[500,999]` (1340 vs 1341) — no within-run drift over 1000 iters ✓
- +5.1% over older `bs=128 reshT` anchor (job 45366, gpu07, 300 iter, `1,276 img/s`) — but comparison **crosses node and date**

**What this establishes**: ran 1000 iters cleanly (no OOM, no retries), throughput competitive, within-run jitter ≈ CV of bs=96 runs.

**What it does NOT establish**: that the +5.1% is a reshF effect (vs gpu05-vs-gpu07 noise); that bs=128 actually beats bs=96 (1,341 vs 1,351 on gpu07 — right comparator is missing); that long-soak memory is OK (eval+ckpt off in 47554; original researcher OOM on 05-12 under unknown reshT/reshF).

**Note**: mean is partly spike-inflated (iter 300 = 1851, iter 700 = 1680); **median 1,312 is the defensible point**.

### 1-GPU vs 8-GPU: The Sign Flip

Same config (`bs=128 reshT sel AC`) produces opposite results depending on parallelism:

| | 1-GPU (no comms) | 8-GPU (with NCCL) |
|-|-----------------|-------------------|
| bs=96 | 448 img/GPU · 20.5% MFU | 169 img/GPU · 7.7% MFU |
| bs=128 | **478 img/GPU · 21.9% MFU (+6.7%)** | **160 img/GPU · 7.3% MFU (−5.6%)** |

Each GPU does only ~38% of what it can do solo. The 22 pp gap between 1-GPU (20–22%) and 8-GPU (7–8%) is almost entirely NCCL overhead not being hidden by compute.

**Update 2026-05-20**: Job 47554 (bs=128 reshF sel, 1000 iter) at `median 1,312 img/s` is the first sign that the 8-GPU sign-flip might partially reverse with reshF — but needs paired same-node A/B confirmation before any claim.

---

## NCCL Sweep Results

**Base config for all stages**: bs=96, reshT, sel AC, gpu07

### Stage A — One knob at a time (300 iter)

| ID | Config | img/s | Δ | Notes |
|----|--------|-------|---|-------|
| A2 | `NCCL_PROTO=LL128` | 1,409 | **+12.1% ★** | Best — not shipped (Stage C ≠ replicate) |
| A5 | `AVOID_RECORD_STREAMS=1` | 1,330 | +5.7% | Parked — stacks badly with LL128 |
| A6 | baseline replicate | 1,257 | +0.0% | Drift check — stable ✓ |
| A0 | baseline | 1,257 | ref | 1,257 img/s · CV 10.8% |
| A1 | `NCCL_NVLS_ENABLE=1` | 1,148 | −8.7% | Autotuner doesn't pick NVLS anyway |
| A3 | `NCCL_NTHREADS=256` | 1,132 | −10.0% | Regression — default is 512 |
| A4 | `NCCL_BUFFSIZE=16M` | 999 | −20.6% | Worst — avoid entirely |

### Stage B — Validate winners + diagnostic cells (300 iter, gpu07)

| ID | Config | img/s | Notes |
|----|--------|-------|-------|
| B3 | `bs=128 reshT sel + LL128` | 1,394 (mixed) | Regime mixed |
| B2 | `bs=96 reshT full + LL128` | 1,355 | LL128 on full AC — held up |
| B1 | `bs=96 reshT sel + LL128 + NORS` | 1,183 | **Stacked regression** — LL128 + NORS conflict |

**Stacking LL128 + AVOID_RECORD_STREAMS**: empirically negative. Both target overlapping aspects of per-collective overhead; LL128's buffer lifecycle inside NCCL conflicts with the stream-tracking changes.

### Stage C — Longer validation (1000 iter, gpu07)

| ID | Config | img/s | Notes |
|----|--------|-------|-------|
| C1 | `bs=96 reshT sel + LL128` | 1,124–1,249 | 6.4–7.1% MFU — **noisy epoch**, CV 18% |
| C2 | `bs=128 reshT sel + LL128` | 1,194 | 6.83% MFU — bs=128 with LL128 **−6.4% vs 45366** |
| C3 | `bs=96 reshT sel + LL128 + NORS` | 1,319 | +5% vs baseline but below either knob alone |

**Stage C caveat**: A2 (LL128, +12.1%) ran on gpu07 at ~1,409 in Stage A. Stage C re-ran the same config on the same node and got ~1,124 (−20%). Cause unresolved: cluster background load vs long-run allocator drift. **Do not ship LL128 without 3× controlled replicates on an idle node.**

---

## Open Questions

**Q1 — Does LL128 hold in longer runs?**  
Stage A said +12.1%, CV 4.9%. Stage C (same config, same node, 1000 iter) got CV 18% with lower throughput. Cause unresolved: cluster background load vs in-process effects (GC, allocator drift, FSDP buffer state). Needs controlled 3× replicates of A0 vs A2 on an idle gpu07.

**Q2 — What caused the original bs=128 OOM?** (RED — blocking)  
A 1000-iter memory profile (job 45282, two forced eval+ckpt cycles) survived at 33 GB peak. But the researcher-reported real-run OOM was in actual production training — longer horizon, real data, possibly specific allocator state after many checkpoint cycles. Now that bs=128 reshF sel AC is back as a candidate (47554, 18.6 GiB peak, eval+ckpt off), the question is: **was the original OOM under reshT or reshF?** Need to ask the researcher. A longer real-eval-cycle validation must precede any production switch.

**Q3 — When does `run.sh` get updated?**  
After: (a) bs=128 reshF + LL128 stacking question resolves, (b) chosen production cell survives longer validation on a clean node (>1000 iter, repeated eval+ckpt), (c) bs=128 OOM history check. Candidates: `bs=96 reshT sel`, `bs=96 reshT sel + LL128`, `bs=128 reshF sel` (±LL128).

**Q4 — Is NCCL the right growth lever at all?**  
NCCL knobs can recover at most ~1–2 pp MFU. Current: ~7–8%. Target: 30%. The 22 pp gap likely requires: gradient accumulation (fewer collectives per effective-batch step), coarser FSDP2 wrapping (fewer large collectives), or more nodes. Knob tuning is exhausted as the primary lever.

**Q5 — Is FSDP2 costing absolute throughput vs DDP at bs=96?**  
Archived DDP screening (jobs 9630/9631/9651) shows 18–23% MFU at `bs=64/128` — materially above any FSDP2 Phase 5 cell. There is no matched DDP bs=96 row. **Job 48312** (DDP bs=96, recipe-matched to cell 45367) is the missing calibration. If DDP bs=96 is dramatically faster and memory is safe across eval/ckpt, the question becomes: accept DDP for ViT-B production while keeping FSDP2 as the scale-up platform?

---

## Concepts & Glossary

### MFU — Model FLOP Utilization

```
MFU = (model_FLOPs × images_per_sec) / hardware_peak_FLOPs
```

FLOPs here are MACs × 2 (DINOv2 paper / fvcore convention). MFU does **not** account for communication time — a system at 8% MFU that's 80% communication-bound is not "wasting" 92% of compute.

**Two ceilings**:
- **989 TFLOPS**: Dense BF16. Standard MFU reporting basis (all Phase 5 numbers use this).
- **794.5 TFLOPS**: MAMF (practical ceiling, memory bandwidth limited). ~23.9% at 989 basis ≈ ~29.8% of MAMF.
- **1979 TFLOPS**: DO NOT USE — assumes 2:4 structured sparsity; misleadingly halves reported MFU.

**Where we stand**:
- 8-GPU current: **7–8%** (~74 TFLOPS actually used)
- 1-GPU compute only: **20–22%** (what the model can do without NCCL)
- Lab target: **30%** (aspirational internal target, not a Meta paper claim)

### FSDP2 — Fully Sharded Data Parallel v2

Each GPU holds 1/N of each layer's parameters. Before each layer's forward, an all-gather reconstructs the full layer. After each layer's backward, a reduce-scatter distributes gradients. ZeRO-3 semantics when `reshard_after_forward=True`.

For ViT-B (12 blocks): **24 collectives per step** (12 AG + 12 RS).

**Knob**: `train.fsdp_reshard_after_forward` (authoritative in our fork — see Upstream PRs section).

### reshT vs reshF

- **reshT** (`reshard_after_forward=True`, default): ZeRO-3 — parameters re-sharded after each layer's forward. Minimal resident memory, maximum communication.
- **reshF** (`reshard_after_forward=False`, "no-release"): Parameters stay unsharded through the full forward pass — closer to DDP memory layout.

At **bs=96**, reshF without AC is the **worst cell** (1,143 vs 1,351 for reshT sel). With AC, roughly tied — FSDP2 prefetch already hides the reshT all-gather cost. At **bs=128**, a possible story flip is under investigation (2026-05-20, job 47554).

### AC — Activation Checkpointing

Frees activation tensors during forward; recomputes during backward. Trades FLOPs for HBM.
- **Selective AC**: saves expensive outputs (matmul, SDPA, reduce-scatter tensors), recomputes the rest. Preferred.
- **Full AC**: recomputes everything. Maximum memory savings.

At bs=96: sel AC saves 10.6 GB (24.9 → 14.3 GB) AND is throughput-positive (+5.6%). Full AC saves an additional 5.4 GB with <1.5% throughput cost.

### NCCL — NVIDIA Collective Communications Library

Implements all-gather (assemble sharded params), reduce-scatter (distribute gradients), all-reduce (sync metrics). Runs as CUDA kernels (`ncclDevKernel_*`).

**Phase 5 profiler finding**: NCCL = 70–80% of kernel time at 8-GPU. NCCL↔compute overlap = only 11–16%. **This is the dominant bottleneck.**

### LL128 / NCCL_PROTO=LL128

Low-latency 128-byte aligned double-buffering. Reduces per-chunk overhead for medium-sized collectives. Forces a specific wire protocol instead of letting NCCL's autotuner choose.

### NORS / AVOID_RECORD_STREAMS

`TORCH_NCCL_AVOID_RECORD_STREAMS=1` — removes PyTorch per-launch stream-record bookkeeping on NCCL buffers. Reduces CPU-side dispatch overhead and allocator churn. **Does not combine well with LL128** (Stage B B1: −6% combined).

### NVLS / NVSwitch SHARP

NVSwitch-aware algorithm that reduces in the switch fabric rather than over NVLink. Set via `NCCL_NVLS_ENABLE=1` or `NCCL_ALGO=NVLS`. Stage A A1: −8.7%. Autotuner didn't select NVLS. This workload's collective sizes don't land in NVLS's sweet spot.

### CV — Coefficient of Variation

Standard deviation ÷ mean, expressed as %. Lower CV = more stable, more trustworthy single-run comparisons. Rule of thumb: **δ < CV is untrustworthy**. Stage A A2 (LL128): CV 4.9% — very stable. Stage C baseline (C1): CV 18% — hard to interpret.

### iBOT — Masked Image Modeling Loss

One of two self-supervised objectives (alongside DINO). Uses stochastic per-batch mask shapes. **This is why `max-autotune` and CUDA graphs are incompatible with this codebase** — stochastic shapes prevent static graph capture. Only `compile_mode=null` (default induction-only) works.

### ZeRO-3

Shards parameters, gradients, AND optimizer states across all GPUs. For ViT-B (85M params) at 8-GPU: each GPU holds ~10.6M params during idle; during each layer's forward, the other ~74.4M are gathered then immediately released.

### MAMF — Maximum Achievable MFU

Practical ceiling accounting for HBM bandwidth limits and realistic matmul efficiency. ~794.5 TFLOPS for H100 BF16. ~23.9% MFU at the 989-TFLOPS basis ≈ ~29.8% of MAMF.

---

## The 4 Bottleneck Hypotheses (Tested Against nsys Traces)

Phase 5 opened with four candidates. nsys traces from jobs 39140 (bs=128) and 39141 (bs=96) tested all four.

**A — Multi-crop Sequential Forward**  
Predicted ≥3× attention kernel bimodality from processing global (197 tokens) and local (37 tokens) crops sequentially. Evidence: nsys showed `flash_fwd_kernel` min/max ratio of only 1.5–1.7×. **Demoted — modest effect, not primary.**

**B — NCCL Serialization ✓ CONFIRMED**  
24 collectives per step; if they dominate step time and can't be overlapped, NCCL becomes the wall-clock bottleneck. Evidence: NCCL = 70% (bs=128) / 80% (bs=96) of kernel time. NCCL↔compute overlap = 16% / 11%. AllGather alone = 347 sec over a 90-sec trace window (37,024 launches). **Dominant bottleneck.**

**C — H2D Not Overlapped**  
H2D transfers not actually overlapping with NCCL/compute. Evidence: H2D↔compute overlap = 0%, but H2D total = ~1% of GPU time. **Real but minor (<1% of step time).**

**D — Python GC / Eval / Checkpoint Stalls**  
GC across ranks at different times creating periodic stragglers. Evidence: bs=128 p99 union gap = 0.5 ms (no visible GC stalls). `gc.disable()` + manual `gc.collect()` already partially mitigated. **Not primary** — but see Upstream PR #340 section below for a GC-related risk.

**Bottom line**: Hypothesis B is the one. Fixing it requires reducing NCCL volume (gradient accumulation, coarser wrapping) or improving overlap — not kernel optimization.

---

## FSDP2 Communication Model

```
ONE TRAINING STEP — ViT-B, 12 transformer blocks, FSDP2 ZeRO-3

NCCL:    [AllGather×12]  [ReduceScatter×12]  [AllReduce (metrics)]
Compute:       [fwd]           [bwd]
Overlap:  ↑~11–16% of NCCL time overlaps with compute (rest is wasted wait)

Total step: ~560–810 ms (varies by batch size and node)
```

| Stat | Value |
|------|-------|
| NCCL fraction of kernel time | 70–80% |
| NCCL↔compute overlap | 11–16% |
| Collectives per step | 24 (12 AG + 12 RS) |

Why bigger batch hurts at 8-GPU: larger activations mean more all-gather payload per step. ViT-B (~85M params) is small, so NCCL volume is already large relative to compute at any batch size — scaling batch increases both, but NCCL grows proportionally more.

---

## Why FSDP2 (Strategic Rationale)

> "I have not used DDP once since 2022 … In FSDP you have the option to not release shards, in which case it's basically equivalent to DDP, with the gather moved before the fwd instead of after the bwd … I think both are fine if both fit."  
> — Tim Darcet, DINOv2/v3 co-author, 2026-04-25

`reshard_after_forward=False` is mathematically equivalent to DDP (same collectives, reordered). At bs=96 with selective AC, the reshard flag barely matters once AC is on. FSDP2 provides the right memory-safety envelope as model scale or batch size increases.

**Caveat (2026-05-20)**: The "closed" claim is being re-examined empirically. `bs=96 reshF noAC` (1,143 img/s) was *worse* than `reshT noAC` (1,279). Archived DDP screening (jobs 9630/9631/9651) shows 18–23% MFU — far above any FSDP2 Phase 5 cell. **Job 48312** (DDP bs=96, recipe-matched to 45367) measures the actual gap. The question, if DDP is dramatically faster and memory-safe: accept DDP for ViT-B production while keeping FSDP2 as the scale-up platform?

**Operational rule**: All Phase 5 experiments use FSDP2 ZeRO-3 (`reshard_after_forward=True` unless explicitly varied). FSDP2 remains the scale-up platform of record.

---

## Upstream DINOv3 PR Findings (2026-05-20)

Source document: `docs/upstream-pr-findings-2026-05-20.md`  
PRs: [#324](https://github.com/facebookresearch/dinov3/pull/324) · [#340](https://github.com/facebookresearch/dinov3/pull/340)

**When these findings conflict with older sections above, treat this section as the newer interpretation.**

### PR #324 — FSDP2 reshard_after_forward Config

**Upstream finding**: Public upstream DINOv3 originally hardcoded `reshard_after_forward=True` in `fully_shard()` calls. The `compute_precision.sharding_strategy` config field existed but **did not change FSDP2 wrapping behavior**. PR #324 adds `train.fsdp_reshard_after_forward` and wires it through the SSL training FSDP wrapping path.

**Our fork status**: Our fork already reads `cfg.train.fsdp_reshard_after_forward` and passes it into `fully_shard()` for trained models. Therefore our Phase 5 `reshF` runs (jobs 45368, 45369, 45370) are **probably not invalidated** — they likely exercised real `reshard_after_forward=False` behavior.

**Authoritative config knob in our fork**:
```
train.fsdp_reshard_after_forward
```

**Legacy vocabulary to retire**: `compute_precision.sharding_strategy = SHARD_GRAD_OP` is not the operative FSDP2 behavior knob. Do not use it to reason about Phase 5 reshT/reshF behavior.

**Documentation corrections**:
- `train.fsdp_reshard_after_forward=true` → ZeRO-3-like (reshard after each block's forward)
- `train.fsdp_reshard_after_forward=false` → no-release / DDP-like (params unsharded through backward)
- Inference-only models (EMA teacher) may still force immediate resharding for memory predictability — does not invalidate trained-model reshF tests

**Important nuance**: `reshF` is "DDP-like" algorithmically, but not a performance guarantee of DDP-level throughput. It still runs through FSDP2's prefetch scheduling, bookkeeping, and different communication ordering. Our Phase 5 observation that `bs=96 reshF noAC` (1,143) was *worse* than `bs=96 reshT noAC` (1,279) is surprising but consistent with this.

### PR #340 — Re-enable Cycle GC to Prevent Worker RSS Leak

**Upstream finding**: Disabling automatic cycle GC (`gc.disable()`) interacts badly with `torchvision.transforms.v2`. Short-lived reference cycles accumulate in DataLoader workers, holding tensor storage references and shared-memory FDs. Observed: **~3.4 MB/iter worker RssAnon growth** with `gc.disable()` vs **~0.01 MB/iter** with cycle GC enabled. Practical rate: **~1.25 GB/hour/worker** with `num_workers=16`. No measurable iteration-time impact from re-enabling automatic cycle GC.

PR #340 removes `gc.disable()` while keeping periodic manual `gc.collect()`.

**Impact on our older docs**: Several docs say `gc.disable() + manual gc.collect() every ~100/150 iters` partially mitigates GC straggler effects. That statement is now **too strong and potentially backwards**.

**Revised interpretation**:
- `gc.disable()` may reduce automatic-GC timing noise in *short runs*
- It can create a **long-run memory leak mechanism** via DataLoader workers
- For long-running training jobs, the memory risk matters more than the noise reduction
- **Future runs should prefer automatic cycle GC enabled**, plus periodic manual collection if needed

**Revised statement on GC**:
> Periodic manual `gc.collect()` can still be useful for predictable rank-synchronized pauses, but disabling automatic cycle GC can cause long-run DataLoader worker RSS growth when `torchvision.transforms.v2` creates reference cycles.

**Relevance to OOM investigation**: Long-run OOM suspicions should now include DataLoader worker RSS growth as a suspect, especially if RSS grew outside CUDA allocator metrics. This is independent of the `bs=128` FSDP2 allocation profile.

### Practical Follow-Ups From Upstream PRs

**Code/config cleanup**:
1. Remove or deprecate `compute_precision.sharding_strategy` from configs
2. Remove stale asserts requiring `cfg.compute_precision.sharding_strategy == "SHARD_GRAD_OP"`
3. Add startup log line printing:
   ```
   train.distributed_strategy
   train.fsdp_reshard_after_forward
   train.checkpointing
   train.checkpointing_full
   ```

**FSDP2 verification**: Add a cheap smoke-test that inspects trained-model FSDP states and confirms `reshard_after_forward` is set as requested (not inferred from throughput alone).

**GC/memory verification**:
1. Remove `gc.disable()` from training loop or gate behind explicit experimental flag
2. Keep periodic manual `gc.collect()`
3. For long soaks, log DataLoader worker RSS periodically
4. Reinterpret old long-run OOM suspicions with worker RSS as a suspect

---

## Full Experiment Ledger (Phase 5)

### DDP Calibration

| ID | Date | Job | Node | Config | MFU | img/s | Peak Mem | Status |
|----|------|-----|------|--------|-----|-------|----------|--------|
| ddp-bs96-calib | 05-20 | 48312 | — | `DDP bs=96 no AC, compile=true, no ES` — matched recipe vs FSDP2 cell 45367; only `distributed_strategy` differs | — | — | — | Queued |

### Missing-Cell Probe — bs=128 × reshF × sel AC

| ID | Date | Job | Node | Config | MFU | img/s | Peak Mem | Status |
|----|------|-----|------|--------|-----|-------|----------|--------|
| bs128-reshF-sel | 05-20 | 47554 | gpu05 | `bs=128 reshF sel AC (1000 i, no NCCL knobs)` — gpu07 occupied (~1.5d wait); landed on gpu05 | 7.67% | 1,341 | 18.6 GB | ★ reopens bs=128 |

### NCCL Stage C — Longer Validation (1000 iter, gpu07)

| ID | Date | Job | Node | Config | MFU | img/s | Peak Mem | Status |
|----|------|-----|------|--------|-----|-------|----------|--------|
| C3 | 05-18 | 46031 | gpu07 | `bs=96 reshT sel + LL128 + NORS (300 i)` | 7.55% | 1,319 | 14.3 GB | done |
| C2 | 05-18 | 46030 | gpu07 | `bs=128 reshT sel + LL128 (1000 i)` | 6.83% | 1,194 | 18.9 GB | done · noisy |
| C1 | 05-18 | 46029 | gpu07 | `bs=96 reshT sel + LL128 (1000 i)` | 6.4–7.1% | 1,124–1,249 | 14.3 GB | done · noisy epoch |

### NCCL Stage B — Winners + Diagnostic (300 iter, gpu07)

| ID | Date | Job | Node | Config | MFU | img/s | Peak Mem | Status |
|----|------|-----|------|--------|-----|-------|----------|--------|
| B3 | 05-18 | 45993 | gpu07 | `bs=128 reshT sel + LL128` | 7.98% | 1,394 (mixed) | 18.9 GB | regime mixed |
| B2 | 05-18 | 45992 | gpu07 | `bs=96 reshT full + LL128` — LL128 on full AC (not sel) | 7.76% | 1,355 | 8.9 GB | done |
| B1 | 05-18 | 45991 | gpu07 | `bs=96 reshT sel + LL128 + NORS` — stacking → regression | 6.77% | 1,183 | 14.3 GB | stacked regression |

### NCCL Stage A — One Knob at a Time (300 iter, gpu07, bs=96 reshT sel AC)

| ID | Date | Job | Node | Config | MFU | img/s | Peak Mem | Status |
|----|------|-----|------|--------|-----|-------|----------|--------|
| A6 | 05-18 | 45987 | gpu07 | `bs=96 reshT sel AC — baseline replicate` — drift check | 7.20% | 1,257 | 14.3 GB | drift ref |
| A5 | 05-18 | 45986 | gpu07 | `bs=96 reshT sel + AVOID_RECORD_STREAMS` | 7.61% | 1,330 | 14.3 GB | +5.7% · parked |
| A4 | 05-18 | 45985 | gpu07 | `bs=96 reshT sel + NCCL_BUFFSIZE=16M` | 5.72% | 999 | 14.3 GB | −20.6% |
| A3 | 05-18 | 45984 | gpu07 | `bs=96 reshT sel + NCCL_NTHREADS=256` | 6.48% | 1,132 | 14.3 GB | −10.0% |
| A2 | 05-18 | 45983 | gpu07 | `bs=96 reshT sel + NCCL_PROTO=LL128` | 8.06% | 1,409 | 14.3 GB | +12.1% ★ |
| A1 | 05-18 | 45982 | gpu07 | `bs=96 reshT sel + NCCL_NVLS_ENABLE=1` | 6.57% | 1,148 | 14.3 GB | −8.7% |
| A0 | 05-18 | 45981 | gpu07 | `bs=96 reshT sel AC — baseline` | 7.20% | 1,257 | 14.3 GB | reference |

### 1-GPU Disambiguator (FSDP2 degenerates to no-shard at 1 rank, gpu07)

| ID | Date | Job | Node | Config | MFU | img/s | Peak Mem | Status |
|----|------|-----|------|--------|-----|-------|----------|--------|
| 1GPU-128 | 05-16 | 45477 | gpu07 | `bs=128 reshT sel — 1 GPU` | 21.88% | 478/GPU | 20.7 GB | +6.7% local |
| 1GPU-96 | 05-16 | 45476 | gpu07 | `bs=96 reshT sel — 1 GPU` | 20.49% | 448/GPU | 16.1 GB | 1-GPU ref |

### reshard × AC Matrix (bs=96)

Note: reshF-sel and reshF-full ran on gpu03/gpu05 — cross-node timing.

| ID | Date | Job | Node | Config | MFU | img/s | Peak Mem | Status |
|----|------|-----|------|--------|-----|-------|----------|--------|
| reshF-full | 05-15 | 45370 | gpu05 | `bs=96 reshF full AC` | 7.59% | 1,326 | 9.1 GB | done · x-node |
| reshF-sel | 05-15 | 45369 | gpu03 | `bs=96 reshF sel AC` | 7.47% | 1,305 | 14.5 GB | done · x-node |
| reshT-noAC | 05-15 | 45367 | gpu07 | `bs=96 reshT no AC` | 7.32% | 1,279 | 24.9 GB | done |
| reshF-noAC | 05-15 | 45368 | gpu07 | `bs=96 reshF no AC` | 6.54% | 1,143 | 25.1 GB | worst cell |

### Batch × AC Matrix (Primary — all reshT, 1000 iter with forced eval+ckpt at 400/800)

| ID | Date | Job | Node | Config | MFU | img/s | Peak Mem | Status |
|----|------|-----|------|--------|-----|-------|----------|--------|
| reshT-sel ★ | 05-15 | 45280 | gpu07 | `bs=96 reshT sel AC` | 7.73% | 1,351 | 14.3 GB | ★ default |
| reshT-full 💾 | 05-15 | 45281 | gpu07 | `bs=96 reshT full AC` | 7.54% | 1,317 | 8.9 GB | 💾 fallback |
| bs128-sel-v2 | 05-15 | 45366 | gpu07 | `bs=128 reshT sel AC` — clean rerun, replaces BAD | 7.30% | 1,276 | 18.9 GB | no advantage |
| bs128-full | 05-15 | 45364 | gpu03 | `bs=128 reshT full AC` | 7.42% | 1,297 | 11.7 GB | dominated · x-node |
| bs128-noAC | 05-15 | 45282 | gpu07 | `bs=128 reshT no AC (memprofile)` | 7.25% | 1,267 | 33.0 GB | OOM unresolved |
| bs128-sel-BAD | 05-15 | 45363 | gpu01 | `bs=128 reshT sel AC` — 8-GPU job (GPU-exclusive); step time halved at iter 400 (604→297 ms); cause unknown (CPU co-scheduling, thermal, or NCCL algo reselection — not another GPU job). Memory (18.9 GB) trustworthy; timing is not. | 19.55% | 3,416 !!! | 18.9 GB | ⚠ timing artifact |

### Early Baselines (FSDP2 ZeRO-3, no AC, reshT, no NCCL overrides)

| ID | Date | Job | Node | Config | MFU | img/s | Peak Mem | Status |
|----|------|-----|------|--------|-----|-------|----------|--------|
| VAR2 | 05-14 | 44794 | gpu05 | `bs=96 reshT no AC` | 7.01% | 1,224 | 24.9 GB | variance ref |
| VAR1 | 05-14 | 44793 | gpu01 | `bs=96 reshT no AC` | 7.39% | 1,291 | 24.9 GB | variance ref |
| P5-00 | 05-13 | 44023 | gpu08 | `bs=96 reshT no AC` | 14.67% | 2,564 | 24.9 GB | ⚠ gpu08 idle outlier |
| P5-stacked | 05-13 | 44024 | gpu08 | `bs=96 reshF + NCCL_ALGO=NVLS + DEBUG=INFO` — 3 knobs at once, confounded | 6.59% | 1,151 | 25.1 GB | confounded |

### ⚠ Archived — Pre-Phase-5 DDP Screening (NOT current path, short soaks, not production-validated)

| ID | Date | Job | Config | MFU | img/s | Peak Mem | Notes |
|----|------|-----|--------|-----|-------|----------|-------|
| DDP-128 | ~04 | 9631 | `DDP bs=128 no AC` | 23.1% | ~4,042 | ~34.0 GB | Archived reference. DDP closed as path (Tim Darcet 2026-04-25). DDP bs=96 calibration run (48312) tests gap at production batch. |
| DDP-ES-128 | ~04 | 9651 | `DDP + expandable_segments bs=128 no AC` | 22.9% | ~3,993 | ~34.1 GB | Archived |
| DDP-64 | ~04 | 9630 | `DDP bs=64 no AC` | 18.1% | ~3,169 | n/a | Archived |

---

## Cluster Variance Notes

The 2× step-time drop in bs128-sel-BAD (job 45363, gpu01, iter 400: 604→297 ms) is an 8-GPU job with exclusive GPU access — **another GPU job cannot have been co-scheduled on that node** (DGX H100 nodes have exactly 8 GPUs; an 8-GPU job holds them exclusively). Non-GPU resources that remain shared/variable:
- 160 of 224 CPUs are not in the GPU allocation — CPU-side jobs can be co-scheduled
- Weka client I/O contention
- Per-node clock state / GPU thermal throttling  
- NCCL algorithm reselection (can happen mid-run)

Cluster inter-run variance is real (range: 1,143–1,351 img/s for bs=96 across Phase 5). Sources: co-scheduled CPU jobs, Weka client contention, per-node clock/throttle variation, NCCL collective algorithm reselection. **Single-run MFU numbers are reliable within ~5–10%; treat differences <CV as noise.**

---

## Model and Training Config Quick Reference

**ViT-B** · `embed_dim=768` · `depth=12` · `num_heads=12` · `patch_size=16` · `in_chans=5` (satellite)  
**Global crop tokens**: 197 (196 patches + 1 CLS + 0 register tokens)  
**Local crop tokens**: 37  
**H100 BF16 peak (dense)**: 989 TFLOPS  
**H100 BF16 MAMF**: ~794.5 TFLOPS

**Key config keys for performance**:
```yaml
train.distributed_strategy: fsdp2       # "fsdp2" or "ddp"
train.fsdp_reshard_after_forward: true   # true=ZeRO-3/reshT; false=no-release/reshF
train.checkpointing: true               # selective AC (preferred)
train.checkpointing_full: false         # full AC (memory fallback only)
train.compile: true                     # torch.compile per block
train.cudagraphs: false                 # incompatible with iBOT dynamic masks
train.batch_size_per_gpu: 96           # safe operating point
compute_precision.param_dtype: bf16
compute_precision.reduce_dtype: fp32
crops.local_crops_number: 8
```

**Do not use**: `compute_precision.sharding_strategy` (legacy, no effect on FSDP2 behavior in our fork).

---

## Sources

- `docs/phase5_perf_plan.md` — full narrative and raw experiment data (source of truth, ~17k words)
- `docs/upstream-pr-findings-2026-05-20.md` — PR #324 and PR #340 findings
- `docs/claude-html-files/status.html` — status dashboard (HTML)
- `docs/claude-html-files/concepts.html` — concepts and glossary (HTML)
- nsys traces from jobs 39140 (bs=128) and 39141 (bs=96) — bottleneck evidence
- Tim Darcet (DINOv2/v3 co-author) on FSDP2 vs DDP — 2026-04-25
