# Phase 5 — FSDP2 + Data/Compute Pipeline Optimization Plan

> **STATUS: CLOSED 2026-05-21.** Phase 5 is concluded by external advisory input from
> **Armen Aghajanyan** (oracle/advisor). The wrap-up grid (jobs 51069 / 51165 / 51166)
> was cancelled; the remaining results no longer drive a decision. The new path is
> **Phase 6 — DDP + `torch.compile(fullgraph=True)`** — see `docs/phase6_perf_plan.md`.
>
> **What Phase 5 settled:**
> - FSDP2 is overkill for ViT-B (~85M params). `reshard_after_forward` cannot pay back
>   the all-gather/reduce-scatter overhead at this scale. Stick to DDP. (Armen, 2026-05-21,
>   converges with Tim Darcet's earlier "either is fine if it fits" but pushes harder.)
> - Job 48312 (DDP bs=96, compile=true, no AC, no ES) is the current verified DDP baseline:
>   iter 400–999 mean **1,387 img/s, 7.94% MFU, step_ms 545, peak alloc 25.8 GB on gpu03**.
>   That number — not the archived 2026-04 ~4 k img/s screening — is what Phase 6 has to beat.
> - The headroom Armen expects from this scale is hidden behind two compile flags we never
>   actually flipped: **`fullgraph=True`** and **`triton.cudagraphs=true`**. Both are wired
>   in the code (`dinov3/fsdp/ac_compile_parallelize.py:72`) but the default config gates
>   them off (`cudagraphs: false` in `dinov3/configs/ssl_default_config.yaml:82`).
> - The principal blocker for fullgraph capture is the **dynamic shape of `mask_indices_list`**
>   (`dinov3/data/collate.py:63`) and the consequent dynamic slice `loss[:n_masked_patches]`
>   (`dinov3/loss/ibot_patch_loss.py:115`). This is the design problem Phase 6 has to solve
>   — or work around — before fullgraph numbers are meaningful.
>
> **What Phase 5 left unfinished (and where it lives now):**
> - Same-shape wrap-up 2×2 grid (bs∈{96,128} × resh∈{T,F}, eval+ckpt OFF). **Cancelled.**
>   The premise (FSDP2 worth optimizing) was retired.
> - The bs=128 FSDP2 OOM root cause (researcher report, 2026-05-12). **Closed without resolution
>   — no longer load-bearing, since DDP doesn't reshard.**
> - Archived 100-iter DDP vs current long-soak convention mismatch (jobs 9631 etc. reporting
>   ~4 k img/s vs 48312's 1.4 k). **Deferred to Phase 6.1 as a diagnostic** — same answer
>   would inform "is the archived screening a fair upper-bound to beat, or just measurement
>   artifact?".
>
> Content below is preserved for reference and to ground Phase 6's open questions. Where it
> conflicts with this header or Phase 6, this header is authoritative.

---

| Field | Value |
|---|---|
| Branch | `perf-fsdp2-pipeline` (branched from `perf-ddp-vs-fsdp` @ `a281fc8`) |
| Date opened | 2026-05-07 |
| Date closed | 2026-05-21 |
| Predecessor | Phase 4 — FSDP2 no-release screening (closed 2026-04-27) |
| Successor | Phase 6 — DDP + `torch.compile(fullgraph=True)` (`docs/phase6_perf_plan.md`) |
| ~~Strategy going forward~~ | ~~FSDP2 ZeRO-3 only.~~ **Reversed 2026-05-21 by Armen Aghajanyan: DDP only for ViT-B; FSDP2 is overkill at 85M params.** |
| Production-viable batch sizes | **bs=64, bs=96** confirmed safe under FSDP2; **bs=96 DDP** confirmed safe (job 48312). bs=128 status under DDP not yet measured. |
| Operating point we're optimizing | (legacy) FSDP2 ZeRO-3 bs=96 — now superseded by DDP bs=96 in Phase 6. |
| Goal (legacy) | Complete the bs=128 AC matrix, choose the production candidate, then run a longer validation before touching `run.sh` |

> **Note**: This file was reconstructed on 2026-05-11 from the conversation transcript after the
> original was lost when a Claude Code process was killed mid-session. Content is recovered;
> a few less-essential paragraphs may be terser than the original.

## Current conclusion (2026-05-15)

- The current safe production baseline remains **FSDP2 ZeRO-3 bs=96 with no AC in `run.sh`**.
- **Selective and full activation checkpointing are both verified at bs=96.** Selective AC is the preferred throughput candidate; full AC is the memory fallback.
- **bs=128 is still investigational.** A 1000-iter no-AC memory profile survived, but that does not retire the earlier researcher-reported long-run OOM.
- The next decision gate is the **bs=128 selective/full AC matrix** (jobs 45363/45364). Do not promote bs=128 or AC into `run.sh` until those results and a longer validation are reviewed.
- Older sections below are retained for history. When they conflict with this block, the status snapshot, the experiment ledger, and §11 are authoritative.

---

## 1. Why a new branch (and not back to `mfu-tracking-baseline`)

`perf-ddp-vs-fsdp` carries the pieces this phase needs:
- MFU instrumentation (`dinov3/utils/mfu.py`, CUDA-event timing in `train.py`)
- FSDP2 wrapping fixes (`ac_compile_parallelize.py`)
- Worst-case memory profiling infra (`scripts/memprofile_*.sh`)
- The validated production config in `run.sh`
- Compile-mode learnings (max-autotune incompatible with iBOT — closed permanently)

Re-branching from `mfu-tracking-baseline` would force re-doing all of that. Merging back to
`master` is deferred per project preference.

---

## 2. Status snapshot — current state (verified 2026-05-15)

| Knob | State | Where |
|---|---|---|
| `num_workers=20` (2.5/GPU) | ✓ on | `run.sh:84` |
| `pin_memory=True` (always) | ✓ on | `dinov3/data/loaders.py:246` |
| `non_blocking=True` for H2D (7 tensors) | ✓ on | `dinov3/train/ssl_meta_arch.py:396-407` |
| `persistent_workers=True` | ✓ on | `run.sh:87` |
| `prefetch_factor=8` | ✓ on | `run.sh:88` |
| `cudnn.benchmark=True` | ✓ on | `dinov3/train/train.py:45` |
| `matmul.allow_tf32=True` | ✓ on | `dinov3/train/train.py:44` |
| FlashAttention2 (via SDPA dispatch on H100/bf16) | ✓ effectively on | `dinov3/layers/attention.py:116,159` |
| Production AC setting | ✗ off in `run.sh` | Default: `train.checkpointing=false`, `train.checkpointing_full=false` in `dinov3/configs/ssl_default_config.yaml:78-79`; no override in `run.sh` |
| Selective activation checkpointing | ✓ verified in screening; not yet promoted | `train.checkpointing=true`, `train.checkpointing_full=false`; wrapper in `dinov3/fsdp/ac_compile_parallelize.py:25-47,59-62`; bs=96 jobs 44795/45280 |
| Full activation checkpointing | ✓ verified in screening; memory fallback | `train.checkpointing=true`, `train.checkpointing_full=true`; same wrapper path; bs=96 job 45281 |
| Memory profiling at eval + checkpoint boundaries | ✓ available and used in AC screens | `DINOV3_MEMORY_PROFILE=1`; phase-boundary logging in `dinov3/train/train.py:571-572` and profiling helpers |
| **Explicit copy_stream / compute_stream** | ✗ **not implemented** | — |
| **Packed multi-crop attention (NestedTensor / FA2 varlen)** | ✗ **not implemented** | — |
| NCCL bucket size tuning | ✗ default | — |
| NVTX annotations | ✓ profiling-gated ranges exist | Train loop ranges in `dinov3/train/train.py:595-662`; SSL phase ranges in `dinov3/train/ssl_meta_arch.py:395-445` |

Most low-hanging pure-config knobs are already pulled. The major May 15 update is that
activation checkpointing is no longer theoretical: both selective and full AC paths run at
bs=96, with selective preferred for throughput and full kept as the memory fallback. AC has
not yet been promoted to the production `run.sh` recipe; the bs=128 AC matrix is the next
decision gate. Remaining non-AC levers require code changes (copy_stream, packed attention)
or diagnostic work (nsys + FSDP2 communication shape).

---

## 3. Bottleneck findings

| Hypothesis | Current verdict | Evidence / next action |
|---|---|---|
| **A. Multi-crop forward structure** | Demoted | nsys showed only modest attention-kernel bimodality (1.5-1.7×), not the expected ≥3× signature. Packed attention remains a future high-effort option, not a current Phase 5 lever. |
| **B. NCCL all-gather serialization** | **Confirmed dominant trace signal** | Jobs 39140/39141 showed NCCL at 70-80% of kernel time with only 11-16% NCCL↔compute overlap. Original no-reshard/NVLS stack regressed, so this is not solved yet. |
| **C. H2D not actually overlapped** | Real but low priority | H2D overlap was ~0%, but H2D was only ~1% of GPU time in the usable traces. Do not build a copy-stream pipeline unless a newer trace says data movement became material. |
| **D. Periodic Python GC / eval / checkpoint stalls** | Not primary in trace; still worth monitoring | Manual GC is already in the train loop. The changing straggler-rank signal is unresolved but not currently the top lever. |

The trace-backed bottleneck is communication, but the active Phase 5 path has shifted to
activation checkpointing because it is the practical route to testing bs=128 safely.

---

## 4. Background articles ingested 2026-05-07

These were the inputs for the original Phase 5 plan. They are kept as background, not as
the current execution order.

### Tier 1 — still relevant
1. **CPU→GPU transfer (Articles 2 & 3)** — three-stage diagnosis: multi-process loader → pinned+non_blocking → explicit CUDA streams. We're at stage 2; stage 3 only if trace shows H2D stalling compute. Also: NCCL `bucket_cap_mb` is under-tuned by default (article reports 4% gain bumping to 100 MB on NVLink).
2. **Pipelining with CUDA streams (Article 4)** — pattern reference for copy_stream if Hypothesis C wins.
3. **Variable-length sequences (Article 5)** — structural lever for Hypothesis A (NestedTensor + FA2 varlen for multi-crop). Reported 2.5–3× attention speedups. **High effort; only pursue if profile confirms forward-pass dominates step time.**

### Tier 2 — diagnostic value
4. **Caching strategy (Article 1)** — bisect I/O time location. Useful as follow-up if nsys says data loading IS bottlenecked but doesn't explain why.

### Tier 3 — not directly relevant
5. Inference-side article on batched transfer.

---

## 5. Current plan of attack

### Step 1 — Finish the bs=128 AC matrix

The immediate open cells are:

| Config | Purpose |
|---|---|
| bs=128 + selective AC (job 45363) | Tests whether the preferred AC variant scales to bs=128. |
| bs=128 + full AC (job 45364) | Tests the memory-fallback variant and completes the 2×3 batch×AC matrix. |

Both should be judged against same-node or replicated comparisons where possible; current
MFU variance is too large for single cross-node deltas to be decisive.

### Step 2 — Choose the candidate

| If results show… | Decision |
|---|---|
| bs=128 + selective AC is stable and throughput-positive | Promote it to the long-validation candidate. |
| bs=128 + selective AC fits but is slower/noisy; full AC also fits | Keep bs=96 + selective AC as the likely production default; keep full AC as fallback. |
| bs=128 + selective AC OOMs but full AC fits | Treat full AC as a memory-rescue candidate; validate longer before promotion. |
| Both bs=128 AC variants are unstable or throughput-negative | Stay at bs=96; promote selective AC only if it improves safety without a clear throughput cost. |

### Step 3 — Long validation before promotion

Run the chosen candidate longer than 1000 iterations with repeated eval + checkpoint cycles.
Only after that should `run.sh` be updated, with a rollback note.

### Parked levers

NVLS, no-reshard, copy-stream H2D overlap, packed multi-crop attention, and allocator tuning
are all parked unless the AC matrix fails or a fresh trace points back to them.

---

## 6. Experiment ledger (Phase 5)

Unless a row says otherwise, the Phase 5 Slurm screening/memory-profile runs use the same
baseline recipe:
- **Model/data**: `vit_base`, 5 input channels, same Weka `MixedSatelliteDataset` paths, no
  pretrained weights for screening jobs (`student.pretrained_weights=""`).
- **Distributed/compile**: `torchrun --nproc_per_node=8`, `train.distributed_strategy=fsdp2`,
  `train.compile=true`, and ZeRO-3/default `train.fsdp_reshard_after_forward=true`.
- **DataLoader**: `train.num_workers=20`, `train.persistent_workers=true`,
  `train.prefetch_factor=8`, `train.cache_dataset=true`.
- **Logging/runtime**: `wandb.enabled=false` for screening jobs; `PYTORCH_CUDA_ALLOC_CONF`
  is unset in the dedicated Phase 5 scripts so DDP-era `expandable_segments:True` does not
  leak into FSDP2.
- **Varies by row**: batch size, AC mode (`checkpointing` / `checkpointing_full`), eval/checkpoint
  periods, memory profiling, nsys profiling, and experimental comm knobs such as NVLS/no-reshard.

| Exp | Date | Config | MFU % | step_ms | max_mem | Notes |
|---|---|---|---|---|---|---|
| P5-01a | 2026-05-07 | nsys: FSDP2 ZeRO-3 bs=128 (job 38829) | — | — | — | **FAILED** — `module load nsight-systems` not available on GPU nodes; nsys unresolved; aborted in ~3s. |
| P5-02a | 2026-05-07 | nsys: FSDP2 ZeRO-3 bs=96 (job 38830) | — | — | — | **FAILED** — same root cause as P5-01a. |
| P5-01 | 2026-05-08 | nsys: FSDP2 ZeRO-3 bs=128 (job 39028) | n/a | n/a | n/a | Trace ran but **multi-process attach failed** — only rank 0 captured; 35 s NCCL hang. Re-run needed. |
| P5-02 | 2026-05-08 | nsys: FSDP2 ZeRO-3 bs=96 (job 39029) | n/a | n/a | n/a | All 8 ranks captured but only ~18 s of GPU activity in the 60 s window; ~3–5 steps. Compile + pinned-pool warmup not stabilized. NCCL↔compute overlap 0% (rank 0). Re-run needed. |
| P5-01b | 2026-05-08 | nsys re-run: FSDP2 ZeRO-3 bs=128 — `NSYS_DELAY=360, NSYS_DURATION=120` (job 39140) | n/a (trace-only) | n/a | n/a | **Trace usable.** 8 ranks captured. NCCL=70% of kernel time, NCCL↔compute overlap=16.3% (rank 0). Util 93–95% per non-straggler rank. Hypothesis B confirmed as dominant. |
| P5-02b | 2026-05-08 | nsys re-run: FSDP2 ZeRO-3 bs=96  — `NSYS_DELAY=360, NSYS_DURATION=120` (job 39141) | n/a (trace-only) | n/a | n/a | **Trace usable.** 8 ranks captured. NCCL=80% of kernel time, overlap=11% (rank 0). Straggler rank changes between runs → non-deterministic. |
| P5-00 | 2026-05-13 | **bs=96 baseline reference** (FSDP2 ZeRO-3, no overrides) — job 44023, `scripts/fsdp2_bs96_baseline.sh` | **14.67%** (steady, iter≥100) | **335.4** | **24,922 MB** | Done, but later P5-VAR showed this gpu08 run is not representative of gpu01/gpu05 (~7.0-7.4% MFU). Use same-node comparisons for decisions. |
| P5-03+P5-04 | 2026-05-13 | **Stacked: `reshard_after_forward=False` + `NCCL_ALGO=NVLS` + `NCCL_DEBUG=INFO`** at bs=96 — job 44024, `scripts/fsdp2_bs96_noreshard_nvls.sh` | **6.59%** (steady, iter≥100) | **662.6** | **25,089 MB** | Done. **Large regression vs baseline: −8.1 pp MFU, 2.0× slower step.** Confounded — three knobs changed at once and `NCCL_DEBUG=INFO` produced a 51 MB log. Deconfound only if we revisit NVLS/no-reshard after AC. |
| P5-04-clean | tbd | **NCCL_ALGO=NVLS only**, no NCCL_DEBUG, `reshard_after_forward=true` at bs=96 | tbd | tbd | tbd | Parked behind AC. Isolates NVLS effect alone if we return to NCCL algorithm testing. |
| P5-03-clean | tbd | **`reshard_after_forward=False` only**, no NCCL_DEBUG, default NCCL algo at bs=96 | tbd | tbd | tbd | Deferred. Expected to be in noise around the baseline (Phase 4 bs=256 was −0.3 pp). |
| P5-AC-selective | 2026-05-14/15 | **Selective AC verified at bs=96** — jobs 44795 and 45280 | **7.00–7.60%** | **577.4** on gpu07 replicate | **14,302 MB** | Done. AC path works and cuts bs=96 peak allocation by ~10.6 GB vs no-AC reference. Preferred throughput default if AC is promoted. |
| P5-AC-full | 2026-05-15 | **Full AC verified at bs=96** — job 45281 | **7.49%** | **579.7** | **8,908 MB** | Done. Saves another ~5.4 GB vs selective at similar measured throughput; keep as memory fallback, not first-choice default. |
| P5-OOM-noAC | 2026-05-15 | **bs=128 FSDP2 no AC memory profile** — job 45282 | **7.25%** | **806.9** | **33,034 MB** | Done for 1000 iters with two forced eval+checkpoint cycles; no OOM, no allocator pathology. Does not prove long-run safety. |
| P5-bs128-AC-selective | submitted 2026-05-15 | bs=128 + selective AC — job 45363 | tbd | tbd | tbd | Pending. Completes the bs=128 selective cell in the 2×3 batch×AC matrix. |
| P5-bs128-AC-full | submitted 2026-05-15 | bs=128 + full AC — job 45364 | tbd | tbd | tbd | Pending. Completes the bs=128 full-AC cell and tests the memory-fallback path. |

---

## 7. Risks and known gotchas

- **Realistic batch-size envelope**: bs=64 and bs=96 are confirmed safe for full training. A 1000-iter bs=128 no-AC profile survived two forced eval+checkpoint cycles on 2026-05-15, but the earlier researcher-reported real-run OOM is still unresolved. Treat bs=128 as an investigation target until the AC matrix and a longer validation close the loop. bs=192/256 remain off the table.
- **FA2 dispatch is conditional**: PyTorch SDPA dispatches to FA2 only when no custom mask is passed and bf16/fp16 dtype matches. iBOT masking happens before attention so SDPA sees clean inputs — but verify in nsys by looking for `flash_fwd_kernel`.
- **NCCL bucket tuning**: in FSDP2, `bucket_cap_mb` doesn't apply directly; module wrap granularity replaces it. `reshard_after_forward=True` (current) means per-block all-gather — that's the "bucketing" we'd tune by changing wrap granularity.
- **`expandable_segments:True` is DDP-only**: noted in `run.sh:32`. Not used in any Phase 5 run.
- **Compile warmup is ~30–40 iters for FSDP2 ZeRO-3**: set nsys delay accordingly.

---

## 8. Definitions of done for Phase 5

- [x] At least one nsys trace captured for FSDP2 ZeRO-3 (jobs 39140 bs=128, 39141 bs=96, 2026-05-08). bs=128 trace is diagnostic only; bs=128 is not promoted.
- [x] One bottleneck identified with trace evidence — **B (NCCL serialization), 70–80% of kernel time, 11–16% overlap**
- [x] One lever verified and re-profiled — activation checkpointing at bs=96 (selective + full AC, jobs 44795/45280/45281)
- [x] Short soak with eval + checkpoint pressure — 1000-iter bs=96 AC runs and bs=128 no-AC memory profile each forced two combined eval+checkpoint cycles
- [ ] Complete the bs=128 AC matrix (jobs 45363/45364 pending as of 2026-05-15)
- [ ] Choose the production candidate and run a longer validation
- [ ] Phase 5 row added to `archive_decisions.md`
- [ ] If a winner emerges: `run.sh` updated with rollback note

---

## 9. Session log (historical)

This is an experiment notebook, not the source of truth for current recommendations. The
current state is summarized in the opening conclusion, §2, §5, §6, and §11.

### 2026-05-07 — Branch + plan + first nsys submission

- Created branch `perf-fsdp2-pipeline` off `perf-ddp-vs-fsdp` @ `a281fc8`.
- Wrote this plan and `scripts/nsys_profile.sh`.
- Submitted **job 38829** (FSDP2 bs=128) and **job 38830** (FSDP2 bs=96).
- **Both failed silently** — empty output dirs. Root cause (`module: command not found`):
  `module` is not available on GPU nodes. `module load nsight-systems` is a no-op; `nsys`
  never resolves; `set -euo pipefail` aborts.

### 2026-05-08 — Fix nsys path; first resubmission

- Replaced `module load nsight-systems` with direct binary path:
  `NSYS_BIN="/mnt/weka/apps/nsight-systems/2026.2.1/install/target-linux-x64/nsys"`
- Resubmitted **39028** (bs=128) + **39029** (bs=96). Both completed; traces produced.
- Lesson: **never use `module load` from a Slurm job script on this cluster.**

### 2026-05-08 — First analyzer pass; both traces underwhelming

- Wrote `scripts/nsys_dinov3_summary.py` — reusable DINOv3-specific nsys SQLite analyzer
  anchored to the 4 hypotheses (A/B/C/D). Output: markdown report next to each `.sqlite`.
  Coverage: per-device util, union/per-rank gap distribution, kernel-class breakdown,
  top-N kernels, NCCL & H2D kernels, NCCL↔compute and H2D↔compute overlap %, attention-kernel
  bimodality probe, top runtime APIs, mechanical verdict against the 4 hypotheses.
- Findings (both traces compromised):
  - **bs=128 (39028)**: nsys multi-process attach failed — only rank 0 visible. Trace
    dominated by a single 35.1 s `ncclDevKernel_AllGather_RING_LL` (hung wait state).
    Structurally unusable.
  - **bs=96 (39029)**: all 8 ranks visible but only ~17.9 s of GPU activity in the 60 s
    window. `cudaHostAlloc_v3020` at 13.4 s → pinned-memory pool still growing in
    "steady state." Only ~3–5 training steps captured.
- Conclusion: **both traces fired too early.** `NSYS_DELAY=180 s` insufficient. Real
  steady state needs ~250–300 s into the job.

### 2026-05-08 — Bump nsys defaults; resubmit; reclaim disk

- `scripts/nsys_profile.sh` defaults: `NSYS_DELAY 180→360 s`, `NSYS_DURATION 60→120 s`,
  `ITERS 1000→2000`. Added `--trace-fork-before-exec=true` to nsys for proper torchrun
  child-process attach. (`--process-scope` is not a valid flag in nsys 2026.2.1.)
- Submitted **job 39140** (FSDP2 bs=128) and **job 39141** (FSDP2 bs=96).
- Reclaimed disk: deleted unusable `.nsys-rep` + `.sqlite` from jobs 39028/39029
  (1.1 GB freed). Summary `.md` reports retained as evidence.

### 2026-05-09 — Re-runs completed; first usable traces

- Job 39140 (bs=128) and 39141 (bs=96) both produced clean .nsys-rep files (619 MB and
  242 MB respectively). 3 oom_kill events in slurm stderr for 39140 — those fired at
  end-of-job cleanup after the trace was sealed.
- SQLite export of 39140 failed twice mid-write on Weka (file got to ~91% with zeroed
  header). Third attempt succeeded: 3.5 GB, 3,396,919 kernels across all 8 devices.
  Root cause not pinned down — likely a Weka write contention quirk.
- 39141 SQLite (1.6 GB, 157,358 kernels, all 8 devices) exported on first try.
- **Important caveat**: during the capture window MFU dropped from ~21% (untraced steady
  state) to ~6.5% (with `--sample=cpu --python-sampling=true`). Profile overhead dilates
  between-kernel gaps and inflates `cudaStreamSynchronize` / NCCL exposure. So *absolute*
  NCCL share is inflated by the trace, but *relative* signatures (overlap %, straggler
  identity, H2D placement) are still diagnostic.

### 2026-05-11 — Analyzer results on both traces

Steady-state window = capture minus first 30 s.

| Signal | bs=128 (39140) | bs=96 (39141) | Notes |
|---|---|---|---|
| Per-device util (median, non-straggler) | 93–95% | 65% | bs=128 keeps compute streams much fuller |
| Straggler | rank 7 @ 33% | rank 4 @ 12% | **Different rank each run → non-deterministic** |
| Union active time | 97.1% of span | 50.0% of span | bs=128 cluster busy almost continuously |
| Union p99 gap | 0.5 ms | 13.1 ms | No GC/eval stalls — Hypothesis D ruled out for bs=128 |
| Kernel-time share — NCCL | 70.1% | 80.4% | Dominant in both (inflated by trace overhead but still 5:1+ vs compute on rank 0) |
| NCCL↔compute overlap (rank 0) | 16.3% | 11.0% | Almost no overlap — **strong Hypothesis B signal** |
| H2D↔compute overlap (rank 0) | 0.6% | 0.0% | But H2D total is ~1% of GPU time → low-priority |
| `flash_fwd_kernel` min/max ratio | 1.5× | 1.7× | Less bimodal than the seq=197/37 spread predicts |
| D2D memcpy total | 2972 GB | 202 GB | Massive — FSDP2 all-gather working buffers |

**Top kernels by total time (bs=128, all ranks, 90 s window)**:
1. `ncclDevKernel_AllGather_RING_LL` — 347 s (37,024 launches)
2. `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL` — 97 s
3. `ncclDevKernel_AllReduce_Sum_f32_RING_LL` — 39 s
4. `sm90_xmma_gemm_bf16…f32_tn_n_tilesize128x128x64` (matmul) — 31 s
5. `triton_red_fused_cat_native_layer_norm_native_layer_norm_backward_8` — 19 s

Matmul kernels combined: ~67 s (9.8% of kernel time). NCCL totals: ~483 s (70.1%).

**Hypothesis verdict**:
- **A — multi-crop sequential forward**: only modest bimodality (1.5–1.7×, not ≥3×).
  Demoted from primary suspect.
- **B — NCCL serialization**: **confirmed as the dominant lever.** Consistent with the
  Tim Darcet hint in `CLAUDE.md` — `reshard_after_forward=True` adds per-block all-gather
  overhead vs DDP-equivalent no-release.
- **C — H2D not overlapped**: real (0%) but small (~1% of GPU time at bs=128). Not worth
  the engineering cost of a custom copy_stream pipeline at this point.
- **D — periodic stalls**: ruled out for bs=128 steady state (p99 union gap = 0.5 ms).
- **Bonus — straggler signal**: a single rank lags hard in each trace, but the lagging
  rank changes between runs (rank 7 in bs=128, rank 4 in bs=96). Likely nsys-induced
  scheduling variance rather than a pinned hardware fault. Re-verify in an untraced soak.

**Next step (P5-03)**: test `train.fsdp_reshard_after_forward=false` on a short MFU
screening run at bs=128 (200–500 iters, no nsys). If MFU climbs from ~24% toward DDP+ES
levels with no memory regression, promote to a 500-iter soak.

Reports written:
- `/mnt/weka/adovlatyan/nsys_profiles/2026-05-08/39140/dinov3-fsdp2-bs128-39140.summary.md`
- `/mnt/weka/adovlatyan/nsys_profiles/2026-05-08/39141/dinov3-fsdp2-bs96-39141.summary.md`

### 2026-05-15 — Three-run gpu07 results + Codex independent verification + matrix completion

#### All-gpu07 same-node comparison (3 jobs: 45280, 45281, 45282)

| Run | MFU mean | MFU σ | img/s | step_ms | rank-0 peak_alloc | peak_reserved | alloc_retries | num_ooms |
|---|---|---|---|---|---|---|---|---|
| bs=96 + selective AC (45280) | 7.60% | 1.60 pp | 1328 | 577.4 | 14,302 MB | 15,526 MB | 0 | 0 |
| bs=96 + full AC (45281) | 7.49% | 1.25 pp | 1310 | 579.7 | **8,908 MB** | 10,062 MB | 0 | 0 |
| bs=128 NO AC (45282) | 7.25% | 1.44 pp | 1267 | 806.9 | 33,034 MB | 34,464 MB | 0 | 0 |

**Codex independent verification (gpt-5.5 high, 2026-05-15)**: independently re-parsed the logs with the same regex; numbers match within rounding. Also surfaced the **per-rank log directory** at `${OUTPUT_DIR}/logs/` (one file per rank) — verified bs=128 noAC steady `max_alloc_mb=33,034 MB` was **identical on all 8 ranks**, no skew, no rank-local hidden peak.

#### Three big findings

1. **bs=128 noAC did NOT OOM** in 1000 iters with two forced eval+checkpoint cycles. 33 GB peak alloc, 34.5 GB reserved on the 80 GB H100. No fragmentation accumulation (`fragmentation_ratio=0.008` stable through iter 999), no `alloc_retries`, no `num_ooms`. The original OOM was either longer-horizon (10k+ iters / hours), workload/data-dependent, or something has shifted in the codebase since.
2. **bs=128 is throughput-NEGATIVE vs bs=96+AC** at face value: 1267 img/s (bs=128 noAC) < 1310–1328 img/s (bs=96 + AC). Bigger batch loses to AC at smaller batch.
3. **Full vs Selective at bs=96**: full saves another 5.4 GB (14.3 → 8.9 GB) at 0.11 pp MFU cost (within noise). Selective is the right throughput default; full is the right memory fallback.

#### Codex's pushback (well-founded — adopting verbatim)

- **0.35 pp MFU gap < 1.4–1.6 pp within-run σ**. The throughput-negative claim is directional, not statistically strong, from a single run per condition.
- **"Eval" in this run was effectively a teacher-checkpoint save**, not a full validation workload. We tested checkpoint pressure, not every production failure mode.
- **1000 iters cannot rule out longer-horizon issues**: rare dynamic mask shapes, allocator state after many more save/delete cycles, dataset-order effects, node thermal variance.
- **Fragmentation has different denominators**. bs=128 noAC's "lower" 0.008 ratio vs the AC runs' 0.028 is from the bigger reserved pool — treat as "healthy/stable," not as proof of a superior allocator regime.
- **Verdict**: keep bs=96 + selective AC as the throughput default; bs=96 + full AC as the safer memory fallback. Do NOT promote bs=128 noAC based on this run.

#### Production recommendation pending matrix completion

User correctly noted we have NOT measured **bs=128 + selective AC** or **bs=128 + full AC**. Filling in those cells closes the 2×3 batch×AC matrix and lets us build the affine memory model Codex asked for. Two new jobs submitted 2026-05-15:

- **Job 45363** — `scripts/fsdp2_bs128_ac_selective.sh`. Same template as the bs=96 AC scripts but `train.batch_size_per_gpu=128`. 1.5 h walltime.
- **Job 45364** — `scripts/fsdp2_bs128_ac_full.sh`. Same but `train.checkpointing_full=true`.

Decision deferred until matrix is complete: it is possible bs=128 + AC reaches a *throughput-positive* operating point where the bigger batch overcomes the recompute cost, or it is possible the throughput-negative pattern holds. We'll know once 45363 / 45364 finish.

### 2026-05-21 — Phase 5 wrap-up plan: same-shape 2×2 grid queued (jobs 51069, 51165, 51166)

External reviewer (gpt-5.5 high-effort) re-read the matrix and pointed out a measurement-shape inconsistency that prevents a clean read of the bs=96 vs bs=128 question:

- `45280` (bs=96 reshT sel) and `45369` (bs=96 reshF sel) ran with **forced eval+ckpt at iter 400/800**.
- `47554` (bs=128 reshF sel) and the just-queued `51069` (bs=128 reshT sel) ran with **eval+ckpt OFF**.

iter 400–999 means therefore include eval-phase recovery in the bs=96 cells but not the bs=128 cells, so the 1,351 (45280) vs 1,341 (47554) "tie" is shape-confounded.

**Wrap-up grid (3 runs, eval+ckpt OFF, sel AC, compile=true, no NCCL knobs, memprofile period 50, 1000 iters)** — one variable changed between adjacent cells:

| Cell | Job | Script | Status |
|---|---|---|---|
| bs=128 reshT sel | **51069** | `scripts/fsdp2_bs128_reshT_sel_resoak.sh` | PENDING (pinned gpu05) |
| bs=96  reshT sel | **51165** | `scripts/fsdp2_bs96_reshT_sel_resoak.sh`  | PENDING (unpinned) |
| bs=96  reshF sel | **51166** | `scripts/fsdp2_bs96_reshF_sel_resoak.sh`  | PENDING (unpinned) |
| bs=128 reshF sel | 47554 (existing) | `scripts/fsdp2_bs128_reshF_sel.sh` | done (gpu05, 1,341/1,312) |

Same-node correctness within the grid was deprioritized — the cluster is saturated (60+ pending jobs from another user as of 2026-05-21), and waiting for gpu05 across three more runs would push wrap-up past several days. Cross-node variance is part of the read; record actual nodes in the analysis.

**What this grid answers**:

1. **`51069` vs `47554` (bs=128 reshT vs reshF, both on gpu05 if 51069 lands on it)** — is reshF a real lever at bs=128, or was `47554`'s lift node/date noise?
2. **`51065` vs `51069` (bs=96 vs bs=128 reshT, matched shape)** — does bs=128 actually beat bs=96 under the production-shaped reshT? Existing 45280 vs 45366 said no (-5.6 %), but with shape confound.
3. **`51166` vs `47554` (bs=96 reshF vs bs=128 reshF, matched shape)** — does reshF help bs=96 disproportionately, or does it amplify the bs=128 win?

**Interpretation pre-commit** (locked in *before* results, to avoid after-the-fact narration):

- 51069 within ~2 % of 47554 → reshF buys nothing at bs=128 same-node; the 47554 lift was likely node/date noise.
- 51069 ≥ 3 % below 47554 → reshF is a real lever at bs=128.
- 51165 ≥ 51069 (i.e. bs=96 reshT ≥ bs=128 reshT in matched shape) → existing "park bs=128" verdict stands; OOM-history question becomes the only reason to revisit bs=128.
- 51165 < 51069 by ≥ 3 % → bs=128 wins on throughput under matched shape; revisit production candidacy *after* OOM-history is settled.
- 51166 ≥ 51165 (reshF helps at bs=96) → memory cost of reshF is the next decision lever, not throughput.

**Mechanism note (from external review, lifted in for future readers)**: 8-GPU traces from jobs 39140/39141 show NCCL ~70–80 % of kernel time with only 11–16 % overlap. The 1-GPU bs=128 win (+6.7 % over bs=96) inverts at 8-GPU (−5.6 %) because the larger batch improves local GEMM efficiency but the 8-GPU critical path is dominated by exposed FSDP all-gathers / reduce-scatters that are ordered around module boundaries — extra compute does not hide tail collectives that don't start until their dependent compute completes. This is the single best-supported mechanism we have for the sign flip; cite it before invoking AC × FSDP scheduling, dynamic-shape costs, or memory pressure.

**What still remains open after this grid finishes** (in priority order):

1. **bs=128 OOM history (researcher 2026-05-12 real-run OOM, reshT vs reshF unknown).** None of these runs have eval+ckpt forced; memprofile gives steady-state peaks only. The OOM-history question requires either talking to the researcher or running one bs=128 cell with eval+ckpt forced + a long horizon, separate from the grid.
2. **Same-node nsys traces on one cell of the grid.** Current trace evidence (NCCL 70–80 %, overlap 11–16 %) is from older jobs 39140/39141. A fresh trace on the winning cell of the grid would be the cleanest path to "what specifically is exposed" — but it requires a tracing script + post-processing, not a wrap-up activity. Move to Phase 5.5 if pursued.
3. **The archived DDP convention question (9631 reported ~4 k img/s at 100 iters; 48312 reports ~1.4 k at 1000 iters on the same MFU formula).** Resolution requires running the original 100-iter `screening_ddp.sh 128` recipe on today's codebase. Until then, archived DDP numbers stay flagged as historical context only (see Q5 caveat in `docs/claude-html-files/status.html`).
4. **Instrumentation upgrade (CUDA-event sub-timers + NVTX ranges around FSDP blocks).** Would meaningfully reduce uncertainty for any future investigation but is a code change, not a wrap-up. Defer to Phase 5.5.

After the grid finishes, the wrap-up writeup should produce: (a) the pre-committed interpretations applied to actual numbers, (b) a single mechanism-anchored "why bs=128 doesn't pay off at 8-GPU" paragraph, (c) the explicit list of items above that we are *not* closing in Phase 5 and would queue as Phase 5.5 if pursued.

---

### 2026-05-20 — DDP bs=96 calibration run submitted (job 48312)

We have no exact **DDP × bs=96** row. bs=96 is the current safe FSDP2 operating point; old DDP screening only covered bs=64/bs=128 (archived jobs 9630/9631/9651, which showed 18–23 % MFU when the batch fits — materially above any FSDP2 cell measured in Phase 5). `reshard_after_forward=false` is "DDP-like" in spirit but did not recover DDP-level MFU in practice (bs=96 reshF noAC = 1,143 was *worse* than reshT noAC = 1,279). So the algorithmic equivalence (Tim Darcet) is not bearing out as a throughput equivalence.

**One run, not a sweep.** Job 48312: `DDP bs=96, no AC, compile=true, no expandable_segments, 1000 iters, forced eval+ckpt at iter 400/800, memory profile on`. Recipe is matched cell-for-cell against FSDP2 `bs=96 reshT noAC` (job 45367 = 1,279 img/s) — the *only* variable changed is `distributed_strategy: fsdp2 → ddp`. Forced eval+ckpt is included deliberately: the question is not just throughput but whether DDP bs=96 is *memory-safe across eval/checkpoint*.

Allocator: `PYTORCH_CUDA_ALLOC_CONF` unset (no ES) — this run isolates DDP itself; ES is a possible follow-up only if DDP bs=96 is throughput-positive and memory is tight.

If DDP bs=96 is dramatically faster and memory is safe, the strategic question becomes real: **accept DDP for ViT-B-scale production while keeping FSDP2 as the scale-up platform?** Script: `scripts/ddp_bs96_calibration.sh`. Node not pinned (cluster saturated). Awaiting allocation — analysis when it finishes.

#### Archived DDP reference (pre-Phase-5 screening — NOT current path)

| Job | Config | MFU | img/s | Peak mem |
|---|---|---|---|---|
| 9631 | DDP bs=128 no AC | 23.1 % | ~4,042 | ~34.0 GB |
| 9651 | DDP+ES bs=128 no AC | 22.9 % | ~3,993 | ~34.1 GB |
| 9630 | DDP bs=64 no AC | 18.1 % | ~3,169 | n/a |

Kept as reference — short soaks, not production-validated. They are the reason the DDP-vs-FSDP2 question at bs=96 is worth one calibration run.

#### 2026-05-21 — Results (job 48312, gpu03, COMPLETED, 18:27 elapsed)

Ran 1000 iters clean: exit `0:0`, no OOM, no `alloc_retries`, no `num_ooms`. Node gpu03 (not pinned). 101 logged points (every 10 iters).

**Throughput** — instantaneous `images_per_sec`, iter 400–999 window (matches the 45367 matrix-column convention):

| Metric | DDP bs=96 (48312, gpu03) | FSDP2 bs=96 reshT noAC (45367, gpu07) | Δ |
|---|---|---|---|
| img/s mean       | 1387   | 1279   | +108 (+8.4 %) |
| img/s median     | 1379   | —      | — |
| step_ms median   | 545    | 600    | −55 |
| MFU mean         | 7.94 % | 7.32 % | +0.62 pp |
| CV (iter 400–999)| 8.2 %  | —      | — |

Steady window iter 100–999: mean 1362 img/s, median 1359, CV 8.5 %, MFU mean 7.79 %.

Throughput rose monotonically across the run: iter 100–399 mean 1310 img/s → iter 410–790 mean 1373 → iter 810–1000 mean 1419.

**Memory** — rank-0 `[MEMPROFILE]`:

| Phase | max_alloc | max_reserved |
|---|---|---|
| steady_state        | 25,843 MB | 26,722 MB |
| pre_eval (both)     | 25,837 MB | 26,724 MB |
| eval_complete       | 1,346 MB  | 26,724 MB |
| checkpoint_complete | 1,349 MB  | 26,724 MB |

Peak alloc 25.8 GB (vs FSDP2 45367 24.9 GB → +0.9 GB). frag 0.014, `alloc_retries=0`, `num_ooms=0`. Eval and checkpoint phases did not exceed the training-steady peak — DDP bs=96 cleared the forced eval+ckpt at iter 400/800 without added memory pressure.

**Caveats**: single-run calibration, cross-node and cross-date comparison. DDP 48312 *observed* +8.4 % img/s versus FSDP2 45367 on the shared `iter 400–999` window, but cross-node (gpu03 vs gpu07), cross-date (2026-05-21 vs 2026-05-15), no within-cell replication. DDP also showed a monotonic upward trend across the run (1310 → 1373 → 1419 img/s) — the trend swing is larger than the reported DDP-vs-FSDP2 delta, so the `iter 400–999` scalar blends two regimes and is biased toward the late-run behavior. Attribution deferred. The eval/checkpoint memory statement is scoped to this 1000-iter recipe; do not generalize to longer training or different crops/configs.

---

### 2026-05-18 — Missing-cell probe: bs=128 + sel AC + reshF (job 47554)

Submitted after the Stage C noise discussion. The `bs=128 × reshF × sel AC` cell had never been run. Hypothesis: reshF (no-release) moves the all-gather to the forward (Tim Darcet's "basically equivalent to DDP" comm pattern). bs=128 has a +6.7 % single-GPU compute win that gets erased at 8-GPU under reshT — reshF is the principled lever to test whether the comm-pattern change recovers that win.

**Design**: one run, no NCCL knobs, 1000 iters, bs=128, sel AC, reshF, eval+ckpt off. Decision rule:
- ≥ ~1330 img/s → clean beat over 45366 (bs=128 reshT sel, 1276 img/s) and matching bs=96 reshF (45369: 1305, 45370: 1326). Queue +LL128 follow-up, reconsider "park bs=128."
- 1250–1300 img/s → noise band; lock in "park bs=128" with full coverage.
- < 1200 or OOM → reshF pathological at bs=128; shelve.

**Node note**: original submit had `--nodelist=gpu07` for same-node anchoring against 45366/46030. gpu07 was occupied (~1.5 day queue wait, abarseghyan job 47330). Resubmitted without nodelist pinning — first available H100 node will pick it up. Cross-node variance is now part of the read; the actual node will be recorded in the post-run analysis. (gpu07 itself showed ~20 % inter-epoch variance Stage A → Stage C, so the same-node argument was weaker than it sounds anyway.)

Script: `scripts/fsdp2_bs128_reshF_sel.sh`. Awaiting allocation.

---

### 2026-05-20 — Missing-cell result: bs=128 + sel AC + reshF (job 47554, gpu05)

Job 47554 ran on **gpu05** (not gpu07 — the original gpu07 pin was dropped because gpu07 was occupied for ~1.5 days; see node-note above). 1000 iters, bs=128 sel AC reshF, no NCCL knobs, eval+ckpt off.

#### Result — clean beat of the decision rule

| Window | n | img/s mean (med, sd) | CV | step_ms | MFU% |
|---|---|---|---|---|---|
| [100, 299] | 20 | **1340** (1283, 194) | 14.4 % | 741 | 7.67 |
| [200, 499] | 30 | **1352** (1305, 201) | 14.9 % | 740 | 7.74 |
| [500, 799] | 30 | **1347** (1352, 139) | 10.3 % | 736 | 7.71 |
| [800, 999] | 21 | **1331** (1298, 149) | 11.2 % | 754 | 7.62 |
| [500, 999] | 51 | **1341** (1312, 142) | 10.6 % | 744 | 7.67 |
| [200, 999] | 81 | **1345** (1311, 165) | 12.3 % | 742 | 7.70 |

max_mem 19068 MB (≈18.6 GiB) on rank 0. data_time ~0.001 s throughout — loader is not the bottleneck. Trajectory swings 1130 ↔ 1500 with occasional bursts to 1700–1850 and dips to ~1130 (per-50-iter dump in repo log `/mnt/weka/adovlatyan/logs/fsdp2-bs128-reshF-sel-47554.out`).

#### Comparison against the matrix cells we already have

| Config | Node | iters | img/s | MFU | CV |
|---|---|---|---|---|---|
| 45366 bs=128 reshT sel AC | gpu07 | 300 | 1276 | ~7.30 % | — |
| **47554 bs=128 reshF sel AC** | **gpu05** | 1000 | **1341** [500,999] | **7.67 %** | **10.6 %** |
| 46030 bs=128 reshT sel AC + LL128 (Stage C C2) | gpu07 | 1000 | 1194 [500,999] | 6.86 % | 8.4 % |
| 45369 bs=96 reshF sel AC | gpu03 | 300 | 1305 | 7.47 % | — |
| 45370 bs=96 reshF full AC | gpu05 | 300 | 1326 | 7.59 % | — |
| 45368 bs=96 reshF no AC (worst cell) | gpu07 | 300 | 1143 | 6.54 % | — |
| A2 bs=96 reshT sel + LL128 (clean Stage A) | gpu07 | 300 | 1409 | 8.06 % | 5.0 % |
| C1 bs=96 reshT sel + LL128 (Stage C) | gpu07 | 1000 | 1249 [500,999] | 7.15 % | 13.2 % |

#### Findings (Codex-tightened 2026-05-20)

1. **bs=128 reshF sel AC scored +5.1 % over an older bs=128 reshT sel AC anchor, cross-node** (1341 gpu05 vs 1276 gpu07). The missing-cell hypothesis is *consistent with* this reading but **not proven** by it — the comparison crosses both nodes and run dates, both of which we have evidence are not exchangeable. Median (1312) trails mean (1341) and the trajectory shows burst spikes to 1680/1851 img/s that inflate the mean — the median is the more defensible point. **The honest read: reshF *may* recover what reshT was costing bs=128, but this single cross-node run cannot distinguish a real reshF effect from gpu05-being-faster-than-gpu07-on-this-day.**
2. **The "47554 matches 45370 on gpu05" framing was wrong.** 45370 is bs=96 reshF *full* AC, not the production comparator (bs=96 reshT sel AC = 1351 on gpu07). +1.1 % over the wrong reference. The right same-node anchor (bs=96 reshT sel AC on gpu05) does not exist in our matrix. So we cannot yet claim bs=128 is "not losing its compute advantage at 8-GPU" — we can only claim bs=128 reshF on gpu05 is in the same throughput band as some bs=96 cells on other nodes.
3. **Strongest internal-validity signal**: [100,299] = 1340 ≈ [500,999] = 1341. The 1000-iter extension showed no late-stage degradation, and the early 300-iter window was representative for *this* run. That's a real piece of evidence — strengthens the run's internal validity, even though it does not rescue cross-job comparisons.
4. **Within-run jitter is in the band but not benign.** CV ~10.6 % at [500,999] matches bs=96 1000-iter Stage C runs (C1 13.2 %, C2 8.4 %). But the median–mean gap (1312 vs 1341) and the iter-300 / iter-700 bursts say a chunk of the mean comes from spikes. Any verdict should quote both median and mean.
5. **Memory is yellow, not green.** 18.6 GiB rank-0 peak in this 1000-iter recipe is reassuring but does not answer the prior real-run bs=128 FSDP2 OOM (researcher report 2026-05-12). Critical unknown: **was that OOM under reshT or reshF?** reshF holds params resident — likely higher long-horizon memory than reshT, so the OOM history matters more here, not less. Until that's resolved, the memory budget is "passed one short profile," not "production-safe."
6. **The "park bs=128" verdict from Stage C is reopened, not revised to "viable candidate."** Stage C's "no 8-GPU upside at bs=128" was correct *for reshT* and did not test the cell that mattered. The right new state for bs=128 is **"reopen investigation, needs paired same-node validation + long-soak memory check before any production promotion."**

#### Caveats (Codex-tightened 2026-05-20)

- **Single run, cross-node, cross-date read.** 47554 is one 1000-iter run on gpu05. The bs=128 reshT comparison (45366) is 300-iter on gpu07. Stage A → Stage C taught us gpu07 alone has ~20 % inter-epoch variance; cross-node variance is on top of that. A single +5.1 % delta is not enough to call the reshF effect real on its own.
- **Mean is partly spike-inflated.** Median [500,999] = 1312 vs mean 1341 — a ~30 img/s gap (~2 %). Burst iters (300=1851, 700=1680) materially help the mean. Any production calculation should use the median.
- **The "matches gpu05 45370" sanity check is weak.** 45370 is bs=96 reshF *full* AC (1326 img/s, 300-iter), not the production reference (bs=96 reshT sel AC = 1351 on gpu07). It establishes that 47554's number is plausible for this node and 1000-iter regime, but it does not establish that bs=128 reshF beats bs=96 in production conditions.
- **Memory is provisionally passed, not green.** 18.6 GiB rank-0 peak in this 1000-iter profile is well within budget on paper, but: (a) reshF likely uses *more* memory than reshT, (b) the original bs=128 real-run OOM (researcher 2026-05-12) was not characterized as reshT or reshF, and (c) eval + sharded-checkpoint cycles were not exercised in 47554 (`period=100000`). The memory question is **open** until either the OOM history is resolved or a longer real-eval-cycle soak passes.
- **Step time per iter is higher (~744 ms vs ~600 ms at bs=96 reshT)**; per-image throughput is better but step walltime is less predictable. Flag for the researcher if step-time predictability matters more than throughput.

#### Next moves (Codex-prioritized 2026-05-20)

The original decision rule was "≥ 1330 → queue +LL128 follow-up." We hit it but Codex's pushback is correct that stacking is the wrong next step:
- LL128 already hurt bs=128 reshT (46030 C2 = 1194 vs 45366 = 1276) — −6.4 %. So LL128 on bs=128 is *not* a known-good knob; stacking it on reshF could regress further.
- The reshF effect itself is not yet validated. Layering a second variable on top of an unvalidated one is the wrong order of operations.

**Highest-value next experiment**: **paired same-node A/B of bs=128 reshT sel AC vs bs=128 reshF sel AC** on whichever H100 node frees up first. Both 1000 iters, sequential on the same node, eval+ckpt off, no NCCL knobs. Report median + mean + CV for both windows.
- If reshF beats reshT cleanly (median ≥ 3–5 %, both windows) → reshF effect is validated; *then* test LL128 stack.
- If within ~2 % → 47554's +5.1 % was cross-node variance; lock in "bs=128 parked, reshF doesn't help."
- If reshF loses → bs=128 stays parked, full coverage achieved.

**Second priority**: resolve the bs=128 OOM history. Was the researcher's 2026-05-12 OOM under reshT or reshF? A direct ask of the researcher is cheap and gates any bs=128 production talk.

**Lower priority** (deliberately deferred): LL128 stacking on bs=128 reshF, longer-than-1000-iter runs, gradient accumulation / joint forward / iBOT max-autotune as bigger MFU levers.

**What we resist**: turning this into a sweep. One paired same-node A/B is one job pair, not a new matrix.

---

### 2026-05-18 — NCCL knob sweep Stage C completed (jobs 46029–46031)

All 3 runs completed on gpu07, sequential chain. Same `--job-name=ncclC-<variant>` resolution trick. Eval+ckpt fully OFF (`period=100000`). Script: `scripts/fsdp2_ncclsweep_stageC.sh`.

| Run | Config | iters | log |
|---|---|---|---|
| C1 | bs=96  sel + LL128                       | 1000 | 46029 |
| C2 | bs=128 sel + LL128                       | 1000 | 46030 |
| C3 | bs=96  sel + LL128 + AVOID_RECORD_STREAMS | 300  | 46031 |

#### Headline numbers — every window we can cut

img/s mean (median in parens), step_ms mean, MFU dense mean, within-window CV.

| Window | C1 (bs=96 LL128) | C2 (bs=128 LL128) | C3 (bs=96 LL128 NORS) |
|---|---|---|---|
| [100,299] | **1124** (1057) · 672 ms · 6.43 % · CV 18.0 % | **1192** (1166) · 849 ms · 6.82 % · CV 12.3 % | **1319** (1271) · 563 ms · 7.55 % · CV 13.9 % |
| [200,499] | 1202 (1188) · 628 ms · 6.88 % · CV 12.2 % | 1185 (1180) · 847 ms · 6.78 % · CV 8.1 % | — |
| [500,999] | 1249 (1216) · 608 ms · 7.15 % · CV 13.2 % | 1199 (1179) · 836 ms · 6.86 % · CV 8.4 % | — |
| [800,999] | 1228 (1220) · 616 ms · 7.03 % · CV  8.9 % | 1194 (1177) · 849 ms · 6.83 % · CV 8.6 % | — |
| [200,999] | 1232 (1205) · 616 ms · 7.05 % · CV 12.9 % | 1194 (1180) · 840 ms · 6.83 % · CV 8.3 % | — |

#### Codex review (2026-05-18) — corrections applied below

This writeup originally framed C1's 20 % shortfall as a "noisy gpu07 epoch" and treated within-stage comparisons as internally valid because they shared that epoch. Codex pushed back hard on both moves. The corrections are:

1. **Cluster-noise was put first; should have been last.** C1's trajectory is internally **structured** (bimodal between ~1100 and ~1250 across the full 1000 iters, see per-50-iter dump below). Structured time series are evidence of **in-process effects** (Python GC desync, allocator fragmentation, FSDP/NCCL buffer state drift, compile-cache churn) before they are evidence of external background load. Authoring the cluster-blame narrative first inverts the right diagnostic order.
2. **Run-order is a first-class confound, not a footnote.** C2 launched after C1 (1000 iters of allocator/cache state on the node), C3 launched after C1+C2. Stage A runs were short, sequential cold-ish starts. Stage C runs are not exchangeable with Stage A runs as "same node, same config." Saying "shared noise epoch makes within-stage comparisons valid" is too strong — shared epoch does not imply shared nuisance variables when runs are sequential and stateful.
3. **C3 vs B1 is not a contradiction; it is a control failure.** Both nominally `bs=96 sel + LL128 + NORS`. B1 ran after Stage A's short pattern; C3 ran after two 1000-iter long runs. Different initial allocator / CUDA-cache / thermal state. The honest verdict on NORS×LL128 is **untested under controlled start state**, not "in the noise, drop on principle."
4. **The 1100–1400 img/s "production band" is a hedge, not a measurement.** It blends Stage A's quiet 1409 with Stage C's structured-degraded 1200–1250. Two regimes glued into one band do not make a production claim — they admit we do not yet know which regime is representative. Codex's two-framing alternative is below.
5. **bs=128 stability is being flattened to a single throughput number.** C2's within-window CV is **8 %** versus C1's **13 %** at the comparable steady-state window. If `bs=128` is genuinely more comm-hidden / compute-dominant (lower variance, slightly lower mean), that is **diagnostic** about where the noise is coming from, not just a go/no-go datum. The "park bs=128" call survives the pushback (mean isn't better, comparison is confounded), but the framing should preserve the variance asymmetry as a clue.

#### The big surprise — C1 fails to reproduce Stage A A2 (same config, same node)

Stage A A2 was `bs=96 sel + LL128` on gpu07 at 300 iters — same config as C1. Direct same-window comparison:

| Window | A2 (Stage A, 45983) | C1 (Stage C, 46029) | Δ |
|---|---|---|---|
| [100,299] | 1409 img/s · 533 ms · MFU 8.06 % · CV **5.0 %** | 1124 img/s · 672 ms · MFU 6.43 % · CV **18.0 %** | **−20.2 % img/s, ~3.6× jitter** |
| [200,299] | 1415 img/s · 535 ms · MFU 8.10 % · CV 4.2 % | 1084 img/s · 692 ms · MFU 6.20 % · CV 7.6 % | **−23.4 % img/s** |

Identical script in all relevant respects (same node, same batch, same sel-AC, same LL128, same 8 GPUs). The two things that differ — period flags (don't fire in 300 iters anyway) and run length — cannot explain a 20 % throughput delta. This is **node-state variance between Stage A's runtime (earlier May 18) and Stage C's runtime (May 18 17:53+)**. Likely causes: background tenant on gpu07, Weka contention from another job, or GPU clock state. We cannot resolve which from existing data — but the inference is unambiguous: **Stage C ran in a slower / noisier epoch than Stage A.**

The Stage A baseline-replicate control (A6, end-of-stage) was inside Stage A's own time window and only bounds intra-stage drift — it does not bound **inter-stage** drift.

Consequence: the C1 1000-iter run does **not** serve as a clean production anchor for the A2 +12 % LL128 claim. The A2 number remains the best reading we have for `bs=96 sel + LL128` under quiet conditions; C1 is what the same config looks like under contention.

#### What *can* be read out of Stage C (within-stage comparisons are internally consistent)

All three Stage C runs share the same noise epoch, so relative comparisons inside Stage C are clean even though the absolute scale is depressed.

1. **bs=128 vs bs=96 with LL128, in a long run (C2 vs C1)** — the original puzzle. Late steady-state [800,999]:
   - C1 bs=96 : 1228 img/s, MFU 7.03 %
   - C2 bs=128: 1194 img/s, MFU 6.83 %
   - **bs=128 is ~−2.8 % img/s vs bs=96** in sustained 8-GPU operation, even with LL128. This **confirms** the Stage B B3 finding that bs=128 + LL128 does not recover the bs=128 deficit, and rules out the "B3 was a 300-iter mixed-regime artifact" possibility. C2 [200,999] is essentially flat: 1194 img/s ±8 %.
   - The B3 early-fast / late-slow split is now visible as run-to-run jitter at this batch size, not a regime change tied to iter count — the 1000-iter C2 trajectory swings between ~1080 and ~1300 img/s throughout (mean 1194, see per-50-iter dump). The "early" 1596 in B3 was likely a transient cold-cache or warm-up artifact, not steady-state.

2. **NORS+LL128 vs LL128 alone (C3 vs C1, 300-iter same window)** — B1 replicate.
   - C1 [100,299]: 1124 img/s
   - C3 [100,299]: 1319 img/s (**+17.3 % vs C1**)
   - C3 [200,299]: 1268 img/s; C1 [200,299]: 1084 img/s (**+17.0 %**)
   - **C3 does NOT reproduce B1's stacked-regression finding.** In the Stage C noise epoch, NORS+LL128 *beats* LL128 alone by ~17 %. This is the opposite sign of Stage B's B1 result. We now have two contradictory readings on this stacking question (B1 says regression, C3 says win). The honest read: **NORS×LL128 interaction is within the run-to-run variance band on this cluster and we cannot call it.** That said, C3 also does not reach Stage A A2's 1409 — neither the "win" nor the "regression" story holds up cleanly across epochs.

3. **Jitter is the dominant noise floor.** Within-window CV at bs=96 / bs=128 ranges 8–18 % across all Stage C cuts. Stage A's A2 had CV 5 % at the same window. Whatever changed between Stage A and Stage C tripled the per-iter variance. This is the **single largest threat to any Phase 5 conclusion** — Stage A's clean 5 % CV was probably the exception, not the rule, on this cluster.

#### Findings (Codex-tightened 2026-05-18)

1. **Stage A A2's +12 % LL128 win did not reproduce in Stage C** at the 1000-iter scale on gpu07. **The cause is unresolved.** Three candidate explanations remain live: (a) cluster background load differed between epochs, (b) long-run in-process effects (GC, allocator, FSDP/NCCL buffer state) that 300-iter Stage A never exposed, (c) cold-vs-hot start state from prior runs sharing the node. Until controlled replicates land, this is not a "noise" story — it is an open regime-shift question.
2. **bs=128 has no demonstrated 8-GPU mean-throughput upside vs bs=96**, even with LL128, in either Stage B (300 iter) or Stage C (1000 iter). Park bs=128 for production but note for the record that C2's CV (~8 %) is materially lower than C1's (~13 %) — possibly indicating bs=128 is more comm-hidden / compute-dominant under stress. Worth revisiting if the noise-floor work uncovers a comm-overlap mechanism.
3. **NORS×LL128 sign is undetermined.** B1 (300 iter, post-Stage-A) showed regression; C3 (300 iter, post-C1+C2) showed +17 %. Different start states. **Do not draw a directional conclusion** from current data. Operational call: default to no NORS (one fewer knob to defend), but this is "parked-unknown," not "drop-on-evidence."
4. **No production throughput claim can be made yet.** The Stage A 1409 number is a best-case ceiling under unidentified favorable conditions; the Stage C 1100–1250 numbers are lower under unidentified unfavorable conditions. Reporting either as the production figure, or splitting the difference into a band, is unsupported.
5. **gpu07 exhibits much higher run-to-run variance than initially assumed.** Single-shot ≤3 % effects on this cluster are not yet distinguishable from the inter-run / inter-stage variance the runs surfaced — but the noise floor itself is not yet calibrated. Avoid generalizing this into a hard threshold until 3× replicates land.

#### Open questions and next experiments (Codex-prioritized 2026-05-18)

The central unresolved question is whether **the Stage A A2 +12 % LL128 gain reproduces under controlled replication**. Until that is answered, every downstream production claim is shaky.

**Highest-value next experiment (Codex's #1)**: `3× A0 / 3× A2` replicate sweep in **one controlled hour**, same node, same launch method, same run length (300 iters is enough), same warmup discard. Add one variant: a deliberate process restart between two of the A2 runs to test cold-vs-hot start effects.
- If quiet-window A2 replicates cluster near 1400 → A2 is real, Stage C was contaminated by *something* external/long-run. Then we investigate what.
- If they cluster around 1200–1250 → A2 was a lucky outlier; the conservative number is the truth.

**Second priority**: per-rank `step_time_ms` breakdown from the C1 log (or a re-run with rank-level logging enabled). If C1's structured 1100↔1250 oscillation is one lagging rank, the search collapses immediately.

**Third priority**: `gc.disable()` + manual `gc.collect()` every ~100 iters as a cheap falsification of the GC-desync hypothesis. The repo's own `learnings/` and CLAUDE.md already flag this pattern. Run on a 500-iter bs=96 sel + LL128 and see if CV drops from ~13 % toward Stage A's ~5 %.

**Lower priority (do not do first)**:
- Node-state snapshot probe (`nvidia-smi -q`, `vmstat`, `iostat`, `lsof`) — useful only after we know whether the regime shift is in-process or external.
- FSDP2 bucket/group-size sweep — premature until the LL128 baseline is stable.

**Specific in-process hypotheses to falsify** (raised by C1's structured trajectory, not by the cluster-noise story):
- Python GC desync across ranks creating periodic per-iter stragglers.
- CUDA caching allocator fragmentation building up over 1000 iters (C1's `current_reserved_mb` did rise — 15526 at iter 999; fragmentation_ratio stayed low at 0.028, but reserved-vs-allocated gap is large).
- FSDP2 all-gather/reduce-scatter buffer state drift over a long run.
- `torch.compile` recapture or guard-failure events firing periodically.

#### Updated production verdict (Codex-tightened 2026-05-18, replaces Stage B verdict)

The earlier draft tried to ship two truths at once (best-case 1409, noisy-epoch 1100–1250) inside one production "band." Codex called this analytically weak. The corrected verdict is:

- **`NCCL_PROTO=LL128` remains promising but is not yet revalidated under controlled replication.** Stage A A2 (1409 img/s, MFU 8.1 % dense) is the only clean reading. Stage C did not reproduce it; the cause (cluster background vs in-process long-run effect vs cold/hot start) is unresolved. **Do not yet ship LL128 as the new default throughput baseline** without the replicate calibration described below.
- **NORS×LL128 interaction is unresolved.** B1 said regression, C3 said win, neither was controlled for start state. Default to the simpler config (no NORS), but treat this as parked-unknown rather than a settled drop-on-principle.
- **bs=128 is not promoted.** C2's mean is not better than C1's (and not better than the prior bs=128 matrix), and the prior B3 mixed-regime concern is consistent with C2's flat-but-still-jittery 1000-iter behavior. **Note for the record**: C2's within-window CV (~8 %) is materially lower than C1's (~13 %) at comparable windows. This may indicate bs=128 is more comm-hidden / compute-dominant under stress — diagnostically interesting, not yet a production case.
- **Single-shot ≤3 % deltas on this cluster are not yet reliably distinguishable from run-to-run noise**, but the noise floor itself is not yet calibrated. Do not generalize this into a hard threshold until replicates land.
- **Reframe the cluster question.** Stage C exposed a **regime-shift problem** that needs to be separated into (a) node-state effects, (b) in-process long-run effects (GC, allocator, FSDP buffer drift), and (c) cold-vs-hot start effects, before any firm production throughput claim is made.

---

### 2026-05-18 — NCCL knob sweep Stage B completed (jobs 45991–45993)

All 3 runs completed on gpu07, sequential chain, 300 iters each. Stage B validated the two Stage A winners along three axes.

#### Results (current-iter values, iter 100–299, n=21)

| Run | Config | MFU | img/s ±σ | step_ms | img/s CV | peak | Δ vs A0/A6 baseline (1257.3) |
|---|---|---|---|---|---|---|---|
| **A2 (ref)** | bs=96 sel + LL128 | 8.06 % | 1408.9 ±68.9 | 533.5 | 4.89 % | 14,307 MB | **+12.05 %** |
| **A5 (ref)** | bs=96 sel + AVOID_RECORD_STREAMS | 7.61 % | 1329.5 ±118.6 | 555.3 | 8.92 % | 14,307 MB | **+5.74 %** |
| **B1** | bs=96 sel + LL128 + AVOID_RECORD_STREAMS | 6.77 % | **1183.1 ±252.5** | 635.8 | **21.34 %** | 14,307 MB | **−5.90 %** |
| **B2** | bs=96 full + LL128 | 7.76 % | 1355.1 ±137.1 | 559.2 | 10.12 % | 8,906 MB | (vs A2: −3.82 %) |
| **B3** | bs=128 sel + LL128 | 7.98 % | 1394.5 ±324.2 | 742.2 | **23.25 %** | 18,895 MB | (vs A2: −1.02 %) |

Memory is exactly the prior-matrix value for each AC×batch combination — confirms NCCL knobs don't move memory. The CV column is critical: B1 and B3 are 4–5× noisier than A2.

#### Three findings (one operational, one suggestive, one untrustworthy)

The wording below is **tightened per Codex's 2026-05-18 Stage B review**: my initial writeup over-claimed mechanism (B1), attribution strength (B2), and recovery magnitude (B3). The corrected reads:

**1. B1 — stacking LL128 + AVOID_RECORD_STREAMS produces a probable regression.** *Not* a proven interaction from a single run.

- Combined: 1183 img/s (−5.90 % vs baseline, −16 % vs LL128 alone). If additive: should be +17.8 %.
- The regression is *not* random scatter — splitting B1's window: iter 100–199 = 1288 img/s, iter 200–299 = **1088 img/s with CV 4.73 %** (low variance late-window worsening = drift, not noise).
- Codex caution: enough evidence to **reject the stacked config operationally**, *not* enough to publish a clean mechanistic story without a replicate. My earlier "stream-bookkeeping interaction" attribution is demoted to working hypothesis. **Operational decision: ship LL128 alone. Drop AVOID_RECORD_STREAMS from production defaults.**

**2. B2 — LL128 helps full AC about the same as sel AC. Suggestive of latency reduction, not proven.**

- A2 (sel + LL128): 1408.9 img/s
- B2 (full + LL128): 1355.1 img/s (−3.82 % relative)
- Prior matrix at no-LL128: bs=96 full = 1317 vs sel = 1351 → full was −2.5 % vs sel
- The sel-vs-full gap is roughly preserved (−3.8 % vs prior −2.5 %, within noise).

**Codex correction**: my "latency not overlap, clean mechanistic attribution" framing overstates what one comparison can prove. Tightened wording:
- *Evidence favors LL128 reducing exposed comm cost in both AC modes similarly.*
- *No evidence here that LL128 preferentially improves overlap under fuller recompute.*

That's enough to call **"latency reduction" the working hypothesis** but not to declare it proven from this single comparison.

**3. B3 — bs=128 + LL128 cannot be characterized from a 300-iter run. The 100–299 aggregate is misleading.**

Codex flagged that B3's 100–299 window mixes two regimes:

| Window | img/s | step_ms | CV |
|---|---|---|---|
| iter 100–199 (early, unstable) | **1596** | 643 | 24.71 % |
| iter 200–299 (late, low-variance) | **1211** | 833 | 4.53 % |
| iter 100–299 (mixed) | 1394.5 | 742 | 23.83 % |

The late window is the genuine steady-state — and it's **slower than the early window, with much lower CV**. The 100–299 aggregate is contaminated by the fast early regime.

Applying late-window numbers to the bs=128 question:

- B3 late window: 1211 img/s
- A2 late window (bs=96 sel + LL128): 1414.8 img/s (Codex re-checked earlier)
- bs=128 vs bs=96 *late window* = **−14.4 %** — i.e., **bs=128 is *worse* than no-LL128 prior-matrix bs=128** (1276 → 1211), once you look at the stabilized window.

My earlier "−5.6 % → −1.02 % partial recovery" claim was an artifact of the fast-early regime. **The actual signal in B3's stable window is that bs=128 + LL128 is not yet a win; it may even regress.** Codex's verdict: not stable enough to support any bs=128 decision. **Need a 1000-iter run to characterize bs=128 + LL128 properly before making any production call.**

#### Production verdict (Codex-tightened 2026-05-18)

- **Ship `NCCL_PROTO=LL128`** at bs=96 sel reshT — this is the only clean positive result Codex endorsed from the entire NCCL sweep.
- **Drop `TORCH_NCCL_AVOID_RECORD_STREAMS=1`** from production defaults — Stage A A5 +5.74 % win does not survive composition with LL128 and is a regression risk.
- New default throughput at bs=96 sel reshT + LL128: **~1409 img/s** (Stage A 300-iter window), corresponding to **~8.1 % MFU dense** (~10.2 % MAMF). For longer-run numbers, a 1000-iter follow-up is needed to anchor production claims.
- **Park bs=128 pending a longer characterization run.** The Stage B B3 result is not stable enough to support any bs=128 production decision (late-window steady-state showed degradation).
- Further gain candidates remain in the FSDP2 prefetch/scheduling space (working hypothesis: LL128 reduced exposed latency, leaving overlap budget unexploited) — but this is now hypothesis-driven and would need its own experiment to confirm.

#### Codex-flagged overinterpretations in this round (logged)

- **Biggest**: I framed B3 as a "+4.6 pp partial recovery" of the bs=128 deficit. Wrong. The 100–299 aggregate mixed a fast early regime with a slow late regime. In the stabilized late window, bs=128 + LL128 is actually slower than the prior no-LL128 bs=128 baseline. The "partial recovery" framing is retracted.
- The B1 "stream-bookkeeping interaction" mechanism was overclaimed from a single run — demoted to working hypothesis. The operational decision (drop the stacked config) is unchanged.
- The B2 "clean mechanistic attribution" framing was too strong from one comparison — softened to "evidence favors latency-reduction in both modes, not overlap improvement".

#### Stage C candidates (not committed)

1. **Longer bs=128 + LL128 (1000 iters)**: highest priority. The B3 300-iter run is not steady-state. Without this we cannot park bs=128 with confidence either.
2. **Longer bs=96 sel + LL128 (1000 iters)**: anchor the production throughput claim against a longer run, replacing the 1351 prior-matrix number with a LL128-enabled equivalent.
3. **B1 replicate**: cheap (one run) to confirm the stacked regression is real before claiming any mechanism.
4. **FSDP2 bucket/group-size sweep** at bs=96 sel + LL128 — Codex flagged this earlier as potentially higher-yield than remaining NCCL knobs.
5. **Per-rank straggler analysis** from existing logs at bs=128 to see if the bs=128 instability is one rank or systemic.
6. **Alternative NCCL protocols** at bs=96 sel: try `LL`, `SIMPLE` for context against LL128.

### 2026-05-18 — NCCL knob sweep Stage A completed (jobs 45981–45987)

All 7 runs completed on gpu07, sequential dependency chain, 300 iters each, `bs=96 + sel AC + reshT`, single knob per run (no stacking), eval+ckpt off. Submission via `--job-name=ncclA-<variant>` after the portal sbatch wrapper rejected `--export=*` with a `user_env_retrieval_failed` hold (BCM portal injects `--get-user-env` on any `--export`).

#### Results table (current-iter values, iter 100–299, n=21 samples per cell)

| Variant | Knob | MFU mean ±σ | img/s mean ±σ | step_ms mean ±σ | Δ vs baseline (img/s) | verdict |
|---|---|---|---|---|---|---|
| **A0** | baseline (defaults) | 7.195 % ±0.77 | 1257.3 ±135.2 | 581.6 ±72.4 | — | reference (start) |
| A1 | `NCCL_NVLS_ENABLE=1` | 6.568 % ±1.16 | 1147.8 ±203.1 | 653.6 ±109.5 | **−8.71 %** | regression |
| **A2** | `NCCL_PROTO=LL128` | **8.062 % ±0.39** | **1408.9 ±68.9** | **533.5 ±26.4** | **+12.05 %** | **WIN** |
| A3 | `NCCL_NTHREADS=256` | 6.476 % ±1.20 | 1131.6 ±209.0 | 675.5 ±99.4 | **−10.00 %** | regression |
| A4 | `NCCL_BUFFSIZE=16777216` | 5.715 % ±0.73 | 998.7 ±127.7 | 754.9 ±104.1 | **−20.57 %** | regression |
| **A5** | `TORCH_NCCL_AVOID_RECORD_STREAMS=1` | 7.608 % ±0.68 | 1329.5 ±118.6 | 555.3 ±62.7 | **+5.74 %** | **WIN** |
| **A6** | baseline replicate (drift control) | 7.195 % ±1.07 | 1257.4 ±186.2 | 599.4 ±70.6 | +0.01 % | reference (end) |

Peak allocation is **14,307 MB** across all 7 variants — confirms NCCL knobs only affect comm timing, not memory.

#### Drift control: clean

- img/s drift A0 → A6 = **+0.01 %** (essentially zero)
- step_ms drift = +3.07 % (within the within-run σ envelope)

The gpu07 thermal/cache state is stable across the ~50-min chain on the canonical metric (img/s). Comparisons against the A0–A6 baseline mean are valid.

#### The two winners (mechanism stories tightened per Codex review 2026-05-18)

1. **A2 — `NCCL_PROTO=LL128`: +12.05 % img/s** at iter 100–299. **Codex re-checked stricter windows**: iter 150–299 gives **+14.3 %**, iter 200–299 gives **+18.4 %**. The 100–299 window is conservative — the steady-state effect is bigger. CV drops from 10.8 % (A0) → 4.9 % (A2) on img/s, and 12.4 % → 4.9 % on step_ms — **the knob halves tailiness/jitter**, not just shifts the mean. **Strong signal, well above the 2–3 % threshold.**

   **Mechanism (tightened)**: NCCL already enables LL128 by default when supported. What this run proved is **not "LL128 is good in general"** but **"forcing protocol selection to LL128 beats NCCL's autotuned mixed-protocol choice for this workload"**. The "FSDP2-per-block-collective" framing remains a plausible inference for *why* (latency-sensitive medium collectives), but not proven from this data alone.

2. **A5 — `TORCH_NCCL_AVOID_RECORD_STREAMS=1`: +5.74 % img/s** (iter 100–299; less clean across windows than A2). Clean win, above threshold.

   **Mechanism (Codex correction)**: the earlier "per-launch overhead" framing was too specific. Safer story: **reduces stream/lifetime sync bookkeeping or allocator interaction around NCCL work**. The exact attribution would need profiling.

#### The three regressions — useful info (stories tightened per Codex)

3. **A1 — `NCCL_NVLS_ENABLE=1`: −8.71 % img/s**. Tightened story: **this workload doesn't land in NVLS's sweet spot, or forcing NVLS interferes with NCCL's normal algorithm choice**. Don't enable explicitly.

4. **A3 — `NCCL_NTHREADS=256`: −10.00 % img/s**. **Codex caught my mechanism error**: NCCL docs say `NCCL_NTHREADS` controls CUDA threads per comm block, with **newer GPUs defaulting to 512**. So setting to 256 is *reducing* thread count — not over-provisioning. Correct story: **reducing NCCL comm kernel thread count hurt throughput**. Don't reduce below default.

5. **A4 — `NCCL_BUFFSIZE=16777216` (16 MiB): −20.57 % img/s**. Tightened story: **larger per-pair buffers worsened this workload's overlap/latency tradeoff**. The specific "kills pipelining" mechanism remains an inference.

#### Methodology notes

- Within-run σ is ~12 % on baseline step_ms (much higher than the 1–2 pp seen in the 1000-iter matrix runs at iter 400–999). This is because the 300-iter runs use iter 100–299 — barely past compile warmup, so the run hasn't fully stabilized. **Codex confirmed re-windowing**: A2 strengthens at iter 150–299 (+14.3 %) and 200–299 (+18.4 %); A5 stays positive in all windows but less clean. A2 is robust; A5 should be validated.
- **CV (not raw σ) is the right tailiness metric**: A0 → A2 img/s CV drops 10.8 % → 4.9 %. Not a trivial "higher mean → lower σ" artifact. Genuine jitter reduction.
- The fact that two knobs improve and three hurt — at the same single-knob single-run cost — confirms there's real signal here, not random noise across the matrix.

#### Codex flagged overinterpretations (2026-05-18 review)

- **Biggest**: the A5 mechanism story ("per-launch overhead") is too specific. Demoted to "reduces stream/lifetime sync bookkeeping or allocator interaction".
- **Smaller**: "LL128 is the right protocol because of FSDP2-per-block" — the data supports "force LL128 for *this workload*", not a broader architectural claim.
- **My A3 mechanism was wrong**: I said "over-provisioning" but NCCL default is 512 on newer GPUs; setting to 256 is a *reduction*. Caught and fixed.

#### Stage A → Stage B plan (Codex-confirmed 2026-05-18)

- **A2 LL128 is the lead signal** — proceed to validate. Robust across windows, halves jitter.
- **A5 AVOID_RECORD_STREAMS is promising but secondary** — less clean across windows.

**Stage B runs** (in priority order):

| Run | Cell | Question |
|---|---|---|
| **B1** | `bs=96 + sel AC + reshT + LL128 + AVOID_RECORD_STREAMS` | Stacking — do A2 and A5 compose? |
| **B2** | `bs=96 + full AC + reshT + LL128` | Does the gain hold (or grow) under more overlap surface? Differential = latency vs scheduling attribution. |
| **B3** | `bs=128 + sel AC + reshT + LL128` | Upside probe — does the lost +6.7 % local compute win re-emerge once comms are cheaper? |

**B4 (bs=128 + sel + LL128 + AVOID_RECORD_STREAMS) dropped** per Codex — not worth running unless B3 is clearly stable and worthwhile.

Optional pre-B1 rigor (Codex suggested, not mandatory): one more short A2 replicate on the same node to bank confidence cheaply.

### 2026-05-16 — single-GPU bs=96 vs bs=128 disambiguator completed (jobs 45476, 45477)

The Codex-recommended decisive experiment (filed 2026-05-15) is in. Both ran on **gpu07**, sequentially via `--dependency=afterany`, 300 iters each, selective AC, same code path as the 8-GPU runs (FSDP2 with 1 rank). Scripts: `scripts/fsdp2_bs{96,128}_singlegpu_sel.sh`.

#### Results (current-iter values, iter 150–299)

| Cell | Node | MFU | img/s | step_ms | peak alloc | within-run σ (MFU) |
|---|---|---|---|---|---|---|
| bs=96  1-GPU sel | gpu07 | 20.4935 % | 447.64 | 214.21 | 16,072 MB | ±0.049 pp |
| bs=128 1-GPU sel | gpu07 | **21.8813 %** | **477.96** | 267.57 | 20,662 MB | ±0.071 pp |

Per-image: **bs=128 is +6.7 % faster than bs=96 on a single GPU**. Step-time ratio 267.6/214.2 = **1.249** vs batch ratio 128/96 = **1.333** — step time scales **sub-linearly** with batch, the textbook "bigger batch → better Tensor-Core tile fill" behavior. Within-run σ on MFU dropped from ±1.0–1.9 pp at 8-GPU to ±0.05–0.07 pp at 1-GPU — i.e. nearly all the noise in the 8-GPU runs came from cross-rank effects (NCCL straggler timing, GC desync, GPU-clock drift), not the compute itself.

#### Cross-scale comparison (selective AC, both columns)

| Scale | bs=96 img/s/GPU | bs=128 img/s/GPU | bs=128 vs bs=96 | per-GPU MFU |
|---|---|---|---|---|
| 1 GPU | 448 | 478 | **+6.7 %** | 20.5 % → 21.9 % |
| 8 GPU | 168.9 (1351/8) | 159.5 (1276/8) | **−5.6 %** | 7.7 % → 7.3 % |

The bs=128-vs-bs=96 throughput delta swings **+6.7 % → −5.6 %** going from 1 to 8 GPUs — a ~12 pp relative swing. The compute side prefers the bigger batch; **the swap to negative at scale is entirely a multi-GPU effect**.

#### Verdict — the bs=128 puzzle is closed

**bs=128 flatness at 8-GPU is NCCL/FSDP scheduling overhead, not local kernel/compile behavior.** Single-GPU eliminates all collective traffic, and as soon as it's gone, bs=128 produces the textbook tile-efficiency win. Therefore:

- Tensor-Core saturation hypothesis (one of three I floated yesterday): **rejected**. bs=128 has tile headroom at this model size.
- `torch.compile` shape-specialization picking a worse kernel at bs=128: **rejected**. The kernels are locally faster per image.
- NCCL/FSDP collective cost scaling unfavorably with batch: **confirmed as the dominant cause**. Either:
  - The all-gather/reduce-scatter volume scales with activation/gradient sizes (which grow with batch), and at ViT-B (~85 M params) the comms-to-compute ratio is bad enough that bigger-batch local-compute gains are erased by bigger-batch comm cost.
  - Or the FSDP2 prefetch schedule has a per-step fixed overhead that becomes a larger fraction of step time when each rank has more work to overlap against (counterintuitive but possible if prefetch windows misalign).

#### The bigger surprise: per-GPU throughput drops 62 % going from 1 GPU to 8 GPUs

At bs=96 selective AC: per-GPU img/s = **448 (1-GPU) → 169 (8-GPU)**, a drop of **62 %**. MFU mirrors this: **20.5 % → 7.7 %**. Each GPU in the 8-GPU run is doing only ~38 % of what it can do solo.

For ViT-B (~85 M params, fairly small as transformers go) this is plausible but worth flagging: at small model sizes the FSDP2 collective volume per step is large relative to compute, so scaling efficiency is poor. This is the same pattern Tim Darcet alluded to ("if both fit, DDP and FSDP are fine") — but for ViT-B specifically, FSDP2 is not "fine" at 8 GPUs vs the theoretical 100 % per-GPU scaling. The 30 % MFU lab target is achievable at single-GPU compute throughput (we're at 20–22 % and selective AC is already on); it's the multi-GPU scaling that's killing us.

#### Production implications

1. **bs=128 stays parked at 8-GPU**. The +6.7 % single-GPU win evaporates at scale. No knob in the FSDP2 × AC × batch space recovers it. `bs=96 + reshT + selective AC` remains the default.

2. **The next gains have to come from the distributed-side, not from the FSDP2/AC/batch knob space.** See revised priority order in the Codex caveats below — nsys at 8-GPU is the unambiguous next experiment. Knob sweeps before that are blind.

3. **The 30 % MFU lab target is theoretically reachable on single-GPU** (we're at 22 % with bs=128 sel AC; another 8 pp is plausible from kernel work, compile tuning, FA tuning). At 8-GPU it's far harder — would need both single-GPU improvements *and* halving comm overhead.

#### Methodology notes

- The single-GPU run is the same code path as 8-GPU (FSDP2 wrapping is invoked but the 1-rank reduce-scatter / all-gather collapse to no-ops), so the comparison is genuinely apples-to-apples for kernel-level questions. The collective code paths are exercised but produce no over-the-wire traffic.
- Within-run σ dropping by ~20× from 8-GPU to 1-GPU is itself informative — it means **most of the 8-GPU noise comes from cross-rank synchronization scatter** (NCCL straggler timing, GC desync, GPU-clock drift), not from intrinsic GPU compute variance.

#### Codex review caveats (2026-05-16)

Two independent Codex reviews returned. Numbers confirmed (with two tiny corrections: img/s = 447.64 not 448; bs=128 peak = 20,662 MB not 20,663). Two material caveats to layer on the verdict above:

1. **bs=128 single-GPU has a dataloader stall that bs=96 does not**: `data:` field at iter ≥ 150 is `0.188 ± 0.066 s` per step at bs=128 vs `0.000167 s` at bs=96. The CUDA-event `images_per_sec` field is computed from `step_time_ms` only (`train.py:668`), so the **+6.7 % compute win is real and isolates GPU step time**, but **end-to-end wall throughput at single-GPU is NOT bs=128-favored** once the dataloader stall is included. Likely cause: the single-GPU script uses `num_workers=12` (vs 20 at 8-GPU) and the same `prefetch_factor=8`, which can't keep up with bs=128 reads at one rank. This doesn't invalidate the disambiguation but tightens the claim: **compute-side prefers bs=128 (+6.7 %); the 8-GPU run is not data-bound; therefore the 8-GPU bs=128 deficit is on the distributed side**.

2. **"NCCL/FSDP scheduling" is an inference, not a proof**: world_size=1 eliminates all collectives, so the single-GPU experiment proves only that **compute scales sub-linearly with batch (textbook tile fill)**. The 8-GPU flatness could be NCCL collectives, FSDP2 per-block scheduling/prefetch overhead, rank stragglers, thermal variance at full 8-GPU power, GC desync, or some combination. Single-GPU can't separate these.

#### Priority order for the next experiment (revised 2026-05-16 after Codex + the catch that nsys is already done)

**Important correction**: nsys at 8-GPU was already run on 2026-05-08 (jobs 39140 bs=128, 39141 bs=96). Headline findings on file (see Phase 5 ledger below):
- **NCCL = 70 % (bs=128) / 80 % (bs=96) of kernel time**
- **NCCL↔compute overlap = 16.3 % (bs=128) / 11 % (bs=96)**
- Straggler rank shifts between runs (rank 7 / rank 4) — non-deterministic

So **the "where does step time go" question is already answered: comms-dominated with bad overlap.** Re-running nsys with current configs would only sharpen the picture marginally. The 62 % per-GPU throughput drop from this 1-GPU vs 8-GPU comparison is consistent with — and re-confirms — the existing nsys finding, it does not generate a new question.

The action items follow from what we already know, not from another profile.

**Revised priority order**:

1. **NCCL knob sweep — two-stage factorial** (top priority, cheap, fast, informed by existing nsys data). Design below.
2. **FSDP2 bucket/group-size sweep** — code currently wraps each transformer block (12 buckets). Codex flagged this as potentially higher-yield than NCCL envs: changes collective count AND payload size simultaneously. Try every-2-blocks / every-3-blocks at `reshT + sel AC`.
3. **Per-rank step-time histogram** from existing 8-GPU logs — cheap (5-min re-parse). If a particular rank is consistently slow → straggler-attribution problem (GC, log I/O on rank-0, etc.); if the slow rank changes — system-level (thermal, fabric).
4. **Another nsys ONLY for refined questions** — e.g. "does `reshard=False + sel AC` reduce NCCL share, or is FSDP2 prefetch already hiding the eliminated all-gathers?" or "has the straggler pattern stabilized under current configs?" These are sharpening questions, not headline questions.
5. **Gradient accumulation: deprioritized** per Codex — does NOT remove per-microbatch FSDP collectives, only saves the optimizer-step reduce (tiny).

#### NCCL sweep — two-stage design (Codex review, 2026-05-16)

**Why both AC modes, why both batch sizes** — initial design used only `bs=96 + sel AC`; Codex pushed back on both choices and the user surfaced the same pushback independently. Revised rationale:

- **`selective AC`** = primary baseline: production-default, less compute → less overlap budget → NCCL wins show in step_ms cleanly (signal-to-noise advantage).
- **`full AC`** = diagnostic contrast: more recompute → more overlap surface for NCCL. **Differential interpretation of a knob's effect**:
  - Helps sel but not full → reduced exposed comm latency (good, comm cost itself dropped)
  - Helps full more than sel → improved overlap/scheduling (different mechanism)
  - Helps neither → knob is irrelevant
- **`bs=96`** = required: tells us whether the safe production cell actually improves.
- **`bs=128`** = upside probe: bs=128 has +6.7 % local compute win that gets erased by comms at 8-GPU; if comms get cheaper, this is where the biggest wallclock win could re-emerge. **More informative cell for "best-case improvement" but not the production-safe baseline.**

**Stage A — screen 5 knobs on the most important cell** (6 runs × ~6 min = ~36 min, sequential on gpu07 to avoid contention/drift):

Cell: `bs=96, sel AC, reshT, 300 iters, eval+ckpt off`. Each knob run varies **one** env var from the baseline (no stacking — prior NVLS attempt was confounded by stacked changes):

| Run | Knob change | Hypothesis |
|---|---|---|
| A0 | baseline (NCCL defaults) | reference |
| A1 | `NCCL_NVLS_ENABLE=1` | NVSwitch-aware reduce-scatter/all-gather — explicit beats `NCCL_ALGO=NVLS` |
| A2 | `NCCL_PROTO=LL128` (vs auto) | many small/medium collectives favor LL128 |
| A3 | `NCCL_NTHREADS=256` | more threads per channel |
| A4 | `NCCL_BUFFSIZE=16777216` | bigger NCCL buffer = fewer chunks per collective |
| A5 | `TORCH_NCCL_AVOID_RECORD_STREAMS=1` | cuts stream-record overhead per launch |

Dropped from original list: `NCCL_NSOCKS_PERTHREAD` (socket-side; low-yield on single-node NVLink/NVSwitch).

**Stage B — validate top 1–2 winners on the upside + diagnostic cells** (~6 runs × ~6 min = ~36 min):

For each Stage A winner with ≥2–3 % improvement (anything below that is single-run noise):
- `bs=96, full AC, reshT` (diagnostic: overlap vs latency attribution)
- `bs=128, sel AC, reshT` (upside: does the +6.7 % local win re-emerge at 8-GPU?)

If only one winner clears the 2–3 % bar: 2 runs. If two: 4 runs. Plus a fresh baseline replicate at the end of Stage A to bound gpu07 thermal/cache drift.

**Pitfalls (Codex-flagged, methodology guardrails)**:
- Do NOT stack knobs in the screen. One change at a time.
- Do NOT use `NCCL_DEBUG=INFO` except in a tiny separate diagnostic run — it perturbs timing and bloats logs.
- Same node (gpu07), same script template, same iter window for parsing, same metric (current-iter values, iter ≥ 100).
- **Baseline at beginning AND end of Stage A** — gpu07 thermal/cache state changes, fake winners are real risk.
- Don't judge on MFU alone — use step_ms, img/s, and within-run σ. A "win" under ~2–3 % from one 300-iter run is suspect.

### 2026-05-15 — reshard × AC × batch matrix completed (jobs 45366–45370)

All 5 follow-up jobs ran 1000 iters with the forced eval+ckpt at iter 400/800. None OOM'd, no `alloc_retries`, no `num_ooms`. 45366 reran bs=128+reshT+selective on **gpu07** (was gpu01 contention previously); 45367–45370 filled in the bs=96 reshard × AC cells.

#### Full results table (steady-state, smoothed-current; "iter 400–999" is the 600-sample post-first-eval window)

| Cell | Job | Node | iter 400–999 MFU | img/s | step_ms | peak alloc | frag |
|---|---|---|---|---|---|---|---|
| bs=96  reshT noAC | 45367 | gpu07 | 7.32 % | 1279 | 600 | 24,919 MB | 0.014 |
| bs=96  reshT sel  | 45280 | gpu07 | **7.73 %** | **1351** | 562 | 14,302 MB | 0.028 |
| bs=96  reshT full | 45281 | gpu07 | 7.54 % | 1317 | 573 | 8,908 MB  | 0.042 |
| bs=96  reshF noAC | 45368 | gpu07 | **6.54 %** | **1143** | 678 | 25,089 MB | 0.006 |
| bs=96  reshF sel  | 45369 | gpu03 | 7.47 % | 1305 | 583 | 14,476 MB | 0.028 |
| bs=96  reshF full | 45370 | gpu05 | 7.59 % | 1326 | 567 | 9,072 MB  | 0.039 |
| bs=128 reshT noAC | 45282 | gpu07 | 7.25 % | 1267 | 809 | 33,034 MB | 0.008 |
| bs=128 reshT sel  | 45366 | gpu07 | 7.30 % | 1276 | 802 | 18,895 MB | 0.030 |
| bs=128 reshT full | 45364 | gpu03 | 7.42 % | 1297 | 787 | 11,720 MB | 0.039 |

(Numbers are **current-iter values** averaged over iter ≥ 100, NOT the rolling-window-20 smoothed values — the smoothed window is contaminated by warmup and eval-phase outliers. Within-run σ on MFU is ~1.0–1.9 pp across all cells — same noise envelope, so cross-cell deltas under ~1.5 pp are not significant.)

**Node-mix confound (Codex flag, 2026-05-15)**: reshT cells all ran on gpu07; reshF sel = gpu03, reshF full = gpu05, reshF noAC = gpu07. The strongest *same-node* AC ladder is the gpu07 reshT row (45367 / 45280 / 45281 / and the reshF-noAC cell 45368). Cross-node reshT-vs-reshF deltas mix node heterogeneity with the actual reshard effect. The findings below stand directionally but the causal stories (Findings 2/3/4) are hypotheses, not proofs.

#### Confirmation: 45366 disproved the gpu01 contention "win"

bs=128 + reshT + sel on **gpu07** = **7.30 % / 1276 img/s / 802 ms**. The earlier gpu01 run (45363) showed 19.55 % / 3416 img/s post-iter-400. **That was 100 % a contention artifact** — gpu07 result is fully consistent with the other bs=128 reshT cells (noAC 1267, full 1297). Killing that ghost.

#### Findings — reshard × AC interaction (bs=96)

1. **`reshard=False` without AC is the worst cell in the entire matrix**: 1143 img/s, **−11 % vs reshT noAC**, **−15 % vs reshT sel**. Counterintuitive — Tim Darcet's heuristic that "no-release is basically DDP" predicts a *win* from comms moved earlier. We see the opposite. Likely explanation: at ViT-B / bs=96 the all-gather→forward→backward path with `reshard=True` is **already comms-overlapped well** by FSDP2's prefetch, so eliminating the post-forward reshard saves nothing while losing the FSDP2 prefetch scheduling, and possibly forcing a different (less optimal) `torch.compile` plan. This is a regime-specific result — the same comparison at much larger models/batches might flip.

2. **`reshard=False` + AC closes the gap to `reshard=True` + AC** (essentially tied within noise):
   - sel: reshT 1351 vs reshF 1305 (−3.4 %, cross-node)
   - full: reshT 1317 vs reshF 1326 (+0.7 %, cross-node)
   - The "AC needs an extra all-gather under reshard=True" hypothesis from my pre-run reasoning is **not observable empirically** at this scale. The AC-time gather is fully hidden by recompute, or FSDP2's prefetch covers it. **No measurable cost.**

3. **AC is throughput-*positive*, not negative, in this codebase at ViT-B / bs=96**. This is the surprise:
   - reshT: noAC 1279 → sel 1351 (+5.6 %), full 1317 (+3.0 %)
   - reshF: noAC 1143 → sel 1305 (+14 %), full 1326 (+16 %)
   - Cause is likely allocator/memory-system not arithmetic: noAC keeps the big activation tensors live for the entire forward → backward window. AC drops them and recomputes — smaller working set → less allocator/cache pressure → better effective throughput. Fragmentation tells a consistent story (`frag` rises 0.014→0.028→0.042 going noAC → sel → full, but throughput rises too — so it's not the *reserved-pool* fragmentation that hurts, it's the *live-set* size).
   - This is the opposite of the "AC pays a few pp MFU for memory" textbook framing. The textbook framing assumes you are compute-bound on a workload where save-and-replay is cheaper than recompute. Here at H100 BF16 throughput with our actual data pipeline, **we are not compute-bound enough for that framing to hold** — saving activations is more expensive than recomputing them.

4. **Memory: reshard=False costs ~150–170 MB of resident param shards** vs reshard=True, consistent across AC modes:
   - noAC: 24,919 → 25,089 (+170 MB)
   - sel: 14,302 → 14,476 (+174 MB)
   - full: 8,908 → 9,072 (+164 MB)
   - This is the "param shards stay resident" cost. Negligible at ViT-B; would be much larger at ViT-7B.

5. **AC reduces memory orthogonally to reshard**:
   - reshT: noAC 24.9 GB → sel 14.3 GB (−43 %) → full 8.9 GB (−64 %)
   - reshF: noAC 25.1 GB → sel 14.5 GB (−42 %) → full 9.1 GB (−64 %)
   - The two knobs are independent — the AC savings come from activation memory, not from FSDP2 param scheduling.

6. **bs=128 doesn't beat bs=96 even at fixed AC mode** (reshT row, all gpu07):
   - noAC: bs=96 1279 vs bs=128 1267 (tied)
   - sel:  bs=96 1351 vs bs=128 1276 (−5.6 %)
   - full: bs=96 1317 vs bs=128 1297 (tied within noise, cross-node)
   - Throughput is **constant in batch** at this regime — step_time scales linearly with batch (562 → 802 ms at +33 % batch ≈ +43 %). The trainer is matmul-arithmetic-limited (good) but not yet on the right side of the Tensor-Core fill-curve; bigger batch buys no per-image speedup. This contradicts the usual "bigger batch = higher MFU" intuition for ViT-B at H100 BF16.

#### Analytical model (what these numbers fit)

The simplest model that explains *all six bs=96 cells*:

```
step_time(reshard, AC) ≈ T_compute(AC)  +  T_resident_cost(reshard) ± noise

where:
  T_compute(noAC) > T_compute(sel) ≈ T_compute(full)   [activation-memory-driven]
  T_resident_cost(reshF) > T_resident_cost(reshT) when no-AC   [no overlap to hide it]
  T_resident_cost(reshF) ≈ T_resident_cost(reshT) when AC      [recompute hides any gap]
```

Restated: the dominant lever at bs=96 ViT-B is **activation working-set size**, not parameter shard scheduling. AC wins because it shrinks the live set. `reshard=False` only matters when AC is off (and then it hurts).

#### Production implication

- **bs=96 + reshT + selective AC remains the throughput default** (1351 img/s) — best cell in the entire matrix.
- **bs=96 + reshF + full AC is the memory-tightest viable cell** (9.1 GB, 1326 img/s) — close to selective on throughput, smallest peak.
- **Avoid reshard=False without AC** — strictly dominated.
- **bs=128 is not throughput-positive** at this regime — no reason to fight its memory cost. Park.
- **Throughput ceiling at this point is ~1350 img/s on a clean gpu07**, ~7.7 % MFU dense — well below the 30 % lab target. The next gains have to come from elsewhere (data pipeline, kernel-level fusion, model arch, or larger global batch via more nodes), not from FSDP2 / AC knob-tuning.

#### Open follow-ups (not committed)

- **Decisive bs=128 puzzle experiment — submitted 2026-05-15 (jobs 45476, 45477)**: single-GPU bs=96 vs bs=128 + selective AC on **gpu07** (sequential via `--dependency=afterany`), 300 iters each, real Weka data, same code path as the 8-GPU runs (FSDP2 with 1 rank degenerates to no-shard but keeps the wrapping/compile path identical so the comparison is apples-to-apples with the multi-GPU rows). Eval+ckpt disabled (period=10000). Scripts: `scripts/fsdp2_bs{96,128}_singlegpu_sel.sh`.

  **What it disambiguates**: in the 8-GPU matrix, `step_time(bs=128)/step_time(bs=96) ≈ 1.348` vs batch ratio 1.333 — step time scales near-perfectly linearly in batch, so img/s is flat. Codex ruled out DataLoader (data: field sub-ms) and allocator (no `alloc_retries`/`num_ooms`). Remaining candidates split into two groups:
  - **Multi-GPU-only**: NCCL collective cost scaling with batch; FSDP2 prefetch scheduling hidden cost at larger working sets
  - **Local kernel/compile**: Tensor-Core M-dim already saturated at bs=96 (global M = 96×197 = 18,912 rows is ~100× the K-dim 768; bigger M may give no further tile-efficiency gain); `torch.compile` shape-specializing to a less-efficient kernel at bs=128 shapes; iBOT Sinkhorn iterations scaling with batch

  Single-GPU kills all collective traffic, so:
  - If **bs=128 wins img/s single-GPU** (per-image throughput improves with larger batch when no comms) → the 8-GPU flatness is **NCCL/FSDP scheduling** (multi-GPU-only group)
  - If **single-GPU is still flat** (per-image throughput equal or worse at bs=128) → it's **local kernel/compile behavior** (Tensor-Core saturation or compile shape choice), which means **bs=128 is genuinely not throughput-positive on this model and no comms tuning will fix it**

  Either outcome closes the question and tells us whether further bs=128 work is worthwhile or should be permanently parked.
- **bs=128 × reshF row** is unmeasured. Lower priority — the bs=96 result shows reshF only matters via AC, and bs=128 reshT row was already flat.
- **bs=96 ViT-L scan** would test whether the activation-memory-dominated regime holds for larger models or whether it's specific to ViT-B.
- **Step-time profiling** (nsys) on `reshF + noAC` to confirm the lost-prefetch hypothesis vs other causes (recompile, allocator stalls).

#### Methodology notes (Codex review, 2026-05-15)

- All MFU/img/s/step_ms numbers in the matrix are **current-iter values** (the unparenthesized field in `helpers.py:105` log lines), averaged over iter ≥ 100. The parenthesized rolling-window-20 smoothed values are contaminated by warmup and eval-phase outliers and should not be used for steady-state characterization.
- The findings labeled "claim" or "hypothesis" (Findings 2/3/4) are directionally supported by the data but the **causal mechanisms are not proven** from single runs at this σ. Treat as working hypotheses to test, not conclusions.

### 2026-05-15 — bs=128 × AC matrix completed (jobs 45363, 45364)

Both jobs ran to 1000 iters with two forced eval+ckpt cycles (`checkpointing.period=400`, `evaluation.eval_period_iterations=400`, `OFFICIAL_EPOCH_LENGTH=1000`). **Neither OOM'd.** Same memprofile env as the bs=96 AC pair.

#### Completed 2×3 batch × AC matrix (steady-state, iter ≥ 100; smoothed-current parsed from `helpers.py:105`)

| Cell | Job | Node | iter 100–399 | iter 400–999 | `steady_state` peak alloc | Notes |
|---|---|---|---|---|---|---|
| bs=96 + selective AC | 45280 | gpu07 | 7.34% / 1283 / 608 ms | 7.73% / 1351 / 562 ms | 14,302 MB | Clean, stable |
| bs=96 + full AC      | 45281 | gpu07 | 7.40% / 1294 / 594 ms | 7.54% / 1317 / 573 ms | 8,908 MB  | Clean, stable; **−38% mem** vs selective for **−2.5% img/s** |
| bs=128 + no AC       | 45282 | gpu07 | 7.25% / 1267 / 802 ms | 7.25% / 1267 / 809 ms | 33,034 MB | Clean, stable; throughput-tied with bs=96+AC |
| **bs=128 + selective AC** | **45363** | **gpu01** | 9.83% / 1718 / 604 ms | **19.55% / 3416 / 297 ms** | **18,895 MB** | ⚠ **gpu01 contention artifact** — 2× step change at iter 400; numbers not directly comparable |
| **bs=128 + full AC**      | **45364** | **gpu03** | 7.42% / 1296 / 784 ms | 7.42% / 1297 / 787 ms | **11,720 MB** | Clean, stable; throughput-tied with bs=96+full despite +33% batch |

(Format: `mfu% / img/s / step_ms`)

#### Three findings from the completed matrix

1. **Memory scales near-linearly in batch for each AC mode** (single-cell estimates, not a fit):
   - Selective: 14,302 MB @ bs=96 → 18,895 MB @ bs=128 → **+143 MB per +1 image** (~+32 % mem for +33 % bs).
   - Full:       8,908 MB @ bs=96 → 11,720 MB @ bs=128 → **+88 MB per +1 image** (~+32 % mem for +33 % bs).
   - No-AC peak at bs=128 (33,034 MB) is ~1.75× the selective bs=128 peak, ~2.8× the full bs=128 peak.
   - **Naive affine projection** (`peak ≈ a·bs + b`, two-point fit per AC variant): bs=192 → ≈28.1 GB (sel) / ≈17.4 GB (full); bs=256 → ≈37.2 GB (sel) / ≈22.9 GB (full). All fit on 80 GB H100. Treat as *direction*, not a guarantee — Codex's caveat (longer-horizon allocator state, eval workload realism) still applies.

2. **Full AC erases the bigger-batch throughput gain** (clean, same-node-class comparison via gpu07 ↔ gpu03):
   - bs=96 full (gpu07): 1,317 img/s. bs=128 full (gpu03): 1,297 img/s. **Throughput-tied within noise**, while memory rises 8.9 → 11.7 GB.
   - This is expected: full block-recompute makes the per-step compute scale ~linearly with batch, so img/s/step saturates. No reason to run bs=128 + full AC.

3. **bs=128 + selective AC is the one cell we can't conclude from yet** — the gpu01 run shows a hard 2× step-time discontinuity at iter 400 (604 → 297 ms, 1718 → 3416 img/s). The post-iter-400 numbers are too good to credit — they sit above every other cell in the matrix and clearly indicate **gpu01 contention was released around the first eval boundary** (someone else's job finished, or a clock/throttle event). The pre-iter-400 numbers (9.83 % / 1,718 img/s / 604 ms) are more plausible but still cross-node vs gpu07. **Memory measurement (18,895 MB) is trustworthy** — that's an allocator fact, not a timing measurement.

#### Production recommendation (matrix-informed)

- **bs=96 + selective AC remains the throughput default** (1,351 img/s steady, 14.3 GB peak — 65 GB headroom on 80 GB H100). Same recommendation Codex returned 2026-05-15; matrix did not overturn it.
- **bs=96 + full AC is the memory fallback** (1,317 img/s, 8.9 GB peak — 71 GB headroom; only 2.5 % img/s cost vs selective in same-node A/B).
- **bs=128 + full AC is dominated** (same throughput as bs=96 + full, more memory). Drop.
- **bs=128 + selective AC verdict is deferred** — needs a same-node rerun on gpu07 (or a gpu08-class baseline + paired bs=128) before promoting. If clean-node img/s matches the pre-iter-400 segment (~1,700 img/s), it would be a **+25 % throughput** win over bs=96 + selective at +32 % memory (18.9 GB peak, still ample headroom). If it matches bs=128 no-AC (1,267 img/s), no win.
- **bs=128 + no AC** stays parked: throughput-tied with bs=96+AC at 2.3× the memory and unresolved long-horizon OOM history.

#### Caveats carried forward

- All three "good" cells (45280, 45281, 45282) were on gpu07. Both new bs=128 cells (45363, 45364) landed elsewhere; the bs=128 full result is corroborated by the bs=96 full gpu07 result (≈1,310 img/s at 7.5 % MFU on both), so node skew here is small. The bs=128 selective cell is the one that needs to come back to gpu07.
- Codex's prior pushback (single-run cells, 1.4–1.6 pp within-run σ, eval-as-ckpt-save, 1000-iter horizon) still applies to every cell in the matrix.
- Fragmentation is healthy across all five cells: `alloc_retries=0`, `num_ooms=0` for the full run; `fragmentation_ratio` stable at 0.030 (sel), 0.039 (full), 0.008 (noAC).

#### Next action

Submit **bs=128 + selective AC** rerun, prefer-host `gpu07` (or accept any non-shared node and re-verify). Then revisit promotion decision. Logged as the open item under §11.

**Update (same day)**: rerun submitted as **job 45366**, pinned via `sbatch --nodelist=gpu07`, started immediately (gpu07 was idle). When it finishes, append the row to the matrix and decide on bs=128 + selective.

#### Open issue — cluster-side inter-run variance (parked for cluster-tuning work)

Throughout Phase 5 we have seen non-trivial throughput swings between runs of the same config that we cannot fully attribute to the workload:
- Job 44023 (gpu08, daytime) hit 14.67 % MFU baseline; reruns on gpu01/gpu05 (jobs 44793/44794) hit ~7 %.
- Job 45363 (gpu01) showed a clean step-time **2× discontinuity at iter 400** (604 → 297 ms), with no corresponding event in our code path — almost certainly another user's job releasing the node.
- Memory peaks on the *same* node are highly reproducible across runs; **timing peaks are not**.

Probable contributors (none verified, all candidates for a separate cluster-tuning pass):
- **Co-scheduled CPU jobs on the same GPU node** — `cpus-per-task=64` requested out of 224; the other 160 cores are scheduled to other users by Slurm (`mixed` state in `sinfo`). Their L3 / memory-controller / NUMA pressure can perturb our DataLoader + H2D copies.
- **Weka client contention** — shared 91 TB filesystem; reads from `re-id/pretraining/*` compete with whatever else is in flight. The h5 datasets are read with `cache_dataset=true` so warm reads should be from page cache, but cold reads during eval/ckpt phases are not.
- **Per-node clock/throttle state** — H100s can drop SM clocks under sustained load or thermal pressure; we have not audited `nvidia-smi -q -d CLOCK,POWER,TEMPERATURE` mid-run.
- **NCCL coll algorithm reselection** — auto-tuning can swap RING_LL ↔ TREE between epochs based on bus contention.

Implication for the matrix: **same-node A/B is the only fair comparison** for throughput claims. Memory is portable across nodes; timing is not. Going forward, pin via `--nodelist=<host>` and prefer idle nodes (check `sinfo -p research -o "%n %T %C"`).

**reshard × AC interaction matrix submitted same day** (per user request — empirical-first, then analyze): 4 new cells at bs=96. Initially queued behind 45366 on gpu07; redistributed across idle nodes after partial runs to **parallelize the wall-clock window and broaden node sampling** (rather than serializing all five runs on gpu07).

| Job | Script | Config | Node |
|---|---|---|---|
| 45367 | `fsdp2_bs96_reshT_noAC.sh` | reshard=True  + no AC | gpu07 (ran first, finished) |
| 45368 | `fsdp2_bs96_reshF_noAC.sh` | reshard=False + no AC | gpu07 (picked up after 45367) |
| 45369 | `fsdp2_bs96_reshF_sel.sh`  | reshard=False + selective AC | gpu03 (relocated via `scontrol`) |
| 45370 | `fsdp2_bs96_reshF_full.sh` | reshard=False + full AC | gpu05 (relocated via `scontrol`) |

Note on gpu04: tried to assign 45369 there first; rejected with `BadConstraints` — `scontrol show node gpu04` reports `Gres=gpu:h100:6` (only 6 GPUs), so jobs requesting `gpu:h100:8` are infeasible. The other research-partition nodes are 8-GPU.

Combined with the existing reshard=True + selective (45280) and reshard=True + full (45281, both on gpu07), this closes the 2×3 reshard × AC matrix at bs=96. **Caveats from parallel execution**:
- **Cross-node** (45369 on gpu03, 45370 on gpu05) — reasonable for memory measurement (portable across nodes) but timing must be interpreted with the node-variance disclaimer.
- **Concurrent Weka reads** — all three running jobs read the same `re-id/pretraining/*` datasets simultaneously. Weka is designed for many-client concurrent IO and the datasets are small enough that `cache_dataset=true` keeps the working set in page cache after warmup, but **cold reads in the first few iters and at eval phase boundaries may now contend** in a way they did not when runs were serial. This is a new source of variance in this batch specifically.
- The user's framing here is the correct one: cluster-wide contention is "almost like a random process" — running same-config replicates is more informative than chasing every confounding factor. The matrix completion is meant to surface *first-order* reshard × AC effects, not to nail down 0.x % differences.

Analytical interpretation deferred until all five cells have numbers.

Mitigations to evaluate later (not part of Phase 5 perf work):
- `--exclusive` flag → reserve the whole GPU node (kills neighbor CPU jobs but costs scheduling latency).
- Pin CPU set via `taskset` / Slurm `--cpu-bind` to the NUMA node closest to the GPUs we use.
- Lock GPU clocks: `nvidia-smi -lgc <max>` for the duration of a perf run (root only).
- `numactl --membind` for the trainer + workers.
- Weka per-client QoS knobs (cluster-admin level).


### 2026-05-14 — Codex pushback applied + 3 follow-up jobs submitted

After Codex (gpt-5.5, high effort) flagged 5 issues with the AC-selective decision framing:
- **(a)** Downgraded "zero MFU cost" / "60+ GB headroom" / "memprofile-v2 done" language in the §9 results entry. Replaced with one-run-paired-on-gpu01 framing, rough-bound projection (no headroom claim), and explicit notes that the bs=96+AC clean fragmentation does NOT diagnose the bs=128 OOM.
- **(b) Job 45280** — `scripts/fsdp2_bs96_ac_selective.sh` resubmitted. Provides A/B replicate of the selective-AC MFU number on same-node baseline (gpu01-class).
- **(c) Job 45281** — `scripts/fsdp2_bs96_ac_full.sh` (new). Mirror of selective but `train.checkpointing_full=true`. Provides the selective-vs-full variant comparison at the same batch size — prerequisite for choosing the AC variant at bs=128 (per user direction 2026-05-14).
- **(d) Job 45282** — `scripts/fsdp2_bs128_memprofile.sh` (new). bs=128 **without** AC, `DINOV3_MEMORY_PROFILE=1`, `DINOV3_MEMORY_PROFILE_PERIOD=10`, 1000 iters with coincident eval+ckpt at iter 400 and 800. Will either OOM (in which case the [MEMPROFILE]/[MEMFRAG] lines up to the failure point categorize the mode: working-set / rank-skew / phase-boundary materialization / fragmentation) or survive 1000 iters (in which case the OOM is longer-horizon than 1000 iters — also data).

All three jobs use the same `unset PYTORCH_CUDA_ALLOC_CONF` sanitization, same 0.05 s SLURM partition, and `train.distributed_strategy=fsdp2`. Walltime: 1 h for the bs=96 AC pair, 1.5 h for the bs=128 memprofile.

Deferred from Codex's recommendations (per user direction):
- **(e) affine memory model** (bs=64 + bs=96 → predict bs=128): nice-to-have but the bs=128 memprofile run (45282) gives the *actual* answer.
- **(f) node diagnostic** (re-baseline on gpu08 + clocks/throttle audit): logged in the plan but not on the active queue.

### 2026-05-14 — P5-VAR + P5-AC-selective results (jobs 44793, 44794, 44795)

#### Result 1: selective AC works cleanly at bs=96 (P5-AC-selective, job 44795)

**Phase-boundary memory** (rank 0, `[MEMPROFILE]` lines):

| Phase | max_alloc_mb | max_reserved_mb |
|---|---|---|
| post_init_weights | 140 | 156 |
| compile_warmup_iter0 | 14,152 | 15,286 |
| **steady_state** | **14,302** | **15,526** |
| pre_eval (iter 399) | 14,302 | 15,526 |
| eval_complete | 386 | 15,526 |
| pre_checkpoint | 386 | 15,526 |
| checkpoint_complete | 389 | 15,526 |
| (same pattern replicated at iter 799–800) | | |

**Fragmentation timeline** (`[MEMFRAG]`, sampled): `alloc_retries=0, num_ooms=0, fragmentation_ratio=0.028` held constant from iter 9 → iter 999. **Zero allocator pathology** across 1000 iters with two eval+checkpoint cycles.

**MFU**: steady mean = 7.00%, std 1.18 pp, range 5.48–11.23. **Identical to the same-node baseline (gpu01 P5-VAR1, job 44793, mean 7.39%).** Step time +29 ms (~5%) is in the noise.

**Memory reduction from selective AC at bs=96 (one run): 24,922 MB → 14,302 MB = −10,620 MB (−42.6%).** MFU 7.00% vs same-node baseline 7.39% in **one paired run only** — within-run iteration σ is 1.18–1.74 pp, but per-iteration σ is autocorrelated and does **not** prove the +29 ms / −0.39 pp mean delta is noise. **Downgrade**: "no clearly detectable MFU cost from one A/B pair on gpu01" — not "zero cost." Replicate needed before treating as a free win (Codex review, gpt-5.5, 2026-05-14, finding #1).

#### Result 2: bs=96 baseline MFU is highly node-dependent (P5-VAR, n=3)

| Job | Node | Time | Steady MFU mean | Within-run σ |
|---|---|---|---|---|
| 44023 (P5-00) | gpu08 | 2026-05-13 00:25 | **14.67%** | 5.05 pp |
| 44793 (P5-VAR1) | gpu01 | 2026-05-14 15:18 | 7.39% | 1.74 pp |
| 44794 (P5-VAR2) | gpu05 | 2026-05-14 15:25 | 7.01% | 1.14 pp |
| 44795 (P5-AC-sel) | gpu01 | 2026-05-14 15:27 | 7.00% | 1.18 pp |

**Run-to-run σ across the 3 baselines = 4.32 pp, range = 7.67 pp.** The original gpu08 number (14.67%) is an outlier vs gpu01/gpu05 (~7.0–7.4% consistently). Within-node variance is small (gpu01: 7.39 baseline vs 7.00 with AC — well within iteration-level noise); between-node variance is huge.

**Implications**:
- The "bs=96 baseline ≈ 14.67% MFU" claim from the 2026-05-13 ledger is **not representative.** The honest baseline is **≈ 7.0–7.4% steady MFU** on gpu01/gpu05.
- The aspirational 30% MFU figure is much further away than the 2026-05-13 framing implied (4× vs 2× away).
- **Any future MFU comparison must hold the GPU node constant** (or replicate across multiple nodes), otherwise inter-node variance swamps the signal.
- gpu08 was running at a fundamentally different speed — possible causes: midnight idle cluster vs 15:00 contention, thermal, NUMA microvariation, Triton/compile cache state. Worth investigating separately but not blocking AC work.

#### Result 3: bs=128 + selective AC — rough bound, NOT a headroom guarantee

A naive linear scaling `14.3 × (128/96) ≈ 19 GB` is suggestive but **not defensible as a production-safety estimate** (Codex review finding #2):
- Treats rank-0 allocated memory as fully scalable; ignores fixed-cost terms (params, optimizer states, FSDP2/NCCL working buffers, EMA teacher, gram teacher) that do *not* scale with batch.
- Ignores per-rank skew; the OOM ranks may not be rank 0.
- Reserved memory may scale differently than allocated.
- The previous worst-case profiler already failed to predict bs=128 OOM by missing eval+checkpoint pressure together — a second point projection inherits the same epistemic risk.

To replace the point estimate, an **affine model** from at least two AC batch sizes (e.g., bs=64 + AC, bs=96 + AC) and rank-max peaks would let us extrapolate `peak ≈ a·bs + b` with a meaningful confidence interval. For now, treat the 19 GB number as a "fits if scaling is roughly linear" hint, not a headroom claim. **The bs=128 + AC run is the only authoritative answer.**

#### Decisions

**Revised decisions** (incorporating user 2026-05-14 + Codex review):

- **P5-AC-full at bs=96 stays in the queue** (Codex #4 + user direction): the selective vs full comparison at the *same batch size* is the prerequisite for choosing the AC variant at bs=128. Dropping it before bs=128 retry would leave us without a same-batch reference if selective at bs=128 fits tight or OOMs.
- **P5-AC-selective replicate at bs=96 on gpu01**: an A/B/A baseline-AC-baseline sequence (or at least one more paired AC run) is required before the "no MFU cost" claim becomes a decision input (Codex #1).
- **P5-OOM-memprofile-v2 stays active** (Codex #3): the bs=96 + AC clean fragmentation result does *not* tell us why the original bs=128 OOM happened. The actual diagnostic is to reproduce the bs=128 OOM **without** AC, with `DINOV3_MEMORY_PROFILE=1` + per-rank `[MEMPROFILE]` + `[MEMFRAG]` + ideally `torch.cuda.memory._record_memory_history()` snapshot, so we can categorize the failure: working-set growth, rank-local skew, eval/checkpoint materialization, or fragmentation. Skipping this diagnostic risks repeating the worst-case profiler's mistake.
- **bs=128 retry deferred** until (a) the bs=96 selective-vs-full comparison is done, (b) the AC MFU-cost claim is replicated, and (c) we have the bs=128-no-AC memory snapshot for OOM characterization.
- **Node-variance investigation deferred but logged** (Codex #5): the 2× node-to-node MFU spread is large enough to invalidate cross-node MFU comparisons. All upcoming AC comparisons fix the node by running on gpu01 (or whichever Slurm picks for the AC run) and being explicit about *same-node* paired comparisons only. A separate gpu08 re-baseline + clocks/throttle audit is on the queue but does not gate AC work.

### 2026-05-14 — P5-04-verify done, P5-VAR + P5-AC-selective submitted

**P5-04-verify result** (SQL on existing job 39141 bs=96 SQLite):

| Kernel | Launches | Total sec |
|---|---|---|
| `ncclDevKernel_AllGather_RING_LL` | 2,847 | 36.2 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL` | 1,252 | 5.6 |
| `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL` | 348 | 0.054 |
| `ncclDevKernel_AllReduce_Sum_u64_RING_LL` | 84 | 0.001 |

**Auto-NCCL chose RING_LL for every dominant collective. Zero NVLS kernels** despite NVLS multicast being available on dev 0..7. NVLS drop is **provisional, not permanent** — the auto-tuner is not picking NVLS on this topology + collective sizes, so a clean (no-`NCCL_DEBUG`) NVLS test would be testing a genuinely different algorithm. Parked at low priority after AC work.

**Submitted**:
- Jobs **44793, 44794** — P5-VAR reruns of `fsdp2_bs96_baseline.sh` (n=3 baseline sample).
- Job **44795** — `scripts/fsdp2_bs96_ac_selective.sh` (P5-AC-selective). bs=96 + `train.checkpointing=true, train.checkpointing_full=false`, eval and checkpoint both forced to period=400 → coincident at iter 400 and 800. `DINOV3_MEMORY_PROFILE=1`, `DINOV3_MEMORY_PROFILE_PERIOD=10` → `[MEMPROFILE]` lines at every phase boundary + `[MEMFRAG]` lines every 10 iters. No codebase changes needed — full memory profile infrastructure already exists at `dinov3/utils/profiling.py:141-211` and `train.py:743-767`.

**Side observation**: `train.py:526` already calls `gc.disable()` and `train.py:588` does manual `gc.collect()` every 150 iters. **P5-05 (GC straggler hypothesis) is partially already in effect.** The straggler variance observed in nsys traces must come from another source — re-scope P5-05 to "diagnose why the existing GC strategy isn't fully suppressing rank straggle."

### 2026-05-14 — Second adversarial review (gpt-5.5 high effort) + plan refinements

Codex was re-run with the default `gpt-5.5` model and `model_reasoning_effort=high` (from `~/.codex/config.toml`). It flagged three high-priority issues and two medium ones against the revised plan as first written:

- **[high]** AC runs forced eval and checkpoint at non-coincident periods (`period=300 / eval=400` → events never overlap inside the window). The actual hypothesized failure mode is *combined* eval+checkpoint pressure. **Fix applied**: aligned periods to `checkpointing.period=400 / eval_period_iterations=400`, run length 1000 iters so we hit the combined boundary twice (at iter 400 and 800).
- **[high]** Snapshot points 50 iters after the phase boundary catch a clean allocator. **Fix applied**: snapshots at steady (iter 200) / pre-phase (iter 399) / immediately post-phase (iter 401) / post-recovery (iter 410), replicated at iter 800. Plus `try/except OutOfMemoryError` to dump `_snapshot.pickle` on OOM.
- **[high]** bs=96 percentage memory drop cannot gate bs=128 causality — activation memory scales with batch. **Fix applied**: gates rewritten to require an actual bs=128 AC measurement or a defensible GiB-scaling projection; bs=96 results are informational only for the bs=128 question.
- **[medium]** P5-VAR n=3 is statistically thin. **User decision**: keep n=3, accept the wider σ uncertainty, replicate borderline effects regardless.
- **[medium]** P5-ALLOC fragmentation thresholds are vague. **User decision (2026-05-14)**: deliberately do not commit thresholds in advance — we have no ground truth for fragmentation patterns in this workload; inspect the actual P5-OOM-memprofile-v2 snapshot and decide ad hoc.

Also added on Codex's prompting:
- **P5-04-verify** as a 5-minute SQL pass on the existing bs=96 nsys SQLite (job 39141) to confirm whether auto-NCCL already selected NVLS for the dominant all-gather / reduce-scatter kernels before deciding whether to park the NVLS experiment.

### 2026-05-14 — Adversarial review pushback + revised plan

After Codex adversarial review and a re-read of ch04 (NVLS caveat) + ch13 (AC guidance) + Meta's own preset configs (`dinov3_vit7b16_pretrain.yaml:54-55` uses selective AC at bs=16), restructured the plan at that point:

- **Park NVLS** (`NCCL_ALGO=NVLS` is a constraint, not a capability; the later P5-04-verify result keeps this as provisional, not permanently dropped).
- **Park standalone P5-03** (no-reshard alone) — Phase 4 already measured −0.3 pp at bs=256 with no memory benefit; below our current variance floor.
- **Insert P5-VAR**: 2 more clean bs=96 baseline runs to bound run-to-run variance. The 8–20% per-iter spread means our current screening can't reliably detect small wins; need a measurement floor before promoting results.
- **Prioritize P5-AC-selective then P5-AC-full**: Meta's repo + ch13 both point here. Selective AC matches the Meta ViT-7B recipe (`checkpointing=true, checkpointing_full=false`). Full AC matches their ViT-L distilled.
- **Insert P5-OOM-memprofile-v2**: extend memprofile to drive eval + checkpoint inside the snapshot window with `torch.cuda.memory._record_memory_history()`. Will distinguish working-set OOM from fragmentation OOM at bs=128.
- **Insert P5-ALLOC (gated)**: `max_split_size_mb` + `roundup_power2_divisions` tuning only if AC alone doesn't make bs=128 fit.
- **Reframe P5-05 as diagnostic**: measure GC correlation with low-MFU iters before intervening (Codex point).
- **PyTorch 2.6.0+cu124 confirmed** as the env — all AC APIs called by `ac_compile_parallelize.py` exist with the right signatures.

### 2026-05-14 — bs=96 baseline + stacked results (jobs 44023, 44024)

Both jobs completed (10:29 and 10:06 elapsed, exit 0). Steady-state numbers, iter ≥ 100 to skip torch.compile warmup, computed from instantaneous (not running-avg) MFU/step values:

| Config | Steady-state MFU | img/s | step_ms | max mem |
|---|---|---|---|---|
| P5-00 baseline (FSDP2 ZeRO-3 bs=96) | **14.67%** (7.97–20.19) | 2,564 | 335.4 | 24,922 MB |
| P5-03+P5-04 stacked (no-reshard + NVLS + DEBUG=INFO) | **6.59%** (5.10–9.28) | 1,151 | 662.6 | 25,089 MB |

**Result: stacked config is ~2× slower at the same memory.** Three confounds in the same run — cannot attribute the regression yet. Smoking-gun candidate: the stacked job's stdout/stderr is **51 MB** (vs 80 KB for the baseline) because `NCCL_DEBUG=INFO` logs every collective. With NCCL kernels at 70–80% of GPU kernel time, the print-overhead path is plausibly a large slice of the slowdown.

NVLS itself initialized cleanly: `NVLS multicast support is available on dev 0..7`, `NVLS comm … nHeads 8 buffSize 1048576 memSize 2097152 nvlsPerRankSize 150994944` — so the algorithm is available on this DGX H100 + NVSwitch box. The regression is not "NVLS unsupported, falling back to something slow" — that would have been visible in the NCCL init log as an explicit fallback line.

**Next action: deconfound.** Two clean re-runs needed:
- P5-04-clean: `NCCL_ALGO=NVLS` only, **no `NCCL_DEBUG`**, `reshard_after_forward=true`.
- P5-03-clean: `reshard_after_forward=False` only, **no `NCCL_DEBUG`**, no `NCCL_ALGO` override.

Memory: the no-reshard config did **not** cost meaningful extra peak memory at bs=96 (25,089 vs 24,922 MB → +167 MB, ~0.7%). This is good news for the memory hypothesis but doesn't justify the throughput cost.

Baseline observation worth recording: steady-state MFU at bs=96 ranges 8–20% iteration-to-iteration — significant variance, consistent with the straggler dynamics seen in the nsys traces. Mean ~15% is the new reference. The earlier ~24% MFU figure from screening was at bs=128 or bs=256 and is not directly comparable.

### 2026-05-13 — Submitted bs=96 baseline + P5-03+P5-04 stacked

- `scripts/fsdp2_bs96_baseline.sh` — job **44023** (FSDP2 ZeRO-3 bs=96, 500 iters, no nsys, no wandb). Established the initial bs=96 reference; later P5-VAR showed this gpu08 run was not representative of gpu01/gpu05.
- `scripts/fsdp2_bs96_noreshard_nvls.sh` — job **44024** (FSDP2 bs=96 + `reshard_after_forward=False` + `NCCL_ALGO=NVLS`, 500 iters). Both Phase 5 levers stacked in one run; `NCCL_DEBUG=INFO` enabled for NVLS verification.
- Both scripts explicitly `unset PYTORCH_CUDA_ALLOC_CONF` so no inherited `expandable_segments:True` leaks into the FSDP2 run (Codex adversarial review finding, 2026-05-13).
- 30-min walltime each, `--partition=research`.

### 2026-05-13 — Plan revision after re-reading ch4 + AC status review

- **Chapter 4 NCCL tuning is multi-node focused** — thread/buffer knobs (`NCCL_NTHREADS`, `NCCL_BUFFSIZE`) target inter-node transport. On a single-node 8×H100 + NVSwitch box, the relevant intra-node lever is the algorithm (`NCCL_ALGO=NVLS` = P5-04). Dropped the generic NCCL thread/buffer experiment from the ranked stack.
- **Activation checkpointing was unverified at this point.** `train.checkpointing=false` is the default, so P5-AC became a first-class experiment: audit the FSDP+AC wrap, enable it, measure peak-memory delta at bs=96, and use AC as the memory-cheap path back to bs=128 under the P5-OOM track. This was later resolved by the May 14-15 AC runs.
- **DALI** parked as a future option only — current profile says H2D is ~1% of GPU time; the loader is not the bottleneck.
- See the superseded stacked optimization plan retained below for history.

### 2026-05-11 — Doc reconstruction

- `docs/phase5_perf_plan.md` was lost when a Claude Code process was killed during the
  previous session (the file was untracked in git and didn't survive the kill). Reconstructed
  in full from the conversation transcript.

---

## 10. Open knowledge — `bs=128 FSDP2 OOM in real training` (2026-05-12)

Researcher whose project this is reported that **`batch_size_per_gpu=128` under FSDP2
went out-of-memory in an actual long training run**. The May 15 memory-profile run did
not reproduce the OOM in 1000 iterations with forced eval+checkpoint cycles, so the
original failure is still unresolved rather than disproven. Earlier diagnostics missed
important parts of the real memory envelope:

- nsys profile captures (60–120 s windows) ran too few steps to hit the OOM regime.
- We did not exercise the **eval cycles + sharded-checkpoint write paths** together
  with steady-state training under the same memory pressure.
- Activation checkpointing is now verified at bs=96 for both selective and full modes
  (jobs 44795/45280/45281). The remaining question is whether bs=128 + AC is fast and
  stable enough to promote.
- The FSDP2 configuration itself may be subtly incorrect for this setup.
- `scripts/memprofile_*.sh` (our worst-case memory profiling) did not mimic enough of
  the real memory variation (eval phases, periodic checkpoint, optimizer state in motion).

### Implications for Phase 5

- **Do not assume bs=128 is production-viable** until OOM root cause is understood and
  fixed. The previous "36.3 GB worst-case, 43 GB headroom" claim from the profiling
  scripts was **not predictive**.
- `reshard_after_forward=False` (the P5-03 lever) **increases** peak memory vs the
  ZeRO-3 default because the unsharded parameters remain resident across the forward.
  Therefore: **P5-03 must be tested at bs=96 first**, not bs=128.
- bs=96 is the current safe operating point until proven otherwise.

### P5-OOM follow-up — separate investigation track

To preserve as future work:

1. **Reproduce or retire** the OOM report under FSDP2 bs=128 with a longer run than the
   current 1000-iter profile, including repeated eval + checkpoint cycles.
2. **Memory snapshot** with `torch.cuda.memory._record_memory_history()` around the
   first eval + checkpoint cycle, dump to `.pickle`, render with `_dump_snapshot()`.
3. **Audit the FSDP2 wrap** in `dinov3/fsdp/ac_compile_parallelize.py`:
   - is each transformer block actually being wrapped with `fully_shard()` separately
     (per ch13 guidance for compile+overlap)?
   - is the outer `fully_shard(model, ...)` call double-wrapping?
   - is `MixedPrecisionPolicy` configured correctly (bf16 param / fp32 reduce)?
4. **Complete the bs=128 AC matrix**: selective and full AC jobs 45363/45364 determine
   whether AC should become the production path or remain only a memory fallback.
5. **Improve the memprofile script** to drive eval + sharded checkpoint inside the
   profiled window, not just steady-state forward/backward.

---

## 11. Current execution plan (updated 2026-05-15)

After the May 14-15 runs, Phase 5 is no longer a generic profiling phase. The active path is:
complete the batch×AC matrix, decide whether bs=128 + AC is worth promoting, then run a
longer validation before editing `run.sh`. References below tie each step to either the
local KB (`~/knowledge-base/`) or Meta's own DINOv3 preset configs.

### Cross-reference: Meta's own AC choices

| Config | bs/GPU | `checkpointing` | `checkpointing_full` |
|---|---|---|---|
| `dinov3/configs/train/dinov3_vit7b16_pretrain.yaml` | 16 | **true** | false (selective) |
| `dinov3/configs/train/dinov3_vitl16_lvd1689m_distilled.yaml` | — | **true** | **true** (full) |
| `dinov3/configs/ssl_default_config.yaml` (our ViT-B default) | 96 | false | false |

Meta enables selective AC the moment training becomes memory-pressured. Our 5-channel ViT-B has an unresolved bs=128 OOM report from real training (§10), so selective AC is the recipe-faithful first response; full AC is the fallback if selective fits too tightly.

### Cross-reference: ch13 canonical guidance

- ch13:1054–1070 — AC trades FLOPS for HBM; "natural fit for the latest generation hardware."
- ch13:1062 — "you don't have to checkpoint everything … most memory savings with minimal recompute overhead." Selective policy maps directly to our `checkpointing_full=false` path with the save-list at `ac_compile_parallelize.py:32–40` (saves `aten.mm`, `aten._scaled_mm`, `_scaled_dot_product_*`, `reduce_scatter_tensor` outputs — i.e. don't recompute the things we already paid the most for).
- ch13:1064 — "FSDP can enable automated checkpointing, which recursively applies checkpointing to multiple layers" — our code does this manually per block (lines 59–62), which is equivalent and more controllable.
- ch13:1112–1179 — older API example (`activation_checkpointing_policy={TransformerEncoderLayer, ...}`); our code uses the newer per-module `checkpoint_wrapper` pattern, which is the same idea applied explicitly.

### Cross-reference: PyTorch 2.6.0+cu124 (the env this project runs)

API surface confirmed in `/mnt/weka/adovlatyan/.conda/envs/test-conda-slurm/lib/python3.11/site-packages/torch/`:
- `torch.distributed.algorithms._checkpoint.checkpoint_wrapper.checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT, ...)` — default is non-reentrant, which is what we want with `torch.compile`.
- `torch.utils.checkpoint.create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False)` — the selective policy entry point.
- Both exist and have the signatures `ac_compile_parallelize.py` calls. **No version-skew concerns.**

### Parked / deprioritized levers

- **P5-04 (NCCL_ALGO=NVLS)** — parked, not permanently dropped. Auto-NCCL chose `RING_LL` in the bs=96 trace even though NVLS multicast was available, so forcing NVLS is a real experiment, but the first stacked run was confounded by `NCCL_DEBUG=INFO`. Revisit only after AC decisions.
- **P5-03 alone (no-reshard at bs=96)** — Phase 4 already measured this at bs=256: −0.3 pp with no memory advantage. With our 8–20% per-iter variance, a +0.3 pp screening is below detection. Deferred unless variance characterization (P5-VAR below) shrinks the noise floor.
- **P5-06 (NCCL thread/buffer tuning)** — chapter 4 NCCL knobs are multi-node focused; single-node 8×H100 + NVSwitch does not benefit.

### Revised sequence (run in order; analyze between each step)

| # | Experiment | KB / repo basis | What we learn |
|---|---|---|---|
| **P5-04-verify** *(done 2026-05-14)* | SQL pass on the existing bs=96 nsys SQLite (job 39141). **Result**: auto-NCCL chose `RING_LL` for **every** dominant collective — `AllGather_RING_LL` (2,847 launches, 36.2 s), `AllReduce_Sum_f32_RING_LL` (1,252 launches, 5.6 s), `ReduceScatter_Sum_f32_RING_LL` (348 launches, 0.05 s). **Zero NVLS kernels** despite NVLS multicast being available. | Codex review (gpt-5.5, 2026-05-14). | NVLS drop is **provisional, not permanent**. Auto-NCCL is not choosing NVLS on this topology + collective sizes, so forcing it was genuinely adding a different algorithm — we just don't know yet whether NVLS *helps* (the regression may have been `NCCL_DEBUG=INFO` overhead). Park `P5-04-clean` (NVLS without DEBUG) at low priority after AC work; do not drop it permanently. |
| **P5-VAR** *(done 2026-05-14)* | Variance baseline: 3 bs=96 baseline samples. | Codex review (small cost; tells us whether ±1 pp screening is detectable). | Node-to-node variance dominates: gpu08 was 14.67% MFU, while gpu01/gpu05 were ~7.0–7.4%. Future MFU comparisons must be same-node or replicated. |
| **P5-AC-selective** *(done 2026-05-14/15)* | bs=96, `train.checkpointing=true`, `train.checkpointing_full=false`, with coincident eval + checkpoint cycles. | ch13:1054–1070; ch13:1007–1010; Meta's `dinov3_vit7b16_pretrain.yaml:54-55`; `project_bs128_fsdp2_oom.md`. | Selective AC works: peak allocation dropped to 14.3 GB, fragmentation stayed stable, and MFU was comparable to same-node runs within current variance. Preferred AC variant if promoted. |
| **P5-AC-full** *(done 2026-05-15)* | bs=96, `train.checkpointing=true`, `train.checkpointing_full=true`, same eval + checkpoint pressure. | ch13:1062; Meta's `dinov3_vitl16_lvd1689m_distilled.yaml:84-85`. | Full AC works and cuts peak allocation further to 8.9 GB at similar measured throughput. Keep as memory fallback unless bs=128 selective is tight or unstable. |
| **P5-07** | Runtime FSDP2 wrap audit: write a small one-shot script that imports the wrapped model, walks every `FSDPState`, and prints (a) which modules got per-block `fully_shard()`, (b) the outer wrap state, (c) whether inference-only models have `reshard_after_forward=True` per `ac_compile_parallelize.py:255–264`. | ch13:1410–1422 wrap-granularity guidance. | Confirm the wrap matches the documented pattern; flag any double-wrap. Read-only, no training cost. |
| **P5-OOM-noAC / memprofile-v2** *(done 2026-05-15)* | bs=128 without AC, 1000 iters, two forced eval+checkpoint cycles, `DINOV3_MEMORY_PROFILE=1`, per-rank log review. | ch13:1007–1010 (memory profiler); existing `learnings/distributed_training.md:273–300` MEMPROFILE table. | Did not reproduce OOM: peak allocation was 33,034 MB on all ranks, no allocator retries/OOMs, fragmentation stable. This narrows the OOM question but does not prove long-run safety. |
| **P5-bs128-AC-selective/full** *(submitted 2026-05-15)* | Retry bs=128 with selective AC (job 45363) and full AC (job 45364). Both run 1000 iters with `checkpointing.period=400`, `eval_period_iterations=400`, and `DINOV3_MEMORY_PROFILE=1`. | bs=96 AC measurements plus user-requested matrix completion. | Completes the 2×3 batch×AC matrix. If bs=128 + AC is stable and throughput-positive, it can become the production candidate; otherwise keep bs=96 + selective AC as default and full AC as fallback. |
| **P5-ALLOC** (gated) | Tune `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,roundup_power2_divisions:[256:1,512:2,1024:4,>:8]`. Only if the bs=128 AC jobs still OOM and the memory snapshot shows fragmentation (large reserved-vs-allocated gap). | ch13:1007 framing — note user observation that ch13 framing is large-scale; we run it as a *small* fragmentation-driven try, not a primary lever. Skip `backend:cudaMallocAsync` (different allocator entirely; large unknowns). | Whether allocator tuning can close the bs=128 gap when AC alone cannot. |
| **P5-05** (deferred, diagnostic) | Per-rank step / data-time logging + `gc.callbacks` instrumentation in an untraced soak. Test `gc.disable()` only if GC events correlate with low-MFU iterations. | Codex review reframe (don't ablate GC before measuring it). | Whether the changing-rank straggler is GC-driven or inherent. Decides whether `gc.disable()` is worth a real intervention. |

### Stop-gates / decision points

- **After P5-VAR (n=3)**: the variance floor is high and node-dependent. Treat single-run MFU deltas as directional only unless the jobs are same-node or replicated; borderline effects get repeated.
- **After P5-AC-selective/full at bs=96**: selective is the preferred throughput default, full is the memory fallback. Do not use the bs=96 memory percentage alone to declare bs=128 safe.
- **After the bs=128 AC jobs (45363/45364)**: decide whether bs=128 + AC is throughput-positive and stable enough to become the production candidate. If not, keep bs=96 + selective AC as the likely default.
- **After P5-OOM-noAC / memprofile-v2**: current evidence shows a healthy 1000-iter no-AC run, not fragmentation. Keep the original OOM report open until a longer run or the bs=128 AC matrix settles it.
- **P5-ALLOC is gated**: only run if AC alone can't unlock bs=128 AND the memprofile-v2 snapshot suggests fragmentation. Don't add allocator knobs to a working config or to a working-set-bound OOM.

### What this plan deliberately does NOT do

- It doesn't chase 0.5–1 pp MFU wins until variance is bounded.
- It doesn't promote `reshard_after_forward=False` without an MFU detection floor that can see its effect.
- It doesn't change recipe (local_crops_number) — accuracy filter per project rules.

---

## 12. Superseded stacked optimization plan (2026-05-13)

This section is superseded by §11, but the details are intentionally retained so the other
agent can still follow the experiment history when updating this document after jobs 45363
and 45364 finish.

What survived:
- AC became the active path and is now verified at bs=96.
- Generic NCCL thread/buffer tuning stayed deprioritized for this single-node NVSwitch setup.
- DALI stayed parked because H2D was only ~1% of GPU time in the usable traces.

What changed:
- The stacked no-reshard + NVLS + `NCCL_DEBUG=INFO` run regressed badly and was too confounded
  to promote.
- NVLS is parked, not permanently dropped.
- No-reshard alone is deferred below the current measurement noise floor.
- The old 30% MFU target framing is not treated as a citable external goal.

### Historical ranked levers

| # | Experiment | Effort | Expected gain | Notes |
|---|---|---|---|---|
| **P5-03** | `reshard_after_forward=False` at bs=96 | trivial (1 flag) | −0.5 to +0.5 pp | Phase 4 (bs=256) gave −0.3 pp; bs=96 comm/compute ratio differs, but this is now deferred until variance is better controlled. |
| **P5-04** | `NCCL_ALGO=NVLS` (verify without `NCCL_DEBUG=INFO`) | trivial (1 env) | unknown | NVLink SHARP/NVLS is relevant to dominant collectives, but the first stacked test was confounded by debug logging. |
| **P5-05** | GC/straggler diagnostics | small code/instrumentation | tail-latency reduction if GC-correlated | Manual GC is already present, so future work should measure correlation before changing behavior. |
| **P5-AC** | Verify + enable activation checkpointing | audit + flags | memory reduction; possible bs=128 path | This is now done at bs=96 for selective and full AC; bs=128 AC matrix is pending. |
| **P5-07** | FSDP2 wrap audit | ~1 hour, read-only | sanity | Check per-block `fully_shard()`, outer wrap, and `MixedPrecisionPolicy`. Still useful if bs=128 behavior is surprising. |
| **P5-08** | Coarser FSDP2 wrap (group 2-3 blocks per shard unit) | moderate | fewer/larger all-gathers | Memory tradeoff; keep parked unless communication remains the top blocker after AC decisions. |

### Historical order of operations

1. Baseline FSDP2 ZeRO-3 bs=96 screening.
2. P5-03 + P5-04 stacked screening.
3. P5-05 diagnostic follow-up only if straggler evidence warrants it.
4. P5-AC audit + screening, then retry bs=128 with AC if memory headroom appears.
5. P5-07 read-only audit in background.
6. P5-08 only if easier levers leave the run short of the aspirational headroom target.

---

## 13. Forward-looking notes

- **NVIDIA DALI** — GPU-side data loading and preprocessing (decode + augmentation on the GPU, fed through a CUDA pipeline). Tracked as a *future* option only. Current profile evidence (H2D ≈ 1% of GPU time, NCCL = 70–80%) says the data pipeline is not the active bottleneck, so DALI is parked until a profile shows the loader becoming the long pole. Reference: ch05 / Fregly notes.

- **SimpleFSDP** (TorchTitan reference impl) — described in
  `~/knowledge-base/ai_systems_perf_engineering/ai_systems_perf_eng_ch13.md:1512–1532`:
  "reimplements FSDP in a torch.compile-friendly way using DTensor … in TorchTitan
  experiments, SimpleFSDP reduced memory usage by up to ~28% … improved training
  throughput by up to ~69% versus the traditional FSDP2 eager path." Adoption cost in
  this codebase is unknown — it would mean swapping `ac_compile_parallelize.py`'s
  `fully_shard()` calls for a different API. Worth tracking but **not a current Phase 5
  task**; revisit if P5-03 + P5-04 + P5-OOM exhaust the easier levers.

- **No "Meta paper 30% MFU target" exists** — earlier in this branch an exchange
  asserted Meta's DINOv3 paper targeted ~30% MFU. That is not in the original Meta
  paper or repo; treat the "30% MFU" figure in `learnings/gpu_performance.md:63–65`
  and `learnings/kernel_optimization.md:206` as an *aspirational headroom calculation*
  (37% of H100 BF16 MAMF), not a citable external target.

---

## 14. References

- `docs/perf_experiment_log.md` — Phases 1–4
- `docs/archive_decisions.md` — compressed history
- `docs/fsdp2_revalidation_tracker.md` — FSDP2 long-soak status
- `learnings/data_pipeline.md` — Ch05 + measured DataLoader benchmarks
- `learnings/distributed_training.md` — DDP vs FSDP2, NCCL overlap notes
- `learnings/profiling_workflow.md` — tool order and diagnostic checklist
- `~/knowledge-base/important_articles_lectures/` — five articles ingested 2026-05-07
- `scripts/nsys_profile.sh` — Slurm script to capture an nsys trace
- `scripts/nsys_dinov3_summary.py` — DINOv3-specific SQLite analyzer (4-hypothesis report)
