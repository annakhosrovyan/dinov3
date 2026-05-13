# Phase 5 — FSDP2 + Data/Compute Pipeline Optimization Plan

| Field | Value |
|---|---|
| Branch | `perf-fsdp2-pipeline` (branched from `perf-ddp-vs-fsdp` @ `a281fc8`) |
| Date opened | 2026-05-07 |
| Predecessor | Phase 4 — FSDP2 no-release screening (closed 2026-04-27) |
| Strategy going forward | **FSDP2 ZeRO-3 only.** DDP is closed — there is always an FSDP2 config equivalent to DDP, and FSDP2 is the long-run platform (Tim Darcet, 2026-04-25). |
| Production-viable batch sizes | **bs=64, bs=96, bs=128.** bs=256 is NOT achievable in real training (OOM in eval/checkpoint cycles per worst-case profiling) — do not use it as a comparison point. |
| Operating point we're optimizing | FSDP2 ZeRO-3 bs=128 (36.3 GB worst-case, 43 GB headroom — production-safe) |
| Goal | Identify and remove the next bottleneck, validated by nsys traces |

> **Note**: This file was reconstructed on 2026-05-11 from the conversation transcript after the
> original was lost when a Claude Code process was killed mid-session. Content is recovered;
> a few less-essential paragraphs may be terser than the original.

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

## 2. Status snapshot — what's already configured (verified 2026-05-07)

| Knob | State | Where |
|---|---|---|
| `num_workers=20` (2.5/GPU) | ✓ on | `run.sh:59` |
| `pin_memory=True` (always) | ✓ on | `dinov3/data/loaders.py:246` |
| `non_blocking=True` for H2D (7 tensors) | ✓ on | `dinov3/train/ssl_meta_arch.py:396–407` |
| `persistent_workers=True` | ✓ on | `run.sh:62` |
| `prefetch_factor=8` | ✓ on | `run.sh:63` |
| `cudnn.benchmark=True` | ✓ on | `dinov3/train/train.py:45` |
| `matmul.allow_tf32=True` | ✓ on | `dinov3/train/train.py:44` |
| FlashAttention2 (via SDPA dispatch on H100/bf16) | ✓ effectively on | `dinov3/layers/attention.py:116,159` |
| **Explicit copy_stream / compute_stream** | ✗ **not implemented** | — |
| **Packed multi-crop attention (NestedTensor / FA2 varlen)** | ✗ **not implemented** | — |
| NCCL bucket size tuning | ✗ default | — |
| NVTX annotations in train loop | ✗ none yet | — |

Most low-hanging pure-config knobs are already pulled. Remaining levers require code changes
(copy_stream, packed attention) or diagnostic work (nsys + bucketing).

---

## 3. Bottleneck hypotheses (unconfirmed without an nsys trace)

| Hypothesis | Mechanism | nsys signal |
|---|---|---|
| **A. Multi-crop forward structure** | `get_student_output()` does two sequential `forward_features_list` passes (seq 197 then seq 37); can't be batched (different seq lengths). | Two distinct attention-kernel sequences per layer per step. |
| **B. NCCL all-gather serialization** | Default bucket size may serialize gradient all-reduce against backward tail. ZeRO-3 also adds per-block all-gather. | Thick NCCL bar at backward-end, or NCCL stalls between backward kernels. |
| **C. H2D not actually overlapped** | `non_blocking=True` is necessary but not sufficient — depends on whether dataloader yields pinned tensors and on stream layout. | Yellow memcpy block + idle SM before first kernel of each step. |
| **D. Periodic Python GC / eval / checkpoint stalls** | Documented in CLAUDE.md. | Periodic step-time spikes uncorrelated with kernel content. |

We do **not** know which dominates without a real trace on the production config.

---

## 4. Articles ingested 2026-05-07 — relevance ranking

### Tier 1 — directly applicable
1. **CPU→GPU transfer (Articles 2 & 3)** — three-stage diagnosis: multi-process loader → pinned+non_blocking → explicit CUDA streams. We're at stage 2; stage 3 only if trace shows H2D stalling compute. Also: NCCL `bucket_cap_mb` is under-tuned by default (article reports 4% gain bumping to 100 MB on NVLink).
2. **Pipelining with CUDA streams (Article 4)** — pattern reference for copy_stream if Hypothesis C wins.
3. **Variable-length sequences (Article 5)** — structural lever for Hypothesis A (NestedTensor + FA2 varlen for multi-crop). Reported 2.5–3× attention speedups. **High effort; only pursue if profile confirms forward-pass dominates step time.**

### Tier 2 — diagnostic value
4. **Caching strategy (Article 1)** — bisect I/O time location. Useful as follow-up if nsys says data loading IS bottlenecked but doesn't explain why.

### Tier 3 — not directly relevant
5. Inference-side article on batched transfer.

---

## 5. Plan of attack

### Step 1 — Profile with nsys (decision gate)

Script: `scripts/nsys_profile.sh`. Profile two FSDP2 configurations at production-viable batches:

| Config | Purpose |
|---|---|
| FSDP2 ZeRO-3 bs=128 | Highest-throughput production-safe operating point (36.3 GB worst-case). |
| FSDP2 ZeRO-3 bs=96  | Conservative operating point. Comparison reveals throughput/MFU scaling. |

What to look for, in order:
1. Step-level GPU-idle gaps + duration distribution
2. H2D timeline: does memcpy overlap kernels?
3. Multi-crop forward: how long does each `forward_features_list` take?
4. NCCL: thick all-reduce at backward end?
5. Eval/checkpoint sync points (if any fall in the capture window)

### Step 2 — Decision tree

| If trace shows… | Lever to pull |
|---|---|
| Idle gap before first kernel of each step | Implement explicit copy_stream pipeline — Article 4 |
| Thick NCCL all-reduce at backward end | Tune `bucket_cap_mb` (try 100 MB) — Article 3; for FSDP2, change wrap granularity / `reshard_after_forward` |
| Forward dominates step time, attention is long pole | Investigate NestedTensor + FA2 varlen — Article 5 (high effort) |
| Periodic spikes with no kernel content | `gc.disable()` + manual `gc.collect()` every 100 iters |
| Idle gap mid-step, workers maxed out | Increase `num_workers` to 32; investigate Weka/HDF5 random-access |
| Trace clean, GPUs near-saturated | MFU-limited by ViT-B/multi-crop math; further gain only via architectural changes |

### Step 3 — Implement chosen lever, re-profile to confirm
One change at a time, re-run the same nsys script, compare traces side-by-side.

### Step 4 — Promote to `run.sh` only after a 500-iter soak test
Confirms zero memory creep + stable MFU. Mirror the Phase 3/4 promotion discipline.

---

## 6. Experiment ledger (Phase 5)

| Exp | Date | Config | MFU % | step_ms | max_mem GB | Notes |
|---|---|---|---|---|---|---|
| P5-01a | 2026-05-07 | nsys: FSDP2 ZeRO-3 bs=128 (job 38829) | — | — | — | **FAILED** — `module load nsight-systems` not available on GPU nodes; nsys unresolved; aborted in ~3s. |
| P5-02a | 2026-05-07 | nsys: FSDP2 ZeRO-3 bs=96 (job 38830) | — | — | — | **FAILED** — same root cause as P5-01a. |
| P5-01 | 2026-05-08 | nsys: FSDP2 ZeRO-3 bs=128 (job 39028) | n/a | n/a | n/a | Trace ran but **multi-process attach failed** — only rank 0 captured; 35 s NCCL hang. Re-run needed. |
| P5-02 | 2026-05-08 | nsys: FSDP2 ZeRO-3 bs=96 (job 39029) | n/a | n/a | n/a | All 8 ranks captured but only ~18 s of GPU activity in the 60 s window; ~3–5 steps. Compile + pinned-pool warmup not stabilized. NCCL↔compute overlap 0% (rank 0). Re-run needed. |
| P5-01b | 2026-05-08 | nsys re-run: FSDP2 ZeRO-3 bs=128 — `NSYS_DELAY=360, NSYS_DURATION=120` (job 39140) | n/a (trace-only) | n/a | n/a | **Trace usable.** 8 ranks captured. NCCL=70% of kernel time, NCCL↔compute overlap=16.3% (rank 0). Util 93–95% per non-straggler rank. Hypothesis B confirmed as dominant. |
| P5-02b | 2026-05-08 | nsys re-run: FSDP2 ZeRO-3 bs=96  — `NSYS_DELAY=360, NSYS_DURATION=120` (job 39141) | n/a (trace-only) | n/a | n/a | **Trace usable.** 8 ranks captured. NCCL=80% of kernel time, overlap=11% (rank 0). Straggler rank changes between runs → non-deterministic. |
| P5-03 | tbd | MFU screening: FSDP2 ZeRO-3 bs=128, `reshard_after_forward=False` (no nsys) | tbd | tbd | tbd | Pending. Test whether Tim Darcet's hint pays off: if MFU climbs vs default ZeRO-3, that's the lever. |

---

## 7. Risks and known gotchas

- **Realistic batch-size envelope is 64–128**: bs=192/256 are NOT production-viable for full training (OOM in eval/checkpoint per worst-case profiling). Do not propose bs=192/256 as production targets.
- **FA2 dispatch is conditional**: PyTorch SDPA dispatches to FA2 only when no custom mask is passed and bf16/fp16 dtype matches. iBOT masking happens before attention so SDPA sees clean inputs — but verify in nsys by looking for `flash_fwd_kernel`.
- **NCCL bucket tuning**: in FSDP2, `bucket_cap_mb` doesn't apply directly; module wrap granularity replaces it. `reshard_after_forward=True` (current) means per-block all-gather — that's the "bucketing" we'd tune by changing wrap granularity.
- **`expandable_segments:True` is DDP-only**: noted in `run.sh:32`. Not used in any Phase 5 run.
- **Compile warmup is ~30–40 iters for FSDP2 ZeRO-3**: set nsys delay accordingly.

---

## 8. Definitions of done for Phase 5

- [x] At least one nsys trace captured for FSDP2 ZeRO-3 bs=128 production config (jobs 39140/39141, 2026-05-08)
- [x] One bottleneck identified with trace evidence — **B (NCCL serialization), 70–80% of kernel time, 11–16% overlap**
- [ ] One lever implemented and re-profiled — pending P5-03 (`reshard_after_forward=False` screen)
- [ ] Soak test on the new config (500+ iters, eval + checkpoint cycles)
- [ ] Phase 5 row added to `archive_decisions.md`
- [ ] If a winner emerges: `run.sh` updated with rollback note

---

## 9. Session log

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

### 2026-05-11 — Doc reconstruction

- `docs/phase5_perf_plan.md` was lost when a Claude Code process was killed during the
  previous session (the file was untracked in git and didn't survive the kill). Reconstructed
  in full from the conversation transcript.

---

## 10. References

- `docs/perf_experiment_log.md` — Phases 1–4
- `docs/archive_decisions.md` — compressed history
- `docs/fsdp2_revalidation_tracker.md` — FSDP2 long-soak status
- `learnings/data_pipeline.md` — Ch05 + measured DataLoader benchmarks
- `learnings/distributed_training.md` — DDP vs FSDP2, NCCL overlap notes
- `learnings/profiling_workflow.md` — tool order and diagnostic checklist
- `~/knowledge-base/important_articles_lectures/` — five articles ingested 2026-05-07
- `scripts/nsys_profile.sh` — Slurm script to capture an nsys trace
- `scripts/nsys_dinov3_summary.py` — DINOv3-specific SQLite analyzer (4-hypothesis report)
