# Phase 7 — The flat-MFU residual investigation

> **Objective (unchanged since 2026-06-11):** explain and shrink the residual that pins DDP+cudagraphs
> MFU flat at ~20–21% across bs=128/160/192, then act on the cheapest lever.
>
> **Thesis pivot (2026-06-15):** Phase 7 *opened* believing the residual was **compute** (head FLOPs /
> the "dynamic iBOT shape" problem → pad → `max-autotune`). The first real nsys trace (job 75082)
> **disproved that**: the dominant residual is **inter-rank communication / synchronization**, not
> compute. The compute analysis is not deleted — it is *why we know* it's comms — but it is archived in
> **Part II**. All live work is **Part I**.
>
> **Why this stays "Phase 7" and not "Phase 8":** a phase boundary tracks an *objective*, not a
> *mechanism*. The objective ("why is MFU flat, how do we fix it") is the same; only the suspected cause
> flipped. Sub-indexing (7.0…7.9) keeps the artifacts that already cite 7.1/7.2 valid. Reserve "Phase 8"
> for a genuinely new objective (multi-node, larger model, a different MFU regime).

## How this doc is organized

- **Ladder & status** (below) — the single source of truth for what each sub-step is and where it stands.
- **Baselines** (below) — the three reference runs and what each is *for*.
- **Part I — LIVE: the communication thesis** — 7.1 attribution, 7.2 env sweep (done: H1 retired), Codex reviews,
  the capture/harness pipeline, consolidated methodology caveats.
- **Part II — ARCHIVE: rejected compute-residual hypothesis (7.0)** — the static-shape-heads / padding /
  autotune ROI analysis. Preserved intact, Codex-reviewed; superseded by 7.1.

## Phase 7 ladder & status

| Step | What | Status |
|---|---|---|
| **7.0** | Compute-residual hypothesis: pad iBOT masks → static-shape heads → unlock `max-autotune`. | **REJECTED** — nothing to pad (K is constant); and 7.1 shows the residual isn't compute anyway. See Part II. |
| **7.1** | nsys residual attribution (job 75082). | **DONE** — residual is comms/sync, not compute. |
| **7.2** | `CUDA_DEVICE_MAX_CONNECTIONS` sweep (=1 vs =8). | **DONE — H1 NOT supported.** No evidence `=8` helps wall throughput (observed favored `=1`; jobs 77671/77672, same node gpu01, 1 run each). Keep `=1`; insufficient signal to fan out to 16/32. |
| **7.3** | Gate `allreduce_metrics` to logging iters (`metrics_sync_period`, keep per-iter NaN guard via period). | **IMPLEMENTED; comparison RETRACTED (harness audit).** Knob lands mechanically (NVTX `allreduce_metrics` → 0 on skipped iters). The "−3.5% exposed NCCL (729→706)" cross-trace comparison is **invalid**: 75082's 729 was stall-inflated window-basis; kept-only the pair is **428.6 vs 705.9 ms/iter** (v2 reports) — the two traces are *incomparable* (different days, cross-capture stall variance), not reversed. The E2E verdict rests on **7.8a**, which is within-trace and unaffected: gating is ≈0 E2E lever. Closed. |
| **7.4** | Sinkhorn collectives (≥3 blocking fp32 `all_reduce`/iter, teacher-side). | **DEAD (80711, 7.8d).** The NVTX tags measured it: `dino_sinkhorn` **0.27%** + `ibot_sinkhorn` **0.31%** of wall (~3.6 ms/iter combined, bs192 v2 report). Sinkhorn is not the residual; the batch-growing residual points to episodic arrival stalls (7.8). No Sinkhorn-math work. Location note kept for the record: the active loss-side collectives are the DINO + iBOT Sinkhorn all-reduces inside `get_teacher_output` (`ssl_meta_arch.py:493/497`), not a center all-reduce in `compute_losses`. |
| **7.5** | Matched `cudagraphs=true`/`false` **control trace** (diagnostic only — not a ship candidate). | DEMOTED — attribution value only; tests the grad/compute-overlap hypothesis. |
| **7.6** | Enable NVLS (stop unsetting `NCCL_NVLS_ENABLE`). | PLANNED — helps *transfer*, not the tiny-payload barrier/skew that dominates; low priority. |
| **7.7** | bs128/160/192 differential capture (separate constant sync from scaling compute). | **DONE — RESTATED (harness audit).** The per-image compute U-shape is **retracted**: the bs192 anchor (80545) was stall-contaminated. Kept-only per-image compute is **monotonic decreasing**: 1.635 → 1.568 → **1.559** ms/img (80711 corroborates 1.564). The 160→192 diminishing return is **residual growth** — episodic multi-second arrival stalls plus exposed comms — not a compute regression. The "rank-6/7 exposed 65/107 vs ~880 ms" arrival-skew datum is also retracted (2-stall artifact; kept-only all ranks 46.3–70.7 ms). See 7.7 below. |
| **7.8** | Straggler attribution + metric-gate E2E quantification (sub-phase). | **DONE (job 80711).** 7.8a: metric-gating ≈0 E2E lever (closes Q2). 7.8d: Sinkhorn NVTX 0.27+0.31% ⇒ 7.4 dead. 7.8c+capture verdict: the straggler is **loader starvation** — rank 7 `[RANKDATA]` data_time 0.37–0.42 s vs ~0.3 ms on ranks 0–6, `[RANKTOPO]` shows all 8 ranks share one identical 192-CPU affinity mask with 8×24=192 unpartitioned workers. See 7.8 verdict below. |
| **7.9** | Straggler fix: DataLoader worker-budget headroom + per-rank NUMA-local CPU slices. | **DONE (2026-07-06, Codex-gated).** **Ship w16 no-slice: −11%/−6% median/mean node-controlled at bs160; −17% median at bs192 (cross-node).** Slicing REJECTED as default under current IRQ placement: storage-NIC IRQ/softirq zone inside job cpusets (gpu08: CPU 28 pin + cores 24-29/32 + CPU 0) turns the rotating loser into a fixed 5-15 s straggler; explains 7.8's rotation. Permanent fix = root IRQ steering to reserved cores 48-55/104-111 (proposed, NOT executed); opportunity signal = 345 ms clean-iter p50 at bs192. Full matrix + verdict in §7.9. |

> Ladder order is the consolidated view of our ranking + Codex's (GPT-5.5, high reasoning) two reviews.
> Where they diverged (7.5), see "Codex peer reviews" in Part I.

## Baselines (reference — keep these distinct)

| Role | Job | Numbers | Use it for |
|---|---|---|---|
| **Operative throughput baseline** | **67639** (torch 2.10, DDP bs128 cg, controlled 24w/8pf, 1500G/192CPU) | ~1,640 img/s run-avg, 2000/2000 PASS | apples-to-apples img/s comparisons. Refreshed by **77672** (=1, 700-iter throughput-only, gpu01): **2,116 wall img/s** steady-state (iter≥200) — higher than 67639's run-avg because it has no ckpt/eval and a shorter window. |
| **Attribution baseline** | **75082** (nsys trace, same champion config) | NVTX shares (Part I); kept-only exposed NCCL **428.6 ms/iter** ≈ 56% of p50 wall (v2 report — the old "93%/729 ms" was stall-inflated, see harness audit) | mechanism / where-time-goes. *Absolute* ms are inflated by nsys — trust shares + wall, not absolute CUDA-event ms. Cite the `.phase7.v2.*` reports for any NCCL/per-rank/kernel/H2D number. |
| Historical screen champion | 53708 | 2,394 img/s | **do not** compare against — torch-2.6-era, no eval/ckpt, different conditions. |

---

# PART I — LIVE: the residual is COMMUNICATION

## 7.1 — nsys residual attribution (job 75082)

**Source of truth:** `/mnt/weka/adovlatyan/nsys_profiles/phase7/75082/ddp-cg-bs128-w20pf8-75082.{nsys-rep,sqlite,phase7.v2.md,phase7.v2.json}`
(the v1 `.phase7.{md,json}` are kept as citations only — their NCCL/per-rank/kernel numbers are
window-basis and stall-inflated; see the harness-audit note below).
Champion config (DDP, `compile=true`, `cudagraphs=true`, `compile_mode=null`, bs=128, 8×H100, torch 2.10),
44 kept steady-state iterations. The analyzer dropped 2 out-of-band segments: 1 corrupt
(nsys/CUDA-graph end-of-session timestamps) and **1 real 13.5 s straggler stall** (rank 7 idle at
258.9 ms busy while rank 6 spun 13,479 ms — surfaced in the v2 stall table; the v1 report hid the
distinction).

### Headline

Phase 7 was created to chase a **compute** residual. The trace shows that framing is wrong: **the dominant
measured residuals are collectives / cross-rank synchronization, not head FLOPs.** (Codex-calibrated
phrasing — the stronger "compute is a minority" overreaches, because `forward_backward`'s 57% bundles
compute *and* the gradient all-reduce, and the 22% GPU-gap is idle wall, not provably communication.)

### Per-iteration wall breakdown (rank 0, wall-clock NVTX ranges — the trustworthy view)

| range | p50 ms | % of wall | what it is |
|---|---:|---:|---|
| **forward_backward** | 384.3 | **57.0%** | all student/teacher compute + graphed backbone + grad all-reduce |
| · teacher_fwd | 70.6 | 10.3% | EMA teacher forward (no_grad) — **also enqueues the Sinkhorn collectives** (see note) |
| · student_fwd | 125.0 | 16.7% | student global+local joint forward |
| · losses_backward | 165.0 | 29.7% | loss compute + `loss.backward()` (incl. exposed grad all-reduce + DINO/iBOT center all-reduces) |
| **allreduce_metrics** | 314.5 | **32.6%** | per-iter blocking `all_gather`(loss)+`all_reduce`(metrics) barrier |
| grad_clip | 15.2 | 2.1% | |
| optimizer_step | 9.7 | 1.7% | |
| ema_update | 0.7 | 0.5% | |
| untracked | 37.6 | 6.0% | dataloader wait / logging / Python |

- **GPU is idle 22.4% of the wall** (gap p50 = 107.6 ms of 769.3 ms; busy 625.4 ms).
- Step wall here (p50 769 ms, ~1,330 img/s) is slower than the bare champion screen (~428 ms, 2,394
  img/s) — expected: nsys attach overhead + node + torch 2.10. The *shares* are the signal.
- A second hand-read of one representative iter (teacher 18 / student 30 / losses_backward 409 /
  allreduce_metrics 359 / grad_clip 6 ms) agrees on the shape: two big ranges (`losses_backward`,
  `allreduce_metrics`), both comms-heavy. Per-iter numbers vary; trust the 44-iter aggregate shares.

> **Key reading-the-trace caveat (why NVTX ranges mislead here):** NVTX ranges are **CPU push/pop**
> times; NCCL collectives are enqueued **async** on a separate stream. The Sinkhorn collectives live in
> `get_teacher_output` (`ssl_meta_arch.py:493,497`), wrapped by `teacher_fwd` (~18 ms CPU) — but that
> 18 ms is only the *enqueue*. Their GPU execution + the wait for stragglers **surfaces later, at the
> first hard sync after backward — `allreduce_metrics`.** So the 314–359 ms there is **"the bill coming
> due"** for deferred async comms + rank skew, **not** the intrinsic cost of its own byte-sized
> collectives. NVTX shows where Python *waited*, not where the responsible GPU work was *created*.

### The two communication sinks

1. **`allreduce_metrics` = 32.6% of wall (~314 ms/iter).** Confirmed in code (`train.py:652-667`): an
   unconditional, **blocking** `all_gather_into_tensor(total_loss)` + `all_reduce(metrics_values)`
   **every iteration**, for NaN-checking and logging. Payload is a handful of scalars, so essentially all
   of it is **synchronization wait**, not transfer. It runs every iter even though metrics are only
   *logged* every ~10. **But** (per the caveat above and Codex) it is largely a *sink point*, not a
   self-cost — see the 7.3 design tension before assuming gating it wins.

2. **NCCL is mostly EXPOSED — corrected magnitude (2026-07-03, harness audit).** Kept-only (v2):
   union **486.2 ms/iter**, exposed **428.6 ms/iter** (11.9% hidden) ≈ **56% of the 769 ms p50 wall**.
   The originally published "union 787 / exposed 729 / ~93% exposed" was window-basis and included the
   dropped 13.5 s stall in the numerator. Direction holds (the residual is exposed collectives);
   magnitude was overstated. Dtype split, kept-only:
   - `AllReduce_Sum_bf16_RING_LL` (**263.3 ms/iter**) = the **DDP gradient all-reduce**. The DDP path casts
     params to bf16 with **no** FSDP-style `reduce_dtype` (`ac_compile_parallelize.py:335,348-353`,
     `static_graph=True`), so grads reduce in bf16.
   - `AllReduce_Sum_f32_RING_LL` (**194.2 ms/iter** kept-only; the published 494 included ~13.2 s of
     stall spin inside one f32 collective) = a **mixture of loss-side fp32 collectives**, NOT gradients —
     DINO Sinkhorn (`dino_clstoken_loss.py:43`), iBOT Sinkhorn (`ibot_patch_loss.py:30/36/42/49`), plus
     the per-iter metrics `all_reduce` (`train.py:663`; the loss `all_gather` at `:654` shows up as the
     separate `AllGather_RING_LL` kernel, 22.9 ms/iter kept-only). The "single largest kernel"
     claim is **retracted for this trace**: kept-only, the bf16 grad all-reduce is larger. Caution on
     reading dtype tables at all: NCCL LL kernel durations are dominated by spin-wait, so the split shows
     *which collective the wait surfaced in*, not intrinsic transfer cost — in the other captures
     (80545/80546/80547/80711 v2) f32 exceeds bf16, because the wait lands wherever the straggler arrives.
     Per 7.8d the Sinkhorn *work* itself is ~3.6 ms/iter (0.58% of wall) — the f32 ms is wait.
   - **Protocol is RING_LL, not NVLS** — we `unset NCCL_NVLS_ENABLE`, so NCCL fell back to ring.
   - **HYPOTHESIS (not proven): CUDA-graph backward reduces DDP grad/compute overlap.** Eager DDP fires
     bucket all-reduces as grads become ready; a graph-captured backward materializes all grads at
     replay-end, so the bf16 grad all-reduce can't hide behind backward compute. The mostly-exposed
     measurement (88% kept-only) is *consistent* with this but not *discriminating* — needs the 7.5
     control trace.

### Compute (for contrast — what Phase 7 originally targeted)

Kernel time by class (per iter, kept-only v2, after the `nvjet_*`→GEMM taxonomy fix): elementwise_triton
72.3 ms, **matmul_gemm 63.7 ms** (Hopper `nvjet_tst_*` cuBLASLt), attention_flash 26.8 ms, reduction
20.1 ms (incl. `indexFuncLargeIndex` 17.8 ms — the `block.py` index_add from the cudagraph-capture fix),
layernorm 10.9 ms. Total non-NCCL compute ≈ 210 ms/iter. Heads remain a tiny slice — consistent with the
~2.8%-of-FLOPs Part II finding. **No head-side or autotune change can touch the communication residual.**

## 7.2 — `CUDA_DEVICE_MAX_CONNECTIONS` sweep (DONE — H1 not supported)

> **Result (2026-06-24): H1 is not supported — no evidence `=8` improves wall throughput; the observed
> result favored `=1`. Keep `=1`; insufficient signal to justify a fan-out to 16/32.** Both jobs landed on
> the **same node (gpu01)**, so this is a matched control. The GPU-timed metric and the wall metric *point
> opposite ways*, and that disagreement — once read correctly — is the lesson.

| Metric (iter≥200, both gpu01) | `=8` (77671) | `=1` (77672) | favors |
|---|---:|---:|---|
| **WALL iter_time, s** (the metric that matters) | 0.5242 (med 0.5213) | **0.4840 (med 0.4488)** | **=1** (−7.7% mean / −14% median) |
| **WALL img/s** = 1024 / iter_time | 1,953 | **2,116** | **=1** |
| CUDA-event step_time_ms (GPU window) | **281.7** | 311.3 | =8 (−9.5%) |
| CUDA-event img/s | **3,570** | 3,245 | =8 |
| MFU % (CUDA-event basis) | **20.4** | 18.6 | =8 |
| data_time, s | 0.0005 | 0.0016 | both negligible — *not* data-bound |

**Why the two metrics disagree — corrected (Codex caught my first version, verified in code).** I originally
claimed `allreduce_metrics` was *outside* the CUDA-event window. **That is false in the current code.**
`step_start_event.record()` is at `train.py:604` (before `zero_grad`) and `step_end_event.record()` is at
`:692` — *after* `forward_backward` (:606), `grad_clip`, `allreduce_metrics` (:652–667), `optimizer_step`
(:688), and `ema_update` (:691). So **the metrics all-reduce/all-gather barrier is captured *inside*
`step_time_ms`.** The GPU-window metric therefore says `=8`'s on-GPU step (compute + in-step collectives)
got ~9.5% faster. Wall `iter_time` (helpers.py:99) covers the whole loop body *and* the host-side tail the
GPU window can't see: the `step_end_event.synchronize()` host wait (:695), MFU compute, `metric_logger`
bookkeeping, logging, periodic GC, and the schedule update. With `data_time ≈ 0`, that host/CPU tail is what
grew under `=8` (wall − GPU-window: ~173 ms at `=1` → ~242 ms at `=8`). **I cannot attribute that growth to a
specific cause** from two runs — candidates include CPU launch/scheduling overhead, host-sync interaction,
periodic GC alignment, or rank skew. The defensible statement is narrow: `=8` improved a GPU-timed slice but
did not improve (and here worsened) end-to-end wall throughput.

**Statistics caveat (do not overclaim).** The logged points are **not** independent samples: `iter_time` is a
20-sample rolling `SmoothedValue.avg` dumped every 10 iters (helpers.py:101,141), and the other meters are
window *medians* (helpers.py:60) — consecutive points share ~half their window, so they're heavily
autocorrelated, and there is only **one run per condition**. Wall variance is high (std ≈ 0.14 s, ~28% of
mean). The mean gap (~0.040 s) is only ~1.4 difference-SEMs even before accounting for autocorrelation, so
this is **not a statistically proven 7.7–14% regression**. The honest claim: *no evidence `=8` helps wall
throughput; the observed result favored `=1`.* The median gap is larger and the direction never reverses,
which is enough — combined with the weak theoretical prior — to retire H1 and not invest more GPU-hours here.

**Why no fan-out to 16/32.** Resource prioritization, not physics: H1 predicted `=1` was the constraint, and
it wasn't, so higher values have weak expected value. CUDA graphs *do* capture stream behavior under the
`CUDA_DEVICE_MAX_CONNECTIONS` in effect at launch, so the var is not provably irrelevant and 16/32 *could*
behave differently — but there is **insufficient signal to justify the spend**. H1 retired.

> **Methodology takeaway (promote to harness doctrine):** report **wall `iter_time`** as the headline. The
> CUDA-event `step_time_ms`/`images_per_sec`/`mfu` are *GPU-window* metrics — even though they include the
> in-step collectives, they exclude the host-side tail (sync wait, logging, GC, scheduling) and can move
> *opposite* to wall. The sweep script already prints both — read the wall line first. And cite metrics by
> their statistic: dumped `iter_time` is a smoothed **avg**, the rest are window **medians**, over one run.

<details><summary>Original H1 hypothesis & experiment design (kept for provenance)</summary>

**Hypothesis (H1, Codex-ranked #1 info-per-GPU-hour).** `export CUDA_DEVICE_MAX_CONNECTIONS=1` is set in
~60 scripts + `CLAUDE.md`, inherited from the FSDP2/Megatron-era recipe (where `=1` forces TP collective
ordering). It collapses all CUDA streams onto **one hardware work queue**, creating a false dependency
that can **serialize the NCCL stream behind the compute stream** — exactly the 93%-exposed-NCCL failure
mode. PyTorch *does* use multiple streams under the hood (NCCL/DDP grad reduction run on dedicated
streams); that separation is the entire overlap mechanism. For pure DDP (no tensor parallelism) `=1` is
likely unnecessary and possibly harmful. NVIDIA default is **8** (range 1–32, sans MPS).

**Experiment.** `scripts/screening/ddp_maxconn_sweep.sh` — single-variable sweep, everything else pinned
to the 67639 baseline (bs=128, 24w/8pf, 1500G/192CPU, no ckpt/eval, 700 iters, skip first 200 for the
steady-state mean). Self-reports steady-state img/s **and wall iter_time** from `training_metrics.json`.

| Job | `CUDA_DEVICE_MAX_CONNECTIONS` | Role | Result (wall img/s) |
|---|---|---|---|
| 77672 | 1 | matched re-baseline (gpu01) | **2,116** ✅ |
| 77671 | 8 | hypothesis (CUDA default, gpu01) | 1,953 ✗ (−7.7%) |
| (next) | 16 / 32 | fan-out only if =8 helps | **not submitted — =8 lost, no fan-out** |

**Honest expected outcome (Codex), confirmed.** Codex predicted flat-to-single-digit more likely than a
big win, with the graph-capture caveat that the env var should largely not matter at replay. The result
landed even slightly worse than flat on wall — fully consistent with that prediction.

</details>

**H2 (lower confidence, not yet swept):** `OMP_NUM_THREADS=8` / `MKL_NUM_THREADS=8` are also inherited
project-wide. Could contribute to rank skew via DataLoader worker oversubscription on the 192-CPU node,
but they don't directly explain exposed NCCL absent per-rank CPU/data-skew evidence. With H1 retired and
the binding cost confirmed to be the post-backward barrier (7.3), H2 is low priority — sweep only if 7.3/7.4
prove inconclusive.

## 7.3 — Gate `allreduce_metrics` (IMPLEMENTED; ≈0 E2E lever per 7.8a — the −3.5% comparison is retracted)

The metrics `all_reduce` is purely for logging (consumed every `print_freq=10`), so reducing it every
iter is a 10× over-call. Both reviews flagged a sharp caveat: the throughput cost is the per-iter
**barrier**, and the per-iter NaN `all_gather` *is* that barrier — so gating the whole block (gather +
reduce + NaN check) and *measuring* is the only honest way to size the gain.

**Implementation (`train.metrics_sync_period`, default 1 = unchanged).** Gates the entire
allreduce_metrics block AND the per-iter loss-side `.item()` logging to every Nth iter; `do_metric_sync`
is rank-invariant (desync-safe), block confirmed outside the cudagraph capture. Codex (GPT-5.5 high)
caught a real off-by-one in the final-iter guard (`iteration==max_iter` was dead; fixed to
`(iteration+1)==max_iter`). Trade-offs: NaN detection lags up to ~period iters; JSON loss fields become
sampled/stale.

**Result (job 80547, bs128, period=10) — restated 2026-07-03 after the harness audit.** The gate works
mechanically: the NVTX `allreduce_metrics` range drops to **0.0 ms** on skipped iters, and the deferred
wait *relocates* into `untracked` (p50 233.5 ms in 80547 vs 37.6 ms in 75082 — both per-kept-iter NVTX
numbers, unaffected by the harness bug). The 32.6% NVTX `allreduce_metrics` share was mostly the
**deferred DDP gradient all-reduce surfacing at that barrier** ("bill coming due"), not metrics traffic.

**The originally quoted "−3.5% GPU-side exposed NCCL (729→706)" is INVALID** (harness audit,
`docs/phase7_harness_audit.md` §4.2): 75082's 729 was a window-basis number inflated by a dropped
13.5 s stall. Kept-only the pair is **428.6 (75082) vs 705.9 (80547) ms/iter** (v2 reports) — different
days, and cross-capture stall variance dominates, so the two traces are **incomparable**, not evidence
that gating hurts. The conclusion does not change, because it no longer rests on this comparison:
**7.8a's within-trace measurement (unaffected by the bug) shows metric-gating is ≈0 as an E2E lever.**
The batch-growing residual points to episodic arrival stalls (7.8), which the gate cannot touch.

*(The NaN-safety design choice — (A) cheap 1-element boolean `all_reduce` per iter vs (B) NaN only on
log iters — is still open and now matters less, since the gate's throughput payoff is small. Default
`period=1` keeps current safety.)*

## 7.7 — bs128/160/192 differential: why batch ↑ gives diminishing returns (DONE — restated 2026-07-03)

> **Restated after the harness audit** (`docs/phase7_harness_audit.md`): the original Finding 1
> (per-image compute U-shape, sweet spot bs160) and the rank-6/7 arrival-skew datum were artifacts of
> the analyzer's mixed time basis — the bs192 anchor (80545) had **two dropped 19–20 s stalls = 67.4%
> of its capture window** leaking into every window-basis metric. All numbers below are kept-only from
> the `*.phase7.v2.{md,json}` reports.

**Captures:** 80545 (bs192), 80546 (bs160), 80547 (bs128, period=10) — **all on gpu02, same date**
(clean same-node comparison) — plus attribution baseline 75082 (bs128, gpu06) and the instrumented
80711 (bs192, gpu03). §6.B.7 soak motivation: wall img/s 1,640→1,840 (**+12%**, 128→160) then
1,840→1,880 (**+2.6%**, 160→192) at flat MFU.

**Methodology caveat (still decisive):** the per-iteration **wall** in these 60 s captures is
non-monotonic and contradicts the soak (kept-iter p50 = bs128 577 / bs160 389 / bs192 341 ms — bigger
batch shows *shorter* iters, impossible as real throughput). This is exposed-NCCL spin-wait dominating
the short window (same artifact as 7.2). So wall/gap/NCCL-time from these captures are **discarded**;
only per-iteration **GPU-kernel durations** are used — and after the audit, only **kept-only (v2)**
kernel durations, since the window-basis v1 numbers absorbed the dropped stalls.

**Finding 1 (RESTATED) — per-image GPU compute is monotonic decreasing; the U-shape is retracted.**
Per-rank non-NCCL GPU-kernel busy ÷ batch, kept-only (v2 per-rank tables, 8-rank mean):

| batch | trace | per-rank compute ms/iter | per-image ms | vs prev |
|---|---|---:|---:|---:|
| 128 | 80547 | 209.3 | **1.635** | — |
| 160 | 80546 | 250.9 | **1.568** | −4.1% |
| 192 | 80545 | 299.3 | **1.559** | −0.6% |
| 192 | 80711 (gpu03) | 300.4 | **1.564** | (corroborates) |

The v1 bs192 value (311.5 ms/iter → 1.622 ms/img) included the two stalls' kernel time; kept-only it is
299.3. Within this evidence set per-image compute **improves monotonically with batch** — there is no
bs160 sweet spot and no bs192 compute regression, so the GEMM wave/tile-quantization hypothesis loses
its motivating observation and is shelved. (Per the audit, one clean bs192 confirmation capture would
make the corrected number final; 80711 on a different node already agrees to within 0.4%.)

**Finding 2 (UPDATED) — the residual grows with batch; the bs192 traces point to episodic stalls as the
major component.**
Reconstructing iter-time from soak throughput (`iter_ms = batch·8 / img_s`) and subtracting kept-only
compute (the soak numbers are harness-independent; the compute anchor changed 311→299 at bs192):

| batch | iter ms (soak) | compute ms (kept-only) | **residual ms** | Δ iter | Δ compute | Δ residual |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 624 | 209 | **415** | — | — | — |
| 160 | 696 | 251 | **445** | +71 | +42 | +30 |
| 192 | 817 | 299 | **518** | +121 | +48 | **+73** |

For 160→192 the +121 ms iter-time growth now splits **compute +48 / residual +73 — residual-dominant**
(the earlier "~equal weight" used the stall-inflated 311 ms compute anchor). The earlier attribution of
the growing part to per-iter f32 loss-side collectives (Sinkhorn, K∝batch) is **dead**: 80711's NVTX
tags put Sinkhorn at 0.27+0.31% of wall (7.8d). What the bs192 traces actually show is **episodic
multi-second arrival stalls**: 80545 lost 38.8 s (67.4% of its window) to two stalls of 19.9 and 18.8 s
with rank 6 idle both times; 80711 has one 15.4 s stall with rank 7 idle (v2 stall tables). Steady-state
exposed NCCL at bs192 is 69.7 (80545) / 274.7 (80711) ms/iter — variable, and small relative to the
stall mass. The honest reading: the capture evidence points to worsening arrival stalls (loader
starvation, 7.8) as the major component of the residual growth, on top of a steady exposed-comms floor —
but the captures are 60 s windows, not the soak itself, so the 7.9 fix experiment in a throughput run is
the confirmation.

**Conclusion (7.7, restated):** the 160→192 diminishing return is **residual growth — with episodic
arrival stalls as the leading candidate for the growing part** — not a per-image compute regression
(compute per image improves monotonically), and **not** a compute straggler (per-rank compute spread ≤1.5% of mean at every batch,
v2). The previous rank-6/7 arrival-skew datum ("bs192: ranks 6/7 NCCL-exposed 65/107 ms vs ranks 0–5
~880 ms") is **retracted** — it was the two 80545 stall events spread over 47 kept iters
(2×19,400/47 ≈ 825 ms/iter); kept-only all eight ranks sit in a **46.3–70.7 ms** band. The arrival-skew
story survives on independent evidence instead: the stall events themselves and the `[RANKDATA]` loader
starvation (7.8).

**H2D (ground truth, all batches ~100% exposed, v2):** 5.7→9.1→12.3 ms/iter exposed, bandwidth
degrading 39.5→30.7→27.1 GB/s — a minor same-direction penalty on 192, reclaimable via a copy-stream
double-buffer but small relative to the residual.

## 7.8 — Straggler attribution + metric-gate E2E quantification (sub-phase, DONE)

Spun off from two open questions after the 7.7 review: (Q2) *how much* E2E does gating `allreduce_metrics`
actually buy, and (straggler) is the rank-6/7 collective-arrival skew a host-side lever, and can we even
separate it from the Sinkhorn cost (they're the same exposed-NCCL ms measured two ways)? *(The specific
per-rank skew datum that motivated the straggler question was later retracted by the harness audit — see
7.7 — but the stall events themselves are real, and the question was answered by 80711 below.)* Plan and the 7.8a
finding were Codex-reviewed (GPT-5.5, high reasoning, read-only) **before** this writeup; Codex's location
correction (Sinkhorn site) and three wording softenings are applied below.

### 7.8a — metric-gating is *not* an E2E lever (DONE, no job)

Method: within job 80547 (period=10, **single node gpu02** — no cross-node confound), classify each iter as
metric-**sync** (has an `allreduce_metrics` NVTX range) vs **non-sync**, then compare host-side CUDA
runtime-API durations per iter, **broken down by API**:

| API | sync ms | non-sync ms | Δms | sync n | non n | Δn |
|---|---:|---:|---:|---:|---:|---:|
| `cudaMemcpyAsync` (the `.item()` D2H reads the gate removes) | 0.37 | 0.36 | **+0.01** | 22 | 7 | +15 |
| `cudaStreamSynchronize` (metric barrier, sync iters only) | 325.35 | 0.00 | +325 | 15 | 0 | +15 |
| `cudaEventSynchronize` (per-iter MFU-timing sync) | 0.23 | 154.28 | −154 | 1 | 1.5 | −0.5 |

Reading: the logging reads the gate removes are **+15 scalar D2H copies costing +0.01 ms ≈ free**. The large
sync-vs-non-sync swing is the *blocking wait relocating* between two sync points — the metric barrier
(`cudaStreamSynchronize`, sync iters) and the per-iter MFU-timing `step_end_event.synchronize()`
(`cudaEventSynchronize`, 154 ms on non-sync iters). The naive "+171 ms on sync iters → ~28% of iter" bound is
**invalid**: it's the deferred DDP/NCCL bill *bunching* at the 1-in-10 sync iters, not removable `.item()`
work. Per-iter average host-block here = (6·325 + 53·154)/59 ≈ 171 ms; at period=1 each iter pays its own
small share, so the average is similar.

**Conclusion (7.8a) — closes Q2:** the *CUDA-side* cost the metric-gate removes is ~free, and combined with
the 7.3 mechanical result (gate works; the wait relocates rather than vanishes — the cross-trace −3.5%
figure is retracted, see 7.3) the evidence **predicts negligible E2E gain** from gating or removing
`allreduce_metrics`. Caveats (Codex): (1) CUDA runtime-API accounting misses pure-Python log
formatting / file I/O, so the precise claim is "the CUDA-side scalar-read cost is ~free," not "all logging
host cost is zero"; (2) this is a *within-trace wait-relocation* proof, not a matched-throughput measurement —
a hard "measured E2E≈0" still needs the matched-node soak (7.8b, demoted to optional); (3) the 154 ms
`cudaEventSynchronize` is irreducible *under the current per-iter MFU-timing design*, not globally.
*(A stale pointer removed 2026-07-03: this section originally redirected to the Sinkhorn lever (7.4);
the 80711 verdict below killed that — the lever is now 7.9.)* Script:
`scratchpad/p78a_hosttail.py` (within-node, per-API breakdown).

### 7.8d — NVTX-tag the loss-side Sinkhorn collectives (DONE)

**Location corrected (Codex, verified in code):** the active loss-side f32 collectives are the **DINO
Sinkhorn** (`ssl_meta_arch.py:493`) and **iBOT Sinkhorn** (`:497`) all-reduces inside `get_teacher_output`
(under the `teacher_fwd` NVTX range) — *not* a DINO center all-reduce in `compute_losses`. With
`centering: sinkhorn_knopp` (asserted at `ssl_meta_arch.py:40`) the async `update_center`/`reduce_center_update`
path is inactive. Each Sinkhorn does ≥3 blocking `dist.all_reduce`s with K∝batch (`ibot_patch_loss.py:36/42/49`).

Tags `dino_sinkhorn` / `ibot_sinkhorn` were added at the **outer call sites** (outside the `torch.compile`'d
`SinkhornKnoppTeacher.forward`, so compilation is not perturbed) and registered in the harness `NESTED_RANGES`
(`phase7_nsys_report.py`). The range bounds Sinkhorn **compute + its collectives**; a follow-up can intersect
it with the NCCL kernel union to isolate the pure collective share (Codex P1) — deferred until the range
proves significant.

### 7.8c — per-rank `data_time` + NUMA/affinity dump (DONE)

To attribute the host-side straggler we need to know whether the late-arriver rank is **loader-starved**
(fixable by worker/NUMA pinning) or **H2D/PCIe-bound** (a different fix). Two env-gated diagnostics
(`DINOV3_PERRANK_DIAG=1`, zero cost when unset), CPU-scalars only — no CUDA, no collectives, no shared-file
writes (Codex P4):
- **`[RANKDATA]`** — every rank prints its own `data_time`/`iter_time` each log step (`helpers.py`).
  `data_time` measures host wait *between DataLoader yields* (`helpers.py:106`), i.e. **loader starvation** —
  it does *not* include the `.cuda(non_blocking=True)` H2D (Codex P3): high `data_time` on the late rank ⇒
  loader/pin-memory starvation likely; low ⇒ not loader (could be H2D/PCIe, launch jitter, mask-shape
  variance, or pure arrival effects).
- **`[RANKTOPO]`** — one-time per-rank dump of `local_rank`, CUDA device, CPU-affinity count + range, and
  `Cpus_allowed_list` (`train.py:_log_rank_topology`). Shows whether workers are NUMA-pinned and which socket
  each rank lands on, to test the socket-1-locality hypothesis for the high ranks.

### 7.8 verdict (job 80711, bs192 instrumented capture) — the straggler is LOADER STARVATION

**Trace:** `/mnt/weka/adovlatyan/nsys_profiles/phase7/80711/ddp-cg-bs192-w24pf8-80711.phase7.v2.{md,json}`;
per-rank diagnostics in `/mnt/weka/adovlatyan/logs/p7-nsys-80711.err`. 59 kept iters; the analyzer
surfaced **one dropped 15.4 s stall: rank 7 idle (315.4 ms busy) while the other ranks spun ~15.4 s in a
single f32 all-reduce** (v2 stall table; max NCCL exposed inside the stall 15,051 ms).

1. **`[RANKDATA]` — the smoking gun, harness-independent (stderr, verified 2026-07-03).** Rank 7's
   `data_time` starves in **episodic bursts**: 0.246 s at iters 430–440, 0.745 s at 630–640, then
   **0.18–0.42 s sustained across iters 660–710** (0.416 / 0.368 / 0.368 s at 690/700/710 — the
   captured tail) vs **~0.2–1 ms on ranks 0–6 throughout**, with quiet stretches in between. Rank 7's
   DataLoader repeatedly cannot keep up; during those episodes it arrives late and the other seven ranks
   spin waiting. This is loader starvation, not H2D/PCIe (`data_time` excludes the H2D copy) and not GPU
   work (per-rank compute spread 0.84% of mean, v2). The episodic shape matches the episodic stalls in
   the traces.
2. **`[RANKTOPO]` — why a loser can exist.** All 8 ranks report the **identical affinity mask**
   `0-47,56-103,112-159,168-215` (n_cpus=192) — no per-rank CPU partition. So 8 ranks × 24 workers =
   **192 DataLoader workers plus 8 main processes (OMP_NUM_THREADS=8) compete unpartitioned for 192
   CPUs**. This is consistent with unpartitioned worker contention starving whichever rank loses the
   scheduling contest (and would explain the rank identity rotating across captures) — a supported
   mechanism, not yet a proven cause; the 7.9 fix experiment is the confirmation.
3. **7.8d answer — Sinkhorn is ~nothing; 7.4 is dead.** NVTX `dino_sinkhorn` = **0.27%** and
   `ibot_sinkhorn` = **0.31%** of wall (means 1.67 + 1.90 ≈ **3.6 ms/iter**, v2 NVTX table). The
   skew-vs-Sinkhorn partition of the exposed-NCCL residual resolves entirely to **arrival skew**.
4. **`allreduce_metrics` share at bs192 = 44.6%** (v2) — consistent with everyone waiting on the
   straggler plus the deferred DDP bill at the barrier. Fixing the straggler should collapse it without
   touching the metrics code (full metric removal stays moot per 7.8a).

**New prime lever (→ 7.9, separate work package):** give the loaders headroom and locality —
**worker-budget headroom** (drop below 24/rank so 8×workers + mains < 192 CPUs) and **per-rank
NUMA-local CPU slices** (workers inherit rank affinity, so partitioning the mask pins each rank's
loaders to a socket-local slice). The Codex P2 caveat stands: one capture identifies the mechanism
(starvation under oversubscription), and the rotating-loser model predicts *some* rank always starves;
the fix experiment itself will confirm.

### 7.9 — straggler fix experiment: worker-budget headroom + per-rank NUMA-local CPU slices

**Question:** does removing cross-rank CPU contention eliminate the episodic loader starvation (7.8
verdict) and shrink the residual — and does bs192 scale properly once the loaders are isolated?
Secondary: separate the two ingredients (fewer workers vs affinity slices).

**Mechanism under test:** at w24, 8×24 = 192 workers + 8 mains (OMP=8) compete unpartitioned for 192
allowed CPUs; the scheduler episodically deschedules one rank's workers and that rank's `data_time`
jumps ~3 orders of magnitude, stalling every collective (80711 `[RANKDATA]`). Treatment = (a)
**worker headroom**: w16/rank so 128 workers + mains < 192 CPUs; (b) **per-rank slices**:
`DINOV3_RANK_CPU_SLICE=1` pins each rank (and its forked workers, which inherit affinity) to a
private 12-physical-core NUMA-local slice — whole cores (both HT siblings via sysfs `core_id`),
ranks 0–3 on node 0, 4–7 on node 1, matching DGX H100 GPU topology. The GPU↔NUMA ordering is
*assumed* (standard DGX layout), not validated in code — `nvidia-smi topo -m` is dumped in the job
preamble as per-run ground truth; if it ever shows a nonstandard order, the run tests cross-NUMA
pinning and must be flagged (Codex review 2026-07-03). Implementation: `_apply_rank_cpu_slice()` in
`dinov3/train/train.py`, called first thing in `main()` so CUDA, NCCL, OMP pools, and DataLoader
workers all inherit the slice; emits `[RANKSLICE]` lines for verification. The capture script
hard-fails slice arms on <192-CPU allocations (the slicer would otherwise partition a small cpuset
and manufacture starvation).

**Arms** (all: capture script, 192 CPU / 1500G, pf8, `DINOV3_PERRANK_DIAG=1`, ITERS=900,
delay 720 s / 60 s window):

| Arm | Job | bs | workers | slices | vs control | Isolates |
|---|---|---|---|---|---|---|
| A | **87757** | 192 | 16 | ON | 80711/80545 (w24, no slice) | full treatment at the stress point |
| B | **87758** | 160 | 16 | ON | 80546 (w24, no slice) | full treatment at the candidate production point |
| C | **84637** | 192 | 16 | OFF | arm A | headroom alone vs headroom+slices |

(84635/84636 were the first A/B submissions — killed at t+1s by the `nproc`-vs-`OMP_NUM_THREADS`
guard bug, no GPU time consumed, resubmitted as 87757/87758 on 2026-07-06.)

**Readouts, in order of trust:**
1. `[RANKDATA]` starvation episodes (harness-independent stderr): any rank with `data_time_max`
   ≥ 50 ms while peers sit at ~0.3 ms. Success = zero episodes in the logged window. (`data_time`
   is a 20-sample rolling avg that dilutes single bursts ~20×; the line now also prints the window
   **max** as the raw episode detector — Codex review 2026-07-03.)
2. v2 report stall table: `dropped_slow` count and total ms (target: 0).
3. Per-rank NCCL-exposed band width (kept-only): should stay narrow AND lose the one-low-rank
   inversion signature.
4. Residual = wall − compute at bs192 (was 518 ms/iter kept-only): material drop expected if
   contention was the cause; per-image compute should stay ~1.56 ms/img (slices must not hurt GPU
   feeding).
5. img/s from `training_metrics.json` steady state vs the matched w24 controls.

**Interpretation guide:** starvation gone in A but wall flat → contention was real but not the
dominant residual term (falsifies the 7.8→residual link). Starvation persists in C but not A →
slices (isolation) are the active ingredient, not headroom. Starvation *uniform* across ranks in
A/B (all `data_time` elevated) → ambiguous between w16 undersupply and intra-slice contention (16
workers + OMP=8 main + pin/NCCL helper threads ≈ 25+ runnable on 24 logical CPUs); a w20+slices
rerun must then also drop OMP/MKL threads or it stays ambiguous (Codex review 2026-07-03). OMP=8 is
held constant across all arms *and* the w24 controls, so cross-arm comparisons stay valid either
way. bs192 residual collapses to ~bs160 levels → the "bs192 doesn't scale" verdict was an
allocation artifact; re-decide the production batch size.

**Caveats:** w16 < w24 cuts aggregate loader throughput ~33% — arm C exists partly to see whether
headroom alone starves uniformly; 60 s windows catch episodic bursts only probabilistically, so
`[RANKDATA]` (whole-run) outranks the trace for the starvation verdict; slices confine NCCL/CUDA
driver threads too, which is intended (isolation) but means a slice regression would also show up
as compute-side slowdown — watch per-image compute.

#### 7.9 execution log (2026-07-06)

**Arm C (84637, bs192 w16, no slices) — headroom alone: big median win, tail not fixed.**
Full-run steady state (iters ≥ 400): iter_time median **0.705 s / mean 0.741 s** vs w24 control
80711's **0.847 / 0.856 s** → **−17% median wall (~2,073 vs ~1,794 img/s)**. Capture window
(kept-only): exposed NCCL **251 ms/iter vs 530** at w24, wall p50 444 vs 472 ms. But
`data_time_max` shows heavier-than-w24 episodes: multi-second bursts (up to 7.5 s), rotating
ranks (1: iters 440–580, 0: 580–790). w24→w16 relieved median-level contention for everyone
while leaving too little supply margin per rank at bs192.

**Arm A as-run (87757, bs192 w16+slice v1) — INVALID as an equal-slice test; valid as a
dose-response confirmation.** gpu08 exposes **4 NUMA domains (Sub-NUMA Clustering)**, not 2:
GPUs 0–2 → domain 0, GPU 3 → domain 1, GPUs 4–6 → domain 2, GPU 7 → domain 3
(`nvidia-smi topo -m`, job preamble), and the Slurm mask covers the domains unevenly (28 allowed
cores on 0/2, 20 on 1/3). Slicer v1 chunked per NUMA node with 2 ranks/node → ranks 2,3,6,7 got
**10-core slices** (vs 14) and ranks 2/6 were pinned cross-domain. Result: ranks 6/7 (10-core)
showed sustained 1.7–2.8 s `data_time_max`; run throughput fell back to control (median
0.803 / mean 0.853 s). **Mechanism confirmation: construct a rank with less CPU supply and it
becomes a persistent straggler** — supply, not scheduling luck, sets who starves.

**Slicer v2** (train.py, Codex-reviewed 2nd pass, no blockers): block ranks by **package
(socket)**, order cores within a socket by (NUMA domain, core_id), chunk **equally** (remainder
spread). On gpu08's topology this yields 8 equal 12-core/24-CPU slices; ranks 3/7 land pure on
their GPUs' domains; exactly one rank per socket straddles an SNC boundary (2/6) — equal supply
beats strict SNC locality, given the v1 evidence. Verified by simulation against the exact gpu08
topology (disjoint, whole-core, exact partition) and reduces to v1 behavior on plain 2-node
machines. Known residual assumption: local-rank→socket blocking (0–3 pkg0, 4–7 pkg1) is
validated externally per run via the preamble topo dump, not inferred in code.

**Resubmissions:** A′ = **88301** (bs192 w16+slice v2), B′ = **88305** (bs160 w16+slice v2,
ITERS=1400 — see below; replaces cancelled 88302).

**Arm A′ (88301, equal v2 slices) — 7 of 8 ranks fixed; the 8th uncovered the real culprit.**
Equal slices worked as designed (`[RANKSLICE]` verified 8×12 cores): ranks 0,1,3–7 had **zero**
starvation episodes — including rank 6, the socket-1 SNC straddler. But **rank 2 starved
continuously from iter 110** (avg `data_time` 0.3–1.1 s, maxes 5–12.8 s in every 10-iter window;
per-rank NCCL-exposed 37 ms vs ~460 = the last-arriver signature). Run median 0.771 / mean
0.836 s — the rank-2 tail ate arm C's gain. The clean-iteration ceiling is however visible in the
capture: **wall p50 = 345 ms** (vs 444 arm C / 472 control), GPU gap p50 2.6%.

**Root cause found by node probes (jobs 88307/88309 on gpu08): the storage NIC's IRQ + NET_RX
softirq zone sits inside rank 2's slice.** `mlx5_comp0` for `0000:56:00.0/.1` is pinned to
**CPU 28**; cumulative NET_RX/mlx interrupt counts concentrate on cores 24–29/32 and HT siblings
(1.0–3.0 B each); socket 1's mirror cores are cold (why rank 6 is clean). Weka client threads
float (0–223 mask, no dedicated cores). **This also explains 7.8's rotating loser**: with
unpartitioned masks the scheduler migrates workers away from the taxed zone, so the loser
rotates; exclusive slices pin the tax to whoever owns the zone. Cores are not fungible — some
are pre-taxed by the kernel.

**Slicer v3 (A″ = 88310, bs192):** `DINOV3_SLICE_EXCLUDE_CPUS` keeps a cpulist out of every
slice (whole-core exclusion, `-xirq` run tag). Exclusion set `26-29,56-57,84-85` is symmetric
across sockets → **equal 11-core slices** on the untaxed cores (v1 taught that unequal supply
starves; 10-core slices starve at bs192/w16, so equality was preserved by matching socket-1
exclusions to its own narrow-pinned IRQ cores 56/84). Excluded cores form a housekeeping pool
that also absorbs nsys, the torchrun parent, and Weka's floating threads. Sim-verified against
the exact gpu08 topology.

**Proposed permanent fix (needs root, not executed):** steer the storage-NIC IRQs to the cores
Slurm already reserves out of every job's mask (48–55/104–111) and ban them in irqbalance. That
fixes the tax for all jobs and all users without any slicing logic; the exclusion knob then
becomes unnecessary. Same pattern as the perf-sysctl cluster change (audit doc + reversal
procedure).

**Arm A″ (88310, bs192, v3 exclusion `26-29,56-57,84-85`, equal 11-core slices) — NEGATIVE,
and it closes the slicing question.** Median 0.946 / mean 0.968 s, the worst arm. Ranks 1,3,4,7
were clean at 11 cores (so 11 cores alone doesn't starve at w16); rank 2 still starved (40
episodes — its slice retains un-excluded hot cores 24/25/32), rank 0 newly starved (20 episodes —
its slice holds **CPU 0**, three more `mlx5_comp0` pins, 6.1 B historical interrupts), rank 6
21 episodes (near the socket-1 NIC queue-pin range 57–83). The IRQ hot spots are scattered
through the mask; excluding them all cuts supply below the ~12-clean-cores/rank viability line
on socket 0. **Conclusion: static exclusive slicing cannot win on this node while the IRQs live
inside the job mask. Unpartitioned masks let the scheduler diffuse the tax — a rotating mild
loser beats a fixed heavy one.** The slicing lever is parked until/unless the root-level IRQ
steering lands; then A′ evidence (clean-iter p50 345 ms) says slices become very attractive.

**Arm D (88313, bs160 w16 no-slice, gpu02)** — median 0.657 / mean 0.668 s; episodes diffuse
across 5 ranks (maxes ≤3.8 s), the rotating-loser signature on gpu02's own IRQ geography.
Node-controlled comparison vs 80546 (bs160 w24, also gpu02): **w16 alone = −11% median / −6%
mean**.

#### 7.9 VERDICT (2026-07-06, Codex-gated)

Full-run steady-state iter_time (median/mean s, iters ≥ 400, same capture-script family):

| Config | bs192 | bs160 | Node |
|---|---|---|---|
| w24 no-slice (control) | 0.847 / 0.856 | 0.739 / 0.710 | gpu03 / gpu02 |
| **w16 no-slice** | **0.705 / 0.741** | **0.657 / 0.668** | gpu07 / gpu02 |
| w16 + slices v1 (uneven) | — | 0.618 / 0.615 | gpu08 |
| w16 + slices v2 (equal 12c) | 0.771 / 0.836 | 0.687 / 0.734 | gpu08 |
| w16 + slices v3 (11c + IRQ excl.) | 0.946 / 0.968 | — | gpu08 |

1. **Ship w16 (worker headroom), no slices.** Node-controlled at bs160 (both gpu02): −11%
   median / −6% mean. bs192 shows −17% median but cross-node (gpu07 vs gpu03) — direction
   supported, magnitude uncertain. Mechanism: 192 workers oversubscribed the mask and taxed
   every iteration, not just the loser rank.
2. **Static CPU slicing is rejected as the default production fix while storage-NIC IRQs live
   inside job cpusets.** Equal and IRQ-exclusion slicing produced *fixed* rank starvation
   (5–15 s bursts pinned to the IRQ-zone owner) and did not robustly beat no-slice controls.
   The bs160 uneven-v1 result (0.618, best in bracket) is an inconclusive exception —
   plausibly accidental tax-splitting, on a different node than the bs160 controls; it would
   need same-node/rotated repeats to rescue slicing.
3. **Root-level IRQ steering is the likely permanent fix** (storage-NIC IRQs → Slurm-reserved
   cores 48–55/104–111 + irqbalance ban). Current data show *opportunity*, not a measured
   post-fix result: non-taxed equal-slice ranks reached ~345 ms p50 clean iterations at bs192
   (vs 472 control p50), suggesting substantial headroom if IRQ interference is removed.
4. **Production: w16 everywhere; do not slice by default; bs128–192 all viable** subject to
   the existing host-RAM constraints (phase 6 §6.B.6). At current numbers bs160-w16 ≈ bs192-w16
   ≈ 1,900–2,100 img/s; the bs160/bs192 ranking is within node noise — re-decide after any
   IRQ fix.

Optional follow-ups (Codex list, not run): same-node w24-vs-w16 repeat at bs192; gpu08 no-slice
anchors; IRQ-steering A/B; uneven-v1 repeat with rotated slice→rank mapping.

**87758 (bs160 w16+slice v1, uneven slices) — secondary point, strong even with bad slices:**
median **0.618 / mean 0.615 s** vs bs160 w24 control 80546's **0.739 / 0.710 s** (−16% median,
~2,081 img/s ≈ arm C's bs192 rate). Episodes persist (ranks 1/2, maxes ≤2.5 s) but bs160 demand
is low enough that even 10-core slices mostly keep up. **Its nsys trace is EMPTY** (433 KB): at
w16 speed with a warm-cache startup (~200 s vs the 480 s the delay was calibrated for), 900 iters
finished at ~772 s — inside the 720 s nsys delay. Primary readouts (`[RANKDATA]`,
`training_metrics.json`) are full-run and unaffected. B′ therefore runs ITERS=1400 so the
720–780 s window lands in steady state on both warm and cold startups.

### Remaining open items

- **7.8b (optional, demoted):** matched-**node** period soak (bs128, period=1 vs 10, raw elapsed img/s) — only
  if a hard "E2E≈0" number is wanted beyond 7.8a's prediction.
- **`CUDA_DEVICE_MAX_CONNECTIONS=8 @ bs192`:** cheap env *screen*, low-EV (CUDA-graph capture pins stream
  topology, 7.2) — not mechanistically closed at bs192, but labeled screening, not diagnosis.

## Codex peer reviews

**Review 1 — GPT-5.5, high reasoning, 2026-06-15** (reviewed §7.1 + harness + grounding code, read-only):
the pivot from head/static-shape compute toward communication/sync is **supported**; the *causal* claim
"CUDA graphs killed DDP overlap" is a **hypothesis needing a matched control trace**, not a finding.
Corrections folded in: (A) softened "compute is a minority" → "dominant measured residuals are
collectives/sync"; (B) union NCCL exposure implemented — Codex independently recomputed 787.3/729.7/7.3%,
matching; (C) NaN-correctness constraint on gating + gain bounded <32.6%; (D) **fixed the dtype
attribution** (bf16 = DDP grads; f32 = loss-side barriers, since the DDP path has no `reduce_dtype`);
(E) outlier filter now classifies corrupt vs slow.

**Review 2 — GPT-5.5, high reasoning, 2026-06-22** (reviewed the env-knob hypotheses + reframed
attribution + ladder):
- **Confirmed** the "`allreduce_metrics` = bill coming due, not intrinsic cost" reframing, *and* the
  corollary that gating it may go flat — *especially* if the per-iter NaN `all_gather` stays.
- **`CUDA_DEVICE_MAX_CONNECTIONS=8` = the #1 experiment** (best info/GPU-hr; one line; directly attacks
  the 93%-exposed NCCL). Strongest pro/anti cases recorded in 7.2.
- **Downgraded the dtype phrasing**: f32 494 ms = loss-side fp32 *mixture* (Sinkhorn/center/metrics),
  not "= iBOT Sinkhorn." Found additional fp32 sites in DINO loss (`:43`, `:111`).
- **Ladder ranking (Codex):** 1) env knob, 2) cg true/false control trace *(for attribution value —
  Codex re-promoted this above our demotion; it's the cleanest test of graph-vs-overlap, still never a
  ship candidate)*, 3) gate metrics (weak if NaN barrier stays), 4) Sinkhorn (risky — touches math).
- **Three methodology caveats** (apply to every experiment): (i) single rank-0 trace can't distinguish
  self-wait from exposing *another* rank's delay — a real fix may need per-rank traces; (ii) the nsys run
  is much slower than champion — validate throughput with **wall `iter_time`**, not just CUDA-event
  `images_per_sec`; (iii) CPU NVTX ranges show where Python *waited*, not where the GPU work was
  *created*.
- **Vindicated by the 7.2 result (2026-06-24):** caveat (ii) was decisive — `=8` *won* on CUDA-event
  `images_per_sec` (+9.8%) but *lost* on wall `iter_time` (−7.7%). Had we trusted the GPU-window metric
  we'd have "confirmed" a win that doesn't exist end-to-end. Codex's "flat-to-single-digit, graph capture
  weakens the claim" expectation for the #1 experiment proved correct; the lever is retired, not promoted.
- **A 3rd Codex review (GPT-5.5 high, 2026-06-24)** of the 7.2 *result* caught two real errors in my first
  writeup, both verified in code and corrected above: (1) `allreduce_metrics` is *inside* the CUDA-event
  window (`step_end_event` at `train.py:692`), so the "off-window bill" decomposition was wrong; (2) the
  logged points are autocorrelated smoothed values over one run, not 51 independent samples — so the
  regression is "observed, favors `=1`," not statistically proven. Conclusion (keep `=1`, retire H1) stands.

> Project rule: review caveats are recorded as notes/ladder items, not auto-run as new experiments.

## Capture pipeline + analysis harness (supports 7.1, 7.5, 7.7)

Single-command pipeline so every ladder experiment produces methodologically identical, machine-comparable
numbers (no manual GUI inspection):

| Artifact | Path | Role |
|---|---|---|
| Capture job | `scripts/profiling/phase7_nsys_ddp_capture.sh` | sbatch: champion DDP+cudagraphs recipe under nsys (`--delay --duration`, NVTX via `--profiling`), then **in-job** `nsys export --type sqlite` + analyzer run |
| Harness | `scripts/profiling/phase7_nsys_report.py` | stdlib-only; emits `<trace>.phase7[.<tag>].md` + `.json` (`--tag v2` = current convention, `schema_version` 2); `--diff a.json b.json` → side-by-side verdict table |
| Outputs | `/mnt/weka/adovlatyan/nsys_profiles/phase7/<jobid>/` | `.nsys-rep`, `.sqlite`, `.phase7.{md,json}` (v1, citations only), `.phase7.v2.{md,json}` (current) |

**Harness methodology** (fixed across traces — what makes 7.5/7.7 verdicts one `--diff` each):
- Iterations segmented from consecutive `forward_backward` NVTX starts on rank 0's timeline (rank 0 via
  device-0 kernels → `correlationId` → CUDA-runtime `globalTid` → PID).
- Per-iter NVTX wall shares for all 6 top-level + 5 nested ranges; `untracked` = wall not inside any
  top-level range.
- GPU busy vs gap per iteration on device 0 (merged kernel+memcpy intervals).
- **NCCL exposure via wall-clock union footprints** (merge NCCL intervals, merge compute intervals, then
  sweep) — replaced the original dimensionally-mixed sum-minus-overlap (Codex rec). Also emits
  `nccl_sum_over_union` (1.0 for 75082 ⇒ serialized channels).
- **Kept-only time basis (F1, since v2 / `schema_version` 2):** every GPU-side aggregate (NCCL
  union/exposed, per-rank compute/busy/exposed, kernel classes, top kernels, H2D) clips kernel/memcpy
  intervals **by overlap to the union of kept iterations** before merging/sweeping — dropped-segment GPU
  time can no longer leak into per-iter averages. Guarantees `all-busy ≤ mean wall` by construction.
- Kernel taxonomy (matmul/attention/nccl/elementwise/…) per iter; `nvjet`→`matmul_gemm`,
  `indexfunc`/`scatter`→`reduction`.
- **Outlier removal with surfaced stalls (F2)**: drop segments outside `0 < wall ≤ max(10 s, 6×median)`
  before aggregating (nsys/CUDA-graph timestamp glitches otherwise poison every mean); splits
  `dropped_corrupt_ms` (≤0 or >30 s) vs `dropped_slow_ms`. Since v2, every dropped **slow-stall** gets
  its own `.md` table row (wall, idle rank, idle/busiest per-rank busy, max NCCL exposed inside the
  stall) plus a stall-inclusive mean wall — real straggler events are signal, not discarded noise.
- Validated on a synthetic nsys-schema sqlite (30 fabricated iters): all shares reproduce constructed
  ground truth, including the `--diff` constant-vs-scaling split.

### Harness audit (2026-07-02) and the v2 report convention

`docs/phase7_harness_audit.md` (Codex-gated, GPT-5.5 high) audited the analyzer against direct sqlite
re-derivations: **1 BUG + 2 MISLEADING**. The bug was a **mixed time basis** — NCCL exposure, per-rank
tables, kernel classes and H2D were capture-window totals divided by *kept*-iteration count, so any
multi-second stall the outlier filter dropped still contaminated them (80711: exposed NCCL published
529.8 ms/iter vs **274.7** honest, 1.93× overstated; 80545: 67.4% of the window was dropped stall time).
The per-iter NVTX accounting (shares, wall percentiles) was verified **exact**, and the capture script
sound. Fixes F1–F5 were applied 2026-07-03 (`schema_version` 2) and all five traces regenerated as
`*.phase7.v2.{md,json}` alongside the untouched originals.

**Citation rule going forward:** for any NCCL/per-rank/kernel-class/H2D number, cite the `.phase7.v2.*`
report; v1 `.phase7.{md,json}` files remain valid only for the per-iter NVTX family (range shares, iter
wall stats, untracked). Conclusions restated from v2: 7.7 Finding 1 (U-shape retracted → monotonic),
7.3 (−3.5% comparison invalid), 7.1 magnitudes (exposed 428.6, not 729/93%), 80545 per-rank skew datum
(retracted). Unchanged: 7.8a, the soak throughput inputs, and the `[RANKDATA]` straggler evidence (all
harness-independent) — though the 7.7 residual table itself was recomputed with the v2 compute anchor
(bs192 residual 506 → 518).

**Capture attempt log** (each failure was a distinct real root cause — kept so we don't repeat them):

| Job | Result | Lesson |
|---|---|---|
| 71041 (06-12) | FAILED — empty trace + host-RAM OOM | startup ~480 s so `delay=360` caught only warmup; rank-5 OOM at 512G default; `set -e` skipped export on the dead-app exit code |
| 71682 (06-13) | FAILED — 0-byte `.nsys-rep`, finalization killed at 48% by 90-min wall | `--python-sampling` + `--sample=cpu --cpuctxsw` across 8 ranks → huge backtrace tables nsys finalizes pathologically slowly (and unneeded: `data_time≈0` ⇒ residual is GPU-side) |
| 74635 (06-15) | FAILED — `.qdstrm` written but `.nsys-rep` finalization crashed | `triton.cudagraphs=True` + default `--cuda-graph-trace=graph` → device-graph event with replay-relative start > absolute end crashes finalization |
| 75082 (06-15) | **SUCCESS** — 56M `.nsys-rep`, 160M `.sqlite`, reports generated | fix: `--cuda-graph-trace=node` (per-kernel activity inside the graph) — sidesteps the crash AND is what kernel taxonomy needs |

---

# PART II — ARCHIVE: rejected compute-residual hypothesis (7.0)

> **REJECTED, preserved as the record of *why* the residual is comms.** This was Phase 7's opening thesis:
> make the head path static-shape (pad masked-patch tensors to fixed K) so it becomes CUDA-graph eligible,
> to shrink a presumed *compute* residual. Two findings killed it: (B) K is already constant — there is
> nothing to pad; and 7.1 (Part I) then showed the residual isn't compute at all. The autotune sub-track it
> would have unlocked is also moot for the residual (heads are ~2.8% of FLOPs). Reviewed by Codex
> (GPT-5.5, high reasoning, 2026-06-11); its corrections are marked ⚖.

## 7.0.0 Verdict

| Quantity | Value | Source |
|---|---|---|
| **Head matmul-MAC fraction** (dino_head + ibot_head, of total step MACs) | **~2.8%** (6.15 / 221.7 GMACs/img; ibot_head 2.44%, dino_head 0.33%). ⚖ matmul-MAC estimate, not a wall-time ceiling — loss/sinkhorn elementwise time is unmeasured. | §2; `ssl_default_config.yaml:13-17,40-44`, `dino_head.py:53-67`; Codex matched (6.147 GMACs, 2.77%) |
| **Padding waste** | **Moot — no padding needed.** K is ALREADY constant: masked-patch count = deterministic collate `upperbound` (7,503 at bs=128) exactly, every iter. Verified 5/5 trials. | §3; `collate.py:42-56`, `masking.py:87-93` |
| **Net MFU ceiling from head CUDA graphs** | ≤ +0.6 pts absolute; realistic **+0.1–0.3 pts** — below variance | §4 |
| **Recommendation** | **Do not build Option B.** Autotune track reopens at near-zero cost, but (see 7.1) it cannot touch the comms residual. | §5, §7 |

## 7.0.1 Finding A — the "per-iteration recompile cliff" does not exist

The brief assumed dynamic K forces Inductor to recompile the head path every iteration. Refuted:
- Heads compiled via plain `module.compile()` (`ac_compile_parallelize.py:228-233`) — automatic dynamic shapes.
- Job 16719 (default mode, DDP bs=128) ran 200 iters at ~23.1% steady-state MFU / ~150 ms/iter
  (`learnings/compile_modes.md:32-38`) — impossible with per-iter recompiles.
- The Phase 4 "minutes" cost was `max-autotune-no-cudagraphs` autotuning at iter-0 (18 min, then SIGABRT
  — `compile_modes.md:41-58`), not a recompile loop.
- ⚖ Codex: "second distinct K triggers exactly one recompile" is asserted, not demonstrated. Moot per B.

## 7.0.2 Head FLOP fraction: ~2.8%

Both heads identical: MLP 768→2048→2048→256 + prototypes 256→65536, no bias.
- MACs/token: MLP 6.29 M + prototypes 16.78 M = **23.07 M** (prototype layer 73%).
- Token-passes/img at bs=128 (K=7,503/rank → 58.6 masked patches/img): student dino 30, teacher dino 2,
  student ibot 176, teacher ibot 58.6 → 266.5 passes → **6.15 GMACs/img**.
- Backbone 215.6 GMACs/img → **head fraction 2.77%** (⚖ Codex matched). Counts matmul MACs only;
  sinkhorn/log-softmax over 65536 prototypes are elementwise, unmeasured in *time*.
- `dino_head` is already static-shape; the dynamic question was always only the iBOT path.
- Even deleting the heads entirely buys ≤2.8% compute.

## 7.0.3 Finding B (decisive) — K is ALREADY constant

The brief's premise ("K varies every iteration") is false at fixed per-GPU batch:
1. Per-sample mask counts are deterministic: `n_samples_masked=int(B·0.5)`, targets `int(196·probs[i+1])`,
   `probs=linspace(0.1,0.5,n+1)` — only *which* patches is random (`collate.py:42-56`).
2. ⚖ `MaskingGenerator.__call__` ends with `complete_mask_randomly()` force-topping each mask to exactly
   its target (`masking.py:87-93`) — Codex-caught; stronger than the draft's "K ≤ upperbound".
3. **Empirical: 5/5 trials give K = 7,503 exactly** at bs=128.

Consequences: `mask_indices_list` is (7503,) every iter; the `collate.py:63` "variable shape" comment is
wrong; Phase 4's "different masked token count" narrative (`compile_modes.md:106`) is wrong at fixed bs and
needs re-attribution. ⚖ Scope: holds for the default path; `get_batch_subset()` (multidist, unused) and
gram `tokens_used` would reintroduce dynamic shapes — both disabled here. Changing bs changes K (one
recompile, fine).

## 7.0.4 Direct gain from head CUDA graphs: noise-level

Upper bound 2.8% of FLOPs → ≈ +0.6 pts at 21%; realistic +0.1–0.3 pts — below §6.B.7 variance.
`ibot_head` is a pure MLP, clean to capture; **Sinkhorn must stay out of capture scope** (three
`dist.all_reduce`s, `ibot_patch_loss.py:36,42,49`).

## 7.0.5 The autotune sub-track (orthogonal to cudagraphs) — moot for the residual

Since K is static, the experiments Option B would unlock are runnable with no collate/loss changes.
Autotuning and CUDA graphs are ORTHOGONAL: `fullgraph+cudagraphs` eliminates launch overhead via replay;
`max-autotune` picks faster Triton matmuls; the target is both. Under `cudagraphs=true`, `compile_mode`
does NOT touch the backbone (blocks take the graph branch first; `compile_mode` reaches only the heads,
hardcoded `use_cuda_graphs=False` at `:233`).

| Experiment | Change | Measures |
|---|---|---|
| (a) `cudagraphs=false` + `compile_mode=max-autotune-no-cudagraphs` | config-only | autotune matmul gain in isolation; NOT production (drops cudagraphs) — a go/no-go for (b) |
| (b) branch 1 → `options={"max_autotune": True, "triton.cudagraphs": True}` | one line in `wrap_compile_block:72` | autotuned kernels inside replayed graphs on the champion path |
| (c) heads → branch 1 | one line at `:233` | noise-level direct gain |

⚖ The 5–20% figure is expected matmul-only gain, unmeasured model-wide. **7.1 supersedes this entire
sub-track for the residual question** — autotune speeds compute, and compute is not the bottleneck.

## 7.0.6 Correctness notes (kept for the record, moot without padding)

- If padding were ever needed: `forward_masked` is pad-safe via zero `masks_weight` (⚖ denominator is
  `student_masks_flat.shape[0]` = n_crops, not B·N); the teacher sinkhorn is NOT pad-safe (pad rows couple
  through `sum_Q`/marginals), and ⚖ the naive "zero the pad rows" divides by zero column sums
  (`ibot_patch_loss.py:54`) → NaNs; a guard would be required, plus a loss-equivalence test.
- The naive K_max `ceil(mask_ratio_max·B·N)` = 25,088 would have cost +5.7% of compute (234% padding on
  the iBOT path) — the trap the deterministic-upperbound observation avoided.
