# Phase 7 profiling-harness audit

**Date**: 2026-07-02 · **Branch**: `exp/ddp-fullgraph-head-static-shapes` · **Auditor**: Claude (per `docs/phase7_harness_audit_handoff.md`)
**Files audited**: `scripts/profiling/phase7_nsys_report.py`, `scripts/profiling/phase7_nsys_ddp_capture.sh`
**Primary evidence**: `/mnt/weka/adovlatyan/nsys_profiles/phase7/80711/ddp-cg-bs192-w24pf8-80711.sqlite` (+ 75082, 80545, 80546, 80547)
**Codex gate**: GPT-5.5, high reasoning, read-only, 2026-07-02 — confirmed the core bug, independently reproduced the headline recomputations, four corrections accepted after self-verification (see §6).

```yaml
# machine-readable summary
verdicts:
  1_heavy_tails: OK           # real skew; reported mean UNDERSTATES tail (worst stall dropped)
  2_dropped_segments: MISLEADING  # correctly kills corrupt segment, also drops a real 15.4s stall; .md hides the distinction
  3_nccl_exposed_math: BUG    # mixed time basis: window numerator / kept-iter denominator; 529.8 -> 274.7 ms/iter honest
  4_nvtx_partition: OK        # exact partition, shares reproduce
  5_cudagraph_attribution: OK # node granularity, per-kernel rows, sane timestamps; minor join fragility
  6_iteration_segmentation: OK  # capture-end garbage NVTX handled; head/tail edges handled
  7_rank_device_mapping: MISLEADING  # mapping itself OK; published per-rank values window-contaminated
  8_h2d_dma: OK               # pinned HtoD, 25.6 GB/s, 0% hidden all verified; minor window-basis caveats
  9_capture_script: OK        # env propagation, NVTX wiring, window landing all verified
affected_conclusions:
  - "7.7 Finding 1 per-image compute U-shape: GONE after correction (monotonic 1.635 -> 1.568 -> 1.559 ms/img)"
  - "7.3 metric-gate -3.5% exposed NCCL (729->706): INVALID comparison (stall-inflated vs uncontaminated trace)"
  - "80545 per-rank 'ranks 6/7 exposed 65/107 vs ~880 ms': artifact of 2 stall events; kept-only all ranks 46-71 ms"
  - "75082 exposed NCCL 729 ms/iter -> 428.6 kept-only; comms-residual conclusion survives directionally"
  - "7.7 Finding 2 residual growth 415->445->506: soak-derived, arithmetic stands; Sinkhorn attribution weakened"
  - "rank-7 straggler: [RANKDATA] stderr evidence stands; harness corroboration survives in weakened form"
```

## 1. The core defect — one root cause, several symptoms

`phase7_nsys_report.py` segments iterations from rank-0 `forward_backward` NVTX starts
(`analyze()`, lines 306–334), drops out-of-band segments (`0 < wall ≤ max(10 s, 6×median)`,
line 322–323), and then computes two families of metrics on **different time bases**:

- **Per-kept-iter basis** (clean): NVTX range stats, iter-wall stats, untracked, GPU busy/gap
  (lines 337–359). These clip to each kept iteration individually.
- **Contiguous-window basis** (the bug): NCCL union/exposed/hidden (lines 365–372), per-rank
  compute/all-busy/NCCL-exposed (lines 409–417), kernel classes and top kernels (lines 427–443),
  H2D union/exposed/top-sizes (lines 380–404). These query
  `[t0, t1] = [first kept iter start, last kept iter end]` (line 351) and divide by
  `n_iters` = kept count. A dropped segment that sits **inside** that window leaves all of its
  GPU time in the numerator while contributing nothing to the denominator basis.

Codex's sharper phrasing, adopted here: this is a **basis-selection bug**, not an
interval-math bug. The merge/sweep/union/overlap-subtraction logic is correct (verified — an
independent reimplementation reproduces the published 554.098 / 529.786 / 4.4% / sum-over-union
1.0 exactly). The wrong part is selecting `[t0, t1]` instead of the union of kept intervals.

### What that means in 80711 concretely

Re-derived by direct sqlite queries (scratch scripts `audit1_segmentation.py` …
`audit5_ushape_ranks.py`; independent merge/sweep implementation):

- 67 `forward_backward` starts on the primary tid → 66 segments → 61 after `skip_iters=5` →
  **59 kept, 2 dropped**: segment idx 13 with wall **15,385.5 ms** (a real stall, fully inside
  the window) and the final segment with wall 2.019×10⁸ ms (corrupt, span-in-window 0 ms).
- Window `t1−t0` = **51,808.9 ms** = 36,423.4 ms (sum of kept walls) + **15,385.5 ms** (the
  dropped stall). Every window-basis metric is a 51.8 s total divided by 59.
- Inside the stall, ranks 0–6 each sit in a single ~14.9 s
  `ncclDevKernel_AllReduce_Sum_f32_RING_LL` spin (per-device busy inside the stall
  15,358–15,380 ms); **rank 7 is idle (315.4 ms busy)** — rank 7 arrived ~15 s late at one
  collective. NCCL union inside the stall on device 0: **15,140.2 ms**.

The published report contradicts itself once you see this: "NCCL exposed **529.786 ms/iter**"
exceeds the iter-wall p50 of 471.8 ms, and per-rank "all-busy **857.3 ms/iter**" exceeds the
mean wall of 617.3 ms. Neither is possible for a typical iteration.

### Headline re-derivations (80711, device 0)

| metric | published (window ÷ 59) | kept-only (clipped to kept iters) |
|---|---:|---:|
| NCCL union ms/iter | 554.098 (reproduced exactly) | 297.5 |
| NCCL exposed ms/iter | **529.786** (reproduced) | **274.7** (published is 1.93× overstated) |
| NCCL hidden % | 4.4 | 7.6 |
| rank 0 all-busy ms/iter | 857.3 | 596.6 |
| rank 1 all-busy ms/iter | 740.6 | 478.5 |
| rank 7 all-busy ms/iter | 473.6 | 467.9 |
| per-rank NCCL-exposed pattern | rank 7 uniquely low (150.9 vs ~530) | ranks 1 **and** 7 low (161.8 / 150.9 vs 268.9–279.1) |

Also reproduced exactly (the clean per-iter family): iter wall p50 471.8 / mean 617.3;
allreduce_metrics share 44.56%; forward_backward 47.33%; untracked 5.94%.

### Cross-trace impact (same recomputation on every Phase 7 sqlite)

| trace | config / node | inside-window dropped time | NCCL exposed: window → kept (ms/iter) | per-rank compute: window → kept (ms/img) |
|---|---|---:|---|---|
| 75082 | bs128 gpu06 | 13,490 ms (26.9%) | 729.1 → 428.6 | 1.681 → 1.643 |
| 80547 | bs128 msp10 gpu02 | 0 | 705.9 → 705.9 | 1.635 → 1.635 |
| 80546 | bs160 gpu02 | 0 | 336.9 → 336.9 | 1.568 → 1.568 |
| 80545 | bs192 gpu02 | **38,753 ms (67.4%)** | **879.2 → 69.7** | **1.622 → 1.559** |
| 80711 | bs192 gpu03 | 15,386 ms (29.7%) | 529.8 → 274.7 | 1.592 → 1.564 |

Two traces are uncontaminated (80546, 80547) because their stalls either didn't happen or
stayed under the 10 s cap. That is itself a comparability trap: 80547's stalls (3.2–6.7 s;
kept-wall p90 3,176 ms, mean 1,044.8 ms per its `.phase7.json`) stayed **inside** both bases,
while 80545's 19–20 s stalls were dropped from one basis and kept in the other. Whether two
traces are comparable currently depends on whether their stalls happen to cross 10 s.

Note on the kept-only method: intervals are attributed by start time, so a kernel starting
inside a dropped stall and extending into kept time would be lost. Codex checked this edge
explicitly: **0 such NCCL kernels** in 80711, 80545, 80546, 75082 — the recomputed numbers are
unbiased here, but the fix should clip by overlap, not start membership (§7 F1).

## 2. Verdict table — the 9 items

| # | item | verdict | one-line basis |
|---|---|---|---|
| 1 | Heavy tails | **OK** (real skew) | Partition exact; tail = 3 real slow iters + one excluded 15.4 s stall; mean *understates* the tail |
| 2 | "dropped 2 out-of-band segment(s)" | **MISLEADING** | One drop was a legitimate 15.4 s stall (JSON tags it `dropped_slow_ms`); the .md hides the distinction; drop biases the skew being measured |
| 3 | Exposed-vs-hidden NCCL math | **BUG** | Interval logic correct; time basis mixed — 529.8 ms/iter published vs 274.7 honest |
| 4 | NVTX partition | **OK** | Σ(top-level)+untracked = wall exactly, every iter; shares reproduce |
| 5 | CUDA-graph attribution | **OK** | node granularity → per-kernel rows, sane timestamps; minor PID-join fragility |
| 6 | Iteration segmentation | **OK** | Head/tail edges handled; capture-end garbage NVTX correctly filtered |
| 7 | Rank↔device mapping | **MISLEADING** (mapping OK, values contaminated) | rank=cuda_dev verified; published table dominated by one stall event |
| 8 | H2D DMA section | **OK** | Pinned HtoD, 25.6 GB/s, 0% hidden all verified; minor window-basis caveats |
| 9 | Capture script | **OK** | Env propagation, NVTX wiring, steady-state window landing all verified |

## 3. Per-item evidence

### Item 1 — heavy tails: real, and actually understated
The NVTX partition is exact (see item 4), so the tail cannot be segmenter misattribution.
Per-iter anatomy of the slowest kept iters (rank-0 NVTX, ms):

| iter | wall | forward_backward | losses_backward | allreduce_metrics |
|---|---:|---:|---:|---:|
| 36 | 4,079.8 | 3,761.4 | 3,714.0 | 284.8 |
| 2 | 2,514.0 | 2,240.2 | 2,176.4 | 223.1 |
| 10 | 2,174.1 | 1,937.8 | 1,865.7 | 221.4 |

Mean wall excluding the top 3 = 493.8 ms (vs p50 471.8) — the 617.3 mean is three slow iters,
all with the CPU stuck inside `loss.backward()` (DDP grad all-reduce wait bunching there).
`losses_backward` mean excluding top 3 = 108.2 ms vs p50 89.3. On top of that the single
largest stall (15.4 s) was dropped from the wall stats entirely, so the published mean and p99
**understate** the true tail. Verdict OK: the skew is real; the caveat cuts the other way.

### Item 2 — the outlier filter drops a legitimate stall
`dropped_walls_ms` = [15,385.545, 201,882,930.564]; the script's own audit trail classifies the
first as `dropped_slow_ms` (real, physically plausible) and the second as `dropped_corrupt_ms`
(lines 329–331). Both classifications are correct. Two problems:
1. The .md renders only "dropped 2 out-of-band segment(s)" (render_md, line 518) — a reader
   cannot tell a garbage timestamp from the single worst straggler event in the capture.
2. Dropping the stall from wall stats while leaving its GPU time in the window-basis metrics
   is exactly the double distortion described in §1. The filter drops the very iters that
   carry the skew under study, then lets their kernels contaminate the per-iter averages of
   everything else.

### Item 3 — NCCL math: logic right, basis wrong (BUG)
Union and exposure re-derived independently: window basis gives union 554.098, exposed 529.786,
hidden 4.4%, sum/union 1.00 — byte-identical with the published numbers, so `merge_intervals`
(10 µs gap), `overlap_sweep`, and the union−overlap subtraction are all correct. Kept-only:
union 297.5, exposed 274.7, hidden 7.6%. The stall contributes 15,140 ms of NCCL union on
device 0, i.e. +257 ms/iter of the published 554. The "sum/union = 1.0×" concurrency check is
fine (NCCL channels serialized).

### Item 4 — NVTX partition: exact
Checked every kept iter: 0 iters where Σ(top-level ranges) > wall (the `max(0, …)` clamp at
line 348 never fired), 0 top-level range instances overhanging their iteration's end (range
attribution by start, line 340–344, never misassigns here — NVTX ranges on one CPU thread are
sequential). Shares re-derived: allreduce_metrics 44.56%, forward_backward 47.33%, untracked
5.94%; with schedule_update 0.01 + grad_clip 0.95 + optimizer_step 1.10 + ema_update 0.12 the
partition sums to ~100%. No double counting: nested (·) ranges are excluded from the partition
sum (`TOP_RANGES` only, line 347); `forward_backward` containing them is by design.
`allreduce_metrics` is a CPU-side wall share — the established "bill coming due" reading of it
is unchanged by this audit.

### Item 5 — CUDA-graph attribution: sound
`--cuda-graph-trace=node` yields per-kernel rows: 54.1% of dev-0 kernels (92,796 / 171,416)
have `graphNodeId` set. Timestamps are sane — 0 kernels with end < start, dev-0 kernel span
62.9 s, consistent with the 60 s capture + finalization. Kernels are attributed to iterations
by real replay timestamps, so no smearing. One fragility: `device0_pid()` (lines 174–189)
joins kernel→runtime on `correlationId` without constraining process; correlationIds collide
across the 8 ranks, so the join fans out (326,475 joined rows for 171,416 dev-0 kernels;
177,393 for the graph-kernel subset). Majority vote still picked the correct PID (1003329 =
dev-0's `globalPid`), verified. Fix F3 hardens this.

### Item 6 — segmentation edges: handled
Head: ranges open when the capture starts are absent from the trace; the first recorded fb
start is a complete one, and `skip_iters=5` discards warm-in anyway. Tail: the 14 NVTX ranges
(7 ranks × forward_backward + losses_backward) still open at capture end get end-of-session
sentinel timestamps ~201,937 s into a 63 s trace (one even has start > end) — these create the
corrupt final segment, which the `0 < wall` / cap filter correctly removes. The final partial
iteration is inherently dropped by fb-start→next-fb-start construction. 67 starts → 66
segments → 61 after skip → 59 kept: reproduced exactly.

### Item 7 — rank↔device mapping: correct; the table, not
Mapping verified three ways: `[RANKTOPO]` stderr shows `rank=local_rank=cuda_dev` for every
rank; kernel `globalPid` per device is consecutive PIDs 1003329–1003336 for dev 0–7; the eight
NVTX fb tids decode to those same PIDs. So row *d* of the per-rank table really is rank *d*.
The **values** are window-basis: rank 7's published 473.6 all-busy vs ~857 for others is
dominated by the single stall event (others spin 15.37 s, rank 7 idles). Kept-only, the
steady-state picture is: ranks 0,2–6 busy 587–597 ms/iter, ranks 1 **and** 7 busy 468–478;
NCCL-exposed 269–279 vs 162/151. "Rank 7 uniquely spins least" is a one-event artifact; this
trace's steady state has two late-arriving ranks. The same contamination is worse in 80545,
where the published "ranks 6/7 exposed 65/107 ms vs ranks 0–5 ~880 ms" collapses kept-only to
**all ranks 46–71 ms** — the ~880 was two ~19.4 s stall events spread over 47 iters
(2 × 19,400 / 47 ≈ 825 ms/iter).

### Item 8 — H2D: classification and bandwidth verified
The large per-iter copies are `copyKind=1` (HtoD) with `srcKind=1` (**PINNED**): 360 large
pinned H2D copies in the window, 20,066.6 MB / 785.3 ms = **25.6 GB/s** — matches the published
25.5, and is plausible pinned-PCIe-gen5 throughput. Not NVLink and not pageable: pageable HtoD
in the window is 180 copies totaling ~0 MB; DtoD traffic (`copyKind=8`, 586 GB) is a separate
kind and never enters `h2d_ops()` (line 241 filters `copyKind=1`). "0% hidden" verified
directly: H2D interval overlap with non-NCCL compute = 0.0 ms **and** with NCCL = 0.0 ms — the
copies land while the GPU runs nothing else, so the "reclaimable by copy-stream double-buffer"
framing is fair. Caveats: `h2d_exposed_ms_per_iter` (13.328) is window-basis (9 stall-interval
copies included — also why it slightly exceeds the per-iter DMA mean of 13.04), and the
top-sizes counts (60) span the window, not the 59 kept iters. Both are cosmetic here.

### Item 9 — capture script: sound
- **Env propagation**: `DINOV3_PERRANK_DIAG` is exported before `torchrun` (capture script line
  98); single-node, all 8 local ranks inherit — confirmed by `[RANKDATA]`/`[RANKTOPO]` lines
  from all 8 ranks in `p7-nsys-80711.err`. `METRICS_SYNC_PERIOD` is not an env dependency at
  runtime: it's passed as the `train.metrics_sync_period` config override (line 170), identical
  argv on every rank; the key exists (`ssl_default_config.yaml:83`, consumed at `train.py:609`).
- **NVTX via `--profiling`**: wired end-to-end — `train.py:932–936` sets `cfg.train._profiling`,
  `train.py:567–587` builds `make_nvtx` and hands it to the model, ranges emitted at
  `train.py:649/660/684/710/746/748`.
- **Window landing**: all five successful captures landed in steady state — 44–85 clean
  iterations each (75082: 44, 80545: 47, 80546: 85, 80547: 53, 80711: 59). The three failures
  (71041/71682/74635) predate the current delay/trace settings, are documented in the script
  header, and left no sqlite. The `|| NSYS_RC=$?` guard correctly keeps `set -e` from skipping
  export/analysis.

## 4. Which Phase 7 conclusions are affected

**Needs restating:**
1. **7.7 Finding 1 — per-image compute U-shape (sweet spot bs160)**: the bs192 anchor (80545)
   is stall-inflated (two dropped stall iterations' compute in the numerator over 47 kept
   iters). Corrected per-image compute: 1.635 → 1.568 → **1.559** ms/img (80545 kept-only),
   corroborated by 80711 at 1.564. Within this evidence set the series is monotonically
   decreasing — **the U-shape is gone**, and the GEMM wave-quantization hypothesis loses its
   motivating observation. bs192 deserves one clean confirmation capture before the corrected
   number is treated as final.
2. **7.3 "metric-gate −3.5% exposed NCCL (729→706)"**: invalid as computed — it compared a
   stall-inflated number (75082) against an uncontaminated one (80547). Kept-only the pair is
   428.6 vs 705.9; per Codex (accepted): call the two traces **incomparable** (different days,
   cross-capture stall variance dominates), not a reversed conclusion. The stronger 7.8a
   within-trace result (metric-gate is not an E2E lever) is unaffected and already superseded
   this number.
3. **Per-rank straggler tables** (memory: "rank ROTATES r6@bs128/r7@bs160/r6+7@bs192"): the
   bs192 leg (80545) was a 2-event artifact (§3 item 7). 75082 kept-only still shows rank 7
   lowest (387.9); 80547 rank 6 lowest (360.5); 80546 rank 7 lowest (178.2); 80711 ranks 1+7.
   The rotation observation partially survives but should be restated from kept-only numbers.

**Survives with reduced magnitude:**
4. **"Residual is COMMS" (75082)**: NVTX shares (32.6% allreduce_metrics) are per-kept-iter and
   stand. Absolute exposed NCCL drops 729 → 428.6 ms/iter kept-only — still ~54% of the 794 ms
   median wall, so the direction holds at bs128; the magnitude was overstated.
5. **7.7 Finding 2 — residual grows 415→445→506 ms**: derived from soak throughput
   (67639/67153/67154), harness-independent; the arithmetic stands. Its *attribution* to
   per-iter f32 loss-side collectives (the 7.4 Sinkhorn promotion) is weakened, not disproven:
   in the bs192 traces steady-state exposed NCCL is small-to-moderate (69.7 ms in 80545, 274.7
   in 80711) and the residual mass shows up as **episodic multi-second arrival stalls** (two
   events = 67% of 80545's window). 80546 (bs160) still carries 336.9 ms/iter exposed with no
   stall, and 80547 has heavy sub-10 s tails, so the honest statement is: the bs192 evidence
   suggests episodic arrival stalls are a major component of the residual there; current nsys
   evidence no longer cleanly supports Sinkhorn as the dominant *growing* term without
   separating episodic stalls from steady comms. (The dino/ibot_sinkhorn NVTX shares measured
   in 80711 are 0.27%/0.31% — also pointing away from Sinkhorn as the big lever.)

**Stands regardless of the harness:**
6. **Rank-7 straggler (host-side arrival)**: the `[RANKDATA]` stderr evidence is independent of
   the analyzer (rank 7 data_time 0.18 s at iters 660/670 vs ~0.3 ms on other ranks; the
   handoff's 0.37–0.42 s at 690/700/710) and the 15.4 s stall event itself — rank 7 idle while
   7 ranks spin in one f32 all-reduce — is raw trace fact, not a computed metric. The harness's
   per-rank *corroboration* survives in weakened form (kept-only rank 7 still lowest exposed,
   rank 1 comparable in 80711).

## 5. Independent re-derivations (summary of method)

Scratch scripts (session scratchpad, `audit1`–`audit5`): fresh implementations of interval
merge, sweep-line overlap, and per-iter attribution written without reference to the analyzer's
internals, run read-only against the sqlite exports. Verification sequence: first reproduce
every published number exactly under the script's own definitions (segmentation counts, wall
percentiles, NVTX shares, NCCL union/exposed, per-rank busy), then recompute under the
corrected kept-only basis. Headline numbers independently re-derived: iter-wall p50/mean,
allreduce_metrics/forward_backward/untracked shares, NCCL union/exposed/hidden, per-rank
busy/exposed (×8), H2D bytes/bandwidth/overlap. Codex independently reproduced the
segmentation (59 kept, dropped [15,385.5, 2.019e8]), the window/kept walls (51,808.9 /
36,423.4), and the five-trace NCCL exposed recomputation to the same three decimals.

## 6. Codex review outcome

`codex exec --model gpt-5.5 -c model_reasoning_effort=high --sandbox read-only` on the draft
findings. Verdict: "the draft's core bug is real, and the main Phase 7 comparisons that used
window-level CUDA totals need restating." All 9 verdicts upheld. Four corrections, each
self-verified before acceptance:
1. Root-cause wording: "mixed time basis / basis selection", not "normalization" — accepted.
2. Kept-only clipping by start membership could drop stall-starting kernels that extend into
   kept time; Codex counted **0** such NCCL kernels in 80711/80545/80546/75082, so the audit
   numbers are unbiased, but fix F1 must clip by overlap — accepted.
3. "80547 is clean" only w.r.t. this bug — verified: its kept walls have p90 3,176 / mean
   1,044.8 ms (`.phase7.json`), heavy sub-cap tails; not a stable baseline — accepted, wording
   adjusted in §4.2.
4. Join-fanout figure: 177,393 is the graph-kernel subset; all-kernel fanout is 326,475 /
   171,416 — verified by direct query; both now labeled (§3 item 5).
Codex also flagged overreach in "the mechanism shifts to episodic stalls" — softened to the
bs192-scoped statement in §4.5 — and cautioned against reading the 7.3 kept-only pair as
"gating hurts"; stated as incomparable instead.

## 7. Proposed fixes (described, not applied)

- **F1 (core)**: compute every window-basis metric over the **union of kept iteration
  intervals**: clip kernel/memcpy intervals to kept iters **by overlap** (splitting intervals
  that cross a kept/dropped boundary), then merge and sweep. Applies to NCCL union/exposed/
  hidden, per-rank compute/busy/exposed, kernel classes, top kernels, H2D union/exposed/
  top-sizes. After F1, `all-busy ≤ mean wall` and `NCCL exposed ≤ wall` hold by construction.
- **F2**: surface `dropped_slow_ms` in the .md — a stall-accounting line per dropped-slow
  segment (wall, its per-rank busy and NCCL content, which rank was idle), and report both the
  kept mean and a stall-inclusive mean wall so the tail is never silently truncated.
- **F3**: harden `device0_pid()`: constrain the correlation join to the same process
  (`(r.globalTid >> 24) & 0xFFFFFF == (k.globalPid >> 24) & 0xFFFFFF`) to remove cross-rank
  correlationId collisions.
- **F4 (cosmetic)**: count `h2d_top_sizes` over kept iters; keep the empirical
  H2D-vs-NCCL-overlap check (it validated the double-buffer framing here).
- **F5 (comparability note, doc or code)**: the 10 s / 6×median cap makes cross-trace
  comparisons depend on whether stalls cross the cap (80547's 3–6.7 s stalls kept vs 80545's
  19–20 s stalls dropped). With F1+F2 in place, consider also emitting stall-inclusive
  aggregate metrics so traces with different stall magnitudes stay comparable.

## Short version

The analyzer's per-iteration NVTX accounting is exact, and the capture script is sound. The
one real bug is a mixed time basis: NCCL exposure, the per-rank table, kernel classes, and H2D
exposure are window totals divided by kept-iteration count, so any multi-second stall that the
outlier filter drops still contaminates them. In 80711 that inflates "NCCL exposed" from an
honest 275 to the published 530 ms/iter and manufactures the "rank 7 uniquely spins least"
table; in 80545 it manufactured both the bs192 per-image compute regression (the U-shape) and
the ranks-6/7 880 ms exposure story. The soak-derived residual-growth numbers and the
stderr-based rank-7 data_time evidence are independent of the harness and stand.
