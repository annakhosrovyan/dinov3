# THP / Hugepages Impact on BCM11 — Scoping Experiment Design

**Date:** 2026-07-20
**Status:** Design (scoping tier), pending final user review. Reviewed by GPT-5.6 Sol twice (high
effort, read-only): design review + workload-taxonomy review. Taxonomy-review fixes applied to
§Step 1, §Step 2, §6, §7, §8, §9. Companion: `~/docs/slurm/cluster-workload-taxonomy-2026-07-20.md`.
**Cluster:** BCM11 — Supermicro H100 nodes (gpu01–08), Weka storage.
**Workload:** DINOv3 satellite fork (`/home/adovlatyan/dinov3-performance-optimizations/dinov3`).

---

## 1. Question

Does Transparent Huge Pages (THP) mode change DINOv3 training throughput on BCM11, and by how
much — measured against the AI Systems Performance Engineering Ch03 estimate of ~3–5% from
hugepages? This is a **scoping** study: a directional, single-node answer that decides whether a
full powered study is worth the node-hours. It is explicitly **not** a cluster-wide claim.

## 2. Why this is easy to get wrong

THP is a **host-memory** knob: it cuts CPU TLB miss rate and first-touch page-fault volume by
backing anonymous mappings with 2 MB pages instead of 4 KB. It only moves wall-clock when
host-memory-bound work sits on the **exposed critical path**. DINOv3 throughput is gated by GPU
compute (bs128 champion) or by CPU *scheduling* contention (bs192 loader starvation), not by
host-memory translation cost. So an effect can be real at the mechanism level (fewer dTLB misses)
yet invisible at the wall-clock level. The design measures both layers so a null is *explained*,
not merely observed.

## 3. Established facts (cited)

| Fact | Source |
|---|---|
| Champion config = DDP **bs128 + cudagraphs** (~1,640 img/s run-avg; 2,116 steady on job 77672) | `docs/phase7_perf_plan.md:48,61` |
| MFU flat ~20–21% across bs128/160/192; 160→192 diminishing returns | `docs/phase7_perf_plan.md:4,269-270` |
| bs192 residual = **loader starvation** (job 80711): rank7 data_time 0.37–0.42 s vs ~0.3 ms; 8 ranks × 24 workers = 192 unpartitioned DataLoader workers + 8 mains contending for 192 CPUs | `docs/phase7_perf_plan.md:38,417-419` |
| §7.9 fix: **w16 no-slice** ships −11%/−6% median/mean at bs160, −17% median at bs192. NUMA slicing **rejected** as default — under current IRQ placement it turns a rotating loser into a fixed straggler | `docs/phase7_perf_plan.md:39` |
| THP state was **never captured** in phase7 diagnostics (`[RANKTOPO]` records CPU affinity only). Prior-session "THP=madvise uniform" is **unverified** | phase7 grep, this session |
| User has **root + exclusive-node** access on gpuNN → can flip `/sys/kernel/mm/transparent_hugepage/enabled` between legs | confirmed this session |

## 4. Design decisions and what was rejected

**THP-mode is the lever; explicit hugepages is not a comparator.** hugetlbfs / `nr_hugepages`
measure a different thing (a pre-reserved pool the app must explicitly map) and are not a drop-in
comparator for THP behavior. Sol confirmed THP is the correct first operational knob. Explicit
hugepages are out of scope for the scoping study.

**Primary end-to-end arm = bs160 w16** (§7.9 shipped worker-budget fix), not bs128. bs128 hides
the loader behind GPU compute, so THP has no path to the critical path there — it would only ever
produce a null. bs160 w16 keeps a real, exposed loader residual where host-memory cost can plausibly
surface, while using the worker-budget fix so we are not measuring on top of the raw oversubscription
pathology. **bs192 w16** is an optional second arm (larger residual, but noisier episodic stalls).

**Rejected for the scoping tier** (deferred to escalation): full statistical power (pilot → powered
rep count), equivalence testing for a formal null, multi-node confirmation, and the `THP × worker-count`
interaction stress arm (w24). NUMA slicing is rejected outright per §7.9.

## 5. The scoping study

### Step 0 — Baseline + full kernel-state capture (read-only)
One probe job on the target node. Record, before touching anything:
- `transparent_hugepage/{enabled,defrag}`, per-size THP `enabled`, `shmem_enabled`
- khugepaged: `defrag`, `scan_sleep_millisecs`, `pages_to_scan`, `alloc_sleep_millisecs`
- `/proc/meminfo` (`AnonHugePages`, `HugePages_*`), `nr_hugepages`, `numa_balancing`
- kernel version, page sizes, swap state, `/proc/buddyinfo`, compaction counters
- confirm a root write to the `enabled` knob is permitted from this context

This finally records THP state in the diagnostics, which phase7 never did.

### Step 1 — Hugepage UPTAKE audit (HARD GATE for the DINOv3 arm only)
Before any timing. madvise-mode THP only backs `MADV_HUGEPAGE` regions; PyTorch/glibc allocations
may never opt in, so `madvise`→`always` might not change the backing of the memory that matters.
Under each mode (`never`/`madvise`/`always`), measure hugepage coverage of the **DataLoader-worker
RSS and the pinned-buffer pool**.
- Do **not** rely on `smaps`/`meminfo` `AnonHugePages` alone — it reports PMD-sized (2 MB) anonymous
  THP and misses smaller multi-size THP (mTHP). Also read per-size counters under
  `/sys/kernel/mm/transparent_hugepage/hugepages-*/stats`, and for the shared DataLoader buffers
  read `ShmemPmdMapped` / `FilePmdMapped` (`smaps` per-mapping and `/proc/meminfo`). A "no coverage"
  reading from `AnonHugePages` alone is not proof of no uptake.
- Compare `AnonHugePages`+mTHP coverage vs `Rss` per relevant process across modes.

**If coverage does not materially change across modes, STOP the DINOv3 timing arm and report that.**
This hard stop bounds only the DINOv3 experiment. It does **not** stop the separate cluster-workload
/ CPU-chemistry investigation (see the taxonomy doc, §11) — a null on DINOv3's loader says nothing
about the CPU-side docking/QM jobs where THP is more likely to bite.

### Step 2 — Slice A: end-to-end AB/BA (directional)
- Config: **bs160 w16** DDP + cudagraphs (champion pipeline, §7.9 worker budget). Optional: bs192 w16.
- **Estimand is "this configuration."** bs160 w16 is deliberately more loader-exposed than the
  production bs128 champion. Every result is stated as "this DINOv3 config and pipeline," never
  "DINOv3 generally" and never "the cluster" — see the proxy-scope limits in the taxonomy doc (§11).
- THP `madvise` vs `always`. **Randomized AB/BA** legs (not deterministic ABAB), a handful of pairs
  (~4–6) on **one exclusive node**.
- Each leg = a **freshly restarted** training process (the leg is the experimental unit, not the
  iteration). ~10–15 min steady-state window; begin measurement only after compile warmup **and**
  hugepage coverage reach a stable regime.
- Before every leg: identical **unmeasured prewarm** over the same shards to equalize Weka page-cache
  warmth. Fresh worker processes after every mode change.
- Capture before+after each leg: `/proc/buddyinfo`, compaction/THP counters, khugepaged/kcompactd CPU
  time, softirq CPU, worker CPU utilization, bytes read.

### Slice B — Mechanism companion (labeled: NOT a predictor)
GPU-idle or paced consumption of DINOv3's own `data/loaders.py` + `transforms.py` on real Weka data,
to explain *why* Slice A moved or didn't. Sol's constraint: a GPU-idle loader benchmark is a
mechanism/capacity demo — a 10% loader-only gain can be 0% end-to-end.
- Preserve the real topology: 8 rank-like consumers, same worker count, multiprocessing/shared-memory
  path, collate path, pin-memory behavior, prefetch depth, CPU affinity.
- Add a **paced-consumer** mode that pulls batches at the observed training cadence; measure
  queue-empty probability and tail (p95) fetch latency, not just saturated throughput.
- Bind ranks+workers **before** first-touch of loader buffers (NUMA first-touch matters).
- `perf stat`: dTLB-load-misses, page-faults — the mechanism endpoints.
- Any training-benefit claim requires confirmation in the full loop (Slice A), never from Slice B alone.

## 6. Variance and confound controls
- Same physical node for all legs of a comparison; randomized AB/BA order.
- Pinned BLAS threads for the microbench: `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`.
- Fixed seed and data order; identical prewarm per leg.
- khugepaged left **untouched** in Slice A — its CPU cost *is* part of `always`. Pinning it is a
  separate mechanistic experiment, not the operational one.
- Report paired differences with a confidence interval and a paired test — never single-run point diffs.
- **Unconditional restore.** The harness records every THP/defrag/khugepaged knob it reads in Step 0
  and installs a trap/epilog that restores all of them on normal exit, cancellation (SIGTERM/SIGINT),
  and failure. The node must never be left in a mutated THP state after any leg, aborted or not.
- **Keep a cold-start measurement.** Steady-state wall img/s stays primary, but because short jobs
  dominate the cluster by count (73% finish <10 min — taxonomy §Duration), also record a secondary
  cold-start/initialization metric (process start → first stable step) under each mode. THP effects
  on first-touch fault volume show up there even when steady-state is flat.

## 7. Metrics and analysis
- **Primary (Slice A):** steady-state wall img/s. Report the paired `always − madvise` difference in
  **log throughput** with a CI, plus per-rank `data_time`.
- **Mechanism (Slice B):** AnonHugePages+mTHP coverage, dTLB-load-misses, page-faults, p50/p95 fetch
  latency, queue-empty probability.
- **Harm-side counters (both slices, per leg):** cgroup `memory.peak` / `memory.current` /
  `memory.events{,.local}` / `memory.stat anon_thp`, THP split and allocation-fallback counters,
  reclaim/compaction stall counters, `kcompactd`/`khugepaged` CPU time, `/proc/PID/numa_maps` +
  per-node `numastat`, and p99 iteration latency. A cluster-default decision is bounded by these,
  not by mean throughput alone (taxonomy §"THP implications", harm model).
- Pre-register the single primary contrast and window before running; report **all** attempted arms,
  including uptake failures and negatives (guards against selective interpretation).

## 8. Decision rules
- **Uptake gate fails** (Step 1): report "THP mode does not change DINOv3 memory backing on this node"
  and stop the DINOv3 arm. Cheapest possible outcome. (Does not close the CPU-chemistry question.)
- **Slice A paired difference is small and its CI crosses 0:** report a **directional null** — the
  Ch03 3–5% claim does not visibly transfer to this GPU/scheduling-bound loop. Be explicit about what
  this is *not*: a CI that crosses zero is **not** an equivalence result. Claiming "THP has no effect"
  requires a formal equivalence test (TOST) with a pre-declared margin (e.g. ±3%) where the **entire**
  confidence interval falls inside ±margin. At the scoping tier we do not have the replication to run
  TOST, so the honest statement is "no effect resolved," not "no effect exists." Formal equivalence is
  an escalation item (§9).
- **Slice A shows a resolvable directional effect:** escalate to the powered study (§9) to confirm and
  quantify.

## 9. Escalation path (only if scoping warrants)
Sol's full redesign, deferred until scoping justifies the node-hours:
1. Pilot ≥4 randomized pairs → estimate paired-difference SD → power the rep count
   (2% SD → ~4 pairs, 3% → ~8, 5% → ~22 for 80% power at 5%).
2. Formal **equivalence test** (TOST) for the null with a pre-declared ±3% margin — the entire CI
   must fall inside ±margin. This is what §8 defers.
3. `THP × worker-count` interaction arm (bs192 w24 as a stress arm).
4. **Multi-node** (2–3 nodes) before any "BCM11 cluster" claim.
5. **Workload panel + canary**, not DINOv3 alone. The workload taxonomy (§11) shows DINOv3 proxies
   only the vision-SSL slice (~21% of GPU-hours). A cluster-default decision needs a panel spanning
   CPU chemistry (docking/QM), GPU chemistry (NEP/VASP), LLM/VLM, and inference; aged-memory and
   co-tenancy tests (several 1-GPU jobs sharing a node); numeric no-harm thresholds; a canary rollout
   with rollback + baseline restoration; and a comparison against the safer `madvise` + per-app
   `MADV_HUGEPAGE` opt-in. `always` must beat opt-in, not merely beat `madvise` on one benchmark.

## 10. Risks / open items
- Uptake gate may fail immediately (plausible under madvise) — accepted; it is the cheapest useful result.
- bs192 episodic multi-second stalls inflate variance; bs160 chosen primary to keep variance resolvable.
- `drop_caches` is **not** used as a reset — it creates an artificial cold-start and may not reset
  Weka-side caching. Prewarm-to-warm is the estimand instead.
- Single-node scope is a deliberate limitation of the scoping tier, not an oversight.

## 11. References
- **`~/docs/slurm/cluster-workload-taxonomy-2026-07-20.md`** (+ `.py`) — 30-day accounting-based
  workload characterization. Establishes what DINOv3 proxies (vision-SSL ~21% of GPU-hours) and what
  it does not (CPU chemistry, LLM host-memory, memory-tight configs), the corrected OOM picture
  (concentrated campaigns, not a baseline), and the decision framing for a cluster default. This
  spec's proxy-scope and escalation claims are grounded there.
- `docs/phase7_perf_plan.md` (§7.7, §7.8c, §7.9) — champion config, loader-starvation attribution, w16 fix.
- `docs/archive/perf_bottleneck_ledger.md` — earlier bs256 ledger (superseded).
- AI Systems Performance Engineering, Ch03 — hugepages/THP ~3–5% estimate (KB:
  `knowledge-base/ai_systems_perf_engineering/`).
- GPT-5.6 Sol reviews, 2026-07-20 (read-only, high effort): design review (findings in §4–§9) and
  taxonomy review (spec fixes in §Step 1, §Step 2, §6, §7, §8, §9).
