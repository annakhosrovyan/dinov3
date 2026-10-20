#!/usr/bin/env python3
"""Phase 7 residual-attribution report for an Nsight Systems SQLite export.

Answers the Phase 7.1 questions (docs/phase7_perf_plan.md §7.1) with one fixed
methodology, so traces from different configs (champion baseline, 7.2 autotune
isolation, 7.3 autotune+graphs, bs differential pairs) are directly comparable:

  1. Per-iteration wall breakdown by NVTX range (rank 0):
     schedule_update / forward_backward (H2D_transfer, teacher_fwd, student_fwd,
     losses_backward) / grad_clip / allreduce_metrics / optimizer_step / ema_update,
     plus the UNTRACKED residual (dataloader wait, logging, Python).
  2. GPU busy vs gap inside each iteration (device 0) — the launch-overhead residual.
  3. NCCL exposure: how much NCCL time is NOT hidden behind compute (device 0).
  4. Kernel-time taxonomy (matmul / attention / nccl / elementwise / ...).

Outputs <trace>.phase7.md (human) and <trace>.phase7.json (machine — feed two
JSONs to --diff for a side-by-side verdict table).

Usage:
  python scripts/profiling/phase7_nsys_report.py trace.sqlite [--label NAME]
  python scripts/profiling/phase7_nsys_report.py --diff a.phase7.json b.phase7.json [--out d.md]

Stdlib-only by design (runs on GPU nodes inside the capture job).
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sqlite3
import sys
from pathlib import Path

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000

# Top-level NVTX ranges emitted once per training iteration (train.py).
TOP_RANGES = [
    "schedule_update",
    "forward_backward",
    "grad_clip",
    "allreduce_metrics",
    "optimizer_step",
    "ema_update",
]
# Nested inside forward_backward (ssl_meta_arch.py).
# dino_sinkhorn / ibot_sinkhorn (Phase 7.8d) are nested inside teacher_fwd; they bound the
# loss-side Sinkhorn collectives (>=3 blocking f32 all_reduces each, K proportional to batch).
NESTED_RANGES = [
    "H2D_transfer",
    "teacher_fwd",
    "student_fwd",
    "gram_fwd",
    "losses_backward",
    "dino_sinkhorn",
    "ibot_sinkhorn",
]
ALL_RANGES = TOP_RANGES + NESTED_RANGES

KERNEL_CLASSIFIERS: list[tuple[str, re.Pattern[str]]] = [
    (
        "nccl",
        re.compile(r"(?i)(ncclKernel|ncclDevKernel|nccl_|allreduce|allgather|reducescatter|all_gather|all_reduce)"),
    ),
    ("attention_flash", re.compile(r"(?i)(flash_(fwd|bwd)|fmha|mha_fwd|mha_bwd|attention.*kernel)")),
    # nvjet_* are the H100 cuBLASLt (Hopper "jet") GEMM kernels — without them most GEMM
    # time lands in "other" (job 75082: matmul_gemm read 0.27 ms/iter while nvjet kernels
    # summed to ~70 ms/iter). Keep this matcher ahead of elementwise so fused GEMM-epilogue
    # names resolve to GEMM.
    (
        "matmul_gemm",
        re.compile(
            r"(?i)(gemm|cutlass|matmul|sgemm|hgemm|bgemm|cublas|nvjet|ampere_|sm80_|sm90_|triton_mm|triton_tem)"
        ),
    ),
    ("conv", re.compile(r"(?i)(conv|implicit_gemm|cudnn)")),
    ("layernorm_norm", re.compile(r"(?i)(layer_norm|rms_norm|layernorm|batchnorm)")),
    ("optimizer_foreach", re.compile(r"(?i)(adam|sgd|foreach|fused_adam|multi_tensor)")),
    (
        "elementwise_triton",
        re.compile(
            r"(?i)(triton_poi|triton_red|triton_per|elementwise|gelu|silu|relu|softmax|dropout|cat_kernel|copy_|to_copy|vectorized)"
        ),
    ),
    # indexFuncLargeIndex = the index_add scatter from the block.py cudagraph-capture fix.
    ("reduction", re.compile(r"(?i)(reduce_kernel|sum_kernel|mean_kernel|argmax|topk|sort|index_|indexfunc|scatter)")),
]


def classify_kernel(name: str) -> str:
    for label, rx in KERNEL_CLASSIFIERS:
        if rx.search(name):
            return label
    return "other"


def percentiles(xs: list[float]) -> dict | None:
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return {
        "n": n,
        "min": xs[0],
        "p50": xs[n // 2],
        "p90": xs[min(n - 1, int(0.9 * n))],
        "p99": xs[min(n - 1, int(0.99 * n))],
        "max": xs[-1],
        "mean": sum(xs) / n,
    }


def merge_intervals(intervals: list[tuple[int, int]], merge_gap_ns: int = 10_000) -> list[list[int]]:
    if not intervals:
        return []
    intervals.sort()
    merged: list[list[int]] = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + merge_gap_ns:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged


def overlap_sweep(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Total ns where at least one interval of `a` overlaps at least one of `b`."""
    events = (
        [(s, "as") for s, _ in a] + [(e, "ae") for _, e in a] + [(s, "bs") for s, _ in b] + [(e, "be") for _, e in b]
    )
    events.sort()
    ao = bo = 0
    last = events[0][0] if events else 0
    ovlp = 0
    for t, ev in events:
        if ao > 0 and bo > 0:
            ovlp += t - last
        if ev == "as":
            ao += 1
        elif ev == "ae":
            ao -= 1
        elif ev == "bs":
            bo += 1
        else:
            bo -= 1
        last = t
    return ovlp


def clip_to_kept(intervals: list[tuple[int, int]], kstarts: list[int], kends: list[int]) -> list[tuple[int, int]]:
    """Clip each (s, e) to the union of kept iteration intervals BY OVERLAP.

    F1 (audit §7): window-basis metrics must be computed over the union of kept
    iteration intervals, not over the raw [first_start, last_end] window. A multi-second
    stall that the outlier filter drops sits *inside* that window, so its kernels land in
    the numerator while contributing nothing to the kept-iter denominator (the 1.93× bug).

    `kstarts`/`kends` are the sorted, disjoint kept iteration bounds. An interval that
    crosses a kept/dropped boundary is split, so this clips by overlap (not by start
    membership) — the edge Codex flagged (a kernel starting in a dropped stall and
    extending into kept time is correctly attributed only for its kept portion).
    """
    if not kstarts:
        return []
    out: list[tuple[int, int]] = []
    n = len(kstarts)
    for s, e in intervals:
        j = bisect.bisect_left(kends, s)
        while j < n and kstarts[j] < e:
            cs = s if s > kstarts[j] else kstarts[j]
            ce = e if e < kends[j] else kends[j]
            if ce > cs:
                out.append((cs, ce))
            j += 1
    return out


def has_table(cur: sqlite3.Cursor, name: str) -> bool:
    return bool(cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)).fetchone())


def table_columns(cur: sqlite3.Cursor, name: str) -> set[str]:
    return {r[1] for r in cur.execute(f"PRAGMA table_info({name})").fetchall()}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def load_nvtx_ranges(cur: sqlite3.Cursor) -> list[tuple[int, int, int, str]]:
    """Return (start, end, globalTid, text) for all named push/pop ranges."""
    if not has_table(cur, "NVTX_EVENTS"):
        return []
    cols = table_columns(cur, "NVTX_EVENTS")
    text_expr = "e.text"
    join = ""
    if "textId" in cols:
        text_expr = "COALESCE(e.text, s.value)"
        join = "LEFT JOIN StringIds s ON s.id = e.textId"
    rows = cur.execute(
        f"""
        SELECT e.start, e.end, e.globalTid, {text_expr} AS name
        FROM NVTX_EVENTS e {join}
        WHERE e.end IS NOT NULL AND name IS NOT NULL
        """
    ).fetchall()
    return [(int(s), int(e), int(tid), str(name)) for s, e, tid, name in rows if name in ALL_RANGES]


def device0_pid(cur: sqlite3.Cursor) -> int | None:
    """PID of the rank driving device 0, via kernel correlationId -> runtime globalTid."""
    if not (has_table(cur, "CUPTI_ACTIVITY_KIND_KERNEL") and has_table(cur, "CUPTI_ACTIVITY_KIND_RUNTIME")):
        return None
    # F3 (audit §7): correlationIds collide across the 8 ranks, so an unconstrained
    # kernel->runtime join on correlationId alone fans out cross-process (326k joined rows
    # for 171k dev-0 kernels). Constrain the join to the same process by comparing the PID
    # field (high bits of globalTid/globalPid) so only the rank driving device 0 matches.
    kcols = table_columns(cur, "CUPTI_ACTIVITY_KIND_KERNEL")
    same_proc = ""
    if "globalPid" in kcols:
        same_proc = "AND ((r.globalTid >> 24) & 0xFFFFFF) = ((k.globalPid >> 24) & 0xFFFFFF)"
    row = cur.execute(
        f"""
        SELECT r.globalTid, COUNT(*) AS c
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
        WHERE k.deviceId = 0 {same_proc}
        GROUP BY r.globalTid ORDER BY c DESC LIMIT 1
        """
    ).fetchone()
    if not row:
        return None
    return (int(row[0]) >> 24) & 0xFFFFFF


def device_intervals(
    cur: sqlite3.Cursor, device_id: int, t0: int, t1: int, nccl_only: bool | None = None
) -> list[tuple[int, int]]:
    """Kernel (+memcpy/memset) intervals on a device OVERLAPPING [t0, t1].
    nccl_only=True -> only NCCL kernels; False -> only non-NCCL; None -> all (incl. memcpy).

    F1: select by overlap (`end>t0 AND start<t1`), not full containment, so an interval that
    crosses the outer window boundary still reaches `clip_to_kept()` and is trimmed to its kept
    portion instead of being dropped whole. (Interior kept/dropped crossings are handled by the
    clipper; this widens the outer edge so nothing is lost before clipping.)"""
    out: list[tuple[int, int]] = []
    if has_table(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        rows = cur.execute(
            """
            SELECT k.start, k.end, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            WHERE k.deviceId=? AND k.end>? AND k.start<? AND k.end IS NOT NULL
            """,
            (device_id, t0, t1),
        ).fetchall()
        for s, e, name in rows:
            is_nccl = bool(name) and ("nccl" in name.lower())
            if nccl_only is None or nccl_only == is_nccl:
                out.append((int(s), int(e)))
    if nccl_only is None:
        for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            if has_table(cur, table):
                out.extend(
                    (int(s), int(e))
                    for s, e in cur.execute(
                        f"SELECT start, end FROM {table} WHERE deviceId=? AND end>? AND start<? AND end IS NOT NULL",
                        (device_id, t0, t1),
                    )
                )
    return out


# CUPTI copyKind enum: 1=HtoD, 2=DtoH, 8=DtoD (the three that appear in these traces).
H2D_COPY_KIND = 1


def h2d_ops(cur: sqlite3.Cursor, device_id: int, t0: int, t1: int) -> list[tuple[int, int, int]]:
    """(start, end, bytes) for Host->Device memcpys on a device within [t0, t1].

    These are the per-iteration input-batch copies (collated crops + masks) issued by
    `.cuda(non_blocking=True)` in ssl_meta_arch.py. The NVTX `H2D_transfer` range around
    those calls measures only CPU *enqueue* time and badly understates the real DMA — the
    actual transfer lands on the GPU copy engine and is only visible here (job 75082:
    NVTX p50 0.327 ms vs real DMA ~5.9 ms/iter). This is the ground truth for H2D cost.
    """
    if not has_table(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        return []
    cols = table_columns(cur, "CUPTI_ACTIVITY_KIND_MEMCPY")
    kind_filter = "AND copyKind=?" if "copyKind" in cols else ""
    params: tuple = (device_id, t0, t1) + ((H2D_COPY_KIND,) if kind_filter else ())
    byte_expr = "bytes" if "bytes" in cols else "0"
    rows = cur.execute(
        f"""SELECT start, end, {byte_expr} FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE deviceId=? AND end>? AND start<? AND end IS NOT NULL {kind_filter}""",
        params,
    ).fetchall()
    return [(int(s), int(e), int(b)) for s, e, b in rows]


def device_busy_ms(
    cur: sqlite3.Cursor, device_id: int, t0: int, t1: int, kstarts: list[int], kends: list[int]
) -> tuple[float, float, float]:
    """(compute-busy ms, all-kernel busy ms, nccl-exposed ms) on a device, kept-iters only.

    For rank skew. `compute` = non-NCCL kernel union (real work); `all` includes NCCL
    spin-wait kernels; `nccl_exposed` = NCCL union not overlapped by compute. Separating
    compute from NCCL is what distinguishes a true slow rank (compute imbalance) from mere
    collective-arrival jitter (NCCL LL kernels spin-wait, inflating `all` busy on whichever
    rank reaches the barrier first). Methodology mirrors the rank-0 headline numbers.

    F1: intervals are queried over [t0, t1] then clipped by overlap to the union of kept
    iteration intervals (`kstarts`/`kends`) so a dropped stall's kernels never contaminate
    the totals. After clipping, all-busy <= sum(kept wall) by construction.
    """
    all_iv = clip_to_kept(device_intervals(cur, device_id, t0, t1, nccl_only=None), kstarts, kends)
    busy = sum(e - s for s, e in merge_intervals(list(all_iv))) / NS_PER_MS if all_iv else 0.0
    nccl_iv = clip_to_kept(device_intervals(cur, device_id, t0, t1, nccl_only=True), kstarts, kends)
    comp_iv = clip_to_kept(device_intervals(cur, device_id, t0, t1, nccl_only=False), kstarts, kends)
    nccl_merged: list[tuple[int, int]] = [(s, e) for s, e in merge_intervals(list(nccl_iv))]
    comp_merged: list[tuple[int, int]] = [(s, e) for s, e in merge_intervals(list(comp_iv))]
    compute_busy = sum(e - s for s, e in comp_merged) / NS_PER_MS
    nccl_union = sum(e - s for s, e in nccl_merged) / NS_PER_MS
    hidden = overlap_sweep(nccl_merged, comp_merged) / NS_PER_MS if nccl_merged and comp_merged else 0.0
    return compute_busy, busy, nccl_union - hidden


def list_devices(cur: sqlite3.Cursor) -> list[int]:
    if not has_table(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return [0]
    return [
        int(r[0]) for r in cur.execute("SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId")
    ]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(sqlite_path: Path, label: str, skip_iters: int) -> dict:
    db = sqlite3.connect(str(sqlite_path))
    cur = db.cursor()

    nvtx = load_nvtx_ranges(cur)
    if not nvtx:
        sys.exit("ERROR: no recognized NVTX ranges in trace — was --profiling passed to train.py?")

    # Pick rank-0's NVTX timeline: prefer the PID owning device 0; fall back to the
    # globalTid with the most forward_backward ranges.
    pid0 = device0_pid(cur)
    fb_by_tid: dict[int, int] = {}
    for _, _, tid, name in nvtx:
        if name == "forward_backward":
            fb_by_tid[tid] = fb_by_tid.get(tid, 0) + 1
    if not fb_by_tid:
        sys.exit("ERROR: no forward_backward NVTX ranges found")
    pool = fb_by_tid
    if pid0 is not None:
        candidates = {t: c for t, c in fb_by_tid.items() if ((t >> 24) & 0xFFFFFF) == pid0}
        if candidates:
            pool = candidates
    primary_tid = max(pool, key=lambda t: pool[t])

    mine = sorted((s, e, name) for s, e, tid, name in nvtx if tid == primary_tid)
    fb_starts = sorted(s for s, _, name in mine if name == "forward_backward")
    # Iteration i spans [fb_start_i, fb_start_{i+1}) — includes everything between
    # steps (optimizer, EMA, logging, next-batch wait), giving true wall per iter.
    iters_all: list[tuple[int, int]] = list(zip(fb_starts, fb_starts[1:]))
    iters_all = iters_all[skip_iters:]
    if len(iters_all) < 10:
        sys.exit(f"ERROR: only {len(iters_all)} iterations in window after skipping {skip_iters} — capture too short")

    # Robust outlier removal. A real training iteration is sub-second to a few seconds;
    # nsys/CUDA-graph timestamp glitches occasionally emit a forward_backward start at a
    # garbage absolute time, yielding one segment with an astronomical (or negative) wall
    # that poisons every mean and sum-based %-of-wall (job 75082: one 1.06e10 ms segment).
    # p50 is already robust; dropping out-of-band segments makes mean/shares trustworthy too.
    wall_all = [(b - a) / NS_PER_MS for a, b in iters_all]
    med = sorted(wall_all)[len(wall_all) // 2]
    cap_ms = max(10_000.0, 6.0 * med)
    keep = [i for i, w in enumerate(wall_all) if 0.0 < w <= cap_ms]
    n_dropped = len(iters_all) - len(keep)
    # Audit trail: record the dropped walls and tag each as 'corrupt' (<=0 or implausibly
    # large vs any real iter) vs 'slow' (out-of-band but physically plausible — could be a
    # real gc/eval/checkpoint pause; for this capture eval/ckpt are disabled). Codex review
    # (2026-06-15) flagged that blind dropping can hide legitimate slow iters.
    dropped = [round(w, 3) for w in wall_all if not (0.0 < w <= cap_ms)]
    dropped_corrupt = [w for w in dropped if w <= 0.0 or w > 30_000.0]
    dropped_slow = [w for w in dropped if 0.0 < w <= 30_000.0]
    # F2: keep the (start, end) of each dropped-slow segment so its per-rank busy / idle-rank
    # can be surfaced in the .md instead of hiding a real multi-second straggler event.
    dropped_slow_intervals = [
        (iters_all[i], wall_all[i])
        for i in range(len(iters_all))
        if not (0.0 < wall_all[i] <= cap_ms) and 0.0 < wall_all[i] <= 30_000.0
    ]
    if len(keep) < 10:
        sys.exit(
            f"ERROR: only {len(keep)} clean iterations after outlier removal (dropped {n_dropped}) — trace corrupt or capture too short"
        )
    iters: list[tuple[int, int]] = [iters_all[i] for i in keep]
    # Sorted, disjoint kept iteration bounds — the F1 clip basis for every window-basis metric.
    kstarts = [a for a, _ in iters]
    kends = [b for _, b in iters]

    # Per-iteration range sums (a range instance belongs to the iter its start falls in).
    range_per_iter: dict[str, list[float]] = {n: [0.0] * len(iters) for n in ALL_RANGES}
    starts_only = [s for s, _ in iters]
    ends_only = [e for _, e in iters]
    for s, e, name in mine:
        i = bisect.bisect_right(starts_only, s) - 1
        if i < 0 or s >= ends_only[i]:
            continue
        range_per_iter[name][i] += (e - s) / NS_PER_MS

    iter_wall_ms = [(b - a) / NS_PER_MS for a, b in iters]
    top_sum_ms = [sum(range_per_iter[n][i] for n in TOP_RANGES) for i in range(len(iters))]
    untracked_ms = [max(0.0, iter_wall_ms[i] - top_sum_ms[i]) for i in range(len(iters))]

    # GPU busy vs gap per iteration (device 0).
    t0, t1 = iters[0][0], iters[-1][1]
    dev_iv = device_intervals(cur, 0, t0, t1, nccl_only=None)
    dev_iv.sort()
    busy_ms: list[float] = []
    for a, b in iters:
        window = [(max(s, a), min(e, b)) for s, e in dev_iv if e > a and s < b]
        merged = merge_intervals(window)
        busy_ms.append(sum(e - s for s, e in merged) / NS_PER_MS)
    gap_ms = [iter_wall_ms[i] - busy_ms[i] for i in range(len(iters))]

    # NCCL exposure on device 0 over the steady window. Use WALL-CLOCK UNION footprints
    # for both NCCL and compute: subtracting overlap (a union quantity) from a raw sum of
    # kernel durations is dimensionally mixed (Codex review 2026-06-15) and over-reports
    # exposure on traces with concurrent NCCL channels. Merge first, then sweep.
    # F1: clip to the union of kept iterations by overlap before merge/sweep, so a dropped
    # stall inside [t0, t1] never leaves its NCCL spin time in the per-iter numerator.
    nccl_iv = clip_to_kept(device_intervals(cur, 0, t0, t1, nccl_only=True), kstarts, kends)
    comp_iv = clip_to_kept(device_intervals(cur, 0, t0, t1, nccl_only=False), kstarts, kends)
    nccl_sum_ms = sum(e - s for s, e in nccl_iv) / NS_PER_MS  # kept for the concurrency check
    nccl_merged: list[tuple[int, int]] = [(s, e) for s, e in merge_intervals(list(nccl_iv))]
    comp_merged: list[tuple[int, int]] = [(s, e) for s, e in merge_intervals(list(comp_iv))]
    nccl_union_ms = sum(e - s for s, e in nccl_merged) / NS_PER_MS
    nccl_hidden_ms = overlap_sweep(nccl_merged, comp_merged) / NS_PER_MS if nccl_merged and comp_merged else 0.0
    nccl_exposed_ms = nccl_union_ms - nccl_hidden_ms
    n_iters = len(iters)

    # H2D reconciliation (device 0). The NVTX `H2D_transfer` range measures CPU enqueue and
    # understates the real cost; these are the GPU copy-engine intervals — the ground truth.
    # Per-iter bytes/DMA/bandwidth + how much DMA is EXPOSED (not overlapped by compute):
    # exposed H2D is wall time the GPU spends moving the input batch with nothing else running,
    # the part a copy-stream double-buffer could reclaim. Methodology mirrors NCCL exposure.
    # F1/F4: keep only copies whose start falls inside a kept iteration, so union/exposed,
    # bytes, bandwidth and the size histogram are all on the kept-iters basis (dropped-stall
    # copies excluded). Per-iter bytes/DMA were already kept-only via the same membership test.
    h2d_all = h2d_ops(cur, 0, t0, t1)
    h2d = []
    h2d_bytes_per_iter = [0.0] * len(iters)
    h2d_dma_per_iter = [0.0] * len(iters)
    for s, e, b in h2d_all:
        i = bisect.bisect_right(starts_only, s) - 1
        if i < 0 or s >= ends_only[i]:
            continue
        h2d.append((s, e, b))
        h2d_bytes_per_iter[i] += b
        h2d_dma_per_iter[i] += (e - s) / NS_PER_MS
    h2d_iv = clip_to_kept([(s, e) for s, e, _ in h2d], kstarts, kends)
    h2d_merged: list[tuple[int, int]] = [(s, e) for s, e in merge_intervals(list(h2d_iv))]
    h2d_union_ms = sum(e - s for s, e in h2d_merged) / NS_PER_MS
    h2d_hidden_ms = overlap_sweep(h2d_merged, comp_merged) / NS_PER_MS if h2d_merged and comp_merged else 0.0
    h2d_exposed_ms = h2d_union_ms - h2d_hidden_ms
    h2d_total_bytes = sum(b for _, _, b in h2d)
    h2d_total_dma_ms = sum((e - s) for s, e, _ in h2d) / NS_PER_MS
    h2d_bandwidth_gbs = (h2d_total_bytes / 1e9) / (h2d_total_dma_ms / 1e3) if h2d_total_dma_ms > 0 else 0.0
    # Distinct transfer sizes (e.g. the global-crop vs local-crop tensors) — the largest few
    # are the per-iter batch payloads; tiny ones are the mask/index scalars.
    size_hist: dict[int, dict] = {}
    for s, e, b in h2d:
        d = size_hist.setdefault(b, {"count": 0, "dma_ms": 0.0})
        d["count"] += 1
        d["dma_ms"] += (e - s) / NS_PER_MS
    top_sizes = sorted(size_hist.items(), key=lambda kv: kv[0] * kv[1]["count"], reverse=True)[:4]

    # Per-rank spread (all devices) — Codex flagged rank skew as a candidate residual driver,
    # especially at larger batch. A straggler shows up as higher busy on one rank and/or higher
    # NCCL-exposed on the others (they wait at the collective). Spread = max-min across ranks.
    devices = list_devices(cur)
    per_rank = {}
    for d in devices:
        compute_d, busy_d, nccl_exp_d = device_busy_ms(cur, d, t0, t1, kstarts, kends)
        per_rank[d] = {
            "compute_ms_per_iter": round(compute_d / n_iters, 3),
            "busy_ms_per_iter": round(busy_d / n_iters, 3),
            "nccl_exposed_ms_per_iter": round(nccl_exp_d / n_iters, 3),
        }

    # F2: characterize each dropped-slow stall — which rank was idle (arrived late) while the
    # others spun in the collective. This is the straggler signal the outlier filter removes;
    # surface it instead of hiding it behind "dropped N out-of-band segment(s)".
    dropped_slow_detail = []
    for (a, b), w in dropped_slow_intervals:
        single_s, single_e = [a], [b]
        busies = {}
        nccls = {}
        for d in devices:
            _, busy_d, nccl_exp_d = device_busy_ms(cur, d, a, b, single_s, single_e)
            busies[d] = busy_d
            nccls[d] = nccl_exp_d
        idle_rank = min(busies, key=lambda k: busies[k])
        dropped_slow_detail.append(
            {
                "wall_ms": round(w, 3),
                "idle_rank": idle_rank,
                "idle_rank_busy_ms": round(busies[idle_rank], 3),
                "busiest_rank": max(busies, key=lambda k: busies[k]),
                "busiest_rank_busy_ms": round(max(busies.values()), 3),
                "max_nccl_exposed_ms": round(max(nccls.values()), 3),
                # F2: full per-rank busy / NCCL-exposed content inside the stall, so the idle
                # (late-arriving) rank vs the spinning ranks is fully auditable, not just summarized.
                "per_rank_busy_ms": {str(d): round(busies[d], 3) for d in busies},
                "per_rank_nccl_exposed_ms": {str(d): round(nccls[d], 3) for d in nccls},
            }
        )
    # Stall-inclusive mean wall (F2/F5): keep the tail visible for cross-trace comparison —
    # traces whose stalls stay under the 10 s cap keep them in every basis, so report the mean
    # that includes dropped-slow walls alongside the kept-only mean.
    stall_incl_walls = iter_wall_ms + [w for _, w in dropped_slow_intervals]
    iter_wall_stall_incl_mean_ms = round(sum(stall_incl_walls) / len(stall_incl_walls), 3)
    # Skew on COMPUTE (real work) — the true straggler signal, free of NCCL spin-wait jitter.
    computes = [v["compute_ms_per_iter"] for v in per_rank.values()]
    rank_compute_spread_ms = round(max(computes) - min(computes), 3) if computes else 0.0
    rank_compute_cv_pct = (
        round(100.0 * (max(computes) - min(computes)) / max(sum(computes) / len(computes), 1e-9), 2)
        if computes
        else 0.0
    )

    # Kernel taxonomy (device 0). F1: durations are clipped by overlap to the kept iterations
    # (the SQL SUM(end-start) over [t0, t1] would fold a dropped stall's kernels into the
    # per-iter totals). A kernel is counted if any of it overlaps a kept iteration.
    class_totals: dict[str, dict] = {}
    top_kernels: list[tuple[str, int, float]] = []
    if has_table(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        rows = cur.execute(
            """
            SELECT s.value, k.start, k.end
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            WHERE k.deviceId=0 AND k.end>? AND k.start<? AND k.end IS NOT NULL
            """,
            (t0, t1),
        ).fetchall()
        n_kept = len(kstarts)
        kernel_totals: dict[str, list] = {}
        for name, ks, ke in rows:
            j = bisect.bisect_left(kends, ks)
            clipped_ns = 0
            while j < n_kept and kstarts[j] < ke:
                cs = ks if ks > kstarts[j] else kstarts[j]
                ce = ke if ke < kends[j] else kends[j]
                if ce > cs:
                    clipped_ns += ce - cs
                j += 1
            if clipped_ns <= 0:
                continue
            cls = classify_kernel(name or "")
            d = class_totals.setdefault(cls, {"count": 0, "total_ms": 0.0})
            d["count"] += 1
            d["total_ms"] += clipped_ns / NS_PER_MS
            kt = kernel_totals.setdefault(name, [0, 0.0])
            kt[0] += 1
            kt[1] += clipped_ns / NS_PER_MS
        top_kernels = sorted(
            ((name, c, ms) for name, (c, ms) in kernel_totals.items()),
            key=lambda t: t[2],
            reverse=True,
        )[:25]

    def stats(xs: list[float]) -> dict:
        p = percentiles(xs) or {}
        return {k: round(v, 3) for k, v in p.items()} if p else {}

    metrics = {
        "schema_version": 2,
        "label": label,
        "sqlite": str(sqlite_path),
        "primary_tid": primary_tid,
        "device0_pid": pid0,
        "n_iterations": n_iters,
        "skip_iters": skip_iters,
        "n_dropped_outliers": n_dropped,
        "dropped_walls_ms": dropped,
        "dropped_corrupt_ms": dropped_corrupt,
        "dropped_slow_ms": dropped_slow,
        "dropped_slow_detail": dropped_slow_detail,
        "iter_wall_ms": stats(iter_wall_ms),
        "iter_wall_stall_incl_mean_ms": iter_wall_stall_incl_mean_ms,
        "gpu_busy_ms": stats(busy_ms),
        "gpu_gap_ms": stats(gap_ms),
        "gpu_gap_share_pct": round(100.0 * sum(gap_ms) / max(sum(iter_wall_ms), 1e-9), 2),
        "untracked_ms": stats(untracked_ms),
        "untracked_share_pct": round(100.0 * sum(untracked_ms) / max(sum(iter_wall_ms), 1e-9), 2),
        "nccl_union_ms_per_iter": round(nccl_union_ms / n_iters, 3),
        "nccl_exposed_ms_per_iter": round(nccl_exposed_ms / n_iters, 3),
        "nccl_hidden_pct": round(100.0 * nccl_hidden_ms / max(nccl_union_ms, 1e-9), 1),
        # concurrency factor: sum/union > 1 means NCCL channels ran in parallel (the sum
        # over-reports wall footprint). ~1.0 here means channels were effectively serialized.
        "nccl_sum_over_union": round(nccl_sum_ms / max(nccl_union_ms, 1e-9), 2),
        # H2D ground truth (device 0). NVTX H2D_transfer (under ranges) is CPU enqueue;
        # these are the real copy-engine numbers. exposed = DMA not overlapped by compute.
        "h2d_bytes_per_iter_mb": stats([b / 1e6 for b in h2d_bytes_per_iter]),
        "h2d_dma_ms_per_iter": stats(h2d_dma_per_iter),
        "h2d_exposed_ms_per_iter": round(h2d_exposed_ms / n_iters, 3),
        "h2d_exposed_share_pct": round(100.0 * h2d_exposed_ms / max(sum(iter_wall_ms), 1e-9), 2),
        "h2d_hidden_pct": round(100.0 * h2d_hidden_ms / max(h2d_union_ms, 1e-9), 1),
        "h2d_bandwidth_gbs": round(h2d_bandwidth_gbs, 1),
        "h2d_top_sizes": [
            {"mb": round(b / 1e6, 3), "count": d["count"], "dma_ms_per_iter": round(d["dma_ms"] / n_iters, 3)}
            for b, d in top_sizes
        ],
        # Per-rank skew across all devices in the trace (spread on real compute, not NCCL spin).
        "rank_compute_spread_ms": rank_compute_spread_ms,
        "rank_compute_cv_pct": rank_compute_cv_pct,
        "per_rank": per_rank,
        "ranges": {},
        "kernel_classes": {
            cls: {"count": d["count"], "total_ms_per_iter": round(d["total_ms"] / n_iters, 3)}
            for cls, d in sorted(class_totals.items(), key=lambda kv: kv[1]["total_ms"], reverse=True)
        },
        "top_kernels": [
            {"name": (n or "")[:100], "count": c, "total_ms_per_iter": round(ms / n_iters, 3)}
            for n, c, ms in top_kernels
        ],
    }
    for name in ALL_RANGES:
        xs = range_per_iter[name]
        if not any(xs):
            continue
        metrics["ranges"][name] = {
            **stats(xs),
            "share_of_wall_pct": round(100.0 * sum(xs) / max(sum(iter_wall_ms), 1e-9), 2),
            "nested": name in NESTED_RANGES,
        }
    db.close()
    return metrics


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_md(m: dict) -> str:
    out = [f"# Phase 7 residual attribution — `{m['label']}`\n\n"]
    out.append(f"_Generated by `scripts/profiling/phase7_nsys_report.py` from `{Path(m['sqlite']).name}`._\n\n")
    n_corrupt = len(m.get("dropped_corrupt_ms", []))
    n_slow = len(m.get("dropped_slow_ms", []))
    out.append(
        f"- Iterations analyzed: **{m['n_iterations']}** "
        f"(skipped first {m['skip_iters']}, dropped {m.get('n_dropped_outliers', 0)} out-of-band "
        f"segment(s): {n_corrupt} corrupt, **{n_slow} slow-stall**)\n"
    )
    stall_incl = m.get("iter_wall_stall_incl_mean_ms")
    stall_note = f"; stall-inclusive mean = **{stall_incl} ms**" if n_slow and stall_incl is not None else ""
    out.append(
        f"- Iter wall (rank 0, kept-only): p50 = **{m['iter_wall_ms'].get('p50')} ms**, "
        f"p90 = {m['iter_wall_ms'].get('p90')} ms, mean = {m['iter_wall_ms'].get('mean')} ms{stall_note}\n"
    )
    out.append(
        f"- GPU busy (dev 0): p50 = **{m['gpu_busy_ms'].get('p50')} ms** | "
        f"GPU gap: p50 = **{m['gpu_gap_ms'].get('p50')} ms** "
        f"(**{m['gpu_gap_share_pct']}%** of wall is GPU-idle on rank 0)\n"
    )
    out.append(
        f"- NCCL per iter (dev 0): union {m['nccl_union_ms_per_iter']} ms, "
        f"exposed **{m['nccl_exposed_ms_per_iter']} ms** ({m['nccl_hidden_pct']}% hidden behind compute; "
        f"sum/union={m.get('nccl_sum_over_union')}× — >1 means concurrent channels)\n\n"
    )

    detail = m.get("dropped_slow_detail", [])
    if detail:
        out.append("## Dropped slow-stall segments (surfaced, not hidden)\n\n")
        out.append(
            "Each row is a multi-second iteration the outlier filter removed from the kept stats. "
            "These are real straggler events: one rank arrives late (idle) while the others spin in "
            "the collective. Window-basis metrics are computed kept-only (F1) so this GPU time never "
            "leaks into the per-iter averages above.\n\n"
        )
        out.append(
            "| stall wall (ms) | idle rank | idle-rank busy (ms) | busiest rank | busiest busy (ms) | max NCCL exposed (ms) |\n"
        )
        out.append("|---:|---:|---:|---:|---:|---:|\n")
        for d in detail:
            out.append(
                f"| {d['wall_ms']} | {d['idle_rank']} | {d['idle_rank_busy_ms']} | "
                f"{d['busiest_rank']} | {d['busiest_rank_busy_ms']} | {d['max_nccl_exposed_ms']} |\n"
            )
        out.append("\n")

    out.append("## NVTX range breakdown (per-iteration wall, rank 0)\n\n")
    out.append("Top-level ranges partition the step; nested ranges (·) live inside forward_backward.\n")
    out.append("`untracked` = wall not inside any top-level range: dataloader wait, logging, Python.\n\n")
    out.append("| range | p50 ms | p90 ms | mean ms | % of wall |\n|---|---:|---:|---:|---:|\n")
    for name, r in m["ranges"].items():
        prefix = "· " if r.get("nested") else ""
        out.append(
            f"| {prefix}{name} | {r.get('p50')} | {r.get('p90')} | {r.get('mean')} | {r['share_of_wall_pct']}% |\n"
        )
    out.append(
        f"| **untracked** | {m['untracked_ms'].get('p50')} | {m['untracked_ms'].get('p90')} | "
        f"{m['untracked_ms'].get('mean')} | **{m['untracked_share_pct']}%** |\n\n"
    )

    out.append("## H2D input-batch transfer (device 0, ground truth vs NVTX)\n\n")
    nvtx_h2d = m["ranges"].get("H2D_transfer", {})
    out.append(
        f"- Real DMA: p50 **{m['h2d_dma_ms_per_iter'].get('p50')} ms/iter** "
        f"(mean {m['h2d_dma_ms_per_iter'].get('mean')}), "
        f"bytes p50 **{m['h2d_bytes_per_iter_mb'].get('p50')} MB/iter**, "
        f"bandwidth **{m['h2d_bandwidth_gbs']} GB/s**\n"
    )
    out.append(
        f"- NVTX `H2D_transfer` (CPU enqueue): p50 {nvtx_h2d.get('p50')} ms — "
        f"understates the real DMA above; use the DMA number for cost.\n"
    )
    out.append(
        f"- Exposed (not hidden by compute): **{m['h2d_exposed_ms_per_iter']} ms/iter** "
        f"(**{m['h2d_exposed_share_pct']}%** of wall; {m['h2d_hidden_pct']}% of DMA hidden) "
        f"— this is the part a copy-stream double-buffer could reclaim.\n"
    )
    if m.get("h2d_top_sizes"):
        out.append("\n| transfer size | count (window) | DMA ms/iter |\n|---|---:|---:|\n")
        for sz in m["h2d_top_sizes"]:
            out.append(f"| {sz['mb']} MB | {sz['count']} | {sz['dma_ms_per_iter']} |\n")
    out.append("\n## Per-rank spread (all devices — straggler / skew check)\n\n")
    out.append(
        f"- **Compute** spread across ranks: **{m['rank_compute_spread_ms']} ms/iter** "
        f"(**{m['rank_compute_cv_pct']}%** of mean). This is real-work imbalance; a large value = a "
        f"true straggler. NCCL-exposed variance (next column) is mostly collective-arrival jitter, "
        f"not a slow rank.\n\n"
    )
    out.append(
        "Values are kept-only (F1); a ⚠ flags a rank that was the idle (late-arriving) rank in a "
        "dropped slow-stall segment above.\n\n"
    )
    idle_ranks = {int(d["idle_rank"]) for d in m.get("dropped_slow_detail", [])}
    out.append("| rank | compute ms/iter | all-busy ms/iter | NCCL exposed ms/iter |\n|---|---:|---:|---:|\n")
    for d, v in sorted(m.get("per_rank", {}).items(), key=lambda kv: int(kv[0])):
        flag = " ⚠" if int(d) in idle_ranks else ""
        out.append(
            f"| {d}{flag} | {v['compute_ms_per_iter']} | {v['busy_ms_per_iter']} | {v['nccl_exposed_ms_per_iter']} |\n"
        )
    out.append("\n## Kernel time by class (device 0, per iteration)\n\n")
    out.append("| class | count (window) | ms/iter |\n|---|---:|---:|\n")
    for cls, d in m["kernel_classes"].items():
        out.append(f"| {cls} | {d['count']} | {d['total_ms_per_iter']} |\n")
    out.append("\n## Top kernels (device 0, per iteration)\n\n")
    out.append("| kernel | count (window) | ms/iter |\n|---|---:|---:|\n")
    for k in m["top_kernels"]:
        out.append(f"| `{k['name']}` | {k['count']} | {k['total_ms_per_iter']} |\n")
    out.append("\n---\n_Machine-readable metrics: same path, `.json`. Compare two runs with `--diff`._\n")
    return "".join(out)


def render_diff(a: dict, b: dict) -> str:
    def row(name: str, va, vb) -> str:
        try:
            d = (vb - va) / va * 100.0 if va else float("nan")
            return f"| {name} | {va} | {vb} | {d:+.1f}% |\n"
        except TypeError:
            return f"| {name} | {va} | {vb} | — |\n"

    out = [f"# Phase 7 trace diff — `{a['label']}` vs `{b['label']}`\n\n"]
    out.append(f"| metric | {a['label']} | {b['label']} | Δ |\n|---|---:|---:|---:|\n")
    out.append(row("iter wall p50 (ms)", a["iter_wall_ms"].get("p50"), b["iter_wall_ms"].get("p50")))
    out.append(row("GPU busy p50 (ms)", a["gpu_busy_ms"].get("p50"), b["gpu_busy_ms"].get("p50")))
    out.append(row("GPU gap p50 (ms)", a["gpu_gap_ms"].get("p50"), b["gpu_gap_ms"].get("p50")))
    out.append(row("GPU gap share (%)", a["gpu_gap_share_pct"], b["gpu_gap_share_pct"]))
    out.append(row("untracked p50 (ms)", a["untracked_ms"].get("p50"), b["untracked_ms"].get("p50")))
    out.append(row("NCCL exposed (ms/iter)", a["nccl_exposed_ms_per_iter"], b["nccl_exposed_ms_per_iter"]))
    out.append(
        row(
            "H2D DMA p50 (ms/iter)",
            a.get("h2d_dma_ms_per_iter", {}).get("p50"),
            b.get("h2d_dma_ms_per_iter", {}).get("p50"),
        )
    )
    out.append(
        row(
            "H2D bytes p50 (MB/iter)",
            a.get("h2d_bytes_per_iter_mb", {}).get("p50"),
            b.get("h2d_bytes_per_iter_mb", {}).get("p50"),
        )
    )
    out.append(row("H2D exposed (ms/iter)", a.get("h2d_exposed_ms_per_iter"), b.get("h2d_exposed_ms_per_iter")))
    out.append(row("H2D bandwidth (GB/s)", a.get("h2d_bandwidth_gbs"), b.get("h2d_bandwidth_gbs")))
    out.append(row("rank compute spread (ms)", a.get("rank_compute_spread_ms"), b.get("rank_compute_spread_ms")))
    out.append(row("rank compute CV (%)", a.get("rank_compute_cv_pct"), b.get("rank_compute_cv_pct")))
    out.append("\n## NVTX ranges (p50 ms)\n\n")
    out.append(f"| range | {a['label']} | {b['label']} | Δ |\n|---|---:|---:|---:|\n")
    for name in ALL_RANGES:
        ra, rb = a["ranges"].get(name), b["ranges"].get(name)
        if not (ra or rb):
            continue
        out.append(row(name, (ra or {}).get("p50"), (rb or {}).get("p50")))
    out.append("\n## Kernel classes (ms/iter)\n\n")
    out.append(f"| class | {a['label']} | {b['label']} | Δ |\n|---|---:|---:|---:|\n")
    for cls in sorted(set(a["kernel_classes"]) | set(b["kernel_classes"])):
        va = a["kernel_classes"].get(cls, {}).get("total_ms_per_iter")
        vb = b["kernel_classes"].get(cls, {}).get("total_ms_per_iter")
        out.append(row(cls, va, vb))
    out.append(
        "\n**Reading a bs differential** (e.g. bs128 vs bs160 at matched loader settings): "
        "components scaling ~with batch ratio are real compute; components staying ~constant "
        "are per-iteration overhead — the residual Phase 7 is hunting.\n"
    )
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "inputs", nargs="+", type=Path, help="one .sqlite to analyze, or (with --diff) two .phase7.json files"
    )
    ap.add_argument("--diff", action="store_true", help="compare two .phase7.json metric files")
    ap.add_argument("--label", default=None, help="run label (default: sqlite stem)")
    ap.add_argument("--out", type=Path, default=None, help="output md path override")
    ap.add_argument(
        "--tag",
        default=None,
        help="insert a tag into the output suffix, e.g. --tag v2 -> .phase7.v2.{md,json} "
        "(keeps regenerated reports alongside originals without overwriting them)",
    )
    ap.add_argument("--skip-iters", type=int, default=5, help="iterations to drop at window start (default 5)")
    args = ap.parse_args()

    if args.diff:
        if len(args.inputs) != 2:
            ap.error("--diff needs exactly two .phase7.json files")
        a, b = (json.loads(p.read_text()) for p in args.inputs)
        out = args.out or args.inputs[0].with_name(f"diff-{a['label']}-vs-{b['label']}.md")
        out.write_text(render_diff(a, b), encoding="utf-8")
        print(f"Wrote: {out}")
        return 0

    sqlite_path = args.inputs[0]
    if not sqlite_path.exists():
        sys.exit(f"not found: {sqlite_path}")
    label = args.label or sqlite_path.stem
    metrics = analyze(sqlite_path, label, args.skip_iters)
    suffix = f".phase7.{args.tag}" if args.tag else ".phase7"
    json_path = sqlite_path.with_suffix(f"{suffix}.json")
    md_path = args.out or sqlite_path.with_suffix(f"{suffix}.md")
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    md_path.write_text(render_md(metrics), encoding="utf-8")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
