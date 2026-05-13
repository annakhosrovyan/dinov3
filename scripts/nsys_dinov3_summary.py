#!/usr/bin/env python3
"""DINOv3-specific summary for an Nsight Systems SQLite export.

Anchored to the four bottleneck hypotheses in `docs/phase5_perf_plan.md`:
  A. Multi-crop forward structure (seq 197 + seq 37 sequential through same blocks)
  B. NCCL all-gather / reduce-scatter serialization (FSDP2 ZeRO-3)
  C. H2D memcpy not actually overlapped with compute
  D. Periodic GC / eval / checkpoint stalls

Output: a markdown report next to the input trace. Designed to complement (not replace)
manual GUI analysis in Nsight Systems.

Usage:
  python scripts/nsys_dinov3_summary.py /path/to/trace.sqlite
  python scripts/nsys_dinov3_summary.py /path/to/trace.sqlite --out report.md
  python scripts/nsys_dinov3_summary.py /path/to/trace.sqlite --steady-skip-s 10
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000


def percentiles(xs: list[float]) -> dict | None:
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return {
        "n": n,
        "min": xs[0],
        "p10": xs[max(0, int(0.1 * n) - 1)],
        "p50": xs[n // 2],
        "p90": xs[min(n - 1, int(0.9 * n))],
        "p99": xs[min(n - 1, int(0.99 * n))],
        "max": xs[-1],
        "mean": sum(xs) / n,
    }


def fmt_pct(p: dict | None, unit: str = "ms") -> str:
    if p is None:
        return "(no data)"
    return (
        f"n={p['n']:>6} min={p['min']:.2f} p10={p['p10']:.2f} p50={p['p50']:.2f} "
        f"p90={p['p90']:.2f} p99={p['p99']:.2f} max={p['max']:.2f} mean={p['mean']:.2f} {unit}"
    )


def has_table(cur: sqlite3.Cursor, name: str) -> bool:
    return bool(
        cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
            (name,),
        ).fetchone()
    )


# Heuristic kernel classification — matches the kernel-name strings nsys records.
# Order matters: NCCL first (often contains "ReduceScatter"/"AllGather" + "Kernel").
KERNEL_CLASSIFIERS: list[tuple[str, re.Pattern[str]]] = [
    ("nccl", re.compile(r"(?i)(ncclKernel|nccl_|allreduce|allgather|reducescatter|reduce_scatter|broadcast|all_gather|all_reduce)")),
    ("attention_flash", re.compile(r"(?i)(flash_(fwd|bwd)|fmha|mha_fwd|mha_bwd|attention.*kernel)")),
    ("matmul_gemm", re.compile(r"(?i)(gemm|cutlass|sgemm|hgemm|bgemm|cublas|cublaslt|ampere_|sm80_|sm90_)")),
    ("conv", re.compile(r"(?i)(conv|implicit_gemm|cudnn)")),
    ("memcpy_d2d", re.compile(r"(?i)(memcpy.*device.*device|memcpy_d2d|copy_kernel)")),
    ("layernorm_norm", re.compile(r"(?i)(layer_norm|rms_norm|layernorm|batchnorm)")),
    ("activation_elementwise", re.compile(r"(?i)(elementwise|gelu|silu|relu|softmax|dropout|add_kernel|mul_kernel|cast_kernel|to_copy)")),
    ("reduction", re.compile(r"(?i)(reduce_kernel|sum_kernel|mean_kernel|argmax|topk|sort)")),
    ("optimizer", re.compile(r"(?i)(adam|sgd|foreach|fused_adam)")),
]


def classify_kernel(name: str) -> str:
    for label, rx in KERNEL_CLASSIFIERS:
        if rx.search(name):
            return label
    return "other"


def merge_intervals(intervals: list[tuple[int, int]], merge_gap_ns: int = 50_000) -> list[list[int]]:
    """Merge overlapping/near-overlapping (start_ns, end_ns) intervals (sorted)."""
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


def get_kernel_intervals(cur: sqlite3.Cursor, device_id: int | None = None) -> list[tuple[int, int]]:
    """Return raw (start_ns, end_ns) for kernel + memcpy + memset on the given device,
    or all devices if device_id is None."""
    out: list[tuple[int, int]] = []
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if not has_table(cur, table):
            continue
        if device_id is not None:
            rows = cur.execute(
                f"SELECT start, end FROM {table} WHERE deviceId = ? AND start IS NOT NULL AND end IS NOT NULL",
                (device_id,),
            )
        else:
            rows = cur.execute(
                f"SELECT start, end FROM {table} WHERE start IS NOT NULL AND end IS NOT NULL"
            )
        out.extend((int(s), int(e)) for s, e in rows)
    return out


def find_attention_kernels(cur: sqlite3.Cursor, t0: int, t1: int) -> list[tuple[str, int]]:
    """Return (name, duration_ns) for kernels matching attention/flash within [t0, t1]."""
    rows = cur.execute(
        """
        SELECT s.value AS name, k.end - k.start AS dur
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        WHERE k.start >= ? AND k.start < ? AND k.end IS NOT NULL
        """,
        (t0, t1),
    ).fetchall()
    out = []
    for name, dur in rows:
        if name and ("flash" in name.lower() or "fmha" in name.lower() or "mha_" in name.lower()):
            out.append((name, int(dur)))
    return out


def main(sqlite_path: Path, out_path: Path, steady_skip_s: float, top_n: int) -> int:
    db = sqlite3.connect(str(sqlite_path))
    cur = db.cursor()

    # --- Trace shape -----------------------------------------------------------
    has_kernel = has_table(cur, "CUPTI_ACTIVITY_KIND_KERNEL")
    has_memcpy = has_table(cur, "CUPTI_ACTIVITY_KIND_MEMCPY")
    has_memset = has_table(cur, "CUPTI_ACTIVITY_KIND_MEMSET")
    has_runtime = has_table(cur, "CUPTI_ACTIVITY_KIND_RUNTIME")
    has_nvtx = has_table(cur, "NVTX_EVENTS")

    meta = cur.execute(
        "SELECT duration, startTime, stopTime FROM ANALYSIS_DETAILS ORDER BY startTime LIMIT 1"
    ).fetchone()
    trace_dur_s = (meta[0] / NS_PER_S) if meta and meta[0] else 0.0

    # Per-device kernel time
    devices: list[int] = []
    if has_kernel:
        devices = [
            r[0]
            for r in cur.execute(
                "SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId"
            ).fetchall()
        ]

    # --- Build steady-state window --------------------------------------------
    # Use earliest kernel start as "t0" of GPU activity, then skip steady_skip_s.
    if has_kernel:
        first_k = cur.execute(
            "SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0]
        last_k = cur.execute(
            "SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0]
    else:
        first_k = last_k = 0

    steady_t0 = (first_k or 0) + int(steady_skip_s * NS_PER_S)
    steady_t1 = last_k or steady_t0
    if steady_t1 <= steady_t0:
        steady_t0 = first_k or 0
        steady_t1 = last_k or 0

    # --- Per-device summary ----------------------------------------------------
    per_device_rows: list[tuple[int, float, float, float]] = []  # (dev, span_s, active_s, util%)
    for dev in devices:
        intervals = get_kernel_intervals(cur, device_id=dev)
        intervals = [(s, e) for (s, e) in intervals if s >= steady_t0 and e <= steady_t1]
        if not intervals:
            continue
        intervals.sort()
        merged = merge_intervals(intervals)
        active_ns = sum(e - s for s, e in merged)
        span_ns = merged[-1][1] - merged[0][0]
        per_device_rows.append(
            (
                dev,
                span_ns / NS_PER_S,
                active_ns / NS_PER_S,
                100.0 * active_ns / span_ns if span_ns > 0 else 0.0,
            )
        )

    # --- Aggregate (any-device-active) timeline for gap analysis --------------
    # Gaps in this timeline mean "no GPU on any rank is doing work" — which on a
    # well-overlapped multi-rank training run should be near zero. Per-rank gaps
    # below tell a different story.
    union_intervals = get_kernel_intervals(cur, device_id=None)
    union_intervals = [(s, e) for (s, e) in union_intervals if s >= steady_t0 and e <= steady_t1]
    union_merged = merge_intervals(union_intervals)
    union_span_s = (union_merged[-1][1] - union_merged[0][0]) / NS_PER_S if union_merged else 0.0
    union_active_s = sum(e - s for s, e in union_merged) / NS_PER_S
    union_gaps_ms = [
        (union_merged[i + 1][0] - union_merged[i][1]) / NS_PER_MS
        for i in range(len(union_merged) - 1)
        if union_merged[i + 1][0] > union_merged[i][1]
    ]
    top_union_gaps = sorted(
        (
            (union_merged[i][1], union_merged[i + 1][0], (union_merged[i + 1][0] - union_merged[i][1]) / NS_PER_MS)
            for i in range(len(union_merged) - 1)
        ),
        key=lambda r: r[2],
        reverse=True,
    )[:20]

    # --- Per-device gap distribution (steady-state, per rank 0) ---------------
    rank0_gaps_ms: list[float] = []
    rank0_top_gaps: list[tuple[float, float, float]] = []  # (start_s, end_s, gap_ms)
    if devices:
        rank0_intervals = get_kernel_intervals(cur, device_id=devices[0])
        rank0_intervals = [(s, e) for (s, e) in rank0_intervals if s >= steady_t0 and e <= steady_t1]
        rank0_merged = merge_intervals(rank0_intervals, merge_gap_ns=10_000)
        for i in range(len(rank0_merged) - 1):
            gap_ms = (rank0_merged[i + 1][0] - rank0_merged[i][1]) / NS_PER_MS
            if gap_ms > 0:
                rank0_gaps_ms.append(gap_ms)
        rank0_top_gaps = sorted(
            (
                (
                    rank0_merged[i][1] / NS_PER_S,
                    rank0_merged[i + 1][0] / NS_PER_S,
                    (rank0_merged[i + 1][0] - rank0_merged[i][1]) / NS_PER_MS,
                )
                for i in range(len(rank0_merged) - 1)
            ),
            key=lambda r: r[2],
            reverse=True,
        )[:15]

    # --- Top kernels (steady-state) --------------------------------------------
    kernel_rows: list[tuple[str, int, float, float]] = []  # (name, count, total_ms, avg_us)
    if has_kernel:
        kernel_rows = cur.execute(
            """
            SELECT s.value AS name, COUNT(*) AS cnt, SUM(k.end - k.start) AS total_ns,
                   AVG(k.end - k.start) AS avg_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            WHERE k.start >= ? AND k.start < ?
            GROUP BY s.value
            ORDER BY total_ns DESC
            LIMIT ?
            """,
            (steady_t0, steady_t1, top_n),
        ).fetchall()
        kernel_rows = [(name, cnt, tot / NS_PER_MS, avg / 1000.0) for name, cnt, tot, avg in kernel_rows]

    # --- Aggregate by kernel class --------------------------------------------
    class_totals: dict[str, dict] = {}
    if has_kernel:
        for name, cnt, total_ns, avg_ns in cur.execute(
            """
            SELECT s.value AS name, COUNT(*) AS cnt, SUM(k.end - k.start) AS total_ns,
                   AVG(k.end - k.start) AS avg_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            WHERE k.start >= ? AND k.start < ?
            GROUP BY s.value
            """,
            (steady_t0, steady_t1),
        ).fetchall():
            cls = classify_kernel(name or "")
            d = class_totals.setdefault(cls, {"count": 0, "total_ms": 0.0})
            d["count"] += cnt
            d["total_ms"] += total_ns / NS_PER_MS

    # Total kernel time across all kernels in steady window — denominator for class %.
    total_kernel_ms = sum(d["total_ms"] for d in class_totals.values()) or 1.0

    # --- NCCL kernel breakdown (per kernel name) ------------------------------
    nccl_rows: list[tuple[str, int, float]] = []
    if has_kernel:
        for name, cnt, tot in cur.execute(
            """
            SELECT s.value AS name, COUNT(*) AS cnt, SUM(k.end - k.start) AS total_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            WHERE k.start >= ? AND k.start < ?
            GROUP BY s.value
            """,
            (steady_t0, steady_t1),
        ).fetchall():
            if name and ("nccl" in name.lower() or "all_reduce" in name.lower() or "all_gather" in name.lower()):
                nccl_rows.append((name, cnt, tot / NS_PER_MS))
        nccl_rows.sort(key=lambda r: r[2], reverse=True)

    # --- NCCL ↔ compute overlap (single-rank: device 0) -----------------------
    overlap_pct: float | None = None
    nccl_total_ms = 0.0
    compute_total_ms = 0.0
    if has_kernel and devices:
        dev0 = devices[0]
        rows = cur.execute(
            """
            SELECT k.start, k.end, s.value AS name
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            WHERE k.deviceId = ? AND k.start >= ? AND k.start < ?
            """,
            (dev0, steady_t0, steady_t1),
        ).fetchall()
        nccl_iv = []
        comp_iv = []
        for s, e, name in rows:
            if not name:
                comp_iv.append((int(s), int(e)))
                continue
            n_lower = name.lower()
            if "nccl" in n_lower or "all_reduce" in n_lower or "all_gather" in n_lower:
                nccl_iv.append((int(s), int(e)))
            else:
                comp_iv.append((int(s), int(e)))
        nccl_total_ms = sum(e - s for s, e in nccl_iv) / NS_PER_MS
        compute_total_ms = sum(e - s for s, e in comp_iv) / NS_PER_MS
        # Compute overlap via interval sweep
        events = (
            [(s, "ns") for s, _ in nccl_iv]
            + [(e, "ne") for _, e in nccl_iv]
            + [(s, "cs") for s, _ in comp_iv]
            + [(e, "ce") for _, e in comp_iv]
        )
        events.sort()
        n_open = c_open = 0
        last = events[0][0] if events else 0
        ovlp_ns = 0
        for t, ev in events:
            if n_open > 0 and c_open > 0:
                ovlp_ns += t - last
            if ev == "ns":
                n_open += 1
            elif ev == "ne":
                n_open -= 1
            elif ev == "cs":
                c_open += 1
            elif ev == "ce":
                c_open -= 1
            last = t
        if nccl_total_ms > 0:
            overlap_pct = 100.0 * (ovlp_ns / NS_PER_MS) / nccl_total_ms

    # --- Memcpy (H2D / D2H / D2D) summary -------------------------------------
    memcpy_rows: list[tuple[str, int, float, float]] = []  # (kind, count, GB, ms)
    h2d_streams: list[tuple[int, int, float, float]] = []  # (deviceId, streamId, GB, ms)
    if has_memcpy:
        # Resolve copyKind ↔ label: copyKind=1 is H2D, =2 is D2H, =8 is D2D in CUPTI conventions.
        kind_map = {1: "H2D", 2: "D2H", 8: "D2D"}
        for ck, cnt, by, dur in cur.execute(
            """
            SELECT copyKind, COUNT(*), COALESCE(SUM(bytes),0), COALESCE(SUM(end-start),0)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE start >= ? AND start < ?
            GROUP BY copyKind ORDER BY 4 DESC
            """,
            (steady_t0, steady_t1),
        ).fetchall():
            memcpy_rows.append((kind_map.get(ck, f"kind={ck}"), cnt, by / 1e9, dur / NS_PER_MS))

        for dev, sid, gb, ms in cur.execute(
            """
            SELECT deviceId, streamId,
                   COALESCE(SUM(bytes),0)/1e9 AS gb,
                   COALESCE(SUM(end-start),0)/1e6 AS ms
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE copyKind=1 AND start >= ? AND start < ?
            GROUP BY deviceId, streamId
            ORDER BY deviceId, ms DESC
            """,
            (steady_t0, steady_t1),
        ).fetchall():
            h2d_streams.append((int(dev), int(sid), float(gb), float(ms)))

    # --- H2D ↔ compute overlap (rank 0) ---------------------------------------
    h2d_overlap_pct: float | None = None
    h2d_total_ms = 0.0
    if has_memcpy and devices:
        dev0 = devices[0]
        h2d = cur.execute(
            """
            SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE copyKind=1 AND deviceId = ? AND start >= ? AND start < ?
            """,
            (dev0, steady_t0, steady_t1),
        ).fetchall()
        kern = cur.execute(
            """
            SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
            WHERE k.deviceId = ? AND k.start >= ? AND k.start < ?
            """,
            (dev0, steady_t0, steady_t1),
        ).fetchall()
        h2d_iv = [(int(s), int(e)) for s, e in h2d]
        kern_iv = [(int(s), int(e)) for s, e in kern]
        h2d_total_ms = sum(e - s for s, e in h2d_iv) / NS_PER_MS
        if h2d_total_ms > 0:
            ev = (
                [(s, "hs") for s, _ in h2d_iv]
                + [(e, "he") for _, e in h2d_iv]
                + [(s, "ks") for s, _ in kern_iv]
                + [(e, "ke") for _, e in kern_iv]
            )
            ev.sort()
            ho = ko = 0
            ovlp_ns = 0
            last = ev[0][0]
            for t, evt in ev:
                if ho > 0 and ko > 0:
                    ovlp_ns += t - last
                if evt == "hs":
                    ho += 1
                elif evt == "he":
                    ho -= 1
                elif evt == "ks":
                    ko += 1
                elif evt == "ke":
                    ko -= 1
                last = t
            h2d_overlap_pct = 100.0 * (ovlp_ns / NS_PER_MS) / h2d_total_ms

    # --- Attention kernel duration distribution (Hypothesis A) ---------------
    attn_durs_ms = []
    attn_kernel_rows = []
    if has_kernel:
        attn_kernel_rows = cur.execute(
            """
            SELECT s.value AS name, COUNT(*) AS cnt, SUM(k.end - k.start) AS total_ns,
                   AVG(k.end - k.start) AS avg_ns, MIN(k.end - k.start) AS min_ns,
                   MAX(k.end - k.start) AS max_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            WHERE k.start >= ? AND k.start < ?
            GROUP BY s.value
            """,
            (steady_t0, steady_t1),
        ).fetchall()
        attn_kernel_rows = [
            (name, cnt, total_ns / NS_PER_MS, avg_ns / 1000.0, min_ns / 1000.0, max_ns / 1000.0)
            for name, cnt, total_ns, avg_ns, min_ns, max_ns in attn_kernel_rows
            if name and ("flash" in name.lower() or "fmha" in name.lower() or "mha_" in name.lower())
        ]
        attn_kernel_rows.sort(key=lambda r: r[2], reverse=True)
        # Per-event duration distribution (rank 0 only) for percentiles
        if devices:
            for s, e, name in cur.execute(
                """
                SELECT k.start, k.end, sn.value
                FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sn ON sn.id = k.shortName
                WHERE k.deviceId = ? AND k.start >= ? AND k.start < ?
                """,
                (devices[0], steady_t0, steady_t1),
            ).fetchall():
                if name and ("flash" in name.lower() or "fmha" in name.lower() or "mha_" in name.lower()):
                    attn_durs_ms.append((e - s) / NS_PER_MS)

    # --- Top runtime APIs (steady-state) --------------------------------------
    runtime_rows = []
    if has_runtime:
        runtime_rows = cur.execute(
            """
            SELECT s.value, COUNT(*), SUM(r.end - r.start) / 1e6
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON s.id = r.nameId
            WHERE r.start >= ? AND r.start < ?
            GROUP BY s.value ORDER BY 3 DESC LIMIT 12
            """,
            (steady_t0, steady_t1),
        ).fetchall()

    # --- Write report ----------------------------------------------------------
    out = []
    out.append(f"# DINOv3 nsys summary — `{sqlite_path.name}`\n")
    out.append(f"_Generated by `scripts/nsys_dinov3_summary.py`._\n\n")
    out.append("## 1. Trace shape\n\n")
    out.append(f"- Trace duration: **{trace_dur_s:.2f} s**\n")
    out.append(f"- Devices (GPUs) detected: **{len(devices)}** ({devices})\n")
    out.append(f"- Tables: KERNEL={has_kernel}  MEMCPY={has_memcpy}  MEMSET={has_memset}  RUNTIME={has_runtime}  NVTX={has_nvtx}\n")
    if first_k:
        out.append(f"- First kernel start: {first_k / NS_PER_S:.2f} s\n")
        out.append(f"- Last  kernel end:   {last_k / NS_PER_S:.2f} s\n")
    out.append(f"- Steady-state window: **[{steady_t0/NS_PER_S:.2f} s, {steady_t1/NS_PER_S:.2f} s] = {(steady_t1-steady_t0)/NS_PER_S:.2f} s**\n")
    out.append(f"  (skipped first {steady_skip_s:.1f} s of GPU activity to bypass compile/init warmup)\n\n")

    out.append("## 2. Per-device GPU utilization (steady-state)\n\n")
    out.append("Active = sum of merged kernel/memcpy intervals on that device.\n")
    out.append("Span   = elapsed time between first and last activity on that device.\n")
    out.append("Util   = 100 × active / span. **Higher is better; near-100% means the GPU is rarely idle on that rank.**\n\n")
    out.append("| device | span s | active s | util % |\n|---:|---:|---:|---:|\n")
    for dev, span, act, util in per_device_rows:
        out.append(f"| {dev} | {span:.3f} | {act:.3f} | {util:.1f} |\n")
    if per_device_rows:
        utils = [u for *_, u in per_device_rows]
        out.append(f"\nUtil spread across ranks: min={min(utils):.1f}%  max={max(utils):.1f}%  range={max(utils)-min(utils):.1f} pp\n")
        out.append("Large spread (≥ a few pp) indicates straggler imbalance — bad for collective-bound training.\n\n")
    else:
        out.append("\n(no per-device data)\n\n")

    out.append("## 3. Union (any-rank) GPU activity & gaps\n\n")
    out.append(f"- Span (any-rank): {union_span_s:.3f} s\n")
    out.append(f"- Active (union): {union_active_s:.3f} s ({100*union_active_s/union_span_s:.1f}% of span)\n" if union_span_s else "")
    out.append(f"- Gap percentiles (ms): {fmt_pct(percentiles(union_gaps_ms))}\n\n")
    out.append("Top 20 union gaps (when **no rank** has any GPU work — typically a global stall):\n\n")
    out.append("| start s | end s | gap ms |\n|---:|---:|---:|\n")
    for s, e, g in top_union_gaps:
        out.append(f"| {s/NS_PER_S:.3f} | {e/NS_PER_S:.3f} | {g:.2f} |\n")
    out.append("\n")

    out.append("## 4. Per-rank gap distribution (rank 0)\n\n")
    out.append(f"- Gap percentiles (ms): {fmt_pct(percentiles(rank0_gaps_ms))}\n\n")
    out.append("Top 15 rank-0 gaps (intervals where rank 0 is idle even if other ranks may be busy):\n\n")
    out.append("| start s | end s | gap ms |\n|---:|---:|---:|\n")
    for s, e, g in rank0_top_gaps:
        out.append(f"| {s:.3f} | {e:.3f} | {g:.2f} |\n")
    out.append("\n")

    out.append("## 5. Kernel time by class (rough taxonomy, all ranks)\n\n")
    out.append("Heuristic name-match. Useful as a coarse breakdown — for absolute numbers, see top-N below.\n\n")
    out.append("| class | count | total ms | % of kernel time |\n|---|---:|---:|---:|\n")
    for cls, d in sorted(class_totals.items(), key=lambda kv: kv[1]["total_ms"], reverse=True):
        out.append(f"| {cls} | {d['count']} | {d['total_ms']:.0f} | {100*d['total_ms']/total_kernel_ms:.1f}% |\n")
    out.append("\n")

    out.append(f"## 6. Top {top_n} kernels by total time (all ranks)\n\n")
    out.append("| name | count | total ms | avg us |\n|---|---:|---:|---:|\n")
    for name, cnt, tot_ms, avg_us in kernel_rows:
        out.append(f"| `{(name or '')[:90]}` | {cnt} | {tot_ms:.0f} | {avg_us:.1f} |\n")
    out.append("\n")

    out.append("## 7. NCCL kernels (FSDP2 ZeRO-3 = all-gather + reduce-scatter)\n\n")
    out.append("If reshard_after_forward=True (current), expect heavy AllGather (forward+backward) and ReduceScatter (gradient reduce).\n\n")
    out.append("| name | count | total ms |\n|---|---:|---:|\n")
    for name, cnt, ms in nccl_rows[:20]:
        out.append(f"| `{(name or '')[:90]}` | {cnt} | {ms:.0f} |\n")
    out.append(f"\n**Rank-0 NCCL total**: {nccl_total_ms:.0f} ms\n")
    out.append(f"**Rank-0 compute total**: {compute_total_ms:.0f} ms\n")
    if overlap_pct is not None:
        out.append(f"**NCCL ↔ compute overlap (rank 0)**: {overlap_pct:.1f}% of NCCL time is hidden behind compute.\n")
        out.append("Higher = better. <60% means a significant share of NCCL serializes against compute → bottleneck candidate B.\n\n")

    out.append("## 8. Memcpy (steady-state, all ranks)\n\n")
    out.append("| kind | count | GB | total ms |\n|---|---:|---:|---:|\n")
    for k, c, gb, ms in memcpy_rows:
        out.append(f"| {k} | {c} | {gb:.2f} | {ms:.0f} |\n")
    out.append("\n### H2D streams (per device)\n\n")
    out.append("If H2D shares the compute stream, GPU memcpy serializes against compute even with non_blocking=True.\n\n")
    out.append("| device | streamId | GB | ms |\n|---|---:|---:|---:|\n")
    for dev, sid, gb, ms in h2d_streams:
        out.append(f"| {dev} | {sid} | {gb:.2f} | {ms:.0f} |\n")
    if h2d_overlap_pct is not None:
        out.append(f"\n**Rank-0 H2D total**: {h2d_total_ms:.0f} ms\n")
        out.append(f"**H2D ↔ compute overlap (rank 0)**: {h2d_overlap_pct:.1f}% of H2D time runs while a kernel is also executing.\n")
        out.append("Low overlap (e.g. <70%) and idle gap before first kernel of each step → bottleneck candidate C (explicit copy stream may help).\n")
    out.append("\n")

    out.append("## 9. Attention kernels (multi-crop signature, Hypothesis A)\n\n")
    out.append("DINOv3 runs 2 globals (seq=197) and 8 locals (seq=37) sequentially through the same blocks. We expect a bimodal distribution of attention-kernel durations.\n\n")
    out.append("| name | count | total ms | avg us | min us | max us |\n|---|---:|---:|---:|---:|---:|\n")
    for name, cnt, tot_ms, avg_us, min_us, max_us in attn_kernel_rows:
        out.append(f"| `{(name or '')[:90]}` | {cnt} | {tot_ms:.0f} | {avg_us:.1f} | {min_us:.1f} | {max_us:.1f} |\n")
    if attn_durs_ms:
        out.append(f"\nRank-0 per-event attention duration percentiles: {fmt_pct(percentiles(attn_durs_ms))}\n")
        # crude bimodality check: ratio max/min on a single kernel name
        if attn_kernel_rows:
            top = attn_kernel_rows[0]
            out.append(f"\nTop-attention kernel `{(top[0] or '')[:60]}` min/max ratio: {top[5] / max(1.0, top[4]):.1f}× — high ratio (e.g. ≥3×) = bimodal seq-length signature confirmed.\n")
    out.append("\n")

    out.append("## 10. Top runtime APIs (steady-state)\n\n")
    out.append("| api | count | total ms |\n|---|---:|---:|\n")
    for api, cnt, ms in runtime_rows:
        out.append(f"| `{api}` | {cnt} | {ms:.0f} |\n")
    out.append("\n")

    # ----- Quick verdict -------------------------------------------------------
    out.append("## 11. Quick verdict (mechanical, take with a grain of salt)\n\n")
    out.append("Anchored to the four hypotheses in `docs/phase5_perf_plan.md`:\n\n")
    if union_gaps_ms:
        gp = percentiles(union_gaps_ms)
        if gp and gp["p99"] > 50:
            out.append(f"- **D (periodic stalls)**: union p99 gap = {gp['p99']:.1f} ms. Rare large gaps point at GC/eval/checkpoint or rank-straggler resync. Inspect top union gaps above.\n")
        else:
            out.append(f"- **D (periodic stalls)**: union p99 gap = {gp['p99'] if gp else 0:.1f} ms — clean, no obvious global stalls.\n")
    if overlap_pct is not None:
        if overlap_pct < 60:
            out.append(f"- **B (NCCL serialization)**: NCCL overlap with compute is **{overlap_pct:.1f}%** — tune wrap granularity / try `reshard_after_forward=False` to test.\n")
        else:
            out.append(f"- **B (NCCL serialization)**: NCCL overlap is **{overlap_pct:.1f}%** — comms reasonably hidden, not the dominant bottleneck.\n")
    if h2d_overlap_pct is not None:
        if h2d_overlap_pct < 70:
            out.append(f"- **C (H2D not overlapped)**: H2D overlap with compute is **{h2d_overlap_pct:.1f}%** — explicit copy_stream pipeline would help.\n")
        else:
            out.append(f"- **C (H2D not overlapped)**: H2D overlap is **{h2d_overlap_pct:.1f}%** — pinned + non_blocking already doing its job.\n")
    if attn_kernel_rows:
        top = attn_kernel_rows[0]
        ratio = top[5] / max(1.0, top[4])
        if ratio >= 3.0:
            out.append(f"- **A (multi-crop sequential forward)**: top-attn kernel min/max ratio = {ratio:.1f}× → bimodal global+local signature confirmed. Packed-attention rework is the structural lever.\n")
        else:
            out.append(f"- **A (multi-crop sequential forward)**: top-attn kernel min/max ratio = {ratio:.1f}× — less bimodal than expected; check whether local crops are batched together or hidden in another kernel name.\n")
    out.append("\n---\n_End of report._\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(out), encoding="utf-8")
    print(f"Wrote: {out_path}")
    db.close()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", type=Path, help="Path to .sqlite from `nsys export --type=sqlite`")
    ap.add_argument("--out", type=Path, default=None, help="Output markdown path (default: <sqlite>.summary.md)")
    ap.add_argument("--steady-skip-s", type=float, default=10.0,
                    help="Seconds of GPU activity to skip at the start (compile + init warmup)")
    ap.add_argument("--top-n", type=int, default=20, help="Top-N kernels to report (by total time)")
    args = ap.parse_args()
    if not args.sqlite.exists():
        print(f"not found: {args.sqlite}", file=sys.stderr)
        sys.exit(2)
    out = args.out or args.sqlite.with_suffix(".summary.md")
    sys.exit(main(args.sqlite, out, args.steady_skip_s, args.top_n))
