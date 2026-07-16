#!/usr/bin/env python3
"""Project-agnostic gate engine for the T2 autoresearch-loop verifier.

This is the REUSABLE core. It knows nothing about any specific training project;
a project plugs in via an `Adapter` (see `adapters/dinov3.py` for the worked
example and `ADAPTERS.md` for the authoring contract). To stand up the verifier
for a new PyTorch training project you copy this file verbatim and write one
small adapter — never edit the logic here.

Design rules (docs/html/explainers/autoresearch-loops-mfu-2026-07-03.html + four
Codex review passes: two GPT-5.5 high 2026-07-03, one GPT-5.6-sol xhigh 2026-07-10,
one GPT-5.6-sol high 2026-07-16 on the committed toolkit — see README validation log):

  * PRIMARY SCORE = steady-state WALL img/s (mean-based, i.e. integrated
    throughput — stalls count). The adapter's diagnostic metric (MFU for dinov3)
    is a DIAGNOSTIC ONLY: phase7 §7.2 (jobs 77671/77672) proved a GPU-window
    metric can improve 9.5% while wall throughput worsens 7.7%. Never optimize a
    partial window.
  * All numbers re-derived from raw JSONL — never trust a run's self-report.
  * Gates are MANDATORY or ADVISORY (see the sets below). A mandatory FAIL makes
    the verdict "invalid"; a mandatory not_evaluated makes it "incomplete" (the
    evidence is missing — NOT a silent pass). Advisory gates never block.
  * objective_integrity gate: the candidate's effective config is diffed against
    the baseline's; any non-allowlisted difference FAILS. This is the anti-gaming
    gate — a faster run that got faster by changing the objective (loss weights,
    crops, arch, schedule length) is invalid, not a win.
  * The completed gate should get --expect-iters FROM THE DRIVER. If it has to
    infer from the run's own config, that value is candidate-controlled and the
    verdict says so in a note.
  * Statistics honesty: logged points are rolling averages (autocorrelated, NOT
    independent samples). A single replicate never promotes; deltas inside the
    noise floor are "no-evidence".

Deployment: run the INSTALLED copy (~/scripts/loop-verifier/, see
install_verifier.sh) when verifying loop candidates. The trusted combined sha256
digest must be recorded in the loop's state.md (outside the verifier directory —
a hash file next to the target only detects accidents, not replacement).

Stdlib only (yaml used opportunistically for the integrity diff). Exit code:
0 = scored (gates may still FAIL — read the JSON), 2 = could not score.
"""

import argparse
import copy
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

SCHEMA_VERSION = 3  # v3: mandatory/advisory gate split (not_evaluated no longer fails open),
#                     timing_coverage gate, type-aware integrity diff, unrounded comparison

# Gate taxonomy (Codex GPT-5.6 review, 2026-07-10, finding 1). A gate that is
# `not_evaluated` used to count as a silent pass, so a candidate could DISABLE a
# gate's input (e.g. EXTRA_ENV=DINOV3_PERRANK_DIAG=0 → straggler gate has no data
# → not_evaluated → "candidate-improvement") and dodge it. MANDATORY gates must
# be an explicit "pass" to certify a comparison verdict; a `not_evaluated`
# mandatory gate downgrades the verdict to "incomplete" (missing evidence), it
# does NOT pass. ADVISORY gates are honest infra stubs (no parser wired yet) and
# stay non-blocking until a mission needs them.
MANDATORY_GATES = frozenset(
    {
        "completed",
        "finite_loss",
        "timing_coverage",
        "wall_tail",
        "data_time_rank0",
        "per_rank_straggler",
        "objective_integrity",
        "loss_envelope",
    }
)
ADVISORY_GATES = frozenset({"peak_vram", "host_rss_drift"})

_MISSING = ("<missing-key>",)  # integrity-diff sentinel: distinct from any real (type, value) pair


@dataclass
class Columns:
    """Maps this project's raw-metrics JSONL keys onto the roles the engine needs.

    `diag_metric_*` is the "MFU-like" GPU-window metric used only for the §7.2
    divergence detector. `extra_diag` are (source_key, output_key) pairs surfaced
    verbatim in the diagnostics block.
    """

    iteration: str
    iter_time: str
    batch_size: str
    loss: str
    data_time: str
    diag_metric_src: str
    diag_metric_out: str
    diag_metric_label: str
    extra_diag: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class Adapter:
    """Everything project-specific the engine needs. See ADAPTERS.md."""

    name: str
    metrics_file: str  # JSONL, one row per logged point (rank 0)
    config_file: str  # serialized effective run config, for the integrity diff
    columns: Columns
    allowed_diff_keys: Set[str]  # config keys a candidate may change (the menu)
    ignored_diff_keys: Set[str]  # run-identity keys with no objective content
    # fn(run_dir) -> expected total iters (or None). Read from the run's config.
    read_expected_iters: Optional[Callable[[str], Optional[int]]] = None
    # fn(slurm_log, skip_iters) -> (per_rank_avg, per_rank_max) dicts[int]->[float].
    parse_per_rank: Optional[Callable[[str, int], Tuple[dict, dict]]] = None
    # fn(run_dir) -> [str]: project-specific advisory notes (e.g. dinov3's nan_logs/
    # check). Kept out of the engine so the core carries no project conventions.
    extra_notes: Optional[Callable[[str], List[str]]] = None
    # Expected number of ranks at the operating point, so the straggler gate can
    # FAIL when the log is missing a rank instead of passing on the survivors
    # (finding 6). None disables the coverage check. --world-size overrides it.
    expected_world_size: Optional[int] = None


# ---------------------------------------------------------------------------
# pure utilities
# ---------------------------------------------------------------------------
def pctl(values, q):
    """Nearest-rank percentile; q in [0,100]. Caller guarantees non-empty."""
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(q / 100.0 * len(s)) - 1))
    return s[k]


def load_rows(metrics_path):
    rows = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def col(rows, key):
    return [r[key] for r in rows if key in r and r[key] is not None and math.isfinite(r[key])]


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def straggler_gate(per_rank_avg, per_rank_max, straggler_max_s, expected_world_size=None, min_points=20):
    """Per-rank loader-starvation gate from already-parsed per-rank data_time.

    Parsing is the adapter's job (log format is project-specific); the STATISTICS
    are generic. Rank-0's data_time in the metrics file cannot see other ranks'
    starvation (job 80711: rank-7 at 0.37-0.42 s/iter was invisible to rank 0).
    """
    if not per_rank_avg:
        return {
            "status": "not_evaluated",
            "value": None,
            "threshold": "no per-rank data_time parsed from the slurm log (adapter.parse_per_rank returned empty)",
        }, None
    # Coverage (finding 6): a nonempty dict does NOT mean every rank was observed.
    # A log that dropped the starving rank, or has too few steady-state points to
    # estimate a p95, must FAIL (blind), not pass on the survivors.
    observed = sorted(per_rank_avg)
    missing = sorted(set(range(expected_world_size)) - set(observed)) if expected_world_size else []
    thin = sorted(r for r, v in per_rank_avg.items() if len(v) < min_points)
    # Gate on RAW p95, NOT median and NOT a rounded p95 (findings 9 + episodic-stall
    # lesson). 80711: every rank median < 2ms but ranks 0/1/2/6/7 p95 up to 0.44s;
    # and a true p95 of 0.10004s must not round down to 0.1000 and pass a 0.1s cap.
    raw_p95 = {r: pctl(v, 95) for r, v in per_rank_avg.items()}
    per_rank_stats = {
        r: {
            "median": round(statistics.median(per_rank_avg[r]), 4),
            "p95": round(raw_p95[r], 4),
            "frac_over_50ms": round(sum(1 for x in per_rank_avg[r] if x > 0.05) / len(per_rank_avg[r]), 2),
            "n": len(per_rank_avg[r]),
        }
        for r in observed
    }
    worst_rank = max(raw_p95, key=lambda r: raw_p95[r])
    worst_p95_raw = raw_p95[worst_rank]
    diag = {
        "per_rank_data_time_s": per_rank_stats,
        "worst_rank": worst_rank,
        "worst_rank_data_time_max_p95_s": (
            round(pctl(per_rank_max[worst_rank], 95), 4) if per_rank_max.get(worst_rank) else None
        ),
        "n_ranks": len(per_rank_stats),
        "expected_world_size": expected_world_size,
        "observed_ranks": observed,
    }
    if missing or thin:
        reason = []
        if missing:
            reason.append(f"missing rank(s) {missing} of world_size {expected_world_size}")
        if thin:
            reason.append(f"< {min_points} steady-state points for rank(s) {thin}")
        return {
            "status": "fail",
            "value": "insufficient per-rank coverage: " + "; ".join(reason),
            "threshold": f"every rank in [0,{expected_world_size}) present with >= {min_points} points and p95 <= {straggler_max_s}s",
        }, diag
    return {
        "status": "pass" if worst_p95_raw <= straggler_max_s else "fail",
        "value": f"worst rank {worst_rank}: p95 data_time {round(worst_p95_raw, 4)}s",
        "threshold": f"every rank's p95 steady-state data_time <= {straggler_max_s}s (episodic-stall detector)",
    }, diag


def integrity_gate(cand_dir, base_dir, config_file, allow_keys, ignore_keys):
    """Diff candidate effective config vs baseline; non-allowlisted diff = FAIL.

    Catches work-reduction gaming (loss weights, crops, arch, schedule length)
    that the loss-envelope gate cannot: lower loss from an easier objective
    passes the envelope but is not the same experiment.
    """
    try:
        import yaml
    except ImportError:
        return {
            "status": "not_evaluated",
            "value": None,
            "threshold": "pyyaml unavailable — run with a python that has it installed",
        }
    c, b = Path(cand_dir) / config_file, Path(base_dir) / config_file
    if not (c.is_file() and b.is_file()):
        return {
            "status": "not_evaluated",
            "value": None,
            "threshold": f"{config_file} missing in candidate or baseline",
        }
    cf, bf = flatten(yaml.safe_load(c.read_text())), flatten(yaml.safe_load(b.read_text()))

    # Type-aware, missing-sentinel comparison (finding 7). Plain `!=` collapses
    # True==1 / False==0 and treats a MISSING key the same as an explicit null, so
    # a type-changing edit to a protected key could disappear from the diff. This
    # anti-gaming gate fails CLOSED: any representational difference is a diff to
    # review; the allowlist still lets intended knobs through.
    def canon(d, k):
        return _MISSING if k not in d else (type(d[k]).__name__, d[k])

    diffs = {
        k: {"baseline": bf.get(k), "candidate": cf.get(k)}
        for k in sorted(set(cf) | set(bf))
        if canon(cf, k) != canon(bf, k) and k not in ignore_keys
    }
    violations = {k: v for k, v in diffs.items() if k not in allow_keys}
    return {
        "status": "fail" if violations else "pass",
        "value": {"allowed_diffs": {k: v for k, v in diffs.items() if k in allow_keys}, "violations": violations},
        "threshold": f"{config_file} may differ from baseline only in {sorted(allow_keys)}",
    }


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score_run(run_dir, args, adapter):
    run_dir = Path(run_dir)
    C = adapter.columns
    metrics_path = run_dir / adapter.metrics_file
    if not metrics_path.is_file():
        raise FileNotFoundError(f"no {adapter.metrics_file} in {run_dir}")
    rows = load_rows(metrics_path)
    if not rows:
        raise ValueError(f"empty {adapter.metrics_file} in {run_dir}")

    skip_iters = args.skip_iters
    expect = args.expect_iters
    expect_source = "driver"
    if expect is None:
        expect = adapter.read_expected_iters(str(run_dir)) if adapter.read_expected_iters else None
        expect_source = f"run {adapter.config_file} (candidate-controlled)"
    max_iter = max(r.get(C.iteration, 0) for r in rows)
    ss = [r for r in rows if r.get(C.iteration, 0) >= skip_iters]

    notes = []
    gates = {}

    # --- gate: run completed (final logged row is iteration n-1) --------------
    if expect is not None:
        gates["completed"] = {
            "status": "pass" if max_iter >= expect - 1 else "fail",
            "value": max_iter,
            "threshold": f">= {expect - 1} (expect_iters={expect}, source: {expect_source})",
        }
        if expect_source != "driver":
            notes.append(
                f"expect_iters inferred from the run's OWN {adapter.config_file} — a candidate that "
                "shortens the schedule would self-attest completion; the driver should "
                "always pass --expect-iters (and integrity_gate flags schedule diffs)"
            )
    else:
        gates["completed"] = {
            "status": "not_evaluated",
            "value": max_iter,
            "threshold": "expect_iters unknown (no config match; pass --expect-iters)",
        }

    if not ss:
        raise ValueError(f"no steady-state points (max_iter={max_iter}, skip_iters={skip_iters}) in {run_dir}")

    # --- gate: finite loss ---------------------------------------------------
    raw_losses = [r.get(C.loss) for r in rows if r.get(C.iteration, 0) > 0]
    nonfinite = [x for x in raw_losses if x is None or not math.isfinite(x)]
    gates["finite_loss"] = {
        "status": "fail" if nonfinite else "pass",
        "value": f"{len(nonfinite)} non-finite of {len(raw_losses)} points",
        "threshold": f"0 non-finite {C.loss} after iter 0",
    }

    if adapter.extra_notes:
        notes.extend(adapter.extra_notes(str(run_dir)))

    # --- steady-state series -------------------------------------------------
    # A row is USABLE for scoring only if it carries BOTH a positive-finite
    # iter_time (the denominator) AND a positive-finite batch_size (the numerator)
    # on the SAME row (Codex 2026-07-16, finding 4). col() alone would (a) accept a
    # zero/negative iter_time — a division blow-up or a negative "score" — and (b)
    # let batch_size appear on a single row while iter_time is present on many, so
    # one inflated batch_size value could set the numerator for the whole run.
    # Pairing per-row closes both.
    def _pos(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0

    paired = [(r[C.iter_time], r[C.batch_size]) for r in ss if _pos(r.get(C.iter_time)) and _pos(r.get(C.batch_size))]
    it = [t for t, _ in paired]
    gbs_series = [b for _, b in paired]
    gbs = gbs_series[0] if gbs_series else None
    if not it or not gbs:
        raise ValueError(f"no steady-state row has positive {C.iter_time} AND {C.batch_size} in {run_dir}")

    # --- gate: timing coverage (findings 2 + 4) ------------------------------
    # Without this gate a run whose steady-state window is usable on only one row
    # would score off that single fast row (wall_tail ratio 1.0, completion pass).
    # Require most steady-state rows to be USABLE (positive iter_time AND batch
    # size paired on the same row), a floor on the count, and a CONSTANT batch
    # size across those usable rows (gbs is the score numerator).
    n_ss = len(ss)
    n_usable = len(paired)
    coverage = n_usable / n_ss if n_ss else 0.0
    gbs_constant = len(set(gbs_series)) == 1
    timing_ok = n_usable >= args.min_ss_points and coverage >= args.min_timing_coverage and gbs_constant
    gates["timing_coverage"] = {
        "status": "pass" if timing_ok else "fail",
        "value": (
            f"{n_usable}/{n_ss} steady-state rows have positive {C.iter_time} AND {C.batch_size} ({coverage:.0%}); "
            + (f"{C.batch_size} constant" if gbs_constant else f"{C.batch_size} VARIES {sorted(set(gbs_series))}")
        ),
        "threshold": (
            f">= {args.min_timing_coverage:.0%} of steady-state rows usable (paired positive "
            f"{C.iter_time}+{C.batch_size}), >= {args.min_ss_points} points, constant {C.batch_size} "
            "(else the score can rest on a few cherry rows)"
        ),
    }

    it_mean, it_med = statistics.mean(it), statistics.median(it)
    it_p95, it_p99 = pctl(it, 95), pctl(it, 99)

    # --- gate: wall tail (catches MAX_CONN=8-style host-tail / episodic stalls)
    tail_ratio = it_p95 / it_med if it_med else None
    gates["wall_tail"] = {
        "status": "pass" if tail_ratio is not None and tail_ratio <= args.tail_ratio_max else "fail",
        "value": round(tail_ratio, 3) if tail_ratio is not None else None,
        "threshold": f"p95(iter_time)/median <= {args.tail_ratio_max}",
    }

    # --- gate: rank-0 data_time (loader starvation, LOCAL view only) ---------
    dt = col(ss, C.data_time)
    dt_med = statistics.median(dt) if dt else None
    gates["data_time_rank0"] = {
        "status": "pass" if dt_med is not None and dt_med <= args.data_time_max_s else "fail",
        "value": round(dt_med, 4) if dt_med is not None else None,
        "threshold": f"median steady-state data_time <= {args.data_time_max_s}s (rank-0 view)",
    }

    # --- gate: per-rank straggler (needs the slurm log + adapter parser) ------
    straggler_diag = None
    if args.slurm_log and adapter.parse_per_rank:
        per_rank_avg, per_rank_max = adapter.parse_per_rank(args.slurm_log, skip_iters)
        ews = args.world_size if args.world_size is not None else adapter.expected_world_size
        gates["per_rank_straggler"], straggler_diag = straggler_gate(
            per_rank_avg, per_rank_max, args.straggler_max_s, expected_world_size=ews
        )
    else:
        gates["per_rank_straggler"] = {
            "status": "not_evaluated",
            "value": None,
            "threshold": "pass --slurm-log (and the adapter must supply parse_per_rank)",
        }

    # Honest blind spots of v2 (parsers to add when a mission needs them):
    gates["peak_vram"] = {
        "status": "not_evaluated",
        "value": None,
        "threshold": "not in metrics file; parse 'max mem' from slurm log",
    }
    gates["host_rss_drift"] = {
        "status": "not_evaluated",
        "value": None,
        "threshold": "needs node-level sampling during the run",
    }

    # --- score + diagnostics --------------------------------------------------
    def dstat(key):
        v = col(ss, key)
        if not v:
            return None
        return {
            "mean": round(statistics.mean(v), 4),
            "median": round(statistics.median(v), 4),
        }

    loss_ss = col(ss, C.loss)
    tail_window = [r for r in ss if r.get(C.iteration, 0) >= max_iter - max(50, int(0.1 * max_iter))]
    # Guard the median: tail_window can be non-empty while col() filters out every
    # loss (all NaN/Inf/missing). statistics.median([]) raises StatisticsError
    # (a ValueError subclass — it'd be caught as "cannot score" and exit 2), but a
    # NaN run should score to a gated invalid verdict (finite_loss=FAIL) WITH JSON,
    # not vanish as unscoreable. So loss_final stays None and scoring continues.
    tail_losses = col(tail_window, C.loss)
    loss_final = statistics.median(tail_losses) if tail_losses else None

    diagnostics = {
        "iter_time_s": {
            "mean": round(it_mean, 4),
            "median": round(it_med, 4),
            "p95": round(it_p95, 4),
            "p99": round(it_p99, 4),
            "std": round(statistics.stdev(it), 4) if len(it) > 1 else None,
        },
        C.diag_metric_out: dstat(C.diag_metric_src),
    }
    for src, out in C.extra_diag:
        diagnostics[out] = dstat(src)
    diagnostics.update(
        {
            "data_time_s": {
                "median": round(dt_med, 4) if dt_med is not None else None,
                "p95": round(pctl(dt, 95), 4) if dt else None,
            },
            "per_rank": straggler_diag,
            "loss_final_window_median": round(loss_final, 4) if loss_final is not None else None,
            "loss_ss_median": round(statistics.median(loss_ss), 4) if loss_ss else None,
            "global_batch_size": gbs,
        }
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "job_id": (re.search(r"(\d+)$", run_dir.name) or [None, None])[1],
        "window": {
            "skip_iters": skip_iters,
            "iter_max": max_iter,
            "n_points": len(it),
            "stats_caveat": "points are rolling averages — autocorrelated, not independent",
        },
        "score": {
            "primary": "wall_imgs_mean",
            "wall_imgs_mean": round(gbs / it_mean, 1),
            "wall_imgs_median": round(gbs / it_med, 1),
            # Full-precision values for the delta math (finding 10): comparing
            # rounded scores lets a delta sitting on the noise floor misclassify.
            "wall_imgs_mean_raw": gbs / it_mean,
            "wall_imgs_median_raw": gbs / it_med,
            "why_mean": "integrated throughput — stalls count; median reported as stall-insensitive diagnostic",
        },
        "diagnostics": diagnostics,
        "gates": gates,
        "notes": notes,
    }
    return result


def finalize_gates(result):
    gates = result["gates"]
    result["gates_passed"] = all(g["status"] != "fail" for g in gates.values())
    result["gates_not_evaluated"] = sorted(k for k, g in gates.items() if g["status"] == "not_evaluated")
    # A mandatory gate is "met" only when it is an explicit pass. not_evaluated is
    # NOT a pass — it means the evidence is missing (finding 1). mandatory_unmet
    # collects every mandatory gate that is present and not a clean pass, split so
    # the verdict can say "failed" (real regression) vs "not evaluated" (blind).
    result["mandatory_failed"] = sorted(k for k, g in gates.items() if k in MANDATORY_GATES and g["status"] == "fail")
    result["mandatory_unevaluated"] = sorted(
        k for k, g in gates.items() if k in MANDATORY_GATES and g["status"] == "not_evaluated"
    )
    # AUTHORITATIVE certification boolean (Codex 2026-07-16, finding 3). gates_passed
    # above means only "no explicit FAIL" — it is still true when a mandatory gate is
    # not_evaluated, so a driver that keyed off it would reopen the finding-1 fail-open
    # hole. mandatory_gates_passed requires every mandatory gate to be an explicit pass
    # (no fail AND no not_evaluated). This is the boolean a driver should gate on; the
    # verdict string carries the same distinction ("invalid"/"incomplete").
    result["mandatory_gates_passed"] = not result["mandatory_failed"] and not result["mandatory_unevaluated"]


def compare(cand, base, base_dir, args, adapter):
    """Attach baseline comparison, integrity gate, and verdict (in place)."""
    C = adapter.columns
    allow_keys = set(adapter.allowed_diff_keys)
    if args.allow_diff_keys:
        allow_keys |= {k.strip() for k in args.allow_diff_keys.split(",") if k.strip()}

    # objective integrity: needs the baseline's config on disk
    if base_dir is not None:
        cand["gates"]["objective_integrity"] = integrity_gate(
            cand["run_dir"], base_dir, adapter.config_file, allow_keys, adapter.ignored_diff_keys
        )
    else:
        cand["gates"]["objective_integrity"] = {
            "status": "not_evaluated",
            "value": None,
            "threshold": "baseline given as JSON — pass a baseline RUN DIR to enable the config diff",
        }

    # Full-precision delta (finding 10). Fall back to the rounded field for old
    # cached baseline JSON that predates the *_raw fields.
    c = cand["score"].get("wall_imgs_mean_raw", cand["score"]["wall_imgs_mean"])
    b = base["score"].get("wall_imgs_mean_raw", base["score"]["wall_imgs_mean"])
    delta_pct = 100.0 * (c - b) / b

    cl = cand["diagnostics"]["loss_final_window_median"]
    bl = base["diagnostics"]["loss_final_window_median"]
    if cl is not None and bl is not None:
        loss_delta_pct = 100.0 * (cl - bl) / abs(bl)
        cand["gates"]["loss_envelope"] = {
            "status": "pass" if loss_delta_pct <= args.loss_tol_pct else "fail",
            "value": round(loss_delta_pct, 2),
            "threshold": f"final-window median loss no more than {args.loss_tol_pct}% above baseline "
            "(a much LOWER loss is also suspicious — integrity gate checks the objective)",
        }
        if loss_delta_pct < -args.loss_tol_pct:
            cand["notes"].append(
                f"loss is {abs(loss_delta_pct):.1f}% BELOW baseline — verify the objective is "
                "unchanged (integrity gate) before reading this as the same experiment"
            )
    else:
        cand["gates"]["loss_envelope"] = {
            "status": "not_evaluated",
            "value": None,
            "threshold": "loss missing in candidate or baseline",
        }

    finalize_gates(cand)

    if cand["mandatory_failed"]:
        verdict = f"invalid (gate failure: {', '.join(cand['mandatory_failed'])})"
    elif cand["mandatory_unevaluated"]:
        # not_evaluated mandatory gate = missing evidence, NOT a pass. Refuse to
        # certify a promote/regress verdict; the driver must supply the missing
        # input (e.g. --slurm-log for per_rank_straggler, a baseline RUN DIR for
        # objective_integrity) and re-score. This is the finding-1 fix: a candidate
        # cannot dodge a gate by disabling its input.
        verdict = (
            f"incomplete (mandatory gate(s) not evaluated: {', '.join(cand['mandatory_unevaluated'])}"
            f"; delta {delta_pct:+.2f}% is provisional — supply the missing input and re-score)"
        )
    elif delta_pct > args.noise_floor_pct:
        verdict = "candidate-improvement (single replicate — confirm with matched-node A/B before accepting)"
    elif delta_pct < -args.noise_floor_pct:
        verdict = "candidate-regression (single replicate)"
    else:
        verdict = "no-evidence (within noise floor)"

    cand["baseline"] = {
        "run_dir": base["run_dir"],
        "wall_imgs_mean": round(b, 1),
        "delta_pct": round(delta_pct, 2),
        "noise_floor_pct": args.noise_floor_pct,
        "self_reported": base_dir is None,  # JSON baseline: numbers are not re-derived (finding 5)
    }
    cand["verdict"] = verdict

    # Baseline trust (finding 5): a JSON baseline's numbers are self-reported (not
    # re-derived from raw artifacts), and even a rescored baseline may itself have
    # failed a gate. Surface both — a delta is only as trustworthy as its baseline.
    if base_dir is None:
        cand["notes"].append(
            "baseline is a self-reported score JSON — its wall_imgs_mean is NOT re-derived from raw "
            "metrics; pass a baseline RUN DIR so the engine rescores it (and enables objective_integrity)"
        )
    else:
        # Only a real FAIL on the baseline is disqualifying. not_evaluated on the
        # baseline is expected here (we rescore it WITHOUT the candidate's slurm
        # log — see load_baseline — so its straggler gate is simply not re-checked;
        # the baseline's straggler was certified when it was promoted).
        base_bad = sorted(base.get("mandatory_failed", []))
        if base_bad:
            cand["notes"].append(
                f"baseline itself FAILED mandatory gate(s): {', '.join(base_bad)} — the delta is "
                "measured against a run that is not certified; re-establish a clean champion"
            )

    # §7.2 divergence detector: GPU-window diagnostic "improved" while wall
    # regressed past the noise floor. Assumes the diagnostic is higher-is-better
    # (true for MFU/img-s; a lower-is-better latency proxy would need a signed
    # direction on Columns — see ADAPTERS.md). BOTH series are noise-floor gated
    # (Codex findings 2026-07-10 #10 + 2026-07-16 #8): the wall must regress past
    # the floor AND the diagnostic must IMPROVE past the floor, so sub-noise wobble
    # in either series (e.g. MFU 20.0000 -> 20.0001) cannot manufacture a divergence.
    label = C.diag_metric_label
    cm, bm = cand["diagnostics"][C.diag_metric_out], base["diagnostics"][C.diag_metric_out]
    diag_gain_pct = 100.0 * (cm["mean"] - bm["mean"]) / abs(bm["mean"]) if (cm and bm and bm["mean"]) else 0.0
    if cm and bm and diag_gain_pct > args.noise_floor_pct and delta_pct < -args.noise_floor_pct:
        cand["notes"].append(
            f"DIVERGENCE (§7.2 pattern): {label} improved while wall img/s regressed "
            f"{abs(delta_pct):.1f}% (past the {args.noise_floor_pct}% floor) — cost moved outside the "
            f"CUDA-event window; do NOT read the {label} gain as a win"
        )


def load_baseline(path, args, adapter):
    """Returns (baseline_result, baseline_run_dir_or_None).

    The baseline is ALWAYS re-derived from raw artifacts when a run directory is
    available — a cached score JSON is never trusted for the delta if the run it
    names still exists on disk (Codex GPT-5.6-sol review, 2026-07-16, finding 1).
    A prior version returned the JSON's stored score while ALSO returning its
    run_dir, so compare() would mark it re-derived (self_reported=false) and use a
    number that was never recomputed: editing wall_imgs_mean_raw in a saved JSON
    (real dir retained) manufactured a "trusted" improvement. Now a JSON that
    points at a live dir is treated exactly like passing that dir directly; only a
    JSON whose run_dir is gone stays self-reported (and compare() warns).
    """

    # Rescore any baseline run dir WITHOUT the candidate's --slurm-log: that log
    # is the CANDIDATE's, so feeding it to the baseline would compute the
    # baseline's straggler gate from the wrong run. The baseline's straggler was
    # certified at promotion time; here it stays not_evaluated (and the note logic
    # in compare() only flags a real baseline FAIL, so this adds no noise).
    def rescore(run_dir):
        base_args = copy.copy(args)
        base_args.slurm_log = None
        r = score_run(run_dir, base_args, adapter)
        finalize_gates(r)
        return r

    p = Path(path)
    if p.is_file() and p.suffix == ".json" and p.name != adapter.metrics_file:
        result = json.loads(p.read_text())
        rd = Path(result.get("run_dir", ""))
        if rd.is_dir():
            # Live run dir: re-derive the score from raw artifacts, ignore the
            # JSON's stored numbers, and enable the objective_integrity diff.
            return rescore(rd), rd
        # Dir is gone: fall back to the self-reported JSON (compare() warns and
        # marks self_reported=true; no integrity diff possible).
        return result, None
    return rescore(p), p


def human_summary(r, adapter):
    C = adapter.columns
    diag = r["diagnostics"]
    extra_out = C.extra_diag[0][1] if C.extra_diag else None
    extra_str = f"  {extra_out} {diag.get(extra_out)}" if extra_out else ""
    lines = [
        f"run       : {r['run_dir']}  (job {r['job_id']})",
        f"window    : iter >= {r['window']['skip_iters']}, n={r['window']['n_points']} points (autocorrelated)",
        f"SCORE     : {r['score']['wall_imgs_mean']} wall img/s (mean)   [median-based: {r['score']['wall_imgs_median']}]",
        f"diag      : {C.diag_metric_label.lower()} {diag[C.diag_metric_out]}{extra_str}",
        f"            iter_time {diag['iter_time_s']}",
        "gates     : " + "  ".join(f"{k}={v['status'].upper()}" for k, v in r["gates"].items()),
        f"gates_passed = {r['gates_passed']}"
        + (f"   (not evaluated: {', '.join(r['gates_not_evaluated'])})" if r["gates_not_evaluated"] else ""),
    ]
    if r.get("baseline"):
        lines.append(
            f"vs base   : {r['baseline']['delta_pct']:+.2f}% (floor ±{r['baseline']['noise_floor_pct']}%) → {r['verdict']}"
        )
    ig = r["gates"].get("objective_integrity", {})
    if ig.get("status") == "fail":
        lines.append(f"INTEGRITY : VIOLATIONS {json.dumps(ig['value']['violations'])}")
    for n in r.get("notes", []):
        lines.append(f"NOTE      : {n}")
    return "\n".join(lines)


def build_parser(adapter):
    ap = argparse.ArgumentParser(
        description=f"Hermetic T2 verifier (adapter: {adapter.name}). {__doc__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("run_dir")
    ap.add_argument("--baseline", help="baseline run dir (enables integrity diff) or a prior score JSON")
    ap.add_argument("--expect-iters", type=int, default=None, help="DRIVER-owned run length (do not omit in loops)")
    ap.add_argument("--slurm-log", help="job stdout/stderr log — enables the per-rank straggler gate")
    ap.add_argument("--skip-iters", type=int, default=200)
    ap.add_argument(
        "--min-ss-points", type=int, default=30, help="min usable steady-state points (timing_coverage gate)"
    )
    ap.add_argument(
        "--min-timing-coverage",
        type=float,
        default=0.9,
        help="min fraction of steady-state rows with a finite iter_time",
    )
    ap.add_argument("--noise-floor-pct", type=float, default=5.0)
    ap.add_argument("--loss-tol-pct", type=float, default=5.0)
    ap.add_argument("--tail-ratio-max", type=float, default=2.0)
    ap.add_argument("--data-time-max-s", type=float, default=0.05)
    ap.add_argument("--straggler-max-s", type=float, default=0.1, help="worst-rank p95 data_time gate")
    ap.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="expected rank count for the straggler coverage check (overrides the adapter)",
    )
    ap.add_argument("--allow-diff-keys", help="extra config keys allowed to differ (comma-separated)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of human summary")
    ap.add_argument("--out", help="also write JSON verdict to this file")
    ap.add_argument("--lineage", help="append one JSON line to this lineage file")
    ap.add_argument("--tag", default=None, help="candidate tag for the lineage row")
    ap.add_argument("--hypothesis", default=None, help="one-line hypothesis for the lineage row")
    return ap


def run_cli(adapter, argv=None):
    """Entry point a project's thin score.py calls with its adapter."""
    args = build_parser(adapter).parse_args(argv)

    try:
        result = score_run(args.run_dir, args, adapter)
        if args.baseline:
            base_result, base_dir = load_baseline(args.baseline, args, adapter)
            compare(result, base_result, base_dir, args, adapter)
        else:
            finalize_gates(result)
            result["baseline"] = None
            result["verdict"] = "scored (no baseline given)"
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"score.py: cannot score: {e}", file=sys.stderr)
        sys.exit(2)

    if args.tag:
        result["tag"] = args.tag
    if args.hypothesis:
        result["hypothesis"] = args.hypothesis

    payload = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(payload + "\n")
    if args.lineage:
        with open(args.lineage, "a") as f:
            f.write(json.dumps(result) + "\n")
    print(payload if args.json else human_summary(result, adapter))
