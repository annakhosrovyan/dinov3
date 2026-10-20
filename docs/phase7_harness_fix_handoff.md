# Handoff: Phase 7 harness fixes (Part A) + conclusion restatement (Part B)

**Date**: 2026-07-03 · **Branch**: `exp/ddp-fullgraph-head-static-shapes` · **Repo**: `~/dinov3-performance-optimizations/dinov3`

Two sequential work packages. Part A is executed first (mechanical fixes + regeneration,
acceptance-tested against numbers already hand-derived in the audit). Part B runs only after
Part A's regenerated reports exist (interpretation + doc restatement, Codex-gated).

Shared context: `docs/phase7_harness_audit.md` is the authoritative audit (Codex-gated,
GPT-5.5 high). It found 1 BUG + 2 MISLEADING in `scripts/profiling/phase7_nsys_report.py`
and describes fixes **F1–F5 in its §7** (described, not applied). Do not re-litigate the
audit; implement it.

## Trace inventory (all under `/mnt/weka/adovlatyan/nsys_profiles/phase7/<jobid>/`)

| Job | Config | Notes |
|---|---|---|
| 75082 | bs128, period=1 | original attribution capture; contains stall(s) |
| 80545 | bs192 | 67% of window in dropped stalls — the U-shape culprit |
| 80546 | bs160 | |
| 80547 | bs128, period=10 | metric-gate capture |
| 80711 | bs192, w24, PERRANK_DIAG, Sinkhorn NVTX | contains the 15.4 s rank-7 stall |

Evidence rule: `.sqlite` / `.nsys-rep` / existing `.phase7.md|.json` are **read-only citations —
never overwrite or delete**. All regenerated output is written alongside as `*.phase7.v2.md` /
`*.phase7.v2.json`.

---

## Part A — apply F1–F5 and regenerate (execution agent)

1. Read `docs/phase7_harness_audit.md` fully; implement **F1–F5 from §7** in
   `scripts/profiling/phase7_nsys_report.py` (and the capture script only if a fix touches it).
   Core intent, in priority order:
   - Per-iter NCCL/exposed/all-busy/H2D statistics must be computed **kept-iters-only**
     (numerator and denominator over the same segment set) — this is the 1.93× BUG.
   - Dropped segments must be **surfaced, not hidden**: the `.md` must report count, total
     `dropped_slow_ms`, and per-drop wall + reason (corrupt vs slow), since slow drops are
     signal for the straggler work.
   - Per-rank tables: kept-only basis, and flag any rank whose window contains a dropped
     stall event.
   - Whatever else F4/F5 specify per the report.
2. Keep the JSON schema additive (new fields fine; don't rename existing ones without a
   `schema_version` bump) — downstream `--diff` consumers exist.
3. Regenerate reports for **all 5 traces** → `*.phase7.v2.{md,json}` next to the originals.
4. **Acceptance tests** (pull exact expected values from the audit report — do not trust this
   summary if they disagree; the report wins):
   - 80711 v2 exposed NCCL ≈ **274.7 ms/iter** (kept-only, rank 0), vs 529.786 in v1.
   - 80545/80546/80547 v2 per-image compute is **monotonic decreasing** ≈ 1.635 → 1.568 →
     1.559 ms/img ordering across bs128→160→192 (per the audit's corrected recomputation);
     80711 corroborates ≈ 1.564 at bs192.
   - 80545 v2 per-rank exposed NCCL: all ranks in a narrow band (~46–71 ms kept-only), no
     65/107-vs-880 inversion.
   - 80711 v2 `.md` explicitly shows the 15.4 s dropped stall with its classification.
   - NVTX partition invariant still holds per iter (Σ top-level + untracked ≈ wall).
5. Hygiene: `ruff check` + `ruff format` clean on the touched files; `python -m py_compile`.
   Scratch work in the session scratchpad only.
6. **Codex review of the patch** before declaring done:
   `codex exec --model gpt-5.5 -c model_reasoning_effort=high --sandbox read-only` on the diff
   + acceptance evidence. Verify any flagged error yourself before accepting/rejecting.
7. No GPU jobs. No git commits.
8. Final reply: fix-by-fix summary (F1–F5 → what changed, file:line), acceptance table
   (expected vs measured v2 numbers), Codex outcome, list of v2 artifacts written.

## Part B — restate conclusions from v2 numbers (interpretation agent, Fable)

Preconditions: Part A's `*.phase7.v2.*` files exist for all 5 traces.

1. Read the audit report, all 5 v2 reports, and `docs/phase7_perf_plan.md` (status table +
   ladder + §7.3 + §7.7 + §7.8).
2. Restate in `docs/phase7_perf_plan.md`, citing v2 numbers:
   - **7.7**: retract the per-image U-shape / bs160 sweet spot; replace with monotonic
     per-image compute + episodic-stall residual reading.
   - **7.3**: mark the "−3.5% NCCL" comparison invalid (basis mismatch), keep the 7.8a
     "gate ≈0 E2E" conclusion (unaffected).
   - **7.8**: write the 80711 verdict — rank-7 loader starvation (`[RANKDATA]` 0.37–0.42 s
     vs ~0.3 ms; `[RANKTOPO]` identical affinity masks, 8×24=192 workers on 192 CPUs),
     Sinkhorn NVTX 0.27+0.31% ⇒ 7.4 dead; new prime lever = worker-budget headroom +
     per-rank NUMA-local CPU slices.
   - **Ladder/status table** rows updated accordingly (7.4 → dead/closed; new lever row).
   - Add a short "harness audit" note pointing at `docs/phase7_harness_audit.md` and the
     v2 convention.
3. **Codex-gate the restated text BEFORE finalizing** (same command as Part A step 6); verify
   flagged mechanism errors in code/data yourself.
4. Session memory (`~/.claude/projects/.../memory/project_phase7_comms_residual.md` + index)
   was **already updated on 2026-07-02** with the retractions and the 80711 verdict — only
   touch it if v2 numbers materially contradict what's written there; report any such
   contradiction prominently.
5. Style: `~/CLAUDE.md` writing rules are binding.
6. Final reply: list of doc sections changed, any v2-vs-audit discrepancies found, Codex
   outcome.

## What is explicitly out of scope for both parts

- The straggler-fix experiment itself (worker budget / NUMA affinity) — separate work package.
- Any new GPU job, sweep, or capture.
- `run.sh`, training code, or config changes.
