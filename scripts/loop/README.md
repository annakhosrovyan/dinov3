# scripts/loop/ — T2 autoresearch screening-loop toolkit

Standing infrastructure for the config/systems screening loop described in
`docs/html/explainers/autoresearch-loops-mfu-2026-07-03.html` (Codex-gated).
Built 2026-07-03, ahead of first loop use, so the loop only has to *drive* it.

The design contract, in one line: **open-ended proposals, closed-form
acceptance.** The proposer (human or agent) may suggest any candidate from the
menu (+ one justified wildcard); only the hermetic verifier decides better /
worse / invalid.

**New here?** Read `QUICKSTART.md` (run one candidate in 5 min). Porting to
another PyTorch project? Read `ADAPTERS.md` (copy `score_core.py`, write a
~40-line adapter). Extending or understanding *why*? Read `DESIGN.md` (rationale,
architecture, and the dinov3-specific vs generalizable split).

## Components

| File | Role |
|------|------|
| `score_core.py` | **The verifier engine — project-agnostic, reusable.** All gate logic, verdict rules, noise-floor comparison, and the §7.2 divergence detector. Copy verbatim into any project; never edit it per-project. Primary score = steady-state WALL img/s (mean-based). Diagnostic metric (MFU here) is reported only. **Mandatory** gates (a `not_evaluated` one downgrades the verdict to `incomplete`, it does NOT pass): completed (exact `expect-1`; expect-iters DRIVER-owned), finite loss, **timing_coverage** (score can't rest on a few cherry rows), wall-tail ratio, rank-0 data_time (median), **per-rank straggler** (worst-rank raw p95 + rank coverage), **objective_integrity** (type-aware config diff — the anti-gaming gate), loss envelope. **Advisory** stubs (never block): peak_vram, host_rss_drift. Emits JSON (`schema_version: 3`) + optional lineage row. |
| `adapters/dinov3.py` | **The project adapter (~40 lines).** The only dinov3-specific code: file names, the metrics-JSONL column map, `[RANKDATA]` parser, `read_expected_iters`, and the integrity key-sets. Porting = write a sibling of this file (see `ADAPTERS.md`). |
| `score.py` | **The thin CLI.** Wires `score_core` to `adapters/dinov3`. CLI/flags/behavior unchanged from the original single-file scorer. |
| `run_candidate.sh` | **The actuator.** One sbatch job per candidate. Pinned to the 67639/77672 recipe; candidate = `EXTRA_ENV` / `EXTRA_OPTS` / sbatch args. Project-specific bits live in marked `# ===== PROJECT-SPECIFIC =====` blocks. 700 iters, skip 200. Job-end self-report is explicitly non-authoritative. |
| `install_verifier.sh` | **The hermeticity step.** Copies `score_core.py` + `score.py` + `adapters/*.py` to `~/scripts/loop-verifier/` (read-only) and records **one combined sha256** so the loop's writer agent cannot edit its own scorer. Driver compares the combined digest each iteration. |
| `candidates.example.yaml` | **The menu.** Named knobs with ranges + the single wildcard slot. Operating point (bs=128) is not searchable. |

## Validation (done 2026-07-03 — trust basis)

`score.py` was validated against three historical runs with known verdicts
before first use (the harness doctrine: validate against ground truth first):

| Run | Expected (phase7_perf_plan.md §7.2 / INDEX) | score.py output |
|-----|---------------------------------------------|-----------------|
| 77672 (maxconn=1) | wall 2,116 img/s (mean), iter_time med 0.4488, MFU 18.6, step 311.3ms | 2115.6 / 0.4488 / 18.57 / 311.3 ✓ |
| 77671 (maxconn=8) vs 77672 | wall −7.7%, MFU +9.5% (divergence) | −7.67%, verdict `candidate-regression`, divergence note fired ✓ |
| 67639 (2000-iter baseline) | completes 2000; ~1,6xx–1,7xx img/s | completed gate PASS, 1740.4 mean ✓ |
| 77671 vs 77672 integrity | same recipe (MAX_CONN is env-only) | `objective_integrity` PASS, no violations ✓ |
| 67639 vs 77672 integrity | different recipe (2000 vs 700 iters, ckpt on) | FAIL: `train.OFFICIAL_EPOCH_LENGTH`, `checkpointing.period`, `evaluation.eval_period_iterations` → verdict `invalid` ✓ |
| 80711 straggler ([RANKDATA] log) | known rank starvation (phase7 7.8) | FAIL: worst rank 6 p95 0.44s ✓ — **only after switching the gate statistic from median to p95**; every rank's median was < 2ms because starvation is episodic. A median gate passed the exact pathology the gate exists for. |

Both 7767x runs carry non-empty `nan_logs/`; the scorer surfaces that as a
note (it did in validation). The 80711 row is the reason the gate doc says
"episodic-stall detector": tail statistics or nothing.

**Re-validated after the core/adapter split (2026-07-03):** all six cases were
re-run through the refactored `score.py` and diffed against the original
single-file scorer (installed copy). Every number, gate status, and verdict is
identical; the only differences are three descriptive threshold strings that were
deliberately generalized out of dinov3-specific wording. See `ADAPTERS.md` for
the porting contract and `DESIGN.md` §5 for the split.

**Re-validated after the schema-v3 hardening (Codex GPT-5.6-sol xhigh review,
2026-07-10):** the primary numbers are unchanged (77672 → 2115.6; 77671-vs-77672
→ −7.67%; 80711 worst rank 6 p95 → 0.4413s). What changed is verdict *semantics*,
by design (finding 1): a `not_evaluated` **mandatory** gate no longer passes
silently. The 77671-vs-77672 comparison now needs `--slurm-log` — without it the
straggler gate is `not_evaluated` and the verdict is `incomplete` (was the
under-verified `candidate-regression`). The 67639-vs-77672 integrity fail now
reports 4 violations (the type-aware diff also surfaces `train.sharded_eval_checkpoint:
false→true`). New gate `timing_coverage` was proven to fire on a synthetic
cherry-row run (1/35 usable → FAIL) and on a non-constant batch size.

## The loop, when it runs

```
propose (menu + ≤1 wildcard w/ justification)
  → CANDIDATE_TAG=... HYPOTHESIS="..." EXTRA_OPTS="..." sbatch scripts/loop/run_candidate.sh
  → poll: ~/scripts/jobcheck <jobid>          (driver sleeps via ScheduleWakeup)
  → score: python3 -I -B ~/scripts/loop-verifier/score.py <outdir> \
        --baseline <champion_dir> --expect-iters 700 --world-size 8 \
        --slurm-log /mnt/weka/adovlatyan/logs/dinov3-loop-cand_<jobid>.out \
        --lineage <loop_dir>/lineage.jsonl --tag <tag> --hypothesis "..."
        # --expect-iters is DRIVER-owned: omitting it lets a candidate that
        # shortened its own schedule self-attest completion. Without --slurm-log
        # the per_rank_straggler gate is not_evaluated → verdict "incomplete"
        # (a candidate cannot dodge the gate by dropping its evidence).
  → decide: verdict improvement? → schedule matched-node A/B replicate;
            confirmed twice → new champion (update candidates yaml baseline);
            regression/no-evidence → next candidate
  → morning checkpoint: human + Codex review lineage.jsonl before any promotion
```

Rules the driver must enforce (from the explainer, Codex-corrected):

- **Never optimize MFU.** Wall img/s is the score; MFU is reported only.
- **Single replicate never promotes.** `candidate-improvement` triggers a
  matched-node A/B (same node via `--nodelist`), A/B/A for champions. Deltas
  under the 5% noise floor stay `no-evidence` (logged points are
  autocorrelated 20-iter rolling averages, not samples).
- **Verifier hash check** each iteration — re-derive the combined digest over the
  installed `.py` files and compare to the digest recorded in the loop's
  `state.md` at loop start (NOT the `verifier.sha256` file next to the verifier —
  adjacent hash = circular; it detects accidents, not replacement):
  `cd ~/scripts/loop-verifier && find . -type f ! -name 'verifier.sha256' | sort | xargs sha256sum | sha256sum`
  (hashes ALL files, not just `*.py`, so a planted `__pycache__/*.pyc` is caught —
  `score.py` sets `sys.dont_write_bytecode` so a legit run never creates any).
  Invoke the scorer as `python3 -I -B ~/scripts/loop-verifier/score.py …`
  (isolated mode — no `PYTHONPATH`/user-site/`sitecustomize` import hijack). Pass
  `--world-size 8` (or the run's rank count) so the straggler gate can fail on a
  missing rank, not pass on the survivors.
- **Integrity allowlist is the menu.** `objective_integrity` fails any config
  diff outside `train.num_workers`/`train.prefetch_factor` by default; widen
  per-mission with `--allow-diff-keys`. Env-channel knobs (MAX_CONN, OMP,
  `DINOV3_RANK_CPU_SLICE`) don't appear in config.yaml — the driver owns those
  and must record them in the lineage row (`--tag`/`--hypothesis`).
- **When a gaming/inflation pattern is found, patch `score_core.py`** (the engine,
  not the thin `score.py` CLI; reviewed, reinstall via `install_verifier.sh`),
  don't prompt-scold the proposer.
- **Batch size is an operating point.** Champions re-confirmed at bs=192
  before any claim generalizes.
- **Mechanical stops:** max candidates per night, forced blocker report if
  two consecutive candidates fail to launch/score.
- Sweep finished runs into `/mnt/weka/adovlatyan/runs/<phase>/` and append the
  INDEX row (standing convention) — lineage.jsonl is the loop's working record,
  INDEX.md is the durable one.

## First mission (pre-scoped)

Worker-budget × affinity grid — the Phase 7 prime lever (job 80711: rank-7
loader starvation; 8×24=192 workers oversubscribe 192 CPUs):

1. `num_workers ∈ {8, 12, 16, 20}` at bs=128 via `EXTRA_OPTS=train.num_workers=N` — runnable **today**.
2. Per-rank NUMA/CPU slices — `DINOV3_RANK_CPU_SLICE=1` is **already implemented**
   on this branch (dinov3/train/train.py, gated, off by default) and §7.9 slice
   arms are running per `docs/phase7_perf_plan.md`. The loop's job is the
   *joint* grid (slice × worker budget), seeded with the 7.9 single-arm results.

## Known blind spots (v2 — evolve, don't gold-plate)

Two gates still report `not_evaluated`: `peak_vram` (parse `max mem` from the
slurm log) and `host_rss_drift` (needs node-level sampling). Add parsers when a
mission needs them. Also invisible to the `objective_integrity` diff: env-channel
knobs (not in config.yaml) and anything hardcoded in code (e.g. `pin_memory` at
`dinov3/data/loaders.py:246`) — code changes between candidate and baseline are
out of scope for this verifier and belong to `loop-build`'s git-diff review.
