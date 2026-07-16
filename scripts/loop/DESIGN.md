# scripts/loop/ — Design & Rationale

Comprehensive record of the T2 autoresearch-loop toolkit: why it exists, how each
piece works, the design decisions, and what is reusable outside this repo. For
running it, see `QUICKSTART.md`; for the loop-driver contract, `README.md`.

Built 2026-07-03 on branch `exp/ddp-fullgraph-head-static-shapes`. The conceptual
synthesis it implements is
`docs/html/explainers/autoresearch-loops-mfu-2026-07-03.html` (Codex-gated,
registered in `docs/html/_index.md`).

---

## 1. The question this answers

The lab wanted to use agent loops (`/loop`, `/goal`) for research automation but
didn't know how open-ended they can feasibly be. The synthesis answer:

**Loop open-endedness is verifier-limited, not intelligence-limited.** Feasibility
= (verifier exactness/hermeticity) × (per-evaluation cost), conditional on a
structured search space. The design contract is **open-ended in proposal space,
closed-form in acceptance**. Spend ~80% of loop-design effort on the verifier.

Three tiers were mapped for this lab:
- **T1** kernel evolution (NVIDIA AVO style) — feasible, but the wrong lever here:
  the residual bottleneck is loader starvation / comms, not compute.
- **T2** config/systems screening — **the fit.** Overnight, checkpointed,
  Slurm-driving. This toolkit implements T2.
- **T3** open algorithm discovery — human-led; agents do legwork inside hypotheses.

External evidence that shaped the design (full cites in the HTML explainer and
memory `project-autoresearch-loops`): NVIDIA AVO (arXiv 2603.24517, agent-as-
mutation-operator, two-stage correctness→TFLOPS reward); Sakana CUDA Engineer
(search exploited an eval-harness bug, 100×→1.49× when corrected); METR
(reward-hack rate 30.4% with a visible/editable scorer vs 0.7% opaque — 43×);
RE-Bench time-crossover (agents ~4× humans at 2h, humans ~2× at 32h → size
autonomous units at overnight with a morning human gate); Paradigma/Flywheel (DAG
of hypothesis nodes over MCP, but no verifier mechanism — humans adjudicate).

---

## 2. Architecture

Three roles, deliberately separated so no single component can both do work and
grade it:

```
 proposer            actuator                 verifier              driver
 (human/agent)  →    run_candidate.sh    →    score.py         →   (loop, TBD)
 picks a knob        one sbatch job           hermetic, outside    submit→poll→
 from the menu       pinned recipe            the editable tree    score→decide
```

| File | Role | Editable by writer agent? |
|------|------|---------------------------|
| `candidates.example.yaml` | The menu — bounds the proposal space | yes (it's a spec) |
| `run_candidate.sh` | Actuator — turns a candidate into a pinned sbatch job | yes |
| `score_core.py` + `adapters/dinov3.py` + `score.py` | Verifier — engine + adapter + thin CLI; the one component that decides better/worse | **no (all installed read-only)** |
| `install_verifier.sh` | Copies the three verifier files out of the repo, records one combined digest | n/a |

### Why the verifier lives outside the repo

METR's 43× finding: an agent that can see and edit its own scorer reward-hacks the
scorer, not the task. `install_verifier.sh` copies `score.py` to
`~/scripts/loop-verifier/` (`chmod 555`) and prints a sha256. The driver runs the
*installed* copy and checks the digest each iteration.

**Trust chain caveat (Codex note 5):** an adjacent `.sha256` file only detects
accidents — anyone who can swap the file can swap the hash. The trusted digest
must live in the loop's `state.md` (written at loop start, human-reviewed), and
the check compares the live file against *that* copy. `chmod 555` stops accidental
edits, not the owning user.

**Bytecode coverage (Codex note, 2026-07-03 review):** the digest hashes ALL
installed files, not just `*.py`. A source-only digest had a hole — Python would
import a pre-planted `__pycache__/*.pyc` without any `.py` (or its hash) changing.
Hashing every file closes it, and `score.py` sets `sys.dont_write_bytecode` (plus
the installer purges `__pycache__`) so a legitimate run never creates bytecode
that would perturb the digest.

**Exact-manifest install (Codex GPT-5.6 review, finding 4):** `install_verifier.sh`
builds the four-file manifest in a staging dir and atomically swaps it into place,
so the installed tree is *nothing but* the manifest. The earlier "replace named
files in place" install could leave a pre-existing stray module behind
(`sitecustomize.py`, a shadow `json.py`/`pathlib.py`) that Python would import
*and* that would be folded into the freshly recorded digest — blessing code that
never came from this repo. The driver invokes the scorer as `python3 -I -B`
(isolated: no `PYTHONPATH`, user-site, or `sitecustomize` import hijack).

### Residual hermeticity gaps (Codex GPT-5.6 review, driver-scope)

The verifier immutability protects the *scorer*, not the *evidence*. Two holes
remain and belong to the not-yet-built driver, not to these files:

- **The actuator is agent-editable (findings 3).** A writer that can edit
  `run_candidate.sh` (or inject `EXTRA_ENV=PATH=…`) can point `torchrun` at a fake
  binary that writes forged `config.yaml`/metrics/`[RANKDATA]` and exits 0 — the
  scorer would then grade fabricated artifacts. `run_candidate.sh` now denylists
  the dangerous env keys (`PATH`, `PYTHONPATH`, `LD_*`, `SLURM_*`, output-dir,
  `DINOV3_PERRANK_DIAG≠1`) as defense-in-depth, but the real barrier is the driver
  **submitting an immutable, content-hashed actuator + training git revision** and
  recording the trusted command and sanitized environment itself, independent of
  candidate output. Do this in `loop-build` before the first unattended night.
- **Interpreter/deps outside the digest (finding 4).** `/usr/bin/env python3` and
  the dynamically imported `yaml` are not attested by the combined digest. `-I -B`
  neutralizes env-based import hijacking, but a fully hermetic deployment would pin
  and hash the interpreter and vendor the YAML parser. Deployment-scope.

**Smoothed-metric window bleed (finding 8):** `training_metrics.json` rows are
20-iter rolling averages, so the first point kept at `iter ≥ skip_iters` still
averages in ~20 pre-skip iters. With the default `skip_iters=200` the warmup/
compile stall (iters 0–~50) is long gone, so the practical bleed is negligible;
keep the skip margin ≥ the smoothing window. A future raw-counter metric would
remove the caveat entirely. The `stats_caveat` in the JSON already flags that the
points are autocorrelated rolling averages.

---

## 3. The verifier (`score.py`) in detail

Schema v3. Reads a run dir's raw `training_metrics.json` (never trusts the job's
self-report). Structure:

**Primary score:** `wall_imgs_mean = global_batch_size / mean(iter_time)` over
iters ≥ `--skip-iters` (200). Wall throughput, mean-based. MFU and step_time are
computed but **diagnostic only**.

**Gates split into MANDATORY and ADVISORY (schema v3, Codex GPT-5.6 review
finding 1).** A `not_evaluated` mandatory gate is **not** a pass — it means the
evidence is missing, and it downgrades a comparison verdict to `incomplete`
rather than letting a candidate through. This closes the fail-open hole where a
candidate could disable a gate's input (e.g. `DINOV3_PERRANK_DIAG=0` → straggler
gate `not_evaluated` → old code counted it as passing). Advisory gates are honest
infra stubs and stay non-blocking.

Mandatory gates (`MANDATORY_GATES` in `score_core.py`):

| Gate | What it checks | Default |
|------|----------------|---------|
| `completed` | `max_iter ≥ expect − 1` (exact, not −10) | `--expect-iters` (driver-owned) |
| `finite_loss` | no NaN/Inf in the loss column | — |
| `timing_coverage` | ≥90% of steady-state rows carry a finite `iter_time`, ≥`--min-ss-points`, constant `global_batch_size` (stops a score resting on one cherry row — finding 2) | `--min-timing-coverage 0.9`, `--min-ss-points 30` |
| `wall_tail` | p95/median iter_time ratio ≤ cap | `--tail-ratio-max 2.0` |
| `data_time_rank0` | rank-0 data_time **median** under cap | `--data-time-max-s 0.05` |
| `per_rank_straggler` | **worst-rank p95** data_time under cap, **and** every expected rank present with ≥20 points (raw p95, not rounded — findings 6, 9; needs `--slurm-log`) | `--straggler-max-s 0.1`, `--world-size`/adapter |
| `objective_integrity` | candidate `config.yaml` diff vs baseline within allowlist; **type-aware** so `True`≠`1` and missing≠null (finding 7) | menu keys |
| `loss_envelope` | candidate final loss not worse than baseline | `--loss-tol-pct 5.0` |

Advisory gates (`ADVISORY_GATES`): `peak_vram`, `host_rss_drift` — **not
implemented**, report `not_evaluated`, never block.

**`compare()`** — delta vs a 5% noise floor, computed from **unrounded** wall
img/s (finding 10). Verdicts: `candidate-improvement`, `candidate-regression`,
`no-evidence` (|delta| < floor), `incomplete` (a mandatory gate `not_evaluated`),
`invalid` (a mandatory gate `fail`ed). Includes the **§7.2 divergence detector**:
if the diagnostic (MFU) moved up while wall img/s regressed **past the floor**, it
fires a note — the exact MAX_CONN=8 trap. It surfaces baseline trust too: a
self-reported JSON baseline, or a baseline that itself failed a gate (finding 5).

### Two design decisions worth understanding

**(a) `objective_integrity` — the anti-gaming gate (Codex note 3).** `EXTRA_OPTS`
is appended last so it wins, which means a candidate could set
`ibot.loss_weight=0` or drop crops to get faster (meaningless) steps. The loss
envelope alone doesn't catch this — work-reduction can *lower* loss. So the gate
flattens both configs (`flatten()`) and fails any key that differs outside the
allowlist (the menu). Validated: 77671-vs-77672 (same recipe) passes;
67639-vs-77672 (different recipe) fails on `train.OFFICIAL_EPOCH_LENGTH`,
`checkpointing.period`, `evaluation.eval_period_iterations` → verdict `invalid`.

**(b) The straggler gate uses p95, not median — and this is the session's main
lesson.** First implementation gated on per-rank *median* data_time. Tested
against job 80711 (known rank-7 loader starvation) it **passed** — every rank's
median was <2ms because starvation is episodic (rank 6/7 p95 up to 0.44s, 15–35%
of points over 50ms). A central-tendency gate is blind to the exact pathology it
exists for. Switched to worst-rank p95; it now correctly fails 80711.
Generalizable rule, now in `learnings/profiling_workflow.md`: **gates for episodic
phenomena need tail statistics, and every gate must be validated against a run
where the pathology is known present. A gate that has never fired is untested.**

---

## 4. Validation (the trust basis)

Six ground-truth cases, all green (table in `README.md`):
- 77672 exact reproduction: 2115.6 img/s / iter_time 0.4488 / MFU 18.57 / 311.3ms.
- 77671-vs-77672: −7.67% wall, MFU +9.5%, verdict `candidate-regression`,
  divergence note fired.
- 67639: `completed` gate passes at 2000 iters.
- integrity pass (77671/77672) and fail (67639/77672) pairs.
- 80711 straggler fail (after the p95 fix).

Two catches during validation proved the "validate against ground truth" doctrine:
the `[RANKDATA]` regex needed the `data_time_max` field made optional (old log
format drift), and the median→p95 switch above.

---

## 5. What is dinov3-specific vs generalizable

Roughly **60% reusable, 40% adapter**. If you stand up a T2 loop for another
project (radio, molecules, VLM), keep the architecture and swap the adapters.

> **This split is now IMPLEMENTED (2026-07-03).** The verifier was refactored
> into a project-agnostic engine (`score_core.py`) + a dinov3 adapter
> (`adapters/dinov3.py`) + a thin CLI (`score.py`). The porting contract is in
> `ADAPTERS.md`; the refactor was validated as behavior-identical to the original
> single-file scorer across all six ground-truth cases (README validation table).
> The lists below describe what landed where.

### Generalizable (the method — lift these directly)
- The **three-role separation** and the open-proposals/closed-acceptance contract.
- The **verifier skeleton**: primary score + hard gates + diagnostics;
  `compare()` with a noise floor; `invalid` on any gate failure.
- **`objective_integrity`** via flattened-config diff against a baseline — any
  project with a serialized run config can use this unchanged in spirit.
- **Tail-statistics gating** for episodic failures; validate-against-known-pathology.
- The **reward-hacking doctrine**: primary metric = the real objective never a
  proxy; hermetic scorer outside the editable tree; single replicate never
  promotes; patch the verifier, not the policy.
- `install_verifier.sh` and the digest-in-state.md **trust chain** — generic.
- The loop workflow (submit→poll→score→decide→human checkpoint).

### dinov3-specific (adapters — rewrite per project)
- **`run_candidate.sh`**: the pinned recipe (DDP+cudagraphs, bs=128, torchrun
  `train.py`), the conda env path, Weka paths, Slurm partition/resources.
- **`candidates.example.yaml`**: the actual knobs (`num_workers`,
  `DINOV3_RANK_CPU_SLICE`, `CUDA_DEVICE_MAX_CONNECTIONS`), the operating point
  (bs=128), baseline job numbers/scores.
- **`score.py` parsers** (the seam between reusable and specific):
  - `load_rows()` / `col()` — assume dinov3's `training_metrics.json` JSONL schema
    (rank-0, smoothed 20-iter rolling averages — autocorrelated, not iid).
  - `RANKDATA_RE` — matches dinov3's `[RANKDATA]` stderr format from
    `dinov3/logging/helpers.py` (gated by `DINOV3_PERRANK_DIAG=1`).
  - `read_expected_iters()` — reads dinov3 config keys.
  The gate *logic* is generic; only these readers are project-bound. This is now
  exactly the seam: `score_core.py` (engine) + `adapters/dinov3.py` (these three
  readers plus the file names, column map, and integrity key-sets). A new project
  copies the engine and writes a sibling adapter — `ADAPTERS.md` is the contract.

---

## 6. Known gaps / how to extend

- **The loop driver itself is not built.** The submit→poll→score→decide policy is
  documented in `README.md` but left for `loop-build` at mission time (it needs a
  red-first gate and per-iteration checkpoints — build it when you run, not
  speculatively).
- **`peak_vram`** — parse `max mem` from the slurm log into the existing
  `not_evaluated` gate slot.
- **`host_rss_drift`** — needs node-level RSS sampling; no slurm-log source today.
- **Invisible to `objective_integrity`**: env-channel knobs (not in config.yaml —
  the driver must record them in the lineage row) and anything hardcoded in code
  (e.g. `pin_memory` at `dinov3/data/loaders.py:246`). Code changes between
  candidate and baseline are out of scope here and belong to `loop-build`'s
  git-diff review.
- **First mission (pre-scoped)**: worker-budget × affinity grid. `num_workers ∈
  {8,12,16,20}` is runnable today; `DINOV3_RANK_CPU_SLICE=1` is already
  implemented on this branch (`train.py`, gated) with §7.9 single-arm results — so
  the loop's job is the *joint* slice × worker grid, seeded by those.

When you find a new gaming/inflation pattern: add a gate to **`score_core.py`**
(the engine — it benefits every project; the thin `score.py` CLI holds no logic),
review the diff, re-run `install_verifier.sh`, update the digest in `state.md`.
The verifier is the one component where a bug corrupts every future verdict — treat
changes to it the way `docs/phase7_harness_audit.md` treats measurement code.
