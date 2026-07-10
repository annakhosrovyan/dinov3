# Porting the T2 verifier to a new project — the adapter contract

The verifier is split into a **reusable engine** (`score_core.py`) and a **tiny
project adapter** (`adapters/dinov3.py`). The engine holds every gate, the
verdict logic, the noise-floor comparison, and the §7.2 divergence detector — it
knows nothing about any project. To score a different PyTorch training run you
**copy `score_core.py` verbatim** and write one adapter. You never edit the
engine.

This is the "one base of autoresearch infra" idea: the base is `score_core.py` +
this contract; each project is ~40 lines.

## What you write for a new project

Three small things:

1. `adapters/<project>.py` — an `Adapter` instance (below).
2. A 4-line `score.py` that imports your adapter and calls `run_cli` (copy the
   dinov3 one, change the import).
3. `run_candidate.sh` + `candidates.example.yaml` — the shell recipe and knob
   menu (project-specific; see "The shell side" below).

Nothing else changes. The CLI, flags, gates, JSON schema, and trust chain are
inherited from the engine.

## The `Adapter` contract

```python
from score_core import Adapter, Columns
ADAPTER = Adapter(
    name="myproject",
    metrics_file="training_metrics.json",  # JSONL, one row per logged point (rank 0)
    config_file="config.yaml",             # serialized effective run config
    columns=Columns(...),                  # see below
    allowed_diff_keys={...},               # config keys a candidate MAY change (the menu)
    ignored_diff_keys={...},               # run-identity keys with no objective content
    read_expected_iters=<fn>,              # optional
    parse_per_rank=<fn>,                   # optional
    extra_notes=<fn>,                      # optional (project advisory notes)
    expected_world_size=8,                 # optional — rank count for the straggler coverage check
)
```

### `Columns` — map your metrics-file keys onto the roles the engine needs

The engine re-derives everything from the raw metrics JSONL. It only needs to
know which key plays which role:

| Field | Role | dinov3 value |
|-------|------|--------------|
| `iteration` | step counter (defines steady-state window + completion) | `iteration` |
| `iter_time` | wall seconds per step — **the primary score is `batch_size / mean(iter_time)`** | `iter_time` |
| `batch_size` | global batch size per step | `global_batch_size` |
| `loss` | scalar loss (finite-loss gate + loss-envelope gate) | `total_loss` |
| `data_time` | rank-0 dataloader wait per step (local starvation gate) | `data_time` |
| `diag_metric_src` | the "MFU-like" GPU-window metric — **diagnostic only**, feeds the §7.2 divergence detector | `mfu` |
| `diag_metric_out` | output key for it in the JSON | `mfu_pct` |
| `diag_metric_label` | human label in notes/summary | `MFU` |
| `extra_diag` | `[(src_key, out_key), ...]` surfaced verbatim in diagnostics | `[("step_time_ms","step_time_ms_cuda_event"), ...]` |

If your project has no GPU-window vanity metric, point `diag_metric_src` at any
throughput proxy you want cross-checked (or a key that doesn't exist — `dstat`
returns `None` and the divergence detector simply never fires).

**Direction assumption (Codex finding 10):** the divergence detector assumes the
diagnostic is *higher-is-better* (true for MFU / images-per-sec). If you point it
at a lower-is-better metric (e.g. GPU-kernel latency), the detector inverts — it
would flag the wrong direction. Use a higher-is-better proxy, or leave
`diag_metric_src` pointing at a missing key to disable the detector.

### `allowed_diff_keys` / `ignored_diff_keys` — the anti-gaming gate

`objective_integrity` flattens the candidate's `config_file` and diffs it against
the baseline's. Any key that differs and is **not** in `allowed_diff_keys` fails
the run as `invalid`. This is what stops a candidate from getting "faster" by
changing the objective (dropping a loss term, shrinking the schedule).

- `allowed_diff_keys` = exactly your searchable knobs (the menu). For dinov3:
  `train.num_workers`, `train.prefetch_factor`. Widen per-mission at the CLI with
  `--allow-diff-keys`.
- `ignored_diff_keys` = keys that always differ and carry no objective content
  (output dir, run name, wandb group). These are dropped from the diff entirely.

Keys are dotted paths into the flattened config (`train.num_workers`,
`optim.base_lr`, ...). Env-channel knobs that never appear in the config file
(env vars, launcher flags) are **invisible** to this gate — the driver must
record those in the lineage row.

### `read_expected_iters(run_dir) -> int | None` (optional)

Best-effort read of the intended run length from the run's own config, used only
as a fallback for the `completed` gate. **The driver should always pass
`--expect-iters`** — a value read from the candidate's own config is
candidate-controlled, and the engine flags it in a note. Return `None` if you
can't determine it; the gate becomes `not_evaluated`. dinov3 reads
`OFFICIAL_EPOCH_LENGTH * epochs` by regex (no yaml dependency).

### `parse_per_rank(slurm_log, skip_iters) -> (per_rank_avg, per_rank_max)` (optional)

The per-rank straggler gate is the session's main lesson: rank-0's `data_time`
in the metrics file **cannot see** another rank starving (job 80711: rank-7 at
0.37–0.42 s/iter, invisible to rank 0). Your training loop must emit per-rank
`data_time` to the job log; the adapter parses it.

Return two `dict[int] -> list[float]` (rank → per-step `data_time`, and rank →
per-step max, empty dict if you don't emit a max). The engine computes worst-rank
**p95** on the *raw* values (not median — starvation is episodic, a median gate
passed 80711; not rounded — a true 0.10004 must not round down under a 0.1 cap,
finding 9). If you don't emit per-rank data, omit this function; the gate reports
`not_evaluated` — which for a **mandatory** gate now downgrades a comparison
verdict to `incomplete` (a candidate cannot dodge the gate by dropping its
evidence — finding 1).

**Rank coverage (finding 6):** set `expected_world_size` so the gate FAILS when
the log is missing a rank (a candidate could otherwise hide a starving rank by
dropping its lines) or has too few steady-state points to estimate a p95. The
driver can override per-run with `--world-size`. `None` disables the check.

dinov3 emits `[RANKDATA] rank=N iter=M data_time=... data_time_max=...` to stderr
when `DINOV3_PERRANK_DIAG=1`; the adapter matches it with `RANKDATA_RE`.

## The shell side (also project-specific)

- **`run_candidate.sh`** — one sbatch job per candidate. Everything is pinned to a
  known baseline recipe; the candidate is only `EXTRA_OPTS` (config overrides) /
  `EXTRA_ENV` (env vars) / sbatch args. Project-specific bits (conda env path,
  torchrun command, dataset/output paths, Slurm resources) are inside a marked
  `# ===== PROJECT-SPECIFIC =====` block. Rewrite that block; keep the scaffolding.
- **`candidates.example.yaml`** — the menu of searchable knobs + one wildcard
  slot. Must match `allowed_diff_keys` (config knobs) plus the env-channel knobs
  the driver records. The operating point (batch size for dinov3) is **not** a
  searchable knob — it's the fixed condition.

## Validate before you trust it (non-negotiable)

The verifier is the one file where a bug corrupts every future verdict. Before
using a new adapter, score at least one run whose verdict you already know from
another source, and confirm the primary score and every gate status match. For
dinov3 the six ground-truth cases are in `README.md`; reproduce the analogous set
for your project (a known-good baseline, a known regression, a known gate
failure). A gate that has never fired is untested.

## Install & trust chain (unchanged)

`install_verifier.sh` copies `score_core.py`, `score.py`, and `adapters/*.py`
into `~/scripts/loop-verifier/` (read-only) and prints one **combined** sha256
over **all** installed files (not just `.py` — so a planted `__pycache__/*.pyc`
is caught; the thin CLI sets `sys.dont_write_bytecode` so a legit run never
creates one). Record that digest in the loop's `state.md`; the driver re-derives
it from the live files each iteration. See `DESIGN.md` §2 for why the verifier
lives outside the editable tree.

## What is NOT in the adapter

Do not copy gate logic, statistics, verdict rules, or the divergence detector
into an adapter — those live once, in the engine. If you find a new
gaming/inflation pattern, add a gate to `score_core.py` (it benefits every
project), review the diff, reinstall, and update the digest. The adapter is only
names, a column map, two parsers, and two key-sets.
