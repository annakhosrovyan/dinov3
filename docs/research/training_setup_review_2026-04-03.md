# Training Setup Review

**Date:** 2026-04-03

## Purpose

This note captures the current training setup, what is and is not enabled today, and the
highest-signal next moves that look safe from an accuracy standpoint.

This document is meant to answer:

1. What is the repo doing today in the main launch scripts?
2. Which optimization ideas are already ruled out or deprioritized?
3. Which next steps are high-return and low-risk?

---

## Direct Answers

### Are we using sharded eval checkpoints right now?

No.

- The default config sets `train.sharded_eval_checkpoint: false`.
- `run.sh` does not override it.
- `latest_run.sh` does not override it.

Operational implication:

- The current eval path falls into the non-sharded branch in `do_test()`.
- That branch iterates EMA `DTensor`s and materializes them with `full_tensor()`.
- This is a credible source of phase-specific memory spikes during training.

### Are we doing async eval right now?

No.

Eval currently runs inline inside the training loop.

Operational implication:

- When `eval_period_iterations` fires, training calls `do_test(...)` directly.
- Eval therefore shares the same process lifetime and GPU memory context as training.
- This is simpler operationally, but it means eval-time memory spikes and synchronization
  happen inside the trainer.

### Are we screening compile modes without CUDA graphs right now?

No.

- Compile is enabled by default.
- The code currently calls plain `module.compile()` with no explicit mode.
- The repo has screened the separate `train.cudagraphs=true` path and found it incompatible
  with the current multi-crop architecture.

Operational implication:

- There is still an unexplored, accuracy-neutral compiler tuning space.
- The most relevant next screen is compile-mode selection without enabling CUDA graphs.

---

## Current Vs Recommended

| Area | Current | Recommended next | Why it matters | Accuracy risk | Expected return | Confidence |
|---|---|---|---|---|---|---|
| Distributed strategy | Main scripts inherit `train.distributed_strategy=fsdp2` from config | Treat `ddp` as the main throughput candidate for single-node ViT-B | ViT-B fits comfortably on 80 GB H100; DDP removes FSDP2 sharding overhead | None | High | High |
| Batch size | `run.sh` uses `bs=64`; `latest_run.sh` sweeps `64/96` | Use `DDP bs=128` as the conservative next operating point; keep `DDP bs=256 + expandable_segments` provisional until worst-case memory profiling lands | Batch scaling has produced the largest measured MFU/runtime gains so far | None | High | Medium-high |
| Allocator tuning | Main launch scripts do not set `PYTORCH_CUDA_ALLOC_CONF` | Use `expandable_segments:True` only in DDP runs that are being explicitly validated | Short screens suggest a real DDP `bs=256` win, but not yet a production guarantee | None | Medium-high | Medium |
| Eval checkpoint path | Default `train.sharded_eval_checkpoint=false` | Flip to `true` for serious runs, especially memory-sensitive ones | Avoids the `full_tensor()` materialization path during eval | None | High for stability / OOM avoidance | High |
| Eval execution model | Inline eval in the trainer | Consider async eval from saved checkpoints if acceptable operationally | Decouples eval memory spikes and eval dataloading from training | None | Medium for stability, low for MFU | High |
| `torch.compile` tuning | Compile is on, but no explicit mode screening | Add a compile-mode knob and benchmark `default` vs `max-autotune-no-cudagraphs` | Safer and cheaper than trying to resurrect CUDA graphs | None | Medium | Medium |
| CUDA graphs | `train.cudagraphs=false`; prior tests failed | Keep off for now | Current multi-crop path breaks CUDAGraph tree assumptions | None | Low near-term | High |
| Activation checkpointing | Off by default | Keep as a memory escape hatch, not a primary throughput plan | Local results say it saves memory but loses throughput at equal batch | None | Low near-term | Medium-high |
| Data pipeline | Already reasonably tuned in `run.sh` | Leave alone unless traces show GPU idle gaps | Current evidence does not make this the active bottleneck | None | Low | High |
| Garbage collection | Trainer already does `gc.disable()` and periodic `gc.collect()` | Keep as is unless long-run traces show a better interval | Already a safe anti-straggler mitigation | None | Low-medium | High |

---

## Current Script Reality

### `run.sh`

Current characteristics:

- Uses default `fsdp2`
- Uses `batch_size_per_gpu=64`
- Uses inline eval and non-sharded eval checkpoint behavior from config defaults
- Does not set allocator tuning
- Does not set compile mode

Interpretation:

- This is still a conservative baseline-style launch script, not the best measured
  throughput configuration from recent experiments.

### `latest_run.sh`

Current characteristics:

- Also inherits default `fsdp2`
- Sweeps worker/prefetch settings and a small batch-size range
- Does not override sharded eval checkpointing
- Does not use async eval
- Does not set allocator tuning
- Does not set compile mode

Interpretation:

- This script is mainly a data-pipeline sweep script, not a current best-known
  throughput script.

---

## High-Signal Safe Moves

These are the best next investments if the goal is runtime improvement without knowingly
changing training semantics or risking accuracy.

### 1. Validate DDP as the main training direction

Reason:

- This is the strongest throughput signal in the repo.
- It directly attacks a real bottleneck rather than a hypothetical one.
- The model is small enough that sharding is not obviously necessary on one node.

Important caveat:

- `DDP bs=256 + expandable_segments` is still not proven production-safe.
- The right status today is "best short-horizon measured candidate," not
  "final recommended default."

### 2. Enable sharded eval checkpointing in serious runs

Reason:

- This is a low-risk systems change.
- It targets a very plausible memory-spike path.
- It should not affect model quality because it changes checkpoint materialization,
  not training math.

This is one of the clearest safe changes available right now.

### 3. Consider async eval from saved checkpoints

Reason:

- This is also accuracy-neutral from the training dynamics perspective.
- It keeps eval-time memory behavior out of the main trainer process.
- It can simplify worst-case memory management for long runs.

This matters more for stability and operational cleanliness than for raw steady-state MFU.

### 4. Screen compile modes without CUDA graphs

Reason:

- The current code has compiler tuning headroom that has not really been exercised.
- This is much safer than trying to make CUDA graphs work on the existing multi-crop path.
- It is still a pure systems optimization, not a modeling change.

Best candidate to screen first:

- `max-autotune-no-cudagraphs`

### 5. Keep activation checkpointing as a fallback, not a mainline plan

Reason:

- Local evidence says it is useful for fitting larger batches.
- Local evidence does not say it is a net throughput win in the tested regime.

So the right framing is:

- use it if memory forces it
- do not lead with it as the main optimization path

---

## Things To Deprioritize

### CUDA graphs

Why not now:

- The repo already tested them and hit a structural failure.
- Fixing that is not a one-line config change.
- It likely requires changing how global and local crops move through the compiled blocks.

This is not low-hanging fruit.

### More data-loader churn without new evidence

Why not now:

- `run.sh` is already using the obvious worker/prefetch/persistence knobs.
- The recent high-signal gains came from distributed strategy and batch size, not from
  the input pipeline.

### Dynamic batch size over the lifetime of a run

Why not now:

- This is not a standard "free systems optimization."
- It would drag optimizer, LR scaling, convergence, and semantics questions into a phase
  that should stay systems-focused.

---

## Recommended Short-Term Order

1. Finish worst-case memory profiling for `DDP` and `FSDP2`.
2. Downgrade any docs that state `DDP bs=256 + expandable_segments` as a settled default.
3. Test `train.sharded_eval_checkpoint=true` on the serious candidate runs.
4. If operationally acceptable, move eval off the critical training process and make it async.
5. Add a compile-mode knob and screen `default` vs `max-autotune-no-cudagraphs`.
6. Only after that decide whether `DDP bs=256 + expandable_segments` is truly safe enough
   to replace the current baseline launch setup.

---

## Bottom Line

The best safe opportunities right now are not exotic kernel work.

They are:

- `DDP` instead of default `FSDP2` for single-node ViT-B
- sharded eval checkpointing
- possibly async eval
- compile-mode screening without CUDA graphs

The best short-run throughput candidate in the repo is still `DDP bs=256 + expandable_segments`,
but until the new memory-profiled runs finish, it should be treated as a candidate rather than a
settled production default.
