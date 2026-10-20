# Handoff: merge-to-mainline plan for MFU + DDP/cudagraphs

## Goal

Figure out the smallest sane PR to merge from `perf-ddp-fullgraph` into current mainline.

Practical constraint:
- In an ideal repo this would be 2 PRs:
  - MFU instrumentation
  - DDP + CUDA-graphs enablement
- For this research repo, one compact code-only PR from one branch is acceptable.

Important repo note:
- The actual base ref in this checkout is `origin/master`, not `main`.

## Current situation

- Anna is about to start full-dataset DINOv3 training.
- Her current sweep script is not aligned with the best validated path:
  - it does not force `train.distributed_strategy=ddp`
  - it does not include the code changes that made `train.cudagraphs=true` work
  - it uses `bs=128` on a codebase where that is not yet the known-good default on mainline
  - it sets `train.cache_dataset=false`, while most of the later high-throughput DDP work used dataset caching
- This branch is very broad versus `origin/master`: about 128 changed files, mostly docs and experiment scripts. It should not be merged wholesale.

## Proven results that matter

- Pre-CUDA-graphs DDP baseline:
  - job `48312`
  - config: `DDP`, `bs=96`, `compile=true`, `compile_mode=null`, `cudagraphs=false`
  - result: about `1387 img/s`, `7.94% MFU`
- CUDA-graphs screen champion:
  - job `53708`
  - config: `DDP`, `bs=128`, `compile=true`, `cudagraphs=true`, no activation checkpointing
  - result: about `2394 img/s`, `13.70% MFU`, `34.1 GB` peak VRAM
- Full sustainability soak:
  - job `60959`
  - completed `4000/4000`
  - no crash / OOM / NaN / worker-restart
  - honest sustained throughput was lower than the short screen:
    - about `1830 img/s` run average
    - about `2055 img/s` late-stage
  - VRAM stayed flat
- Important operational caveat from the soak:
  - with `cache_dataset=true`, host RAM drifted upward over time
  - do not set a tight Slurm `--mem` cap if reproducing the cached-data DDP path

## What mainline is missing

### MFU path

Minimal merge-worthy MFU pieces appear to be:
- `dinov3/utils/mfu.py`
- `dinov3/train/train.py`
- `tests/test_mfu.py`

What this adds:
- FLOP/MAC accounting helper for DINOv3 training
- per-step `step_time_ms`
- `images_per_sec`
- `mfu`
- WandB logging for those metrics

Potential cleanup before PR:
- The current `train.py` diff also contains profiling and memory-instrumentation additions. Those are useful, but they are not required for MFU. It may be worth trimming them out if the goal is a smaller PR.

### DDP + cudagraphs path

Minimal merge-worthy DDP/cudagraph files appear to be:
- `dinov3/configs/ssl_default_config.yaml`
- `dinov3/fsdp/ac_compile_parallelize.py`
- `dinov3/train/ssl_meta_arch.py`
- `dinov3/layers/block.py`

Possibly also include:
- `dinov3/data/datasets/satlas_datasets.py`

Why each matters:
- `ssl_default_config.yaml`
  - adds `train.compile_mode`
  - adds `train.distributed_strategy`
  - adds `train.fsdp_reshard_after_forward`
- `ac_compile_parallelize.py`
  - adds real `ddp` wrapping path
  - threads compile mode through block compilation
  - keeps FSDP2 path configurable instead of hard-coded
- `ssl_meta_arch.py`
  - calls `torch.compiler.cudagraph_mark_step_begin()` at the top of `forward_backward()`
  - adds DDP-safe unwrapping / init handling
  - fixes EMA init/state-dict behavior when `student.*` modules are wrapped
- `block.py`
  - replaces advanced indexing with `torch.index_select(...)` in the drop-path subset logic
  - this was required to avoid the bad `index_put_(accumulate=True)` backward pattern that blocked CUDA-graph capture
- `satlas_datasets.py`
  - not part of CUDA graphs directly
  - but it fixed the deterministic rank-6 crash during long DDP soaks by making corrupt-tile logging best-effort instead of fatal for non-dataset-owners

## Strong candidate PR scope

If keeping this as one compact PR, the best scope is probably:

1. MFU instrumentation:
   - `dinov3/utils/mfu.py`
   - minimal `train.py` changes
   - `tests/test_mfu.py`
2. DDP/cudagraph enablement:
   - `ssl_default_config.yaml`
   - `ac_compile_parallelize.py`
   - `ssl_meta_arch.py`
   - `block.py`
3. Robustness fix that proved necessary in real DDP soaks:
   - `satlas_datasets.py`

Everything else on `perf-ddp-fullgraph` should be treated as supporting material, not merge scope, unless the stronger model finds a hidden dependency.

## Recommended merge framing

Suggested framing for the stronger-model discussion:

- Do not ask "how do we merge this branch".
- Ask "what is the smallest production-worthy PR slice from this branch that gets us:"
  - MFU metrics on mainline
  - working `train.distributed_strategy=ddp`
  - working `train.cudagraphs=true`
  - no regression to existing FSDP2 behavior
  - no dependence on the giant docs / scripts diff

## Interim advice for Anna before merge

If Anna must train on current mainline before these changes land, the safest recommendation from this chat was:

- If she wants the known pre-cudagraph DDP path:
  - `train.distributed_strategy=ddp`
  - `train.batch_size_per_gpu=96`
  - `train.compile=true`
  - `train.compile_mode=null`
  - `train.cudagraphs=false`
  - `train.cache_dataset=true` if node memory allows it
  - `train.num_workers=20`
  - `train.prefetch_factor=8`
  - `train.persistent_workers=true`
- Keep activation checkpointing off on the speed path:
  - `train.checkpointing=false`
  - `train.checkpointing_full=false`
- Keep checkpoint saving if she needs resume/recovery:
  - `checkpointing.period=...`
  - `checkpointing.max_to_keep=...`

Important clarification:
- `train.checkpointing` / `train.checkpointing_full` are activation-recompute knobs.
- `checkpointing.period` / `checkpointing.max_to_keep` are actual checkpoint-save behavior.

SBATCH guidance discussed in chat:
- `--nodes=1`
- `--ntasks-per-node=1`
- `--cpus-per-task=64`
- `--gres=gpu:h100:8` or equivalent `--gpus-per-node=8`
- avoid a tight `--mem` cap if using `cache_dataset=true`

## Questions for the stronger model

- Which hunks in `train.py` are truly required for MFU, and which should be dropped from the PR as profiling-only noise?
- Is `satlas_datasets.py` worth including in the same PR, or should it be a separate small robustness PR even in a research workflow?
- Are there any hidden dependencies between the DDP path in `ac_compile_parallelize.py` and the `ssl_meta_arch.py` DDP unwrap / EMA changes?
- Is there any reason to merge `compile_mode` plumbing now if mainline will still default to `compile_mode=null`?
- What is the cleanest validation matrix for the PR before merge?
  - current FSDP2 baseline still runs
  - DDP non-cudagraph path still runs
  - DDP cudagraph path runs on the branch code

## Short version

The branch is too large to merge directly. The real task is to carve out a small code-only PR containing:
- MFU instrumentation
- DDP path support
- the two CUDA-graphs-critical fixes:
  - `cudagraph_mark_step_begin()`
  - `index_select` in `block.py`
- optionally the proved-in-production `save_error_path` robustness fix

Meanwhile, if Anna stays on current mainline, treat `DDP + bs=96 + compile=true + cudagraphs=false` as the known safer DDP recommendation, and do not confuse activation checkpointing with checkpoint saving.
