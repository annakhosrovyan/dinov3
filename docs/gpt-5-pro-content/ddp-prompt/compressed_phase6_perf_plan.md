# Compressed Phase 6 Context For DDP + CUDA Graph Ideation

This is the small Phase 6 context bundle to paste into GPT-5 Pro instead of the full
`docs/phase6_perf_plan.md`.

## Current Direction

Phase 6 pivoted away from "make FSDP2 bs=128 work" as the main performance path.

Current speed path:

- Single node, 8x H100.
- DDP, not FSDP2.
- `train.compile=true`.
- `train.cudagraphs=true`.
- `train.batch_size_per_gpu=128`.
- Activation checkpointing off.
- Loader: `num_workers=20`, `prefetch_factor=8`, `persistent_workers=true`.

FSDP2 remains the conservative long-training baseline in `run.sh`, but the performance work is
now centered on DDP + CUDA graphs.

## Phase 6.A Scoreboard

| Phase | Run | bs | Config | img/s | MFU | Peak VRAM |
|---|---:|---:|---|---:|---:|---:|
| 6.A.0 baseline | 48312 | 96 | DDP, compile=true, cudagraphs=false | 1,387 | 7.94% | 25.8 GB |
| 6.A.3 cudagraphs win | 53681 | 96 | + cudagraphs=true, after `block.py` `index_select` fix | 2,009 | 11.50% | ~25.8 GB |
| 6.A.4 screen champion | 53708 | 128 | + bs=128, no AC | 2,394 | 13.70% | 34.1 GB |
| 6.A.5.b | 53999 | 96 | + AC=full | 1,378 | 7.89% | 9.6 GB |
| 6.A.5.d | 54000 | 128 | + AC=full | 1,654 | 9.46% | 12.4 GB |
| 6.A.6 | 56131 | 192 | AC=full | OOM | - | host-RAM OOM |
| 6.A.4.d | 53739 | 192 | no AC | OOM | - | host-RAM OOM |

Observed facts:

- Backbone CUDA graphs were a large real win once the `index_put` / advanced-indexing issue was
  fixed by replacing `x[indices]` with `torch.index_select(...)` in the stochastic-depth block path.
- The 53708 `bs=128` screening run was the best short-run throughput result.
- AC=full cuts VRAM dramatically but costs about 31% throughput at bs=96/128. It is a VRAM-budget
  mode, not the current speed mode.
- bs=192 failed from host-RAM / Slurm cgroup behavior, not CUDA VRAM. AC did not help because the
  binding resource was host memory, not activations.

Inference:

- HYPOTHESIS: the next speed lever is not activation strategy. It is the still-dynamic,
  launch-heavy, or host-visible part of the DINO/iBOT head/loss/training-step path.

## Phase 6.B Sustainability Soaks

Why Phase 6.B existed:

- Phase 6.A runs were throughput screens, mostly 1000 iterations.
- Prior FSDP2 bs=128 experience showed that short screens can miss late allocator behavior,
  checkpoint/eval spikes, host-RAM growth, and dataset failures.
- Phase 6.B asked whether DDP + cudagraphs + bs=128 can survive a production-shaped 4000-iter
  run with 8 checkpoint saves and 4 eval runs.

The soak instrumentation:

- `DINOV3_MEMORY_PROFILE=1` logs `[MEMPROFILE]` at checkpoint/eval phase boundaries.
- `[MEMFRAG]` logs fragmentation every 50 iterations.
- A shell sidecar samples host RAM and GPU memory every 10 seconds.

Key result table:

| Exp | Job | Config | Wall/GPU throughput | Peak VRAM | Host/cgroup signal | Verdict |
|---|---:|---|---|---|---|---|
| 6.B.1 | 57299 | bs=128, no-AC, 20w/8pf | partial | 36.3 GB stable | cgroup not measured correctly | crashed at rank 6, iter ~1310 |
| 6.B.2 attempts | 58188 / 59273 / 59274 | bs=128, AC=full, varied loader | loader/AC-limited | ~12-15 GB | cgroup pegged at limits, page-cache contaminated | rank-6 crash repeated |
| 6.B.5 interim | 60590 | bs=128, no-AC, 20w/8pf, fix applied | ~1,940 avg | 37.9 GB reserved, flat | ~570 GB host plateau | passed old crash point, killed by time limit at iter 3770 |
| 6.B.5 full | 60959 | bs=128, no-AC, 20w/8pf, fix applied, 4h walltime | ~1,830 run avg / ~2,055 late median | 36.3 GB reserved, flat | 613 GB peak, +15 GB/hr drift | completed 4000/4000, 6.B closed |

## Rank-6 Crash Root Cause And Fix

Observed failure before the fix:

- Four bs=128 soaks died at rank 6, iter ~1310, across multiple nodes.
- The failure initially looked suspicious because it was deterministic and occurred under the
  DDP+cudagraph soak, but it was not an iBOT or CUDA-graph issue.

Traceback captured in job 59847:

```text
libpng error: Read Error
PermissionError: Caught PermissionError in DataLoader worker process 11.
  mixed_satlas_dataset.py:169   self.datasets[i][base_idx]
  satlas_datasets.py:342        self.save_error_path(tci_path)
  satlas_datasets.py:58         with open(save_path, "a") as f:
PermissionError: [Errno 13] Permission denied:
  '/mnt/weka/akhosrovyan/re-id/pretraining/satlas-dataset-v1-naip-2020/naip_error_paths.txt'
```

Root-cause chain:

1. A corrupt/truncated NAIP PNG caused `cv2.imread` to return `None`.
2. The dataset recovery path tried to call `save_error_path(...)`.
3. `save_error_path(...)` attempted to append to an error log inside the dataset owner's
   read-only Weka directory.
4. `PermissionError` killed the DataLoader worker, which killed rank 6, which killed the job.

Fix:

- `dinov3/data/datasets/satlas_datasets.py`.
- Redirect bad-tile logs to writable `_ERROR_LOG_DIR`.
- Default: `/mnt/weka/adovlatyan/logs/dataset_errors`.
- Env override: `DINOV3_ERROR_LOG_DIR`.
- Wrap the write in `try/except OSError` so bad-tile logging can never be fatal.

Validation:

- `scripts/debug/test_save_error_path_fix.py` passed 3/3:
  - reproduces the original raw-write crash;
  - proves redirect writes to the writable dir;
  - proves an unwritable redirect target only warns and continues.
- Job 60959 saw one `libpng error: Read Error`; training self-healed and completed 4000/4000.

Follow-up not yet implemented:

- The `while True` recovery loops in `satlas_datasets.py` can theoretically spin forever if an
  entire shard/source is unreadable. That is a latent DDP deadlock risk and should be bounded
  separately, but it is not the bottleneck for the current performance ideation.

## Full Soak Validation: Job 60959

Config:

- Same recipe as job 53708 / 60590.
- DDP, bs=128, cudagraphs=true, no AC.
- 20 DataLoader workers, prefetch factor 8.
- 4000 iterations.
- 8 checkpoint saves.
- 4 eval runs.
- 4h Slurm walltime.

Outcome:

- Completed the full 4000/4000 iterations.
- No crash, OOM, NaN, worker restart, or traceback.
- Exactly one `libpng error: Read Error`; it self-healed and did not kill training.
- VRAM was dead-flat:
  - `[MEMPROFILE] max_reserved_mb=36266` across all 8 checkpoints and 4 evals.
  - one-time warmup spike to 36454 MB at iter 0, then 36266 MB for the rest.
  - `[MEMFRAG] fragmentation_ratio=0.040`, `alloc_retries=0`, `num_ooms=0`.
- Host RAM:
  - peak ~612.8 GB on a ~2 TB node;
  - slow +15 GB/hr drift, likely `train.cache_dataset=true` filling cache;
  - safe for the 4000-iter soak, worth monitoring on multi-hour production runs.

Throughput correction:

- Honest full-soak throughput is below the optimistic short-screen champion.
- 53708 1000-iter screen: 2,394 img/s, 13.70% MFU.
- 60959 full soak:
  - ~1,830 img/s run average, ~10.5% MFU;
  - ~2,055 img/s late median, ~11.8% MFU;
  - max observed late spikes around 2,658 img/s.
- Throughput improved over the run instead of thermally throttling:

| Window | Median img/s | Median MFU | vs 2,394 screen |
|---|---:|---:|---:|
| early, iter 200-1000 | 1,636 | 9.4% | -31.7% |
| mid, iter 1500-2500 | 1,857 | 10.6% | -22.4% |
| late, iter 3000-3999 | 2,055 | 11.8% | -14.2% |
| all steady, iter >=100 | 1,830 | 10.5% | -23.5% |

Interpretation:

- Observed: DDP + cudagraphs + bs=128 no-AC is production-sustainable on one 8x H100 node.
- Observed: the honest sustained rate is lower than the favorable 53708 short-screen number.
- HYPOTHESIS: early slow iterations are dominated by CUDA graph / cache / ramp effects rather than
  iBOT mask-count variance, because slow compute stalls concentrated before iter 1000 and nearly
  disappeared after iter 2000.
- HYPOTHESIS: the next performance work should target static-shape heads/iBOT, host-visible
  wall-clock gaps, and profiling of uncaptured step components rather than AC or larger batch.

## Current Next Questions For Ideation

Ask for ideas around these areas:

1. Static-shape iBOT masked-token handling:
   - pad to fixed upper bound;
   - fixed-K masking;
   - bucketed masking only if justified.
2. Extending fullgraph / cudagraph behavior beyond backbone blocks into DINO/iBOT heads and losses.
3. Reducing iBOT masked-token gather/head/loss HBM traffic and materialization.
4. Explaining and reducing GPU-event vs wall-clock gaps in the full soak.
5. Measuring head/loss/optimizer/EMA/collective overhead outside the captured backbone graph.
6. DDP all-reduce and loss collective overlap, especially if moving beyond one node.

Avoid for now:

- Treating AC as the speed path. It is a memory path here.
- Treating bs=192 as the obvious next step; prior failures were host-memory bound.
- Inferring precise loader memory requirements from cgroup `memory.current` plateaus.
- Optimizing only short-screen GPU-event throughput without checking delivered wall-clock throughput.
- Changing the SSL objective or mask distribution without labeling the convergence risk.
