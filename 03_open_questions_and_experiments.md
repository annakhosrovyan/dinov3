# DINOv3 Satellite Fork: Open Questions And Experiments

## Highest-leverage open questions

Code fact: The codebase contains both FSDP2 and DDP training paths, and the repo includes short-run screening scripts for both (`dinov3/fsdp/ac_compile_parallelize.py:187-302`, `scripts/screening_run.sh:52-80`, `scripts/screening_ddp.sh:44-70`).

Inference: The first systems question is still whether FSDP2 is the right steady-state choice for single-node ViT-B. The code is already structured to answer that experimentally, which usually means the answer is not settled.

Code fact: `train.py` only asserts `torch.__version__ >= (2, 1)` (`dinov3/train/train.py:52`), but `ac_compile_parallelize.py` imports `register_fsdp_forward_method` from `torch.distributed.fsdp` at module import time (`dinov3/fsdp/ac_compile_parallelize.py:13-16`).

Inference: The effective minimum PyTorch version is probably higher than the asserted one. That matters for portability, compile behavior, and whether benchmark results are comparable across environments.

Code fact: The multi-crop ViT path tries to share some work across resolutions by concatenating tensors in `SelfAttentionBlock._forward_list()`, but attention itself still runs per shape in `SelfAttention.forward_list()` (`dinov3/layers/block.py:126-198`, `dinov3/layers/attention.py:94-118`).

Inference: An unresolved question is whether the current list-based fusion is already close to optimal for mixed resolutions, or whether it leaves significant kernel-launch and memory-planning overhead on the table.

Code fact: The repo has compile toggles, activation-checkpoint toggles, and optional cudagraphs, but the default config still ships with `checkpointing=false` and `cudagraphs=false` (`dinov3/configs/ssl_default_config.yaml:77-86`).

Inference: Those knobs exist because their tradeoffs are unresolved for the target workload, not because they are universally beneficial.

Code fact: The data path mixes multiple storage/layout regimes: PNG-tree Satlas datasets, recursive MAID image walks, and HDF5-backed datasets (`dinov3/data/datasets/satlas_datasets.py`, `dinov3/data/datasets/maid_dataset.py`, `dinov3/data/datasets/hdf5_dataset.py`).

Inference: The repo has no code-level proof that one source type is dominating loader stalls. The mixed input regime makes “data pipeline” a composite bottleneck until measured per source.

## Experiments implied by the code

Code fact: The repo already exposes the experiment knobs needed for a compact matrix:

- Code fact: `train.distributed_strategy` toggles FSDP2 versus DDP (`dinov3/configs/ssl_default_config.yaml:82`, `dinov3/fsdp/ac_compile_parallelize.py:143-145`).
- Code fact: `train.checkpointing` and `train.checkpointing_full` control selective versus aggressive recompute (`dinov3/configs/ssl_default_config.yaml:78-79`, `dinov3/fsdp/ac_compile_parallelize.py:25-47`).
- Code fact: `train.cudagraphs` changes compile options for backbone blocks (`dinov3/configs/ssl_default_config.yaml:81`, `dinov3/fsdp/ac_compile_parallelize.py:65-70`).
- Code fact: `student.fp8_enabled` activates the optional FP8 linear replacement path (`dinov3/configs/ssl_default_config.yaml:114-115`, `dinov3/models/__init__.py:22-32`).

Inference: The smallest high-value experiment grid is:

1. Inference: DDP vs FSDP2 at the same batch size and compile settings.
2. Inference: FSDP2 with selective activation checkpointing on versus off.
3. Inference: FSDP2 with `train.cudagraphs=true` versus `false`.
4. Inference: ViT-B baseline versus a larger model where FP8 becomes relevant enough to matter.

Code fact: The scripts already encode short steady-state screening runs and a profiling run that skips early compile warmup in the profiler window (`scripts/screening_run.sh:52-80`, `scripts/profiling_run.sh:35-67`).

Inference: Those scripts are the right starting point for controlled MFU experiments because they minimize confounding from long training schedules and checkpoint overhead.

## Good places to instrument next

Code fact: The current training loop has GPU step timing and host `data_time`, but it does not split the loader path into file-read, augmentation, collate, and H2D subcomponents (`dinov3/train/train.py:656-660`, `dinov3/logging/helpers.py:65-133`).

Inference: Add timing inside or around these points first:

1. Inference: `make_data_loader()` worker output latency and queue depth in `dinov3/data/loaders.py:196-260`.
2. Inference: CPU crop generation inside `MAIDAugmentation.__call__` and `HDF5Augmentation.__call__` in `dinov3/data/datasets/augmentation.py:59-81` and `dinov3/data/datasets/hdf5_augmentation.py:55-77`.
3. Inference: Batch mask generation inside `collate_data_and_cast()` in `dinov3/data/collate.py:11-78`.
4. Inference: H2D transfer plus first-stall timing around `SSLMetaArch.forward_backward()`’s `non_blocking=True` copies in `dinov3/train/ssl_meta_arch.py:395-402`.

Code fact: The model-side NVTX hooks are already present in the training loop and meta-architecture (`dinov3/train/train.py:523-542`, `dinov3/train/ssl_meta_arch.py:375-377`).

Inference: The next useful extension is finer NVTX around `get_student_output()` internals, especially the boundary between backbone forward, iBOT masked-patch head work, and DINO head work.

Code fact: There is no explicit instrumentation around FSDP collectives, only indirect observation through step time and profiler traces (`dinov3/fsdp/ac_compile_parallelize.py`, `dinov3/utils/profiling.py`).

Inference: Add rank-local markers before and after forward-prefetch-heavy regions, or analyze profiler traces for block-level all-gather / reduce-scatter overlap, before making architectural changes.

## Open questions from the codebase

Code fact: `compute_dino_flops_per_image()` treats head cost as a flat `head_overhead_pct=0.05` estimate rather than deriving it from model structure (`dinov3/utils/mfu.py:52-105`).

Inference: MFU comparisons are directionally useful, but the absolute denominator still deserves validation against a traced or counted FLOP reference for the exact config under study.

Code fact: The default profiling and MFU timing window starts after the batch is already yielded by the iterable (`dinov3/train/train.py:563-660`, `dinov3/logging/helpers.py:93-129`).

Inference: If the real problem is CPU starvation, `step_time_ms` can look healthy while throughput remains poor. That is why loader-side instrumentation is the next gap to close.

Code fact: The training code treats Gram teacher, distillation teacher, and multidistillation as optional branches, but the default path still carries those abstractions in the hot codebase (`dinov3/train/ssl_meta_arch.py`, `dinov3/train/multidist_meta_arch.py`).

Inference: There may be simplification opportunities for the baseline satellite path, but that should only be pursued after profiling proves control-flow complexity is causing graph breaks or compile fragmentation.

Code fact: Satlas datasets can build manifest caches on first access by scanning directory trees and writing `.npz` manifests into dataset, home-cache, or tmp-cache locations (`dinov3/data/datasets/satlas_datasets.py:61-103`).

Inference: Startup latency and first-epoch instability may partly be a dataset-manifest problem rather than a model-performance problem, especially on fresh nodes or ephemeral storage.

Code fact: The repo’s benchmark scripts explicitly use short runs and often clear pretrained weights (`scripts/mfu_validation_run.sh:30-47`, `scripts/screening_run.sh:54-80`).

Inference: The intended engineering workflow here is “measure the systems shape first, then train for quality.” Any further optimization work should preserve that discipline rather than jumping straight to long end-to-end runs.
