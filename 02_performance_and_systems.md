# DINOv3 Satellite Fork: Performance And Systems

## Distributed training stack

Code fact: Distributed initialization is centralized in `dinov3/distributed/torch_distributed_wrapper.py`. The wrapper supports TorchElastic, Slurm, and single-node manual launch, sets CUDA device from `LOCAL_RANK`, initializes NCCL, and can optionally replace `print` on non-main ranks (`dinov3/distributed/torch_distributed_wrapper.py:114-272`).

Code fact: The default training strategy is FSDP2, selected by `train.distributed_strategy: fsdp2` in `dinov3/configs/ssl_default_config.yaml:78-86` and dispatched in `dinov3/fsdp/ac_compile_parallelize.py:126-203`.

Code fact: `SSLMetaArch.prepare_for_distributed_training()` shards the student over the process subgroup, while EMA teacher, optional Gram teacher, and optional distillation teacher are passed as inference-only models, potentially on different process groups (`dinov3/train/ssl_meta_arch.py:842-859`).

Code fact: FSDP2 wrapping is block-level for ViTs. Each transformer block is `fully_shard(...)`’d, forward/backward prefetch relations are registered between neighboring blocks, and the enclosing backbone is then sharded as a parent module (`dinov3/fsdp/ac_compile_parallelize.py:110-124`).

Code fact: Inference-only FSDP models are forced to reshard immediately after forward by mutating their FSDP state after wrapping (`dinov3/fsdp/ac_compile_parallelize.py:250-259`).

Code fact: The repo also has a DDP alternative. `_ac_compile_parallelize_ddp()` wraps only the student submodules in `DistributedDataParallel(static_graph=True, gradient_as_bucket_view=True)` and leaves teacher-like models as plain CUDA modules (`dinov3/fsdp/ac_compile_parallelize.py:261-302`).

Inference: For ViT-B, the DDP path is not a fallback afterthought. The code comments and dedicated scripts indicate it is a first-class experiment for reducing sharding overhead on single-node H100 runs (`dinov3/fsdp/ac_compile_parallelize.py:268-273`, `scripts/screening_ddp.sh:44-70`).

## Precision and numerics

Code fact: The default precision policy is BF16 parameters with FP32 gradient reduction (`dinov3/configs/ssl_default_config.yaml:6-9`). FSDP2 converts that into `MixedPrecisionPolicy(param_dtype=..., reduce_dtype=...)` in `dinov3/fsdp/ac_compile_parallelize.py:221-245`.

Code fact: Global numerics knobs are enabled early in `train.py`: `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.benchmark = True` (`dinov3/train/train.py:52-54`).

Code fact: The data collate path casts crop tensors to the target parameter dtype before the model sees them (`dinov3/data/collate.py:67-78`), and `forward_backward()` then performs non-blocking H2D copies (`dinov3/train/ssl_meta_arch.py:395-402`).

Code fact: KoLeo loss explicitly disables autocast and normalizes features in FP32 before distance computation (`dinov3/loss/koleo_loss.py:38-43`, `dinov3/loss/koleo_loss.py:72-112`).

Code fact: Teacher centering for DINO and iBOT uses cross-rank collectives. DINO’s center update uses an async all-reduce handle (`dinov3/loss/dino_clstoken_loss.py:101-124`), while iBOT’s Sinkhorn path does distributed normalization and is compiled as its own module (`dinov3/loss/ibot_patch_loss.py:20-58`, `dinov3/loss/ibot_patch_loss.py:71-73`).

## Compiler stack and kernel choices

Code fact: Compilation is applied per module, not around the whole training step. `ac_compile_parallelize()` first optionally wraps blocks for activation checkpointing, then calls `.compile()` on backbone blocks and heads, then applies FSDP2 or DDP (`dinov3/fsdp/ac_compile_parallelize.py:133-203`).

Code fact: When `cfg.train.cudagraphs` is enabled, only backbone blocks are compiled with `fullgraph=True`, `dynamic=False`, and `options={"triton.cudagraphs": True}`. Other modules call `.compile()` with defaults (`dinov3/fsdp/ac_compile_parallelize.py:65-70`).

Inference: This is not full-step CUDA graph capture. The code is targeting block-local compiled graphs inside Inductor rather than wrapping the optimizer step, data movement, or full iteration in a single graph.

Code fact: `SelfAttention` uses `scaled_dot_product_attention`, so training attention performance depends on PyTorch’s backend kernel choice for the current shape and dtype (`dinov3/layers/attention.py:106-118`).

Code fact: The selective activation checkpoint wrapper preserves a short allowlist of expensive ops, including `aten.mm`, `aten._scaled_mm`, flash/efficient SDPA ops, and `reduce_scatter_tensor` (`dinov3/fsdp/ac_compile_parallelize.py:25-47`).

Inference: The selective-checkpoint policy is trying to recompute cheap glue code while retaining high-cost matmuls, SDPA, and communication-heavy ops. That should reduce memory with less recompute penalty than full checkpointing, but the actual tradeoff is architecture- and backend-dependent.

Code fact: The model code disables Dynamo automatic dynamic shapes and raises the accumulated cache limit in `dinov3/layers/block.py:17-18`.

Code fact: The optional FP8 path replaces matching linear layers with a custom autograd function built on `torch._scaled_mm`, enables `torch._inductor.config.triton.multi_kernel = 1`, and resets Dynamo/Inductor caches after conversion (`dinov3/models/__init__.py:22-32`, `dinov3/layers/fp8_linear.py:24-140`).

Inference: FP8 is a serious throughput path for large-model experiments in this repo, but it is not part of the default ViT-B satellite baseline because `student.fp8_enabled` is `False` in `ssl_default_config.yaml` (`dinov3/configs/ssl_default_config.yaml:113-115`).

## Data/input pipeline costs

Code fact: `make_data_loader()` always enables `pin_memory=True`; `prefetch_factor` is only passed when `num_workers > 0`; and persistent workers are configurable (`dinov3/data/loaders.py:196-260`).

Code fact: `build_data_loader_from_cfg()` wires the loader with `num_workers`, `persistent_workers`, `prefetch_factor`, a `MaskingGenerator`, and dataset-specific augmentations derived from the model (`dinov3/train/train.py:298-372`).

Code fact: `collate_data_and_cast()` is CPU-side Python that stacks all crops, creates per-sample masks, flattens mask indices, and emits `masks_weight` for the masked-patch loss (`dinov3/data/collate.py:11-78`).

Code fact: The Satlas-style datasets lazily build manifest caches by scanning directory trees when cached manifests are missing (`dinov3/data/datasets/satlas_datasets.py:61-103`, `dinov3/data/datasets/satlas_datasets.py:146-168`, `dinov3/data/datasets/satlas_datasets.py:232-249`).

Inference: First-run startup cost can include substantial filesystem enumeration before steady-state training even begins, especially for Satlas-based sources without manifest cache files.

## Profiling and benchmarking hooks

Code fact: MFU is computed in the training loop from a precomputed MACs-per-image constant and CUDA-event step timing, not from PyTorch profiler FLOP attribution (`dinov3/train/train.py:464-483`, `dinov3/train/train.py:656-660`, `dinov3/utils/mfu.py:52-134`).

Code fact: The CUDA timing window starts immediately before `optimizer.zero_grad(set_to_none=True)` and ends after `model.update_ema(mom)` (`dinov3/train/train.py:594-654`). That means the reported step time includes forward, backward, optimizer step, and EMA update, but excludes data loading outside the loop body.

Code fact: `MetricLogger` still records host wall-clock `iter_time` and `data_time` with `time.time()` in parallel with the CUDA-event timing (`dinov3/logging/helpers.py:65-133`).

Inference: `step_time_ms` and `iter_time` answer different questions. Divergence between them is a direct clue for host-side stalls, dataloader starvation, or synchronization outside the CUDA-event window.

Code fact: Opt-in profiling mode enables PyTorch profiler traces, NVTX ranges, graph-break logging, and per-iteration memory metrics (`dinov3/train/train.py:521-542`, `dinov3/train/train.py:706-713`, `dinov3/utils/profiling.py:16-150`).

Code fact: The repo also includes dedicated launch scripts for profiling and DDP/FSDP2 screening on short runs (`scripts/profiling_run.sh:35-67`, `scripts/screening_run.sh:52-80`, `scripts/screening_ddp.sh:44-70`).

## Likely MFU bottlenecks

### Model compute

Code fact: Student compute is dominated by the joint global+local backbone path in `SSLMetaArch.get_student_output()` and the per-block list-based transformer execution in `DinoVisionTransformer.forward_features_list()` (`dinov3/train/ssl_meta_arch.py:561-613`, `dinov3/models/vision_transformer.py:222-261`).

Inference: The main math bottleneck is still transformer block throughput on two sequence lengths, not the DINO/iBOT heads.

### Memory bandwidth pressure

Code fact: FSDP2 with `reshard_after_forward=True` is used across every block in the default path (`dinov3/fsdp/ac_compile_parallelize.py:110-124`, `dinov3/fsdp/ac_compile_parallelize.py:239-245`).

Inference: For ViT-B on a single node, parameter all-gather / reshard traffic can compete with pure compute efficiency and lower MFU relative to DDP.

### Communication overhead

Code fact: The training loop does an all-gather of per-rank total loss and an all-reduce of stacked metrics every iteration (`dinov3/train/train.py:613-630`). DINO/iBOT centering and Sinkhorn also use subgroup collectives (`dinov3/loss/dino_clstoken_loss.py:43-70`, `dinov3/loss/ibot_patch_loss.py:29-58`).

Inference: The dominant collectives are probably FSDP/gradient traffic first, then the smaller per-step metric and centering collectives. The latter still matter because they are serialized every iteration.

### Data/input pipeline overhead

Code fact: CPU work per batch includes file decode or HDF5 reads, Albumentations multi-crop augmentation, Python mask generation, crop stacking, and dtype conversion before H2D (`dinov3/data/datasets/*.py`, `dinov3/data/collate.py:11-78`, `dinov3/train/ssl_meta_arch.py:395-402`).

Inference: If GPUs show intermittent starvation rather than a flat low-compute plateau, the highest-probability host-side causes are augmentation plus mask generation rather than raw `DataLoader` plumbing alone.

### Host-side orchestration overhead

Code fact: Automatic Python GC is disabled and manual `gc.collect()` is forced every 150 iterations (`dinov3/train/train.py:517-520`, `dinov3/train/train.py:575-579`).

Inference: The code already treats Python GC as a performance hazard. If periodic MFU dips remain, this area is still suspect, but less so than it would be in a default Python training loop.

Code fact: Eval and checkpoint paths call `torch.cuda.synchronize()` before their work (`dinov3/train/train.py:716-737`).

Inference: These syncs are not steady-state bottlenecks, but they will create visible periodic step spikes in end-to-end traces and aggregate throughput logs.
