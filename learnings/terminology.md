# Terminology and Concepts — DINOv3

Ongoing glossary for recurring terms in this repo. Keep entries short, code-grounded, and biased toward the meanings used here rather than generic ML prose.

---

## Backbone

The main feature extractor before task-specific heads.

In this repo:
- `SSLMetaArch` builds `student["backbone"]` and `teacher["backbone"]` in `dinov3/train/ssl_meta_arch.py:47-56`.
- For the current recipe, the backbone is a `DinoVisionTransformer` built from `student.arch=vit_base` in `dinov3/models/__init__.py:33-61`.

Example:
- The student forward path calls `self.student.backbone(...)` in `dinov3/train/ssl_meta_arch.py:543-547`.

Why this matters:
- Performance work often targets the backbone first because it dominates repeated compute.
- In this codebase, CUDA-graphs mode is applied to backbone blocks, not to the whole training step.

---

## Block

One repeated layer unit inside a backbone.

In this repo:
- ViT backbones are made of repeated `SelfAttentionBlock`s in `dinov3/models/vision_transformer.py:131-148`.
- Those blocks are stored in `self.blocks` in `dinov3/models/vision_transformer.py:151`.

Example:
- `dinov3/fsdp/ac_compile_parallelize.py:84-86` loops over `model.blocks` and compiles each block individually.

Why this matters:
- When someone says "compile per block" or "FSDP per block", they mean these repeated transformer units, not the whole model at once.

---

## Head

A task- or loss-specific projection stack that sits on top of the backbone features.

In this repo:
- `SSLMetaArch` attaches `dino_head` and `ibot_head` in `dinov3/train/ssl_meta_arch.py:67-77` and `dinov3/train/ssl_meta_arch.py:113-123`.
- The head class is `DINOHead` from `dinov3/layers/dino_head.py`.

Example:
- Student CLS features are passed through `self.student.dino_head(...)` in `dinov3/train/ssl_meta_arch.py:570-571`.

Why this matters:
- Heads are not the same thing as the backbone.
- `head_hidden_dim=2048` refers to head MLP width, not ViT backbone width.

---

## Embed Dimension (`embed_dim`)

The backbone token width, sometimes also called model dimension or hidden width of the transformer trunk.

In this repo:
- `DinoVisionTransformer` takes `embed_dim` in `dinov3/models/vision_transformer.py:68`.
- `vit_base(...)` sets `embed_dim=768` in `dinov3/models/vision_transformer.py:344-349`.

Example:
- `build_model_from_cfg(...)` returns `embed_dim`, and `SSLMetaArch` stores it as `self.embed_dim` in `dinov3/train/ssl_meta_arch.py:47-61`.

Why this matters:
- This is the `768` used in MFU for `vit_base`.
- It is easy to confuse with `head_hidden_dim`, which is a different dimension.

---

## Hidden Dimension (`hidden_dim`)

A generic term whose meaning depends on context.

In this repo it appears in at least two different ways:
- ViT/MFU context: backbone width, e.g. `hidden_dim=768` in `dinov3/train/train.py:457-468` and `dinov3/utils/mfu.py:58-60`
- Head MLP context: projection-head width, e.g. `head_hidden_dim: 2048` in `dinov3/configs/ssl_default_config.yaml:17` and `:44`

Example:
- `DINOHead(..., hidden_dim=cfg.dino.head_hidden_dim, ...)` in `dinov3/train/ssl_meta_arch.py:71-76`

Why this matters:
- Always ask "hidden dimension of what?" in this codebase.

---

## Depth

The number of repeated blocks in the backbone.

In this repo:
- `DinoVisionTransformer(..., depth=12, ...)` is defined in `dinov3/models/vision_transformer.py:69`.
- `vit_base(...)` sets `depth=12` in `dinov3/models/vision_transformer.py:344-349`.

Example:
- The model stores this as `self.n_blocks = depth` in `dinov3/models/vision_transformer.py:94`.

Why this matters:
- This is the `num_layers=12` used in the current MFU code for `vit_base`.

---

## Student / Teacher

The two main networks used in DINO-style self-supervised training.

In this repo:
- The student is trained directly by gradient descent.
- The teacher is an EMA copy updated from the student.

Examples:
- Student and teacher are built in `dinov3/train/ssl_meta_arch.py:47-56`.
- EMA update happens in `dinov3/train/train.py:606-607` and the parameter update logic lives in `dinov3/train/ssl_meta_arch.py:720-733`.

Why this matters:
- Teacher forward cost exists in MFU accounting, but the teacher does not take gradients.

---

## Global Crop / Local Crop

The multi-crop views used for DINO/iBOT training.

In this repo:
- Global crops are larger views, default size `224`.
- Local crops are smaller views, default size `96`.

Examples:
- Defaults live in `dinov3/configs/ssl_default_config.yaml:141-147`.
- The training step sets `n_global_crops = 2` and `n_local_crops = self.n_local_crops` in `dinov3/train/ssl_meta_arch.py:369-370`.
- Student forward jointly processes `[global_crops, local_crops]` in `dinov3/train/ssl_meta_arch.py:543-547`.

Why this matters:
- Global and local crops have different sequence lengths, which affects both performance and MFU.

---

## Patch Size

The spatial stride used to turn an image into patch tokens.

In this repo:
- `patch_size` is part of the backbone architecture in `dinov3/models/vision_transformer.py:57`.
- The current recipe uses `patch_size: 16` in `dinov3/configs/ssl_default_config.yaml:88`.

Example:
- Sequence length for MFU depends on `crop_size // patch_size`, implemented in `dinov3/utils/mfu.py:93-96`.

Why this matters:
- Patch size directly changes token count, which strongly changes attention and FFN cost.

---

## Compile

Using `torch.compile` to generate optimized code for repeated modules.

In this repo:
- `train.compile: true` is the default in `dinov3/configs/ssl_default_config.yaml:80`.
- Compilation is applied during distributed model setup in `dinov3/fsdp/ac_compile_parallelize.py:170-181`.

Example:
- Backbone blocks go through `wrap_compile_block(...)` in `dinov3/fsdp/ac_compile_parallelize.py:65-69`.

Why this matters:
- `compile=true` is already part of the baseline. CUDA graphs are a stricter layer on top of that, not a replacement for it.

---

## CUDA Graphs

A stricter execution mode aimed at reducing per-iteration launch overhead when the repeated workload is shape-stable.

In this repo:
- The relevant flag is `train.cudagraphs`, default `false`, in `dinov3/configs/ssl_default_config.yaml:81`.
- If enabled, backbone blocks are compiled with `fullgraph=True`, `dynamic=False`, and `triton.cudagraphs=True` in `dinov3/fsdp/ac_compile_parallelize.py:66-67`.

Example:
- The code applies the cudagraph path only to backbone blocks, not to heads or the entire training loop.

Why this matters:
- This is currently a viable optimization experiment for the fixed-shape ViT-B satellite recipe, but it should be benchmarked rather than assumed.

---

## DDP

Distributed Data Parallel: each rank keeps a full replica of the trainable model and synchronizes gradients with an all-reduce.

In this repo:
- The strategy switch is `train.distributed_strategy: ddp` in `dinov3/configs/ssl_default_config.yaml:82`.
- The DDP path wraps each trained sub-model with `DistributedDataParallel(..., gradient_as_bucket_view=True, static_graph=True)` in `dinov3/fsdp/ac_compile_parallelize.py:261-293`.

Example:
- `ac_compile_parallelize(...)` dispatches to `_ac_compile_parallelize_ddp(...)` when `distributed_strategy == "ddp"` in `dinov3/fsdp/ac_compile_parallelize.py:187-193`.

Why this matters:
- For single-node ViT-B, DDP can be the faster choice because the model already fits and DDP avoids FSDP2's repeated all-gather / reduce-scatter overhead.

---

## FSDP2

PyTorch's composable Fully Sharded Data Parallel path: parameters, gradients, and optimizer state are partitioned across ranks instead of fully replicated.

In this repo:
- The default strategy is `train.distributed_strategy: fsdp2` in `dinov3/configs/ssl_default_config.yaml:82`.
- The implementation uses `fully_shard`, `MixedPrecisionPolicy`, and a one-dimensional device mesh in `dinov3/fsdp/ac_compile_parallelize.py:11-14` and `dinov3/fsdp/ac_compile_parallelize.py:201-239`.

Example:
- Transformer backbones are sharded block-by-block in `fsdp_transformer(...)` in `dinov3/fsdp/ac_compile_parallelize.py:97-111`.

Why this matters:
- FSDP2 is the right tool when replication does not fit comfortably, but it is not automatically the fastest choice on one node.

---

## All-Reduce

A collective that combines values across all ranks and returns the full reduced result to every rank.

In this repo:
- The DDP helper describes the DDP path as "no sharding, just all-reduce gradients" in `dinov3/fsdp/ac_compile_parallelize.py:267`.

Example:
- In DDP, gradient synchronization is the main extra communication added to the backward pass.

Why this matters:
- If the model fits in memory, one gradient all-reduce per step is often cheaper than FSDP2's repeated parameter all-gathers.

---

## All-Gather / Reduce-Scatter

Two complementary collectives commonly used by sharded training.
- All-gather reconstructs a full tensor on each participating rank from sharded pieces.
- Reduce-scatter reduces values across ranks and leaves each rank with only its shard of the result.

In this repo:
- The DDP helper explicitly contrasts DDP with FSDP2's `all_gather / reduce_scatter` overhead in `dinov3/fsdp/ac_compile_parallelize.py:269-271`.
- Selective activation checkpointing explicitly lists `torch.ops._c10d_functional.reduce_scatter_tensor.default` in `dinov3/fsdp/ac_compile_parallelize.py:34-35`.

Example:
- FSDP2 uses these collectives so that each block can materialize parameters briefly, run compute, then return to a sharded state.

Why this matters:
- These collectives are the core reason FSDP2 saves memory and the core reason it can lose throughput when sharding is not needed.

---

## Replication / Sharding

Two opposite memory layouts for distributed training.
- Replication means every rank keeps a full copy of a tensor or module.
- Sharding means each rank keeps only one slice, and collectives rebuild or combine values as needed.

In this repo:
- `ddp` is the replicated path and `fsdp2` is the sharded path in `dinov3/fsdp/ac_compile_parallelize.py`.

Example:
- The DDP path keeps each student sub-model as a full `DistributedDataParallel` replica, while the FSDP2 path applies `fully_shard(...)` to blocks and heads.

Why this matters:
- Most DDP vs FSDP2 tradeoffs reduce to replication being simpler and faster, while sharding is more memory-efficient.

---

## Optimizer State

The tensors maintained by the optimizer in addition to model parameters and gradients.

In this repo:
- The optimizer is built in `dinov3/train/train.py:430`.
- The DDP vs FSDP2 decision must include optimizer state, not just raw model weights.

Example:
- Even when a ViT-B checkpoint is small, Adam-style optimizers can still add substantial memory overhead through extra moment tensors.

Why this matters:
- "The model fits" is not enough. The real question is whether parameters, gradients, optimizer state, and activations all fit together at the target batch size.

---

## VRAM / OOM

`VRAM` means GPU memory. `OOM` means an out-of-memory failure when allocations exceed available memory.

In this repo:
- The strategy knob for dealing with memory pressure is `train.distributed_strategy` in `dinov3/configs/ssl_default_config.yaml:82`.

Example:
- If DDP only works at a batch size that is too small to be useful, that is still effectively a memory-limited case and a reason to test FSDP2.

Why this matters:
- The main reason to move from DDP to FSDP2 is memory pressure, not the fact that the job is distributed.

---

## MAC

Multiply-Accumulate. In this repo's MFU math, `1 MAC = 1 multiply-add`.

In this repo:
- `vit_forward_flops(...)` defines the MAC convention in `dinov3/utils/mfu.py:26-39`.
- `compute_dino_flops_per_image(...)` returns MACs per image in `dinov3/utils/mfu.py:52-104`.
- `compute_mfu(...)` converts MACs to hardware FLOPs with a `2x` factor in `dinov3/utils/mfu.py:107-134`.

Example:
- The training loop logs `GMACs/image` before entering the main loop in `dinov3/train/train.py:459-480`.

Why this matters:
- Repo-level GMAC numbers are not directly comparable to hardware TFLOPS numbers until the `2x` MAC-to-FLOP conversion is applied.

---

## MFU

Model FLOP Utilization: achieved training FLOPs divided by a hardware peak denominator.

In this repo:
- FLOP counting utilities live in `dinov3/utils/mfu.py`.
- MFU is computed in the training loop in `dinov3/train/train.py:610-614`.

Example:
- `compute_dino_flops_per_image(...)` estimates MACs per image in `dinov3/utils/mfu.py:53-108`.

Why this matters:
- MFU math in this repo starts from MACs, not directly from hardware FLOPs.
- MFU is useful for relative comparisons inside this repo, but it is only one lens on performance.
