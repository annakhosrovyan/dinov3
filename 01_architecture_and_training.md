# DINOv3 Satellite Fork: Architecture And Training

## Repository purpose

Code fact: The default training stack is a self-supervised DINO + iBOT setup for satellite imagery, with a ViT student, an EMA teacher, optional Gram regularization, five-channel inputs, and multi-crop training configured in `dinov3/configs/ssl_default_config.yaml`, `dinov3/train/ssl_meta_arch.py`, and `dinov3/models/vision_transformer.py`.

Code fact: The main runtime entrypoint is `dinov3/train/train.py`. `main()` sets up distributed state, merges config, instantiates the meta-architecture on the `meta` device, prepares distributed wrapping, materializes parameters on CUDA, and then enters `do_train()` (`dinov3/train/train.py:753-821`).

## Main training entrypoints

Code fact: `dinov3/train/train.py:423-750` is the core training loop. The sequence per iteration is schedule update, `optimizer.zero_grad(set_to_none=True)`, `model.forward_backward(...)`, optional gradient clipping, metric all-reduce, `optimizer.step()`, EMA update, CUDA-event timing, logging, periodic eval, and periodic checkpoint save.

Code fact: `setup_job()` and `setup_config()` in `dinov3/configs/config.py:93-106` and `dinov3/configs/config.py:173-217` are the configuration and process bootstrap path. Config resolution is `default yaml -> experiment yaml -> CLI dot overrides`, using OmegaConf.

Code fact: Learning-rate scaling is not baked into YAML constants. `apply_scaling_rules_to_cfg()` mutates `cfg.optim.lr` for legacy schedules, while `build_schedulers_v2()` applies scaling at schedule-construction time for the v2 path (`dinov3/configs/config.py:40-54`, `dinov3/train/train.py:185-254`).

Code fact: The repo still has a second meta-architecture, `MultiDistillationMetaArch`, which subclasses the SSL path and adds subgroup broadcast logic for multi-distillation (`dinov3/train/multidist_meta_arch.py:16-165`). The default config uses `MODEL.META_ARCHITECTURE: SSLMetaArch` (`dinov3/configs/ssl_default_config.yaml:1-2`).

## Model architecture

Code fact: `SSLMetaArch` builds three backbone-bearing module dicts: student, teacher/EMA, and optional Gram teacher (`dinov3/train/ssl_meta_arch.py:34-63`). The student and teacher each include a backbone, a DINO head, and a separate iBOT head (`dinov3/train/ssl_meta_arch.py:75-129`).

Code fact: The default backbone is `vit_base`, which maps to `DinoVisionTransformer(embed_dim=768, depth=12, num_heads=12, ffn_ratio=4)` in `dinov3/models/vision_transformer.py:344-353`. The default config sets `student.arch=vit_base`, `patch_size=16`, `in_chans=5`, and `n_storage_tokens=0` (`dinov3/configs/ssl_default_config.yaml:87-115`).

Code fact: `build_model_from_cfg()` instantiates both student and teacher on the `meta` device and sizes the backbone from the configured global crop size (`dinov3/models/__init__.py:83-97`). This keeps model construction cheap before FSDP/DDP wrapping.

Code fact: `DinoVisionTransformer` uses a convolutional patch embed, a learned CLS token, optional storage tokens, axial RoPE, a stack of `SelfAttentionBlock`s, and a final norm (`dinov3/models/vision_transformer.py:59-179`, `dinov3/layers/patch_embed.py:21-89`, `dinov3/layers/rope_position_encoding.py:16-121`).

Code fact: The attention implementation uses `torch.nn.functional.scaled_dot_product_attention` rather than a handwritten attention kernel (`dinov3/layers/attention.py:106-118`). The RoPE path rotates Q and K only after token-prefix handling for CLS and storage tokens (`dinov3/layers/attention.py:66-85`).

Code fact: Multi-crop is handled inside the backbone rather than above it. `forward_features_list()` prepares tokens for each crop tensor, computes RoPE per resolution, then runs every transformer block across the list of inputs (`dinov3/models/vision_transformer.py:222-261`).

Code fact: `SelfAttentionBlock._forward_list()` concatenates crop tensors to share normalization and MLP elementwise work, but `SelfAttention.forward_list()` still splits by sequence shape before calling attention (`dinov3/layers/block.py:126-210`, `dinov3/layers/attention.py:94-118`).

Inference: The intended optimization is not “one giant mixed-resolution attention kernel.” The code only fuses the parts that can share shape-agnostic tensor work; attention itself remains resolution-specific.

## Forward/backward structure

Code fact: `SSLMetaArch.forward_backward()` moves collated tensors to GPU with `non_blocking=True`, runs teacher forward on global crops, runs student forward on global plus local crops, optionally runs Gram teacher forward, computes DINO + KoLeo + iBOT + optional Gram losses, and then calls `loss.backward()` (`dinov3/train/ssl_meta_arch.py:379-460`, `dinov3/train/ssl_meta_arch.py:741-742`).

Code fact: Teacher forward consumes global crops only and extracts CLS, storage, and patch tokens before applying DINO and iBOT heads (`dinov3/train/ssl_meta_arch.py:462-505`).

Code fact: Student forward calls the backbone once with a list containing global crops and local crops, then applies iBOT to masked global patches and DINO to concatenated global/local CLS tokens (`dinov3/train/ssl_meta_arch.py:561-613`).

Code fact: Loss wiring is in `compute_losses()` and matches the repo’s training objective structure: DINO local, DINO global, KoLeo on global CLS features, iBOT masked-patch loss, and optional Gram loss (`dinov3/train/ssl_meta_arch.py:615-715`).

Code fact: Teacher EMA updates are done with fused foreach ops, not Python-side per-parameter loops, after the optimizer step (`dinov3/train/ssl_meta_arch.py:744-757`).

## Optimizer, schedules, and checkpointing

Code fact: The optimizer is AdamW built over parameter groups emitted by `model.get_params_groups()` (`dinov3/train/train.py:107-108`, `dinov3/train/train.py:430`). Parameter groups carry per-layer LR multipliers, per-group weight-decay multipliers, and “last layer” flags from `dinov3/train/param_groups.py:125-170`.

Code fact: The default scheduler family is WSD, not cosine. `build_schedulers()` chooses `WSDLRScheduler` when `cfg.optim.lr_scheduler == "wsd"` (`dinov3/train/train.py:146-167`, `dinov3/configs/ssl_default_config.yaml:132-149`).

Code fact: The config defaults include `layerwise_decay=0.9`, `patch_embed_lr_mult=0.2`, `clip_grad=3.0`, and `multi_tensor_optim=true`, so the optimizer path is explicitly tuned for per-layer scaling and fused foreach/fused AdamW param groups (`dinov3/configs/ssl_default_config.yaml:143-152`, `dinov3/train/ssl_meta_arch.py:817-839`).

Code fact: Checkpoints are written through PyTorch Distributed Checkpoint (DCP), not ad hoc rank-0 `torch.save`, using `dcpsd.get_model_state_dict()` and `dcpsd.get_optimizer_state_dict()` (`dinov3/checkpointer/checkpointer.py:191-282`).

Code fact: Resume is automatic when `find_latest_checkpoint(ckpt_dir)` finds an integer-named checkpoint directory (`dinov3/train/train.py:445-457`, `dinov3/checkpointer/checkpointer.py:336-351`).

Code fact: Consolidated pretrained checkpoints are adapted to the satellite fork during load. `adapt_patch_embed_input_channels()` expands `patch_embed.proj.weight` from 3 input channels to 5 by copying RGB weights and filling the extra channels with the RGB mean (`dinov3/checkpointer/checkpointer.py:107-158`, `dinov3/checkpointer/checkpointer.py:427-510`).

## Data path that changes training behavior

Code fact: The default dataset path is a single encoded string that is parsed into `MixedSatelliteDataset` plus per-source kwargs (`dinov3/configs/ssl_default_config.yaml:64-86`, `dinov3/data/loaders.py:46-128`).

Code fact: `MixedSatelliteDataset` mixes Sentinel-1, Sentinel-2A/B, BEN, Sen12MS, Intelinair, MAID, and NAIP by assigning each source an “effective size” of `int(raw_size * weight)` and then modulo-mapping back into the raw dataset (`dinov3/data/datasets/mixed_satlas_dataset.py:14-180`).

Inference: Dataset weighting changes both optimization behavior and I/O locality, because oversampled datasets will repeat samples more often within a nominal epoch without duplicating files on disk.

Code fact: All dataset types are normalized to 5 channels through `to_five_channels()` before model ingestion (`dinov3/data/datasets/channel_utils.py:15-26`, `dinov3/data/datasets/mixed_satlas_dataset.py:172-177`).

Code fact: The augmentation path is dataset-aware. `SSLMetaArch.build_data_augmentation_dino()` routes MAID samples to `MAIDAugmentation` and all others to `HDF5Augmentation` (`dinov3/train/ssl_meta_arch.py:778-815`).

Code fact: Both augmentation classes use Albumentations-based random resized crops, flips/rotations, Gaussian blur, and `ToTensorV2` to emit 2 global crops and `local_crops_number` local crops on CPU (`dinov3/data/datasets/augmentation.py:9-81`, `dinov3/data/datasets/hdf5_augmentation.py:10-77`).

Code fact: The collate path materializes all crops for the batch, builds iBOT masks in Python/NumPy, casts crop tensors to the target precision on CPU, and returns mask indices plus weights for the masked-token loss (`dinov3/data/collate.py:11-78`, `dinov3/data/masking.py:12-94`).

Inference: For this repo, the “data loader” is not just file I/O. It also includes multi-crop augmentation, mask generation, dtype conversion, and batch reshaping before H2D transfer.
