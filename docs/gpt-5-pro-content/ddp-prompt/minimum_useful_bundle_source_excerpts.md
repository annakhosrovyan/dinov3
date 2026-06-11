# GPT-5 Pro DDP Prompt Minimum Useful Bundle Source Excerpts

Purpose: paste this after `ddp-prompt/compressed_phase6_perf_plan.md` when using
`ddp-prompt/ideation_prompt.md`. This file consolidates the Python source excerpts
called out by the DDP prompt's "Minimum useful bundle" into one markdown artifact.

Notes:

- These are cropped excerpts, not full files.
- The cuts are centered on the DDP + CUDA-graph execution path, dynamic iBOT masking,
  loss/collective behavior, and the train-loop timing/logging boundary.
- `compressed_phase6_perf_plan.md` remains a separate companion file.


## 1. Compile and CUDA-Graph Gating for Backbone Blocks

Source: `dinov3/fsdp/ac_compile_parallelize.py:24-126`

----- BEGIN dinov3/fsdp/ac_compile_parallelize.py:24-126 -----

```python

def get_activation_checkpoint_wrapper(cfg):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    if cfg.train.checkpointing_full:
        _checkpointing_wrapper = checkpoint_wrapper
        logger.info("using selective checkpointing on backbone with full checkpointing policy")
    else:
        _save_list = [
            # mm
            torch.ops.aten.mm.default,
            torch.ops.aten._scaled_mm.default,
            # attentions
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
        ]
        _checkpointing_wrapper = partial(
            checkpoint_wrapper,
            context_fn=partial(create_selective_checkpoint_contexts, _save_list),
            preserve_rng_state=True,
        )
        logger.info("using selective checkpointing on backbone with selective policy")
    return _checkpointing_wrapper


def activation_checkpoint_convnext(cfg, model: nn.Module):
    _checkpointing_wrapper = get_activation_checkpoint_wrapper(cfg)
    for stage_id, stage in enumerate(model.stages):
        for block_id, block in enumerate(stage):
            model.stages[stage_id][block_id] = _checkpointing_wrapper(block)
    for dsl_id, dsl in enumerate(model.downsample_layers):
        model.downsample_layers[dsl_id] = _checkpointing_wrapper(dsl)


def activation_checkpoint_transformer(cfg, model: nn.Module):
    _checkpointing_wrapper = get_activation_checkpoint_wrapper(cfg)
    for block_id, b in enumerate(model.blocks):
        model.blocks[block_id] = _checkpointing_wrapper(b)


def wrap_compile_block(
    module: nn.Module,
    use_cuda_graphs: bool,
    is_backbone_block: bool,
    compile_mode: str | None = None,
) -> nn.Module:
    if use_cuda_graphs and is_backbone_block:
        module.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": True})
    elif compile_mode is not None:
        module.compile(mode=compile_mode)
    else:
        module.compile()
    return module


def _get_compile_mode(cfg) -> str | None:
    """Return the compile mode string, or None for PyTorch default."""
    mode = getattr(cfg.train, "compile_mode", None)
    return None if (mode is None or mode == "default") else str(mode)


def _compile_path_desc(use_cuda_graphs: bool, is_backbone_block: bool, compile_mode: str | None) -> str:
    """Human-readable description of which wrap_compile_block branch will run."""
    if use_cuda_graphs and is_backbone_block:
        return "fullgraph=True, dynamic=False, triton.cudagraphs=True"
    elif compile_mode is not None:
        return f"mode={compile_mode!r}"
    else:
        return "default (module.compile(), dynamic=True)"


def compile_convnext(cfg, model: nn.Module):
    compile_mode = _get_compile_mode(cfg)
    use_cuda_graphs = getattr(cfg.train, "cudagraphs", False)
    n_stages = len(model.stages)
    n_dsl = len(model.downsample_layers)
    logger.info(
        "[COMPILE] compile_convnext: %d stages + %d downsample layers → path: %s",
        n_stages, n_dsl, _compile_path_desc(use_cuda_graphs, is_backbone_block=False, compile_mode=compile_mode),
    )
    assert isinstance(model.stages, nn.ModuleList)
    # Compile at stage level
    for stage_id, stage in enumerate(model.stages):
        model.stages[stage_id] = wrap_compile_block(stage, use_cuda_graphs, is_backbone_block=False, compile_mode=compile_mode)
    assert isinstance(model.downsample_layers, nn.ModuleList)
    for dsl_id, dsl in enumerate(model.downsample_layers):
        model.downsample_layers[dsl_id] = wrap_compile_block(dsl, use_cuda_graphs, is_backbone_block=False, compile_mode=compile_mode)


def compile_transformer(cfg, model: nn.Module):
    compile_mode = _get_compile_mode(cfg)
    use_cuda_graphs = getattr(cfg.train, "cudagraphs", False)
    n_blocks = len(model.blocks)
    logger.info(
        "[COMPILE] compile_transformer: %d backbone blocks → path: %s  (cudagraphs=%s, compile_mode=%s)",
        n_blocks,
        _compile_path_desc(use_cuda_graphs, is_backbone_block=True, compile_mode=compile_mode),
        use_cuda_graphs,
        compile_mode,
    )
    assert isinstance(model.blocks, nn.ModuleList)
    for block_id, block in enumerate(model.blocks):
```

----- END dinov3/fsdp/ac_compile_parallelize.py:24-126 -----

## 2. DDP Wrapping Path Used by the Current Champion Configuration

Source: `dinov3/fsdp/ac_compile_parallelize.py:315-353`

----- BEGIN dinov3/fsdp/ac_compile_parallelize.py:315-353 -----

```python

def _ac_compile_parallelize_ddp(
    trained_model: nn.ModuleDict,
    inference_only_models: List[nn.ModuleDict],
    cfg: Any,
    trained_model_process_group: dist.ProcessGroup | None = None,
    inference_only_models_process_groups: List[dist.ProcessGroup] | None = None,
) -> None:
    """DDP distributed wrapping — no sharding, just all-reduce gradients.

    ViT-B is ~172 MB in BF16 and fits trivially on 80 GB H100. DDP avoids the
    all_gather / reduce_scatter overhead of FSDP2 and replaces it with a single
    gradient all-reduce per step.
    """
    all_models = [trained_model] + inference_only_models

    # Move all models to CUDA first
    for model in all_models:
        model.to_empty(device="cuda")

    # Cast parameters to the configured dtype (bf16 by default)
    DTYPE_MAP = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    param_dtype = DTYPE_MAP[cfg.compute_precision.param_dtype]
    for model in all_models:
        model.to(param_dtype)

    # Wrap each student sub-model with DDP
    pg = trained_model_process_group
    for k in trained_model.keys():
        trained_model[k] = nn.parallel.DistributedDataParallel(
            trained_model[k],
            process_group=pg,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
```

----- END dinov3/fsdp/ac_compile_parallelize.py:315-353 -----

## 3. Forward/Backward Step Skeleton in SSLMetaArch

Source: `dinov3/train/ssl_meta_arch.py:379-465`

----- BEGIN dinov3/train/ssl_meta_arch.py:379-465 -----

```python
    def forward_backward(
        self, data, *, teacher_temp, iteration=0, **ignored_kwargs
    ) -> tuple[Tensor, dict[str, float | Tensor]]:
        del ignored_kwargs
        # Required when triton.cudagraphs=True (train.cudagraphs=true): tells the Inductor
        # CUDA graph tree that a new step is starting so it does not alias output buffers
        # across the teacher forward, student forward, and the two crop-size sub-graphs
        # inside _forward_list. Safe to call unconditionally — no-op when cudagraphs are off.
        torch.compiler.cudagraph_mark_step_begin()
        metrics_dict = {}

        # Shapes
        n_global_crops = 2
        n_local_crops = self.n_local_crops  # self.cfg.crops.local_crops_number
        B = data["collated_local_crops"].shape[0] // n_local_crops
        assert data["collated_global_crops"].shape[0] == n_global_crops * B
        metrics_dict["local_batch_size"] = B
        metrics_dict["global_batch_size"] = data["global_batch_size"]

        nvtx = self._nvtx

        with nvtx("H2D_transfer"):
            global_crops = data["collated_global_crops"].cuda(non_blocking=True)
            local_crops = data["collated_local_crops"].cuda(non_blocking=True)
            masks = data["collated_masks"].cuda(non_blocking=True)
            mask_indices_list = data["mask_indices_list"].cuda(non_blocking=True)
            masks_weight = data["masks_weight"].cuda(non_blocking=True)
            n_masked_patches_tensor = data["n_masked_patches"].cuda(non_blocking=True)

            if self.has_gram_teacher:
                assert "collated_gram_teacher_crops" in data, (
                    "no gram teacher crops in the data, have you set cfg.crops.gram_teacher_crops_size?"
                )
                gram_teacher_crops = data["collated_gram_teacher_crops"].cuda(non_blocking=True)
            else:
                gram_teacher_crops = None

        # Teacher output (will trigger an all-gather to unshard)
        with nvtx("teacher_fwd"):
            teacher_global = self.get_teacher_output(
                global_crops.unflatten(0, (n_global_crops, B)),
                teacher_temp=teacher_temp,
                n_masked_patches_tensor=n_masked_patches_tensor,
                mask_indices_list=mask_indices_list,
                upperbound=data["upperbound"],
            )

        # Student output (will trigger an all-gather to unshard)
        with nvtx("student_fwd"):
            student_global, student_local = self.get_student_output(
                global_crops=global_crops.unflatten(0, (n_global_crops, B)),
                local_crops=local_crops.unflatten(0, (n_local_crops, B)),
                upperbound=data["upperbound"],
                masks=masks,
                mask_indices_list=mask_indices_list,
            )

        # Gram output
        if self.gram_use_loss:
            with nvtx("gram_fwd"):
                gram_global = self.get_gram_teacher_output(
                    gram_teacher_crops.unflatten(0, (n_global_crops, B)) if gram_teacher_crops is not None else None,
                    masks=masks,
                    teacher_global=teacher_global,
                    student_global=student_global,
                    student_global_crops_size=global_crops.shape[-1],
                )
        else:
            gram_global = {}

        # Compute losses and backprop
        with nvtx("losses_backward"):
            loss_accumulator, loss_dict = self.compute_losses(
                teacher_global=teacher_global,
                student_global=student_global,
                student_local=student_local,
                gram_global=gram_global,
                masks=masks,
                mask_indices_list=mask_indices_list,
                masks_weight=masks_weight,
                iteration=iteration,
            )

            self.backprop_loss(loss_accumulator)

        # Return total weighted loss and a dict of metrics to log
        return loss_accumulator, metrics_dict | loss_dict
```

----- END dinov3/train/ssl_meta_arch.py:379-465 -----

## 4. Teacher Path: Masked-Patch Gather and Centering

Source: `dinov3/train/ssl_meta_arch.py:467-510`

----- BEGIN dinov3/train/ssl_meta_arch.py:467-510 -----

```python
    @torch.no_grad()
    def get_teacher_output(
        self,
        images,
        *,
        upperbound,
        mask_indices_list,
        teacher_temp,
        n_masked_patches_tensor,
    ):
        n_crops, B, rgb, H, W = images.shape
        images = images.flatten(0, 1)

        backbone_out = self.teacher.backbone(images, is_training=True)
        cls = backbone_out["x_norm_clstoken"]  # [n_crops * B, D]
        reg = backbone_out["x_storage_tokens"]  # [n_crops * B, R, D]
        ibot_patch = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P, D]

        # IBOT head only on patches that are masked for the student
        buffer = torch.index_select(ibot_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        masked_patch_after_head = self.teacher.ibot_head(buffer)

        # DINO head on CLS tokens
        cls_after_head = self.teacher.dino_head(cls)  # [n_crops * B, K]

        # Center with sinkhorn-knopp
        cls_centered = self.dino_loss.sinkhorn_knopp_teacher(
            cls_after_head, teacher_temp=teacher_temp
        )  # [n_crops * B, K]
        cls_centered = cls_centered.unflatten(0, (n_crops, B))  # [n_crops, B, K]
        masked_patch_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
            masked_patch_after_head,
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
        )  # [n_masked_patches, K]

        return {
            "cls_pre_head": cls.unflatten(0, [n_crops, B]),  # [n_crops, B, D]
            "reg_pre_head": reg.unflatten(0, [n_crops, B]),  # [n_crops, B, R, D]
            "patch_pre_head": ibot_patch.unflatten(0, [n_crops, B]),  # [n_crops, B, P, D]
            "cls_after_head": cls_after_head.unflatten(0, [n_crops, B]),  # [n_crops, B, K]
            "cls_centered": cls_centered,  # [n_crops, B, K]
            "masked_patch_centered": masked_patch_centered,  # [n_masked_patches, K]
        }
```

----- END dinov3/train/ssl_meta_arch.py:467-510 -----

## 5. Student Path: Joint Global/Local Backbone Pass and Head Dispatch

Source: `dinov3/train/ssl_meta_arch.py:566-618`

----- BEGIN dinov3/train/ssl_meta_arch.py:566-618 -----

```python
    def get_student_output(self, *, global_crops, local_crops, upperbound, masks, mask_indices_list):
        n_global_crops, B, rgb, H, W = global_crops.shape
        n_local_crops, B, rgb, H, W = local_crops.shape

        global_crops = global_crops.flatten(0, 1)

        # Forward global and local crops through the student backbone jointly
        global_out, local_out = self.student.backbone(
            [global_crops, local_crops.flatten(0, 1)],
            masks=[masks if not self.is_distillation_enabled else None, None],
            is_training=True,
        )
        g_cls, g_reg, g_patch = (
            global_out["x_norm_clstoken"],
            global_out["x_storage_tokens"],
            global_out["x_norm_patchtokens"],
        )
        l_cls, l_reg, l_patch = (
            local_out["x_norm_clstoken"],
            local_out["x_storage_tokens"],
            local_out["x_norm_patchtokens"],
        )

        # IBOT head only on masked patches
        masked_patches_pre_head = torch.index_select(g_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        global_masked_patch_after_head = self.student.ibot_head(masked_patches_pre_head)

        # DINO head on CLS tokens (all in one pass)
        buffer = [
            g_cls,  # [n_global_crops * B, D]
            l_cls,  # [n_local_crops * B, D]
        ]
        sizes = [x.shape[0] for x in buffer]
        buffer = torch.cat(buffer, dim=0)  # [n_global_crops * B + n_local_crops * B, D]
        buffer = self.student.dino_head(buffer)  # [n_global_crops * B + n_local_crops * B, K]
        buffer = torch.split_with_sizes(buffer, sizes, dim=0)

        global_out = {
            "cls_pre_head": g_cls.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, D]
            "reg_pre_head": g_reg.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, R, D]
            "patch_pre_head": g_patch.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, P, D]
            "cls_after_head": buffer[0].unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, K],
            "masked_patch_after_head": global_masked_patch_after_head,  # [n_masked_patches, K]
            "masked_patch_pre_head": masked_patches_pre_head,  # [n_masked_patches, D]
        }
        local_out = {
            "cls_pre_head": l_cls.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, D]
            "reg_pre_head": l_reg.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, R, D]
            "patch_pre_head": l_patch.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, P, D]
            "cls_after_head": buffer[1].unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, K],
        }

        return global_out, local_out
```

----- END dinov3/train/ssl_meta_arch.py:566-618 -----

## 6. Loss Assembly: DINO, iBOT, and Weighting Logic

Source: `dinov3/train/ssl_meta_arch.py:620-720`

----- BEGIN dinov3/train/ssl_meta_arch.py:620-720 -----

```python
    def compute_losses(
        self,
        *,
        teacher_global,
        student_global,
        student_local,
        gram_global,
        masks,
        mask_indices_list,
        masks_weight,
        iteration,
    ):
        n_global_crops = student_global["cls_after_head"].shape[0]
        n_local_crops = student_local["cls_after_head"].shape[0]
        loss_dict = {}
        loss_accumulator = 0.0

        # Loss scales like in DINOv2, these are multiplied with the loss weights from the config
        dino_global_terms = (
            n_global_crops * (n_global_crops - 1) if self.dino_global_ignore_diagonal else n_global_crops**2
        )
        dino_local_terms = n_global_crops * n_local_crops
        dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
        dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)
        koleo_scale = n_global_crops

        # DINO local loss: compare post-head CLS tokens: student(local crops) vs. teacher(global crops)
        dino_local_crops_loss = self.dino_loss(
            student_logits=student_local["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
        )
        loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

        # Reweighting of DINO loss
        if self.cfg.dino.reweight_dino_local_loss:
            local_weight = self.dino_local_loss_schedule[iteration]
        else:
            local_weight = 1.0

        loss_dict["dino_local_loss_weight"] = local_weight
        loss_accumulator += self.dino_loss_weight * dino_local_scale * local_weight * dino_local_crops_loss

        # DINO global loss: compare post-head CLS tokens: student(global crops) vs. teacher(global crops)
        dino_global_crops_loss = self.dino_loss(
            student_logits=student_global["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
            ignore_diagonal=self.dino_global_ignore_diagonal,
        )
        loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
        loss_accumulator += self.dino_loss_weight * dino_global_scale * dino_global_crops_loss

        # Koleo: regularize pre-head CLS tokens of student(global crops)
        koleo_loss = sum(self.koleo_loss(x) for x in student_global["cls_pre_head"]) / n_global_crops
        loss_dict["koleo_loss"] = koleo_loss
        loss_accumulator += self.dino_koleo_loss_weight * koleo_scale * koleo_loss

        # IBOT loss
        ibot_patch_loss = self.ibot_patch_loss.forward_masked(
            student_global["masked_patch_after_head"],
            teacher_global["masked_patch_centered"],
            student_masks_flat=masks,
            n_masked_patches=mask_indices_list.shape[0],
            masks_weight=masks_weight,
        )
        loss_dict["ibot_loss"] = ibot_patch_loss
        loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        # Gram loss
        if self.gram_use_loss:
            gram_loss = self.gram_loss(
                gram_global["student_patches"],
                gram_global["teacher_patches"],
                img_level=self.gram_img_level,
            )

            if self.gram_loss_schedule is not None:
                gram_loss_weight = self.gram_loss_schedule[iteration]
            else:
                gram_loss_weight = self.gram_loss_weight

            loss_dict["gram_loss_weight"] = gram_loss_weight
            loss_accumulator += gram_loss * gram_loss_weight
            loss_dict["gram_loss"] = gram_loss

            if self.gram_compute_stats:
                with torch.no_grad():
                    # Save stats over masked / unmasked tokens
                    gram_loss_masked = self.gram_loss(
                        gram_global["orig_student_patches"][masks].detach(),
                        gram_global["orig_teacher_patches"][masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/masked_gram_loss"] = gram_loss_masked
                    gram_loss_unmasked = self.gram_loss(
                        gram_global["orig_student_patches"][~masks].detach(),
                        gram_global["orig_teacher_patches"][~masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/unmasked_gram_loss"] = gram_loss_unmasked

        return loss_accumulator, loss_dict
```

----- END dinov3/train/ssl_meta_arch.py:620-720 -----

## 7. Collate Path: Dynamic iBOT Mask Construction and n_masked Derivation

Source: `dinov3/data/collate.py:10-77`

----- BEGIN dinov3/data/collate.py:10-77 -----

```python

def collate_data_and_cast(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
    random_circular_shift=False,
    local_batch_size=None,
):
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack(
        [s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
    )  # [n_global_crops, B, ...]
    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    if "gram_teacher_crops" in samples_list[0][0]:
        collated_gram_teacher_crops = torch.stack(
            [s[0]["gram_teacher_crops"][i] for i in range(n_global_crops) for s in samples_list]
        )  # [n_global_crops, B, ...]
    else:
        collated_gram_teacher_crops = None

    if local_batch_size is not None:
        # multi-distillation case, number of masks is different because the number of samples masked
        # is different of the number of samples passed into the teacher initially
        B = n_global_crops * local_batch_size
    else:
        B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_max = probs[i + 1]
        mask = torch.BoolTensor(mask_generator(int(N * prob_max)))
        if random_circular_shift:  # apply le random circular shift to
            shift_x, shift_y = (
                random.randint(0, mask.shape[0] - 1),
                random.randint(0, mask.shape[1] - 1),
            )
            mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
        masks_list.append(mask)
        upperbound += int(N * prob_max)
    for _ in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    out = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
    if collated_gram_teacher_crops is not None:
        out["collated_gram_teacher_crops"] = collated_gram_teacher_crops.to(dtype)
```

----- END dinov3/data/collate.py:10-77 -----

## 8. iBOT Patch Loss and Center Update Path

Source: `dinov3/loss/ibot_patch_loss.py:60-141`

----- BEGIN dinov3/loss/ibot_patch_loss.py:60-141 -----

```python

class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.full((1, 1, patch_out_dim), math.nan))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None
        self.sinkhorn_knopp_teacher = SinkhornKnoppTeacher()
        self.sinkhorn_knopp_teacher.compile()

    def init_weights(self) -> None:
        self.center.zero_()

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp, update_centers=True):
        if update_centers:
            self.apply_center_update()
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks_flat):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_patch_tokens: (B, N, D) tensor
        teacher_patch_tokens: (B, N, D) tensor
        student_masks_flat: (B, N) tensor
        """
        t = teacher_patch_tokens
        s = student_patch_tokens
        loss = lossfunc(t, s, self.student_temp)
        loss = torch.sum(loss * student_masks_flat.float(), dim=-1) / student_masks_flat.sum(dim=-1).clamp(min=1.0)
        return -loss.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        # loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = lossfunc(t, s, self.student_temp)
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0]

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(teacher_patch_tokens.mean(1), dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True, group=get_process_subgroup())

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = get_subgroup_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

```

----- END dinov3/loss/ibot_patch_loss.py:60-141 -----

## 9. DINO CLS-Token Loss and Async Center All-Reduce Path

Source: `dinov3/loss/dino_clstoken_loss.py:15-123`

----- BEGIN dinov3/loss/dino_clstoken_loss.py:15-123 -----

```python

class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.full((1, out_dim), math.nan))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    def init_weights(self) -> None:
        self.center.zero_()

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp, update_centers=True):
        if update_centers:
            self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        # teacher_output: [batch, prototypes]
        teacher_output = teacher_output.float()
        world_size = get_subgroup_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q, group=get_process_subgroup())
        Q /= sum_Q

        for _ in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows, group=get_process_subgroup())
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, student_logits, teacher_probs, ignore_diagonal=False):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_logits: [student crops, batch, prototypes]
        teacher_probs:  [teacher crops, batch, prototypes] must sum to 1 over the last dim

        loss = 0
        count = 0
        for each sample `b` in the batch:
            for each student crop `s` of this sample:
                for each teacher crop `t` of this sample:
                    if ignore_diagonal and s == t:
                        continue
                    loss += cross_entropy(softmax(student_logits[s, b] / student_temp), teacher_probs[t, b])
                    count += 1
        return loss / count
        """
        student_crops, B, K = student_logits.shape
        teacher_crops, _, _ = teacher_probs.shape
        student_logits = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        if not ignore_diagonal:
            loss = -torch.einsum("s b k, t b k -> ", student_logits, teacher_probs)
            return loss / (B * student_crops * teacher_crops)
        else:
            loss = -torch.einsum("s b k, t b k -> s t", student_logits, teacher_probs)
            min_st = min(student_crops, teacher_crops)
            loss = torch.diagonal_scatter(loss, loss.new_zeros(min_st))
            return loss.sum() / (B * student_crops * teacher_crops - B * min_st)

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True, group=get_process_subgroup())

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = get_subgroup_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

```

----- END dinov3/loss/dino_clstoken_loss.py:15-123 -----

## 10. Train Loop: MFU Setup, CUDA-Event Timing, Logging, and Wall-Clock Boundary

Source: `dinov3/train/train.py:427-747`

----- BEGIN dinov3/train/train.py:427-747 -----

```python
def do_train(cfg, model, resume=False):
    _mem_profile = memory_profile_enabled()
    # Period for fragmentation tracking (every N iters). Default 25; override via env.
    _mem_profile_period = int(os.environ.get("DINOV3_MEMORY_PROFILE_PERIOD", "25"))
    process_subgroup = distributed.get_process_subgroup()
    ckpt_dir = Path(cfg.train.output_dir, "ckpt").expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    # Optimizer
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)
    if cfg.multidistillation.enabled:
        register_dont_save_hooks(
            model,
            dont_save=[k for k, _ in model.state_dict().items() if k.startswith("teacher")],
        )
    model.init_weights()
    if _mem_profile:
        log_phase_memory("post_init_weights")
    start_iter = 0
    if resume and (last_checkpoint_dir := find_latest_checkpoint(ckpt_dir)):
        logger.info(f"Checkpoint found {last_checkpoint_dir}")
        start_iter = (
            load_checkpoint(
                last_checkpoint_dir,
                model=model,
                optimizer=optimizer,
                strict_loading=False,
                process_group=process_subgroup,
            )
            + 1
        )
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    if cfg.multidistillation.enabled:
        global_batch_size = cfg.multidistillation.global_batch_size
    else:
        global_batch_size = cfg.train.batch_size_per_gpu * distributed.get_world_size()

    # Precompute MAC constant for MFU tracking (done once, outside the loop)
    num_gpus = distributed.get_world_size()
    # Unwrap DDP if present to access backbone attributes
    _backbone_raw = model.student.backbone
    if hasattr(_backbone_raw, "module"):
        _backbone_raw = _backbone_raw.module
    macs_per_image = compute_dino_flops_per_image(
        global_crop_size=cfg.crops.global_crops_size,
        local_crop_size=cfg.crops.local_crops_size,
        patch_size=cfg.student.patch_size,
        n_global_crops=2,
        n_local_crops=cfg.crops.local_crops_number,
        hidden_dim=_backbone_raw.embed_dim,
        num_layers=_backbone_raw.n_blocks,
        ffn_ratio=getattr(cfg.student, "ffn_ratio", 4.0),
        n_registers=cfg.student.n_storage_tokens,
        gram_enabled=cfg.gram.use_loss,
        head_overhead_pct=0.05,
    )
    logger.info(f"MFU tracking: {macs_per_image/1e9:.1f} GMACs/image, {num_gpus} GPUs")

    # Build data loader
    data_loader = build_multi_resolution_data_loader_from_cfg(
        cfg=cfg,
        model=model,
        start_iter=start_iter,
    )

    # Metric logging
    logger.info("Starting training from iteration %d", start_iter)
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    wandb_module = None
    wandb_run = None
    if cfg.wandb.enabled and distributed.is_main_process():
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise RuntimeError(
                "wandb logging is enabled but the wandb package is not installed."
            ) from exc
        init_kwargs = {
            "project": cfg.wandb.project,
            "name": cfg.wandb.run_name or None,
            "group": cfg.wandb.group or None,
            "dir": cfg.train.output_dir,
            "resume": "allow",
        }
        if cfg.wandb.entity:
            init_kwargs["entity"] = cfg.wandb.entity
        if cfg.wandb.tags:
            init_kwargs["tags"] = list(cfg.wandb.tags)
        wandb_run = wandb_module.init(**init_kwargs)
    # Manual garbage collection
    gc.disable()
    gc.collect()

    # Profiling setup (gated by args.profiling passed via cfg)
    profiling_enabled = getattr(cfg.train, "_profiling", False)
    nvtx = make_nvtx(profiling_enabled)
    profiler = None
    if profiling_enabled:
        logger.info("Profiling mode enabled")
        profiler = build_profiler(
            output_dir=cfg.train.output_dir,
            warmup=int(getattr(cfg.train, "_profiler_warmup", 5)),
            active=int(getattr(cfg.train, "_profiler_active", 3)),
            repeat=int(getattr(cfg.train, "_profiler_repeat", 1)),
            rank=distributed.get_rank(),
        )
        if getattr(cfg.train, "_graph_break_log", False):
            enable_graph_break_logging()
        # Log static run metadata once
        run_meta = get_run_metadata(cfg)
        logger.info("Run metadata: %s", run_meta)
        # Pass nvtx factory to model for inner NVTX ranges
        if hasattr(model, "set_nvtx"):
            model.set_nvtx(nvtx)

    # Training loop
    student = model.student
    iteration = start_iter
    num_gram_updates = 0
    if (
        cfg.gram.use_loss
        and model.has_gram_teacher
        and cfg.gram.rep_update
        and start_iter > 0
        and start_iter >= cfg.gram.it_first_update
    ):
        # If `start_iter == it_first_update`, we have performed one gram teacher update after
        # iteration `start_iter - 1`, except if we are starting training from scratch and `start_iter == 0`.
        num_gram_updates = math.ceil((start_iter + 1 - cfg.gram.it_first_update) / cfg.gram.update_frequency)
        logger.info(f"Gram was updated {num_gram_updates} times before iteration {start_iter}")
    consecutive_nan_count = 0
    step_start_event = torch.cuda.Event(enable_timing=True)
    step_end_event = torch.cuda.Event(enable_timing=True)
    if profiler is not None:
        profiler.start()
    if _mem_profile:
        log_phase_memory("pre_training_loop")
    for data in metric_logger.log_every(
        data_loader,
        print_freq=10,
        header="Training",
        n_iterations=max_iter,
        start_iteration=start_iter,
    ):
        it = iteration
        data["global_batch_size"] = global_batch_size
        if iteration > max_iter:
            return

        # Garbage collection (trigger manually so it happens on all ranks at the same time)
        if (iteration + 1) % 150 == 0:
            logger.info("Garbage collection")
            gc.collect()

        if cfg.gram.use_loss and model.gram_it_load_ema_teacher == it:
            logger.info(f"Loading EMA teacher into Gram teacher before iteration {it}")
            model.gram_load_ema_teacher()

        # Learning rates and other schedules
        with nvtx("schedule_update"):
            lr = lr_schedule[it]
            wd = wd_schedule[it]
            mom = momentum_schedule[it]
            teacher_temp = teacher_temp_schedule[it]
            last_layer_lr = last_layer_lr_schedule[it]
            apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # Forward backward
        step_start_event.record()
        optimizer.zero_grad(set_to_none=True)
        with nvtx("forward_backward"):
            try:
                total_loss, metrics_dict = model.forward_backward(data, teacher_temp=teacher_temp, iteration=it)
            except BaseException:
                _r = distributed.get_rank()
                logger.error("[STEPDIAG] exception in forward_backward iter=%d rank=%d", it, _r)
                _items = list(data.items()) if isinstance(data, dict) else []
                for _k, _v in _items:  # CPU metadata first (no CUDA ops)
                    if torch.is_tensor(_v):
                        logger.error("[STEPDIAG]  %s shape=%s dtype=%s dev=%s", _k, tuple(_v.shape), _v.dtype, _v.device)
                    else:
                        logger.error("[STEPDIAG]  %s = %r", _k, _v)
                for _k, _v in _items:  # CUDA stats second (guarded — may be wedged)
                    if torch.is_tensor(_v):
                        try:
                            _f = _v.detach().float()
                            logger.error("[STEPDIAG]  %s nan=%s inf=%s min=%.4g max=%.4g sum=%.4g",
                                         _k, bool(torch.isnan(_f).any()), bool(torch.isinf(_f).any()),
                                         _f.min().item(), _f.max().item(), _f.sum().item())
                        except BaseException as _e:
                            logger.error("[STEPDIAG]  %s stats unavailable: %s", _k, _e)
                raise

        # Gradient clipping
        with nvtx("grad_clip"):
            if cfg.optim.clip_grad:
                for k, v in student.items():
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        v.parameters(),
                        max_norm=cfg.optim.clip_grad,
                    )
                    metrics_dict[f"{k}_grad_norm"] = (
                        grad_norm.full_tensor().item()
                        if isinstance(grad_norm, torch.distributed.tensor.DTensor)
                        else grad_norm.item()
                    )

        # Reduce total_loss to check for NaNs, reduce metrics for logging
        with nvtx("allreduce_metrics"):
            total_loss_all_ranks = total_loss.new_empty(distributed.get_subgroup_size())
            torch.distributed.all_gather_into_tensor(
                total_loss_all_ranks,
                total_loss.detach(),
                group=distributed.get_process_subgroup(),
            )
            total_loss = total_loss_all_ranks.mean()
            metrics_values = torch.stack(
                [torch.as_tensor(v, dtype=torch.float32, device=total_loss.device).detach() for v in metrics_dict.values()]
            )
            torch.distributed.all_reduce(
                metrics_values,
                op=torch.distributed.ReduceOp.AVG,
                group=distributed.get_process_subgroup(),
            )
        metrics_dict = dict(zip(metrics_dict.keys(), metrics_values))
        if total_loss_all_ranks.isnan().any():
            consecutive_nan_count += 1
            which_ranks = total_loss_all_ranks.isnan().nonzero().flatten().tolist()
            logger.warning("NaN loss detected on ranks: %s", which_ranks)
            logger.warning("Consecutive NaNs: %d", consecutive_nan_count)
            metrics_dict_str = "\n".join([f"{k}: {v}" for k, v in metrics_dict.items()])
            logger.warning("All-reduced metrics:\n%s", metrics_dict_str)
            if consecutive_nan_count > 2 and not cfg.multidistillation.enabled:
                msg = "Too many consecutive nans detected in loss, aborting..."
                logger.error(msg)
                if profiler is not None:
                    try:
                        profiler.stop()
                    except Exception:
                        pass
                raise RuntimeError(msg)
        else:
            consecutive_nan_count = 0
        # Step optimizer
        with nvtx("optimizer_step"):
            optimizer.step()
        with nvtx("ema_update"):
            model.update_ema(mom)
        step_end_event.record()

        # Compute step time and MFU using CUDA events (GPU-synchronized timing)
        step_end_event.synchronize()
        step_time_ms = step_start_event.elapsed_time(step_end_event)
        images_per_sec = global_batch_size / (step_time_ms / 1000.0)
        mfu = compute_mfu(images_per_sec, macs_per_image, num_gpus)

        # Per-phase peak memory markers (only active when DINOV3_MEMORY_PROFILE=1)
        if _mem_profile:
            if iteration == start_iter:
                log_phase_memory("compile_warmup_iter0")
            elif iteration == start_iter + 10:
                log_phase_memory("steady_state")
            # Periodic fragmentation tracking: logs [MEMFRAG] every N iters.
            # Does NOT reset peak stats — tracks cumulative allocator fragmentation.
            # Key metric: alloc_retries + inactive_split_mb growing = fragmentation building.
            if (iteration + 1) % _mem_profile_period == 0:
                log_fragmentation_stats(iteration)

        # [GRAM] Update gram teacher when using gram teacher and frequent updates
        if (
            cfg.gram.use_loss
            and model.gram_rep_update
            and (it + 1) >= model.gram_it_first_update
            and (it + 1) % model.gram_update_frequency == 0
            and (cfg.gram.max_updates is None or num_gram_updates < cfg.gram.max_updates)
        ):
            logger.info(f"Updating Gram teacher from EMA teacher after iteration {it}")
            model.update_gram()
            num_gram_updates += 1

        # Log metrics
        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(total_loss=total_loss, **metrics_dict)
        metric_logger.update(
            mfu=mfu * 100,
            images_per_sec=images_per_sec,
            step_time_ms=step_time_ms,
        )
        if wandb_run is not None:
            wandb_metrics = {
                "iteration": float(iteration),
                "lr": float(lr),
                "wd": float(wd),
                "mom": float(mom),
                "last_layer_lr": float(last_layer_lr),
                "total_loss": float(total_loss.item()),
            }
            for key, value in metrics_dict.items():
                if isinstance(value, DTensor):
                    wandb_metrics[key] = float(value.full_tensor().item())
                elif torch.is_tensor(value):
                    wandb_metrics[key] = float(value.item())
                else:
                    wandb_metrics[key] = float(value)
            wandb_metrics["mfu_pct"] = float(mfu * 100)
            wandb_metrics["images_per_sec"] = float(images_per_sec)
            wandb_metrics["step_time_ms"] = float(step_time_ms)
            wandb_module.log(wandb_metrics, step=iteration)
```

----- END dinov3/train/train.py:427-747 -----

## 11. Train Loop Tail: Eval and Checkpoint Synchronization Points

Source: `dinov3/train/train.py:758-789`

----- BEGIN dinov3/train/train.py:758-789 -----

```python
        # Submit evaluation jobs
        if (
            cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0
            # and iteration != max_iter - 1
        ):
            if _mem_profile:
                log_phase_memory("pre_eval")
            do_test(cfg, model, f"training_{iteration}", process_group=process_subgroup)
            torch.cuda.synchronize()
            if _mem_profile:
                log_phase_memory("eval_complete")

        # Checkpointing
        if (iteration + 1) % cfg.checkpointing.period == 0:
            if _mem_profile:
                log_phase_memory("pre_checkpoint")
            torch.cuda.synchronize()
            save_checkpoint(
                ckpt_dir / str(iteration),
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                overwrite=True,
                process_group=process_subgroup,
            )
            if distributed.is_subgroup_main_process():
                keep_last_n_checkpoints(ckpt_dir, cfg.checkpointing.max_to_keep)
                if "keep_every" in cfg.checkpointing and (iteration + 1) % cfg.checkpointing.keep_every == 0:
                    keep_checkpoint_copy(ckpt_dir / str(iteration))
            if _mem_profile:
                log_phase_memory("checkpoint_complete")

```

----- END dinov3/train/train.py:758-789 -----
