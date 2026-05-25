# GPT-5 Pro DINOv3 Lateral Systems Source Excerpts

Purpose: paste this after the DINOv3 lateral-systems context pack and minimal evidence bundle. These are selected source excerpts for reasoning about multi-crop ViT execution, attention/block scheduling, iBOT/DINO losses, FSDP2 wrapping, and MFU measurement.

Note: `dinov3/train/ssl_meta_arch.py` is intentionally omitted because it was extracted separately.


----- BEGIN dinov3/models/vision_transformer.py:180-265 -----

```python
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
```

----- END dinov3/models/vision_transformer.py:180-265 -----

----- BEGIN dinov3/layers/block.py:21-210 -----

```python
class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            # No batch dimension, do not index
            return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward(self, x: Tensor, rope=None) -> Tensor:
        """
        This is the reference implementation for a single tensor, matching what is done below for a list.
        We call the list op on [x] instead of this function.
        """
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(self, x_list: List[Tensor], rope_list=None) -> List[Tensor]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(self, x_or_x_list, rope_or_rope_list=None) -> List[Tensor]:
        if isinstance(x_or_x_list, Tensor):
            # for reference:
            # return self._forward(x_or_x_list, rope=rope_or_rope_list)
            # in order to match implementations we call the list op:
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            # return [self._forward(x, rope=rope) for x, rope in zip(x_or_x_list, rope_or_rope_list)]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
```

----- END dinov3/layers/block.py:21-210 -----

----- BEGIN dinov3/layers/attention.py:43-118 -----

```python
class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])
```

----- END dinov3/layers/attention.py:43-118 -----

----- BEGIN dinov3/loss/ibot_patch_loss.py:61-142 -----

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

            self.updated = True
```

----- END dinov3/loss/ibot_patch_loss.py:61-142 -----

----- BEGIN dinov3/loss/dino_clstoken_loss.py:16-124 -----

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

            self.updated = True
```

----- END dinov3/loss/dino_clstoken_loss.py:16-124 -----

----- BEGIN dinov3/data/collate.py:11-120 -----

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
    return out


# def get_batch_subset(collated_data_batch, target_bs):
def get_batch_subset(collated_data_batch, divide_by):
    old_bs = collated_data_batch["collated_global_crops"].shape[0] // 2
    target_bs = (old_bs + divide_by - 1) // divide_by
    collated_global_crops = (
        collated_data_batch["collated_global_crops"].unflatten(0, (2, old_bs)).narrow(1, 0, target_bs).flatten(0, 1)
    )
    collated_local_crops = (
        collated_data_batch["collated_local_crops"].unflatten(0, (-1, old_bs)).narrow(1, 0, target_bs).flatten(0, 1)
    )

    masks_old_bs = collated_data_batch["collated_masks"].shape[0] // 2
    masks_target_bs = masks_old_bs // divide_by
    collated_masks = (
        collated_data_batch["collated_masks"]
        .unflatten(0, (2, masks_old_bs))
        .narrow(1, 0, masks_target_bs)
        .flatten(0, 1)
    )
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    while mask_indices_list.shape[0] == 0:
        _unbind = list(collated_data_batch["collated_masks"].unbind(0))
        random.shuffle(_unbind)
        _bind = torch.stack(_unbind, dim=0)
        collated_masks = _bind.unflatten(0, (2, masks_old_bs)).narrow(1, 0, masks_target_bs).flatten(0, 1)
        mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    upperbound = collated_data_batch["upperbound"]

    new_batch = {
        "collated_global_crops": collated_global_crops,
        "collated_local_crops": collated_local_crops,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
```

----- END dinov3/data/collate.py:11-120 -----

----- BEGIN dinov3/fsdp/ac_compile_parallelize.py:25-124 -----

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


def compile_convnext(cfg, model: nn.Module):
    compile_mode = _get_compile_mode(cfg)
    assert isinstance(model.stages, nn.ModuleList)
    # Compile at stage level
    for stage_id, stage in enumerate(model.stages):
        model.stages[stage_id] = wrap_compile_block(stage, cfg.train.cudagraphs, is_backbone_block=False, compile_mode=compile_mode)
    assert isinstance(model.downsample_layers, nn.ModuleList)
    for dsl_id, dsl in enumerate(model.downsample_layers):
        model.downsample_layers[dsl_id] = wrap_compile_block(dsl, cfg.train.cudagraphs, is_backbone_block=False, compile_mode=compile_mode)


def compile_transformer(cfg, model: nn.Module):
    compile_mode = _get_compile_mode(cfg)
    assert isinstance(model.blocks, nn.ModuleList)
    for block_id, block in enumerate(model.blocks):
        model.blocks[block_id] = wrap_compile_block(block, cfg.train.cudagraphs, is_backbone_block=True, compile_mode=compile_mode)


def fsdp_convnext(fsdp_config: Dict[str, Any], model: nn.Module, reshard_after_forward: bool = True):
    stages = model.stages
    assert isinstance(stages, nn.ModuleList)
    # FSDP wrap at stage level
    for stage_id, stage in enumerate(stages):
        stages[stage_id] = fully_shard(stage, **fsdp_config, reshard_after_forward=reshard_after_forward)
    downsample_layers = model.downsample_layers
    assert isinstance(downsample_layers, nn.ModuleList)
    for dsl_id, dsl in enumerate(downsample_layers):
        downsample_layers[dsl_id] = fully_shard(dsl, **fsdp_config, reshard_after_forward=reshard_after_forward)
    dsl: FSDPState
    stage: FSDPState
    for dsl, stage in zip(downsample_layers, stages):
        dsl.set_modules_to_forward_prefetch([stage])
        stage.set_modules_to_backward_prefetch([dsl])
    fully_shard(model, **fsdp_config, reshard_after_forward=reshard_after_forward)
    register_fsdp_forward_method(model, "get_intermediate_layers")


def fsdp_transformer(fsdp_config: Dict[str, Any], model: nn.Module, reshard_after_forward: bool = True):
    # Backbone - FSDP every block
```

----- END dinov3/fsdp/ac_compile_parallelize.py:25-124 -----

----- BEGIN dinov3/utils/mfu.py:20-139 -----

```python
def vit_forward_flops(
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    ffn_ratio: float = 4.0,
) -> int:
    """MACs for one forward pass of a ViT on one image (bidirectional attention).

    Uses MAC convention: 1 MAC = 1 multiply-add (fvcore / DINOv2 paper convention).
    The DINOv2 paper reports ~17.4 GFLOPs for ViT-B/16 global crop using this convention.
    Hardware TFLOPS specs count each MAC as 2 FLOPs (1 multiply + 1 add); see
    compute_mfu() for the 2× conversion factor. Use H100_BF16_TFLOPS (989, dense)
    as the denominator — not NVIDIA's published 1979 which assumes 2:4 sparsity.

    Per-layer breakdown (all in MACs):
      QKV projections: 3 × seq_len × D²   (three [D→D] projections)
      O projection:    1 × seq_len × D²
      Attn QKᵀ + AV:  2 × seq_len² × D   (bidirectional, full square)
      FFN up + down:   2 × seq_len × D × ffn_dim
    """
    ffn_dim = int(hidden_dim * ffn_ratio)
    # Linear: Q, K, V, O projections each [seq_len, D] × [D, D] → seq_len × D² MACs each.
    #   QKV = 3 × seq_len × D², O = 1 × seq_len × D²  →  4 × seq_len × D²
    # Attention: QKᵀ and AV each need seq_len × seq_len × D MACs (per-head × num_heads).
    #   Combined: 2 × seq_len² × D
    # FFN: up [D→ffn_dim] + down [ffn_dim→D] each seq_len × D × ffn_dim MACs → 2× total.
    attn_linear = 4 * seq_len * hidden_dim * hidden_dim   # QKV + O projections
    attn_scores = 2 * seq_len * seq_len * hidden_dim       # QKᵀ + AV (bidirectional, full sq)
    ffn = 2 * seq_len * hidden_dim * ffn_dim               # FFN up + down / MLP layer
    return num_layers * (attn_linear + attn_scores + ffn)


def compute_dino_flops_per_image(
    global_crop_size: int = 224,
    local_crop_size: int = 96,
    patch_size: int = 16,
    n_global_crops: int = 2,
    n_local_crops: int = 8,
    hidden_dim: int = 768,
    num_layers: int = 12,
    ffn_ratio: float = 4.0,
    n_registers: int = 0,
    gram_enabled: bool = False,
    head_overhead_pct: float = 0.05,
) -> int:
    """Total MACs for one full training step per original image in the batch.

    Uses MAC convention (1 MAC = 1 multiply-add). To convert to hardware FLOPs
    for MFU computation, multiply by 2 — see compute_mfu().

    Accounts for:
      - Student forward (global + local crops) + backward (~2× forward)
      - Teacher forward (global crops only, no grad)
      - Optional gram teacher forward (global crops only, no grad)
      - Head overhead (~5% of backbone by default)

    Args:
        global_crop_size: Global crop image size in pixels (default 224).
        local_crop_size: Local crop image size in pixels (default 96).
        patch_size: ViT patch size (default 16).
        n_global_crops: Number of global crops per image (default 2).
        n_local_crops: Number of local crops per image (default 8).
        hidden_dim: ViT hidden dimension (default 768 for ViT-B).
        num_layers: Number of transformer layers (default 12 for ViT-B).
        ffn_ratio: FFN hidden dim ratio (default 4.0 for standard MLP, not SwiGLU).
        n_registers: Number of register tokens (default 0; satellite fork drops them).
        gram_enabled: Whether gram teacher is enabled (default False).
        head_overhead_pct: Fraction added for DINO/iBOT heads (default 0.05 = 5%). (overhead estimate)

    Returns:
        Total MACs per image as an integer.
    """
    # fitting patch sizes into X crop sizes to determine # of tokens
    # + 1 for the class token (CLS) token
    # + n_registers for the register tokens (usually 0 in default config)
    global_seq = (global_crop_size // patch_size) ** 2 + 1 + n_registers
    local_seq = (local_crop_size // patch_size) ** 2 + 1 + n_registers

    global_fwd = vit_forward_flops(global_seq, hidden_dim, num_layers, ffn_ratio)
    local_fwd = vit_forward_flops(local_seq, hidden_dim, num_layers, ffn_ratio)

    student_fwd = n_global_crops * global_fwd + n_local_crops * local_fwd
    student_bwd = 2 * student_fwd          # backward ≈ 2× forward
    teacher_fwd = n_global_crops * global_fwd
    gram_fwd = n_global_crops * global_fwd if gram_enabled else 0

    backbone_flops = student_fwd + student_bwd + teacher_fwd + gram_fwd
    # NOTE: + head_overhead_pct for the DINO/iBOT loss functions (default 0.05 = 5%)
    return int(backbone_flops * (1.0 + head_overhead_pct)) # (returning MACs, not FLOPs technically)


def compute_mfu(
    images_per_sec: float,
    macs_per_image: int,
    num_gpus: int,
    peak_tflops: float = H100_BF16_TFLOPS,
) -> float:
    """Compute Model FLOP Utilization (MFU) as a fraction 0.0–1.0.

    MFU = actual_hardware_flops_per_sec / peak_hardware_flops_per_sec

    The 2× factor converts MACs (from compute_dino_flops_per_image) to hardware FLOPs:
    hardware vendors count each multiply-add as 2 FLOPs (1 multiply + 1 add).
    H100_BF16_TFLOPS = 989.0 is the dense (no 2:4 sparsity) peak — NVIDIA's published
    1979 TFLOPS assumes structured sparsity that standard dense matmuls do not use.

    Args:
        images_per_sec: Total images processed per second across all GPUs.
        macs_per_image: MACs per image per step (from compute_dino_flops_per_image).
        num_gpus: Number of GPUs in the run.
        peak_tflops: Peak TFLOPS per GPU (default H100 BF16 dense = 989.0).

    Returns:
        MFU as a fraction (multiply by 100 to get percentage).
    """
    # 2× converts MACs → hardware FLOPs (1 MAC = 2 hardware FLOPs: 1 multiply + 1 add)
    actual_tflops = (images_per_sec * 2 * macs_per_image) / 1e12
    theoretical_peak = num_gpus * peak_tflops
    return actual_tflops / theoretical_peak
```

----- END dinov3/utils/mfu.py:20-139 -----
