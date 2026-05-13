# Architecture And MFU Teaching Guide For This Repo

This file is meant to be taught from.

It explains the model we train in this repository, the training loop around it, and the exact MFU measurement implementation we use here. The source of truth is the code in this repo, not generic DINOv3 descriptions.

Main code paths:

- `dinov3/configs/ssl_default_config.yaml`
- `dinov3/models/vision_transformer.py`
- `dinov3/layers/block.py`
- `dinov3/layers/attention.py`
- `dinov3/train/ssl_meta_arch.py`
- `dinov3/train/train.py`
- `dinov3/utils/mfu.py`
- `tests/test_mfu.py`

## Short version

We are training a Vision Transformer (ViT) on 5-channel satellite images, not on normal 3-channel RGB images. The core model is a ViT-B/16: embedding size 768, depth 12, attention heads 12, patch size 16, and no register/storage tokens by default (`dinov3/configs/ssl_default_config.yaml:88-115`, `dinov3/models/vision_transformer.py:344-353`).

The training setup uses a student network and a teacher network with the same backbone architecture. The student is optimized by gradient descent. The teacher is not optimized directly; instead it is updated as an exponential moving average (EMA) of the student after every step (`dinov3/train/ssl_meta_arch.py:744-757`).

The student sees 2 global crops and 8 local crops of each image. The teacher sees only the 2 global crops. Training combines:

- DINO loss on CLS tokens
- iBOT loss on masked patch tokens
- KoLeo regularization on global CLS tokens
- optional Gram anchoring on patch-feature relationships

The MFU implementation does two things:

1. It estimates MACs per image from the architecture and crop counts (`dinov3/utils/mfu.py:20-109`).
2. It measures step time with CUDA events inside the training loop, converts throughput to TFLOPS, and divides by H100 dense BF16 peak, 989 TFLOPS/GPU (`dinov3/train/train.py:472-491`, `dinov3/train/train.py:567-670`, `dinov3/utils/mfu.py:112-137`).

## 1. What is actually being trained here

There are two separate questions:

1. What is the model architecture?
2. How do we currently launch training?

Those are related, but they are not the same thing.

### 1.1 Architecture defaults

From `ssl_default_config.yaml`, the default model is:

| Item | Value | Code reference |
|---|---|---|
| Backbone | `vit_base` | `dinov3/configs/ssl_default_config.yaml:88-90` |
| Patch size | `16` | `dinov3/configs/ssl_default_config.yaml:89-90` |
| Input channels | `5` | `dinov3/configs/ssl_default_config.yaml:105`, `:123` |
| FFN type | `mlp` | `dinov3/configs/ssl_default_config.yaml:94-95` |
| FFN ratio | `4.0` | `dinov3/configs/ssl_default_config.yaml:95` |
| Register/storage tokens | `0` | `dinov3/configs/ssl_default_config.yaml:101` |
| Global crops | `2` crops of size `224` | `dinov3/configs/ssl_default_config.yaml:154-163` |
| Local crops | `8` crops of size `96` | `dinov3/configs/ssl_default_config.yaml:158-163` |
| Gram loss | disabled by default | `dinov3/configs/ssl_default_config.yaml:45-63` |

`vit_base` resolves to:

- embed dim `768`
- depth `12`
- heads `12`
- FFN ratio `4`

Source: `dinov3/models/vision_transformer.py:344-353`.

### 1.2 Current production launch

The launch script we currently use in this repo is `run.sh`, and it overrides some system/training settings:

- `train.distributed_strategy=ddp`
- `train.batch_size_per_gpu=256`
- `train.num_workers=20`
- `train.prefetch_factor=8`
- `train.sharded_eval_checkpoint=true`

Source: `run.sh:41-69`.

Important distinction:

- These overrides change how training is run.
- They do not change the basic backbone architecture unless an override explicitly changes an architecture field.

In `run.sh`, the backbone is still `vit_base` with 5 input channels (`run.sh:44-47`).

## 2. Jargon map

This section is here because most confusion comes from vocabulary before it comes from math.

| Term | Plain meaning | What it means in this repo | Code reference |
|---|---|---|---|
| backbone | The main feature extractor, the “body” of the model | `DinoVisionTransformer` | `dinov3/models/vision_transformer.py:59-330` |
| head | A small module on top of the backbone that converts features into training targets | `DINOHead` for DINO and iBOT | `dinov3/train/ssl_meta_arch.py:62-129`, `dinov3/layers/dino_head.py:11-50` |
| patch | A small square cut from the image | Here the model cuts the image into `16x16` tiles | `dinov3/layers/patch_embed.py:21-76` |
| patch size | The width and height of each tile | `16` pixels by default | `dinov3/configs/ssl_default_config.yaml:89-90` |
| token | One vector the transformer processes | CLS token, optional storage tokens, and patch tokens | `dinov3/models/vision_transformer.py:190-220` |
| CLS token | A special learned summary token | Prepended before patch tokens; later used for DINO and KoLeo | `dinov3/models/vision_transformer.py:112`, `:211-218`, `:253-256` |
| register / storage token | Extra learned tokens that act like scratchpad memory | Supported as `n_storage_tokens`, but default is `0` here | `dinov3/models/vision_transformer.py:113-116`, `dinov3/configs/ssl_default_config.yaml:101` |
| block | One repeated transformer unit | norm -> attention -> residual -> norm -> FFN -> residual | `dinov3/layers/block.py:21-124` |
| attention head | One subspace in multi-head attention | `12` heads in ViT-B, so each head sees `768 / 12 = 64` dims | `dinov3/models/vision_transformer.py:74-77`, `dinov3/layers/attention.py:56-58` |
| Q, K, V | Query, key, and value vectors used in attention | Produced by one linear layer and reshaped per head | `dinov3/layers/attention.py:61`, `:106-118` |
| FFN | Feed-forward network, the per-token MLP after attention | Here the default FFN is a standard MLP, not SwiGLU | `dinov3/models/vision_transformer.py:137-155`, `dinov3/configs/ssl_default_config.yaml:94-95` |
| MLP | The ordinary 2-layer FFN used in many transformers | `fc1 -> GELU -> fc2` | `dinov3/layers/ffn_layers.py:24-49` |
| SwiGLU | A gated FFN variant | Supported in code, but not used by default here | `dinov3/models/vision_transformer.py:19-25`, `dinov3/layers/ffn_layers.py:52-77` |
| RoPE | Rotary positional embedding | Encodes 2D position by rotating Q and K using image coordinates | `dinov3/layers/rope_position_encoding.py:14-121`, `dinov3/layers/attention.py:66-85` |
| prototypes | Large output dictionary/bins used by self-supervised heads | `65536` for both DINO and iBOT heads by default | `dinov3/configs/ssl_default_config.yaml:13`, `:40` |
| EMA teacher | Teacher weights are a moving average of student weights | Updated after every optimizer step | `dinov3/train/ssl_meta_arch.py:744-757` |
| Gram matrix | A matrix of feature-to-feature similarities | Used for optional Gram anchoring | `dinov3/loss/gram_loss.py:34-84` |

Two especially important clarifications:

- In this repo, `backbone` does not mean “the whole training system.” It means the ViT that turns images into features.
- In this repo, `head` does not mean attention head. Those are two completely different uses of the word “head.”

## 3. From a satellite image to tokens

The easiest way to understand the backbone is to follow one image through it.

### 3.1 Patch embedding

The input image arrives as a tensor of shape:

```text
[B, C, H, W]
```

For this repo, `C = 5` because the satellite pipeline normalizes different sensors into five channels (`dinov3/configs/ssl_default_config.yaml:105`, `dinov3/data/datasets/channel_utils.py:15-26`).

The patch embedding layer is a `Conv2d` whose kernel size and stride are both equal to the patch size (`dinov3/layers/patch_embed.py:61`).

That means it is doing:

```text
image -> non-overlapping 16x16 tiles -> one 768-d vector per tile
```

For a global crop of `224x224`:

- `224 / 16 = 14` patches per side
- `14 * 14 = 196` patch tokens
- plus `1` CLS token
- plus `0` storage/register tokens in this repo

So the global sequence length is:

```text
196 + 1 + 0 = 197 tokens
```

For a local crop of `96x96`:

- `96 / 16 = 6`
- `6 * 6 = 36` patch tokens
- plus `1` CLS token

So the local sequence length is:

```text
36 + 1 = 37 tokens
```

These exact token counts show up later in the MFU math.

### 3.2 Token preparation in code

Token preparation happens in `prepare_tokens_with_masks()` (`dinov3/models/vision_transformer.py:190-220`):

1. `PatchEmbed` converts image pixels into patch embeddings.
2. The patch grid is flattened from `[H_patches, W_patches, D]` to `[N_patches, D]`.
3. A learned CLS token is prepended.
4. Optional storage tokens would be added here, but default is zero.
5. If iBOT masking is active, masked patch positions are replaced with a learned mask token.

Visually:

```text
[5, 224, 224] image
    ->
PatchEmbed(kernel=16, stride=16)
    ->
[14, 14, 768] patch grid
    ->
[196, 768] patch tokens
    ->
prepend [CLS]
    ->
[197, 768] token sequence
```

Good analogy:

- Patch embedding is like cutting a satellite tile into a grid of square tiles.
- Each tile gets translated from raw pixels into a learned descriptor vector.
- The transformer no longer works on pixels directly. It works on those descriptor vectors.

## 4. What happens inside one transformer block

The backbone is a stack of 12 `SelfAttentionBlock`s (`dinov3/models/vision_transformer.py:140-160`).

One block is:

```text
x
 -> norm
 -> self-attention
 -> add residual
 -> norm
 -> FFN
 -> add residual
```

Source: `dinov3/layers/block.py:43-66`, `:120-124`.

### 4.1 Attention

Inside attention, the token sequence is linearly projected into Q, K, and V (`dinov3/layers/attention.py:61`, `:88-90`).

Then it is reshaped into 12 heads:

```text
[B, N, 768]
    ->
[B, N, 3, 12, 64]
    ->
q, k, v each become [B, 12, N, 64]
```

Source: `dinov3/layers/attention.py:111-118`.

Plain-English meaning:

- Each attention head is one separate “way of comparing tokens.”
- Head 1 may learn one kind of spatial or semantic relation.
- Head 2 may learn another.
- The model then combines all 12 heads back into one 768-d representation.

Analogy:

- Imagine 12 analysts looking at the same map.
- One focuses on texture.
- One focuses on geometry.
- One focuses on boundaries.
- One focuses on repeated patterns.
- At the end, their notes are merged.

### 4.2 RoPE

This repo uses RoPE, rotary positional embeddings (`dinov3/models/vision_transformer.py:124-136`, `dinov3/layers/rope_position_encoding.py:16-121`).

RoPE does not append a learned position vector to each token. Instead, it rotates Q and K vectors based on the patch coordinates before attention is computed (`dinov3/layers/attention.py:66-85`, `:114-116`).

Why that matters:

- Without position information, two identical patches in different locations would look identical to attention.
- RoPE injects “where” information into the comparison itself.

Analogy:

- Think of RoPE as attaching a directional compass angle to each patch before asking “which other patches are related to me?”
- The content still matters, but now location matters too.

This repo uses `pos_embed_rope_normalize_coords: separate`, which normalizes x and y coordinates independently (`dinov3/configs/ssl_default_config.yaml:106-114`, `dinov3/layers/rope_position_encoding.py:71-78`).

### 4.3 FFN / MLP / SwiGLU

After attention, each token goes through an FFN. In this repo, the default FFN is a plain MLP:

```text
Linear -> GELU -> Linear
```

Source: `dinov3/layers/ffn_layers.py:24-49`.

The hidden size is:

```text
ffn_hidden_dim = 768 * 4 = 3072
```

Source: `dinov3/layers/block.py:57-65`.

SwiGLU is another FFN option supported by the code (`dinov3/layers/ffn_layers.py:52-77`), but it is not the default model we train here (`dinov3/configs/ssl_default_config.yaml:94-95`).

Important distinction:

- `MLP` is the actual default here.
- `SwiGLU` is available, but not active unless you change `student.ffn_layer`.

## 5. Student, teacher, and why there are multiple crops

The core training logic is in `SSLMetaArch.forward_backward()` (`dinov3/train/ssl_meta_arch.py:379-460`).

The structure is:

```text
input batch
  -> move crops and masks to GPU
  -> teacher forward on 2 global crops
  -> student forward on 2 global + 8 local crops
  -> optional Gram-teacher forward
  -> compute losses
  -> backward
```

### 5.1 Why multiple crops?

For each original image, the augmentation pipeline creates:

- 2 large views of the image: global crops
- 8 small views of the image: local crops

Source: `dinov3/configs/ssl_default_config.yaml:154-163`.

Idea:

- The teacher sees the larger, richer context.
- The student must learn representations that stay consistent across different views and scales.

### 5.2 Teacher forward

Teacher forward is in `get_teacher_output()` (`dinov3/train/ssl_meta_arch.py:462-505`).

The teacher:

- runs only on the global crops
- extracts CLS tokens and patch tokens
- applies the DINO head to CLS tokens
- applies the iBOT head to the masked patch tokens
- turns teacher outputs into probability targets using Sinkhorn-Knopp

### 5.3 Student forward

Student forward is in `get_student_output()` (`dinov3/train/ssl_meta_arch.py:561-613`).

The student:

- runs on both global and local crops
- applies masking on global crops for iBOT
- computes CLS-token outputs for DINO
- computes masked patch outputs for iBOT

One subtle but important implementation detail:

- The student backbone is called once with a list containing global crops and local crops (`dinov3/train/ssl_meta_arch.py:567-572`).
- The backbone processes both resolutions through the same block stack using `forward_features_list()` (`dinov3/models/vision_transformer.py:222-261`).
- Some elementwise work is shared across the list, but attention still has to respect the different sequence lengths (`dinov3/layers/block.py:126-198`, `dinov3/layers/attention.py:94-104`).

That means:

- the code is trying to be efficient
- but global and local attention are still fundamentally different workloads because `197 != 37`

## 6. What each loss is teaching the model

Loss wiring is in `compute_losses()` (`dinov3/train/ssl_meta_arch.py:615-715`).

### 6.1 DINO loss

DINO uses the CLS token. The teacher produces soft targets, and the student tries to match them (`dinov3/train/ssl_meta_arch.py:641-664`, `dinov3/loss/dino_clstoken_loss.py:72-99`).

There are two pieces:

- local DINO loss: student local CLS vs teacher global CLS
- global DINO loss: student global CLS vs teacher global CLS

Intuition:

- The model should recognize that different views of the same image still belong to the same underlying scene.

### 6.2 iBOT loss

iBOT uses masked patch tokens (`dinov3/train/ssl_meta_arch.py:671-680`).

Intuition:

- Hide some student patches.
- Ask the student to predict what the teacher believes those hidden patches should look like in feature space.

So:

- DINO mostly teaches image-level agreement through CLS tokens.
- iBOT mostly teaches patch-level understanding through masked patches.

### 6.3 KoLeo loss

KoLeo regularizes the student global CLS tokens (`dinov3/train/ssl_meta_arch.py:666-669`, `dinov3/loss/koleo_loss.py:14-43`).

It pushes representations to spread out instead of collapsing into the same point.

Analogy:

- If every image ended up with almost the same embedding, the model would be useless.
- KoLeo pushes embeddings to occupy space more evenly.

### 6.4 EMA teacher update

After the optimizer updates the student, the teacher is updated by EMA:

```text
teacher = m * teacher + (1 - m) * student
```

Source: `dinov3/train/ssl_meta_arch.py:744-757`.

This makes the teacher a smoother, slower-moving target than the student.

Analogy:

- The student is a person learning quickly and making noisy updates.
- The teacher is a long-exposure average of the student’s recent states.

## 7. Gram anchoring

This is the part many people do not understand at first, because it is not “match features directly.” It is “match relationships between features.”

### 7.1 What it is

The Gram loss implementation is in `dinov3/loss/gram_loss.py:11-84`.

It does this:

1. Take student patch features.
2. Take teacher patch features.
3. Optionally L2-normalize them.
4. Build a similarity matrix for each side using:

```text
similarity = features @ features^T
```

5. Minimize the MSE between the student similarity matrix and the teacher similarity matrix.

In code:

- teacher similarity: `target_feats @ target_feats.transpose(-1, -2)` (`dinov3/loss/gram_loss.py:61-63`)
- student similarity: `output_feats @ output_feats.transpose(-1, -2)` (`dinov3/loss/gram_loss.py:72-73`)
- loss: `MSE(student_sim, target_sim)` (`dinov3/loss/gram_loss.py:24`, `:84`)

### 7.2 Why that is called “anchoring”

It anchors the student to the teacher’s internal geometry.

Not:

```text
student patch 17 must equal teacher patch 17 exactly
```

But:

```text
if the teacher thinks patch A and patch B are similar,
the student should also think patch A and patch B are similar
```

And similarly:

```text
if the teacher thinks patch A and patch C are not similar,
the student should preserve that relation too
```

So Gram anchoring is about preserving structure in feature space.

Analogy:

- Suppose three cities form a triangle on a map.
- Gram anchoring cares more about preserving the triangle shape than about forcing every city to have the exact same coordinates in two different maps.

### 7.3 Where the teacher patches come from

That logic is in `get_gram_teacher_output()` (`dinov3/train/ssl_meta_arch.py:507-559`).

There are two modes:

- `gram.ema_teacher=true`: use the normal EMA teacher’s patch features
- otherwise: use a separate Gram teacher backbone

If a separate Gram teacher is used and its crop resolution differs from the student’s, the teacher patch map can be resized to the student patch grid (`dinov3/train/ssl_meta_arch.py:524-541`).

That means Gram anchoring is flexible:

- same teacher as DINO/iBOT
- or a separate teacher
- same patch grid
- or resized patch grid

### 7.4 Which patches are included

The config can choose:

- `all`
- `masked`
- `unmasked`

Source: `dinov3/configs/ssl_default_config.yaml:60`, `dinov3/train/ssl_meta_arch.py:543-551`.

Default note:

- In the default YAML, `gram.use_loss=false`, so Gram anchoring is available but off by default (`dinov3/configs/ssl_default_config.yaml:45-46`).

### 7.5 One subtle point about `img_level`

`GramLoss.forward()` can operate:

- per image, if `img_level=True`
- across the flattened batch of selected patches, if `img_level=False`

Source: `dinov3/loss/gram_loss.py:34-70`.

In the default config, `gram.img_level=false` (`dinov3/configs/ssl_default_config.yaml:57`), so if Gram is enabled without further changes, it compares relationships over the flattened patch set rather than image-by-image.

## 8. The training loop, in plain English

The outer training loop is in `do_train()` (`dinov3/train/train.py:434-670`).

Per iteration, the important order is:

1. Update learning-rate and other schedules.
2. Record CUDA start event.
3. `optimizer.zero_grad(set_to_none=True)`.
4. Run `model.forward_backward(...)`.
5. Clip gradients.
6. All-reduce metrics.
7. `optimizer.step()`.
8. Update EMA teacher.
9. Record CUDA end event.
10. Compute `step_time_ms`, `images_per_sec`, and `mfu`.

Source: `dinov3/train/train.py:594-670`.

That ordering matters for MFU:

- the timed region is the actual training step on GPU
- it starts right before `zero_grad`
- it ends right after the EMA update

So the timing includes:

- forward
- backward
- optimizer step
- EMA update

It does not directly include:

- Python time before the event starts
- data loading while the next batch is still being prepared
- periodic eval/checkpoint syncs outside the measured step

## 9. MFU: what it means here

MFU stands for Model FLOP Utilization.

Plain-English meaning:

- “Given how much math this model step should do, and how fast the step actually ran, how much of the GPU’s peak compute rate are we using?”

It is not a measure of model quality.
It is a measure of hardware utilization.

### 9.1 The two ingredients

This repo computes MFU from:

1. `macs_per_image`: estimated once from architecture and crop counts
2. `images_per_sec`: measured from CUDA event timing

Then:

```text
actual_hardware_flops_per_sec = images_per_sec * 2 * macs_per_image
mfu = actual_hardware_flops_per_sec / (num_gpus * peak_tflops_per_gpu)
```

Source: `dinov3/utils/mfu.py:112-137`.

The `2 *` is because:

- the FLOP estimator uses MAC convention
- `1 MAC = 1 multiply-add`
- hardware TFLOPS specs count multiply and add separately
- so `1 MAC = 2 hardware FLOPs`

Source: `dinov3/utils/mfu.py:26-33`, `:118-136`.

### 9.2 Why the denominator is 989 TFLOPS, not 1979

The code uses:

```text
H100_BF16_TFLOPS = 989.0
```

Source: `dinov3/utils/mfu.py:9`.

That is the dense BF16 peak. NVIDIA’s larger number assumes structured sparsity. Standard dense transformer training should use the dense number, so `989` is the denominator this repo intentionally uses (`dinov3/utils/mfu.py:1-14`).

## 10. How `macs_per_image` is estimated

### 10.1 Per-image ViT forward MACs

`vit_forward_flops()` estimates MACs for one forward pass of a ViT on one image (`dinov3/utils/mfu.py:20-52`).

Per layer, it counts:

- QKV and output projections
- attention score and value mixing terms
- FFN up and down projections

Formula:

```text
attn_linear = 4 * seq_len * D * D
attn_scores = 2 * seq_len * seq_len * D
ffn         = 2 * seq_len * D * (D * ffn_ratio)
layer_total = attn_linear + attn_scores + ffn
total       = num_layers * layer_total
```

Where:

- `seq_len` is number of tokens
- `D` is hidden size, here `768`

### 10.2 Step-level MACs per original image

`compute_dino_flops_per_image()` then builds the full training-step estimate (`dinov3/utils/mfu.py:55-109`).

For this repo’s default ViT-B/16 settings:

```text
global_seq = (224 / 16)^2 + 1 + 0 = 197
local_seq  = (96  / 16)^2 + 1 + 0 = 37
```

Then:

```text
student_fwd = 2 * global_fwd + 8 * local_fwd
student_bwd = 2 * student_fwd
teacher_fwd = 2 * global_fwd
gram_fwd    = 2 * global_fwd if Gram is enabled else 0
backbone    = student_fwd + student_bwd + teacher_fwd + gram_fwd
total_macs  = backbone * 1.05
```

The final `1.05` is the approximate 5% head overhead for DINO/iBOT heads (`dinov3/utils/mfu.py:64-71`, `:100-109`).

Important caveat:

- This is an engineering estimate, not an exact kernel-by-kernel hardware trace.
- It is good enough for MFU tracking and comparisons across runs.

### 10.3 Why patch size and crop counts matter so much

Now architecture and MFU connect directly:

- smaller patch size -> more patches -> longer sequence -> more compute
- more local crops -> much more student compute
- register/storage tokens -> larger sequence -> more compute
- Gram enabled -> one extra teacher-style forward on 2 global crops

This is why architecture terminology is not just vocabulary. It changes the FLOP denominator.

## 11. How step time is measured in code

Inside `do_train()`, the code creates CUDA events once:

- `step_start_event`
- `step_end_event`

Source: `dinov3/train/train.py:567-568`.

Then each iteration does:

```text
step_start_event.record()
optimizer.zero_grad(...)
model.forward_backward(...)
optimizer.step()
model.update_ema(...)
step_end_event.record()
step_end_event.synchronize()
step_time_ms = step_start_event.elapsed_time(step_end_event)
```

Source: `dinov3/train/train.py:604-669`.

Then:

```text
images_per_sec = global_batch_size / (step_time_ms / 1000.0)
mfu = compute_mfu(images_per_sec, macs_per_image, num_gpus)
```

Source: `dinov3/train/train.py:669-670`.

This means the MFU number is tied to the actual configured global batch size and measured step duration.

## 12. What the tests are checking

The MFU unit tests are useful because they encode the intended interpretation of the formula (`tests/test_mfu.py:1-139`).

Key tests:

- global ViT-B/16 forward should be about `17.5` GMACs (`tests/test_mfu.py:8-13`)
- local crop should be much cheaper than global crop (`tests/test_mfu.py:14-18`)
- FLOPs should grow faster than linearly with sequence length because attention has a quadratic term (`tests/test_mfu.py:20-31`)
- enabling Gram should add the cost of another global teacher-style forward (`tests/test_mfu.py:56-65`)
- MFU should scale linearly with throughput and inversely with GPU count (`tests/test_mfu.py:97-107`)

These tests are a good way to separate “the formula we meant” from “the formula we accidentally typed.”

## 13. A concrete mental model you can keep in your head

If you only want one mental picture, use this one:

```text
original satellite image
    ->
augment into 2 big views + 8 small views
    ->
student ViT processes all of them
teacher ViT processes only the 2 big views
    ->
student tries to match teacher on:
  - image summaries (DINO / CLS)
  - masked patches (iBOT)
  - representation spread (KoLeo)
  - optional patch-relationship geometry (Gram)
    ->
after the optimizer step, the teacher becomes a slow EMA copy of the student
```

And for MFU:

```text
how much math should one image cost?
    +
how many images per second did we actually process?
    =
what fraction of H100 dense BF16 peak did we use?
```

## 14. Common misunderstandings

### “Backbone” means the whole training system

No. Here it means the ViT feature extractor only. The DINO and iBOT heads are separate modules on top (`dinov3/train/ssl_meta_arch.py:62-129`).

### “Head” always means attention head

No. There are:

- attention heads inside self-attention
- model heads like `dino_head` and `ibot_head`

Different concepts, same word.

### “RoPE” means adding position embeddings to tokens

Not in the usual additive sense. Here RoPE rotates Q and K based on coordinates before attention (`dinov3/layers/attention.py:66-85`).

### “Gram anchoring” means patch vectors are copied exactly from the teacher

No. It matches similarity structure, not raw vectors (`dinov3/loss/gram_loss.py:61-84`).

### “SwiGLU” is part of the default model here

No. The default config uses `mlp`, not `swiglu` (`dinov3/configs/ssl_default_config.yaml:94-95`).

### “Registers” are an important part of the default satellite model

No. This repo’s default is `n_storage_tokens=0`, so the standard default training path has no extra register/storage tokens (`dinov3/configs/ssl_default_config.yaml:101`).

## 15. Suggested reading order in code

If you want to read the code in the least confusing order, use this sequence:

1. `dinov3/configs/ssl_default_config.yaml`
2. `dinov3/models/vision_transformer.py`
3. `dinov3/layers/patch_embed.py`
4. `dinov3/layers/attention.py`
5. `dinov3/layers/block.py`
6. `dinov3/train/ssl_meta_arch.py`
7. `dinov3/train/train.py`
8. `dinov3/utils/mfu.py`
9. `tests/test_mfu.py`

That order goes from “what model do we mean?” to “how do we train it?” to “how do we measure its compute use?”

## Conclusion

The cleanest way to think about this repo is:

- backbone = the ViT that turns 5-channel satellite crops into token features
- heads = small output modules that turn those features into self-supervised targets
- teacher = a slow EMA copy of the student
- DINO = align image-level summaries
- iBOT = align masked patch predictions
- KoLeo = stop collapse by spreading embeddings out
- Gram anchoring = preserve the teacher’s patch-to-patch relationship geometry
- MFU = estimated math per image times measured images/sec, divided by H100 dense peak

If you can explain those seven bullets in your own words, you already understand most of the architecture and most of the MFU implementation in this codebase.
