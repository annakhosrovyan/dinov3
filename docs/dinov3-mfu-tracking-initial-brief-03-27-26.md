# DINOv3 MFU Tracking — Context Brief for Claude Code

**Goal**: Add MFU tracking to the `rs_foundation_models` DINOv3 training run on 8x H100 GPUs.  
**Repo**: Meta DINOv3 fork (satellite imagery specialization), 5-channel ViT-B  
**Target**: Establish baseline MFU, then drive toward 30%+

---

> **ERRATA (2026-03-31)** — Three errors in this brief were found after implementation:
>
> 1. **Section 3 FLOP formula**: The `vit_b_forward_flops` code snippet uses an inconsistent
>    mixed convention — `attn_linear = 8 × seq × D²` and `ffn = 4 × seq × D × ffn_dim` treat
>    each multiply-add as 2 hardware FLOPs, but `attn_scores = 2 × seq² × D` uses the MAC
>    convention (1 multiply-add = 1 op). The correct consistent approach (MAC convention,
>    matching fvcore / DINOv2 paper) is:
>    `attn_linear = 4 × seq × D²`, `attn_scores = 2 × seq² × D`, `ffn = 2 × seq × D × ffn_dim`
>    → yields ~17.4 GMACs/image for global crop ✓. The mixed formula yielded ~34 GFLOPs, which
>    is wrong. The actual implementation in `dinov3/utils/mfu.py` uses the correct MAC convention.
>
> 2. **Section 4 compute_mfu**: The formula `actual_tflops = (images_per_sec × flops) / 1e12`
>    is missing a **2× factor**. Each MAC = 2 hardware FLOPs (1 multiply + 1 add), but
>    `flops_per_image` returns MACs. The corrected formula is:
>    `actual_tflops = (images_per_sec × 2 × macs_per_image) / 1e12`
>    Without this factor, reported MFU is half the true value.
>
> 3. **Section 9 baseline estimates**: The 10–14% MFU estimate assumed the old (uncorrected)
>    formula. Actual measured baseline at 8×H100 with compile is **~11.3% hardware MFU (dense)**.
>    See `docs/mfu-results-2026-03-30.md` for actual numbers.
>
> 4. **H100 denominator** (2026-04-01): `H100_BF16_TFLOPS = 1979.0` throughout this brief is wrong.
>    NVIDIA's 1979 TFLOPS assumes 2:4 structured sparsity. Dense BF16 (standard transformer
>    training) is **989 TFLOPS**. Similarly `A100_BF16_TFLOPS = 312.0` should be `156.0`.
>    The roofline table (Section 8) and ridge point formula are also affected — see inline notes.
>
> Everything else in this brief (architecture, token counts, forward pass structure, FSDP2
> setup, key code paths) is correct and was verified against the implementation.

---

---

## 1. Why PaLM's 6N Formula Doesn't Apply Here

The standard `MFU = tokens/sec × 6N / (num_gpus × peak_TFLOPS)` is for decoder-only causal LMs.  
DINOv3 is fundamentally different:

| Property | Causal LM | DINOv3 |
|---|---|---|
| Attention | Causal (masked) | Bidirectional |
| Passes per step | 1 forward + 1 backward | 3+ forward passes (student, teacher, optional gram teacher) + 1 backward |
| Batch structure | Uniform tokens | Multi-crop: 2 global (224px) + K local (96px) |
| Loss target | Next token | Patch/CLS representation distillation |
| Throughput unit | tokens/sec | images/sec (per original image, not per crop) |

**MFU formula for DINOv3:**
```
MFU = (images_per_sec × flops_per_image_step) / (num_gpus × peak_TFLOPS)
```

`flops_per_image_step` must be derived from scratch — see Section 3.

---

## 2. Architecture Details (from repo code)

### ViT-B Config (vision_transformer.py → vit_base)
```
embed_dim  = 768
depth      = 12 (layers)
num_heads  = 12
head_dim   = 64  (768 / 12)
patch_size = 16
ffn_ratio  = 4.0  (MLP FFN, NOT SwiGLU for ViT-B)
in_chans   = 5    (satellite: Sentinel-1 + Sentinel-2 bands)
n_storage_tokens = 0  (no register tokens in this satellite fork — confirmed from ssl_default_config.yaml)
```

**Token counts per crop:**
- Global (224px): `(224/16)² = 196` patches + 1 CLS + 0 registers = **197 tokens**
- Local (96px): `(96/16)² = 36` patches + 1 CLS + 0 registers = **37 tokens**

**RoPE positional encoding** (not learned absolute PE) — compute cost is negligible vs attention.

### Multi-Model Structure (ssl_meta_arch.py)
Three ViT instances per run:
1. **Student** (`self.student.backbone`) — trained, FSDP2-wrapped
2. **Teacher / EMA** (`self.teacher.backbone`) — EMA of student, no grad
3. **Gram teacher** (`self.gram_teacher.backbone`) — optional frozen reference, no grad

Each has a DINO head + iBOT head on top of backbone.

---

## 3. FLOP Estimation — Step-by-Step

### ViT-B Forward FLOPs Per Image (one resolution)

```python
def vit_b_forward_flops(seq_len: int, D: int = 768, L: int = 12, ffn_ratio: float = 4.0) -> int:
    """
    FLOPs for a single forward pass of ViT-B on one image.
    seq_len = num_patches + 1 (CLS) + n_registers
    
    Each linear op counts as 2 FLOPs per multiply-add.
    """
    ffn_dim = int(D * ffn_ratio)
    
    # Per-layer:
    # QKV projection: seq_len × D × 3D × 2  = 6 × seq_len × D²
    # O projection:   seq_len × D × D × 2   = 2 × seq_len × D²
    # Attn scores:    seq_len × seq_len × D × 2  (QKᵀ, bidirectional, full square)
    # Attn weighted:  seq_len × seq_len × D × 2  (softmax(QKᵀ)V)
    # FFN up:         seq_len × D × ffn_dim × 2
    # FFN down:       seq_len × ffn_dim × D × 2
    attn_linear = 8 * seq_len * D * D          # QKV + O
    attn_scores = 2 * seq_len * seq_len * D    # QKᵀ + weighted sum (bidirectional → full seq²)
    ffn         = 4 * seq_len * D * ffn_dim    # up + down
    
    per_layer = attn_linear + attn_scores + ffn
    return L * per_layer


# Global crop (224px, 197 tokens):
global_fwd = vit_b_forward_flops(seq_len=197)  # ≈ 17.4 GFLOPs

# Local crop (96px, 37 tokens):
local_fwd  = vit_b_forward_flops(seq_len=37)   # ≈ 3.2 GFLOPs
```

**Key insight**: bidirectional attention means full `seq²` — no factor-of-2 saving from causal mask.

### Full Step FLOPs Per Original Image in Batch

```python
def flops_per_image_step(
    n_global: int = 2,
    n_local: int = 8,   # cfg.crops.local_crops_number
    gram_enabled: bool = False,
) -> int:
    """
    FLOPs for one full training step, per image in the original batch.
    
    Backward is ~2× the student forward (one matmul per grad, two per weight grad).
    Teacher and gram teacher are no_grad → forward only.
    """
    global_fwd  = vit_b_forward_flops(seq_len=197)  # ~17.4 GFLOPs
    local_fwd   = vit_b_forward_flops(seq_len=37)   # ~3.2 GFLOPs

    # Student: forward (global + local, run jointly) + backward (~2× forward)
    student_fwd = n_global * global_fwd + n_local * local_fwd  # ~60.4 GFLOPs
    student_bwd = 2 * student_fwd                               # ~120.8 GFLOPs

    # Teacher: 2 global crops, no grad
    teacher_fwd = n_global * global_fwd                         # ~34.8 GFLOPs

    # Gram teacher (optional): 2 global crops, no grad
    gram_fwd = (n_global * global_fwd) if gram_enabled else 0   # ~34.8 GFLOPs if enabled
    
    # DINO/iBOT heads: small relative to backbone, estimate ~5% of backbone
    head_overhead = 0.05 * (student_fwd + teacher_fwd)
    
    return student_fwd + student_bwd + teacher_fwd + gram_fwd + int(head_overhead)


# Default config (gram disabled): ~221 GFLOPs per image per step
# With gram teacher: ~256 GFLOPs per image per step
```

**Sanity check**: DINOv2 paper reports ~17.5 GFLOPs per forward pass for ViT-B/16 (197 tokens, no registers) — our 17.4 is consistent.

---

## 4. MFU Calculation Function

```python
# H100 SXM5 BF16 peak
H100_BF16_TFLOPS = 1979.0  # WRONG: 1979 is with 2:4 sparsity; dense = 989 — see ERRATA item 4

def compute_mfu(
    images_per_sec: float,          # total across all GPUs
    batch_size_per_gpu: int,
    num_gpus: int,
    n_global_crops: int = 2,
    n_local_crops: int = 8,
    gram_enabled: bool = False,
    peak_tflops: float = H100_BF16_TFLOPS,
) -> float:
    """
    Returns MFU as a fraction (0.0–1.0).
    """
    flops = flops_per_image_step(n_global_crops, n_local_crops, gram_enabled)
    
    actual_tflops = (images_per_sec * flops) / 1e12
    theoretical_peak = num_gpus * peak_tflops
    
    return actual_tflops / theoretical_peak


# Example: 8 GPUs, 64 img/GPU/step, 1-second steps
# images_per_sec = 8 * 64 = 512
# compute_mfu(512, 64, 8, gram_enabled=False) ≈ ?
# 512 * 221e9 * 2 / 1e12 / (8 * 989) ≈ 2.86%  — floor estimate (assumes 1-second/step)
# NOTE: original had 1979 (sparsity) and missing 2× — see ERRATA items 2 and 4
# Current config already has compile=True; expect first measurement well above this
```

**Reference point**: PaLM paper gets ~46% MFU on dense LLM. For ViT SSL with multi-crop complexity, 20–35% is a realistic optimized target on H100.

---

## 5. Where to Instrument in the Repo

### Primary: `dinov3/train/train.py` — inside `do_train()`

The training loop at line ~414. Key measurement points:

```python
# Add timing around the core iteration:
# (1) Before the loop, initialize:
step_start = torch.cuda.Event(enable_timing=True)
step_end   = torch.cuda.Event(enable_timing=True)

# (2) Inside the for loop, wrap forward_backward:
step_start.record()
total_loss, metrics_dict = model.forward_backward(data, teacher_temp=teacher_temp, iteration=it)
# ... (grad clip, optimizer, EMA) ...
optimizer.step()
model.update_ema(mom)
step_end.record()
torch.cuda.synchronize()

step_time_ms = step_start.elapsed_time(step_end)
images_this_step = global_batch_size  # already computed in do_train
images_per_sec = images_this_step / (step_time_ms / 1000.0)

mfu = compute_mfu(images_per_sec, cfg.train.batch_size_per_gpu, distributed.get_world_size(), ...)

# (3) Add to metric_logger:
metric_logger.update(mfu=mfu, images_per_sec=images_per_sec, step_time_ms=step_time_ms)
```

**Alternative**: use `time.perf_counter()` to avoid GPU sync cost during normal training — only use CUDA events for profiling.

### Secondary: Add MFU computation utility to `dinov3/utils/`

Create `dinov3/utils/mfu.py` — keeps it modular and testable.

---

## 6. Key Code Paths to Understand

### Student forward is JOINT for global + local crops
In `ssl_meta_arch.py → get_student_output()`:
```python
global_out, local_out = self.student.backbone(
    [global_crops, local_crops.flatten(0, 1)],  # list → forward_features_list()
    masks=[masks, None],
    is_training=True,
)
```
`forward_features_list()` in `vision_transformer.py` processes **both resolutions in a single block loop** but sequentially (not batched together). This is a key perf consideration — global (197-token) and local (37-token) can't be batched into one matmul because they have different sequence lengths.

### Teacher is called separately, no grad
In `ssl_meta_arch.py → get_teacher_output()`:
```python
@torch.no_grad()
def get_teacher_output(self, images, ...):
    backbone_out = self.teacher.backbone(images, is_training=True)
```
Called *before* student forward in `forward_backward()`. So the order is: **teacher forward → student forward → student backward**.

### torch.compile wraps individual blocks
From `ac_compile_parallelize.py → compile_transformer()`:
```python
for block_id, block in enumerate(model.blocks):
    model.blocks[block_id] = wrap_compile_block(block, cfg.train.cudagraphs, is_backbone_block=True)
```
Compile is per-block, not full model. `cfg.train.compile = True` in the active config — already enabled.

### FSDP2 with SHARD_GRAD_OP
```python
assert cfg.compute_precision.sharding_strategy == "SHARD_GRAD_OP"
mp_policy = MixedPrecisionPolicy(
    param_dtype=DTYPE_MAP[cfg.compute_precision.param_dtype],  # "bf16"
    reduce_dtype=DTYPE_MAP[cfg.compute_precision.reduce_dtype],  # "fp32"
)
```
Params are BF16, gradient reduction is FP32. This is the right setting for numerical stability.

### Checkpointing (activation checkpointing, not model checkpointing)
Two modes in `ac_compile_parallelize.py`:
- **Full checkpointing** (`cfg.train.checkpointing_full = True`): recomputes all activations
- **Selective** (default): only saves matmul and flash attention outputs; recomputes everything else

Selective checkpointing is the default and is more compute-efficient.

---

## 7. Configuration Findings (Resolved from Codebase)

All questions resolved by reading `dinov3/configs/ssl_default_config.yaml` and `run.sh`:

| Parameter | Confirmed Value | Impact on MFU |
|---|---|---|
| Deployment | **Single node, 8× H100** (`--nodes=1` in `run.sh`) | No inter-node communication overhead |
| `local_crops_number` | **8** | Config default; FLOP formula correct as written |
| Gram loss | **Disabled** (`gram.use_loss: false`) | No third backbone forward; use ~221 GFLOPs figure |
| `n_storage_tokens` | **0** (satellite fork drops register tokens) | Corrects seq_len: 197 global, 37 local (was 201/41) |
| `compile` | **True** | Baseline already includes compile; not a remaining optimization |
| Activation checkpointing | **Off** (`checkpointing: false`, `checkpointing_full: false`) | No recompute overhead; step time is pure forward+backward |
| Batch size per GPU | **64** (set in `run.sh`) | Global batch = 512 images across 8 GPUs |
| Dataset storage | **Weka** (`/mnt/weka/...`); `cache_dataset=false` in `latest_run.sh` (sweep) | IO is live from Weka; no warm-up cache |
| Local crop size | **96px only** (`global_local_crop_pairs_ratios: 1.0` is scalar → single pair) | No multi-resolution; FLOP formula needs no weighted sum |

---

## 8. Implementation Plan (Ordered)

### Step 1: Write `dinov3/utils/mfu.py`

```python
"""MFU tracking utilities for DINOv3 / iBOT SSL training."""

H100_BF16_TFLOPS = 1979.0  # WRONG: dense = 989 (1979 assumes 2:4 sparsity) — see ERRATA item 4
A100_BF16_TFLOPS = 312.0   # WRONG: dense = 156 (312 assumes 2:4 sparsity) — see ERRATA item 4


def vit_forward_flops(
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    ffn_ratio: float = 4.0,
) -> int:
    """FLOPs for one forward pass of a ViT on one image (bidirectional attention)."""
    ffn_dim = int(hidden_dim * ffn_ratio)
    attn_linear = 8 * seq_len * hidden_dim * hidden_dim
    attn_scores  = 2 * seq_len * seq_len * hidden_dim
    ffn          = 4 * seq_len * hidden_dim * ffn_dim
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
    n_registers: int = 0,  # this satellite fork uses n_storage_tokens=0
    gram_enabled: bool = False,
    head_overhead_pct: float = 0.05,
) -> int:
    """
    Total FLOPs for one full training step per original image in the batch.
    Accounts for student (fwd + bwd), teacher (fwd only), and optional gram teacher (fwd only).
    """
    global_seq = (global_crop_size // patch_size) ** 2 + 1 + n_registers  # patches + CLS + regs
    local_seq  = (local_crop_size // patch_size) ** 2 + 1 + n_registers

    global_fwd = vit_forward_flops(global_seq, hidden_dim, num_layers, ffn_ratio)
    local_fwd  = vit_forward_flops(local_seq, hidden_dim, num_layers, ffn_ratio)

    student_fwd = n_global_crops * global_fwd + n_local_crops * local_fwd
    student_bwd = 2 * student_fwd         # backward ≈ 2× forward
    teacher_fwd = n_global_crops * global_fwd
    gram_fwd    = n_global_crops * global_fwd if gram_enabled else 0

    backbone_flops = student_fwd + student_bwd + teacher_fwd + gram_fwd
    return int(backbone_flops * (1.0 + head_overhead_pct))


def compute_mfu(
    images_per_sec: float,
    flops_per_image: int,
    num_gpus: int,
    peak_tflops: float = H100_BF16_TFLOPS,
) -> float:
    """Returns MFU as a fraction 0.0–1.0."""
    actual_tflops     = (images_per_sec * flops_per_image) / 1e12
    theoretical_peak  = num_gpus * peak_tflops
    return actual_tflops / theoretical_peak
```

### Step 2: Integrate into `do_train()` in `train.py`

```python
# At top of do_train(), before the loop — precompute FLOP constant
from dinov3.utils.mfu import compute_dino_flops_per_image, compute_mfu

flops_per_image = compute_dino_flops_per_image(
    global_crop_size=cfg.crops.global_crops_size,
    local_crop_size=cfg.crops.local_crops_size,
    patch_size=cfg.student.patch_size,
    n_global_crops=2,
    n_local_crops=cfg.crops.local_crops_number,
    hidden_dim=768,  # or read from model config
    num_layers=12,
    ffn_ratio=4.0,
    n_registers=cfg.student.n_storage_tokens,  # 0 in current config
    gram_enabled=cfg.gram.use_loss,
)
num_gpus = distributed.get_world_size()

# Inside the loop, after optimizer.step():
_t1 = time.perf_counter()
# [existing training code]
_t2 = time.perf_counter()

step_time_sec = _t2 - _t1
images_per_sec = global_batch_size / step_time_sec
mfu = compute_mfu(images_per_sec, flops_per_image, num_gpus)

metric_logger.update(
    mfu=mfu * 100,            # log as %
    images_per_sec=images_per_sec,
    step_time_ms=step_time_sec * 1000,
)
```

### Step 3: Write a test

```python
# tests/test_mfu.py
from dinov3.utils.mfu import compute_dino_flops_per_image, compute_mfu

def test_flop_estimate_sanity():
    """ViT-B forward FLOPs should be ~17.5 GFLOPs per image (matches DINOv2 paper)."""
    flops = compute_dino_flops_per_image(
        n_global_crops=1, n_local_crops=0,  # just one global crop forward
        gram_enabled=False, head_overhead_pct=0.0,
    )
    # Forward only is 1/3 of student_fwd + bwd; teacher = same as student global
    # So for 1 global, 0 local: student_fwd + student_bwd + teacher_fwd = 3 * global_fwd
    # Expect total ≈ 3 × 17.4 GFLOPs ≈ 52.2 GFLOPs  (seq_len=197, n_registers=0)
    assert 48e9 < flops < 58e9, f"Expected ~52 GFLOPs, got {flops/1e9:.1f}"

def test_mfu_range():
    """MFU should be between 0 and 1 for sensible inputs."""
    flops = compute_dino_flops_per_image()
    mfu = compute_mfu(images_per_sec=512, flops_per_image=flops, num_gpus=8)
    assert 0.0 < mfu < 1.0
```

---

## 9. Expected Baseline MFU Numbers

Based on the architecture, **before any optimization**, rough estimate on 8x H100:

| Config | images/sec (est.) | MFU estimate |
|---|---|---|
| No compile, selective checkpoint | ~400 | ~7–10% |
| With torch.compile | ~550 | ~10–14% |
| Compile + batched multi-crop | ~700 | ~13–18% |
| Compile + FlashAttention-3 | ~900 | ~17–23% |
| All optimizations | 1200+ | ~23–32% |

These are rough. The actual baseline is the first thing to measure.

**Current config status**: `compile=True` already enabled, no activation checkpointing, single node — this puts the **starting point at or above the "With torch.compile" row**. First measurement should land in the 10–14% range or higher. The remaining high-ROI steps are batched multi-crop (fusing the two `forward_features_list()` calls) and FlashAttention-3.

**Key context**: ViT-B is compute-small (86M params, ~17 GFLOPs/image forward). At 8×H100, the theoretical ceiling is enormous — the bottleneck is likely Python overhead and memory latency on small local crops (37-token sequences). No inter-node communication to worry about (single node confirmed).

---

## 10. Hardware Reference

| GPU | BF16 TFLOPS | HBM BW | NVLink BW |
|---|---|---|---|
| H100 80GB SXM5 | ~~1,979~~ **989** (dense) | 3.35 TB/s | 900 GB/s |
| A100 80GB SXM4 | ~~312~~ **156** (dense) | 2.0 TB/s | 600 GB/s |

**Ridge point** (H100): 989 TFLOPS / 3.35 TB/s ≈ **295 FLOP/byte** (dense; original used sparsity number — see ERRATA item 4)
- Ops with arithmetic intensity < 591 are memory-bound (LayerNorm, softmax, small matmuls)
- Large matmuls in ViT-B are compute-bound at typical batch sizes

---

## 11. Key Files in the Repo

```
dinov3/train/train.py               ← do_train() — add MFU logging here
dinov3/train/ssl_meta_arch.py       ← forward_backward(), get_student_output(), get_teacher_output()
dinov3/models/vision_transformer.py ← DinoVisionTransformer, forward_features_list()
dinov3/fsdp/ac_compile_parallelize.py ← FSDP2 wrapping, torch.compile per-block
dinov3/configs/ssl_default_config.yaml ← all hyperparams including crops, compile, checkpointing
dinov3/logging/metric_logger.py     ← MetricLogger.update() — accepts any kwargs
run.sh                              ← SLURM config, check nproc_per_node and num-nodes
```

---

## 12. Profiling Commands (Once Baseline MFU is Logged)

```bash
# Basic GPU util + tensor core activity (run while training)
dcgmi dmon -e 203,1003 -d 1000
# 203 = GPU utilization %, 1003 = FP16/tensor core active %

# PyTorch profiler trace (add --profiling flag if available, or wrap manually)
torchrun --nproc_per_node=8 dinov3/train/train.py \
  --config-file dinov3/configs/ssl_default_config.yaml \
  --output-dir ./output_profile \
  --opts train.batch_size_per_gpu=32  # smaller batch for profiling

# Quick memory check
python -c "
import torch
torch.cuda.set_device(0)
print(f'Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')
print(f'Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB')
"
```

---

## 13. Latest Run Context (`latest_run.sh`)

`latest_run.sh` is a **Slurm array job** (`--array=0-5`, 1 hour each) sweeping dataloader parameters:

| Array ID | `num_workers` | `prefetch_factor` | `batch_size_per_gpu` | run name |
|---|---|---|---|---|
| 0 | 8 | 2 | 64 | `nw8_pf2_bs64` |
| 1 | 12 | 2 | 64 | `nw12_pf2_bs64` |
| 2 | 16 | 2 | 64 | `nw16_pf2_bs64` |
| 3 | 12 | 4 | 64 | `nw12_pf4_bs64` |
| 4 | 12 | 2 | 96 | `nw12_pf2_bs96` |
| 5 | 16 | 4 | 64 | `nw16_pf4_bs64` |

**Notable differences from `run.sh`:**
- **Dataset**: intelinair + MAID + Sentinel-1 + NAIP only — **no Sentinel-2** in this sweep
- **`cache_dataset=false`** — live reads from Weka, no caching
- **`OMP_NUM_THREADS=1`** (not 8)
- **Pretrained weights**: `./pretrained_weights/dinov3_vitb16_pretrain.pth` — fine-tuning from DINOv2, not scratch
- **WandB group**: `satellite_sweep` (not `satellite_only`)
- `wandb.group=satellite_sweep`; outputs to `./output_${RUN_NAME}/`

This sweep is specifically designed to find the optimal dataloader config before committing to a long training run — results here will inform the `num_workers` / `prefetch_factor` settings used in the MFU baseline.

---

## 14. Resolved Questions

1. **Multi-resolution crops?** ✅ **No.** `global_local_crop_pairs_ratios: 1.0` is a scalar float. `train.py:384-386` wraps it as `[1.0]` (single pair) when the value is `int | float`. FLOP formula needs no weighted sum — single global size (224px) and single local size (96px).

2. **iBOT masking reduces effective FLOPs?** ✅ **No.** Masking controls which patch predictions are used in the iBOT loss, but all tokens still pass through the full backbone unchanged. FLOPs are unaffected by the mask.

3. **Gram teacher process group** ✅ **Moot — gram is disabled** (`gram.use_loss: false`). For reference: `ssl_meta_arch.py:820-828` shows gram teacher uses `default_process_group` while the student uses `process_subgroup`; on single-node (confirmed) these are identical.

4. **WandB logging** ✅ **Confirmed active.** `wandb.enabled: true`, project `dinov3-satellite`, run name `satellite_8xh100` (set in `run.sh:41-43`). MFU will land in WandB automatically once added to `metric_logger`.
