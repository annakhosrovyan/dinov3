# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Satellite-specialized fork of Meta's DINOv3 self-supervised vision foundation model. Trains ViT models on multi-sensor satellite imagery (Sentinel-1, Sentinel-2, NAIP) using DINO + iBOT objectives with FSDP2 distributed training on H100 GPUs.

## Environment Setup

**Working env (2026-03-30)**: `~/.conda/envs/test-conda-slurm` — torch 2.6.0+cu124, all deps installed.

```bash
# In Slurm scripts — use PATH prepend, NOT conda activate (fails on GPU nodes):
export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

# Required env vars
export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
```

**Requires PyTorch >= 2.6** (not 2.1 — the codebase uses `register_fsdp_forward_method` added in 2.6).

**The shared Weka env `/mnt/weka/shared-cache/miniforge3/envs/dinov3` does NOT have torch installed** — do not use it for running jobs.

Check `run.sh` for SLURM configuration (8x H100, 64 CPUs).

## Training Commands

```bash
# Submit to cluster (preferred)
sbatch run.sh

# Direct multi-GPU training
torchrun --nproc_per_node=8 dinov3/train/train.py \
  --config-file dinov3/configs/ssl_default_config.yaml \
  --output-dir ./output_satellite_s1_s2ab \
  --opts student.in_chans=5 teacher.in_chans=5 train.batch_size_per_gpu=64

# Resume from checkpoint
torchrun ... train.py ... --opts train.output_dir=./output_satellite_s1_s2ab  # finds latest ckpt automatically

# Smoke test (100 iters, no pretrained weights, synthetic data):
sbatch scripts/mfu_validation_run.sh
```

Config CLI overrides use OmegaConf dot-notation: `section.key=value` pairs passed **directly** (no `--opts` prefix — positional after the required args). Example: `student.pretrained_weights="" train.batch_size_per_gpu=32`.

**Pretrained weights**: Default config points to `./pretrained_weights/dinov3_vitb16_pretrain.pth`. The actual file lives at `/auto/home/anna.khosrovyan/dinov3/pretrained_weights/dinov3_vitb16_pretrain.pth` (accessible from GPU nodes; not from headnode container). Use `student.pretrained_weights=""` to skip for smoke tests — the config guards with `if self.cfg.student.pretrained_weights:`.

**Dataset access**: `run.sh` uses `/mnt/weka/akhosrovyan/re-id/pretraining/...` — these paths are **permission-denied for adovlatyan** from the headnode container, but are accessible from GPU nodes when the job is submitted. A synthetic dataset exists at `/mnt/weka/adovlatyan/synthetic_intelinair.h5` for local testing.

**Storage**: Use `/mnt/weka/adovlatyan/` for logs and outputs (not `/data/adovlatyan/` which is NFS and deprecated). Log path: `/mnt/weka/adovlatyan/logs/`.

## Code Quality

```bash
ruff check dinov3/        # lint
ruff format dinov3/       # format (120-char lines, Python 3.11)
mypy dinov3/              # type checking
```

Config: `pyproject.toml`. Pylint is scoped to similarities/misc only.

## Architecture

### Configuration System (`dinov3/configs/`)
- **`ssl_default_config.yaml`** — master defaults; all keys documented there
- **`config.py`** — `setup_config()` merges default → file → CLI; `apply_scaling_rules_to_cfg()` auto-scales LR: `LR *= 4 * sqrt(batch_per_gpu * world_size / 1024)`
- Section breakdown: `student`/`teacher` (arch), `crops` (augmentation), `dino`/`ibot`/`gram` (losses), `optim` (scheduler, LR), `train` (dataset paths, checkpointing), `compute_precision` (bf16/fp32 mix)

### Training Loop (`dinov3/train/train.py`)
- `main()` → `do_train()` at line 414 is the core loop
- Each iteration: schedule update → zero_grad → `model.forward_backward()` → grad clip → all-reduce → optimizer step → EMA teacher update
- **NaN detection**: aborts after 2 consecutive NaN losses
- **Checkpointing**: every 3750 iters; evals every 12500 iters

### Model Architecture (`dinov3/train/ssl_meta_arch.py`)
- `SSLMetaArch`: student ViT + EMA teacher + DINO head + iBOT head + optional Gram head
- Default: `vit_base`, patch 16, **5 input channels** (satellite), RoPE positional embeddings
- `forward_backward()` computes DINO loss (CLS token distillation) + iBOT loss (masked patch prediction) + KoLeo (collapse prevention)
- FSDP2 with `SHARD_GRAD_OP` strategy, bf16 params / fp32 reduction

### LR Schedulers (`dinov3/train/cosine_lr_scheduler.py`)
- **`WSDLRScheduler`** (primary): Warmup → Stable → Decay (10% of steps). Set via `cfg.optim.lr_scheduler = "wsd"`
- **`CosineScheduler`** (legacy): linear warmup → cosine decay
- Separate schedules for weight decay, EMA momentum, teacher temperature

### Satellite Data Pipeline (`dinov3/data/`)
- **`MixedSatelliteDataset`** (`datasets/mixed_satlas_dataset.py`): composes multiple satellite sources with `dataset_weight` for effective size scaling
- Dataset spec string format (passed in config): `MixedSatelliteDataset:sen1_data_path=/path:sen1_weight=1.0:sen2a_data_path=...`
- **`to_five_channels()`** (`datasets/channel_utils.py`): normalizes all sources to 5-channel tensor; each dataset type has its own per-channel statistics
- Data loaders: `Sen1Dataset`, `Sen2Dataset`, `NaipDataset`, `HDF5Dataset` (BEN/Intelinair/Sen12MS)
- Multi-crop collation: 2 global (224px) + 8 local (96px) crops with iBOT masks applied in `collate.py`

### Distributed Infra
- **`dinov3/fsdp/ac_compile_parallelize.py`**: wraps model with FSDP2, optional `torch.compile`
- **`dinov3/checkpointer/`**: DCP (Distributed Checkpoint Protocol) for sharded/consolidated saves
- Sharded checkpoints in `{output_dir}/ckpt/`; last 3 kept by default

## Key Design Decisions

- **5-channel unification**: All satellite sources → 5ch via `to_five_channels()` so a single ViT handles all sensors
- **Weighted dataset mixing**: `dataset_weight` in `MixedSatelliteDataset` controls effective frequency without full data copies
- **WSD over cosine**: Better handles large-scale satellite training where cosine decay may be too aggressive
- **RoPE with `separate` normalization**: X and Y coordinates normalized independently; controlled by `pos_embed_rope_normalize_coords`
- **Gram loss** (optional): anchors representations to a reference teacher; enable via `cfg.gram`

---

## MFU Tracking & Performance Optimization

> **MFU tracking is implemented** (2026-03-30). See `docs/mfu-results-2026-03-30.md` for baseline
> numbers and validation results. See `docs/dinov3-mfu-tracking-initial-brief-03-27-26.md` for
> the full FLOP formula derivation. Utilities in `dinov3/utils/mfu.py`; tests in `tests/test_mfu.py`.

### ViT-B at a Glance (for FLOP counting)
| Property | Value |
|---|---|
| `embed_dim` | 768 |
| `depth` | 12 layers |
| `num_heads` | 12 |
| `patch_size` | 16 |
| `in_chans` | 5 (satellite) |
| `n_storage_tokens` | **0** (satellite fork drops register tokens — confirmed from `ssl_default_config.yaml`) |
| Global crop tokens | **197** (196 patches + 1 CLS + 0 registers) |
| Local crop tokens | **37** (36 patches + 1 CLS + 0 registers) |
| H100 BF16 peak | 1979 TFLOPS |

`vit_base` defined at `dinov3/models/vision_transformer.py:344`

### Key Entry Points for MFU / Timing Instrumentation

#### Training loop — key locations
| What | File | Line(s) |
|---|---|---|
| `do_train()` function start | `dinov3/train/train.py` | 416 |
| MFU `flops_per_image` precomputed | `dinov3/train/train.py` | 459 |
| CUDA event init (`step_start_event`) | `dinov3/train/train.py` | 526 |
| `step_start_event.record()` | `dinov3/train/train.py` | 558 |
| `optimizer.zero_grad()` | `dinov3/train/train.py` | 559 |
| **`model.forward_backward()`** — core compute | `dinov3/train/train.py` | 560 |
| `optimizer.step()` + `model.update_ema()` | `dinov3/train/train.py` | 606–607 |
| `step_end_event.record()` + MFU compute | `dinov3/train/train.py` | 608–614 |
| `metric_logger.update(mfu=, images_per_sec=, ...)` | `dinov3/train/train.py` | 629–637 |
| WandB log (includes mfu_pct, images_per_sec) | `dinov3/train/train.py` | 639–658 |
| Eval sync point (`cuda.synchronize`) | `dinov3/train/train.py` | 666 |
| Checkpoint sync point (`cuda.synchronize`) | `dinov3/train/train.py` | 670 |

#### SSLMetaArch — forward pass breakdown
| What | File | Line(s) |
|---|---|---|
| `forward_backward()` entry | `dinov3/train/ssl_meta_arch.py` | 362 |
| Teacher forward (`@no_grad`) | `dinov3/train/ssl_meta_arch.py` | 391–398 |
| Student forward (global + local, joint) | `dinov3/train/ssl_meta_arch.py` | 400–407 |
| Gram teacher forward (optional) | `dinov3/train/ssl_meta_arch.py` | 409–419 |
| `compute_losses()` — all 4 loss terms | `dinov3/train/ssl_meta_arch.py` | 421–431 |
| `backprop_loss()` — `loss.backward()` | `dinov3/train/ssl_meta_arch.py` | 433 |
| Teacher EMA update (`_foreach_mul_/add_`) | `dinov3/train/ssl_meta_arch.py` | 720–733 |
| `prepare_for_distributed_training()` | `dinov3/train/ssl_meta_arch.py` | 818–835 |

#### Student forward — joint global+local pass
`get_student_output()` at `dinov3/train/ssl_meta_arch.py:537` calls
`backbone.forward_features_list([global_crops, local_crops])` — both resolutions processed
sequentially in one block loop (they cannot be batched because seq lengths differ: 197 vs 37).
This is the key perf bottleneck for multi-crop.

#### Loss implementations
| Loss | File | Lines |
|---|---|---|
| DINO CLS token loss (cross-entropy + centering) | `dinov3/loss/dino_clstoken_loss.py` | 16–124 |
| iBOT masked patch loss (Sinkhorn-Knopp) | `dinov3/loss/ibot_patch_loss.py` | 61–142 |
| KoLeo diversity regularization | `dinov3/loss/koleo_loss.py` | 14–113 |
| `compute_losses()` — wires all losses together | `dinov3/train/ssl_meta_arch.py` | 591–691 |

### Distributed & Compilation Infrastructure
| What | File | Line(s) |
|---|---|---|
| FSDP2 wrapping (block-level, with prefetch) | `dinov3/fsdp/ac_compile_parallelize.py` | 110–124 |
| `torch.compile` per block (backbone) | `dinov3/fsdp/ac_compile_parallelize.py` | 65–87 |
| Selective activation checkpointing setup | `dinov3/fsdp/ac_compile_parallelize.py` | 25–47 |
| MixedPrecisionPolicy (BF16 param, FP32 reduce) | `dinov3/fsdp/ac_compile_parallelize.py` | 183–201 |
| Inference-only model optimization (EMA teacher) | `dinov3/fsdp/ac_compile_parallelize.py` | 212–220 |

### Logging & Metrics
| What | File | Line(s) |
|---|---|---|
| `MetricLogger` class (iter_time, data_time, mem) | `dinov3/logging/helpers.py` | 19–133 |
| `SmoothedValue` (window=20 rolling avg) | `dinov3/logging/helpers.py` | 136–203 |
| Memory tracking (`cuda.memory_allocated`) | `dinov3/logging/helpers.py` | 87–89 |
| WandB init (only rank 0) | `dinov3/train/train.py` | 466–486 |
| JSON metrics file (`training_metrics.json`) | `dinov3/train/train.py` | 464–465 |

### Config Keys Relevant to Performance
```yaml
train.compile: true             # torch.compile per transformer block (already enabled)
train.cudagraphs: false         # CUDA graph capture (disabled by default)
train.checkpointing: false      # selective activation checkpointing (off by default)
train.checkpointing_full: false # full recompute checkpointing
train.num_workers: 20           # DataLoader workers (run.sh override)
train.prefetch_factor: 8        # Prefetch batches per worker (run.sh override)
train.persistent_workers: true  # Keep workers alive between epochs
compute_precision.param_dtype: bf16   # Parameter dtype
compute_precision.reduce_dtype: fp32  # Gradient reduction dtype
crops.local_crops_number: 8     # 8 local crops (affects student FLOP count significantly)
```

### MFU — Baseline Results (2026-03-30)

Measured on job 6585, gpu02, 2× H100, bs=32/GPU, torch.compile=True, synthetic data:

| Metric | Value |
|---|---|
| MFU (post-warmup avg) | **2.48%** |
| images/sec (total, 2 GPU) | ~434 |
| step_time_ms | ~147ms |
| FLOPs/image | 226.4 GFLOPs |

**FLOP formula uses MAC convention** (1 FLOP = 1 multiply-add), consistent with DINOv2 paper (~17.4 GFLOPs global forward). H100 BF16 peak = 1979 TFLOPS. To hit **10% MFU at 8 GPUs**: need ~8800 img/s total (~1100/GPU); current is ~217/GPU, so ~5× gap.

**Note**: First iteration is ~35 seconds (torch.compile warmup). Skip first 1–2 logged points for any analysis.

### Known Performance Considerations
- **CUDA event timing implemented**: `torch.cuda.Event` pairs wrap `zero_grad` → `update_ema` — GPU-synchronized step time. The wall-clock `time.time()` in `MetricLogger.log_every()` (`helpers.py:69`) is separate and still present (used for eta/data-time).
- **Two `cuda.synchronize()` calls** in the training loop (`train.py:666,670`) — only at eval/checkpoint, not every iteration. The MFU timing uses `step_end_event.synchronize()` which adds one sync per iter — acceptable overhead.
- **Async center all-reduce** in DINO/iBOT losses overlaps with computation (uses `async_op=True` in `dino_clstoken_loss.py:101` and `ibot_patch_loss.py`).
- **`torch._foreach_mul_/add_`** for EMA update (`ssl_meta_arch.py:728–730`) — fused multi-tensor ops, efficient.
- **`cudnn.benchmark = True`** set globally (`train.py:45`) — auto-selects fastest conv algo.
- **`matmul.allow_tf32 = True`** set globally (`train.py:44`) — uses TF32 for linear layers.

### Data Pipeline Performance Notes
- `pin_memory=True` always (`loaders.py:246`)
- `prefetch_factor=4` default, overridden to 8 in `run.sh`
- `persistent_workers=True` in `run.sh` — avoids worker respawn overhead
- No profiling/timing instrumentation in data loading code — if throughput is a bottleneck, instrument `collate_data_and_cast()` (`collate.py:11`) and `make_data_loader()` (`loaders.py:196`)
