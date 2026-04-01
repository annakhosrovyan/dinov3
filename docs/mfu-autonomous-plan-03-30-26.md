# MFU Tracking — Autonomous Iteration Plan
**Created**: 2026-03-30
**Purpose**: Self-contained task doc for a Claude Code auto-mode session to implement, test,
iterate on, and validate MFU tracking in the DINOv3 satellite training codebase.

---

> **ERRATA (2026-03-31)** — Several errors in this plan were found during or after implementation.
> The plan is preserved as-is for historical reference; do not use the test code or expected
> values below without reading these corrections first.
>
> 1. **Section 2.3 smoke test expected value**: `expect ~7%` is wrong. The correct hardware MFU
>    at 512 img/s, 8 GPUs, ~226 GMACs/image is:
>    `512 × 226e9 × 2 / (8 × 1979e12) = 1.46%`. The 7% figure came from using MAC counts
>    against a hardware-FLOP denominator without the 2× conversion factor.
>
> 2. **Section 3 test code — three bugs** (all corrected in the actual `tests/test_mfu.py`):
>    - `compute_mfu(images_per_sec=512, flops_per_image=flops, ...)`: the kwarg is now
>      `macs_per_image`, not `flops_per_image`
>    - `test_floor_estimate`: bounds `0.05 <= mfu <= 0.20` (5–20%) are wrong. Correct bounds
>      for 512 img/s at 8×H100: `0.008 <= mfu <= 0.025` (~1.46% with hardware FLOP convention)
>    - `test_perfect_mfu_is_1`: `perfect_ips = (gpus × peak × 1e12) / flops` is wrong. With
>      the 2× MAC→hardware-FLOP conversion, correct is `/ (2 × macs)`
>
> 3. **Section 2.2 formula table**: `QKV+O projections: 8 × seq × D²` uses hardware FLOP
>    convention (×2 per MAC). The rest of the table and the actual implementation use MAC
>    convention: `4 × seq × D²` for QKV+O, `2 × seq × D × ffn_dim` for FFN.
>
> See `docs/mfu-results-2026-03-30.md` for the actual implementation and corrected results.

---

---

## 0. How to Launch This Session (Instructions for Aram)

Auto mode is already configured in `~/.claude/settings.json` (`permissions.defaultMode: "auto"`).
The `autoMode.environment` block tells the classifier this is a trusted HPC research cluster.

Start a new Claude Code session from the repo root:

```bash
cd /home/adovlatyan/dinov3-performance-optimizations/dinov3
claude
```

That's it. Auto mode activates automatically. The classifier will approve routine operations
(file edits, pytest, sbatch, squeue, log reads) without prompting, while still blocking
anything that looks like exfiltration or destructive actions.

**Model**: Sessions use the model set in `~/.claude/settings.json`. For best results on this
multi-step task, override to Opus at startup if not already set:
```bash
claude --model claude-opus-4-6
```

**Paste this prompt at the start of the new session**:

```
Follow the autonomous iteration plan at:
  docs/mfu-autonomous-plan-03-30-26.md

Execute all phases in order. Submit Slurm jobs, read logs, iterate until MFU tracking
is verified correct. Leave results in docs/mfu-results-<date>.md. Do not ask me
questions — make reasonable decisions and document them.
```

---

## 1. Context Summary (Read This Before Doing Anything)

**Repo**: Meta DINOv3 fork, satellite imagery specialization.
**Model**: ViT-B, 5-channel input, patch=16, no register tokens (`n_storage_tokens=0`).
**Training**: 8×H100 single node, FSDP2 SHARD_GRAD_OP, torch.compile=True.
**Config**: `dinov3/configs/ssl_default_config.yaml` + CLI overrides in `run.sh`.
**Key brief**: Full MFU derivation and FLOP math in `docs/dinov3-mfu-tracking-initial-brief-03-27-26.md` — **read it first before writing any code**.

**What does NOT exist yet** (as of 2026-03-30):
- `dinov3/utils/mfu.py` — needs to be created
- MFU logging in `do_train()` — needs to be added to `dinov3/train/train.py`
- Any test file for MFU — needs to be created
- A short-run Slurm script — needs to be created

---

## 2. Phase 1 — Implement `dinov3/utils/mfu.py`

### 2.1 Read first

Before writing a single line, read these files in full:
- `docs/dinov3-mfu-tracking-initial-brief-03-27-26.md` (full FLOP derivation)
- `dinov3/train/train.py` lines 414–652 (the `do_train()` function)
- `dinov3/train/ssl_meta_arch.py` lines 362–440 (the `forward_backward()` function)
- `dinov3/configs/ssl_default_config.yaml` (confirm n_storage_tokens=0, crops config, gram disabled)
- `dinov3/utils/__init__.py` (understand existing utils structure)

### 2.2 Create `dinov3/utils/mfu.py`

The file should export exactly:
- `vit_forward_flops(seq_len, hidden_dim, num_layers, ffn_ratio) -> int`
- `compute_dino_flops_per_image(...) -> int`
- `compute_mfu(images_per_sec, flops_per_image, num_gpus, peak_tflops) -> float`

**Critical correctness requirements** — verify these match the brief exactly:

| Term | Formula | Notes |
|---|---|---|
| QKV+O projections | `8 * seq_len * D²` | bidirectional, no causal mask saving |
| Attention scores (QKᵀ + weighted sum) | `2 * seq_len² * D` | full square, not triangular |
| FFN (up + down) | `4 * seq_len * D * ffn_dim` | ffn_dim = D × ffn_ratio |
| Student backward | `2 × student_fwd` | standard rule of thumb |
| Teacher forward | `n_global × global_fwd` | no grad, no backward |
| Gram teacher | `n_global × global_fwd` if enabled | disabled in current config |

**ViT-B parameters** (hardcode only as defaults, accept as args):
```
hidden_dim = 768
num_layers = 12
ffn_ratio = 4.0
patch_size = 16
n_registers = 0
```

**H100 peak**: `H100_BF16_TFLOPS = 1979.0`
Also define `A100_BF16_TFLOPS = 312.0` for reference.

**Sanity check numbers** (must hold when function is called with these inputs):
- `vit_forward_flops(seq_len=197, hidden_dim=768, num_layers=12, ffn_ratio=4.0)` → ~17.4 GFLOPs (DINOv2 paper reference)
- `vit_forward_flops(seq_len=37, ...)` → ~3.2 GFLOPs
- `compute_dino_flops_per_image(n_global=2, n_local=8, gram_enabled=False)` → ~221 GFLOPs

### 2.3 Verify with a quick Python check

After writing the file, run:
```bash
cd /home/adovlatyan/dinov3-performance-optimizations/dinov3
python -c "
from dinov3.utils.mfu import vit_forward_flops, compute_dino_flops_per_image, compute_mfu
g = vit_forward_flops(197, 768, 12, 4.0)
l = vit_forward_flops(37, 768, 12, 4.0)
total = compute_dino_flops_per_image(n_global_crops=2, n_local_crops=8)
mfu_ex = compute_mfu(512, total, 8)
print(f'Global fwd: {g/1e9:.2f} GFLOPs (expect ~17.4)')
print(f'Local fwd:  {l/1e9:.2f} GFLOPs (expect ~3.2)')
print(f'Total step: {total/1e9:.1f} GFLOPs/image (expect ~221)')
print(f'MFU @512 img/s, 8 GPUs: {mfu_ex*100:.2f}% (expect ~7% — floor estimate)')
"
```

Fix any discrepancies before moving on.

---

## 3. Phase 2 — Write Tests (`tests/test_mfu.py`)

Create `tests/test_mfu.py`. These tests must all pass with `python -m pytest tests/test_mfu.py -v`.

### Required test cases

```python
# tests/test_mfu.py
"""
Unit tests for MFU computation correctness.
Run with: python -m pytest tests/test_mfu.py -v
"""
from dinov3.utils.mfu import vit_forward_flops, compute_dino_flops_per_image, compute_mfu, H100_BF16_TFLOPS

class TestVitForwardFlops:
    def test_global_crop_matches_dinov2_paper(self):
        """DINOv2 paper reports ~17.5 GFLOPs for ViT-B/16 global crop (197 tokens)."""
        flops = vit_forward_flops(seq_len=197, hidden_dim=768, num_layers=12, ffn_ratio=4.0)
        assert 16e9 <= flops <= 19e9, f"Expected ~17.4 GFLOPs, got {flops/1e9:.2f}"

    def test_local_crop_smaller_than_global(self):
        """Local crop (37 tokens) should be much cheaper than global (197 tokens)."""
        g = vit_forward_flops(197, 768, 12, 4.0)
        l = vit_forward_flops(37, 768, 12, 4.0)
        assert l < g / 3, "Local crop should be < 1/3 of global (37² << 197²)"

    def test_scales_quadratically_with_seq_len_dominated_by_attn(self):
        """For large seq_len, attention (seq²) dominates — doubling seq_len ~4× FLOPs."""
        f1 = vit_forward_flops(100, 768, 12, 4.0)
        f2 = vit_forward_flops(200, 768, 12, 4.0)
        # Attn part goes 4× but FFN goes 2× so ratio should be between 2 and 4
        assert 2.5 <= f2 / f1 <= 4.0, f"Ratio was {f2/f1:.2f}"

    def test_scales_linearly_with_num_layers(self):
        f6 = vit_forward_flops(197, 768, 6, 4.0)
        f12 = vit_forward_flops(197, 768, 12, 4.0)
        assert abs(f12 / f6 - 2.0) < 0.01, "FLOPs must scale exactly 2× with layers"

    def test_no_registers_vs_registers(self):
        """n_storage_tokens=0 in this satellite fork — confirms our 197-token baseline."""
        f_no_reg = vit_forward_flops(197, 768, 12, 4.0)   # 196 patches + 1 CLS
        f_4reg   = vit_forward_flops(201, 768, 12, 4.0)   # 196 + 1 + 4 registers
        assert f_4reg > f_no_reg, "More tokens = more FLOPs"

class TestComputeDinoFlopsPerImage:
    def test_default_config_range(self):
        """Default config (2 global + 8 local, gram disabled) should be ~221 GFLOPs."""
        flops = compute_dino_flops_per_image(
            global_crop_size=224, local_crop_size=96, patch_size=16,
            n_global_crops=2, n_local_crops=8,
            hidden_dim=768, num_layers=12, ffn_ratio=4.0,
            n_registers=0, gram_enabled=False, head_overhead_pct=0.05,
        )
        assert 200e9 <= flops <= 245e9, f"Expected ~221 GFLOPs, got {flops/1e9:.1f}"

    def test_gram_adds_flops(self):
        """Enabling gram teacher adds one more teacher-forward (2 global crops)."""
        base = compute_dino_flops_per_image(gram_enabled=False)
        with_gram = compute_dino_flops_per_image(gram_enabled=True)
        global_fwd = vit_forward_flops(197, 768, 12, 4.0)
        gram_added = with_gram - base
        # Gram adds 2 global forwards + head overhead
        expected_added = 2 * global_fwd
        assert abs(gram_added - expected_added * 1.05) / expected_added < 0.10, \
            f"Gram overhead was {gram_added/1e9:.1f} GFLOPs, expected ~{expected_added*1.05/1e9:.1f}"

    def test_no_local_crops_is_less_than_default(self):
        """Zero local crops should give fewer FLOPs than 8 local crops."""
        no_local = compute_dino_flops_per_image(n_local_crops=0)
        with_local = compute_dino_flops_per_image(n_local_crops=8)
        assert no_local < with_local

    def test_backward_is_2x_student_forward(self):
        """
        Verify the backward=2×forward assumption by decomposing the formula.
        With no local crops, 1 global, no gram, no overhead:
          student_fwd = 1 × global_fwd
          student_bwd = 2 × student_fwd
          teacher_fwd = 1 × global_fwd
          total = student_fwd + student_bwd + teacher_fwd = 4 × global_fwd
        """
        flops = compute_dino_flops_per_image(
            n_global_crops=1, n_local_crops=0, gram_enabled=False, head_overhead_pct=0.0
        )
        global_fwd = vit_forward_flops(197, 768, 12, 4.0)
        expected = 4 * global_fwd  # student_fwd + 2*student_fwd + teacher_fwd
        assert abs(flops - expected) / expected < 0.01, \
            f"Expected {expected/1e9:.2f} GFLOPs, got {flops/1e9:.2f}"

class TestComputeMfu:
    def test_mfu_is_fraction_between_0_and_1(self):
        flops = compute_dino_flops_per_image()
        mfu = compute_mfu(images_per_sec=512, flops_per_image=flops, num_gpus=8)
        assert 0.0 < mfu < 1.0, f"MFU={mfu:.4f} should be in (0, 1)"

    def test_mfu_scales_linearly_with_throughput(self):
        flops = compute_dino_flops_per_image()
        mfu_1x = compute_mfu(512, flops, 8)
        mfu_2x = compute_mfu(1024, flops, 8)
        assert abs(mfu_2x / mfu_1x - 2.0) < 0.01

    def test_mfu_scales_inversely_with_more_gpus(self):
        flops = compute_dino_flops_per_image()
        mfu_8  = compute_mfu(512, flops, 8)
        mfu_16 = compute_mfu(512, flops, 16)
        assert abs(mfu_8 / mfu_16 - 2.0) < 0.01

    def test_floor_estimate_at_expected_baseline(self):
        """
        At 512 img/s (8 GPUs × 64 img/GPU, ~1 s/step), MFU floor ≈ 7%.
        With torch.compile running, real number should be higher.
        This test just checks the formula is not wildly wrong.
        """
        flops = compute_dino_flops_per_image()
        mfu = compute_mfu(512, flops, 8, H100_BF16_TFLOPS)
        assert 0.05 <= mfu <= 0.20, \
            f"Floor MFU at 512 img/s should be ~7%, got {mfu*100:.1f}%"

    def test_perfect_mfu_is_1(self):
        """If images_per_sec = (num_gpus × peak_tflops × 1e12) / flops_per_image, MFU=1."""
        flops = 100e9
        peak = 1000.0  # 1000 TFLOPS per GPU
        gpus = 2
        perfect_ips = (gpus * peak * 1e12) / flops
        mfu = compute_mfu(perfect_ips, int(flops), gpus, peak_tflops=peak)
        assert abs(mfu - 1.0) < 1e-6
```

Run all tests after writing:
```bash
cd /home/adovlatyan/dinov3-performance-optimizations/dinov3
python -m pytest tests/test_mfu.py -v 2>&1
```

All tests must pass before proceeding. If any fail, fix `mfu.py` to match the expected values
from the brief — the tests are the ground truth.

---

## 4. Phase 3 — Integrate MFU into `do_train()` in `train.py`

### 4.1 Read `train.py` lines 414–652 again carefully

Note these key variables already present:
- `global_batch_size` — computed at line 453, total images across all GPUs per step
- `iteration` — the current step count (0-indexed)
- `metric_logger` — `MetricLogger` instance, accepts `metric_logger.update(key=value)` with floats
- `wandb_metrics` dict — built at lines 606–620, logged with `wandb_module.log()`

### 4.2 Changes to make in `train.py`

**Addition 1**: Import at the top of the file (near other imports):
```python
import time
from dinov3.utils.mfu import compute_dino_flops_per_image, compute_mfu
```
(Check if `import time` is already present — if yes, don't add it again.)

**Addition 2**: After `global_batch_size` is computed (after line 453), before the data loader
is built — precompute the FLOPs constant:
```python
flops_per_image = compute_dino_flops_per_image(
    global_crop_size=cfg.crops.global_crops_size,
    local_crop_size=cfg.crops.local_crops_size,
    patch_size=cfg.student.patch_size,
    n_global_crops=2,
    n_local_crops=cfg.crops.local_crops_number,
    hidden_dim=768,
    num_layers=12,
    ffn_ratio=cfg.student.get("ffn_ratio", 4.0),
    n_registers=cfg.student.n_storage_tokens,
    gram_enabled=cfg.gram.use_loss,
    head_overhead_pct=0.05,
)
num_gpus = distributed.get_world_size()
logger.info(f"MFU tracking: {flops_per_image/1e9:.1f} GFLOPs/image, {num_gpus} GPUs")
```

**Addition 3**: Around the core iteration — add CUDA event timing. Place `step_start.record()`
BEFORE `optimizer.zero_grad()` (line 537) and `step_end.record()` AFTER `model.update_ema(mom)`
(line 585). Initialize the events once BEFORE the for loop:

```python
# Before the for loop:
step_start_event = torch.cuda.Event(enable_timing=True)
step_end_event   = torch.cuda.Event(enable_timing=True)
```

Then inside the loop, wrap the core compute:
```python
step_start_event.record()
optimizer.zero_grad(set_to_none=True)
total_loss, metrics_dict = model.forward_backward(data, teacher_temp=teacher_temp, iteration=it)
# ... (grad clip, all-reduce — existing code unchanged) ...
optimizer.step()
model.update_ema(mom)
step_end_event.record()
```

Then after `model.update_ema(mom)`, compute MFU:
```python
# Compute step time and MFU (avoid sync overhead on every step — use CPU timer for logging,
# CUDA event for accuracy when we actually need it)
step_end_event.synchronize()
step_time_ms = step_start_event.elapsed_time(step_end_event)
images_per_sec = global_batch_size / (step_time_ms / 1000.0)
mfu = compute_mfu(images_per_sec, flops_per_image, num_gpus)
```

**Addition 4**: Add to `metric_logger.update()` (near lines 599–604):
```python
metric_logger.update(
    mfu=mfu * 100,           # log as percentage
    images_per_sec=images_per_sec,
    step_time_ms=step_time_ms,
)
```

**Addition 5**: Add to `wandb_metrics` dict (in the `if wandb_run is not None:` block):
```python
wandb_metrics["mfu_pct"] = float(mfu * 100)
wandb_metrics["images_per_sec"] = float(images_per_sec)
wandb_metrics["step_time_ms"] = float(step_time_ms)
```

### 4.3 Concerns and decisions

- **CUDA event sync per iteration**: `step_end_event.synchronize()` blocks the CPU until the GPU
  finishes. This adds a small sync overhead every iteration. This is acceptable for now since we
  need accurate timing. If it hurts throughput significantly (>2%), switch to `time.perf_counter()`
  wall-clock timing instead and note this in the results doc.

- **cfg.student.get("ffn_ratio", 4.0)**: OmegaConf `DictConfig` doesn't have `.get()` by default.
  Use `getattr(cfg.student, 'ffn_ratio', 4.0)` or check if key exists:
  `cfg.student.ffn_ratio if hasattr(cfg.student, 'ffn_ratio') else 4.0`.

- **Placement of CUDA events**: The events must wrap ONLY the compute (zero_grad through update_ema),
  not the logging or all-reduce calls. This gives the GPU compute time, not wall-clock overhead.

---

## 5. Phase 4 — Create Short Validation Slurm Script

Create `scripts/mfu_validation_run.sh` — a minimal 2-GPU, 30-minute run to validate MFU logging.

```bash
#!/bin/bash
#SBATCH --job-name=dinov3-mfu-val
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:2
#SBATCH --time=00:30:00
#SBATCH --output=/data/adovlatyan/logs/mfu-val-%j.out
#SBATCH --error=/data/adovlatyan/logs/mfu-val-%j.err

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /home/akhosrovyan/.conda/envs/dinov3_env

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "Starting MFU validation run"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"

torchrun --nproc_per_node=2 dinov3/train/train.py \
  --config-file dinov3/configs/ssl_default_config.yaml \
  --output-dir ./output_mfu_val_${SLURM_JOB_ID} \
  student.arch=vit_base \
  student.in_chans=5 \
  teacher.in_chans=5 \
  student.pretrained_weights=./pretrained_weights/dinov3_vitb16_pretrain.pth \
  "train.dataset_path=MixedSatelliteDataset:intelinair_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/intelinair/intelinair.h5:maid_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/maid:sen1_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/sentinel1:sen1_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/sentinel1_stats:naip_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas-dataset-v1-naip-2020/naip:naip_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/naip_stats:naip_weight=1.0" \
  train.batch_size_per_gpu=32 \
  train.num_workers=8 \
  train.OFFICIAL_EPOCH_LENGTH=100 \
  optim.epochs=1 \
  train.persistent_workers=false \
  train.prefetch_factor=4 \
  train.cache_dataset=false \
  train.compile=true \
  wandb.enabled=false \
  checkpointing.period=99999
```

**Notes on this script**:
- `OFFICIAL_EPOCH_LENGTH=100` + `epochs=1` = exactly 100 training iterations (~3-5 min of compute)
- `wandb.enabled=false` avoids wandb credentials issues in the validation run
- `checkpointing.period=99999` prevents checkpoint writes during the short run
- `train.compile=true` — keeps torch.compile enabled so MFU matches real production config
- 100 iterations is enough to see MFU stabilize (compile warmup is ~10-20 iters)
- Output dir is `./output_mfu_val_<jobid>/` — unique per run

Submit with:
```bash
cd /home/adovlatyan/dinov3-performance-optimizations/dinov3
sbatch scripts/mfu_validation_run.sh
```

Check submission:
```bash
squeue -u adovlatyan
```

---

## 6. Phase 5 — Monitor and Parse Logs

### 6.1 Watch the job

```bash
# Get the job ID from sbatch output or squeue
JOB_ID=<paste here>

# Tail the output log
tail -f /data/adovlatyan/logs/mfu-val-${JOB_ID}.out

# Or check the local output (if --output is relative)
tail -f slurm-${JOB_ID}.out
```

### 6.2 What to look for in the logs

The MetricLogger prints every 10 iterations (print_freq=10). Look for lines like:
```
Training  [  10/100]  eta: 0:00:20  mfu: 12.34 (11.50)  images_per_sec: 612.5 (598.2)  step_time_ms: 104.5 (107.3)  total_loss: 8.234  ...
```

The format is `current_value (smoothed_avg)` from `SmoothedValue`.

**Expected ranges** (2 GPUs, bs=32/GPU = 64 global batch, torch.compile):
- `images_per_sec`: 100–400 (2 GPUs, so 1/4 of the 8-GPU rate)
- `step_time_ms`: ~150–600 ms per step
- `mfu_pct`: should appear as a %. At 2 GPUs with 100 img/s: ~2-5%. At 200 img/s: ~4-10%.

**Red flags**:
- `mfu_pct` is NaN → timing event issue; switch to wall-clock
- `mfu_pct` is 0.0 → formula bug; debug `flops_per_image`
- `step_time_ms` is wildly fluctuating (>50% variance) → data loading is the bottleneck

### 6.3 Parse the training_metrics.json

After the job, each logged iteration is a JSON line in `output_mfu_val_<jobid>/training_metrics.json`:
```bash
cat output_mfu_val_*/training_metrics.json | python3 -c "
import sys, json
lines = [json.loads(l) for l in sys.stdin if l.strip()]
mfus = [l['mfu'] for l in lines if 'mfu' in l]
ips  = [l.get('images_per_sec', 0) for l in lines if 'images_per_sec' in l]
print(f'MFU: min={min(mfus):.2f}% max={max(mfus):.2f}% avg={sum(mfus)/len(mfus):.2f}%')
print(f'img/s: min={min(ips):.1f} max={max(ips):.1f} avg={sum(ips)/len(ips):.1f}')
print(f'Iterations logged: {len(mfus)}')
"
```

---

## 7. Phase 6 — Validation Criteria

The implementation is correct if ALL of the following hold:

### 7.1 Unit tests pass
```bash
python -m pytest tests/test_mfu.py -v
# Must show: X passed, 0 failed, 0 errors
```

### 7.2 Python smoke test passes
```bash
python -c "
from dinov3.utils.mfu import vit_forward_flops, compute_dino_flops_per_image, compute_mfu
assert 16e9 <= vit_forward_flops(197, 768, 12, 4.0) <= 19e9
assert 200e9 <= compute_dino_flops_per_image() <= 245e9
print('Smoke test PASSED')
"
```

### 7.3 MFU appears in training logs
Slurm log must contain lines with `mfu:` and a non-NaN float value. Check:
```bash
grep "mfu:" /data/adovlatyan/logs/mfu-val-*.out | head -5
# Should show something like: mfu: 8.34 (7.90)
```

### 7.4 MFU values are plausible
After torch.compile warmup (skip first 20 iters), the average MFU across remaining iterations should be:
- At 2 GPUs, bs=32/GPU: > 1% (any non-trivial positive number confirms correctness)
- Should NOT exceed 80% (no ViT-B SSL run achieves near-theoretical on H100)
- Smoothed MFU should be stable (variance < 20% of mean after warmup)

### 7.5 images_per_sec is internally consistent
`images_per_sec ≈ global_batch_size / (step_time_ms / 1000)`.
Since `global_batch_size = 64` (32/GPU × 2 GPUs):
- If `step_time_ms ≈ 320 ms` → `images_per_sec ≈ 200` ✓
- If they disagree by >10%, there's a bug in the timing code.

---

## 8. Phase 7 — Iteration Loop

If the first run reveals problems, iterate in this order:

### Problem: MFU is NaN every iteration
- Root cause: `step_start_event.elapsed_time()` returns NaN if events weren't recorded on the same stream, or if `.synchronize()` was called too early.
- Fix: Switch to wall-clock timing. Replace CUDA events with:
  ```python
  _t_start = time.perf_counter()
  # ... all compute ...
  torch.cuda.synchronize()  # ensure GPU work is done before timing
  step_time_ms = (time.perf_counter() - _t_start) * 1000
  ```

### Problem: MFU is unreasonably high (>60%)
- Root cause: Timing wraps only part of the step (e.g., forgetting to include backward).
- Fix: Ensure `step_start_event.record()` is BEFORE `zero_grad()` and `step_end_event.record()` is AFTER `update_ema()`.

### Problem: MFU is unreasonably low (<1% on 2 GPUs with compile)
- Root cause: `flops_per_image` calculation is wrong, or `global_batch_size` is being passed incorrectly.
- Debug:
  ```python
  logger.info(f"Debug MFU: flops_per_image={flops_per_image:.2e}, global_batch_size={global_batch_size}, step_time_ms={step_time_ms:.1f}")
  ```
- Check the formula: `mfu = (images_per_sec × flops) / (num_gpus × 1979e12)`

### Problem: MFU is very noisy (>50% variance after warmup)
- Root cause: Data loading variability is included in step time.
- This is expected behavior for wall-clock timing — CUDA events give more accurate GPU-only compute time.
- Fix: Ensure CUDA events are used (not wall-clock), and that the event pair wraps only GPU compute.

### Problem: Job fails before producing MFU output
- Check Slurm error log: `cat /data/adovlatyan/logs/mfu-val-<jobid>.err`
- Common issues:
  - Import error for `dinov3.utils.mfu` → PYTHONPATH not set; check script has `export PYTHONPATH=.`
  - Dataset path not found → the akhosrovyan Weka paths might not be accessible to adovlatyan; try `ls /mnt/weka/akhosrovyan/re-id/pretraining/intelinair/` to check
  - Conda env issue → try `conda activate /home/akhosrovyan/.conda/envs/dinov3_env` manually first

### If dataset paths are not accessible:
Create a minimal synthetic dataset test instead. Add a config option or override:
```bash
# Fallback: test with just one small dataset that adovlatyan owns
"train.dataset_path=HDF5Dataset:data_path=/mnt/weka/adovlatyan/some_data.h5"
```
Or if no data is available at all, check if there's a way to run with a synthetic DataLoader
(look in the codebase for any `--dry-run` or synthetic data options).

---

## 9. Phase 8 — Submit Follow-Up Runs (If Time Permits)

After the baseline 2-GPU run succeeds, submit a 4-GPU run to verify MFU scales correctly:

**Expected behavior**: MFU should be approximately the same at 4 GPUs as at 2 GPUs (assuming
the job can sustain the same per-GPU throughput). If MFU drops significantly at 4 GPUs, it
indicates communication overhead — but on a single node with NVLink, this should be minimal.

Create `scripts/mfu_val_4gpu.sh` (copy and modify `mfu_validation_run.sh`):
- Change `--gres=gpu:h100:4`
- Change `torchrun --nproc_per_node=4`
- Change `run_name` to include `4gpu`
- Everything else the same

Submit after the 2-GPU run completes (or while it's running).

---

## 10. Phase 9 — Write Results

Create `docs/mfu-results-<date>.md` with:
- Date and job IDs
- Unit test results (paste pytest output)
- Actual MFU numbers observed (from log parsing)
- Comparison to expected baseline in the brief (Section 9 of the brief doc)
- Whether CUDA events or wall-clock timing was used (and why)
- Any problems encountered and how they were resolved
- Recommended next steps (e.g., profiling, batched multi-crop optimization)

---

## 11. Environment and Execution Notes

### Conda environment
The researchers use: `/home/akhosrovyan/.conda/envs/dinov3_env`
Aram's own env (if different): `~/.conda/envs/dinov3` — check if it exists: `ls ~/.conda/envs/`

If neither is accessible, look for the shared env:
```bash
ls /mnt/weka/shared-cache/miniforge3/envs/ | grep -i dino
```

### Running `pytest` in the right env
```bash
# Activate the right env first:
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /home/akhosrovyan/.conda/envs/dinov3_env
cd /home/adovlatyan/dinov3-performance-optimizations/dinov3
python -m pytest tests/test_mfu.py -v
```

### PYTHONPATH
Always set `export PYTHONPATH=.` from the repo root before running any Python.

### Slurm output location
- Structured logs: `/data/adovlatyan/logs/mfu-val-<jobid>.out`
- If the log path doesn't exist yet: `mkdir -p /data/adovlatyan/logs/`

### Checking job status and output
```bash
squeue -u adovlatyan          # running/pending jobs
sacct -u adovlatyan --format=JobID,JobName,State,Elapsed,ExitCode  # completed jobs
```

---

## 12. Files to Create or Modify (Checklist)

- [ ] `dinov3/utils/mfu.py` — NEW: MFU utility functions
- [ ] `tests/test_mfu.py` — NEW: unit tests (all must pass)
- [ ] `dinov3/train/train.py` — MODIFY: add CUDA event timing + MFU logging in `do_train()`
- [ ] `scripts/mfu_validation_run.sh` — NEW: 2-GPU, 100-iter Slurm validation job
- [ ] `docs/mfu-results-<date>.md` — NEW: results after job completes

**Do not modify**: `dinov3/configs/ssl_default_config.yaml`, `run.sh`, `latest_run.sh`.

---

## 13. Success Definition

The task is COMPLETE when:
1. `python -m pytest tests/test_mfu.py -v` → all green
2. A Slurm job has run and produced a log with `mfu:` values
3. Logged MFU values are plausible (1–60% range, stable after warmup)
4. `docs/mfu-results-<date>.md` exists with actual numbers

At that point, leave the session. Aram will review the results and continue.
