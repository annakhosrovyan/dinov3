#!/bin/bash
#SBATCH --job-name=dinov3-ddp-cg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --partition=research
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-ddp-%j.out
#SBATCH --error=slurm-ddp-%j.err

# Production training recipe — DDP + torch.compile + CUDA graphs (bs=128)
# ======================================================================
# This is the validated high-throughput path for single-node 8xH100 training.
# It is the sibling of run.sh: run.sh stays on the conservative FSDP2 bs=96
# default; this script runs the faster DDP + cudagraphs config.
#
#   Strategy:   DDP (single all-reduce, no parameter sharding)
#   Batch/GPU:  128
#   Compile:    torch.compile (compile_mode=null) + train.cudagraphs=true
#   AC:         OFF (activation checkpointing not needed — model fits at bs=128)
#
# WHY THIS CONFIG — provenance:
#   - Screen (job 53708, 1000 iters): 2,394 img/s, 13.70% MFU, 34.1 GB peak VRAM.
#   - Sustainability soak (job 60959, 4000 iters): completed 4000/4000 with no
#     crash / OOM / NaN / worker restart. Honest sustained throughput was lower
#     than the short screen — ~1,830 img/s run-average, ~2,055 img/s late-stage —
#     and VRAM stayed flat. This is the number to expect for a real long run.
#   - For ViT-B (~85M params) FSDP2's sharding is overkill; DDP + fullgraph compile
#     is the throughput winner. (Tim Darcet / Armen Aghajanyan guidance, 2026-05.)
#
# IMPORTANT OPERATIONAL CAVEAT:
#   With train.cache_dataset=true, host RAM drifts upward over the run (page-cache
#   + worker RSS). Do NOT impose a tight Slurm --mem cap here, or a long run can be
#   OOM-killed at the host level. This script deliberately sets no --mem.
#
# TUNABLES (env-overridable so Anna can size the run without editing the script):
#   OFFICIAL_EPOCH_LENGTH  iters per epoch  (default 23412 — full dataset)
#   EPOCHS                 number of epochs (default 10)
#   BATCH_SIZE             per-GPU batch    (default 128)
#   CKPT_PERIOD            save every N     (default 3750, config default)
#   MAX_TO_KEEP            checkpoints kept (default 3, config default)
#   EVAL_PERIOD            eval every N     (default 12500, config default)
#   To shrink for a quick functional check: OFFICIAL_EPOCH_LENGTH=50 EPOCHS=1 sbatch run_ddp.sh
#
# PORTABILITY (env-overridable so a different user need not edit this file):
#   DINOV3_ENV          conda env prefix — REQUIRED, no default (set it to YOUR env)
#   DINOV3_OUTPUT_ROOT  root for outputs (default /mnt/weka/$(whoami))
#   student.pretrained_weights below points at Anna's weights; override on CLI if needed.
#   (The #SBATCH --output/--error lines can't read env vars — they land in the
#    submit dir as slurm-ddp-<jobid>.out; edit those two lines if you want them elsewhere.)
#
# Optional profiling (off by default, zero-overhead when unset):
#   export DINOV3_MEMORY_PROFILE=1          # per-phase + periodic VRAM markers
#   export DINOV3_MEMORY_PROFILE_PERIOD=50  # fragmentation log cadence

# >>> SET YOUR CONDA ENV (torch >= 2.6) <<<
#   export DINOV3_ENV=/home/<you>/.conda/envs/<your-env>   before sbatch,
#   or hardcode your path on the line below. Activated via PATH-prepend (NOT
#   `conda activate`, which fails on the bare-metal GPU nodes). Fails fast if unset.
DINOV3_ENV="${DINOV3_ENV:?Set DINOV3_ENV to your conda env prefix (torch>=2.6), e.g. /home/$(whoami)/.conda/envs/dinov3}"
export PATH="${DINOV3_ENV}/bin:$PATH"
export CONDA_PREFIX="${DINOV3_ENV}"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# `expandable_segments:True` is a DDP+very-large-batch knob; it is not needed at
# bs=128 and can perturb allocator behavior. Drop any inherited value so this run
# uses the default caching allocator that the soak was validated with.
unset PYTORCH_CUDA_ALLOC_CONF
# Let NCCL auto-select its algorithm for this DGX H100 + NVSwitch topology; clear
# any inherited overrides from prior screening runs.
unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

OFFICIAL_EPOCH_LENGTH="${OFFICIAL_EPOCH_LENGTH:-23412}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CKPT_PERIOD="${CKPT_PERIOD:-3750}"
MAX_TO_KEEP="${MAX_TO_KEEP:-3}"
EVAL_PERIOD="${EVAL_PERIOD:-12500}"
# Output root auto-derives the submitting user ($(whoami)) so this file has no
# hardcoded username; override with DINOV3_OUTPUT_ROOT to point at any storage
# you own. (The #SBATCH --output/--error lines above can't read env vars — Slurm
# parses them before the shell runs — so they land relative to the submit dir.)
OUTPUT_ROOT="${DINOV3_OUTPUT_ROOT:-/mnt/weka/$(whoami)}"
OUTPUT_DIR="${OUTPUT_ROOT}/output_ddp_cg_${SLURM_JOB_ID}"

mkdir -p "${OUTPUT_ROOT}"

echo "=== DINOv3 Satellite Training — DDP + CUDA graphs ==="
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     ${SLURM_NODELIST}"
echo "Config:   DDP, bs=${BATCH_SIZE}, compile=true, cudagraphs=true, AC=off"
echo "Schedule: OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH}, epochs=${EPOCHS}"
echo "Ckpt:     period=${CKPT_PERIOD}, max_to_keep=${MAX_TO_KEEP}; eval period=${EVAL_PERIOD}"
echo "Output:   ${OUTPUT_DIR}"
echo "Date:     $(date)"
echo "--- Effective env (perf-critical) ---"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-<unset>}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>} MKL_NUM_THREADS=${MKL_NUM_THREADS:-<unset>}"
echo "DINOV3_MEMORY_PROFILE=${DINOV3_MEMORY_PROFILE:-<unset>}"
echo "-------------------------------------"

torchrun --nproc_per_node=8 dinov3/train/train.py \
  --config-file dinov3/configs/ssl_default_config.yaml \
  --output-dir "${OUTPUT_DIR}" \
  student.arch=vit_base \
  student.in_chans=5 \
  teacher.in_chans=5 \
  student.pretrained_weights=/auto/home/anna.khosrovyan/dinov3/pretrained_weights/dinov3_vitb16_pretrain.pth \
  "train.dataset_path=MixedSatelliteDataset:\
intelinair_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/intelinair/intelinair.h5:\
maid_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/maid:\
sen1_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/sentinel1:\
sen1_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/sentinel1_stats:\
naip_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas-dataset-v1-naip-2020/naip:\
naip_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/naip_stats:\
naip_weight=1.0" \
  train.batch_size_per_gpu="${BATCH_SIZE}" \
  train.OFFICIAL_EPOCH_LENGTH="${OFFICIAL_EPOCH_LENGTH}" \
  optim.epochs="${EPOCHS}" \
  train.num_workers=20 \
  train.prefetch_factor=8 \
  train.persistent_workers=true \
  train.cache_dataset=true \
  train.compile=true \
  train.cudagraphs=true \
  train.checkpointing=false \
  train.checkpointing_full=false \
  train.distributed_strategy=ddp \
  train.sharded_eval_checkpoint=true \
  checkpointing.period="${CKPT_PERIOD}" \
  checkpointing.max_to_keep="${MAX_TO_KEEP}" \
  evaluation.eval_period_iterations="${EVAL_PERIOD}" \
  wandb.enabled=true \
  wandb.project=dinov3-satellite \
  wandb.run_name=satellite_ddp_cg_bs${BATCH_SIZE}_${SLURM_JOB_ID} \
  wandb.group=satellite_ddp_cg

echo "=== Training complete: $(date) ==="
