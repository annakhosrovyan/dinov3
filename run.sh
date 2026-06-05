#!/usr/bin/env bash
#SBATCH --job-name=dinov3-satellite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --partition=research
#SBATCH --partition=research
#SBATCH --time=7-00:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/dinov3-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/dinov3-%j.err

# Production config — revised 2026-05-12 (Phase 5)
# ==================================================
#   Strategy:        FSDP2 (ZeRO-3 per-block, `reshard_after_forward=True` default)
#   Batch / GPU:     96   (bs=128 OOM'd in a real long training run; bs=96 is the current safe ceiling)
#   ES:              OFF  (`expandable_segments:True` is DDP-only and hurts FSDP2)
#
# Rollback note: the previous default was DDP + expandable_segments + bs=256 (short 500-iter
# screening at ~23.9% MFU). That config was retired because:
#   (a) DDP is closed as a path forward (FSDP2 has an equivalent for every DDP operating
#       point per Tim Darcet, DINOv2/v3 co-author);
#   (b) bs=256 / bs=192 are NOT achievable in long training runs (researcher report);
#   (c) bs=128 FSDP2 also OOM'd in a real long training run — root cause open (see
#       docs/phase5_perf_plan.md §10, P5-OOM).
# To restore the old DDP+ES+bs=256 config for screening only:
#   - export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   - train.distributed_strategy=ddp, train.batch_size_per_gpu=256
#   - DO NOT use for a real long training run until the FSDP2 OOM is understood.

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Sanitize allocator env for FSDP2: `expandable_segments:True` is DDP-only and
# hurts FSDP2. If the submit shell or job env inherited it (e.g. from a prior
# DDP+ES screening), drop it here so a long FSDP2 run cannot silently inherit
# the wrong allocator config.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  echo "WARN: PYTORCH_CUDA_ALLOC_CONF contained 'expandable_segments:True' — unsetting for FSDP2 run."
fi
unset PYTORCH_CUDA_ALLOC_CONF

# NCCL — no env overrides until P5-04 NVLS screening lands. NCCL auto-selects algorithm
# per topology; on this DGX H100 + NVSwitch node that usually means NVLS for all-gather /
# reduce-scatter. Use `NCCL_DEBUG=INFO` in a probe run to confirm.

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 Satellite Training ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Date: $(date)"
echo "Config: FSDP2 ZeRO-3 + bs=96 + sharded_eval_checkpoint"
echo "--- Effective env (perf-critical) ---"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-<unset>}"
echo "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-<unset>}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>} MKL_NUM_THREADS=${MKL_NUM_THREADS:-<unset>}"
echo "NCCL_ALGO=${NCCL_ALGO:-<unset>} NCCL_DEBUG=${NCCL_DEBUG:-<unset>} NCCL_NTHREADS=${NCCL_NTHREADS:-<unset>} NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-<unset>}"
echo "-------------------------------------"

torchrun --nproc_per_node=8 dinov3/train/train.py \
  --config-file dinov3/configs/ssl_default_config.yaml \
  --output-dir /mnt/weka/adovlatyan/output_satellite_${SLURM_JOB_ID} \
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
  train.batch_size_per_gpu=96 \
  "train.dataset_path=MixedSatelliteDataset:\
intelinair_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/intelinair/intelinair.h5:\
maid_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/maid:\
sen1_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/sentinel1:\
sen1_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/sentinel1_stats:\
naip_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas-dataset-v1-naip-2020/naip:\
naip_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/naip_stats:\
naip_weight=1.0" \
  train.batch_size_per_gpu=96 \
  train.num_workers=20 \
  train.OFFICIAL_EPOCH_LENGTH=23412 \
  optim.epochs=10 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  train.compile=true \
  train.distributed_strategy=fsdp2 \
  train.fsdp_reshard_after_forward=true \
  train.sharded_eval_checkpoint=true \
  train.compile=true \
  train.distributed_strategy=fsdp2 \
  train.fsdp_reshard_after_forward=true \
  train.sharded_eval_checkpoint=true \
  wandb.enabled=true \
  wandb.project=dinov3-satellite \
  wandb.run_name=satellite_fsdp2_bs96_${SLURM_JOB_ID} \
  wandb.group=satellite_fsdp2

echo "=== Training complete: $(date) ==="
  wandb.run_name=satellite_fsdp2_bs96_${SLURM_JOB_ID} \
  wandb.group=satellite_fsdp2

echo "=== Training complete: $(date) ==="
