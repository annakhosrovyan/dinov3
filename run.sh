#!/usr/bin/env bash
#SBATCH --job-name=dinov3-satellite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --partition=research
#SBATCH --time=7-00:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/dinov3-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/dinov3-%j.err

# Best validated config (2026-04-07):
#   DDP + expandable_segments + bs=256 → 24.5% MFU, 4229 img/s (vs 11.3% FSDP2/bs=64 baseline)
#   expandable_segments eliminates allocator fragmentation stalls in the compiled DDP path.
#   sharded_eval_checkpoint avoids full_tensor() materialization during eval phases.
#   Soak-test (long-run memory stability) is in progress — treat this as a validated candidate.

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# DDP + expandable_segments: fixes allocator fragmentation stalls at bs=256 in compiled DDP path.
# NOTE: this setting hurts FSDP2 — only set it when using distributed_strategy=ddp.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 Satellite Training ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Date: $(date)"
echo "Config: DDP + expandable_segments + bs=256 + sharded_eval_checkpoint"

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
  train.batch_size_per_gpu=256 \
  train.num_workers=20 \
  train.OFFICIAL_EPOCH_LENGTH=23412 \
  optim.epochs=10 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  train.compile=true \
  train.distributed_strategy=ddp \
  train.sharded_eval_checkpoint=true \
  wandb.enabled=true \
  wandb.project=dinov3-satellite \
  wandb.run_name=satellite_ddp_bs256_${SLURM_JOB_ID} \
  wandb.group=satellite_ddp

echo "=== Training complete: $(date) ==="
