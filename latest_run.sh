#!/bin/bash -l
#SBATCH --job-name=dinov3-sweep
#SBATCH --nodes=1
#SBATCH --partition=all
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=0-5

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /home/akhosrovyan/.conda/envs/dinov3_env

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -------------------------------
# Sweep definitions
# Format per run:
#   NUM_WORKERS PREFETCH_FACTOR BATCH_SIZE_PER_GPU
# -------------------------------
CONFIGS=(
  "8 2 64"
  "12 2 64"
  "16 2 64"
  "12 4 64"
  "12 2 96"
  "16 4 64"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r NUM_WORKERS PREFETCH_FACTOR BATCH_SIZE <<< "$CONFIG"

RUN_NAME="nw${NUM_WORKERS}_pf${PREFETCH_FACTOR}_bs${BATCH_SIZE}_1h"

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "CONFIG=${CONFIG}"
echo "RUN_NAME=${RUN_NAME}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "PREFETCH_FACTOR=${PREFETCH_FACTOR}"
echo "BATCH_SIZE=${BATCH_SIZE}"

torchrun --nproc_per_node=8 dinov3/train/train.py \
  --config-file dinov3/configs/ssl_default_config.yaml \
  --output-dir "./output_${RUN_NAME}" \
  student.arch=vit_base \
  student.in_chans=5 \
  teacher.in_chans=5 \
  student.pretrained_weights=./pretrained_weights/dinov3_vitb16_pretrain.pth \
  "train.dataset_path=MixedSatelliteDataset:intelinair_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/intelinair/intelinair.h5:maid_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/maid:sen1_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/sentinel1:sen1_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/sentinel1_stats:naip_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas-dataset-v1-naip-2020/naip:naip_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/naip_stats:naip_weight=1.0" \
  train.batch_size_per_gpu="${BATCH_SIZE}" \
  train.num_workers="${NUM_WORKERS}" \
  train.OFFICIAL_EPOCH_LENGTH=23412 \
  optim.epochs=10 \
  train.persistent_workers=true \
  train.prefetch_factor="${PREFETCH_FACTOR}" \
  train.cache_dataset=false \
  wandb.enabled=true \
  wandb.project=dinov3-satellite \
  wandb.run_name="${RUN_NAME}" \
  wandb.group=satellite_sweep