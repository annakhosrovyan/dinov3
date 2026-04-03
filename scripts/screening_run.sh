#!/bin/bash
# Parametric screening script for Step 5.
# Usage: sbatch scripts/screening_run.sh <batch_size> <cudagraphs> <checkpointing>
# Example: sbatch scripts/screening_run.sh 128 false false
#
# Parameters:
#   $1 = batch_size_per_gpu (default: 64)
#   $2 = cudagraphs (true/false, default: false)
#   $3 = checkpointing (true/false, default: false)
#
#SBATCH --job-name=dinov3-screen
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/screen-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/screen-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Parse parameters (with defaults)
BATCH_SIZE="${1:-64}"
CUDAGRAPHS="${2:-false}"
CHECKPOINTING="${3:-false}"

RUN_TAG="bs${BATCH_SIZE}_cg${CUDAGRAPHS}_ckpt${CHECKPOINTING}"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_screen_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 Screening Run: ${RUN_TAG} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Batch size: ${BATCH_SIZE}"
echo "CUDA graphs: ${CUDAGRAPHS}"
echo "Checkpointing: ${CHECKPOINTING}"
echo "Output: ${OUTPUT_DIR}"
echo "Date: $(date)"

# 100 iters: ~35s compile warmup on iter 0, then ~70 stable iters for measurement
# Shorter than baseline 300 iters since we're screening, not benchmarking
torchrun --nproc_per_node=8 dinov3/train/train.py \
  --config-file dinov3/configs/ssl_default_config.yaml \
  --output-dir "${OUTPUT_DIR}" \
  student.arch=vit_base \
  student.in_chans=5 \
  teacher.in_chans=5 \
  student.pretrained_weights="" \
  "train.dataset_path=MixedSatelliteDataset:\
intelinair_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/intelinair/intelinair.h5:\
maid_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/maid:\
sen1_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/sentinel1:\
sen1_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/sentinel1_stats:\
naip_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas-dataset-v1-naip-2020/naip:\
naip_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/naip_stats:\
naip_weight=1.0" \
  train.batch_size_per_gpu="${BATCH_SIZE}" \
  train.num_workers=20 \
  train.OFFICIAL_EPOCH_LENGTH=100 \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  train.compile=true \
  train.cudagraphs="${CUDAGRAPHS}" \
  train.checkpointing="${CHECKPOINTING}" \
  wandb.enabled=false \
  checkpointing.period=99999

echo "=== Screening ${RUN_TAG} complete: $(date) ==="
