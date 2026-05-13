#!/bin/bash
# FSDP2 no-release + expandable_segments screening — bs=256.
# Tests whether ES is neutral/helpful with no-release FSDP2 (unlike ZeRO-3 where ES hurts 7-12%).
# With reshard_after_forward=false, per-block allgather overhead is gone, so ES allocator pressure
# should be much lower. See pytorch/pytorch#137151 for the ZeRO-3 issue.
#
# Usage: sbatch scripts/screening_fsdp2_norelease_es.sh [batch_size]
# Default: bs=256
#
#SBATCH --job-name=dinov3-fsdp2-nr-es
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/fsdp2-norelease-es-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/fsdp2-norelease-es-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BATCH_SIZE="${1:-256}"

RUN_TAG="fsdp2_norelease_es_bs${BATCH_SIZE}"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 FSDP2 no-release + expandable_segments screening: ${RUN_TAG} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Batch size: ${BATCH_SIZE}"
echo "fsdp_reshard_after_forward: false (no-release / DDP-like)"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "Date: $(date)"

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
  train.distributed_strategy=fsdp2 \
  train.fsdp_reshard_after_forward=false \
  wandb.enabled=false \
  checkpointing.period=99999

echo "=== FSDP2 no-release+ES ${RUN_TAG} complete: $(date) ==="
