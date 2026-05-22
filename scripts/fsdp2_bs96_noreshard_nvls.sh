#!/bin/bash
# Phase 5 screening: FSDP2 bs=96 with P5-03 + P5-04 stacked.
#   P5-03: train.fsdp_reshard_after_forward=false  (no-release / DDP-like comm)
#   P5-04: NCCL_ALGO=NVLS                           (NVLink SHARP collectives)
#
# Compare against scripts/fsdp2_bs96_baseline.sh (pure ZeRO-3 bs=96).
#
# Notes:
# - Phase 4 (job 26, 2026-04-27) tested reshard_after_forward=false at bs=256 → −0.3 pp
#   vs ZeRO-3 with no memory advantage. At bs=96 the comm/compute ratio differs; retesting.
# - NCCL_ALGO=NVLS targets the all-gather / reduce-scatter dominance we saw in nsys (ch4:620).
#
#SBATCH --job-name=dinov3-fsdp2-bs96-nr-nvls
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/fsdp2-bs96-nr-nvls-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/fsdp2-bs96-nr-nvls-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Sanitize allocator env: expandable_segments:True is DDP-only and hurts FSDP2.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  echo "WARN: PYTORCH_CUDA_ALLOC_CONF contained 'expandable_segments:True' — unsetting for FSDP2."
fi
unset PYTORCH_CUDA_ALLOC_CONF

# P5-04: force NCCL to NVLS (NVLink SHARP) algorithm for collectives.
# DGX H100 + NVSwitch supports this; if it fails, NCCL would normally fall back, but we
# want to detect that explicitly — surface the per-collective algorithm choice in logs.
export NCCL_ALGO=NVLS
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,TUNING

BATCH_SIZE=96
RUN_TAG="fsdp2_bs${BATCH_SIZE}_noreshard_nvls"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 FSDP2 bs=96 P5-03 + P5-04 stacked screening: ${RUN_TAG} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Batch size: ${BATCH_SIZE}"
echo "fsdp_reshard_after_forward: false (P5-03 / no-release)"
echo "NCCL_ALGO: ${NCCL_ALGO}"
echo "NCCL_DEBUG: ${NCCL_DEBUG}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF:-<unset>}"
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
  train.batch_size_per_gpu=${BATCH_SIZE} \
  train.num_workers=20 \
  train.OFFICIAL_EPOCH_LENGTH=500 \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  train.compile=true \
  train.distributed_strategy=fsdp2 \
  train.fsdp_reshard_after_forward=false \
  wandb.enabled=false \
  checkpointing.period=99999

echo "=== ${RUN_TAG} complete: $(date) ==="
