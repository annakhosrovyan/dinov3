#!/bin/bash
# Phase 6.A.4.d — DDP bs=192, no AC, fullgraph+triton.cudagraphs=True.
#
# THE ONE THING THIS RUN IS TESTING:
#   Job 53708 (bs=128) used 34.1 GB / 80 GB. Linear extrapolation: bs=192 ≈ 51 GB.
#   Does pushing batch further keep amortizing per-step overhead, or do we run into
#   a different ceiling (NCCL bandwidth, scheduler jitter, loader saturation)?
#
# Comparators:
#   - 48312 baseline:  bs=96,  cg=false → 1,387 img/s,  7.94% MFU, 545 ms, 25.8 GB
#   - 53681:           bs=96,  cg=true  → 2,009 img/s, 11.50% MFU, 379 ms, ? GB
#   - 53708:           bs=128, cg=true  → 2,394 img/s, 13.70% MFU, ~390 ms, 34.1 GB
#
# Expected: img/s gain smaller than 96→128 step (compute now dominates step time).
# If gain < 5%, the bottleneck has fully shifted to per-step compute/comms, and the
# next lever is Phase 6.B (extend fullgraph to heads, static-shape iBOT).
#
#SBATCH --job-name=dinov3-ddp-bs192-cudagraphs
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/ddp-bs192-cudagraphs-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/ddp-bs192-cudagraphs-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

unset PYTORCH_CUDA_ALLOC_CONF
unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

BATCH_SIZE=192
RUN_TAG="ddp_bs${BATCH_SIZE}_cudagraphs"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 DDP bs=192 fullgraph+triton.cudagraphs: ${RUN_TAG} ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "Strategy:  DDP, compile=true, cudagraphs=true, AC=off"
echo "Batch size: ${BATCH_SIZE}  (global = ${BATCH_SIZE} * 8 = $((BATCH_SIZE * 8)))"
echo "Comparator: job 53708 (same recipe, bs=128) — 2394 img/s, 13.70% MFU, 34.1 GB"
echo "Predicted peak alloc: ~51 GB (linear from bs=128)"
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
  train.OFFICIAL_EPOCH_LENGTH=1000 \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  train.compile=true \
  train.cudagraphs=true \
  train.distributed_strategy=ddp \
  train.checkpointing=false \
  train.sharded_eval_checkpoint=true \
  checkpointing.period=400 \
  evaluation.eval_period_iterations=400 \
  wandb.enabled=false

echo ""
echo "=== ${RUN_TAG} complete: $(date) ==="
grep "Training" /mnt/weka/adovlatyan/logs/ddp-bs192-cudagraphs-${SLURM_JOB_ID}.out \
  | awk 'NR>20 {print}' | tail -20 || true
echo ""
grep "\[COMPILE\]" /mnt/weka/adovlatyan/logs/ddp-bs192-cudagraphs-${SLURM_JOB_ID}.out || true
