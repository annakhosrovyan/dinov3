#!/bin/bash
# Phase 5 P5-AC-selective: FSDP2 bs=128 with selective activation checkpointing.
#
# Goals (per docs/phase5_perf_plan.md §10c):
#   1. Measure activation memory drop from selective AC at bs=128
#   2. Measure MFU cost (expected: a few pp from recompute on unsaved ops)
#   3. Capture memory pattern around the COINCIDENT eval + checkpoint boundary
#      (the failure regime documented in project_bs128_fsdp2_oom.md)
#
# Method:
#   - train.checkpointing=true, train.checkpointing_full=false  (Meta ViT-7B recipe)
#   - checkpointing.period=400, evaluation.eval_period_iterations=400
#       → eval AND checkpoint both fire at iter 400 and 800 → combined-phase pressure
#   - OFFICIAL_EPOCH_LENGTH=1000  → boundary hit twice → A/B replication
#   - DINOV3_MEMORY_PROFILE=1     → [MEMPROFILE] lines at every phase boundary
#   - DINOV3_MEMORY_PROFILE_PERIOD=10 → [MEMFRAG] line every 10 iters for fragmentation tracking
#
# The codebase already has gc.disable() + manual gc.collect() every 150 iters
# (train.py:526, 588) — so we are NOT testing P5-05 (gc.disable) here.
#
#SBATCH --job-name=dinov3-fsdp2-bs128-ac-sel
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/fsdp2-bs128-ac-sel-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/fsdp2-bs128-ac-sel-%j.err

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

# Memory profiling — emits [MEMPROFILE] / [MEMFRAG] lines for grep analysis.
export DINOV3_MEMORY_PROFILE=1
export DINOV3_MEMORY_PROFILE_PERIOD=10

BATCH_SIZE=128
RUN_TAG="fsdp2_bs${BATCH_SIZE}_ac_selective"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 FSDP2 bs=128 + selective AC screening: ${RUN_TAG} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Batch size: ${BATCH_SIZE}"
echo "AC: train.checkpointing=true, train.checkpointing_full=false (selective; Meta ViT-7B recipe)"
echo "Aligned eval + ckpt at period=400 → combined-phase pressure at iter 400 and 800"
echo "OFFICIAL_EPOCH_LENGTH=1000 (A/B replication of phase boundary)"
echo "Memory profile: DINOV3_MEMORY_PROFILE=1, DINOV3_MEMORY_PROFILE_PERIOD=10"
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
  train.OFFICIAL_EPOCH_LENGTH=1000 \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  train.compile=true \
  train.distributed_strategy=fsdp2 \
  train.fsdp_reshard_after_forward=true \
  train.checkpointing=true \
  train.checkpointing_full=false \
  train.sharded_eval_checkpoint=true \
  checkpointing.period=400 \
  evaluation.eval_period_iterations=400 \
  wandb.enabled=false

echo "=== ${RUN_TAG} complete: $(date) ==="
echo "--- [MEMPROFILE] summary (rank 0) ---"
grep "MEMPROFILE.*rank=0" /mnt/weka/adovlatyan/logs/fsdp2-bs128-ac-sel-${SLURM_JOB_ID}.out | tail -20 || true
echo "--- [MEMFRAG] summary (rank 0, last 5) ---"
grep "MEMFRAG.*rank=0" /mnt/weka/adovlatyan/logs/fsdp2-bs128-ac-sel-${SLURM_JOB_ID}.out | tail -5 || true
