#!/bin/bash
# Phase 6.A.4.a — DDP bs=128, no AC, torch.compile(fullgraph=True, triton.cudagraphs=True).
#
# THE ONE THING THIS RUN IS TESTING:
#   Does pushing batch from 96 → 128 (with the same fullgraph+cudagraphs path that won in
#   job 53681) further raise throughput, without hitting OOM on H100 80GB?
#
# Comparators:
#   - Job 48312: DDP bs=96, cudagraphs=false  → 1,387 img/s,  7.94% MFU, 545 ms, 25.8 GB
#   - Job 53681: DDP bs=96, cudagraphs=true   → 2,009 img/s, 11.50% MFU, 379 ms, peak alloc TBD
#   - Linear bs scaling from 53681 peak alloc: bs=128 ≈ 34 GB expected (rough; cudagraph
#     workspaces also live in allocator, so add some slack). 80 GB / GPU = plenty of room
#     if linear, but cudagraph trees can hold multiple sub-graphs concurrently.
#
# Hypothesis:
#   Larger batch amortizes the fixed per-step Python / launch overhead — the same lever
#   cudagraphs targets, via a different mechanism. Expect modest +10-20% img/s on top of
#   53681 if memory holds.
#
# If this OOMs: that's the answer to Phase 6.A.4.c (OOM control). Re-queue with AC.
#
#SBATCH --job-name=dinov3-ddp-bs128-cudagraphs
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/ddp-bs128-cudagraphs-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/ddp-bs128-cudagraphs-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# No expandable_segments — matching job 53681 exactly.
unset PYTORCH_CUDA_ALLOC_CONF

# No NCCL knobs — clean baseline.
unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

BATCH_SIZE=128
RUN_TAG="ddp_bs${BATCH_SIZE}_cudagraphs"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 DDP bs=128 fullgraph+triton.cudagraphs: ${RUN_TAG} ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "Strategy:  DDP, compile=true, cudagraphs=true, AC=off"
echo "Compile path (backbone): fullgraph=True, dynamic=False, triton.cudagraphs=True"
echo "Compile path (heads):    module.compile() default (dynamic=True)"
echo "Batch size: ${BATCH_SIZE}"
echo "Comparator: job 53681 (same recipe, bs=96) — 2009 img/s, 11.50% MFU"
echo "Date: $(date)"
echo ""
echo "Verify compile path from log:"
echo "  grep '\[COMPILE\]' \${OUTPUT_DIR}/training.log | head -10"

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
echo "--- Steady-state MFU (skip first ~20 iters for compile warmup) ---"
grep "Training" /mnt/weka/adovlatyan/logs/ddp-bs128-cudagraphs-${SLURM_JOB_ID}.out \
  | awk 'NR>20 {print}' | tail -20 || true
echo ""
echo "--- [COMPILE] path confirmation ---"
grep "\[COMPILE\]" /mnt/weka/adovlatyan/logs/ddp-bs128-cudagraphs-${SLURM_JOB_ID}.out || true
