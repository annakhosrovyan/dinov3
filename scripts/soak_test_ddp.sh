#!/bin/bash
# Long-run soak test for DDP + expandable_segments + bs=256.
#
# Purpose: validate that the best-known DDP config does not accumulate memory
# over hundreds of iterations (no slow fragmentation creep, no late OOM).
# Complements memprofile_ddp.sh (which checks per-phase peak memory in 70 iters)
# by running many more iterations across multiple eval + checkpoint cycles.
#
# The memprofile scripts confirm worst-case-per-phase peak memory.
# This script confirms that those peaks don't grow iteration over iteration.
#
# Memory profile output is enabled. Extract with:
#   grep '\[MEMPROFILE\]' <logfile> | grep 'rank=0'
#
# Usage: sbatch scripts/soak_test_ddp.sh [batch_size] [expand_segments]
# Example:
#   sbatch scripts/soak_test_ddp.sh 256 true     # recommended config
#   sbatch scripts/soak_test_ddp.sh 128 true     # conservative config
#
#SBATCH --job-name=dinov3-soak-ddp
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/soak-ddp-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/soak-ddp-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Enable per-phase memory profiling instrumentation in train.py
export DINOV3_MEMORY_PROFILE=1

BATCH_SIZE="${1:-256}"
EXPAND_SEG="${2:-true}"

if [ "${EXPAND_SEG}" = "true" ]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

RUN_TAG="soak_ddp_bs${BATCH_SIZE}_es${EXPAND_SEG}"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 DDP Soak Test: ${RUN_TAG} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Batch size: ${BATCH_SIZE}"
echo "Expand segments: ${EXPAND_SEG}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF:-not set}"
echo "Output: ${OUTPUT_DIR}"
echo "Date: $(date)"
echo ""
echo "Phase schedule (500 iters, 1 epoch):"
echo "  Iters   0-9:  torch.compile warmup"
echo "  Iters  10-99: steady state (batch 1)"
echo "  Iter   100:   eval phase fires (first)"
echo "  Iter   200:   checkpoint fires (first)"
echo "  Iter   200:   eval phase fires (second)"
echo "  Iter   300:   eval phase fires (third)"
echo "  Iter   400:   checkpoint fires (second) + eval (fourth)"
echo "  Iters 400-499: final steady state"
echo ""
echo "If max_reserved_mb grows monotonically across [MEMPROFILE] checkpoints,"
echo "there is memory creep. Stable peaks = production-safe."
echo ""

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
  train.OFFICIAL_EPOCH_LENGTH=500 \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  train.compile=true \
  train.distributed_strategy=ddp \
  train.sharded_eval_checkpoint=true \
  wandb.enabled=false \
  evaluation.eval_period_iterations=100 \
  checkpointing.period=200

echo ""
echo "=== ${RUN_TAG} complete: $(date) ==="
echo ""
echo "Check for memory creep:"
echo "  grep '\[MEMPROFILE\]' /mnt/weka/adovlatyan/logs/soak-ddp-${SLURM_JOB_ID}.out | grep 'rank=0'"
echo ""
echo "Steady-state MFU (skip first 20 iters):"
echo "  grep 'mfu_pct' /mnt/weka/adovlatyan/logs/soak-ddp-${SLURM_JOB_ID}.out | awk 'NR>20'"
