#!/bin/bash
# Compile-mode screening: benchmarks "default" vs "max-autotune-no-cudagraphs".
#
# max-autotune-no-cudagraphs enables Triton kernel autotuning (tile sizes, warp config, etc.)
# without CUDA graph capture. CUDA graphs are confirmed broken on the multi-crop architecture
# (see learnings/cuda_graphs.md). This is the safe autotuning path.
#
# IMPORTANT — max-autotune Triton OOM issue (2026-04-07, job 16720):
#   When all 8 ranks autotune concurrently during iter-0 compile, each rank benchmarks every
#   kernel variant with temporary allocations on top of the model. Combined per-GPU pressure
#   OOMs before training even starts (SIGABRT on rank 2 after 18 min).
#
#   Fix: single-rank warmup (Phase 1) populates ~/.cache/torch/inductor with autotuned kernels.
#   The 8-rank run (Phase 2) then finds the cache and skips benchmarking entirely. Cache entries
#   are keyed on (op shape, hardware, PyTorch version). Since per-GPU batch size is identical
#   between nproc=1 and nproc=8, cache entries are valid for both.
#
# Usage: sbatch scripts/screening_compile_modes.sh <mode> [batch_size]
#   mode: "default" or "max-autotune-no-cudagraphs"
#   batch_size: per-GPU batch size, default 128
# Example:
#   sbatch scripts/screening_compile_modes.sh default 128
#   sbatch scripts/screening_compile_modes.sh max-autotune-no-cudagraphs 128
#
#SBATCH --job-name=dinov3-compile-mode
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/compile-mode-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/compile-mode-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMPILE_MODE="${1:-default}"
BATCH_SIZE="${2:-128}"

if [ "${COMPILE_MODE}" = "default" ]; then
    COMPILE_MODE_ARG="train.compile_mode=null"
    SAFE_TAG="default"
    NEEDS_WARMUP=false
else
    COMPILE_MODE_ARG="train.compile_mode=${COMPILE_MODE}"
    SAFE_TAG="${COMPILE_MODE//[^a-zA-Z0-9_-]/}"
    NEEDS_WARMUP=true
fi

RUN_TAG="compile_mode_${SAFE_TAG}_bs${BATCH_SIZE}"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"
WARMUP_OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_warmup_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 Compile Mode Screening: ${RUN_TAG} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Compile mode: ${COMPILE_MODE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Needs warmup: ${NEEDS_WARMUP}"
echo "Triton cache: ${HOME}/.cache/torch/inductor"
echo "Output: ${OUTPUT_DIR}"
echo "Date: $(date)"
echo ""

# Shared torchrun args
COMMON_ARGS=(
  --config-file dinov3/configs/ssl_default_config.yaml
  student.arch=vit_base
  student.in_chans=5
  teacher.in_chans=5
  student.pretrained_weights=""
  "train.dataset_path=MixedSatelliteDataset:\
intelinair_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/intelinair/intelinair.h5:\
maid_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/maid:\
sen1_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/sentinel1:\
sen1_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/sentinel1_stats:\
naip_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas-dataset-v1-naip-2020/naip:\
naip_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/naip_stats:\
naip_weight=1.0"
  train.batch_size_per_gpu="${BATCH_SIZE}"
  train.num_workers=4
  train.persistent_workers=false
  train.prefetch_factor=2
  train.cache_dataset=false
  train.compile=true
  train.distributed_strategy=ddp
  train.sharded_eval_checkpoint=true
  "${COMPILE_MODE_ARG}"
  wandb.enabled=false
  checkpointing.period=99999
  evaluation.eval_period_iterations=99999
)

# Phase 1: single-rank Triton cache warmup (only for max-autotune variants)
if [ "${NEEDS_WARMUP}" = "true" ]; then
    echo "--- Phase 1: Single-rank Triton cache warmup (GPU 0 only) ---"
    echo "Populating ~/.cache/torch/inductor for ${COMPILE_MODE}..."
    echo "Start: $(date)"

    # Use only GPU 0, single rank — same per-GPU batch size, same tensor shapes.
    # This causes Triton to benchmark and cache all kernel variants without competing
    # with 7 other ranks for memory.
    CUDA_VISIBLE_DEVICES=0 torchrun \
      --nproc_per_node=1 \
      --master_port=29501 \
      dinov3/train/train.py \
      --output-dir "${WARMUP_OUTPUT_DIR}" \
      "${COMMON_ARGS[@]}" \
      train.OFFICIAL_EPOCH_LENGTH=2 \
      optim.epochs=1

    echo "Phase 1 complete: $(date)"
    echo "Triton cache entries written to: ${HOME}/.cache/torch/inductor"
    echo ""
fi

# Phase 2: full 8-rank screening run
echo "--- Phase 2: 8-GPU screening run (200 iters) ---"
echo "Start: $(date)"

torchrun --nproc_per_node=8 dinov3/train/train.py \
  --output-dir "${OUTPUT_DIR}" \
  "${COMMON_ARGS[@]}" \
  train.OFFICIAL_EPOCH_LENGTH=200 \
  optim.epochs=1 \
  train.num_workers=20 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true

echo ""
echo "=== ${RUN_TAG} complete: $(date) ==="
echo ""
echo "Steady-state MFU (skip first ~20 iters for compile warmup):"
echo "  grep 'Training' /mnt/weka/adovlatyan/logs/compile-mode-${SLURM_JOB_ID}.out | awk 'NR>2'"
