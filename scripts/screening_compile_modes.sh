#!/bin/bash
# Compile-mode screening: benchmarks "default" vs "max-autotune-no-cudagraphs".
#
# max-autotune-no-cudagraphs enables Triton kernel autotuning (tile sizes, warp config, etc.)
# without CUDA graph capture. CUDA graphs are already confirmed broken on the multi-crop
# architecture (see learnings/cuda_graphs.md). This is the safe autotuning path.
#
# Expected: first-run compile warmup for max-autotune is ~10-20 min longer than default.
# Subsequent runs reuse the Triton cache (~/.cache/torch/inductor on NFS, shared across nodes).
# Potential steady-state gain: 5-20% faster kernel execution within compiled backbone blocks.
#
# Usage: sbatch scripts/screening_compile_modes.sh <mode> [batch_size]
#   mode: "default" or "max-autotune-no-cudagraphs"
#   batch_size: default 128
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

# DDP + expandable_segments for clean isolation of compile mode effect.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMPILE_MODE="${1:-default}"
BATCH_SIZE="${2:-128}"

# Normalize: "default" → null config value (PyTorch default)
if [ "${COMPILE_MODE}" = "default" ]; then
    COMPILE_MODE_ARG="train.compile_mode=null"
    SAFE_TAG="default"
else
    COMPILE_MODE_ARG="train.compile_mode=${COMPILE_MODE}"
    SAFE_TAG="${COMPILE_MODE//[^a-zA-Z0-9_-]/}"
fi

RUN_TAG="compile_mode_${SAFE_TAG}_bs${BATCH_SIZE}"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 Compile Mode Screening: ${RUN_TAG} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Compile mode: ${COMPILE_MODE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Output: ${OUTPUT_DIR}"
echo "Date: $(date)"
echo ""
echo "Note: max-autotune-no-cudagraphs adds 10-20 min of Triton autotuning on first run."
echo "      Triton cache lives in ~/.cache/torch/inductor (NFS-shared across nodes)."
echo "      Subsequent runs reuse the cache and start at steady-state speed immediately."
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
  train.OFFICIAL_EPOCH_LENGTH=200 \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  train.compile=true \
  train.distributed_strategy=ddp \
  train.sharded_eval_checkpoint=true \
  "${COMPILE_MODE_ARG}" \
  wandb.enabled=false \
  checkpointing.period=99999 \
  evaluation.eval_period_iterations=99999

echo ""
echo "=== ${RUN_TAG} complete: $(date) ==="
echo ""
echo "Extract steady-state MFU (skip first ~20 iters for compile warmup):"
echo "  grep 'mfu_pct' /mnt/weka/adovlatyan/logs/compile-mode-${SLURM_JOB_ID}.out | tail -150"
