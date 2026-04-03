#!/bin/bash
# DDP worst-case memory profiling script.
# Runs 70 iterations with eval triggered at iter 39 and checkpoint at iter 49,
# capturing per-phase peak GPU memory to identify the true OOM risk point.
#
# Usage: sbatch scripts/memprofile_ddp.sh <batch_size> [checkpointing] [expand_segments]
# Example: sbatch scripts/memprofile_ddp.sh 128 false true
#
# Output: grep '[MEMPROFILE]' <logfile> | grep 'rank=0' for per-phase peaks.
#
#SBATCH --job-name=dinov3-memprof-ddp
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/memprof-ddp-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/memprof-ddp-%j.err

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

BATCH_SIZE="${1:-128}"
CHECKPOINTING="${2:-false}"
EXPAND_SEG="${3:-true}"

if [ "${EXPAND_SEG}" = "true" ]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

RUN_TAG="memprof_ddp_bs${BATCH_SIZE}_ckpt${CHECKPOINTING}_es${EXPAND_SEG}"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 DDP Worst-Case Memory Profiling: ${RUN_TAG} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Batch size: ${BATCH_SIZE}"
echo "Checkpointing: ${CHECKPOINTING}"
echo "Expand segments: ${EXPAND_SEG}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF:-not set}"
echo "Output: ${OUTPUT_DIR}"
echo "Date: $(date)"
echo ""
echo "Phase schedule:"
echo "  Iters 0-9:  torch.compile warmup + early steady state"
echo "  Iter  10:   steady_state memory marker logged"
echo "  Iter  39:   eval phase fires (do_test + full_tensor on EMA)"
echo "  Iter  49:   checkpoint phase fires (DCP save, model + optimizer)"
echo "  Iters 50-69: post-checkpoint steady state"
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
  train.OFFICIAL_EPOCH_LENGTH=70 \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  train.compile=true \
  train.distributed_strategy=ddp \
  train.checkpointing="${CHECKPOINTING}" \
  wandb.enabled=false \
  evaluation.eval_period_iterations=40 \
  checkpointing.period=50

echo ""
echo "=== ${RUN_TAG} complete: $(date) ==="
echo ""
echo "Extract memory profile results with:"
echo "  grep '\[MEMPROFILE\]' /mnt/weka/adovlatyan/logs/memprof-ddp-${SLURM_JOB_ID}.out | grep 'rank=0'"
