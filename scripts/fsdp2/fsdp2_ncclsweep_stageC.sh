#!/bin/bash
# Phase 5 NCCL knob sweep — Stage C (long-run anchoring + B1 replicate).
#
# Stage B (2026-05-18) results required tighter wording on three findings:
#   - B1 stacking is a regression at 300 iters; mechanism unproven from one run
#   - B2 LL128-under-full-AC: latency-reduction hypothesis (not proven)
#   - B3 bs=128 + LL128 at 300 iters mixed regimes; late-window degraded
#
# Stage C closes the loop:
#   C1: bs=96  sel + LL128                       1000 iters   anchor production claim
#   C2: bs=128 sel + LL128                       1000 iters   characterize bs=128 properly
#   C3: bs=96  sel + LL128 + AVOID_RECORD_STREAMS 300 iters   B1 replicate (confirm regression)
#
# Variant resolution via SLURM_JOB_NAME (--job-name=ncclC-<variant>).
#
#SBATCH --job-name=dinov3-nccl-sweepC
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --nodelist=gpu07
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/fsdp2-nccl-%x-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/fsdp2-nccl-%x-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  echo "WARN: PYTORCH_CUDA_ALLOC_CONF contained 'expandable_segments:True' — unsetting for FSDP2."
fi
unset PYTORCH_CUDA_ALLOC_CONF

unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

if [[ "${SLURM_JOB_NAME:-}" == ncclC-* ]]; then
  VARIANT="${SLURM_JOB_NAME#ncclC-}"
else
  VARIANT="C1"
fi

BATCH_SIZE=96
AC_FULL=false
EPOCH_LEN=1000
KNOB_DESC=""

case "${VARIANT}" in
  C1)
    BATCH_SIZE=96
    EPOCH_LEN=1000
    export NCCL_PROTO=LL128
    KNOB_DESC="bs=96  sel + LL128 (1000 iters — production anchor)"
    ;;
  C2)
    BATCH_SIZE=128
    EPOCH_LEN=1000
    export NCCL_PROTO=LL128
    KNOB_DESC="bs=128 sel + LL128 (1000 iters — bs=128 characterization)"
    ;;
  C3)
    BATCH_SIZE=96
    EPOCH_LEN=300
    export NCCL_PROTO=LL128
    export TORCH_NCCL_AVOID_RECORD_STREAMS=1
    KNOB_DESC="bs=96  sel + LL128 + AVOID_RECORD_STREAMS (300 iters — B1 replicate)"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}" >&2
    exit 2
    ;;
esac

export DINOV3_MEMORY_PROFILE=1
export DINOV3_MEMORY_PROFILE_PERIOD=50

RUN_TAG="fsdp2_ncclsweepC_${VARIANT}"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== NCCL Sweep Stage C: ${VARIANT} — ${KNOB_DESC} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Batch size: ${BATCH_SIZE}  AC_full: ${AC_FULL}  iters: ${EPOCH_LEN}"
echo "NCCL env after override:"
env | grep -E '^(NCCL_|TORCH_NCCL_)' | sort || echo "  (no overrides)"
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
  train.OFFICIAL_EPOCH_LENGTH=${EPOCH_LEN} \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  train.compile=true \
  train.distributed_strategy=fsdp2 \
  train.fsdp_reshard_after_forward=true \
  train.checkpointing=true \
  train.checkpointing_full=${AC_FULL} \
  train.sharded_eval_checkpoint=true \
  checkpointing.period=100000 \
  evaluation.eval_period_iterations=100000 \
  wandb.enabled=false

echo "=== ${RUN_TAG} complete: $(date) ==="
