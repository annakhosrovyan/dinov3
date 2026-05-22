#!/bin/bash
# Phase 5 NCCL knob sweep — Stage A.
#
# Codex-designed two-stage factorial (2026-05-16). Stage A screens 5 NCCL/torch
# knobs on the production-default cell (bs=96, selective AC, reshard=True).
# One knob per run, no stacking. Plus a baseline replicate at the END to bound
# gpu07 thermal/cache drift (Codex flagged single-node baseline drift as fake-
# winner risk).
#
# Variants (set via NCCL_VARIANT env var passed through sbatch --export):
#   A0       - baseline (NCCL defaults, no overrides)
#   A1_NVLS  - NCCL_NVLS_ENABLE=1
#   A2_LL128 - NCCL_PROTO=LL128
#   A3_NT256 - NCCL_NTHREADS=256
#   A4_BUF16 - NCCL_BUFFSIZE=16777216 (16 MiB)
#   A5_NORS  - TORCH_NCCL_AVOID_RECORD_STREAMS=1
#   A6       - baseline REPLICATE (end-of-stage drift control)
#
# Methodology guardrails (Codex):
#   - Same node (gpu07), same iter window, same script.
#   - No NCCL_DEBUG=INFO (perturbs timing).
#   - 300 iters; eval+ckpt OFF (period=10000); steady-state window iter >= 100.
#   - Win threshold: >= 2-3% on step_ms; below is single-run noise.
#
#SBATCH --job-name=dinov3-nccl-sweepA
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --nodelist=gpu07
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=00:30:00
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

# Clear any pre-set NCCL knobs from the parent env so we start clean
unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

# Variant resolution: prefer explicit NCCL_VARIANT env var; otherwise derive from
# job name (e.g. SLURM_JOB_NAME="ncclA-A1_NVLS" -> A1_NVLS). The BCM portal sbatch
# wrapper injects --get-user-env whenever --export is used, which fails on this
# cluster, so we pass the variant via --job-name=ncclA-<variant> instead.
if [ -z "${NCCL_VARIANT:-}" ]; then
  if [[ "${SLURM_JOB_NAME:-}" == ncclA-* ]]; then
    NCCL_VARIANT="${SLURM_JOB_NAME#ncclA-}"
  else
    NCCL_VARIANT="A0"
  fi
fi

case "${NCCL_VARIANT}" in
  A0|A6)
    KNOB_DESC="baseline (NCCL defaults)"
    ;;
  A1_NVLS)
    export NCCL_NVLS_ENABLE=1
    KNOB_DESC="NCCL_NVLS_ENABLE=1"
    ;;
  A2_LL128)
    export NCCL_PROTO=LL128
    KNOB_DESC="NCCL_PROTO=LL128"
    ;;
  A3_NT256)
    export NCCL_NTHREADS=256
    KNOB_DESC="NCCL_NTHREADS=256"
    ;;
  A4_BUF16)
    export NCCL_BUFFSIZE=16777216
    KNOB_DESC="NCCL_BUFFSIZE=16777216 (16 MiB)"
    ;;
  A5_NORS)
    export TORCH_NCCL_AVOID_RECORD_STREAMS=1
    KNOB_DESC="TORCH_NCCL_AVOID_RECORD_STREAMS=1"
    ;;
  *)
    echo "Unknown NCCL_VARIANT=${NCCL_VARIANT}" >&2
    exit 2
    ;;
esac

export DINOV3_MEMORY_PROFILE=1
export DINOV3_MEMORY_PROFILE_PERIOD=50

BATCH_SIZE=96
RUN_TAG="fsdp2_ncclsweepA_${NCCL_VARIANT}"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== NCCL Sweep Stage A: ${NCCL_VARIANT} — ${KNOB_DESC} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Batch size: ${BATCH_SIZE} (sel AC, reshT, 300 iters, eval+ckpt off)"
echo "NCCL env after override:"
env | grep -E '^(NCCL_|TORCH_NCCL_)' | sort || echo "  (no NCCL overrides set — baseline)"
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
  train.OFFICIAL_EPOCH_LENGTH=300 \
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
  checkpointing.period=10000 \
  evaluation.eval_period_iterations=10000 \
  wandb.enabled=false

echo "=== ${RUN_TAG} complete: $(date) ==="
