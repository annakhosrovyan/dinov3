#!/bin/bash
# Phase 5 — missing-cell probe: bs=128 + selective AC + reshard_after_forward=false.
#
# Hypothesis: reshF (no-release) keeps params resident and moves the all-gather to the
# forward (Tim Darcet 2026-04-25 — "basically equivalent to DDP, with the gather moved
# before the fwd instead of after the bwd"). bs=128 has a +6.7 % single-GPU compute win
# that gets erased at 8-GPU under reshT — reshF is the principled lever to test whether
# the comm-pattern change recovers that win.
#
# This is ONE run, no NCCL knobs, no LL128. Compared against:
#   45366  bs=128 reshT sel AC          1276 img/s  same-node rerun
#   45369  bs=96  reshF sel AC          1305 img/s  cross-node
#   46030  bs=128 reshT sel AC + LL128  1194 img/s  1000-iter (Stage C C2)
#
# Decision rule:
#   img/s ≥ ~1330 → clean beat; queue +LL128 follow-up, reconsider "park bs=128"
#   img/s ~1250–1300 → in noise band; lock in "park bs=128" with full coverage
#   img/s < 1200 or OOM → reshF pathological at bs=128; shelve
#
# NODE NOTE (2026-05-18): gpu07 was occupied (~1.5 day queue wait), so this run was
# submitted WITHOUT --nodelist=gpu07 — will land on first available H100 node.
# Comparison anchors 45366 (bs=128 reshT sel) and 46030 (C2) both ran on gpu07.
# Cross-node variance is now part of the read; record the actual node in any
# downstream writeup. (gpu07 itself has shown ~20 % inter-epoch variance, so the
# same-node argument was weaker than it sounds anyway.)
#
#SBATCH --job-name=dinov3-bs128-reshF-sel
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/fsdp2-bs128-reshF-sel-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/fsdp2-bs128-reshF-sel-%j.err

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

# Clear any inherited NCCL knobs — this is a clean reshF probe, no NCCL overrides
unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

export DINOV3_MEMORY_PROFILE=1
export DINOV3_MEMORY_PROFILE_PERIOD=50

RUN_TAG="fsdp2_bs128_reshF_sel"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== bs=128 sel AC reshF — missing cell probe ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Config: bs=128, selective AC, reshard_after_forward=false, 1000 iters, no NCCL knobs"
echo "NCCL env (should be empty/default):"
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
  train.batch_size_per_gpu=128 \
  train.num_workers=20 \
  train.OFFICIAL_EPOCH_LENGTH=1000 \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  train.compile=true \
  train.distributed_strategy=fsdp2 \
  train.fsdp_reshard_after_forward=false \
  train.checkpointing=true \
  train.checkpointing_full=false \
  train.sharded_eval_checkpoint=true \
  checkpointing.period=100000 \
  evaluation.eval_period_iterations=100000 \
  wandb.enabled=false

echo "=== ${RUN_TAG} complete: $(date) ==="
