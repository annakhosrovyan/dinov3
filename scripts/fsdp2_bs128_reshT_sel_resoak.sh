#!/bin/bash
# Phase 5 — paired same-node A/B: bs=128 + sel AC + reshT (re-soak vs 47554 anchor).
#
# Purpose: close the cross-node confound in the bs=128 reshT-vs-reshF comparison.
#   45366 (bs=128 reshT sel AC, gpu07, 1000 i) = 1,276 img/s, MFU 7.30 %.
#   47554 (bs=128 reshF sel AC, gpu05, 1000 i) = 1,341 img/s mean / 1,312 median, MFU 7.67 %.
# The +5.1 % reshF→reshT delta is currently cross-node, cross-date. This run reruns the
# reshT half on gpu05 (47554's node) so we get a clean variable-isolated A/B.
#
# Also addresses two secondary questions:
#   - bs=128 vs bs=96: 47554 hit 1,341 vs bs=96 best 1,351 (45280, gpu07). Inside noise.
#     A clean same-node bs=128 reshT number lets us rule out reshF as the lever.
#   - OOM history (researcher 05-12, reshT-vs-reshF unknown): 1000-iter memprofile of
#     bs=128 reshT confirms the steady-state and post-warmup peak — secondary read.
#
# ONLY VARIABLE CHANGED vs 47554: train.fsdp_reshard_after_forward false -> true.
# Everything else (bs, AC, compile, NCCL knobs, eval+ckpt off, memprofile) matched
# to 47554 cell-for-cell.
#
# Node pinned to gpu05 deliberately. If gpu05 queues too long the user can resubmit
# without the --nodelist; same-node anchoring is the WHOLE POINT of this run so prefer
# waiting over relaunching on a different node.
#
# Decision rule (after run finishes):
#   bs=128 reshT on gpu05 ≈ 47554 (within ~2 %) → reshF buys nothing at bs=128 same-node
#   bs=128 reshT on gpu05 << 47554 (≥ 3 % lower) → reshF is the real lever at bs=128
#   bs=128 reshT on gpu05 ≥ bs=96 best (1,351) → bs=128 is competitive without reshF
#
#SBATCH --job-name=dinov3-bs128-reshT-sel-resoak
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --nodelist=gpu05
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/fsdp2-bs128-reshT-sel-resoak-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/fsdp2-bs128-reshT-sel-resoak-%j.err

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

# Clear any inherited NCCL knobs — clean reshT probe matched against 47554, no overrides
unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

export DINOV3_MEMORY_PROFILE=1
export DINOV3_MEMORY_PROFILE_PERIOD=50

RUN_TAG="fsdp2_bs128_reshT_sel_resoak"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== bs=128 sel AC reshT — paired same-node A/B vs 47554 (gpu05) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Config: bs=128, selective AC, reshard_after_forward=true, 1000 iters, no NCCL knobs"
echo "Anchor: 47554 (bs=128 reshF sel AC, gpu05) = 1,341 img/s mean / 1,312 median"
echo "Only variable vs 47554: fsdp_reshard_after_forward true (vs false)"
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
  train.fsdp_reshard_after_forward=true \
  train.checkpointing=true \
  train.checkpointing_full=false \
  train.sharded_eval_checkpoint=true \
  checkpointing.period=100000 \
  evaluation.eval_period_iterations=100000 \
  wandb.enabled=false

echo "=== ${RUN_TAG} complete: $(date) ==="
