#!/bin/bash
# Phase 5 wrap-up — §6 exp 2: bs=96 reshT sel AC, eval+ckpt-OFF measurement shape.
#
# Closes the missing same-shape bs=96 anchor for the gpu05 2x2 grid. Existing
# 45280 is bs=96 reshT sel AC but ran with FORCED eval+ckpt at iter 400/800,
# which makes its iter 400-999 window measurement-shape-inconsistent with 51069
# (bs=128 reshT, eval+ckpt OFF) and 47554 (bs=128 reshF, eval+ckpt OFF).
#
# Pair: this run + 51069 answers "does bs=128 beat bs=96 under reshT+sel AC at
# matched measurement shape." Then the bs=96 reshF re-soak closes the reshard
# axis for bs=96 too.
#
# ONLY VARIABLE CHANGED vs 51069: train.batch_size_per_gpu 128 -> 96.
# Everything else (sel AC, reshT, compile=true, no NCCL knobs, eval+ckpt OFF,
# memprofile period 50) matched to 51069.
#
# Node: NOT pinned (cluster saturated, user accepts cross-node variance for the
# wrap-up grid). Record the actual node in any downstream writeup.
#
#SBATCH --job-name=dinov3-bs96-reshT-sel-resoak
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/fsdp2-bs96-reshT-sel-resoak-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/fsdp2-bs96-reshT-sel-resoak-%j.err

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

export DINOV3_MEMORY_PROFILE=1
export DINOV3_MEMORY_PROFILE_PERIOD=50

RUN_TAG="fsdp2_bs96_reshT_sel_resoak"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== bs=96 sel AC reshT — wrap-up §6 exp 2 (eval+ckpt OFF shape, pair to 51069) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Config: bs=96, selective AC, reshard_after_forward=true, 1000 iters, no NCCL knobs, no eval/ckpt"
echo "Pair: 51069 (bs=128 reshT sel, gpu05) — only variable changed is bs"
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
  train.batch_size_per_gpu=96 \
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
