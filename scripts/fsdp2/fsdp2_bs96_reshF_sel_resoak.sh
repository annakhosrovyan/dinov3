#!/bin/bash
# Phase 5 wrap-up — §6 exp 3: bs=96 reshF sel AC, eval+ckpt-OFF measurement shape.
#
# Closes the bs=96 reshF cell for the 2x2 grid. Existing 45369 is bs=96 reshF
# sel AC but ran with FORCED eval+ckpt at iter 400/800 — measurement-shape-
# inconsistent with 47554 (bs=128 reshF) which had eval+ckpt OFF.
#
# Together with the bs=96 reshT re-soak, 51069, and 47554, this gives a clean
# 2x2 grid (batch x reshard) under one measurement shape — the cleanest answer
# to whether reshF helps bs=96 vs bs=128 differently.
#
# ONLY VARIABLE CHANGED vs 47554: train.batch_size_per_gpu 128 -> 96.
# Everything else (sel AC, reshF, compile=true, no NCCL knobs, eval+ckpt OFF,
# memprofile period 50) matched to 47554.
#
# Node: NOT pinned (cluster saturated, user accepts cross-node variance for the
# wrap-up grid).
#
#SBATCH --job-name=dinov3-bs96-reshF-sel-resoak
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/fsdp2-bs96-reshF-sel-resoak-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/fsdp2-bs96-reshF-sel-resoak-%j.err

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

RUN_TAG="fsdp2_bs96_reshF_sel_resoak"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== bs=96 sel AC reshF — wrap-up §6 exp 3 (eval+ckpt OFF shape, pair to 47554) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Config: bs=96, selective AC, reshard_after_forward=false, 1000 iters, no NCCL knobs, no eval/ckpt"
echo "Pair: 47554 (bs=128 reshF sel, gpu05) — only variable changed is bs"
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
  train.fsdp_reshard_after_forward=false \
  train.checkpointing=true \
  train.checkpointing_full=false \
  train.sharded_eval_checkpoint=true \
  checkpointing.period=100000 \
  evaluation.eval_period_iterations=100000 \
  wandb.enabled=false

echo "=== ${RUN_TAG} complete: $(date) ==="
