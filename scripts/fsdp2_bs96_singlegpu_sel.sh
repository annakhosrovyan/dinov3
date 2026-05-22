#!/bin/bash
# Phase 5 single-GPU bs=96 + selective AC.
#
# Purpose (Codex disambiguation, 2026-05-15): isolate kernel/compile behavior from
# NCCL/FSDP scheduling. The 8-GPU matrix showed bs=128 step_time scales LINEARLY
# in batch vs bs=96 — i.e., no per-image throughput gain from larger batch.
# Single-GPU eliminates all collective traffic. Compared with the bs=128 sibling
# script, this answers: is the bs=128 flatness a NCCL/FSDP-scheduling effect or
# a local kernel/compile effect?
#
# Method: same code path as the 8-GPU runs (FSDP2 with reshard=True; 1 rank
# degenerates to no-shard but keeps the wrapping/compile path identical). Real
# Weka data (data: field was sub-ms in 8-GPU runs, not a confound). Eval and
# checkpoint disabled — pure throughput run.
#
#SBATCH --job-name=dinov3-fsdp2-bs96-sg
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --nodelist=gpu07
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/fsdp2-bs96-sg-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/fsdp2-bs96-sg-%j.err

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

export DINOV3_MEMORY_PROFILE=1
export DINOV3_MEMORY_PROFILE_PERIOD=10

BATCH_SIZE=96
RUN_TAG="fsdp2_bs${BATCH_SIZE}_singlegpu_sel"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 single-GPU bs=${BATCH_SIZE} + selective AC: ${RUN_TAG} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Batch size: ${BATCH_SIZE}"
echo "GPU count: 1  (sibling of bs=128 single-GPU job — same gpu07 node)"
echo "AC: selective"
echo "Eval+ckpt disabled (period=10000 > 300 iters total)"
echo "Date: $(date)"

torchrun --nproc_per_node=1 dinov3/train/train.py \
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
  train.num_workers=12 \
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
