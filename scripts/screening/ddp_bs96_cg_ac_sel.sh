#!/bin/bash
# Phase 6.A.5.a — DDP bs=96, AC=selective, cudagraphs=true.
#
# Question: how much throughput does selective AC cost when paired with cudagraphs?
# Compare vs job 53681 (bs=96, AC=off, cg=true) — 2,009 img/s, 11.50% MFU.
#
# AC's value is memory ↓ in exchange for backward time ↑ (recompute). At bs=96 we
# already had 25.8 GB headroom — AC's memory win is unused, so this run isolates
# the pure throughput cost of AC.
#
# Compatibility risk: checkpoint_wrapper introduces a control-flow boundary that
# may force graph breaks inside backbone blocks, partially undoing the
# fullgraph+triton.cudagraphs win. The result will tell us whether AC + cudagraphs
# can coexist on the same blocks.
#
#SBATCH --job-name=dinov3-ddp-bs96-cg-ac-sel
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/ddp-bs96-cg-ac-sel-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/ddp-bs96-cg-ac-sel-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

unset PYTORCH_CUDA_ALLOC_CONF
unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

BATCH_SIZE=96
RUN_TAG="ddp_bs${BATCH_SIZE}_cg_ac_sel"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== ${RUN_TAG} ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "Config:    DDP, bs=${BATCH_SIZE}, compile=true, cudagraphs=true, checkpointing=true (selective)"
echo "Comparator: 53681 (same bs, AC=off, cg=true) — 2009 img/s, 11.50% MFU, 379 ms"
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
  train.OFFICIAL_EPOCH_LENGTH=1000 \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  train.compile=true \
  train.cudagraphs=true \
  train.checkpointing=true \
  train.checkpointing_full=false \
  train.distributed_strategy=ddp \
  train.sharded_eval_checkpoint=true \
  checkpointing.period=400 \
  evaluation.eval_period_iterations=400 \
  wandb.enabled=false

echo ""
echo "=== ${RUN_TAG} complete: $(date) ==="
grep "Training" /mnt/weka/adovlatyan/logs/ddp-bs96-cg-ac-sel-${SLURM_JOB_ID}.out \
  | awk 'NR>20 {print}' | tail -20 || true
echo ""
grep "\[COMPILE\]" /mnt/weka/adovlatyan/logs/ddp-bs96-cg-ac-sel-${SLURM_JOB_ID}.out || true
