#!/bin/bash
# Phase 6 — DDP bs=96, torch.compile(fullgraph=True, triton.cudagraphs=True) on backbone.
#
# THE ONE THING THIS RUN IS TESTING:
#   Does entering the fullgraph+triton.cudagraphs path in wrap_compile_block actually
#   improve throughput over the job 48312 baseline (1,387 img/s, 7.94% MFU, step_ms 545)?
#
# How it works:
#   train.cudagraphs=true → wrap_compile_block takes the first branch for backbone blocks:
#     module.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": True})
#   Heads (DINO, iBOT) still compile with module.compile() default — dynamic=True, no fullgraph.
#   The iBOT dynamic-mask shape issue lives in the heads/loss, NOT the backbone, so this
#   is safe to test without any code changes.
#
# The [COMPILE] log lines at job start confirm which path was taken:
#   grep '\[COMPILE\]' <log> | head -10
#
# Comparator: job 48312 (DDP bs=96, compile=true, cudagraphs=false)
#   iter 400-999 mean: 1,387 img/s, 7.94% MFU, step_ms 545 ms, peak_alloc 25.8 GB
#
#SBATCH --job-name=dinov3-ddp-bs96-cudagraphs
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/ddp-bs96-cudagraphs-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/ddp-bs96-cudagraphs-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# No expandable_segments — matching job 48312 exactly.
unset PYTORCH_CUDA_ALLOC_CONF

# No NCCL knobs — clean baseline matching job 48312.
unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

BATCH_SIZE=96
RUN_TAG="ddp_bs${BATCH_SIZE}_cudagraphs"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 DDP bs=96 fullgraph+triton.cudagraphs: ${RUN_TAG} ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "Strategy:  DDP, compile=true, cudagraphs=true"
echo "Compile path (backbone): fullgraph=True, dynamic=False, triton.cudagraphs=True"
echo "Compile path (heads):    module.compile() default (dynamic=True)"
echo "Batch size: ${BATCH_SIZE}"
echo "Comparator: job 48312 (same recipe, cudagraphs=false) — 1387 img/s, 7.94% MFU"
echo "Date: $(date)"
echo ""
echo "Verify compile path from log:"
echo "  grep '\[COMPILE\]' \${OUTPUT_DIR}/training.log | head -10"

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
  train.distributed_strategy=ddp \
  train.checkpointing=false \
  train.sharded_eval_checkpoint=true \
  checkpointing.period=400 \
  evaluation.eval_period_iterations=400 \
  wandb.enabled=false

echo ""
echo "=== ${RUN_TAG} complete: $(date) ==="
echo "--- Steady-state MFU (skip first ~20 iters for compile warmup) ---"
grep "Training" /mnt/weka/adovlatyan/logs/ddp-bs96-cudagraphs-${SLURM_JOB_ID}.out \
  | awk 'NR>20 {print}' | tail -20 || true
echo ""
echo "--- [COMPILE] path confirmation ---"
grep "\[COMPILE\]" /mnt/weka/adovlatyan/logs/ddp-bs96-cudagraphs-${SLURM_JOB_ID}.out || true
