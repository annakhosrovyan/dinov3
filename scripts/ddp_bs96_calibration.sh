#!/bin/bash
# Phase 5 — DDP bs=96 calibration run (one run, not a sweep).
#
# Purpose: we have NO exact DDP bs=96 row. bs=96 is the current safe FSDP2
# operating point. Old DDP screening only covered bs=64 / bs=128 (archived
# jobs 9630 / 9631 / 9651). This run answers one clean question:
#
#   "At the actual production batch size (bs=96), how big is the absolute
#    throughput gap between DDP and FSDP2 — measured under matched conditions?"
#
# Matched against the FSDP2 matrix cell fsdp2_bs96_reshT_noAC (job 45367,
# 1,279 img/s): same bs, same no-AC, same compile=true, same loader settings,
# same forced eval+ckpt at iter 400/800, same memory profiling. The ONLY
# variable changed is distributed_strategy: fsdp2 -> ddp.
#
# Allocator: PYTORCH_CUDA_ALLOC_CONF is UNSET (no expandable_segments). The
# old DDP+ES screening used ES; this run isolates DDP itself first. ES can be
# a follow-up if DDP bs=96 is throughput-positive and memory is tight.
#
# Node: NOT pinned — cluster is saturated; first free H100 node takes it.
# Record the actual node in the post-run analysis (cross-node variance applies,
# same as job 47554 on 2026-05-20).
#
#SBATCH --job-name=dinov3-ddp-bs96-calib
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/ddp-bs96-calib-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/ddp-bs96-calib-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Allocator: unset expandable_segments. This run isolates DDP without ES.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  echo "NOTE: PYTORCH_CUDA_ALLOC_CONF contained 'expandable_segments:True' — unsetting (DDP-without-ES calibration)."
fi
unset PYTORCH_CUDA_ALLOC_CONF

# Clear any inherited NCCL knobs — clean DDP baseline, no NCCL overrides.
unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

# Memory profiling — emits [MEMPROFILE] / [MEMFRAG] lines for grep analysis.
export DINOV3_MEMORY_PROFILE=1
export DINOV3_MEMORY_PROFILE_PERIOD=10

BATCH_SIZE=96
RUN_TAG="ddp_bs${BATCH_SIZE}_calibration"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 DDP bs=96 calibration: ${RUN_TAG} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Strategy: DDP (no AC, compile=true, no expandable_segments)"
echo "Batch size: ${BATCH_SIZE}"
echo "Forced eval + ckpt at period=400 -> combined-phase pressure at iter 400 and 800"
echo "OFFICIAL_EPOCH_LENGTH=1000 (A/B replication of phase boundary)"
echo "Comparator: fsdp2_bs96_reshT_noAC job 45367 (1,279 img/s) — identical recipe, fsdp2 vs ddp"
echo "Memory profile: DINOV3_MEMORY_PROFILE=1, DINOV3_MEMORY_PROFILE_PERIOD=10"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF:-<unset>}"
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
  train.distributed_strategy=ddp \
  train.checkpointing=false \
  train.sharded_eval_checkpoint=true \
  checkpointing.period=400 \
  evaluation.eval_period_iterations=400 \
  wandb.enabled=false

echo "=== ${RUN_TAG} complete: $(date) ==="
echo "--- [MEMPROFILE] summary (rank 0) ---"
grep "MEMPROFILE.*rank=0" /mnt/weka/adovlatyan/logs/ddp-bs96-calib-${SLURM_JOB_ID}.out | tail -20 || true
echo "--- [MEMFRAG] summary (rank 0, last 5) ---"
grep "MEMFRAG.*rank=0" /mnt/weka/adovlatyan/logs/ddp-bs96-calib-${SLURM_JOB_ID}.out | tail -5 || true
