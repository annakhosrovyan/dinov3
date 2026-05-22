#!/bin/bash
# Phase 5 P5-OOM-memprofile-v2: FSDP2 bs=128 WITHOUT activation checkpointing,
# full memory profiling enabled. Goal is to reproduce (or, on this run length,
# get close to) the original bs=128 OOM and capture per-rank [MEMPROFILE] +
# [MEMFRAG] lines so we can categorize the failure mode:
#   - working-set growth (peak alloc keeps climbing)
#   - rank-local skew (one rank's peak >> others)
#   - eval/checkpoint materialization spike (peak at phase boundaries)
#   - fragmentation (inactive_split_mb growth, alloc_retries > 0)
#
# This is the diagnostic Codex review (gpt-5.5, 2026-05-14) insisted on
# keeping active — bs=96+AC clean fragmentation does NOT tell us why the
# original bs=128 OOM happened without AC.
#
# Method:
#   - train.checkpointing=false (replicate original failing config)
#   - bs=128 (the failing config)
#   - checkpointing.period=400, evaluation.eval_period_iterations=400 (combined
#     phase pressure at iter 400, 800 — matches what AC runs exercised)
#   - OFFICIAL_EPOCH_LENGTH=1000  (two phase boundaries → A/B)
#   - DINOV3_MEMORY_PROFILE=1     → [MEMPROFILE] at every phase
#   - DINOV3_MEMORY_PROFILE_PERIOD=10 → [MEMFRAG] every 10 iters → fragmentation
#     timeline before the suspected failure point
#
# Walltime 90 min — bs=128 should be slower per step but if it OOMs in
# < 30 min that's data; if it survives 1000 iters, we have a clean trace.
#
#SBATCH --job-name=dinov3-fsdp2-bs128-memprofile
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/fsdp2-bs128-memprofile-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/fsdp2-bs128-memprofile-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Sanitize allocator env (FSDP2; do NOT inherit DDP-only expandable_segments).
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  echo "WARN: PYTORCH_CUDA_ALLOC_CONF contained 'expandable_segments:True' — unsetting for FSDP2."
fi
unset PYTORCH_CUDA_ALLOC_CONF

# Memory profiling
export DINOV3_MEMORY_PROFILE=1
export DINOV3_MEMORY_PROFILE_PERIOD=10

BATCH_SIZE=128
RUN_TAG="fsdp2_bs${BATCH_SIZE}_memprofile_noAC"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 FSDP2 bs=128 memprofile (NO AC) — reproduce original OOM ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Batch size: ${BATCH_SIZE}"
echo "AC: train.checkpointing=false (replicate original failing config)"
echo "Aligned eval + ckpt at period=400 → combined-phase pressure at iter 400 and 800"
echo "Memory profile: DINOV3_MEMORY_PROFILE=1, DINOV3_MEMORY_PROFILE_PERIOD=10"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF:-<unset>}"
echo "Date: $(date)"
echo "EXPECTED: this run may OOM (replicating production failure). If it OOMs,"
echo "  the [MEMPROFILE]/[MEMFRAG] lines up to that point tell us the failure mode."
echo "  If it survives 1000 iters, that itself is data — the OOM may be longer-horizon."

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
  train.distributed_strategy=fsdp2 \
  train.fsdp_reshard_after_forward=true \
  train.checkpointing=false \
  train.checkpointing_full=false \
  train.sharded_eval_checkpoint=true \
  checkpointing.period=400 \
  evaluation.eval_period_iterations=400 \
  wandb.enabled=false || EXIT_CODE=$?

echo "=== ${RUN_TAG} exit code: ${EXIT_CODE:-0} at $(date) ==="
echo "--- All-rank [MEMPROFILE] markers (any rank, includes OOMing rank if applicable) ---"
grep "MEMPROFILE" /mnt/weka/adovlatyan/logs/fsdp2-bs128-memprofile-${SLURM_JOB_ID}.out | tail -40 || true
echo "--- All-rank [MEMFRAG] last 10 lines ---"
grep "MEMFRAG" /mnt/weka/adovlatyan/logs/fsdp2-bs128-memprofile-${SLURM_JOB_ID}.out | tail -10 || true
echo "--- CUDA OOM lines (if any) ---"
grep -i "out of memory\|OutOfMemoryError\|CUDA out of memory" /mnt/weka/adovlatyan/logs/fsdp2-bs128-memprofile-${SLURM_JOB_ID}.out /mnt/weka/adovlatyan/logs/fsdp2-bs128-memprofile-${SLURM_JOB_ID}.err 2>/dev/null | tail -10 || true
