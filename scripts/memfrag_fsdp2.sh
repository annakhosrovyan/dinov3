#!/bin/bash
# FSDP2 long-run fragmentation tracking script.
#
# MOTIVATION:
# Our 70-iter worst-case profiling captures single-step peak memory correctly but
# misses long-run allocator fragmentation. FSDP2 executes ~24 alloc/free cycles
# per training step (12 transformer blocks × per-block all_gather+reduce_scatter
# during forward+backward). Without expandable_segments, each cycle can leave
# fragments in the caching allocator that can't be reused for differently-sized
# allocations. Over thousands of steps, these fragments accumulate until the
# allocator cannot satisfy a new 112 MB all_gather even though 44 GB appears "free".
# This is the fragmentation-induced OOM that our 70-iter runs cannot observe.
#
# This script runs 500 iterations and logs [MEMFRAG] lines every LOG_PERIOD iters.
# Post-process with:
#   grep '\[MEMFRAG\]' <logfile> | grep 'rank=0' | \
#     awk '{for(i=1;i<=NF;i++) if($i~/iter=|alloc_retries=|inactive_split_mb=|fragmentation_ratio=/) printf $i " "; print ""}'
#
# Key signals to watch:
#   alloc_retries: cumulative retries due to fragmentation. Growing = bad.
#   inactive_split_mb: MB stuck in fragments. Growing = fragmentation building.
#   fragmentation_ratio > 0.3: dangerous, OOM risk on real-length runs.
#
# Usage: sbatch scripts/memfrag_fsdp2.sh [batch_size] [expand_segments] [n_iters] [log_period]
# Example (reproduce colleague's OOM config): sbatch scripts/memfrag_fsdp2.sh 128 false 500 10
# Example (ES control):                      sbatch scripts/memfrag_fsdp2.sh 128 true  500 10
#
#SBATCH --job-name=dinov3-memfrag-fsdp2
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/memfrag-fsdp2-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/memfrag-fsdp2-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Enable per-phase + fragmentation tracking
export DINOV3_MEMORY_PROFILE=1

BATCH_SIZE="${1:-128}"
EXPAND_SEG="${2:-false}"
N_ITERS="${3:-500}"
LOG_PERIOD="${4:-10}"

# Log every LOG_PERIOD iterations — controls [MEMFRAG] emission frequency
export DINOV3_MEMORY_PROFILE_PERIOD="${LOG_PERIOD}"

if [ "${EXPAND_SEG}" = "true" ]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

RUN_TAG="memfrag_fsdp2_bs${BATCH_SIZE}_es${EXPAND_SEG}_n${N_ITERS}"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 FSDP2 Fragmentation Study: ${RUN_TAG} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Batch size: ${BATCH_SIZE} | Expand segments: ${EXPAND_SEG}"
echo "Iterations: ${N_ITERS} | Log period: every ${LOG_PERIOD} iters"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF:-not set}"
echo "Output: ${OUTPUT_DIR}"
echo "Date: $(date)"
echo ""
echo "Theory: FSDP2 executes ~24 alloc/free cycles per step (12 blocks × fwd+bwd)."
echo "Without ES, these fragment the allocator pool over thousands of steps."
echo "alloc_retries and inactive_split_mb growing = fragmentation accumulating."
echo ""
echo "Post-process results with:"
echo "  grep '\[MEMFRAG\]' /mnt/weka/adovlatyan/logs/memfrag-fsdp2-${SLURM_JOB_ID}.out | grep 'rank=0'"
echo ""

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
  train.batch_size_per_gpu="${BATCH_SIZE}" \
  train.num_workers=20 \
  train.OFFICIAL_EPOCH_LENGTH="${N_ITERS}" \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  train.compile=true \
  train.distributed_strategy=fsdp2 \
  train.checkpointing=false \
  wandb.enabled=false \
  checkpointing.period=99999

echo ""
echo "=== ${RUN_TAG} complete: $(date) ==="
echo ""
echo "Fragmentation summary (rank=0 only):"
grep '\[MEMFRAG\]' "/mnt/weka/adovlatyan/logs/memfrag-fsdp2-${SLURM_JOB_ID}.out" | grep 'rank=0' || true
