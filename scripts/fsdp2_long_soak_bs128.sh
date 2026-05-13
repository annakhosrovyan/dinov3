#!/bin/bash
# FSDP2 long-run OOM/fragmentation soak test — bs=128, 5000 iterations.
#
# MOTIVATION:
# Our 500-iter fragmentation study (jobs 16750/16751) showed zero alloc_retries
# and flat inactive_split_mb through 500 iters. The colleague's OOM at FSDP2 bs=128
# remains unexplained. This soak runs 5000 iters (beyond the 3750-iter checkpoint
# cadence of real training) with eval and checkpoint phases to:
#   1. Rule out or confirm fragmentation accumulation at longer horizons.
#   2. Simulate real training memory behavior (eval + checkpoint load).
#
# MEMFRAG logged every 50 iters (100 data points total).
# MEMPROFILE logged at compile_warmup, steady_state, each eval, each checkpoint.
#
# Key signals:
#   alloc_retries: growing = fragmentation induced OOM risk
#   inactive_split_mb: growing = fragmentation accumulating
#   max_reserved_mb: growing across eval/checkpoint = long-run leak
#
# Post-process:
#   grep '\[MEMFRAG\]' <log> | grep 'rank=0'
#   grep '\[MEMPROFILE\]' <log> | grep 'rank=0'
#
#SBATCH --job-name=dinov3-fsdp2-longsoak
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/fsdp2-longsoak-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/fsdp2-longsoak-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# No expandable_segments — testing baseline FSDP2 fragmentation behavior
# (ES confounds the comparison by zeroing inactive_split_mb)

export DINOV3_MEMORY_PROFILE=1
export DINOV3_MEMORY_PROFILE_PERIOD=50   # MEMFRAG every 50 iters = 100 data points over 5000 iters

RUN_TAG="fsdp2_longsoak_bs128_noes_5000"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== DINOv3 FSDP2 Long Soak — bs=128, no ES, 5000 iters ==="
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     ${SLURM_NODELIST}"
echo "Date:     $(date)"
echo "Goal:     rule out / confirm fragmentation OOM beyond 500 iters"
echo "Output:   ${OUTPUT_DIR}"
echo ""
echo "MEMFRAG every 50 iters | MEMPROFILE at each eval/checkpoint"
echo "eval_period_iterations=500 (10 eval cycles)"
echo "checkpointing.period=1000 (5 checkpoint cycles)"
echo ""

echo "=== GPU memory baseline (before torchrun) ==="
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader,nounits \
    | awk -F',' '{printf "  GPU %s: used=%s MB  free=%s MB  total=%s MB\n", $1, $2, $3, $4}'
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
  train.batch_size_per_gpu=128 \
  train.num_workers=20 \
  train.OFFICIAL_EPOCH_LENGTH=5000 \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  train.compile=true \
  train.distributed_strategy=fsdp2 \
  train.checkpointing=false \
  wandb.enabled=false \
  evaluation.eval_period_iterations=500 \
  checkpointing.period=1000

echo ""
echo "=== ${RUN_TAG} complete: $(date) ==="
echo ""
echo "Fragmentation summary (rank=0):"
grep '\[MEMFRAG\]'   "/mnt/weka/adovlatyan/logs/fsdp2-longsoak-${SLURM_JOB_ID}.out" | grep 'rank=0' | tail -20 || true
echo ""
echo "Memory profile phases (rank=0):"
grep '\[MEMPROFILE\]' "/mnt/weka/adovlatyan/logs/fsdp2-longsoak-${SLURM_JOB_ID}.out" | grep 'rank=0' || true
