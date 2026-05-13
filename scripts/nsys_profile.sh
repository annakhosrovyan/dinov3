#!/bin/bash
# DINOv3 — Nsight Systems profile of an 8-GPU FSDP2 ZeRO-3 training run.
#
# Captures a steady-state window (well after torch.compile + FSDP2 init warmup) on a ~100M-param
# ViT-B model across 8×H100. Writes a .nsys-rep file to Weka. Designed for Phase 5 bottleneck
# investigation.
#
# **FSDP2 only.** DDP is closed for this codebase (Phase 4); there is always an FSDP2 config
# equivalent to DDP, and FSDP2 is the long-run platform.
# **Production-viable batch sizes only: 64, 96, 128.** bs=192/256 are NOT achievable in real
# training runs (worst-case memory profiling shows OOM during eval/checkpoint cycles).
#
# Usage:
#   sbatch scripts/nsys_profile.sh                          # default: bs=128
#   BATCH_SIZE=96  sbatch scripts/nsys_profile.sh           # conservative
#   BATCH_SIZE=64  sbatch scripts/nsys_profile.sh           # most conservative
#   NSYS_DELAY=240 NSYS_DURATION=90 sbatch scripts/nsys_profile.sh   # longer window
#
# Env knobs:
#   BATCH_SIZE      per-GPU batch size              (default 128; allowed: 64, 96, 128)
#   ITERS           total training iters to run     (default 2000; must outlast delay+duration)
#   NSYS_DELAY      seconds to wait before capture  (default 360 — covers compile + dataloader + pinned-pool warmup)
#   NSYS_DURATION   capture duration in seconds     (default 120 — ~250–400 steady-state iters)
#
# 2026-05-08: bumped delay 180→360 and duration 60→120 after first re-run (jobs 39028/39029)
# showed only ~18 s of actual GPU activity inside the 60 s window — pinned-memory pool was still
# growing in the supposed "steady state" (cudaHostAlloc_v3020 = 13 s of host time). 360 s puts the
# capture comfortably past the warmup tail. Multi-process capture flags added below.
#
# Output:
#   /mnt/weka/adovlatyan/nsys_profiles/<YYYY-MM-DD>/<jobid>/<run-tag>.nsys-rep
#
#SBATCH --job-name=dinov3-nsys
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/nsys-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/nsys-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

# nsys 2026.2.1 (cluster-wide install on Weka).
# `module` is not available on GPU nodes — use the direct binary path.
NSYS_BIN="/mnt/weka/apps/nsight-systems/2026.2.1/install/target-linux-x64/nsys"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

BATCH_SIZE="${BATCH_SIZE:-128}"
ITERS="${ITERS:-2000}"
NSYS_DELAY="${NSYS_DELAY:-360}"
NSYS_DURATION="${NSYS_DURATION:-120}"

# Reject non-production batch sizes loudly — bs=192/256 are not achievable in real training
# (worst-case memory profiling: OOM in eval/checkpoint cycles).
case "${BATCH_SIZE}" in
  64|96|128) ;;
  *)
    echo "ERROR: BATCH_SIZE=${BATCH_SIZE} is not production-viable. Allowed: 64, 96, 128." >&2
    exit 2
    ;;
esac

# Output paths — Weka, NOT NFS home
DATE_TAG="$(date +%Y-%m-%d)"
RUN_TAG="dinov3-fsdp2-bs${BATCH_SIZE}"
PROFILE_DIR="/mnt/weka/adovlatyan/nsys_profiles/${DATE_TAG}/${SLURM_JOB_ID}"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_nsys_${SLURM_JOB_ID}"
NSYS_OUT="${PROFILE_DIR}/${RUN_TAG}-${SLURM_JOB_ID}"

mkdir -p "${PROFILE_DIR}" /mnt/weka/adovlatyan/logs

echo "=== DINOv3 nsys profile (FSDP2 ZeRO-3) ==="
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Node:             ${SLURM_NODELIST}"
echo "Date:             $(date)"
echo "Batch / GPU:      ${BATCH_SIZE}"
echo "Iters:            ${ITERS}"
echo "nsys delay/dur:   ${NSYS_DELAY}s / ${NSYS_DURATION}s"
echo "nsys output:      ${NSYS_OUT}.nsys-rep"
echo "torchrun output:  ${OUTPUT_DIR}"
echo "nsys version:     $(${NSYS_BIN} --version | head -1)"
echo

# nsys flags:
#   --trace=cuda,nvtx,osrt,cudnn,cublas,nccl  → kernel + comms + OS-level stalls
#   --sample=cpu                              → CPU stack sampling for Python frames
#   --capture-range=none                      → use --delay/--duration (no in-code markers required)
#   --delay/--duration                        → skip compile warmup, then capture steady-state window
#   --force-overwrite=true                    → overwrite previous trace at this path
#   --python-sampling=true                    → resolve Python frames in CPU samples (best-effort)
#
# Optional flags worth flipping when investigating specific hypotheses:
#   --cuda-memory-usage=true                  → memory-track every alloc (doubles trace size)
#   --gpu-metrics-device=all                  → SM/DRAM utilization counters (may need driver perms)
#   --export=sqlite                           → SQLite dump for scripted post-processing
#
# Per-GPU vs full-tree: nsys traces the entire process tree by default — torchrun + 8 ranks land
# in one .nsys-rep. The viewer separates them by process; comms are visible across ranks.
"${NSYS_BIN}" profile \
  --output="${NSYS_OUT}" \
  --trace=cuda,nvtx,osrt,cudnn,cublas,nccl \
  --sample=cpu \
  --python-sampling=true \
  --capture-range=none \
  --delay="${NSYS_DELAY}" \
  --duration="${NSYS_DURATION}" \
  --force-overwrite=true \
  --trace-fork-before-exec=true \
  -- \
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
  train.OFFICIAL_EPOCH_LENGTH="${ITERS}" \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  train.compile=true \
  train.distributed_strategy=fsdp2 \
  train.sharded_eval_checkpoint=true \
  wandb.enabled=false \
  checkpointing.period=99999

echo
echo "=== nsys profile complete: $(date) ==="
ls -lh "${NSYS_OUT}".* 2>/dev/null || echo "(no nsys output found at ${NSYS_OUT})"
echo
echo "To copy to local machine:"
echo "  rsync -avh adovlatyan@cluster.ysu.am:${NSYS_OUT}.nsys-rep ./"
echo "Or run: ~/scripts/rsync-nsys ${SLURM_JOB_ID}"
