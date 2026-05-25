#!/bin/bash
#SBATCH --job-name=dinov3-profile
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/profile-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/profile-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Profiling output under Weka
PROFILE_DIR="/mnt/weka/adovlatyan/profiler_traces/$(date +%Y-%m-%d)/${SLURM_JOB_ID}"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_profile_${SLURM_JOB_ID}"
mkdir -p "${PROFILE_DIR}" /mnt/weka/adovlatyan/logs

echo "=== DINOv3 Profiling Run ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Profile dir: ${PROFILE_DIR}"
echo "Date: $(date)"

# Short run: 15 iters total
#   - warmup=5 (PyTorch profiler warmup, separate from torch.compile warmup)
#   - active=3 (profiler records these iterations)
#   - repeat=1
# The first ~1 iteration is torch.compile warmup (~35s), so the profiler
# warmup=5 ensures we skip that plus a few more stabilization iters.
# Total: OFFICIAL_EPOCH_LENGTH=15, epochs=1 → 15 iterations
torchrun --nproc_per_node=8 dinov3/train/train.py \
  --config-file dinov3/configs/ssl_default_config.yaml \
  --output-dir "${OUTPUT_DIR}" \
  --profiling \
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
  train.batch_size_per_gpu=64 \
  train.num_workers=20 \
  train.OFFICIAL_EPOCH_LENGTH=15 \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  train.compile=true \
  wandb.enabled=false \
  checkpointing.period=99999

# Copy traces to the canonical profiling directory
if [ -d "${OUTPUT_DIR}/profiler_traces" ]; then
  cp -r "${OUTPUT_DIR}/profiler_traces/"* "${PROFILE_DIR}/" 2>/dev/null || true
  echo "Traces copied to: ${PROFILE_DIR}"
fi

echo "=== Profiling complete: $(date) ==="
echo "Output dir: ${OUTPUT_DIR}"
echo "Profile dir: ${PROFILE_DIR}"
