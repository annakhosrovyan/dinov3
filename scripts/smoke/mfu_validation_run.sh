#!/bin/bash
#SBATCH --job-name=dinov3-mfu-val
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:2
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/mfu-val-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/mfu-val-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "Starting MFU validation run"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"

mkdir -p /mnt/weka/adovlatyan/logs

torchrun --nproc_per_node=2 dinov3/train/train.py \
  --config-file dinov3/configs/ssl_default_config.yaml \
  --output-dir ./output_mfu_val_${SLURM_JOB_ID} \
  student.arch=vit_base \
  student.in_chans=5 \
  teacher.in_chans=5 \
  student.pretrained_weights="" \
  "train.dataset_path=MixedSatelliteDataset:intelinair_data_path=/mnt/weka/adovlatyan/synthetic_intelinair.h5" \
  train.batch_size_per_gpu=32 \
  train.num_workers=8 \
  train.OFFICIAL_EPOCH_LENGTH=100 \
  optim.epochs=1 \
  train.persistent_workers=false \
  train.prefetch_factor=4 \
  train.cache_dataset=false \
  train.compile=true \
  wandb.enabled=false \
  checkpointing.period=99999
