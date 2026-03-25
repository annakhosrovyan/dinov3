#!/usr/bin/env bash
#SBATCH --job-name=dinov3-satellite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

conda activate dinov3
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8

torchrun --nproc_per_node=8 dinov3/train/train.py \
  --config-file dinov3/configs/ssl_default_config.yaml \
  --output-dir ./output_satellite_s1_s2ab \
  student.arch=vit_base \
  student.in_chans=5 \
  teacher.in_chans=5 \
  student.pretrained_weights=/auto/home/anna.khosrovyan/dinov3/pretrained_weights/dinov3_vitb16_pretrain.pth \
  train.dataset_path='MixedSatelliteDataset:'\
'intelinair_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/intelinair/intelinair.h5:'\
'maid_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/maid:'\
'sen1_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/sentinel1:'\
'sen1_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/sentinel1_stats:'\
'naip_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas-dataset-v1-naip-2020/naip:'\
'naip_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/naip_stats:'\
'naip_weight=1.0' \
  train.batch_size_per_gpu=64 \
  train.num_workers=20 \
  train.OFFICIAL_EPOCH_LENGTH=23412 \
  optim.epochs=10 \
  train.persistent_workers=true \
  train.prefetch_factor=8 \
  train.cache_dataset=true \
  wandb.enabled=true \
  wandb.project=dinov3-satellite \
  wandb.run_name=satellite_8xh100 \
  wandb.group=satellite_only
