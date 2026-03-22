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

# torchrun --nproc_per_node=8 dinov3/train/train.py \
#   --config-file dinov3/configs/ssl_default_config.yaml \
#   --output-dir ./output_satellite_s1_s2ab_naip_maid_intelinair \
#   student.arch=vit_base \
#   student.in_chans=5 \
#   teacher.in_chans=5 \
#   student.pretrained_weights=/auto/home/anna.khosrovyan/dinov3/pretrained_weights/dinov3_vitb16_pretrain.pth \
#   train.dataset_path='MixedSatelliteDataset:'\
# 'sen1_data_path=/nfs/h100/raid/rs/satlas_dataset/sentinel1:'\
# 'sen1_stats_dir=/nfs/h100/raid/rs/satlas_dataset/stats/sentinel1_stats:'\
# 'sen2a_data_path=/nfs/h100/raid/rs/satlas_dataset/sentinel2a:'\
# 'sen2a_stats_dir=/nfs/h100/raid/rs/satlas_dataset/stats/sen2a_stats:'\
# 'sen2b_data_path=/nfs/h100/raid/rs/satlas_dataset/sentinel2b:'\
# 'sen2b_stats_dir=/nfs/h100/raid/rs/satlas_dataset/stats/sen2b_stats:'\
# 'intelinair_data_path=/nfs/ap/mnt/frtn/rs-multiband/intelinair.h5:'\
# 'maid_data_path=/nfs/h100/raid/rs/maid:'\
# 'naip_data_path=/nfs/ap/mnt/frtn/rs-multiband/satlas-dataset-v1-naip-2020/naip:'\
# 'naip_stats_dir=/nfs/h100/raid/rs/satlas_dataset/stats/naip_stats:'\
# 'naip_weight=1.0' \
#   train.batch_size_per_gpu=16 \
#   train.num_workers=8 \
#   wandb.enabled=true \
#   wandb.project=dinov3-satellite \
#   wandb.run_name=full_satellite_s1_s2ab_naip_maid_intelinair \
#   wandb.group=full_train


rchrun --nproc_per_node=8 dinov3/train/train.py \
  --config-file dinov3/configs/ssl_default_config.yaml \
  --output-dir ./output_satellite_s1_s2ab_naip_maid_intelinair \
  student.arch=vit_base \
  student.in_chans=5 \
  teacher.in_chans=5 \
  student.pretrained_weights=/auto/home/anna.khosrovyan/dinov3/pretrained_weights/dinov3_vitb16_pretrain.pth \
  train.dataset_path='MixedSatelliteDataset:'\
'sen1_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/sentinel1:'\
'sen1_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/stats/sentinel1_stats:'\
'sen2a_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/sentinel2a:'\
'sen2a_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/stats/sen2a_stats:'\
'sen2b_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/sentinel2b:'\
'sen2b_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/stats/sen2b_stats:'\
'intelinair_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/intelinair/intelinair.h5:'\
'maid_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/maid:'\
'naip_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas-dataset-v1-naip-2020/naip:'\
'naip_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/stats/naip_stats:'\
'naip_weight=1.0' \
  train.batch_size_per_gpu=16 \
  train.num_workers=8 \
  wandb.enabled=true \
  wandb.project=dinov3-satellite \
  wandb.run_name=full_satellite_s1_s2ab_naip_maid_intelinair \
  wandb.group=full_train
