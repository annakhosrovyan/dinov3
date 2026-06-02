#!/bin/bash
# Capture the rank-6 / iter~1310 crash exception. main() is now @record-decorated and
# torchrun runs with --tee=3 --redirects=3 --log-dir so each rank's stdout+stderr +
# the real traceback land in per-rank files. forward_backward is wrapped with a
# [STEPDIAG] batch dumper. no-AC + eval/ckpt off so we reach iter ~1310 fast (~15 min).
# Sampler stream matches the crashing soaks (default seed, bs=128, world=8).
#SBATCH --job-name=dinov3-rank6-capture
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=320G
#SBATCH --gres=gpu:h100:8
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/rank6-capture-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/rank6-capture-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
# NOTE: deliberately NOT setting CUDA_LAUNCH_BLOCKING=1 here — it can interfere with
# CUDA-graph capture/replay and risks altering or masking the crash. Keep this run
# faithful to the crashing config; if we get a deferred async CUDA error, rerun with
# cudagraphs=false + CUDA_LAUNCH_BLOCKING=1 to localize it.

unset PYTORCH_CUDA_ALLOC_CONF
unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

OUTPUT_DIR="/mnt/weka/adovlatyan/output_rank6_capture_${SLURM_JOB_ID}"
LOGDIR="/mnt/weka/adovlatyan/logs/rank6-capture-${SLURM_JOB_ID}-perrank"
mkdir -p /mnt/weka/adovlatyan/logs "${LOGDIR}"

echo "=== rank6-capture job ${SLURM_JOB_ID} on ${SLURM_NODELIST} $(date) ==="
echo "Per-rank logs + tracebacks -> ${LOGDIR}"

torchrun --nproc_per_node=8 --redirects=3 --tee=3 --log-dir="${LOGDIR}" \
  dinov3/train/train.py \
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
  train.num_workers=12 \
  train.OFFICIAL_EPOCH_LENGTH=4000 \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=4 \
  train.cache_dataset=true \
  train.compile=true \
  train.cudagraphs=true \
  train.checkpointing=false \
  train.distributed_strategy=ddp \
  checkpointing.period=100000 \
  evaluation.eval_period_iterations=100000 \
  wandb.enabled=false

echo "=== finished (no crash?) $(date) ==="
echo "--- grep tracebacks across per-rank logs ---"
grep -rl "Traceback\|STEPDIAG\|Error\|error_file" "${LOGDIR}" || true
