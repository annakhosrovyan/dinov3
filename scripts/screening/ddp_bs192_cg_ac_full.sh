#!/bin/bash
# Phase 6.A.6 — DDP bs=192, AC=full, cudagraphs=true.
#
# THE ONE THING THIS RUN IS TESTING:
#   At what batch size does AC=full + cudagraphs start beating no-AC + cudagraphs?
#   bs=128 AC=full = 1,654 img/s; bs=128 no-AC = 2,394 img/s — AC is losing.
#   Extrapolation says bs≈224 break-even, bs≈256 first solid win. But bs=192 no-AC
#   OOM'd from cudagraph workspace bloat (53739), so the memory ceiling under
#   AC=full at larger batches is uncharacterized.
#
# Comparators:
#   - 53708 (bs=128, AC=off, cg=true): 2,394 img/s, 13.70% MFU, 34.1 GB peak  ← current champ
#   - 53739 (bs=192, AC=off, cg=true): OOM at >80 GB                          ← the wall
#   - 54000 (bs=128, AC=full, cg=true): 1,654 img/s, 9.46% MFU, 12.4 GB peak  ← AC=full baseline
#   - 53999 (bs=96,  AC=full, cg=true): 1,378 img/s, 7.89% MFU,  9.6 GB peak
#
# Predicted from linear slope (8.625 img/s per unit batch, 0.0875 GB per unit batch):
#   img/s ≈ 2,206 (-7.8% vs 53708)
#   mem   ≈ 18.0 GB (best case — assumes cudagraph workspace also benefits from AC)
#   mem   ≈ 58 GB   (worst case — applying no-AC cudagraph bloat slope)
#
# Outcomes we're discriminating between:
#   A) img/s >= 2,394 (beats no-AC bs=128): AC=full becomes the production path,
#      next test bs=224 or bs=256.
#   B) img/s < 2,394 but mem << 80 GB: characterize slope to bs=224. Predict crossover.
#   C) OOM: cudagraph workspace ceiling is the binding constraint, not activations.
#      Closes the AC-as-batch-unlocker thesis. bs=128 no-AC stays the champion.
#
#SBATCH --job-name=dinov3-ddp-bs192-cg-ac-full
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/ddp-bs192-cg-ac-full-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/ddp-bs192-cg-ac-full-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

unset PYTORCH_CUDA_ALLOC_CONF
unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

BATCH_SIZE=192
RUN_TAG="ddp_bs${BATCH_SIZE}_cg_ac_full"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"

mkdir -p /mnt/weka/adovlatyan/logs

echo "=== ${RUN_TAG} ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "Config:    DDP, bs=${BATCH_SIZE}, compile=true, cudagraphs=true, checkpointing=true, checkpointing_full=true"
echo "Comparator: 54000 (bs=128, AC=full, cg=true) — 1654 img/s, 9.46% MFU, 12.4 GB"
echo "Goal: characterize AC=full at bs=192 to find crossover vs no-AC bs=128 (2394 img/s)."
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
  train.cudagraphs=true \
  train.checkpointing=true \
  train.checkpointing_full=true \
  train.distributed_strategy=ddp \
  train.sharded_eval_checkpoint=true \
  checkpointing.period=400 \
  evaluation.eval_period_iterations=400 \
  wandb.enabled=false

echo ""
echo "=== ${RUN_TAG} complete: $(date) ==="
grep "Training" /mnt/weka/adovlatyan/logs/ddp-bs192-cg-ac-full-${SLURM_JOB_ID}.out \
  | awk 'NR>20 {print}' | tail -20 || true
echo ""
grep "\[COMPILE\]" /mnt/weka/adovlatyan/logs/ddp-bs192-cg-ac-full-${SLURM_JOB_ID}.out || true
grep -E "selective checkpointing|using.*checkpointing" /mnt/weka/adovlatyan/logs/ddp-bs192-cg-ac-full-${SLURM_JOB_ID}.out || true
