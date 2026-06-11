#!/bin/bash
# Phase 6.B.3 — Exploratory soak: DDP bs=192, cudagraphs=true, no AC, REDUCED LOADER.
#
# OBJECTIVE: Test whether bs=192 is viable once the host-RAM blocker is removed.
# The Phase 6.A.4.d (job 53739) and 6.A.6 (job 56131) OOMs were *host RAM*,
# NOT CUDA VRAM — the DataLoader prefetch pool exhausted the Slurm cgroup budget.
#
# FIX APPLIED:
#   num_workers: 20 → 12   (40% fewer workers)
#   prefetch_factor: 8 → 4 (50% smaller per-worker queue)
#   Combined: 12×4=48 batches/rank in flight vs 160/rank previously — 3.3× less.
#   Trade-off: may introduce loader stalls (data pipeline becomes the bottleneck
#   at some step rate); NVTX profiling of data_time will reveal if this fires.
#
# ONLY RUN THIS AFTER 6.B.1 PASSES CLEANLY.
# bs=192 is exploratory; bs=128 remains the production candidate until validated.
#
# EXPECTED OUTCOME (if no OOM):
#   - VRAM: ~51 GB (linear extrapolation from bs=128 at 34.1 GB)
#   - img/s: ~3,000-3,200 (linear extrapolation: +58% over bs=128)
#   - MFU: ~17-19%
#   If these hold → significant win over bs=128. Warrants longer soak.
#   If VRAM spikes non-linearly > 70 GB → cudagraph workspace bloat at large batch.
#   If data_time >> step_time → loader stall from reduced workers.
#
# SHORTER RUN (2000 iters, 4 checkpoint saves) — exploratory, not a full soak.
# If it survives clean, promote to 4000-iter soak.
#
#SBATCH --job-name=dinov3-soak-bs192-cg
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/soak-bs192-cg-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/soak-bs192-cg-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

export DINOV3_MEMORY_PROFILE=1
export DINOV3_MEMORY_PROFILE_PERIOD=50

unset PYTORCH_CUDA_ALLOC_CONF
unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

BATCH_SIZE=192
RUN_TAG="soak_ddp_bs192_cg"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"
MEMLOG="/mnt/weka/adovlatyan/logs/${RUN_TAG}-${SLURM_JOB_ID}-memlog.jsonl"

mkdir -p /mnt/weka/adovlatyan/logs

host_ram_monitor() {
    local logfile="$1"
    while true; do
        local ts; ts=$(date +%s)
        local mem_total mem_avail mem_used
        mem_total=$(awk '/MemTotal/  {printf "%.0f", $2/1024}' /proc/meminfo)
        mem_avail=$(awk '/MemAvailable/ {printf "%.0f", $2/1024}' /proc/meminfo)
        mem_used=$(( mem_total - mem_avail ))
        local gpu_used
        gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
                   | tr '\n' ',' | sed 's/,$//')
        printf '{"ts":%d,"host_ram_used_mb":%d,"host_ram_total_mb":%d,"gpu_used_mb":[%s]}\n' \
               "$ts" "$mem_used" "$mem_total" "$gpu_used" >> "$logfile"
        sleep 10
    done
}

echo "=== ${RUN_TAG} ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "Config:    DDP, bs=192, compile=true, cudagraphs=true, AC=off"
echo "DataLoader: num_workers=12, prefetch_factor=4 (REDUCED from 20/8 to clear host-RAM)"
echo "Duration:  2000 iters (exploratory), 4 checkpoint events (period=500)"
echo "Comparator: 53708 (bs=128, no-AC) — 2394 img/s, 13.70% MFU, 34.1 GB"
echo "Memlog:    ${MEMLOG}"
echo "Date: $(date)"
echo ""
echo "PREREQUISITE: Run 6.B.1 (ddp_bs128_cg_soak.sh) first to confirm bs=128 baseline."
echo "This is an exploratory run — if it OOMs, the MEMLOG will reveal whether it's"
echo "CUDA (gpu_used_mb near 80 GB) or host RAM (host_ram_used_mb near limit)."
echo ""

host_ram_monitor "${MEMLOG}" &
MONITOR_PID=$!
trap "kill ${MONITOR_PID} 2>/dev/null || true" EXIT

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
  train.num_workers=12 \
  train.OFFICIAL_EPOCH_LENGTH=2000 \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=4 \
  train.cache_dataset=true \
  train.compile=true \
  train.cudagraphs=true \
  train.checkpointing=false \
  train.distributed_strategy=ddp \
  train.sharded_eval_checkpoint=true \
  checkpointing.period=500 \
  evaluation.eval_period_iterations=1000 \
  wandb.enabled=false

echo ""
echo "=== ${RUN_TAG} complete: $(date) ==="
echo ""
echo "--- VRAM phase markers (rank 0) ---"
grep '\[MEMPROFILE\]' "/mnt/weka/adovlatyan/logs/soak-bs192-cg-${SLURM_JOB_ID}.out" \
  | grep 'rank=0' || true
echo ""
echo "--- Steady-state MFU (iters 100+) ---"
grep 'images_per_sec' "/mnt/weka/adovlatyan/logs/soak-bs192-cg-${SLURM_JOB_ID}.out" \
  | awk 'NR > 20' | tail -10 || true
echo ""
echo "--- data_time vs step_time (check for loader stalls from reduced workers) ---"
grep 'data_time' "/mnt/weka/adovlatyan/logs/soak-bs192-cg-${SLURM_JOB_ID}.out" \
  | awk 'NR > 20' | tail -5 || true
echo ""
echo "--- Host RAM memlog: ${MEMLOG} ---"
python3 -c "
import json
data = [json.loads(l) for l in open('${MEMLOG}')]
peak = max(d['host_ram_used_mb'] for d in data)
peak_gpu = [max(d['gpu_used_mb'][i] for d in data) for i in range(len(data[0]['gpu_used_mb']))]
print(f'  Peak host RAM used:  {peak:,} MB ({peak/1024:.1f} GB)')
print(f'  Peak GPU used (max): {max(peak_gpu):,} MB ({max(peak_gpu)/1024:.1f} GB)')
print(f'  Per-GPU peaks: {peak_gpu}')
" 2>/dev/null || true
