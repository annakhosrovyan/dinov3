#!/bin/bash
# Phase 6.B.2 — Sustainability soak: DDP bs=128, cudagraphs=true, AC=full.
#
# OBJECTIVE: Validate that the bs=128 AC=full config (Phase 6.A.5.d, job 54000)
# — which uses 12.4 GB VRAM vs 34.1 GB for no-AC — is also production-sustainable.
# If AC keeps VRAM stable, it becomes the preferred config when training with
# checkpoints or when co-locating multiple jobs on a node.
#
# AC=full re-executes the entire transformer block forward during backward pass.
# This reduces saved activation tensors to near-zero but increases compute ~31%.
# Peak VRAM is dominated by the cudagraph workspace + model params, not activations.
#
# WHAT THIS RUN COVERS (identical schedule to 6.B.1):
#   - 4000 iters, 8 checkpoint saves (period=500), 4 eval runs (period=1000)
#   - MEMPROFILE markers: pre/post checkpoint + eval
#   - Shell sidecar: host RAM sampled every 10s
#
# EXPECTED OUTCOME:
#   - VRAM: ~12-15 GB peak (64% less than no-AC, per 6.A.5.d measurement)
#   - img/s: ~1,600-1,700 (31% slower than no-AC, per 6.A.5.d: 1,654 img/s)
#   - MFU: ~9-10%
#
# DECISION CRITERION post-soak:
#   If VRAM stable AND img/s stable → AC=full is the safe-mode config:
#     use when VRAM budget is tight (co-location, future larger models).
#   If OOM or severe throughput degradation over time → no-AC only path.
#
# Comparator: job 54000 (1000-iter screen) — 1654 img/s, 9.46% MFU, 12.4 GB
#
#SBATCH --job-name=dinov3-soak-bs128-cg-ac
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --gres=gpu:h100:8
#SBATCH --time=01:45:00
#
# MEMORY NOTE (2026-05-25): job 57799 OOM-killed at --mem=380G during compile
# warmup before any training iter ran. Cgroup IS strictly enforced. The actual
# DataLoader+Python heap at bs=128 with 20 workers / prefetch=8 exceeds 380 GB.
# Mitigation below: reduced loader to 12 workers / prefetch=4 (3.3x less in flight).
# Expected actual usage: ~150-200 GB, well under the 300 GB cgroup limit.
#SBATCH --output=/mnt/weka/adovlatyan/logs/soak-bs128-cg-ac-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/soak-bs128-cg-ac-%j.err

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

BATCH_SIZE=128
RUN_TAG="soak_ddp_bs128_cg_ac_full"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"
MEMLOG="/mnt/weka/adovlatyan/logs/${RUN_TAG}-${SLURM_JOB_ID}-memlog.jsonl"

mkdir -p /mnt/weka/adovlatyan/logs

# Read THIS job's cgroup memory usage (NOT node-wide /proc/meminfo, which would
# include other jobs sharing the node and give a misleading view of our usage).
# Tries cgroup v2 first, falls back to v1. Records both cgroup and node-wide
# numbers for context.
read_cgroup_bytes() {
    local cg_self; cg_self=$(awk -F: '$2 == "" {print $3}' /proc/self/cgroup 2>/dev/null)
    if [ -r "/sys/fs/cgroup${cg_self}/memory.current" ]; then
        cat "/sys/fs/cgroup${cg_self}/memory.current"; return
    fi
    local cg_v1; cg_v1=$(awk -F: '/memory/{print $3}' /proc/self/cgroup 2>/dev/null)
    if [ -n "$cg_v1" ] && [ -r "/sys/fs/cgroup/memory${cg_v1}/memory.usage_in_bytes" ]; then
        cat "/sys/fs/cgroup/memory${cg_v1}/memory.usage_in_bytes"; return
    fi
    echo 0
}

host_ram_monitor() {
    local logfile="$1"
    while true; do
        local ts; ts=$(date +%s)
        local node_total node_avail node_used cgroup_bytes cgroup_mb
        node_total=$(awk '/MemTotal/  {printf "%.0f", $2/1024}' /proc/meminfo)
        node_avail=$(awk '/MemAvailable/ {printf "%.0f", $2/1024}' /proc/meminfo)
        node_used=$(( node_total - node_avail ))
        cgroup_bytes=$(read_cgroup_bytes)
        cgroup_mb=$(( cgroup_bytes / 1048576 ))
        local gpu_used
        gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
                   | tr '\n' ',' | sed 's/,$//')
        printf '{"ts":%d,"cgroup_used_mb":%d,"node_used_mb":%d,"node_total_mb":%d,"gpu_used_mb":[%s]}\n' \
               "$ts" "$cgroup_mb" "$node_used" "$node_total" "$gpu_used" >> "$logfile"
        sleep 10
    done
}

echo "=== ${RUN_TAG} ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "Config:    DDP, bs=128, compile=true, cudagraphs=true, AC=full"
echo "Duration:  4000 iters, 8 checkpoint events (period=500), 4 eval events (period=1000)"
echo "Comparator: 54000 (1000-iter screen) — 1654 img/s, 9.46% MFU, 12.4 GB"
echo "Memlog:    ${MEMLOG}"
echo "Date: $(date)"
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
  train.batch_size_per_gpu=128 \
  train.num_workers=12 \
  train.OFFICIAL_EPOCH_LENGTH=4000 \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=4 \
  train.cache_dataset=true \
  train.compile=true \
  train.cudagraphs=true \
  train.checkpointing=true \
  train.checkpointing_full=true \
  train.distributed_strategy=ddp \
  train.sharded_eval_checkpoint=true \
  checkpointing.period=500 \
  evaluation.eval_period_iterations=1000 \
  wandb.enabled=false

echo ""
echo "=== ${RUN_TAG} complete: $(date) ==="
echo ""
echo "--- VRAM phase markers (rank 0) ---"
grep '\[MEMPROFILE\]' "/mnt/weka/adovlatyan/logs/soak-bs128-cg-ac-${SLURM_JOB_ID}.out" \
  | grep 'rank=0' || true
echo ""
echo "--- Steady-state MFU (iters 100+) ---"
grep 'images_per_sec' "/mnt/weka/adovlatyan/logs/soak-bs128-cg-ac-${SLURM_JOB_ID}.out" \
  | awk 'NR > 20' | tail -10 || true
echo ""
echo "--- Host RAM memlog: ${MEMLOG} ---"
python3 -c "
import json
data = [json.loads(l) for l in open('${MEMLOG}')]
peak_cg = max(d['cgroup_used_mb'] for d in data)
peak_node = max(d['node_used_mb'] for d in data)
print(f'  Peak cgroup (our job): {peak_cg:,} MB ({peak_cg/1024:.1f} GB)')
print(f'  Peak node-wide:        {peak_node:,} MB ({peak_node/1024:.1f} GB)')
print(f'  Samples: {len(data)}')
" 2>/dev/null || true
