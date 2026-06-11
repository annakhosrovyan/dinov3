#!/bin/bash
# Phase 6.B.2.b — Loader sweep point 2: DDP bs=128, AC=full, 16 workers / 4 prefetch.
#
# CONTEXT: Two prior datapoints on bs=128 + AC=full + cudagraphs:
#   - 12w/4pf (job 58188): 300 GB cgroup exact, wall 1,158 img/s, GPU idle 61%.
#   - 20w/8pf (job 57799): OOM at --mem=380G before any iter ran. Needed >>380G.
# This run probes the middle: more workers (closes data-rate gap) at modest prefetch
# (controls memory).
#
# HYPOTHESIS: num_workers is the binding throughput knob (controls decode rate).
# Going 12→16 (+33% worker count) should drop p50 data_time from ~100 ms toward
# <30 ms, lifting wall throughput from 1,158 → ~1,500-1,700 img/s.
#
# MEMORY PREDICTION (from 58188 fit: 1.75 GB/worker × 8 ranks + 240 MB × batch × 8):
#   worker overhead:  16 × 8 × 1.75 = 224 GB
#   batch buffers:    16 × 4 × 8 × 0.24 = 123 GB
#   baseline:         ~40 GB
#   total est:        ~387 GB → --mem=430G (with headroom)
#
# DECISION CRITERION:
#   - If wall throughput >= 1,500 img/s with data_time p50 < 50 ms:
#       worker-count was the binding constraint; this is the better Pareto point.
#   - If wall throughput stays near 1,200 img/s:
#       data pipeline is variance-limited (tail latencies), not mean-decode-limited;
#       need higher prefetch (then 16w/8pf is the better test).
#
#SBATCH --job-name=dinov3-soak-128-acf-16w4pf
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=430G
#SBATCH --gres=gpu:h100:8
#SBATCH --time=01:45:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/soak-128-acf-16w4pf-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/soak-128-acf-16w4pf-%j.err

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
NUM_WORKERS=16
PREFETCH=4
RUN_TAG="soak_ddp_bs${BATCH_SIZE}_acf_${NUM_WORKERS}w${PREFETCH}pf"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_${RUN_TAG}_${SLURM_JOB_ID}"
MEMLOG="/mnt/weka/adovlatyan/logs/${RUN_TAG}-${SLURM_JOB_ID}-memlog.jsonl"

mkdir -p /mnt/weka/adovlatyan/logs

# Cgroup-aware memory sampler (NOT /proc/meminfo — that's node-wide and misleading
# when sharing a node with other jobs). Falls back to cgroup v1 if v2 not present.
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
echo "Config:    DDP, bs=${BATCH_SIZE}, compile=true, cudagraphs=true, AC=full"
echo "Loader:    num_workers=${NUM_WORKERS}, prefetch_factor=${PREFETCH}"
echo "Memory:    --mem=430G (predicted ~387 GB usage)"
echo "Duration:  4000 iters, 8 checkpoint events (period=500), 4 eval events (period=1000)"
echo "Comparators: 58188 (12w/4pf): 1,158 wall img/s, 300 GB cgroup, data_time p50=100ms"
echo "             54000 screen at 20w/8pf: 1,654 MetricLogger img/s, 12.4 GB VRAM"
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
  train.batch_size_per_gpu=${BATCH_SIZE} \
  train.num_workers=${NUM_WORKERS} \
  train.OFFICIAL_EPOCH_LENGTH=4000 \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor=${PREFETCH} \
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
grep '\[MEMPROFILE\]' "/mnt/weka/adovlatyan/logs/soak-128-acf-16w4pf-${SLURM_JOB_ID}.out" \
  | grep 'rank=0' || true
echo ""
echo "--- Steady-state MFU (iters 100+) ---"
grep 'images_per_sec' "/mnt/weka/adovlatyan/logs/soak-128-acf-16w4pf-${SLURM_JOB_ID}.out" \
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
