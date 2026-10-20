#!/bin/bash
# =============================================================================
# Phase 7.2 — CUDA_DEVICE_MAX_CONNECTIONS sweep (DDP + cudagraphs, bs=128)
# =============================================================================
# WHY: the 75082 nsys trace showed NCCL ~93% EXPOSED (comm not overlapping with
# compute). Hypothesis (H1, Codex-ranked #1 info/GPU-hr): the project-wide
# `export CUDA_DEVICE_MAX_CONNECTIONS=1` — inherited from the FSDP2/Megatron-era
# recipe — collapses all CUDA streams onto ONE hardware work queue, serializing
# the NCCL stream behind the compute stream and DESTROYING comm/compute overlap.
# For pure DDP (no tensor parallelism) =1 is likely unnecessary and possibly
# harmful. Default 8 is CUDA's own default (NVIDIA: 1..32, default 8 sans MPS).
#
# DESIGN: a controlled single-variable sweep. Everything else is pinned to the
# torch-2.10 controlled baseline (job 67639: ddp_memval_soak.sh @ bs=128,
# 24w/8pf, 1500G/192CPU, ~1,640 img/s run-avg, 2000/2000 PASS). The ONLY thing
# this script varies is CUDA_DEVICE_MAX_CONNECTIONS (env-overridable below).
# Submit one job per value; compare steady-state img/s + wall iter_time.
#
#   sbatch scripts/screening/ddp_maxconn_sweep.sh                 # =8 (default)
#   MAX_CONN=1  sbatch scripts/screening/ddp_maxconn_sweep.sh     # =1 re-baseline (matched control)
#   MAX_CONN=16 sbatch scripts/screening/ddp_maxconn_sweep.sh
#   MAX_CONN=32 sbatch scripts/screening/ddp_maxconn_sweep.sh
#
# NOTE (Codex caveat): with cudagraphs the stream topology is captured at
# capture time, so the env var governs how those captured streams map to queues.
# Expected outcome per review: flat-to-single-digit gain more likely than a big
# win; double-digit only if stream serialization is genuinely dominant. We
# report BOTH CUDA-event img/s AND wall iter_time (the nsys run is slower than
# champion; trust shares + wall, not absolute CUDA-event ms alone).
# =============================================================================
#SBATCH --job-name=dinov3-maxconn-sweep
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=1500G
#SBATCH --gres=gpu:h100:8
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/%x_%j.out

set -uo pipefail

ENV=/mnt/weka/adovlatyan/.conda/envs/dinov3_env_210clone
export PATH="$ENV/bin:$PATH"
export CONDA_PREFIX="$ENV"
export PYTHONNOUSERSITE=1
export PYTHONPATH=.

# ---- THE swept variable ---------------------------------------------------
MAX_CONN="${MAX_CONN:-8}"
export CUDA_DEVICE_MAX_CONNECTIONS="${MAX_CONN}"

# Pinned to the 67639 baseline (do NOT vary these in this sweep).
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
unset PYTORCH_CUDA_ALLOC_CONF
unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

# ---- Config (pinned to 67639) ---------------------------------------------
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-24}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"
ITERS="${ITERS:-700}"                  # steady-state by ~iter 200; 700 keeps the run cheap
SKIP_ITERS="${SKIP_ITERS:-200}"        # warmup/compile iters to drop from the steady-state mean
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"

OUTPUT_ROOT="${DINOV3_OUTPUT_ROOT:-/mnt/weka/$(whoami)}"
RUN_TAG="maxconn${MAX_CONN}_ddp_bs${BATCH_SIZE}_${NUM_WORKERS}w${PREFETCH_FACTOR}pf"
OUTPUT_DIR="${OUTPUT_ROOT}/output_${RUN_TAG}_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_ROOT}" "/mnt/weka/adovlatyan/logs"

echo "=== ${RUN_TAG} (CUDA_DEVICE_MAX_CONNECTIONS sweep) ==="
echo "Job ID:                       ${SLURM_JOB_ID}"
echo "Node:                         ${SLURM_NODELIST}"
echo "CUDA_DEVICE_MAX_CONNECTIONS:  ${CUDA_DEVICE_MAX_CONNECTIONS}   <-- swept variable"
echo "Config:                       DDP, bs=${BATCH_SIZE}, compile=true, cudagraphs=true, AC=off"
echo "Loader:                       num_workers=${NUM_WORKERS}, prefetch_factor=${PREFETCH_FACTOR}"
echo "Schedule:                     ${ITERS} iters (no ckpt/eval); steady-state mean skips first ${SKIP_ITERS}"
echo "Baseline to beat:             job 67639 (=1) ~1,640 img/s run-avg @ bs128/24w/8pf"
echo "Output:                       ${OUTPUT_DIR}"
echo "Date:                         $(date)"
echo ""

# ckpt/eval periods set beyond ITERS so neither fires — pure throughput.
torchrun --nproc_per_node=8 dinov3/train/train.py \
  --config-file dinov3/configs/ssl_default_config.yaml \
  --output-dir "${OUTPUT_DIR}" \
  student.arch=vit_base \
  student.in_chans=5 \
  teacher.in_chans=5 \
  student.pretrained_weights="${PRETRAINED_WEIGHTS}" \
  "train.dataset_path=MixedSatelliteDataset:\
intelinair_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/intelinair/intelinair.h5:\
maid_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/maid:\
sen1_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/sentinel1:\
sen1_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/sentinel1_stats:\
naip_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas-dataset-v1-naip-2020/naip:\
naip_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/naip_stats:\
naip_weight=1.0" \
  train.batch_size_per_gpu="${BATCH_SIZE}" \
  train.OFFICIAL_EPOCH_LENGTH="${ITERS}" \
  optim.epochs=1 \
  train.num_workers="${NUM_WORKERS}" \
  train.prefetch_factor="${PREFETCH_FACTOR}" \
  train.persistent_workers=true \
  train.cache_dataset=true \
  train.compile=true \
  train.cudagraphs=true \
  train.checkpointing=false \
  train.checkpointing_full=false \
  train.distributed_strategy=ddp \
  checkpointing.period=999999 \
  evaluation.eval_period_iterations=999999 \
  wandb.enabled=false

RC=$?
echo ""
echo "=== ${RUN_TAG} complete (rc=${RC}): $(date) ==="

# ---- Steady-state self-report (machine-readable training_metrics.json) ------
python3 - "${OUTPUT_DIR}/training_metrics.json" "${SKIP_ITERS}" "${MAX_CONN}" "${BATCH_SIZE}" <<'PY' || echo "(metrics parse failed — inspect training_metrics.json)"
import json, sys, statistics
path, skip, maxconn, bs = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
rows = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
ss = [r for r in rows if r.get("iteration", 0) >= skip]
def col(key):
    return [r[key] for r in ss if key in r and r[key] is not None]
def rep(key, fmt="{:.1f}"):
    v = col(key)
    return fmt.format(statistics.mean(v)) if v else "n/a"
print(f"\n--- STEADY-STATE (iter >= {skip}, n={len(ss)} log-points) | CUDA_DEVICE_MAX_CONNECTIONS={maxconn} ---")
print(f"  images_per_sec (CUDA-event) : {rep('images_per_sec')}")
print(f"  mfu %%                       : {rep('mfu')}")
print(f"  step_time_ms (CUDA-event)   : {rep('step_time_ms','{:.2f}')}")
print(f"  iter_time s  (WALL)         : {rep('iter_time','{:.4f}')}   <-- Codex caveat: trust this too")
itw = col('iter_time')
if itw:
    bs_total = bs * 8
    print(f"  wall img/s  (=1024/iter_time): {bs_total/statistics.mean(itw):.1f}")
PY
echo ""
echo "Compare across MAX_CONN values: grep 'images_per_sec (CUDA-event)' and 'wall img/s' from each job's log."
