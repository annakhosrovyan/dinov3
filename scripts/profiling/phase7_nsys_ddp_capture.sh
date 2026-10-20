#!/bin/bash
# Phase 7.1 — Nsight Systems capture of the DDP+cudagraphs champion config (residual attribution).
#
# OBJECTIVE: attribute the per-iteration wall-time residual that keeps MFU flat (~20-21%)
# across bs=128/160/192 (jobs 67639/67153/67154). All Phase 7 numbers so far are matmul-MAC
# estimates; this capture measures actual wall shares: NVTX range breakdown, GPU busy vs gap,
# NCCL exposure, head/loss wall share. See docs/phase7_perf_plan.md §7.1.
#
# SINGLE-COMMAND PIPELINE: capture -> sqlite export -> phase7_nsys_report.py analysis,
# all inside this job. The report (.md) and machine-readable metrics (.json) land next to
# the trace — no manual log decoding afterward.
#
# Champion recipe is copied from scripts/soak/ddp_bs128_cg_soak.sh (jobs 53708/60959),
# run on the torch-2.10 canonical env (dinov3_env_210clone — matches Anna's stack).
# NVTX ranges are enabled via train.py --profiling (no-op cost when nsys is not attached).
#
# Usage:
#   sbatch scripts/profiling/phase7_nsys_ddp_capture.sh                  # champion: bs=128
#   BATCH_SIZE=160 NUM_WORKERS=24 sbatch --cpus-per-task=192 --mem=1500G \
#     scripts/profiling/phase7_nsys_ddp_capture.sh                       # differential pair
#
#   NOTE on bs>128: must be submitted with the validated provisioning (192 CPU / 1500G,
#   NUM_WORKERS=24) or the DataLoader prefetch budget OOMs host RAM (phase6 §6.B.6).
#   For a clean bs=128-vs-160 differential, capture BOTH at the same loader settings.
#
# LESSONS FROM JOB 71041 (2026-06-12, failed first attempt):
#   - Startup is ~480s on gpu06 (dataset listing ~4 min on Weka + iter-0 compile 226s), so
#     delay=360 captured warmup only (trace was 305K, zero kernel/NVTX rows). Default now 720.
#   - Host-RAM OOM at 512G (partition default) ~33 min in: nsys collection residency on top
#     of the known +15 GB/hr DDP host-RAM drift. --mem=1000G now requested explicitly.
#   - nsys exits non-zero if the app dies; export/analysis must not be skipped by set -e.
#
# LESSONS FROM JOB 71682 (2026-06-13, second failed attempt):
#   - delay=720 landed the window on steady state correctly (captured iters ~450-690,
#     mfu ~11%, img/s ~1900, no OOM — the --mem=1000G fix held). BUT the job hit the 90-min
#     wall while nsys was *finalizing* the trace: .qdstrm->.nsys-rep stuck at 48% for ~73 min,
#     killed by TIME LIMIT -> 0-byte .nsys-rep, nothing to analyze.
#   - ROOT CAUSE: --python-sampling=true + --sample=cpu --cpuctxsw=process-tree across 8 ranks
#     produce huge backtrace tables that nsys post-processes pathologically slowly. They are
#     also UNNEEDED here: data_time ~0 in every log line => DataLoader is not the bottleneck,
#     so the residual is GPU-side (kernel gaps, NCCL exposure, NVTX wall shares), all available
#     from the CUDA+NVTX+NCCL streams alone.
#   - FIX: drop python/cpu sampling (trace = cuda,nvtx,nccl,cudnn,cublas only), halve the
#     capture to 60s, and raise --time to 03:00:00 for finalization headroom. Finalizing this
#     lighter trace is minutes, not hours. (If host-side attribution is ever needed, add
#     --sample=cpu back in a short targeted rerun — the analyzer tolerates missing tables.)
#
# Env knobs:
#   BATCH_SIZE      per-GPU batch size            (default 128 = champion)
#   ITERS           training iters                (default 900 — ends soon after capture,
#                                                  minimizing post-capture OOM exposure)
#   NSYS_DELAY      seconds before capture        (default 720 — measured warmup ~480s + margin)
#   NSYS_DURATION   capture seconds               (default 60 — ~120 steady iters at bs=128;
#                                                  halved from 120 to speed nsys finalization)
#   NUM_WORKERS     DataLoader workers            (default 20 = champion)
#   PREFETCH        DataLoader prefetch_factor    (default 8 = champion)
#   METRICS_SYNC_PERIOD  gate metric all-reduce/NaN-check + loss logging to every Nth iter
#                                                  (default 1 = every iter / unchanged; Phase 7.3)
#
# Output:
#   /mnt/weka/adovlatyan/nsys_profiles/phase7/<jobid>/<run-tag>.nsys-rep
#   /mnt/weka/adovlatyan/nsys_profiles/phase7/<jobid>/<run-tag>.sqlite
#   /mnt/weka/adovlatyan/nsys_profiles/phase7/<jobid>/<run-tag>.phase7.md   (+ .json)
#
#SBATCH --job-name=dinov3-p7-nsys
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH --gres=gpu:h100:8
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/p7-nsys-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/p7-nsys-%j.err

# Canonical torch-2.10 env (matches Anna's production stack) — PATH prepend, not conda activate.
ENV=/mnt/weka/adovlatyan/.conda/envs/dinov3_env_210clone
export PATH="$ENV/bin:$PATH"
export CONDA_PREFIX="$ENV"
export PYTHONNOUSERSITE=1

NSYS_BIN="/mnt/weka/apps/nsight-systems/2026.2.1/install/target-linux-x64/nsys"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

unset PYTORCH_CUDA_ALLOC_CONF
unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

# Phase 7.8c: per-rank data_time + NUMA/affinity diagnostics. Off by default (no-op in train.py);
# set DINOV3_PERRANK_DIAG=1 at submit to emit [RANKDATA]/[RANKTOPO] lines for straggler attribution.
export DINOV3_PERRANK_DIAG="${DINOV3_PERRANK_DIAG:-0}"

# Phase 7.9: per-rank NUMA-local CPU slices (whole physical cores, workers inherit). Off by
# default; set DINOV3_RANK_CPU_SLICE=1 at submit. Emits [RANKSLICE] lines on stderr.
# DINOV3_SLICE_EXCLUDE_CPUS: cpulist kept out of all slices (IRQ/softirq housekeeping pool);
# gpu08 storage-NIC softirq zone = 24-29,32 (job 88301 + IRQ probe 88309).
export DINOV3_RANK_CPU_SLICE="${DINOV3_RANK_CPU_SLICE:-0}"
export DINOV3_SLICE_EXCLUDE_CPUS="${DINOV3_SLICE_EXCLUDE_CPUS:-}"

BATCH_SIZE="${BATCH_SIZE:-128}"
ITERS="${ITERS:-900}"
NSYS_DELAY="${NSYS_DELAY:-720}"
NSYS_DURATION="${NSYS_DURATION:-60}"
NUM_WORKERS="${NUM_WORKERS:-20}"
PREFETCH="${PREFETCH:-8}"
METRICS_SYNC_PERIOD="${METRICS_SYNC_PERIOD:-1}"

# Phase 7.9 guard: slice arms are only meaningful on the validated 192-CPU provisioning.
# On a smaller cpuset the slicer would partition e.g. 64 CPUs into 8-CPU slices and
# manufacture exactly the starvation we're trying to remove (Codex review 2026-07-03).
# NOTE: do NOT use `nproc` here — it honors OMP_NUM_THREADS (exported above) and returned 8
# on a correct 192-CPU allocation, killing jobs 84635/84636 at t+1s. Count the real cpuset.
NCPU=$(python -c "import os; print(len(os.sched_getaffinity(0)))")
if [ "${DINOV3_RANK_CPU_SLICE}" = "1" ] && [ "${NCPU}" -lt 192 ]; then
  echo "ERROR: DINOV3_RANK_CPU_SLICE=1 requires >=192 CPUs (got ${NCPU}). Submit with --cpus-per-task=192." >&2
  exit 1
fi

RUN_TAG="ddp-cg-bs${BATCH_SIZE}-w${NUM_WORKERS}pf${PREFETCH}"
[ "${METRICS_SYNC_PERIOD}" != "1" ] && RUN_TAG="${RUN_TAG}-msp${METRICS_SYNC_PERIOD}"
[ "${DINOV3_RANK_CPU_SLICE}" = "1" ] && RUN_TAG="${RUN_TAG}-slice"
[ -n "${DINOV3_SLICE_EXCLUDE_CPUS}" ] && RUN_TAG="${RUN_TAG}-xirq"
PROFILE_DIR="/mnt/weka/adovlatyan/nsys_profiles/phase7/${SLURM_JOB_ID}"
OUTPUT_DIR="/mnt/weka/adovlatyan/output_p7nsys_${SLURM_JOB_ID}"
NSYS_OUT="${PROFILE_DIR}/${RUN_TAG}-${SLURM_JOB_ID}"

mkdir -p "${PROFILE_DIR}" /mnt/weka/adovlatyan/logs

echo "=== Phase 7.1 nsys capture (DDP+cudagraphs champion) ==="
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Node:             ${SLURM_NODELIST}"
echo "Date:             $(date)"
echo "Batch / GPU:      ${BATCH_SIZE}   workers=${NUM_WORKERS} prefetch=${PREFETCH}"
echo "Per-rank diag:    DINOV3_PERRANK_DIAG=${DINOV3_PERRANK_DIAG}"
echo "CPU slices:       DINOV3_RANK_CPU_SLICE=${DINOV3_RANK_CPU_SLICE}"
# Phase 7.9: GPU<->NUMA mapping ground truth for the slice assignment (best-effort).
nvidia-smi topo -m 2>/dev/null | head -14 || true
echo "Iters:            ${ITERS}"
echo "nsys delay/dur:   ${NSYS_DELAY}s / ${NSYS_DURATION}s"
echo "nsys output:      ${NSYS_OUT}.nsys-rep"
echo "torch:            $(python -c 'import torch; print(torch.__version__)')"
echo "nsys version:     $(${NSYS_BIN} --version | head -1)"
echo

# GPU-side residual attribution only: CUDA + NVTX + NCCL (+cublas/cudnn for kernel taxonomy).
# CPU/Python sampling is deliberately OFF — it bloats the trace and makes nsys finalization
# pathologically slow (job 71682 stuck at 48% for 73 min), and data_time~0 means the host
# side is not where the residual lives. Re-add --sample=cpu in a targeted rerun if needed.
# --cuda-graph-trace=node is REQUIRED for this config: triton.cudagraphs=True means the
# backbone runs as CUDA graphs, and nsys 2026.2.1's default 'graph' granularity emits a
# single device-graph event whose replay-relative start timestamp exceeds its end timestamp,
# crashing .nsys-rep finalization (job 74635: "CudaDeviceGraph event's start > end timestamp").
# 'node' collects per-kernel activity inside the graph — fixes the crash AND is what the
# residual attribution needs (graph granularity would collapse the backbone into one opaque
# event, defeating the kernel taxonomy).
"${NSYS_BIN}" profile \
  --output="${NSYS_OUT}" \
  --trace=cuda,nvtx,nccl,cudnn,cublas \
  --cuda-graph-trace=node \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=none \
  --delay="${NSYS_DELAY}" \
  --duration="${NSYS_DURATION}" \
  --force-overwrite=true \
  --trace-fork-before-exec=true \
  -- \
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
  train.batch_size_per_gpu="${BATCH_SIZE}" \
  train.num_workers="${NUM_WORKERS}" \
  train.metrics_sync_period="${METRICS_SYNC_PERIOD}" \
  train.OFFICIAL_EPOCH_LENGTH="${ITERS}" \
  optim.epochs=1 \
  train.persistent_workers=true \
  train.prefetch_factor="${PREFETCH}" \
  train.cache_dataset=true \
  train.compile=true \
  train.cudagraphs=true \
  train.checkpointing=false \
  train.distributed_strategy=ddp \
  train.sharded_eval_checkpoint=true \
  checkpointing.period=99999 \
  evaluation.eval_period_iterations=99999 \
  wandb.enabled=false || NSYS_RC=$?

# nsys propagates the app's exit code (e.g. if a rank dies after the capture window) —
# the trace may still be perfectly valid, so always attempt export+analysis (job 71041 lesson).
echo
echo "=== capture complete: $(date) (nsys rc=${NSYS_RC:-0}) — exporting sqlite + running analyzer ==="
ls -lh "${NSYS_OUT}".* 2>/dev/null || { echo "ERROR: no nsys output at ${NSYS_OUT}" >&2; exit 1; }

"${NSYS_BIN}" export --type sqlite --force-overwrite=true \
  --output="${NSYS_OUT}.sqlite" "${NSYS_OUT}.nsys-rep"

python scripts/profiling/phase7_nsys_report.py "${NSYS_OUT}.sqlite" \
  --label "${RUN_TAG}-${SLURM_JOB_ID}" || echo "WARN: analyzer failed — trace is still usable" >&2

echo
echo "=== Phase 7.1 artifacts ==="
ls -lh "${PROFILE_DIR}"/ 2>/dev/null
echo
echo "Report:  ${NSYS_OUT}.phase7.md"
echo "Metrics: ${NSYS_OUT}.phase7.json   (feed two of these to --diff)"
echo "GUI:     ~/scripts/rsync-nsys ${SLURM_JOB_ID}"
