#!/bin/bash
# =============================================================================
# T2 screening-loop candidate runner — one candidate config, one 700-iter job
# =============================================================================
# Generalizes scripts/screening/ddp_maxconn_sweep.sh from "sweep one hardcoded
# knob" to "run any candidate expressed as env vars + OmegaConf overrides".
# Everything not overridden is PINNED to the torch-2.10 controlled baseline
# recipe (job 67639 / maxconn=1 winner 77672: DDP, cudagraphs, bs=128, 24w/8pf,
# MAX_CONN=1). The candidate is the diff, nothing else moves.
#
# Usage (the loop driver or a human submits):
#   CANDIDATE_TAG=nw16 EXTRA_OPTS="train.num_workers=16" \
#     sbatch scripts/loop/run_candidate.sh
#   CANDIDATE_TAG=mc4 EXTRA_ENV="CUDA_DEVICE_MAX_CONNECTIONS=4" \
#     sbatch scripts/loop/run_candidate.sh
#   # Slurm-level knobs (cpus, mem) go on the sbatch command line:
#   CANDIDATE_TAG=cpu96 sbatch --cpus-per-task=96 scripts/loop/run_candidate.sh
#
# Contract with the loop:
#   * CANDIDATE_TAG   — short id; becomes part of the output dir name
#   * HYPOTHESIS      — one line, recorded in the job log (lineage row cites it)
#   * EXTRA_ENV       — space-separated K=V pairs exported before torchrun
#   * EXTRA_OPTS      — extra OmegaConf overrides appended LAST (they win)
#   * BATCH_SIZE      — operating point, default 128. NOT a search knob
#                       (champions get re-confirmed at 192 by the driver).
#   * The job's end-of-log self-report is NON-AUTHORITATIVE. The verifier is
#     the installed ~/scripts/loop-verifier/score.py run by the driver against
#     the raw output dir. See scripts/loop/README.md.
# =============================================================================
#SBATCH --job-name=dinov3-loop-cand
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=1500G
#SBATCH --gres=gpu:h100:8
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/%x_%j.out

set -uo pipefail

# ===================== PROJECT-SPECIFIC (rewrite per project) =================
# Porting to another PyTorch project? Rewrite this block, the recipe pins below,
# and the torchrun invocation further down (also marked). The #SBATCH resources
# at the top are project-specific too. Everything else (candidate channels,
# output-dir naming, self-report, exit-code propagation) is generic scaffolding
# you keep as-is. See ADAPTERS.md "The shell side".
ENV=/mnt/weka/adovlatyan/.conda/envs/dinov3_env_210clone
export PATH="$ENV/bin:$PATH"
export CONDA_PREFIX="$ENV"
export PYTHONNOUSERSITE=1
export PYTHONPATH=.

# ---- Pinned baseline recipe (67639 / 77672) --------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=1     # §7.2 winner; override via EXTRA_ENV
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export DINOV3_PERRANK_DIAG=1             # [RANKDATA] lines → straggler gate (stderr only, no training effect)
unset PYTORCH_CUDA_ALLOC_CONF
unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-24}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"
ITERS="${ITERS:-700}"                  # steady-state by ~iter 200
SKIP_ITERS="${SKIP_ITERS:-200}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"

# ===================== GENERIC SCAFFOLDING (keep as-is) ======================
# ---- The candidate ----------------------------------------------------------
CANDIDATE_TAG="${CANDIDATE_TAG:-cand}"
HYPOTHESIS="${HYPOTHESIS:-<none given>}"
EXTRA_ENV="${EXTRA_ENV:-}"
EXTRA_OPTS="${EXTRA_OPTS:-}"

# EXTRA_ENV denylist (Codex GPT-5.6 review, finding 3): a candidate must not be
# able to redirect the runner, hijack imports, move the output dir, or blind a
# gate via the env channel. This is defense-in-depth only — the actuator is
# agent-editable, so the REAL barrier is the driver submitting an immutable,
# hashed actuator + training revision and recording the trusted command itself
# (see DESIGN.md "Residual hermeticity gaps").
if [ -n "${EXTRA_ENV}" ]; then
  for kv in ${EXTRA_ENV}; do
    case "${kv}" in
      *=*) : ;;
      *) echo "FATAL: EXTRA_ENV token '${kv}' is not K=V (values must not contain spaces)"; exit 64 ;;
    esac
    key="${kv%%=*}"; val="${kv#*=}"
    case "${key}" in
      PATH|PYTHONPATH|PYTHONHOME|PYTHONSTARTUP|PYTHONNOUSERSITE|LD_*|SLURM_*|DINOV3_OUTPUT_ROOT|HOME|CONDA_PREFIX)
        echo "FATAL: EXTRA_ENV may not set '${key}' — it can redirect the runner, imports, or output path (measurement integrity)"; exit 65 ;;
    esac
    if [ "${key}" = "DINOV3_PERRANK_DIAG" ] && [ "${val}" != "1" ]; then
      echo "FATAL: DINOV3_PERRANK_DIAG must stay 1 — disabling it blinds the per-rank straggler gate (score_core finding 1)"; exit 65
    fi
    export "${kv?}"
  done
fi

OUTPUT_ROOT="${DINOV3_OUTPUT_ROOT:-/mnt/weka/$(whoami)}"
RUN_TAG="loop_${CANDIDATE_TAG}_bs${BATCH_SIZE}"
OUTPUT_DIR="${OUTPUT_ROOT}/output_${RUN_TAG}_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_ROOT}" "/mnt/weka/adovlatyan/logs"

echo "=== T2 loop candidate: ${CANDIDATE_TAG} ==="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURM_NODELIST}"
echo "Hypothesis:    ${HYPOTHESIS}"
echo "EXTRA_ENV:     ${EXTRA_ENV:-<none>}"
echo "EXTRA_OPTS:    ${EXTRA_OPTS:-<none>}"
echo "Operating pt:  DDP+cudagraphs, bs=${BATCH_SIZE}, ${NUM_WORKERS}w/${PREFETCH_FACTOR}pf, MAX_CONN=${CUDA_DEVICE_MAX_CONNECTIONS}"
echo "Schedule:      ${ITERS} iters, steady-state skips first ${SKIP_ITERS}"
echo "Output:        ${OUTPUT_DIR}"
echo "Date:          $(date)"
echo ""

# ===================== PROJECT-SPECIFIC (rewrite per project) =================
# The training invocation itself. Everything below through the closing EXTRA_OPTS
# is the dinov3 recipe. Keep the two contract rules: (1) set the run length and
# push ckpt/eval periods beyond it so only throughput is measured; (2) append
# ${EXTRA_OPTS} LAST, unquoted, so candidate overrides win.
# ckpt/eval periods beyond ITERS so neither fires — pure throughput.
# EXTRA_OPTS is intentionally UNQUOTED and LAST: word-split into overrides that win.
# shellcheck disable=SC2086
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
  wandb.enabled=false \
  ${EXTRA_OPTS}

RC=$?
# ===================== GENERIC SCAFFOLDING (keep as-is) ======================
echo ""
echo "=== candidate ${CANDIDATE_TAG} complete (rc=${RC}): $(date) ==="

# ---- Self-report (NON-AUTHORITATIVE — the driver re-scores hermetically) ----
SCORER="${HOME}/scripts/loop-verifier/score.py"
[ -f "${SCORER}" ] || SCORER="scripts/loop/score.py"
echo ""
echo "--- self-report via ${SCORER} (non-authoritative) ---"
# -I -B: isolated mode so PYTHONPATH / user-site / sitecustomize cannot hijack the
# verifier's imports; no bytecode written (finding 4).
python3 -I -B "${SCORER}" "${OUTPUT_DIR}" --skip-iters "${SKIP_ITERS}" --expect-iters "${ITERS}" \
  || echo "(self-report failed — driver must inspect ${OUTPUT_DIR})"
echo ""
echo "Authoritative scoring (driver-owned expect-iters + slurm log for the straggler gate):"
echo "  python3 -I -B ~/scripts/loop-verifier/score.py ${OUTPUT_DIR} --baseline <baseline_dir> \\"
echo "      --expect-iters ${ITERS} --world-size 8 --slurm-log /mnt/weka/adovlatyan/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"

# Propagate training's real exit code to Slurm (a failed torchrun must not
# leave a COMPLETED-looking job).
exit "${RC}"
