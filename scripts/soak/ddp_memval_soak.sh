#!/bin/bash
# ddp_memval_soak.sh — Memory-validation soak (short, high-event-density).
# =======================================================================
# PURPOSE
#   Answer one question for a given batch size, BEFORE committing to a multi-day
#   run: "Will DDP + cudagraphs + AC=off at this batch survive real training
#   without OOM — through checkpoint saves, eval runs, and host-RAM drift?"
#
#   This is a *sibling* of run_ddp.sh, not a replacement. run_ddp.sh is the
#   production recipe (bs=128, full schedule). This script reuses the identical
#   training path but shrinks the schedule to 2000 iters while KEEPING the same
#   number of memory-stress events as the validated 4000-iter soak (job 60959):
#
#       checkpoint saves: 8   (period=250  → 8 saves in 2000 iters)
#       eval runs:        4   (period=500  → 4 evals in 2000 iters)
#
#   The iteration count was never the thing being measured — the EVENTS were.
#   Fragmentation accumulates across checkpoint/eval transitions, not uniformly
#   per-iter, so halving iters + halving periods gives equivalent coverage in
#   ~half the wall time (~45–55 min vs ~2 h).
#
# WHY THIS EXISTS — the bs=128 result that motivates it:
#   The 4000-iter bs=128 soak (job 60959) peaked at only 36 GB VRAM on an 80 GB
#   H100 — ~44 GB sat idle. The reason we could not spend that headroom on a
#   bigger batch was NOT VRAM: bs=192 OOM'd twice (jobs 53739, 56131), and the
#   corrected diagnosis (phase6_perf_plan.md §6.A.6) is that BOTH were *host-RAM*
#   cgroup OOMs, not CUDA OOMs. The binding resource is the DataLoader's host
#   working set:  8 ranks × num_workers × prefetch_factor × batch_bytes.
#   So to climb to bs=160/192 you must shrink the loader (w×pf) to free host RAM,
#   then confirm the freed VRAM is actually usable. This script measures both.
#
# WHAT IT MEASURES (and why the instrumentation is non-obvious)
#   The OOM-predicting number is cgroup ANONYMOUS memory, NOT cgroup.current and
#   NOT node MemAvailable. cgroup.memory.current is page-cache-dominated (the
#   Weka HDF5/tile reads) and pegs at whatever --mem you set — it told us nothing
#   in 6.B.2 and that per-worker model was retracted. Linux only OOM-kills when
#   *anon* (non-reclaimable) memory can't be satisfied. So the sidecar logs the
#   `anon` field of the job's cgroup memory.stat. anon-near-cap ⇒ will OOM;
#   page-cache fill ⇒ benign. The analysis block fits the anon slope and projects
#   it to a full 10-epoch run, because +15 GB/hr drift over a 2000-iter window is
#   invisible but fatal over ~234k iters.
#
# LOADER SIZING (env-tunable — this is the lever for bs>128)
#   Heuristic: hold the in-flight working set roughly constant vs the validated
#   bs=128 / 20w / 8pf point by keeping (num_workers × prefetch_factor × batch)
#   near 20×8×128 = 20480. This is a STARTING POINT to be confirmed by the anon
#   measurement, not a guarantee (you cannot predict absolute anon from the
#   multiplication — see the §6.B retraction). Keep prefetch_factor ≥ 6: the
#   6.B.2.b/c sweep showed prefetch DEPTH, not worker count, is what hides the
#   loader. Recommended starting points:
#       bs=160 → 16w × 8pf = 128 batches/rank   (this script's default)
#       bs=192 → 12w × 8pf =  96 batches/rank
#   Watch the data_time column: too few workers at high batch can starve the GPUs.
#
# USAGE
#   DINOV3_ENV=/mnt/weka/<you>/.conda/envs/<env> sbatch scripts/soak/ddp_memval_soak.sh
#   # probe a bigger batch:
#   DINOV3_ENV=... BATCH_SIZE=192 NUM_WORKERS=12 PREFETCH_FACTOR=8 \
#       sbatch scripts/soak/ddp_memval_soak.sh
#
# HOST RAM + CPUs: we own the whole node (gpu:h100:8), and a DGX H100 here has
#   ~2 TB RAM + 224 CPUs. The default (no --mem) resolves to cpus×DefMemPerCPU =
#   512 GB, which is the WRONG ceiling for a bs>128 probe — jobs 64978/64979
#   OOM'd against it (host RAM, not VRAM). So we claim it explicitly:
#     --mem=1500G       → removes the host-RAM ceiling (well under the ~2 TB cap)
#     --cpus-per-task=192 → ~24 decode cores/rank (8 ranks), so the DataLoader can
#                           actually feed bs≥160 instead of starving (the 64-CPU
#                           default gave only 8 cores/rank → loader-bound at bs=192).
#   These two together let us scale the loader UP (more workers + deep prefetch)
#   rather than down. #SBATCH lines can't read env vars, so edit them here if you
#   want different values.
#
#SBATCH --job-name=dinov3-memval-soak
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=1500G
#SBATCH --gres=gpu:h100:8
#SBATCH --time=02:00:00
#SBATCH --output=slurm-memval-%j.out
#SBATCH --error=slurm-memval-%j.err

# >>> SET YOUR CONDA ENV (torch >= 2.6) <<<  (PATH-prepend; conda activate fails on GPU nodes)
DINOV3_ENV="${DINOV3_ENV:?Set DINOV3_ENV to your conda env prefix (torch>=2.6), e.g. /mnt/weka/$(whoami)/.conda/envs/dinov3_env_210clone}"
export PATH="${DINOV3_ENV}/bin:$PATH"
export CONDA_PREFIX="${DINOV3_ENV}"
export PYTHONNOUSERSITE=1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH=.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Per-phase + periodic VRAM markers from inside training (rank-0 [MEMPROFILE]).
export DINOV3_MEMORY_PROFILE=1
export DINOV3_MEMORY_PROFILE_PERIOD="${DINOV3_MEMORY_PROFILE_PERIOD:-50}"

# Match the validated allocator/NCCL conditions of the 60959 soak.
unset PYTORCH_CUDA_ALLOC_CONF
unset NCCL_NVLS_ENABLE NCCL_PROTO NCCL_NTHREADS NCCL_BUFFSIZE NCCL_ALGO \
      NCCL_NSOCKS_PERTHREAD TORCH_NCCL_AVOID_RECORD_STREAMS 2>/dev/null || true

# ---- Tunables (env-overridable) ------------------------------------------------
BATCH_SIZE="${BATCH_SIZE:-160}"
NUM_WORKERS="${NUM_WORKERS:-24}"      # ~1 worker per decode core (192 CPUs / 8 ranks)
PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"
ITERS="${ITERS:-2000}"
CKPT_PERIOD="${CKPT_PERIOD:-250}"     # 8 saves in 2000 iters (matches 60959 event count)
EVAL_PERIOD="${EVAL_PERIOD:-500}"     # 4 evals in 2000 iters (matches 60959 event count)
MAX_TO_KEEP="${MAX_TO_KEEP:-3}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"   # default scratch — memory behavior is identical
# Full-run size used only for the anon-drift projection in the analysis block.
FULL_RUN_ITERS="${FULL_RUN_ITERS:-234120}"     # 10 epochs × 23412

OUTPUT_ROOT="${DINOV3_OUTPUT_ROOT:-/mnt/weka/$(whoami)}"
RUN_TAG="memval_ddp_bs${BATCH_SIZE}_${NUM_WORKERS}w${PREFETCH_FACTOR}pf"
OUTPUT_DIR="${OUTPUT_ROOT}/output_${RUN_TAG}_${SLURM_JOB_ID}"
MEMLOG="${OUTPUT_ROOT}/${RUN_TAG}_${SLURM_JOB_ID}_memlog.jsonl"
mkdir -p "${OUTPUT_ROOT}"

BATCHES_PER_RANK=$(( NUM_WORKERS * PREFETCH_FACTOR ))

# ---- Host-RAM sidecar ----------------------------------------------------------
# Logs cgroup ANON (the OOM-predicting number) from the job's own cgroup, plus
# cgroup file-cache and current for context, plus per-GPU VRAM. cgroup v2.
resolve_cgroup_dir() {
    local rel
    rel="$(awk -F: '/^0::/{print $3}' /proc/self/cgroup 2>/dev/null)"
    [ -n "${rel}" ] && [ -r "/sys/fs/cgroup${rel}/memory.stat" ] && { echo "/sys/fs/cgroup${rel}"; return; }
    # walk up until memory.stat is readable (some layouts put it on a parent)
    local d="/sys/fs/cgroup${rel}"
    while [ "${d}" != "/sys/fs/cgroup" ] && [ "${d}" != "/" ]; do
        d="$(dirname "${d}")"
        [ -r "${d}/memory.stat" ] && { echo "${d}"; return; }
    done
    echo ""   # not resolvable
}

host_ram_monitor() {
    local logfile="$1"; local cgdir="$2"
    while true; do
        local ts; ts=$(date +%s)
        local anon=-1 file=-1 cur=-1
        if [ -n "${cgdir}" ] && [ -r "${cgdir}/memory.stat" ]; then
            anon=$(awk '/^anon /{printf "%.0f",$2/1048576}' "${cgdir}/memory.stat")
            file=$(awk '/^file /{printf "%.0f",$2/1048576}' "${cgdir}/memory.stat")
            [ -r "${cgdir}/memory.current" ] && cur=$(awk '{printf "%.0f",$1/1048576}' "${cgdir}/memory.current")
        fi
        local host_used
        host_used=$(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{printf "%.0f",(t-a)/1024}' /proc/meminfo)
        local gpu_used
        gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
                   | tr '\n' ',' | sed 's/,$//')
        printf '{"ts":%d,"cg_anon_mb":%s,"cg_file_mb":%s,"cg_current_mb":%s,"host_used_mb":%s,"gpu_used_mb":[%s]}\n' \
               "$ts" "${anon:-entry}" "${file:-entry}" "${cur:-entry}" "${host_used:-entry}" "${gpu_used:-0}" >> "$logfile"
        sleep 10
    done
}

CGDIR="$(resolve_cgroup_dir)"

echo "=== ${RUN_TAG} (memory-validation soak) ==="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "Config:     DDP, bs=${BATCH_SIZE}, compile=true, cudagraphs=true, AC=off"
echo "Loader:     num_workers=${NUM_WORKERS}, prefetch_factor=${PREFETCH_FACTOR}  (${BATCHES_PER_RANK} batches/rank; validated point = 160/rank @ bs128)"
echo "Schedule:   ${ITERS} iters; ckpt period=${CKPT_PERIOD} (8 saves), eval period=${EVAL_PERIOD} (4 evals)"
echo "cgroup dir: ${CGDIR:-<UNRESOLVED — anon logging disabled, will fall back>}"
echo "Resources:  --cpus-per-task=192 (~24 decode cores/rank), --mem=1500G (host-RAM ceiling removed; node has ~2 TB)"
echo "Output:     ${OUTPUT_DIR}"
echo "Memlog:     ${MEMLOG}"
echo "Date:       $(date)"
echo ""

host_ram_monitor "${MEMLOG}" "${CGDIR}" &
MONITOR_PID=$!
trap "kill ${MONITOR_PID} 2>/dev/null || true" EXIT

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
  train.sharded_eval_checkpoint=true \
  checkpointing.period="${CKPT_PERIOD}" \
  checkpointing.max_to_keep="${MAX_TO_KEEP}" \
  evaluation.eval_period_iterations="${EVAL_PERIOD}" \
  wandb.enabled="${WANDB_ENABLED}" \
  wandb.project=dinov3-satellite \
  wandb.run_name="${RUN_TAG}_${SLURM_JOB_ID}" \
  wandb.group=satellite_memval

echo ""
echo "=== ${RUN_TAG} complete: $(date) ==="
LOG="slurm-memval-${SLURM_JOB_ID}.out"

echo ""
echo "--- VRAM phase markers (rank 0: pre/post checkpoint & eval) ---"
grep '\[MEMPROFILE\]' "${LOG}" 2>/dev/null | grep 'rank=0' || echo "(none — check ${LOG})"

echo ""
echo "--- data_time vs step_time (loader-stall check at this loader size) ---"
grep 'data_time' "${LOG}" 2>/dev/null | awk 'NR>20' | tail -5 || true

echo ""
echo "--- Memory verdict (cgroup anon = OOM predictor; VRAM = headroom) ---"
python3 - "${MEMLOG}" "${ITERS}" "${FULL_RUN_ITERS}" <<'PY' || echo "(memlog parse failed — inspect ${MEMLOG})"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
iters, full = int(sys.argv[2]), int(sys.argv[3])
if not rows:
    print("  empty memlog"); sys.exit(0)
t0 = rows[0]["ts"]
def col(k):
    return [(r["ts"]-t0, r[k]) for r in rows if isinstance(r.get(k), (int, float)) and r[k] >= 0]
anon = col("cg_anon_mb")
gpus = [max((r["gpu_used_mb"][i] for r in rows if r.get("gpu_used_mb")), default=0)
        for i in range(len(rows[0].get("gpu_used_mb", [])))]
CAP_GB = 1500   # matches #SBATCH --mem=1500G
if anon:
    peak_anon = max(v for _, v in anon)
    # steady-state slope: skip first 25% (compile warmup + cache fill)
    ss = anon[len(anon)//4:]
    if len(ss) >= 2:
        (x0, y0), (x1, y1) = ss[0], ss[-1]
        dt_hr = max((x1 - x0)/3600.0, 1e-9)
        slope = (y1 - y0)/1024.0 / dt_hr            # GB/hr
        wall_hr = rows[-1]["ts"] - rows[0]["ts"]
        full_hr = (wall_hr/3600.0) * (full/iters) if iters else 0
        proj_anon_gb = peak_anon/1024.0 + slope*full_hr
        print(f"  peak cgroup anon : {peak_anon/1024:6.1f} GB   (cap {CAP_GB} GB)")
        print(f"  anon drift       : {slope:+6.1f} GB/hr  (steady-state fit)")
        print(f"  projected anon   : {proj_anon_gb:6.1f} GB  at full run ({full:,} iters, ~{full_hr:.1f} h)")
        margin = CAP_GB - proj_anon_gb
        print(f"  VERDICT          : {'PASS' if margin > 32 else 'RISK' if margin > 0 else 'FAIL'}"
              f"  (projected margin {margin:+.0f} GB vs {CAP_GB} GB cap; PASS wants >32 GB)")
    else:
        print(f"  peak cgroup anon : {peak_anon/1024:.1f} GB  (too few samples for a drift fit)")
else:
    print("  cgroup anon unavailable (cgroup dir unresolved) — re-check resolve_cgroup_dir on this node")
if gpus:
    print(f"  peak VRAM (max GPU): {max(gpus)/1024:6.1f} GB   (of 80 GB — first real number at this batch)")
    print(f"  per-GPU peaks GB  : {[round(g/1024,1) for g in gpus]}")
PY
echo ""
echo "Full memlog: ${MEMLOG}"
