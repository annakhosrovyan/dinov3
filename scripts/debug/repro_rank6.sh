#!/bin/bash
# Reproduce the rank-6 / iter~1310 crash on CPU (no GPU, no DDP). Fast turnaround.
# Replays rank-6's exact ShardedInfiniteSampler stream and calls dataset[idx] directly,
# so a bad sample throws here with a full synchronous traceback + resolved source.
#SBATCH --job-name=dinov3-repro-rank6
#SBATCH --nodes=1
#SBATCH --partition=research
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/weka/adovlatyan/logs/repro-rank6-%j.out
#SBATCH --error=/mnt/weka/adovlatyan/logs/repro-rank6-%j.err

export PATH="/home/adovlatyan/.conda/envs/test-conda-slurm/bin:$PATH"
export CONDA_PREFIX="/home/adovlatyan/.conda/envs/test-conda-slurm"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH=.
export OMP_NUM_THREADS=8

echo "=== repro-rank6 job ${SLURM_JOB_ID} on ${SLURM_NODELIST} $(date) ==="
# Wide window around the observed crash (rank 0 logged 1300, rank 6 died before 1310).
python scripts/debug/repro_rank6.py --rank 6 --world-size 8 --start-batch 1280 --end-batch 1340
echo "=== done $(date) ==="
