#!/bin/bash
#SBATCH --job-name=colloid_merge
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err
# ----------------------------------------------------------------------------
# Merge the per-chunk detection Parquets into detections.parquet + job_meta.json.
#
# This runs on a COMPUTE NODE, submitted by run_oscar_job.py with a dependency
# on the detection array (--dependency=afterok:<arrayjobid>), so NO non-trivial
# work ever touches the Oscar login node. If any array task fails, the afterok
# dependency is never satisfied and Slurm cancels this job (DependencyNever-
# Satisfied) — the launcher detects that and reports the array failure.
#
# No GPU needed. Manual submit (rarely needed):
#   sbatch --dependency=afterok:<arrayjobid> merge.slurm
# ----------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Activate the Python environment (edit oscar/env.sh — one place for both jobs).
source "${SLURM_SUBMIT_DIR}/env.sh"
if [ "${COLLOID_ENV_READY:-0}" != "1" ]; then
  echo "ERROR: edit oscar/env.sh to activate your Python env and set" >&2
  echo "       COLLOID_ENV_READY=1. See oscar/README.md step 3."      >&2
  exit 1
fi

python merge_chunks.py --chunks-dir chunks --job job.json \
       --out detections.parquet --meta-out job_meta.json
