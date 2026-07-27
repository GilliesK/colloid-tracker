#!/bin/bash
#SBATCH --job-name=colloid_detect
#SBATCH --array=0-0                     # OVERRIDDEN by the launcher: --array=0-(nchunks-1)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/detect_%A_%a.out
#SBATCH --error=logs/detect_%A_%a.err
# ----------------------------------------------------------------------------
# One array task = one contiguous frame range ("chunk") of the video.
# Reads chunk_size / frame_start / frame_end from job.json and computes its own
# range from SLURM_ARRAY_TASK_ID, then runs the shared detection core.
#
# CPU by default (the speedup is PARALLELISM across many tasks, not per-task
# GPU). Partition comes from the launcher's -p; a GPU is requested only when the
# launcher is run with --gpu (adds --gres=gpu:1 and sets job.json "gpu": true,
# which also passes --gpu to the worker). GPU needs cupy installed in the env.
#
# Submit via run_oscar_job.py (recommended) which sets --array and the job dir.
# Manual:  cd <jobdir> && sbatch --array=0-$((NCHUNKS-1)) -p batch submit_detect.slurm
# ----------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs chunks obs

# Activate the Python environment (edit oscar/env.sh — one place for both jobs).
source "${SLURM_SUBMIT_DIR}/env.sh"
if [ "${COLLOID_ENV_READY:-0}" != "1" ]; then
  echo "ERROR: edit oscar/env.sh to activate your Python env and set" >&2
  echo "       COLLOID_ENV_READY=1. See oscar/README.md step 3."      >&2
  exit 1
fi

read FRAME_START FRAME_END CHUNK_SIZE VIDEO DFS GPU < <(python - <<'PY'
import json
j=json.load(open("job.json"))
print(j["frame_start"], j["frame_end"], j["chunk_size"], j["video"],
      "1" if j.get("decode_from_start") else "0",
      "1" if j.get("gpu") else "0")
PY
)
DFS_FLAG=""; if [ "$DFS" = "1" ]; then DFS_FLAG="--decode-from-start"; fi
GPU_FLAG="";  if [ "$GPU" = "1" ]; then GPU_FLAG="--gpu"; fi

TASK=${SLURM_ARRAY_TASK_ID:-0}
S=$(( FRAME_START + TASK * CHUNK_SIZE ))
E=$(( S + CHUNK_SIZE - 1 ))
if [ "$E" -gt "$FRAME_END" ]; then E=$FRAME_END; fi
if [ "$S" -gt "$FRAME_END" ]; then
  echo "task $TASK: range $S..$E is past end ($FRAME_END) — nothing to do"; exit 0
fi

OUT=$(printf "chunks/chunk_%05d.parquet" "$TASK")
OBS=$(printf "obs/obs_%05d.parquet" "$TASK")
echo "task $TASK -> frames $S..$E -> $OUT"
python detect_worker.py \
     --video "$VIDEO" \
     --params job.json \
     --frames "${S}:${E}" \
     --out "$OUT" \
     --obs-out "$OBS" \
     $GPU_FLAG $DFS_FLAG
