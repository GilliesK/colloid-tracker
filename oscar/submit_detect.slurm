#!/bin/bash
#SBATCH --job-name=colloid_detect
#SBATCH --array=0-0                     # OVERRIDDEN by the launcher: --array=0-(nchunks-1)
#SBATCH --partition=gpu                 # Brown CCV Oscar GPU partition (see README if yours differs)
#SBATCH --gres=gpu:1
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
# Submit via run_oscar_job.py (recommended) which sets --array and the job dir.
# Manual:  cd <jobdir> && sbatch --array=0-$((NCHUNKS-1)) submit_detect.slurm
# ----------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs chunks

# ==== EDIT THIS BLOCK for your Oscar environment ============================
# Activate the Python env that has: numpy pandas opencv-python scipy pyarrow
# and (optionally, for GPU) cupy matching the cluster CUDA.
module load cuda 2>/dev/null || true
# Option A - conda:
#   module load miniconda3 && source activate colloid
# Option B - venv:
#   source "$HOME/envs/colloid/bin/activate"
# Option C - module python + user site packages already on PYTHONPATH
# Set this to 1 in the block above once you have activated the right env, so the
# job fails fast with a clear message instead of a cryptic ImportError:
: "${COLLOID_ENV_READY:=0}"
# ===========================================================================

if [ "${COLLOID_ENV_READY}" != "1" ]; then
  echo "ERROR: edit the env block in submit_detect.slurm to activate your Python" >&2
  echo "       env (numpy pandas opencv scipy pyarrow [cupy]) and set"           >&2
  echo "       COLLOID_ENV_READY=1. See oscar/README.md step 3."                 >&2
  exit 1
fi

read FRAME_START FRAME_END CHUNK_SIZE VIDEO DFS < <(python - <<'PY'
import json
j=json.load(open("job.json"))
print(j["frame_start"], j["frame_end"], j["chunk_size"], j["video"],
      "1" if j.get("decode_from_start") else "0")
PY
)
DFS_FLAG=""
if [ "$DFS" = "1" ]; then DFS_FLAG="--decode-from-start"; fi

TASK=${SLURM_ARRAY_TASK_ID:-0}
S=$(( FRAME_START + TASK * CHUNK_SIZE ))
E=$(( S + CHUNK_SIZE - 1 ))
if [ "$E" -gt "$FRAME_END" ]; then E=$FRAME_END; fi
if [ "$S" -gt "$FRAME_END" ]; then
  echo "task $TASK: range $S..$E is past end ($FRAME_END) — nothing to do"; exit 0
fi

OUT=$(printf "chunks/chunk_%05d.parquet" "$TASK")
echo "task $TASK -> frames $S..$E -> $OUT"
python detect_worker.py \
     --video "$VIDEO" \
     --params job.json \
     --frames "${S}:${E}" \
     --out "$OUT" \
     --gpu $DFS_FLAG
