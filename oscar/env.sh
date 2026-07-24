# Shared Oscar environment for the colloid detection + merge jobs.
#
# Sourced by BOTH submit_detect.slurm and merge.slurm on the compute node, so
# the Python environment is configured in this single place. EDIT HERE (the PC
# copy) — the launcher ships this file to Oscar on every run, so a copy edited
# inside the job directory would be overwritten.

# GPU runtime for cupy detection (harmless / skipped if the name differs or the
# task lands on a CPU node).
module load cuda 2>/dev/null || true

# Conda (Brown Oscar). `source .../conda.sh` initialises conda so that
# `conda activate` works in a NON-interactive job shell (without it you get
# CommandNotFoundError: conda activate).
module load anaconda3/2023.09-0-aqbc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate colloidcv

# Ignore ~/.local user-site packages so a stray `pip install --user` (numpy,
# opencv, ...) can never shadow the env's versions on a compute node.
export PYTHONNOUSERSITE=1

# Mark the env ready — the jobs fail fast with a clear message if this is not 1.
COLLOID_ENV_READY=1
