# Shared Oscar environment for the colloid detection + merge jobs.
#
# EDIT THIS ONCE. It is sourced by BOTH submit_detect.slurm and merge.slurm (on
# the compute node), so you configure your Python environment in a single place.
#
# Activate the Python env that has: numpy pandas opencv-python(-headless) scipy
# pyarrow  (and, for GPU detection, cupy matching the cluster CUDA), then set
# COLLOID_ENV_READY=1 at the bottom.

module load cuda 2>/dev/null || true

# --- activate your Python env: uncomment/edit ONE of these -------------------
# conda:
#   module load miniconda3 && source activate colloid
# venv:
#   source "$HOME/envs/colloid/bin/activate"
# module python + user site-packages already on PYTHONPATH: nothing to do here
# ----------------------------------------------------------------------------

# Flip to 1 AFTER the env above is active, so the jobs fail fast with a clear
# message instead of a cryptic ImportError partway through a compute-node run.
COLLOID_ENV_READY=0
