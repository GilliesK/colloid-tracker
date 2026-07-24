# Oscar offload — detection on the cluster, analysis on your desktop

Long videos (thousands of 2840×2840 frames) spend almost all their time in
**per-frame particle detection**. That step is embarrassingly parallel, so this
toolkit ships it to Brown's **Oscar** cluster as a Slurm **GPU array** — one task
per frame chunk — and returns only the detected coordinates. Linking and all
structural analysis (g(r), ψ₆, 5–7 pairs, LAGB, Zhang–Nelson) then run **locally**,
exactly as in a normal run, and the result loads straight into the app's
**Video Analysis** tab.

Detection on the cluster is **bit-identical** to the desktop app: both import the
same Qt-free core, `colloid_detect.py`.

```
 desktop                         Oscar (GPU array)                 desktop
┌──────────────┐  video+params  ┌───────────────────────┐  coords ┌──────────────┐
│ run_oscar_   │ ─────────────▶ │ detect_worker.py ×N    │ ──────▶ │ Load cluster │
│ job.py       │                │  (colloid_detect core) │         │ detections…  │
│ (launcher)   │ ◀───────────── │ merge_chunks.py        │         │  link+analyse│
└──────────────┘  detections    └───────────────────────┘         │  → Video tab │
                  .parquet                                          └──────────────┘
```

## What runs where

| Stage | Where | Code |
|---|---|---|
| Preprocess + Crocker–Grier detection (ring / dark / plain) | **Oscar**, GPU array | `detect_worker.py` → `colloid_detect.py` |
| Merge chunks → one coordinate table | Oscar (or local) | `merge_chunks.py` |
| Edge filter, linking, ψ₆/g(r)/LAGB, playback | **Local** | the app's existing pipeline |

Only coordinates come back (a few MB), never the video — the app already has it.

## One-time setup on Oscar

1. **SSH key** so the launcher isn't prompting for a password every step:
   `ssh-copy-id you@ssh.ccv.brown.edu`
2. **Python env** with: `numpy pandas opencv-python scipy pyarrow` and, for GPU,
   `cupy` matching Oscar's CUDA (`module load cuda`). Example:
   ```bash
   module load miniconda3
   conda create -n colloid python=3.11 numpy pandas scipy pyarrow -y
   conda activate colloid
   pip install opencv-python-headless
   pip install cupy-cuda12x        # match `module load cuda` version
   ```
3. **Point the Slurm script at that env**: edit the marked block in
   `submit_detect.slurm` (the `module load` / `conda activate` lines) **and set
   `COLLOID_ENV_READY=1`** in that block — the job fails fast with a clear message
   if you forget, instead of a cryptic `ImportError`. Set `--partition` /
   `--account` to what your CCV allocation uses (`sinfo -s` lists partitions;
   common GPU names are `gpu`, `gpu-he`).

**Locally** (your PC) you also need `pandas` + **`pyarrow`** — the launcher reads
the returned `detections.parquet`, and so does the app's loader
(`pip install pyarrow`).

`colloid_detect.py` is shipped automatically by the launcher; you do **not**
install it — it is the detection core split out of the desktop app.

## Run a job (from your PC)

Export your tuned detection parameters (the app's **Save settings** writes them,
or copy a preset's `params` block) to `params.json`, then:

```bash
cd "Microscope App/oscar"
python run_oscar_job.py \
    --video "D:/LAGB/long_run.mp4" \
    --params params.json \
    --host ssh.ccv.brown.edu --user YOURID \
    --chunk-size 400 --partition gpu
```

It probes the video, stages files, submits `--array=0-(nchunks-1)`, waits, merges,
and downloads `results/<jobname>/{detections.parquet, job_meta.json}`.

- `--no-submit` stages only and prints the exact `sbatch` line (dry run).
- Ctrl-C during the wait just detaches — the array keeps running on Oscar. To
  pick it back up, finish by hand (merge + `scp` below); **re-running the launcher
  starts a NEW job** unless you pass the same `--jobname` (the default name embeds
  a timestamp). There is no automatic resume.
- `--chunk-size` trades array width vs per-task overhead. ~400 frames/task is a
  good start; 12 k frames → 33 tasks. The launcher refuses to exceed the cluster
  `MaxArraySize` (`--max-array`, default 1001) — raise `--chunk-size` if it warns.
- `--decode-from-start` forces exact sequential frame decode on every task. The
  worker already **auto-detects** an inaccurate seek and falls back per-chunk, so
  you only need this if you want to force it fleet-wide (slower).
- The launcher checks `sacct` after the array leaves the queue and **refuses to
  merge unless every task COMPLETED** — a failed/timed-out array won't masquerade
  as a small, valid result.

## Load the result

In the desktop app: **File → Load cluster detections…**, pick the fetched
`job_meta.json` (with the original video present locally). The app rebuilds
tracks and every observable from the cluster coordinates and populates the
Video Analysis tab — playback, overlays, plots, and CSV export all work.

## Manual fallback (no launcher)

```bash
# on Oscar, in a job dir holding the video, job.json, and the 4 .py/.slurm files:
mkdir -p logs chunks          # Slurm opens logs/*.out BEFORE the script runs
sbatch --array=0-32 -p gpu -t 02:00:00 submit_detect.slurm
# after it finishes (check: sacct -j <jobid> -X -o State):
python merge_chunks.py --chunks-dir chunks --job job.json \
       --out detections.parquet --meta-out job_meta.json
# then scp detections.parquet + job_meta.json back to your PC.
```

## Notes & caveats

- **Frame seeking**: each task seeks to its chunk start. If your codec seeks
  inaccurately (rare for H.264/HEVC MP4), pass `--decode-from-start` to
  `detect_worker.py` for exact-but-slower sequential decode.
- **Empty frames**: a frame with zero detections isn't an error; `merge_chunks.py`
  reports which frames had none so you can tell that apart from a task that failed
  (check `logs/` on Oscar).
- **ROI**: if your `params.json` contains a `roi_polygon`, the cluster applies the
  same analysis region, so detections match a local ROI run.
- **Result format** (`schema_version: 1`): `detections.parquet` columns
  `frame,x,y,mass,ecc,signal` (float32, original-resolution px, 0-based absolute
  frame index); `job_meta.json` carries video name, W/H, fps, the full param dict,
  and completion stats.
