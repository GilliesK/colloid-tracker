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
| Preprocess + Crocker–Grier detection (ring / dark / plain) | **Oscar** array (CPU default) | `detect_worker.py` → `colloid_detect.py` |
| Per-frame ψ₆ / defect / 5-7 / boundary counts (`--analyze`) | **Oscar** array | `detect_worker.py` → `colloid_analysis.py` |
| Merge chunks → coordinates + observables | **Oscar compute node** (dependent Slurm job) | `merge.slurm` → `merge_chunks.py` |
| Linking, g(r), overlays, playback (a frame window) | **Local** | the app's existing pipeline |

Only coordinates come back (a few MB), never the video — the app already has it.

## One-time setup on Oscar

1. **SSH key** so the launcher isn't prompting for a password every step:
   `ssh-copy-id you@ssh.ccv.brown.edu`
2. **Python env** with: `numpy pandas opencv-python-headless scipy pyarrow`
   (add `cupy` matching Oscar's CUDA *only* if you'll use `--gpu`). Example:
   ```bash
   module load miniconda3
   conda create -n colloid python=3.11 numpy pandas scipy pyarrow -y
   conda activate colloid
   pip install opencv-python-headless
   pip install cupy-cuda12x        # match `module load cuda` version
   ```
3. **Point the jobs at that env**: edit **`env.sh`** (one file, sourced by both
   the detection and merge jobs) — uncomment your `conda activate` / `venv`
   line **and set `COLLOID_ENV_READY=1`**. The jobs fail fast with a clear
   message if you forget, instead of a cryptic `ImportError`. Set `--partition`
   / `--account` to what your CCV allocation uses (`sinfo -s` lists partitions;
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
    --chunk-size 400 --partition batch
```

**CPU by default — the speedup is parallelism, not the GPU.** Each task detects
at about local speed; the win is running *many* tasks at once. CPU partitions are
far less restricted than GPU, so you get more tasks running concurrently (and a
shorter queue) than fighting for scarce GPU nodes. Only add `--gpu` (which adds
`--gres=gpu:1` and uses the cupy path) if cupy is installed in the env and you've
set a GPU `--partition` — otherwise you queue for a GPU and still run on CPU.
Tune throughput with `--chunk-size`: smaller = more tasks = more parallelism, up
to the array-size cap.

**Analysis region (ROI).** To confine the cluster run to a freehand region — the
same as drawing one locally — draw it in the app with the **✎ ROI** button, then
**File → Export ROI for cluster…** to save a `*.roi.json`, and pass it:

```bash
python run_oscar_job.py ... --roi "D:/LAGB/long_run.roi.json"
```

The ROI is session-only in the app (not in saved settings), which is why it is
exported separately. The polygon is in full-resolution frame pixels; the worker
builds the *same* mask the app would and passes it to detection, so a cluster
ROI run detects exactly the particles a local ROI run would. The launcher checks
the ROI's frame size against the video and warns if the polygon falls outside it.

It probes the video, stages files, submits the detection array **and a dependent
merge job**, waits, and downloads `results/<jobname>/{detections.parquet,
job_meta.json}`.

While it waits it prints a live progress line each poll — tasks done, how many
are **running** (concurrency) vs **queued**, the per-frame detection time, and a
time-averaged **ETA** to the finished result:

```
[3/5] waiting (array -> merge; Ctrl-C to detach, jobs keep running)
      14:22:37  5/33 tasks done | 8 running, 20 queued | 0.041 s/frame/task | ETA 9m30s (~14:32:07)
      ...
      14:31:10  33/33 tasks done | 0 running, 0 queued | merging (running)...
```

**Nothing non-trivial runs on the login node.** Detection runs in the Slurm
array; the chunk merge is a *separate Slurm job* gated on the array succeeding
(`--dependency=afterok`), so even the concatenation happens on a compute node.
On the login node the launcher only ever calls `sbatch` / `squeue` / `sacct` /
`mkdir`. If any array task fails, the merge's `afterok` dependency is never
satisfied and Slurm cancels it — the launcher detects that and reports the
array failure instead of fetching a partial result. Use `--merge-partition` to
send the (CPU-only, GPU-free) merge to a batch partition.

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
tracks from the cluster coordinates and populates the Video Analysis tab —
playback, overlays, plots, and CSV export all work.

**Frame range.** For a long run the app first asks for a frame window to load
(default = the whole run). Linking + per-frame analysis of 100k+ frames locally
is heavy, so you can load, say, just the last 10k for detailed inspection.

**Whole-run graphs come from the cluster.** With `--analyze` (default), the
cluster computes per-frame **hexatic ψ₆, defect fraction, 5-7 pair count, and
boundary count** in parallel and returns them as `observables.parquet`. The
defect/hexatic plot is populated from that over the **entire** run — even if you
only loaded a frame subset — with no heavy local recompute. Pass `--no-analyze`
to skip it.

## Manual fallback (no launcher)

```bash
# on Oscar, in a job dir holding the video, job.json, and the shipped files:
mkdir -p logs chunks          # Slurm opens logs/*.out BEFORE the script runs
AID=$(sbatch --parsable --array=0-32 -p gpu -t 02:00:00 submit_detect.slurm)
# merge runs on a COMPUTE NODE after the array succeeds — never on the login node:
sbatch --dependency=afterok:$AID -p batch merge.slurm
# when the merge job COMPLETES (sacct -j <mergeid> -X -o State),
# scp detections.parquet + job_meta.json back to your PC.
```
Do **not** run `python merge_chunks.py` directly on the login node — submit
`merge.slurm` (or wrap it in `srun`/`salloc`). The launcher does this for you.

## Notes & caveats

- **Frame seeking**: each task seeks to its chunk start. If your codec seeks
  inaccurately (rare for H.264/HEVC MP4), pass `--decode-from-start` to
  `detect_worker.py` for exact-but-slower sequential decode.
- **Empty frames**: a frame with zero detections isn't an error; `merge_chunks.py`
  reports which frames had none so you can tell that apart from a task that failed
  (check `logs/` on Oscar).
- **ROI (analysis region)**: to confine a cluster run to a freehand region, draw
  it in the app (the “✎ ROI” button), then **File → Export ROI for cluster…** to
  save `something.roi.json`, and pass it to the launcher:
  ```
  python run_oscar_job.py … --roi "D:/LAGB/LAGB_long.roi.json"
  ```
  The worker builds the exact same polygon mask the app does (`np.rint`-rounded
  vertices, verified byte-identical), so cluster detection matches a local ROI
  run. The ROI is session-only in the app, which is why it exports separately
  rather than riding along in `--params`. `--roi` overrides any `roi_polygon`
  already in `--params`; omit it for full-frame detection.
- **Result format** (`schema_version: 1`): `detections.parquet` columns
  `frame,x,y,mass,ecc,signal` (float32, original-resolution px, 0-based absolute
  frame index); `observables.parquet` (when `--analyze`) columns
  `frame,n_particles,mean_psi6,frac_defect,n_57,n_lagb` (one row per frame);
  `job_meta.json` carries video name, W/H, fps, the full param dict, and
  completion stats. Observables are computed on the edge-filtered detections, so
  they match a local `analyze_frame` exactly (verified byte-for-byte).
