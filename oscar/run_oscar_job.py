#!/usr/bin/env python3
"""Local launcher: offload detection of a long video to Brown's Oscar cluster.

Stages the video + shared detection code to Oscar, submits a Slurm GPU array
(one task per frame chunk), waits for it, merges the chunks, and fetches back a
compact detections bundle (detections.parquet + job_meta.json). Load that bundle
in the desktop app via  File -> Load cluster detections…  to get the full
Video Analysis tab (playback, overlays, g(r)/psi6, LAGB, CSV export) computed
locally from the cluster-detected coordinates.

Only OpenSSH (ssh/scp, built into Windows 10+/macOS/Linux) and cv2 are needed
locally. Set up passwordless SSH to Oscar first (ssh-copy-id / an ssh key), or
you'll be prompted for your password at each step.

Example:
  python run_oscar_job.py --video "D:/LAGB/run12.mp4" --params params.json \
      --host ssh.ccv.brown.edu --user YOURID --chunk-size 400

Dry run (stage + print commands, do not submit):
  python run_oscar_job.py ... --no-submit
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, shlex
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIP = ["detect_worker.py", "merge_chunks.py",
        "submit_detect.slurm", "merge.slurm", "env.sh"]
CORE = "colloid_detect.py"          # shared detection core, lives one dir up


def sh(cmd, **kw):
    print("  $", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, **kw)


def ssh(host, user, remote_cmd, capture=False):
    cmd = ["ssh", f"{user}@{host}", remote_cmd]
    if capture:
        r = sh(cmd, capture_output=True, text=True)
        return r.stdout.strip(), r.returncode
    return sh(cmd).returncode


def scp(src, host, user, dst):
    # dst is a REMOTE path parsed by the remote shell -> quote it (spaces/meta).
    return sh(["scp", "-C", str(src), f"{user}@{host}:{shlex.quote(dst)}"]).returncode


def scp_from(host, user, remote_path, local_dst):
    return sh(["scp", "-C", f"{user}@{host}:{shlex.quote(remote_path)}", str(local_dst)]).returncode


def _dur(sec):
    """Compact duration, e.g. 1h03m / 4m12s / 45s."""
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"


def _clock(epoch):
    return time.strftime("%H:%M:%S", time.localtime(epoch))


def probe_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, fr = cap.read()
    if not ok:
        sys.exit("ERROR: cannot read first frame")
    h, w = fr.shape[:2]
    cap.release()
    return n, float(fps), int(w), int(h)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--params", required=True,
                    help="JSON of the detection param dict (export from the app's Save settings, "
                         "or a saved preset's 'params').")
    ap.add_argument("--host", default="ssh.ccv.brown.edu")
    ap.add_argument("--user", required=True)
    ap.add_argument("--remote-base", default="~/colloid_jobs")
    ap.add_argument("--jobname", default=None, help="default: video stem + timestamp")
    ap.add_argument("--chunk-size", type=int, default=400, help="frames per Slurm array task")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1, help="-1 = last frame")
    ap.add_argument("--partition", default="gpu", help="partition for the detection array")
    ap.add_argument("--merge-partition", default=None,
                    help="partition for the (CPU-only) merge job; defaults to --partition. "
                         "Set a CPU/batch partition to avoid holding a GPU for the merge.")
    ap.add_argument("--account", default=None)
    ap.add_argument("--time", default="02:00:00", help="per-task walltime")
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--no-submit", action="store_true", help="stage everything but do not sbatch")
    ap.add_argument("--poll", type=int, default=30, help="seconds between squeue polls")
    ap.add_argument("--decode-from-start", action="store_true",
                    help="force exact sequential frame decode on every task (slower) instead of "
                         "seek-and-verify. Use if your codec seeks inaccurately.")
    ap.add_argument("--max-array", type=int, default=1001,
                    help="cluster MaxArraySize (default Slurm value); the run refuses to submit "
                         "more tasks than this — raise --chunk-size instead.")
    ap.add_argument("--roi", default=None,
                    help="JSON file with a drawn ROI polygon (from the app's File -> "
                         "'Export ROI for cluster…'). Confines cluster detection to that region, "
                         "identical to a local ROI run. Overrides any roi_polygon in --params.")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"ERROR: video not found: {video}")
    core = HERE.parent / CORE
    if not core.exists():
        sys.exit(f"ERROR: {CORE} not found next to the app ({core}). "
                 "It is created by splitting the detection core out of colloid_app.py — "
                 "run the desktop app once after updating, or copy it here.")
    with open(args.params) as f:
        params = json.load(f)
    params = params.get("params", params)          # accept a full settings file too

    n, fps, w, h = probe_video(video)

    # Optional ROI: confine detection to a drawn polygon (session-only in the
    # app, so it isn't in --params — it's exported separately). The polygon is
    # in full-resolution frame pixels; the worker builds the same mask the app
    # would and passes it to _fast_locate, so detection matches a local ROI run.
    if args.roi:
        with open(args.roi) as f:
            roi = json.load(f)
        poly = roi.get("roi_polygon", roi) if isinstance(roi, dict) else roi
        if not (isinstance(poly, list) and len(poly) >= 3
                and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in poly)):
            sys.exit(f"ERROR: --roi {args.roi} must contain a polygon of >=3 [x,y] points "
                     f"(a bare list or {{'roi_polygon': [...]}}).")
        rw = roi.get("W") if isinstance(roi, dict) else None
        rh = roi.get("H") if isinstance(roi, dict) else None
        if (rw and rw != w) or (rh and rh != h):
            sys.exit(f"ERROR: ROI was drawn on a {rw}x{rh} frame but the video is {w}x{h}. "
                     "Re-export the ROI on this video (the polygon is in frame pixels).")
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        if min(xs) < 0 or max(xs) > w or min(ys) < 0 or max(ys) > h:
            print(f"WARNING: ROI extends outside the {w}x{h} frame — it will be clipped.")
        params = dict(params, roi_polygon=poly)
        print(f"roi     : {len(poly)}-point polygon, "
              f"x {min(xs):.0f}..{max(xs):.0f}  y {min(ys):.0f}..{max(ys):.0f}")
    s = max(0, args.start)
    e = (n - 1) if args.end < 0 else min(args.end, n - 1)
    if s > e:
        sys.exit(f"ERROR: start ({s}) is past end ({e}); nothing to process.")
    nchunks = (e - s) // args.chunk_size + 1
    if nchunks > args.max_array:
        sys.exit(f"ERROR: {nchunks} array tasks exceeds --max-array ({args.max_array}). "
                 f"Raise --chunk-size (e.g. {(e - s)//(args.max_array - 1) + 1}) to fit.")
    ts = time.strftime("%Y%m%d_%H%M%S")
    jobname = args.jobname or f"{video.stem}_{ts}"
    # Resolve a leading ~ to the remote $HOME so paths can be safely quoted
    # (a quoted "~/x" is NOT tilde-expanded by the remote shell). One round-trip.
    remote_base = args.remote_base.rstrip("/")
    if remote_base.startswith("~"):
        home, hrc = ssh(args.host, args.user, "printf %s \"$HOME\"", capture=True)
        if hrc != 0 or not home:
            sys.exit(f"ERROR: could not resolve remote $HOME on {args.host} "
                     f"(ssh rc={hrc}). Check connectivity / --user.")
        remote_base = home + remote_base[1:]
    remote_dir = f"{remote_base}/{jobname}"
    # Quote every path interpolated into a REMOTE shell command (ssh string /
    # scp remote target). Guards against spaces and shell-metacharacter
    # injection flowing from the video name -> jobname.
    rq_dir = shlex.quote(remote_dir)

    job = {
        "video": video.name, "params": params,
        "n_frames": n, "fps": fps, "W": w, "H": h,
        "frame_start": s, "frame_end": e,
        "chunk_size": args.chunk_size, "n_chunks": nchunks,
        "decode_from_start": bool(args.decode_from_start),
        "created": ts, "schema_version": 1,
    }
    local_job = HERE / f"job_{jobname}.json"
    local_job.write_text(json.dumps(job, indent=2))

    print(f"\nvideo   : {video.name}  ({w}x{h}, {n} frames, {fps:.2f} fps)")
    print(f"range   : {s}..{e}  ->  {nchunks} chunks of {args.chunk_size} frames")
    print(f"oscar   : {args.user}@{args.host}:{remote_dir}")
    print(f"backend : GPU array on partition '{args.partition}'\n")

    # ---- stage ----
    print("[1/5] staging files to Oscar")
    ssh(args.host, args.user, f"mkdir -p {rq_dir}/logs {rq_dir}/chunks")
    for fn in SHIP:
        scp(HERE / fn, args.host, args.user, f"{remote_dir}/{fn}")
    scp(core, args.host, args.user, f"{remote_dir}/{CORE}")
    scp(local_job, args.host, args.user, f"{remote_dir}/job.json")
    scp(video, args.host, args.user, f"{remote_dir}/{video.name}")

    dfs = " --decode-from-start" if args.decode_from_start else ""
    merge_part = args.merge_partition or args.partition
    if args.no_submit:
        print("\n--no-submit: staged only. To run manually on Oscar (all compute on")
        print("nodes — do NOT run merge_chunks.py on the login node):")
        print(f"  ssh {args.user}@{args.host}")
        print(f"  cd {remote_dir} && mkdir -p logs chunks")
        print(f"  AID=$(sbatch --parsable --array=0-{nchunks-1} "
              f"-p {args.partition} -t {args.time} submit_detect.slurm)")
        print(f"  sbatch --dependency=afterok:$AID "
              f"-p {merge_part} merge.slurm   # merge runs on a compute node")
        return

    # ---- submit: detection array, then a dependent merge job ----
    # ALL non-trivial work runs in submitted jobs on compute nodes. The login
    # node only ever runs sbatch / squeue / sacct / mkdir here. The merge is a
    # separate Slurm job gated on the array succeeding (afterok), so nothing
    # heavy — not even the chunk concatenation — touches the login node.
    print("[2/5] submitting detection array + dependent merge job")
    acct = f"-A {shlex.quote(args.account)} " if args.account else ""
    submit = (f"cd {rq_dir} && sbatch --parsable --array=0-{nchunks-1} "
              f"-p {shlex.quote(args.partition)} -t {shlex.quote(args.time)} "
              f"{acct}submit_detect.slurm")
    out, rc = ssh(args.host, args.user, submit, capture=True)
    if rc != 0 or not out:
        sys.exit(f"ERROR: sbatch (array) failed (rc={rc}). Output:\n{out}")
    array_id = out.split(";")[0].split()[-1].strip()
    print(f"      array job {array_id} ({nchunks} tasks)")

    merge_submit = (f"cd {rq_dir} && sbatch --parsable "
                    f"--dependency=afterok:{shlex.quote(array_id)} "
                    f"-p {shlex.quote(merge_part)} {acct}merge.slurm")
    out, rc = ssh(args.host, args.user, merge_submit, capture=True)
    if rc != 0 or not out:
        sys.exit(f"ERROR: sbatch (merge) failed (rc={rc}). Output:\n{out}")
    merge_id = out.split(";")[0].split()[-1].strip()
    print(f"      merge job {merge_id} (runs after the array succeeds)")

    # ---- wait: array -> merge, with a live progress readout ----
    # Each poll reports: how many tasks are done, how many are RUNNING (usage /
    # concurrency) vs PENDING (queue depth), the per-frame detection time, and a
    # time-averaged ETA to the finished result. All from trivial squeue calls.
    print("[3/5] waiting (array -> merge; Ctrl-C to detach, jobs keep running)")
    total = nchunks
    t_wait0 = time.time()
    hist = []                                # (t, done_tasks) for a windowed rate
    eta_ema = None                           # smoothed ETA seconds
    window = max(90.0, args.poll * 4)        # rate-averaging window (s)
    while True:
        # Array element states. -r expands the array so a PENDING range counts
        # as its individual tasks, not one line. Completed/failed tasks have
        # left the queue, so done = total - still-in-queue.
        aq, _ = ssh(args.host, args.user,
                    f"squeue -j {shlex.quote(array_id)} -h -r -o %T 2>/dev/null "
                    f"| sort | uniq -c", capture=True)
        counts = {}
        for ln in aq.splitlines():
            parts = ln.split()
            if len(parts) == 2 and parts[0].isdigit():
                counts[parts[1]] = int(parts[0])
        running = counts.get("RUNNING", 0)
        pending = sum(v for k, v in counts.items() if k not in ("RUNNING",))
        in_queue = sum(counts.values())
        done = max(0, total - in_queue)

        merge_state, _ = ssh(args.host, args.user,
                             f"squeue -j {shlex.quote(merge_id)} -h -o %T 2>/dev/null",
                             capture=True)
        merge_state = merge_state.strip()

        if in_queue == 0 and not merge_state:
            print(f"      {time.strftime('%H:%M:%S')}  all tasks left the queue")
            break

        now = time.time()
        hist.append((now, done))
        while len(hist) > 2 and now - hist[0][0] > window:
            hist.pop(0)
        # Time-averaged completion rate (tasks/s): windowed slope, falling back
        # to the run-long average until the window fills.
        rate = 0.0
        if len(hist) >= 2 and hist[-1][1] > hist[0][1] and hist[-1][0] > hist[0][0]:
            rate = (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0])
        elif done > 0:
            rate = done / max(1e-9, now - t_wait0)

        # Per-task detection cost: with `running` tasks in flight, one task
        # clears chunk_size frames every running/rate seconds. (ASCII only —
        # a Windows console mangles non-ASCII glyphs.)
        if rate > 0 and args.chunk_size > 0:
            spf = (max(running, 1) / rate) / args.chunk_size
            spf_str = f"{spf:.3f} s/frame/task"
        else:
            spf_str = "s/frame n/a"

        if in_queue == 0 and merge_state:
            eta_str = f"merging ({merge_state.lower()})..."
        elif len(hist) < 2:
            eta_str = "ETA measuring..."
        elif rate > 0:
            eta = (total - done) / rate
            eta_ema = eta if eta_ema is None else 0.5 * eta_ema + 0.5 * eta
            eta_str = f"ETA {_dur(eta_ema)} (~{_clock(now + eta_ema)})"
        elif running == 0:
            eta_str = "ETA n/a (waiting in queue for a node)"
        else:
            eta_str = "ETA n/a (no tasks finished yet)"

        print(f"      {time.strftime('%H:%M:%S')}  {done}/{total} tasks done "
              f"| {running} running, {pending} queued | {spf_str} | {eta_str}")
        time.sleep(args.poll)

    # ---- verify on the accounting DB (still trivial login-node calls) ----
    print("[4/5] verifying completion")
    ast, _ = ssh(args.host, args.user,
                 f"sacct -j {shlex.quote(array_id)} -n -X -o State 2>/dev/null | sort | uniq -c",
                 capture=True)
    mst, _ = ssh(args.host, args.user,
                 f"sacct -j {shlex.quote(merge_id)} -n -X -o State 2>/dev/null",
                 capture=True)
    print(f"      array states:\n{ast}\n      merge state: {mst.strip()}")
    if "COMPLETED" not in mst:
        # afterok not satisfied (an array task failed) -> merge is CANCELLED /
        # DependencyNeverSatisfied; or the merge itself failed (e.g. a missing
        # chunk, which merge_chunks.py refuses to merge).
        sys.exit("ERROR: the merge job did not COMPLETE — the result is not ready.\n"
                 f"       array states:\n{ast}\n"
                 f"       merge state : {mst.strip()}\n"
                 f"       If the array had FAILED/TIMEOUT tasks the merge is cancelled by its\n"
                 f"       afterok dependency. Inspect {remote_dir}/logs on Oscar, fix the\n"
                 f"       failed range, and re-run.")

    # ---- fetch ----
    print("[5/5] fetching results")
    rdir = Path(args.results_dir) / jobname
    rdir.mkdir(parents=True, exist_ok=True)
    for fn in ("detections.parquet", "job_meta.json"):
        scp_from(args.host, args.user, f"{remote_dir}/{fn}", rdir / fn)

    print(f"\nDONE. Bundle: {rdir}")
    print("Open the desktop app -> File -> Load cluster detections… and pick "
          f"{rdir / 'job_meta.json'} (with the original video available locally).")


if __name__ == "__main__":
    main()
