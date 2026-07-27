#!/usr/bin/env python3
"""Merge per-chunk detection Parquets into one detections.parquet + job_meta.json.

Runs as a compute-node Slurm job (merge.slurm). STREAMS the chunks — appends one
at a time to the output file with a single pyarrow ParquetWriter — so peak memory
is ~one chunk, not the whole dataset. A long run is hundreds of millions of
detection rows (tens of GB); loading them all and pd.concat'ing OOM-kills the job.

Chunks are already in frame order (chunk_00000 = frames [s..], chunk_00001 =
[s+chunk..], and each worker writes its frames in increasing order), so appending
in sorted-filename order yields a frame-sorted file with no global sort.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True, help="dir holding chunk_*.parquet")
    ap.add_argument("--job", required=True, help="job.json produced by the launcher")
    ap.add_argument("--out", required=True, help="output detections.parquet")
    ap.add_argument("--meta-out", required=True, help="output job_meta.json")
    ap.add_argument("--obs-dir", default=None, help="dir holding obs_*.parquet (optional)")
    ap.add_argument("--obs-out", default=None, help="output observables.parquet (optional)")
    args = ap.parse_args()

    with open(args.job) as f:
        job = json.load(f)

    files = sorted(glob.glob(os.path.join(args.chunks_dir, "chunk_*.parquet")))
    if not files:
        sys.stderr.write(f"ERROR: no chunk_*.parquet in {args.chunks_dir}\n"); sys.exit(2)

    # A missing chunk = a task that failed or timed out (the worker writes its
    # parquet only at the very end, so a killed task leaves NO file). Its frames
    # would otherwise vanish and look identical to genuinely empty frames. Refuse
    # to merge a partial result rather than emit a small, real-looking dataset.
    n_expected = int(job.get("n_chunks", 0))
    if n_expected and len(files) != n_expected:
        present = {int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files}
        missing_tasks = sorted(set(range(n_expected)) - present)
        sys.stderr.write(
            f"ERROR: found {len(files)} chunk files but expected {n_expected}. "
            f"Missing array task(s): {missing_tasks}\n"
            f"These tasks failed or timed out — inspect logs/ and re-run them "
            f"before merging. Refusing to write a partial detections.parquet.\n")
        sys.exit(3)

    # ---- streaming merge: one chunk resident at a time ----
    writer = None
    schema = None
    total_rows = 0
    frames_seen: set = set()
    for fp in files:
        t = pq.read_table(fp)            # one chunk (~chunk_size * particles rows)
        if schema is None:
            schema = t.schema
        if t.num_rows == 0:
            continue                     # empty chunk (all its frames had 0 detections)
        if writer is None:
            writer = pq.ParquetWriter(args.out, schema)
        writer.write_table(t)
        total_rows += t.num_rows
        frames_seen.update(np.unique(t.column("frame").to_numpy()).tolist())
    if writer is not None:
        writer.close()
    elif schema is not None:
        pq.ParquetWriter(args.out, schema).close()   # all empty -> valid empty file
    else:
        sys.stderr.write("ERROR: chunks contained no readable schema.\n"); sys.exit(2)

    s, e = int(job["frame_start"]), int(job["frame_end"])
    expected = set(range(s, e + 1))
    missing = sorted(expected - frames_seen)   # frames with zero detections

    meta = dict(job)
    meta.update({
        "n_detections": int(total_rows),
        "frames_with_detections": int(len(frames_seen)),
        "frames_expected": int(len(expected)),
        # A frame may legitimately have 0 detections; report the list so the
        # loader/user can tell "empty frame" from "chunk never ran".
        "frames_without_detections": missing,
        "n_chunks_merged": len(files),
        "schema_version": 1,
    })
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    # ---- optional: merge per-frame observables (one row/frame — small) ----
    if args.obs_dir and args.obs_out:
        obs_files = sorted(glob.glob(os.path.join(args.obs_dir, "obs_*.parquet")))
        if obs_files:
            import pyarrow as pa
            tables = [pq.read_table(f) for f in obs_files]
            obs = pa.concat_tables(tables)
            # frames are contiguous per chunk in filename order, so already sorted
            pq.write_table(obs, args.obs_out)
            print(f"  observables: {obs.num_rows:,} frame rows -> {args.obs_out}")
        else:
            print("  observables: none found (job ran without --analyze?)")

    print(f"merged {len(files)} chunks -> {args.out}")
    print(f"  {total_rows:,} detections over {len(frames_seen):,}/{len(expected):,} frames")
    if missing:
        print(f"  NOTE: {len(missing)} frames had no detections "
              f"(empty frames or an incomplete array job): "
              f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")


if __name__ == "__main__":
    main()
