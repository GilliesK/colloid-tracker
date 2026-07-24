#!/usr/bin/env python3
"""Merge per-chunk detection Parquets into one detections.parquet + job_meta.json.

Runs on Oscar after the Slurm array finishes (or locally after fetch). Verifies
every expected chunk is present, concatenates in frame order, and writes a
metadata sidecar the desktop loader reads to reconstruct AnalysisData.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True, help="dir holding chunk_*.parquet")
    ap.add_argument("--job", required=True, help="job.json produced by the launcher")
    ap.add_argument("--out", required=True, help="output detections.parquet")
    ap.add_argument("--meta-out", required=True, help="output job_meta.json")
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

    parts = [pd.read_parquet(f) for f in files]
    det = pd.concat(parts, ignore_index=True)
    det = det.sort_values(["frame"]).reset_index(drop=True)
    det["frame"] = det["frame"].astype("int32")

    frames_present = np.unique(det["frame"].to_numpy())
    s, e = int(job["frame_start"]), int(job["frame_end"])
    expected = set(range(s, e + 1))
    got = set(int(x) for x in frames_present)
    missing = sorted(expected - got)     # frames with zero detections look "missing" too

    det.to_parquet(args.out, index=False)

    meta = dict(job)
    meta.update({
        "n_detections": int(len(det)),
        "frames_with_detections": int(len(got)),
        "frames_expected": int(len(expected)),
        # A frame may legitimately have 0 detections; report the list so the
        # loader/user can tell "empty frame" from "chunk never ran".
        "frames_without_detections": missing,
        "n_chunks_merged": len(files),
        "schema_version": 1,
    })
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"merged {len(files)} chunks -> {args.out}")
    print(f"  {len(det):,} detections over {len(got):,}/{len(expected):,} frames")
    if missing:
        print(f"  NOTE: {len(missing)} frames had no detections "
              f"(empty frames or an incomplete array job): "
              f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")


if __name__ == "__main__":
    main()
