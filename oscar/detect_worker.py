#!/usr/bin/env python3
"""Oscar-side particle-detection worker (headless, no Qt).

Runs the SAME Crocker-Grier detection core as the desktop app (imported from
`colloid_detect`, the Qt-free module shared with colloid_app.py) on one frame
range of a video, and writes the raw per-frame detections to a Parquet chunk.

This is the "heavy lifting" half of the pipeline. Linking and structural
analysis run later on the local host, which resumes exactly where the desktop
app's TrackingWorker does after detection (edge filter -> link -> analyse), so
loading a cluster result into the Video Analysis tab is identical to a local
run — only faster, because thousands of frames were detected in parallel across
Slurm array tasks (optionally on GPU via cupy).

CLI:
  python detect_worker.py --video V.mp4 --params job.json \
         --frames START:END --out chunk_00007.parquet [--gpu] [--decode-from-start]

Output Parquet columns: frame(int32), x, y, mass, ecc, signal (float32).
Positions are in ORIGINAL full-resolution pixels, 0-based absolute frame index
— matching what the desktop app stores in AnalysisData.feats before linking.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import cv2

# The detection core. On Oscar, ship colloid_detect.py alongside this file (see
# oscar/README.md). It pulls in numpy/pandas/opencv/scipy (+ optional cupy) and
# imports NO Qt — that is the whole point of the split.
try:
    import colloid_detect as CD
except Exception as exc:                       # pragma: no cover
    sys.stderr.write(
        "ERROR: cannot import colloid_detect. Copy colloid_detect.py next to "
        "detect_worker.py (or add it to PYTHONPATH). Original error:\n"
        f"  {exc}\n")
    raise


def _roi_mask_from_polygon(poly, H, W):
    """Boolean detection mask from a [[x,y],...] polygon (or None -> full frame).

    Byte-for-byte identical to colloid_app._roi_mask_from_polygon (rounds float
    vertices with np.rint, NOT truncation) so a job that used a drawn ROI detects
    exactly the same particles on the cluster as it would locally."""
    if not poly or len(poly) < 3:
        return None
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.rint(np.asarray(poly, dtype=np.float64)).astype(np.int32)], 1)
    return m.astype(bool)


def detect_one_frame(gray32, p, roi_mask, use_gpu):
    """Detect on a single grayscale float32 frame. Byte-for-byte the same
    preprocessing + _fast_locate + minmass handling as
    colloid_app.TrackingWorker (full resolution, no cam_scale).

    The lattice-prediction search_mask is deliberately omitted: it is only a
    per-frame SPEED optimisation that always falls back to a full-frame search,
    so full detection here yields identical particles."""
    diam = max(3, int(p["diameter"]) | 1)
    sep  = max(diam, int(p["separation"]))
    invert = bool(p.get("invert", False))
    ring   = bool(p.get("ring_mode", False))
    dark   = bool(p.get("dark_disk_mode", False))
    ls, ll = p.get("lshort", 1), p.get("llong", 53)

    if dark:
        proc = CD._preprocess(gray32, False, ls, ll, False, 2.0,
                              denoise=p.get("denoise_method", "off"),
                              denoise_strength=p.get("denoise_strength", 10.0),
                              gamma=1.0, sharpen=0.0,
                              flatten_illum=bool(p.get("flatten_illum", False)),
                              flatten_sigma_frac=p.get("flatten_sigma_frac", 0.15))
        proc = CD._dark_disk_to_spot(proc, diam)
    else:
        g = gray32
        if invert:
            g = g.max() - g
        proc = CD._preprocess(g, p.get("use_bandpass", True), ls, ll, False, 2.0,
                              denoise=p.get("denoise_method", "off"),
                              denoise_strength=p.get("denoise_strength", 10.0),
                              gamma=p.get("gamma", 1.0),
                              sharpen=p.get("sharpen_amount", 0.0),
                              flatten_illum=bool(p.get("flatten_illum", False)),
                              flatten_sigma_frac=p.get("flatten_sigma_frac", 0.15))
        if ring:
            proc = CD._ring_to_spot(proc, diam)

    # minmass is the user's hard outlier floor, honored for plain and ring
    # (ring's plain matched-filter masses are on a normal scale). Only dark-disk
    # forces 0 (its ^4 response needs the adaptive filter below). Kept identical
    # to colloid_app.TrackingWorker so cluster detections match a local run.
    feats = CD._fast_locate(
        proc, diameter=diam, separation=sep,
        minmass=(0.0 if dark else float(p["minmass"])),
        percentile=int(p["percentile"]), invert=False,
        ecc_max=(p.get("ecc_max", 0.8) if p.get("use_ecc_filter", False) else None),
        reject_size_outliers=bool(p.get("reject_size_outliers", False)),
        size_outlier_mad_mult=p.get("size_outlier_mad_mult", 2.5),
        search_mask=roi_mask)
    if dark:
        feats = CD._dark_disk_minmass_filter(feats)
    return feats


def main():
    ap = argparse.ArgumentParser(description="Oscar detection worker (one frame range).")
    ap.add_argument("--video", required=True)
    ap.add_argument("--params", required=True, help="job.json holding the detection param dict")
    ap.add_argument("--frames", required=True, help="START:END inclusive, 0-based absolute frame index")
    ap.add_argument("--out", required=True, help="output .parquet chunk path")
    ap.add_argument("--gpu", action="store_true",
                    help="advisory: cupy is used automatically if importable; this just logs intent")
    ap.add_argument("--decode-from-start", action="store_true",
                    help="decode sequentially from frame 0 instead of seeking (exact but slower; "
                         "use if your codec seeks inaccurately)")
    args = ap.parse_args()

    with open(args.params) as f:
        job = json.load(f)
    p = job["params"] if "params" in job else job     # accept bare param dict too
    s, e = (int(v) for v in args.frames.split(":"))

    cap = cv2.VideoCapture(args.video, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        sys.stderr.write(f"ERROR: cannot open video {args.video}\n"); sys.exit(2)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, fr0 = cap.read()
    if not ok:
        sys.stderr.write("ERROR: cannot read first frame\n"); sys.exit(2)
    H, W = fr0.shape[:2]
    roi_mask = _roi_mask_from_polygon(p.get("roi_polygon"), H, W)
    e = min(e, N - 1)

    gpu = "cupy" if getattr(CD, "CUPY_OK", False) else "cpu"
    sys.stderr.write(f"[worker] {os.path.basename(args.video)} frames {s}..{e} "
                     f"({W}x{H}, {N} total) backend={gpu}\n"); sys.stderr.flush()

    # Position the reader at the chunk start.
    def _decode_from_start():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        c = 0
        while c < s:
            if not cap.grab():
                break
            c += 1
        return c

    if args.decode_from_start:
        cur = _decode_from_start()
    else:
        # Seek to the chunk start, then VERIFY the reader actually landed on
        # frame `s`. cv2 CAP_PROP_POS_FRAMES seeks to the nearest keyframe on
        # some codec/ffmpeg builds and can be off by a few frames — which would
        # silently mislabel every frame in this chunk and corrupt frame-index
        # parity with a local sequential-decode run. If the readback disagrees,
        # fall back to exact sequential decode rather than emit wrong indices.
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        landed = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        if landed != s:
            sys.stderr.write(
                f"[worker] WARNING: seek to frame {s} landed on {landed}; "
                f"falling back to exact sequential decode for this chunk. "
                f"(Pass --decode-from-start to all tasks to silence this.)\n")
            cur = _decode_from_start()
        else:
            cur = s

    frames_out, t0, ndet = [], time.time(), 0
    fr = cur
    while fr <= e:
        ok, bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        feats = detect_one_frame(gray, p, roi_mask, args.gpu)
        if feats is not None and len(feats):
            df = feats[["x", "y", "mass", "ecc", "signal"]].copy()
            df.insert(0, "frame", np.int32(fr))
            frames_out.append(df)
            ndet += len(df)
        if (fr - s) % 50 == 0:
            rate = (fr - s + 1) / max(1e-9, time.time() - t0)
            sys.stderr.write(f"[worker] frame {fr} ({rate:.1f} fps, {ndet} dets)\n")
            sys.stderr.flush()
        fr += 1
    cap.release()

    if frames_out:
        out = pd.concat(frames_out, ignore_index=True)
    else:
        out = pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                            [("frame", "int32"), ("x", "float32"), ("y", "float32"),
                             ("mass", "float32"), ("ecc", "float32"), ("signal", "float32")]})
    for c in ("x", "y", "mass", "ecc", "signal"):
        out[c] = out[c].astype("float32")
    out["frame"] = out["frame"].astype("int32")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False)
    sys.stderr.write(f"[worker] DONE frames {s}..{e}: {len(out)} detections -> {args.out} "
                     f"({time.time()-t0:.1f}s)\n")


if __name__ == "__main__":
    main()
