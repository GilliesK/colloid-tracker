"""
grab_raw.py  -  capture RAW, lossless frames from a Basler camera (pypylon)
===========================================================================
Saves 16-bit TIFFs (Mono12, uncompressed) that feed straight into
buckling_analysis.py (set FRAMES_GLOB) and bond_asymmetry.py.

Why: the pylon Viewer video recorder writes 8-bit compressed mp4, which
smears the faint dark ("down") particles and defeats particle detection.
Mono12 + TIFF keeps every bit.

Install once:
    pip install pypylon tifffile numpy

Usage:
    python grab_raw.py single                       # one raw frame
    python grab_raw.py sequence  --n 300 --dt 0.2   # 300 frames, 0.2 s apart
    python grab_raw.py zstack    --n 25             # 25 focus slices (manual)

Common options:
    --exposure 8000     exposure time in microseconds (set for good, non-saturated contrast)
    --gain 0            analog gain (dB); keep low
    --out raw_frames    output folder
"""
import os
import sys
import time
import argparse

import numpy as np
import tifffile

try:
    from pypylon import pylon
except ImportError:
    sys.exit("pypylon not installed.  Run:  pip install pypylon tifffile")


def open_camera(exposure_us, gain_db, pixfmt="Mono12"):
    cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    cam.Open()
    print("Camera:", cam.GetDeviceInfo().GetModelName(),
          "SN", cam.GetDeviceInfo().GetSerialNumber())

    def try_set(node, value):
        try:
            getattr(cam, node).SetValue(value)
            return True
        except Exception as e:
            print(f"  (could not set {node}={value}: {e})")
            return False

    # full bit depth
    if not try_set("PixelFormat", pixfmt):
        try_set("PixelFormat", "Mono8")
        print("  !! fell back to Mono8 - check camera supports Mono12")
    # deterministic brightness: no auto anything
    try_set("ExposureAuto", "Off")
    try_set("GainAuto", "Off")
    # exposure node name varies by camera generation
    for n in ("ExposureTime", "ExposureTimeAbs"):
        if try_set(n, float(exposure_us)):
            break
    try_set("Gain", float(gain_db))
    print(f"  PixelFormat={cam.PixelFormat.GetValue()}  "
          f"Exposure~{exposure_us}us  Gain={gain_db}dB")
    return cam


def grab_one(cam):
    res = cam.GrabOne(5000)
    if not res.GrabSucceeded():
        raise RuntimeError("grab failed: " + res.GetErrorDescription())
    arr = res.Array.copy()
    res.Release()
    return arr  # numpy uint16 (Mono12 -> values 0..4095) or uint8


def save_tif(path, arr):
    tifffile.imwrite(path, arr)  # lossless, preserves 16-bit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["single", "sequence", "zstack"])
    ap.add_argument("--n", type=int, default=1, help="frames (sequence) or slices (zstack)")
    ap.add_argument("--dt", type=float, default=0.2, help="seconds between frames (sequence)")
    ap.add_argument("--exposure", type=float, default=8000, help="exposure, microseconds")
    ap.add_argument("--gain", type=float, default=0.0, help="gain, dB")
    ap.add_argument("--out", default="raw_frames")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cam = open_camera(args.exposure, args.gain)
    try:
        if args.mode == "single":
            a = grab_one(cam)
            p = os.path.join(args.out, "frame_single.tif")
            save_tif(p, a)
            print(f"saved {p}  shape={a.shape} dtype={a.dtype} "
                  f"min={a.min()} max={a.max()}  (max should be < {(1<<12)-1 if a.dtype==np.uint16 else 255} and not clipping)")

        elif args.mode == "sequence":
            print(f"grabbing {args.n} frames, dt={args.dt}s ...")
            t0 = time.perf_counter()
            for i in range(args.n):
                target = t0 + i * args.dt
                a = grab_one(cam)
                save_tif(os.path.join(args.out, f"frame_{i:05d}.tif"), a)
                if i % 20 == 0:
                    print(f"  {i}/{args.n}  max={a.max()}")
                dtsleep = target + args.dt - time.perf_counter()
                if dtsleep > 0:
                    time.sleep(dtsleep)
            print(f"done -> {args.out}\\frame_*.tif   "
                  f"(point buckling_analysis.py FRAMES_GLOB here)")

        elif args.mode == "zstack":
            print(f"Z-STACK: {args.n} slices. Step the MICROSCOPE focus by a fixed")
            print("amount (e.g. 0.2 um) between slices. Record your step size!")
            for i in range(args.n):
                input(f"  slice {i+1}/{args.n}: set focus, then press Enter...")
                a = grab_one(cam)
                save_tif(os.path.join(args.out, f"z_{i:03d}.tif"), a)
                print(f"    saved z_{i:03d}.tif  max={a.max()}")
            print(f"done -> {args.out}\\z_*.tif")
    finally:
        cam.Close()


if __name__ == "__main__":
    main()
