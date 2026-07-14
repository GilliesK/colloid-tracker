"""
run_capture.py  -  turnkey Basler capture I (Claude) run for you.
Opens the camera, forces Mono12 + fixed exposure (no auto), auto-checks that the
exposure isn't clipping, then saves N lossless 16-bit TIFFs at a set interval.

Works for BOTH:
  * dynamics series  -> you touch nothing
  * z-stack          -> you slowly sweep the focus knob during the capture

Usage:
    python run_capture.py --out dyn_raw   --n 360 --dt 0.25          # dynamics
    python run_capture.py --out zstack_raw --n 120 --dt 0.25          # z-stack (sweep focus)
    (add --exposure 8000 to override; default = keep camera's current value)
"""
import os, sys, time, argparse
import numpy as np
import tifffile
try:
    from pypylon import pylon
except ImportError:
    sys.exit("pypylon is required for camera capture — install with: pip install pypylon")


def try_set(cam, node, value):
    try:
        getattr(cam, node).SetValue(value); return True
    except Exception as e:
        print(f"  (couldn't set {node}={value}: {e})"); return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=360)
    ap.add_argument("--dt", type=float, default=0.25)      # seconds between frames
    ap.add_argument("--exposure", type=float, default=0)   # 0 => keep current
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    cam.Open()
    print("opened", cam.GetDeviceInfo().GetModelName(), cam.GetDeviceInfo().GetSerialNumber())

    try_set(cam, "PixelFormat", "Mono12")
    for n in ("ExposureAuto", "GainAuto"):
        try_set(cam, n, "Off")
    if args.exposure > 0:
        for n in ("ExposureTime", "ExposureTimeAbs"):
            if try_set(cam, n, float(args.exposure)):
                break
    try:
        exp = cam.ExposureTime.GetValue()
    except Exception:
        exp = "?"
    print(f"  PixelFormat={cam.PixelFormat.GetValue()}  Exposure={exp}us")

    # exposure sanity: grab a test frame, warn if clipping
    r = cam.GrabOne(2000)
    test = r.Array; r.Release()
    hi = int(test.max()); clip = float((test >= 65520).mean() * 100)
    print(f"  test frame: max={hi}/65535  clipping={clip:.2f}%  "
          f"({'OK' if clip < 0.5 else 'TOO BRIGHT - lower exposure!'})")

    print(f"\ncapturing {args.n} frames, every {args.dt}s "
          f"(~{args.n*args.dt:.0f}s total) -> {args.out}\\")
    print(">>> GO <<<  (z-stack: start sweeping focus NOW, one smooth pass)")
    t0 = time.perf_counter()
    saved = 0
    for i in range(args.n):
        target = t0 + i * args.dt
        r = cam.GrabOne(3000)
        if not r.GrabSucceeded():
            r.Release()
            print(f"  frame {i}: grab failed", flush=True)
            continue
        arr = r.Array.copy()          # copy BEFORE release
        r.Release()
        tifffile.imwrite(os.path.join(args.out, f"f_{i:04d}.tif"), arr)
        saved += 1
        if i % 40 == 0:
            print(f"  {i}/{args.n}  max={int(arr.max())}", flush=True)
        s = target + args.dt - time.perf_counter()
        if s > 0:
            time.sleep(s)
    print(f"  saved {saved} frames")
    cam.Close()
    print(f"done. {args.n} TIFFs in {args.out}\\  (elapsed {time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
