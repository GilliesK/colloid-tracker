"""
dynamics_raw.py  -  flip dynamics from a RAW fixed-focus TIFF time-series.
Detection-light: the headline results (activity map, correlation time) need
NO particle detection, so they are robust to the foam-like morphology.
Outputs go to buckling_out/.
"""
import glob, re, os, sys
import numpy as np
import cv2
import tifffile
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GLOB = r"C:/Users/Ling_/Basler_a2A2840-48umPRO__40287597__20260703_150226278_*.tiff"
DS   = 2                     # downsample factor (memory/speed)
FPS  = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0   # ASSUMED capture rate
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "buckling_out")


def main():
    os.makedirs(OUT, exist_ok=True)
    fs = sorted(glob.glob(GLOB), key=lambda p: int(re.search(r"_(\d+)\.tiff$", p).group(1)))
    n = len(fs)
    a0 = tifffile.imread(fs[0])
    H, W = a0.shape[0] // DS, a0.shape[1] // DS
    print(f"{n} frames, working at {H}x{W} (downsample {DS}), assumed {FPS} fps")

    stack = np.empty((n, H, W), np.float32)
    ref = None
    for i, f in enumerate(fs):
        a = tifffile.imread(f).astype(np.float32)
        if DS > 1:
            a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
        # integer-pixel drift correction to frame 0
        if ref is None:
            ref = a.copy()
        else:
            dx, dy = cv2.phaseCorrelate(ref, a)[0]
            a = np.roll(np.roll(a, -int(round(dy)), 0), -int(round(dx)), 1)
        stack[i] = a
    dt = 1.0 / FPS

    # --- detection-free activity + correlation time ---
    mean = stack.mean(0)
    activity = stack.std(0)                      # per-pixel temporal fluctuation
    # normalise activity map for display
    amap = np.clip((activity - np.percentile(activity, 1)) /
                   (np.percentile(activity, 99) - np.percentile(activity, 1)), 0, 1)

    # temporal autocorrelation on the ACTIVE (flipping) pixels only, so the
    # frozen-region camera noise doesn't dominate and bias tau downward
    noise0 = np.percentile(activity, 20)
    act_ys, act_xs = np.where(activity > 4 * noise0)
    if len(act_ys) > 12000:
        sel = np.random.default_rng(0).choice(len(act_ys), 12000, replace=False)
        act_ys, act_xs = act_ys[sel], act_xs[sel]
    ys, xs = act_ys, act_xs
    print(f"autocorrelation on {len(ys)} active (flipping) pixels")
    ts = stack[:, ys, xs]                         # (n, npix)
    ts = ts - ts.mean(0, keepdims=True)
    F = np.fft.rfft(ts, n=2 * n, axis=0)
    ac = np.fft.irfft(F * np.conj(F), axis=0)[:n]
    ac /= np.arange(n, 0, -1)[:, None]
    C = ac.mean(1)
    C /= C[0]
    lags = np.arange(n) * dt

    def modelfun(t, tau, beta, c0):
        return c0 + (1 - c0) * np.exp(-(t / tau) ** beta)
    tau = beta = c0 = np.nan
    try:
        popt, _ = curve_fit(modelfun, lags, C, p0=[max(lags[1], dt*5), 0.7, max(C[-1], 0)],
                            bounds=([dt, 0.2, 0], [lags[-1]*5, 2, 1]), maxfev=20000)
        tau, beta, c0 = popt
    except Exception as e:
        print("  C(t) fit failed:", e)

    # figures
    plt.figure(figsize=(8, 8))
    plt.imshow(amap, cmap="inferno")
    plt.title("flip-activity map (bright = fluctuating, dark = frozen)")
    plt.axis("off")
    plt.colorbar(label="temporal fluctuation (norm.)", fraction=0.046)
    plt.savefig(os.path.join(OUT, "raw_activity_map.png"), dpi=130, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(lags, C, "o", ms=3, label="C(t) data")
    if np.isfinite(tau):
        tt = np.linspace(0, lags[-1], 300)
        plt.plot(tt, modelfun(tt, tau, beta, c0), "-", color="crimson",
                 label=f"tau={tau:.1f}s  beta={beta:.2f}  plateau={c0:.2f}")
    plt.xlabel(f"lag (s, assuming {FPS} fps)"); plt.ylabel("intensity autocorrelation")
    plt.title("temporal correlation (detection-free)")
    plt.legend(); plt.ylim(0, 1.02)
    plt.savefig(os.path.join(OUT, "raw_autocorrelation.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # fraction of the field that is "active" (fluctuation above a noise floor)
    noise = np.percentile(activity, 20)          # quiet pixels ~ camera noise
    active_frac = (activity > 3 * noise).mean()

    print("\n================ RAW DYNAMICS SUMMARY ================")
    print(f"frames                 : {n}   duration ~{n*dt:.1f}s (if {FPS} fps)")
    print(f"active-area fraction    : {active_frac*100:.1f}%  (pixels fluctuating > 3x noise)")
    if np.isfinite(tau):
        print(f"correlation time tau    : {tau:.1f}s   (in frames: {tau*FPS:.0f})")
        print(f"stretch beta            : {beta:.2f}   (<1 => glassy/heterogeneous)")
        print(f"frozen plateau c0       : {c0:.2f}   (fraction that never decorrelates)")
    print(f"\nfigures: raw_activity_map.png , raw_autocorrelation.png")
    print("NOTE: tau in seconds assumes %.0f fps - give me the real capture rate to fix the axis." % FPS)


if __name__ == "__main__":
    main()
