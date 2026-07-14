"""
buckling_analysis.py
=====================
Quantify up/down "buckling" dynamics in a brightfield video of a confined
colloidal monolayer (silica between coverslips), following the analysis logic
of Han et al., Nature 456, 898 (2008).

What it does
------------
1. Locates every particle on a reference frame (detects BOTH bright and dark
   centered particles, so "up" and "down" are both found).
2. Classifies each particle up/down from its centre brightness using a global
   bimodal (Otsu) threshold -> Ising spin s = +/-1.
3. Because the lattice barely moves laterally, it samples the SAME sites in
   every frame (with optional drift correction) -> a spin-vs-time series.
4. Outputs:
     - up-fraction map        (shows the active buckling band vs jammed band)
     - flip-activity map       (flips per particle per second)
     - spin autocorrelation C(t) with a stretched-exponential fit -> tau, beta
     - <N_f>, mean frustrated bonds per particle (Delaunay), + a map
     - a detection-overlay and a brightness histogram for sanity-checking
   plus a printed summary.

Usage
-----
    python buckling_analysis.py [path_to_video.mp4]

Tune the CONFIG block below. The ONE parameter that matters most is
DIAMETER_PX (apparent particle diameter in pixels). Check detection_overlay.png
after the first run and adjust if particles are missed or over-counted.

Requires: numpy, scipy, opencv-python (cv2), trackpy, pandas, matplotlib.
(For raw TIFF/PNG frame folders instead of mp4, see FRAMES_GLOB below - use
those if you have them; mp4 compression smears the faint flips.)
"""

import os
import sys
import glob
import numpy as np
import cv2
import pandas as pd
import trackpy as tp
from scipy.spatial import Delaunay
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------ CONFIG ------------------------------
VIDEO = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\Ling_\Basler_a2A2840-48umPRO__40287597__20260703_135804200.mp4"
FRAMES_GLOB = None          # e.g. r"C:\path\frames\*.tif" ; overrides VIDEO if set

OUTDIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "buckling_out")

UM_PER_PIXEL = 3.0 / 59.0   # calibration: 3 um particle ~ 59 px spacing (edit!)
FPS          = 25.0         # camera frame rate

DIAMETER_PX  = 45           # apparent particle diameter, ODD int. <-- KEY KNOB
SEPARATION   = 50           # min centre-to-centre separation (px), ~0.85*spacing
MINMASS_PCT  = 40           # keep detections above this percentile of integrated mass

DOWNSCALE    = 1.0          # 0.5 = analyse at half res (4x faster); 1.0 = full
FRAME_STRIDE = 5            # analyse every Nth frame (25 fps / 5 = 5 fps)
MAX_FRAMES   = 400          # cap number of analysed frames
REF_FRAME    = 0            # frame index used to detect the lattice

SAMPLE_RADIUS = 3           # px disk radius for centre-intensity sampling
SMOOTH_WIN    = 3           # temporal median-filter window (odd) to kill 1-frame noise
DRIFT_CORRECT = True        # integer-pixel drift correction via phase correlation

ROI = None                  # (x0, y0, x1, y1) in ORIGINAL px to restrict analysis, or None
# --------------------------------------------------------------------


def odd(n):
    n = int(round(n))
    return n + 1 if n % 2 == 0 else max(1, n)


def log(msg):
    print(msg, flush=True)


# ----------------------------- I/O ----------------------------------
def load_frames():
    """Yield (index, gray_float32) for the analysed frames, plus total count."""
    frames = []
    if FRAMES_GLOB:
        files = sorted(glob.glob(FRAMES_GLOB))[::FRAME_STRIDE][:MAX_FRAMES]
        for f in files:
            im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            frames.append(im)
    else:
        cap = cv2.VideoCapture(VIDEO)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {VIDEO}")
        i = 0
        while len(frames) < MAX_FRAMES:
            ok, im = cap.read()
            if not ok:
                break
            if i % FRAME_STRIDE == 0:
                frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
            i += 1
        cap.release()

    proc = []
    for im in frames:
        if ROI:
            x0, y0, x1, y1 = ROI
            im = im[y0:y1, x0:x1]
        if DOWNSCALE != 1.0:
            im = cv2.resize(im, None, fx=DOWNSCALE, fy=DOWNSCALE,
                            interpolation=cv2.INTER_AREA)
        proc.append(im.astype(np.float32))
    return proc


def estimate_drift(ref, frame):
    """Integer (dx, dy) shift of frame relative to ref via phase correlation."""
    (dx, dy), _ = cv2.phaseCorrelate(ref, frame)
    return int(round(dx)), int(round(dy))


# --------------------------- detection ------------------------------
def detect_sites(img, diam, sep):
    """Detect all particle centres: bright-centred (up) AND dark-centred (down)."""
    parts = []
    for invert in (False, True):
        f = tp.locate(img, diam, invert=invert, separation=sep, engine="auto")
        if len(f):
            f = f[f["mass"] > np.percentile(f["mass"], MINMASS_PCT)]
            parts.append(f[["x", "y"]].values)
    pts = np.vstack(parts) if parts else np.empty((0, 2))

    # dedupe: greedily drop points closer than sep*0.6
    if len(pts):
        from scipy.spatial import cKDTree
        keep = np.ones(len(pts), bool)
        tree = cKDTree(pts)
        for i, j in sorted(tree.query_pairs(r=sep * 0.6)):
            if keep[i] and keep[j]:
                keep[j] = False
        pts = pts[keep]
    return pts


def sample_centres(img, coords, r):
    """Mean intensity in a small disk around each (x, y). Vectorised."""
    ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
    mask = (xs**2 + ys**2) <= r**2
    ox, oy = xs[mask], ys[mask]
    H, W = img.shape
    X = np.clip(coords[:, 0][:, None] + ox[None, :], 0, W - 1).astype(int)
    Y = np.clip(coords[:, 1][:, None] + oy[None, :], 0, H - 1).astype(int)
    return img[Y, X].mean(axis=1)


def otsu_threshold(vals, nbins=256):
    hist, edges = np.histogram(vals, bins=nbins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = np.cumsum(hist)
    wb = w / w[-1]
    mean = np.cumsum(hist * centres)
    mtot = mean[-1] / w[-1]
    mb = np.divide(mean, np.maximum(w, 1))
    mf = np.divide(mtot * w[-1] - mean, np.maximum(w[-1] - w, 1))
    between = wb * (1 - wb) * (mb - mf) ** 2
    return centres[np.argmax(between)]


# ------------------------------ fits --------------------------------
def stretched(t, c0, tau, beta):
    return c0 + (1 - c0) * np.exp(-(np.abs(t) / tau) ** beta)


# ------------------------------ main --------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    diam = odd(DIAMETER_PX * DOWNSCALE)
    sep = max(1, SEPARATION * DOWNSCALE)
    dt = FRAME_STRIDE / FPS
    px_um = UM_PER_PIXEL / DOWNSCALE  # analysed-pixel size in um

    log(f"Loading frames from: {VIDEO if not FRAMES_GLOB else FRAMES_GLOB}")
    frames = load_frames()
    if not frames:
        raise RuntimeError("No frames loaded.")
    log(f"  {len(frames)} frames, size {frames[0].shape}, dt={dt:.3f}s, "
        f"detect diameter={diam}px")

    ref = frames[min(REF_FRAME, len(frames) - 1)]
    log("Detecting particles on reference frame (both polarities)...")
    coords = detect_sites(ref, diam, sep)
    log(f"  {len(coords)} particles detected")
    if len(coords) < 50:
        log("  !! Very few particles - adjust DIAMETER_PX / SEPARATION and re-run.")

    # overlay for sanity check
    plt.figure(figsize=(8, 8))
    plt.imshow(ref, cmap="gray")
    plt.scatter(coords[:, 0], coords[:, 1], s=6, facecolors="none",
                edgecolors="lime", linewidths=0.4)
    plt.title(f"detection overlay  (n={len(coords)})")
    plt.axis("off")
    plt.savefig(os.path.join(OUTDIR, "detection_overlay.png"), dpi=130,
                bbox_inches="tight")
    plt.close()

    # drift + intensity sampling over all frames
    log("Sampling centre intensities across frames (drift-corrected)...")
    ref32 = ref.astype(np.float32)
    intens = np.zeros((len(coords), len(frames)), np.float32)
    for k, fr in enumerate(frames):
        c = coords.copy()
        if DRIFT_CORRECT and k > 0:
            dx, dy = estimate_drift(ref32, fr.astype(np.float32))
            c[:, 0] += dx
            c[:, 1] += dy
        intens[:, k] = sample_centres(fr, c, SAMPLE_RADIUS)

    # global threshold from pooled intensities -> spins
    thr = otsu_threshold(intens.ravel())
    log(f"  Otsu brightness threshold = {thr:.1f} (0-255)")
    states = np.where(intens > thr, 1, -1).astype(np.int8)  # +1 = bright, -1 = dark

    # temporal median smoothing to remove single-frame noise flips
    if SMOOTH_WIN >= 3:
        from scipy.ndimage import median_filter
        states = median_filter(states, size=(1, SMOOTH_WIN))

    up_frac_site = (states == 1).mean(axis=1)           # per particle, over time
    flips = np.count_nonzero(np.diff(states, axis=1), axis=1)
    flip_rate = flips / (dt * (states.shape[1] - 1))    # flips / particle / s
    global_up = (states == 1).mean()

    # brightness histogram
    plt.figure(figsize=(7, 4))
    plt.hist(intens.ravel(), bins=120, color="#444")
    plt.axvline(thr, color="crimson", lw=2, label=f"threshold {thr:.0f}")
    plt.xlabel("centre intensity (0-255)")
    plt.ylabel("count")
    plt.title("brightness histogram (bimodal => two heights)")
    plt.legend()
    plt.savefig(os.path.join(OUTDIR, "brightness_histogram.png"), dpi=130,
                bbox_inches="tight")
    plt.close()

    # up-fraction map
    plt.figure(figsize=(8, 8))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=up_frac_site, s=10,
                     cmap="coolwarm", vmin=0, vmax=1)
    plt.gca().invert_yaxis()
    plt.colorbar(sc, label="time-avg up-fraction")
    plt.title("up-fraction map (blue=down/frozen, red=up)")
    plt.axis("equal"); plt.axis("off")
    plt.savefig(os.path.join(OUTDIR, "up_fraction_map.png"), dpi=130,
                bbox_inches="tight")
    plt.close()

    # flip-activity map
    plt.figure(figsize=(8, 8))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=flip_rate, s=10, cmap="viridis")
    plt.gca().invert_yaxis()
    plt.colorbar(sc, label="flips / particle / s")
    plt.title("flip-activity map (bright=flipping band)")
    plt.axis("equal"); plt.axis("off")
    plt.savefig(os.path.join(OUTDIR, "flip_activity_map.png"), dpi=130,
                bbox_inches="tight")
    plt.close()

    # spin autocorrelation C(t) via FFT, averaged over particles
    S = states.astype(np.float64)
    n = S.shape[1]
    F = np.fft.rfft(S, n=2 * n, axis=1)
    ac = np.fft.irfft(F * np.conj(F), axis=1)[:, :n]
    ac /= np.arange(n, 0, -1)[None, :]      # unbiased per lag
    C = ac.mean(axis=0)
    C /= C[0]
    lags = np.arange(n) * dt

    tau = beta = c0 = np.nan
    fitmask = lags <= lags[-1]
    try:
        p0 = [max(C[-1], 0.0), max(lags[1], dt * 3), 0.6]
        popt, _ = curve_fit(stretched, lags[fitmask], C[fitmask], p0=p0,
                            bounds=([0, dt, 0.2], [1, lags[-1] * 5, 2.0]),
                            maxfev=20000)
        c0, tau, beta = popt
    except Exception as e:
        log(f"  (C(t) fit failed: {e})")

    plt.figure(figsize=(7, 5))
    plt.plot(lags, C, "o", ms=4, label="C(t) data")
    if np.isfinite(tau):
        tt = np.linspace(0, lags[-1], 200)
        plt.plot(tt, stretched(tt, c0, tau, beta), "-", color="crimson",
                 label=f"fit: tau={tau:.1f}s, beta={beta:.2f}, plateau={c0:.2f}")
    plt.xlabel("lag t (s)"); plt.ylabel("C(t) = <s(0)s(t)>")
    plt.title("spin (up/down) autocorrelation")
    plt.legend(); plt.ylim(0, 1.02)
    plt.savefig(os.path.join(OUTDIR, "spin_autocorrelation.png"), dpi=130,
                bbox_inches="tight")
    plt.close()

    # <N_f>: frustrated (same-state) bonds per particle, Delaunay neighbours
    tri = Delaunay(coords)
    eset = set()
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            eset.add((min(s[a], s[b]), max(s[a], s[b])))
    edges = np.array(sorted(eset))
    elen = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    edges = edges[elen < 1.4 * np.median(elen)]         # drop long boundary bonds

    same = (states[edges[:, 0]] == states[edges[:, 1]])  # (n_edges, n_frames)
    # frustrated (same-state) bonds per particle, averaged over frames
    frust_mean_per_edge = same.mean(axis=1)
    Nf_site = np.zeros(len(coords))
    for (i, j), fm in zip(edges, frust_mean_per_edge):
        Nf_site[i] += fm
        Nf_site[j] += fm
    Nf_mean = Nf_site.mean()

    plt.figure(figsize=(8, 8))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=Nf_site, s=10, cmap="magma")
    plt.gca().invert_yaxis()
    plt.colorbar(sc, label="frustrated bonds / particle")
    plt.title(f"frustration map  <N_f> = {Nf_mean:.2f}")
    plt.axis("equal"); plt.axis("off")
    plt.savefig(os.path.join(OUTDIR, "frustration_map.png"), dpi=130,
                bbox_inches="tight")
    plt.close()

    # ------------------------- summary -------------------------
    active = flip_rate > np.percentile(flip_rate, 75)
    log("\n================ SUMMARY ================")
    log(f"particles analysed         : {len(coords)}")
    log(f"frames / duration          : {states.shape[1]}  /  {states.shape[1]*dt:.1f}s")
    log(f"global up-fraction         : {global_up:.3f}   (0.5 = symmetric buckling)")
    log(f"mean flip rate             : {flip_rate.mean():.4f} flips/particle/s")
    log(f"  active band (top 25%)    : {flip_rate[active].mean():.4f} flips/particle/s")
    log(f"  frozen (bottom 25%)      : {flip_rate[flip_rate<np.percentile(flip_rate,25)].mean():.4f}")
    if np.isfinite(tau):
        log(f"C(t) stretched-exp fit     : tau={tau:.1f}s  beta={beta:.2f}  plateau c0={c0:.2f}")
        log(f"  (plateau c0 ~ frozen fraction; beta<1 => glassy/heterogeneous)")
    log(f"<N_f> frustrated bonds/part: {Nf_mean:.2f}   "
        f"(Han et al.: ~2 = frustrated ground state, ~3 = random)")
    log(f"lattice spacing (median)   : {np.median(elen)*px_um:.2f} um  "
        f"({np.median(elen):.1f} analysed-px)")
    log(f"\nFigures written to: {OUTDIR}")
    log("  detection_overlay.png  <-- CHECK THIS FIRST (are particles found correctly?)")
    log("  brightness_histogram.png  up_fraction_map.png  flip_activity_map.png")
    log("  spin_autocorrelation.png  frustration_map.png")

    # save raw per-particle table
    pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "up_fraction": up_frac_site, "flip_rate": flip_rate,
        "Nf": Nf_site,
    }).to_csv(os.path.join(OUTDIR, "per_particle.csv"), index=False)
    log("  per_particle.csv")


if __name__ == "__main__":
    main()
