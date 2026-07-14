#!/usr/bin/env python3
"""
colloid_tracking.py

Colloidal particle tracker with structural defect analysis.
Extends Owen Tower's PhD_Code/Colloid_Analysis.py (explicit permission granted).

Identifies and marks in the annotated output video:
  ● 5-fold particles   (blue circles)   ─┐ adjacent pair = edge dislocation
  ● 7-fold particles   (red circles)    ─┘ (magenta connecting line)
  ● Low-angle grain boundaries  (yellow line, misorientation < LAGB_ANGLE_DEG)
  ● High-angle grain boundaries (cyan line)

Algorithm:
  1. Chunked batch detection with trackpy + optional bandpass
  2. Adaptive linking (trackpy) with affine drift + cage-relative correction
  3. Per-frame Delaunay triangulation on ALL detections → coordination numbers
  4. ψ₆ hexatic order parameter per particle
  5. 5-7 pair identification (must be Delaunay neighbours within PAIR_DIST_PX)
  6. DBSCAN clustering of dislocation midpoints → grain boundary segments
  7. Misorientation angle from ψ₆ → LAGB vs HAGB classification
  8. Annotated MP4 output + CSV catalogues

Requirements:
    pip install trackpy pims imageio-ffmpeg scipy scikit-learn
                opencv-python-headless numpy pandas matplotlib

Usage:
    python colloid_tracking.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import trackpy as tp
import pims
from pims import pipeline
from pathlib import Path
from scipy.spatial import Delaunay, cKDTree
from sklearn.cluster import DBSCAN

tp.quiet()


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  —  edit these for your setup
# ═══════════════════════════════════════════════════════════════════════════════

VIDEO_PATH  = Path("Basler_a2A2600-64umBAS__40312155__20260609_101036115.mp4")
OUT_VIDEO   = Path("colloid_tracked.mp4")
OUT_TRACKS  = Path("tracks.csv")
OUT_DEFECTS = Path("defects.csv")

# Physical calibration
PX_UM = 0.11        # µm per pixel  (calibrate to your objective)
DT    = None        # s per frame;  None → read from video fps

# ── Detection ─────────────────────────────────────────────────────────────────
DIAMETER     = 19   # odd; approximate particle diameter [px]
SEPARATION   = 19   # minimum centre-to-centre distance [px]
MINMASS      = 3000 # minimum integrated brightness (raise to reject dim spots)
PERCENTILE   = 80
USE_BANDPASS = True
LSHORT       = 1
LLONG        = 53
INVERT       = False  # True if particles appear dark on bright background

# ── Linking ───────────────────────────────────────────────────────────────────
MAX_STEP_UM = 3.0   # maximum displacement between frames [µm]
MEMORY      = 3     # frames a particle may disappear and still be linked
MIN_LEN     = 15    # minimum trajectory length (points) for track output
EDGE_PX     = 2 * DIAMETER  # discard detections within this edge margin

# ── Frame window  (set END_FR = None to process all frames) ───────────────────
START_FR = 0
END_FR   = 500      # ~8 s at 65 fps; set None for the full ~27 k-frame video

# ── Defect detection ──────────────────────────────────────────────────────────
PAIR_DIST_PX   = int(2.5 * DIAMETER)  # max 5-7 centre distance to form a pair [px]
LAGB_EPS_PX    = int(4.0 * DIAMETER)  # DBSCAN ε for clustering dislocations [px]
LAGB_MIN_N     = 3                    # minimum dislocations per boundary segment
LAGB_ASPECT    = 2.0                  # PCA aspect-ratio threshold to call a cluster a boundary
LAGB_ANGLE_DEG = 15.0                 # misorientation < 15° → LAGB; ≥ 15° → HAGB

# ── Output ────────────────────────────────────────────────────────────────────
CHUNK      = 50     # frames per detection batch (tune for RAM)
SAVE_VIDEO = True
VIDEO_FPS  = None   # None → match input fps; or set e.g. 15.0 to slow down

# OpenCV colours (BGR)
CLR_6    = (160, 160, 160)  # 6-fold: grey
CLR_5    = (220,  60,  20)  # 5-fold: blue
CLR_7    = ( 20,  60, 220)  # 7-fold: red
CLR_PAIR = (220,  20, 220)  # 5-7 dislocation link: magenta
CLR_LAGB = (  0, 220, 220)  # low-angle GB: yellow
CLR_HAGB = (200, 180,   0)  # high-angle GB: cyan

RADIUS   = max(3, DIAMETER // 3)  # circle radius in rendered video [px]


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO I/O
# ═══════════════════════════════════════════════════════════════════════════════

def open_video(path: Path):
    """Try multiple pims backends; return (frames, dt_per_frame_s)."""
    path = Path(path)
    assert path.is_file(), f"Video not found: {path}"
    for opener in (
        lambda: pims.Video(str(path), plugin="ffmpeg"),
        lambda: pims.PyAVVideoReader(str(path)),
        lambda: pims.OpenCVVideoReader(str(path)),
        lambda: pims.Video(str(path)),
    ):
        try:
            v = opener()
            _ = v[0]
            fps = getattr(v, "frame_rate", None)
            dt  = (1.0 / float(fps)) if (fps and fps > 0) else 1.0
            return v, dt
        except Exception:
            pass
    raise RuntimeError(
        "Could not open video. Install imageio-ffmpeg or av.\n"
        "  pip install imageio-ffmpeg"
    )


@pipeline
def _to_gray(img):
    """Convert a video frame to float32 greyscale (applied lazily by pims)."""
    if getattr(img, "ndim", 2) == 3:
        img = img.mean(axis=2)
    if INVERT:
        img = img.max() - img
    return img.astype(np.float32)


def _bandpass(img):
    return tp.bandpass(img, lshort=LSHORT, llong=LLONG) if USE_BANDPASS else img


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION & LINKING
# ═══════════════════════════════════════════════════════════════════════════════

def detect_features(gray_frames, start: int, end: int, H: int, W: int) -> pd.DataFrame:
    """
    Batch-detect particles over frames [start, end] using trackpy.
    Returns a DataFrame with columns: frame, x_px, y_px, x_um, y_um, mass, size, ecc.
    """
    diam = int(DIAMETER) | 1  # must be odd
    chunks = []

    for a in range(start, end + 1, CHUNK):
        b   = min(a + CHUNK - 1, end)
        imgs = [_bandpass(gray_frames[i]) for i in range(a, b + 1)]
        kw  = dict(diameter=diam, separation=int(SEPARATION),
                   minmass=float(MINMASS), percentile=int(PERCENTILE),
                   preprocess=False, invert=bool(INVERT), processes=1)
        try:
            chunk = tp.batch(imgs, engine="numba", **kw)
        except Exception:
            chunk = tp.batch(imgs, engine="python", **kw)

        if chunk is None or chunk.empty:
            continue
        chunk["frame"] += a
        chunks.append(chunk)

    if not chunks:
        raise RuntimeError(
            "No particles detected. Adjust DIAMETER / MINMASS / PERCENTILE."
        )

    f = pd.concat(chunks, ignore_index=True)

    # discard edge detections
    e  = EDGE_PX
    ok = (f.x >= e) & (f.x <= W - 1 - e) & (f.y >= e) & (f.y <= H - 1 - e)
    f  = f.loc[ok].rename(columns={"x": "x_px", "y": "y_px"}).copy()
    f["x_um"] = f["x_px"] * PX_UM
    f["y_um"] = f["y_px"] * PX_UM
    return f.reset_index(drop=True)


def link_tracks(feats: pd.DataFrame, dt_eff: float) -> pd.DataFrame:
    """Link detections into trajectories with adaptive search range."""
    search_px = MAX_STEP_UM / PX_UM
    linked = tp.link_df(
        feats[["frame", "x_px", "y_px", "x_um", "y_um"]].copy(),
        search_range=search_px,
        memory=MEMORY,
        pos_columns=["x_px", "y_px"],
        t_column="frame",
        adaptive_stop=2.5 / PX_UM,
        adaptive_step=0.9,
    )
    linked["t"] = (linked["frame"] - linked["frame"].min()) * dt_eff
    keep = linked.groupby("particle").filter(lambda g: len(g) >= MIN_LEN)
    return keep.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DRIFT CORRECTION  (adapted from Owen Tower's PhD_Code/Colloid_Analysis.py)
# ═══════════════════════════════════════════════════════════════════════════════

def affine_flow_correct(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Frame-to-frame affine flow fit  (ux = a₀ + a₁x + a₂y, etc.).
    Adds columns x_um_aff, y_um_aff.
    """
    df = tracks.sort_values(["frame", "particle"]).copy()
    df["x_um_aff"] = df["x_um"].values
    df["y_um_aff"] = df["y_um"].values

    frs = np.sort(df["frame"].unique())
    for fr, frn in zip(frs[:-1], frs[1:]):
        g0 = df[df.frame == fr][["particle", "x_um", "y_um", "t"]]
        g1 = df[df.frame == frn][["particle", "x_um", "y_um", "t"]]
        m  = g0.merge(g1, on="particle", suffixes=("", "_n"))
        if len(m) < 20:
            continue
        dt = float(np.median(m.t_n - m.t))
        if not (np.isfinite(dt) and dt > 0):
            continue

        X = np.c_[m.x_um, m.y_um, np.ones(len(m))]
        a = np.linalg.lstsq(X, (m.x_um_n - m.x_um) / dt, rcond=None)[0]
        b = np.linalg.lstsq(X, (m.y_um_n - m.y_um) / dt, rcond=None)[0]

        sel = df.frame == frn
        X1  = np.c_[df.loc[sel, "x_um"], df.loc[sel, "y_um"], np.ones(sel.sum())]
        df.loc[sel, "x_um_aff"] = df.loc[sel, "x_um"] - (X1 @ a) * dt
        df.loc[sel, "y_um_aff"] = df.loc[sel, "y_um"] - (X1 @ b) * dt
        df.loc[sel, "x_um"]     = df.loc[sel, "x_um_aff"]
        df.loc[sel, "y_um"]     = df.loc[sel, "y_um_aff"]

    return df


def cage_relative(tracks: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """
    Subtract mean position of k nearest neighbours per frame (cage correction).
    Adds columns x_um_cr, y_um_cr.
    """
    out = []
    for fr, g in tracks.groupby("frame"):
        P  = g[["x_um_aff", "y_um_aff"]].values
        kk = min(k + 1, len(P))
        gg = g.copy()
        if kk < 2:
            gg["x_um_cr"] = P[:, 0]
            gg["y_um_cr"] = P[:, 1]
        else:
            _, idx = cKDTree(P).query(P, k=kk)
            Q = P - P[idx[:, 1:]].mean(axis=1)
            gg["x_um_cr"] = Q[:, 0]
            gg["y_um_cr"] = Q[:, 1]
        out.append(gg)
    return pd.concat(out, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CRYSTAL STRUCTURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def delaunay_analysis(pts_px: np.ndarray):
    """
    Delaunay triangulation of particle positions.

    Parameters
    ----------
    pts_px : (N, 2) float array  [x_px, y_px]

    Returns
    -------
    coord  : (N,) int    – coordination number (Delaunay neighbour count)
    nbrs   : list[set]   – neighbour indices per particle
    psi6   : (N,) complex – hexatic bond-orientation order parameter ψ₆
    """
    N = len(pts_px)
    if N < 4:
        return np.full(N, 6, int), [set()] * N, np.zeros(N, complex)

    tri  = Delaunay(pts_px)
    nbrs = [set() for _ in range(N)]
    for a, b, c in tri.simplices:
        nbrs[a].update({b, c})
        nbrs[b].update({a, c})
        nbrs[c].update({a, b})

    coord = np.array([len(s) for s in nbrs], dtype=int)

    # ψ₆ = ⟨exp(6iθ)⟩ over Delaunay bonds
    psi6 = np.zeros(N, dtype=complex)
    for i, ns in enumerate(nbrs):
        if ns:
            angles = [
                np.arctan2(pts_px[j, 1] - pts_px[i, 1],
                           pts_px[j, 0] - pts_px[i, 0])
                for j in ns
            ]
            psi6[i] = np.mean(np.exp(6j * np.array(angles)))

    return coord, nbrs, psi6


def find_57_pairs(pts_px: np.ndarray, coord: np.ndarray, nbrs: list) -> list:
    """
    Find 5-7 pairs: a 5-fold and 7-fold particle that are Delaunay neighbours
    AND within PAIR_DIST_PX of each other.

    Each such pair is an edge dislocation in the 2D hexagonal crystal.

    Returns list of (i5, i7) index tuples.
    """
    pairs = []
    for i in range(len(pts_px)):
        if coord[i] != 5:
            continue
        for j in nbrs[i]:
            if coord[j] == 7:
                dist = np.hypot(pts_px[j, 0] - pts_px[i, 0],
                                pts_px[j, 1] - pts_px[i, 1])
                if dist <= PAIR_DIST_PX:
                    pairs.append((i, j))
    return pairs


def _misorientation_6fold_deg(psi6_i: complex, psi6_j: complex) -> float:
    """
    Misorientation angle between two hexagonal crystal grains in degrees.

    Δθ = |arg(ψ₆ᵢ · ψ₆ⱼ*)| / 6,  range [0°, 30°] (fundamental zone of 6mm).
    Returns NaN if either ψ₆ magnitude is below 0.1 (disordered region).
    """
    if abs(psi6_i) < 0.1 or abs(psi6_j) < 0.1:
        return float("nan")
    return float(np.degrees(abs(np.angle(psi6_i * np.conj(psi6_j))) / 6))


def find_grain_boundaries(
    pts_px: np.ndarray,
    pairs57: list,
    psi6: np.ndarray,
) -> list:
    """
    Cluster 5-7 dislocation midpoints with DBSCAN.
    Classify each elongated cluster as LAGB or HAGB based on
    the mean ψ₆ misorientation across its constituent pairs.

    Returns a list of boundary dicts with keys:
        kind                – "LAGB" or "HAGB"
        p1, p2              – end-points of the principal axis [px]
        n                   – number of dislocations in the segment
        aspect              – PCA aspect ratio (larger = more linear)
        misorientation_deg  – mean misorientation across the cluster
    """
    if len(pairs57) < LAGB_MIN_N:
        return []

    mids   = np.array([0.5 * (pts_px[i] + pts_px[j]) for i, j in pairs57])
    labels = DBSCAN(
        eps=float(LAGB_EPS_PX), min_samples=LAGB_MIN_N
    ).fit_predict(mids)

    boundaries = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        mask    = labels == lbl
        cluster = mids[mask]
        if len(cluster) < LAGB_MIN_N:
            continue

        # PCA to test linearity
        c0   = cluster - cluster.mean(0)
        cov  = c0.T @ c0
        eigs, vecs = np.linalg.eigh(cov)           # ascending order
        aspect = float(np.sqrt(eigs[-1] / eigs[-2])) if eigs[-2] > 1e-9 else 999.0
        if aspect < LAGB_ASPECT:
            continue

        # principal-axis end-points for drawing
        axis = vecs[:, -1]                          # eigenvector of largest eigenvalue
        proj = c0 @ axis
        ctr  = cluster.mean(0)
        p1, p2 = ctr + proj.min() * axis, ctr + proj.max() * axis

        # mean misorientation across pairs in this cluster
        pair_indices = np.where(mask)[0]
        mis_vals = []
        for k in pair_indices:
            i5, i7 = pairs57[k]
            m = _misorientation_6fold_deg(psi6[i5], psi6[i7])
            if np.isfinite(m):
                mis_vals.append(m)
        mean_mis = float(np.mean(mis_vals)) if mis_vals else 0.0

        kind = "LAGB" if mean_mis < LAGB_ANGLE_DEG else "HAGB"
        boundaries.append({
            "kind": kind,
            "p1":   p1,
            "p2":   p2,
            "n":    int(len(cluster)),
            "aspect": aspect,
            "misorientation_deg": mean_mis,
        })

    return boundaries


def analyze_frame(pts_px: np.ndarray):
    """
    Full per-frame crystal structure analysis.

    Returns (coord, psi6, pairs57, boundaries)
    """
    coord, nbrs, psi6 = delaunay_analysis(pts_px)
    pairs57            = find_57_pairs(pts_px, coord, nbrs)
    boundaries         = find_grain_boundaries(pts_px, pairs57, psi6)
    return coord, psi6, pairs57, boundaries


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

def render_overlay(
    frame_bgr: np.ndarray,
    pts_px:    np.ndarray,
    coord:     np.ndarray,
    pairs57:   list,
    boundaries: list,
) -> np.ndarray:
    """Draw all overlays onto a copy of frame_bgr and return it."""
    out = frame_bgr.copy()

    # ── particle circles (coloured by coordination) ──────────────────────────
    for i in range(len(pts_px)):
        cx, cy = int(round(pts_px[i, 0])), int(round(pts_px[i, 1]))
        c = int(coord[i]) if i < len(coord) else 6
        if c == 5:
            clr, thick = CLR_5, 2
        elif c == 7:
            clr, thick = CLR_7, 2
        else:
            clr, thick = CLR_6, 1
        cv2.circle(out, (cx, cy), RADIUS, clr, thick, lineType=cv2.LINE_AA)

    # ── 5-7 pair links (dislocations) ────────────────────────────────────────
    for i5, i7 in pairs57:
        p1 = (int(round(pts_px[i5, 0])), int(round(pts_px[i5, 1])))
        p2 = (int(round(pts_px[i7, 0])), int(round(pts_px[i7, 1])))
        cv2.line(out, p1, p2, CLR_PAIR, 2, cv2.LINE_AA)

    # ── grain boundary lines ──────────────────────────────────────────────────
    for gb in boundaries:
        clr = CLR_LAGB if gb["kind"] == "LAGB" else CLR_HAGB
        q1  = (int(round(gb["p1"][0])), int(round(gb["p1"][1])))
        q2  = (int(round(gb["p2"][0])), int(round(gb["p2"][1])))
        cv2.line(out, q1, q2, clr, 3, cv2.LINE_AA)
        # misorientation label near the midpoint
        mx  = int(round(0.5 * (gb["p1"][0] + gb["p2"][0])))
        my  = int(round(0.5 * (gb["p1"][1] + gb["p2"][1])))
        cv2.putText(out,
                    f"{gb['misorientation_deg']:.1f}°",
                    (mx + 4, my - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1, cv2.LINE_AA)

    return out


def draw_legend(img: np.ndarray) -> np.ndarray:
    items = [
        (CLR_5,    "5-fold particle"),
        (CLR_7,    "7-fold particle"),
        (CLR_PAIR, "5-7 dislocation"),
        (CLR_LAGB, f"Low-angle GB  (<{LAGB_ANGLE_DEG:.0f}°)"),
        (CLR_HAGB, f"High-angle GB (≥{LAGB_ANGLE_DEG:.0f}°)"),
    ]
    for k, (clr, txt) in enumerate(items):
        y = 12 + k * 18
        cv2.circle(img, (12, y + 4), 4, clr, -1)
        cv2.putText(img, txt, (22, y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1, cv2.LINE_AA)
    return img


def draw_hud(img: np.ndarray, fr: int, n_part: int,
             n_57: int, n_gb: int, W: int, H: int) -> np.ndarray:
    info = (f"fr {fr}  |  n={n_part}  |  "
            f"57-pairs={n_57}  |  GBs={n_gb}")
    cv2.putText(img, info, (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Colloid Tracking + Defect Analysis")
    print("=" * 60)

    # ── 1. Load video ─────────────────────────────────────────────────────────
    print(f"\n[1] Opening {VIDEO_PATH} …")
    vid, dt_eff = open_video(VIDEO_PATH)
    if DT is not None:
        dt_eff = float(DT)

    gray    = _to_gray(vid)
    N_total = len(gray)
    H, W    = gray[0].shape

    s = int(START_FR)
    e = N_total - 1 if END_FR is None else min(int(END_FR), N_total - 1)
    print(f"    {N_total} frames  |  analysing [{s}, {e}]  |  "
          f"{W}×{H} px  |  dt = {dt_eff:.5f} s  "
          f"({1/dt_eff:.1f} fps)")

    # ── 2. Detect ─────────────────────────────────────────────────────────────
    print(f"\n[2] Detecting (diam={DIAMETER} px, minmass={MINMASS}) …")
    feats = detect_features(gray, s, e, H, W)
    print(f"    {len(feats):,} detections  |  "
          f"{feats.frame.nunique()} frames with hits  |  "
          f"~{len(feats)/max(1,feats.frame.nunique()):.0f} per frame")

    # ── 3. Link ───────────────────────────────────────────────────────────────
    print(f"\n[3] Linking (max_step={MAX_STEP_UM} µm, memory={MEMORY}) …")
    tracks = link_tracks(feats, dt_eff)
    print(f"    {tracks.particle.nunique()} tracks  (len ≥ {MIN_LEN})")

    # ── 4. Drift correction ───────────────────────────────────────────────────
    print("\n[4] Affine drift + cage-relative correction …")
    tracks = affine_flow_correct(tracks)
    tracks = cage_relative(tracks)
    tracks.to_csv(OUT_TRACKS, index=False)
    print(f"    Saved tracks → {OUT_TRACKS}")

    # ── 5. Per-frame defect analysis + annotated video ────────────────────────
    print(f"\n[5] Defect analysis + rendering → {OUT_VIDEO}")

    fps_out = VIDEO_FPS
    if fps_out is None:
        raw_fps = getattr(vid, "frame_rate", None)
        fps_out = float(raw_fps) if (raw_fps and raw_fps > 0) else (1.0 / dt_eff)

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, fps_out, (W, H))

    defect_rows = []
    n_frames_done = 0

    for fr in range(s, e + 1):
        # raw frame → 8-bit BGR
        raw   = gray[fr]
        u8    = np.clip(raw / (raw.max() + 1e-9) * 255, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)

        # ALL detections in this frame for structural analysis
        sub = feats[feats.frame == fr]
        if len(sub) < 4:
            if writer:
                draw_legend(frame)
                draw_hud(frame, fr, 0, 0, 0, W, H)
                writer.write(frame)
            n_frames_done += 1
            continue

        pts    = sub[["x_px", "y_px"]].values.astype(float)
        coord, psi6, pairs57, boundaries = analyze_frame(pts)

        # collect defect rows
        for i5, i7 in pairs57:
            mx = 0.5 * (pts[i5, 0] + pts[i7, 0])
            my = 0.5 * (pts[i5, 1] + pts[i7, 1])
            mis = _misorientation_6fold_deg(psi6[i5], psi6[i7])
            defect_rows.append({
                "frame": fr,
                "type": "dislocation_57",
                "x_px": float(mx),       "y_px": float(my),
                "x_um": float(mx * PX_UM), "y_um": float(my * PX_UM),
                "misorientation_deg": mis,
            })
        for gb in boundaries:
            mx = 0.5 * (gb["p1"][0] + gb["p2"][0])
            my = 0.5 * (gb["p1"][1] + gb["p2"][1])
            defect_rows.append({
                "frame": fr,
                "type":  gb["kind"],
                "x_px":  float(mx),        "y_px": float(my),
                "x_um":  float(mx * PX_UM), "y_um": float(my * PX_UM),
                "n_dislocations": gb["n"],
                "aspect": gb["aspect"],
                "misorientation_deg": gb["misorientation_deg"],
            })

        # render
        if writer:
            ann = render_overlay(frame, pts, coord, pairs57, boundaries)
            draw_legend(ann)
            draw_hud(ann, fr, len(pts), len(pairs57), len(boundaries), W, H)
            writer.write(ann)

        n_frames_done += 1
        if n_frames_done % 50 == 0 or fr == s:
            pct = 100.0 * (fr - s) / max(1, e - s)
            gb_l = sum(1 for g in boundaries if g["kind"] == "LAGB")
            gb_h = sum(1 for g in boundaries if g["kind"] == "HAGB")
            print(f"    fr {fr:5d}  ({pct:5.1f}%)  "
                  f"n={len(pts):4d}  "
                  f"57-pairs={len(pairs57):3d}  "
                  f"LAGB={gb_l}  HAGB={gb_h}")

    if writer:
        writer.release()
        print(f"    Video saved → {OUT_VIDEO}")

    # ── 6. Save defects ───────────────────────────────────────────────────────
    defects_df = pd.DataFrame(defect_rows)
    defects_df.to_csv(OUT_DEFECTS, index=False)
    print(f"\n[6] Saved defects → {OUT_DEFECTS}  ({len(defects_df):,} rows)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Frames analysed:      {n_frames_done}")
    print(f"  Tracks (len≥{MIN_LEN}):     {tracks.particle.nunique()}")

    if not defects_df.empty:
        d57  = defects_df[defects_df.type == "dislocation_57"]
        lagb = defects_df[defects_df.type == "LAGB"]
        hagb = defects_df[defects_df.type == "HAGB"]
        print(f"  5-7 disloc. events:   {len(d57):,}")
        if len(d57):
            print(f"  Mean 57-pairs/frame:  "
                  f"{d57.groupby('frame').size().mean():.1f}")
            valid_mis = d57.misorientation_deg.dropna()
            if len(valid_mis):
                print(f"  Median misorientation:{valid_mis.median():.1f}°")
        print(f"  LAGB events:          {len(lagb):,}")
        print(f"  HAGB events:          {len(hagb):,}")

    print(f"\nOutputs:")
    print(f"  {OUT_TRACKS.resolve()}")
    print(f"  {OUT_DEFECTS.resolve()}")
    if SAVE_VIDEO:
        print(f"  {OUT_VIDEO.resolve()}")
    print("Done.\n")


if __name__ == "__main__":
    main()
