"""
bond_asymmetry.py
=================
Single-frame test for BUCKLING using the bond-length asymmetry of
Han et al., Nature 456, 898 (2008).

Physics: in a buckled monolayer, an up-down (satisfied) neighbour pair
projects to a shorter in-plane distance, sqrt(d^2-(h-d)^2), than a
same-state (up-up / down-down, "frustrated") pair, which stays ~d. So:

    mean(bright<->dark spacing)  <  mean(same-type spacing)   => BUCKLING
    (a few % shorter; ratio 0.87 at h/d=1.5)

A second layer sitting in the lattice hollows would instead put the
bright particles at ~0.58 d from neighbours (a huge, not few-%, offset),
so this test also separates "buckle" from "second layer".

Usage:  python bond_asymmetry.py image.png [diameter_px]
"""
import sys, os
import numpy as np
import cv2
import trackpy as tp
from scipy.spatial import Delaunay, cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG   = sys.argv[1] if len(sys.argv) > 1 else "active_full.png"
DIAM  = int(sys.argv[2]) if len(sys.argv) > 2 else 45      # odd
SEP   = 52
MINMASS_PCT = 30
UM_PER_PIXEL = 3.0 / 59.0
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "buckling_out")


def odd(n):
    n = int(round(n)); return n + (n % 2 == 0)


def detect(img, diam, sep):
    pts = []
    for inv in (False, True):
        f = tp.locate(img, diam, invert=inv, separation=sep, engine="auto")
        if len(f):
            f = f[f["mass"] > np.percentile(f["mass"], MINMASS_PCT)]
            pts.append(f[["x", "y"]].values)
    pts = np.vstack(pts)
    keep = np.ones(len(pts), bool)
    for i, j in sorted(cKDTree(pts).query_pairs(r=sep * 0.6)):
        if keep[i] and keep[j]:
            keep[j] = False
    return pts[keep]


def sample(img, coords, r=3):
    ys, xs = np.mgrid[-r:r+1, -r:r+1]; m = xs**2 + ys**2 <= r**2
    H, W = img.shape
    X = np.clip(coords[:, 0][:, None] + xs[m][None], 0, W-1).astype(int)
    Y = np.clip(coords[:, 1][:, None] + ys[m][None], 0, H-1).astype(int)
    return img[Y, X].mean(1)


def otsu(v, nb=256):
    h, e = np.histogram(v, nb); c = .5*(e[:-1]+e[1:])
    w = np.cumsum(h); mean = np.cumsum(h*c); mt = mean[-1]
    wb = w/w[-1]
    mb = mean/np.maximum(w, 1); mf = (mt-mean)/np.maximum(w[-1]-w, 1)
    return c[np.argmax(wb*(1-wb)*(mb-mf)**2)]


def load_img(path):
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        a = tifffile.imread(path)
    else:
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    a = np.asarray(a).astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    return a


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    img = load_img(IMG)
    diam = odd(DIAM)
    print(f"image {img.shape}  diameter={diam}px")

    coords = detect(img, diam, SEP)
    inten = sample(img, coords)
    thr = otsu(inten)
    up = inten > thr                       # True = bright
    print(f"particles={len(coords)}  bright(up)-fraction={up.mean():.3f}  thr={thr:.0f}")

    # Delaunay neighbour bonds, drop long boundary edges
    tri = Delaunay(coords)
    es = set()
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            es.add((min(s[a], s[b]), max(s[a], s[b])))
    E = np.array(sorted(es))
    L = np.linalg.norm(coords[E[:, 0]] - coords[E[:, 1]], axis=1)
    good = L < 1.4 * np.median(L)
    E, L = E[good], L[good]

    mixed = up[E[:, 0]] != up[E[:, 1]]      # up-down = satisfied
    same = ~mixed                            # up-up / down-down = frustrated
    Lm, Ls = L[mixed], L[same]
    Lm_um, Ls_um = Lm*UM_PER_PIXEL, Ls*UM_PER_PIXEL

    pct = (Ls.mean() - Lm.mean()) / Ls.mean() * 100.0
    print("\n---------------- BOND-LENGTH ASYMMETRY ----------------")
    print(f"up-down (satisfied) bonds : n={mixed.sum():5d}  mean={Lm_um.mean():.3f} um")
    print(f"same-type (frustrated)    : n={same.sum():5d}  mean={Ls_um.mean():.3f} um")
    print(f"satisfied shorter by      : {pct:+.1f}%   "
          f"(Han et al. ~3-4%; buckle predicts a few % SHORTER)")
    # infer h/d from the ratio if it looks like a buckle: ratio = sqrt(1-((h-d)/d)^2)
    ratio = Lm.mean()/Ls.mean()
    if 0 < ratio < 1:
        hd = 1 + np.sqrt(max(0.0, 1 - ratio**2))
        print(f"implied h/d (if buckling) : {hd:.2f}  (target 1.3-1.6)")
    # verdict
    if pct >= 1.5 and pct < 25:
        print("VERDICT: consistent with BUCKLING (satisfied bonds a few % shorter).")
    elif pct >= 25:
        print("VERDICT: shortening too large -> looks like INTERSTITIAL 2nd layer, not a buckle.")
    else:
        print("VERDICT: no significant asymmetry -> NOT clearly buckling (contrast/disorder).")

    # figures
    plt.figure(figsize=(7, 4))
    bins = np.linspace(min(Lm_um.min(), Ls_um.min()), np.percentile(np.r_[Lm_um, Ls_um], 99), 60)
    plt.hist(Ls_um, bins, alpha=.6, color="#c0392b", label=f"same-type (frustrated)  {Ls_um.mean():.3f}um")
    plt.hist(Lm_um, bins, alpha=.6, color="#1d9e75", label=f"up-down (satisfied)  {Lm_um.mean():.3f}um")
    plt.axvline(Ls_um.mean(), color="#c0392b", ls="--"); plt.axvline(Lm_um.mean(), color="#1d9e75", ls="--")
    plt.xlabel("bond length (um)"); plt.ylabel("count")
    plt.title(f"bond-length asymmetry: satisfied {pct:+.1f}% shorter")
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(OUTDIR, "bond_asymmetry_hist.png"), dpi=130, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 9))
    plt.imshow(img, cmap="gray")
    for (i, j), mx in zip(E, mixed):
        plt.plot(coords[[i, j], 0], coords[[i, j], 1],
                 color=("#1d9e75" if mx else "#c0392b"), lw=0.5, alpha=.7)
    plt.scatter(coords[up, 0], coords[up, 1], s=8, c="yellow", label="up (bright)")
    plt.scatter(coords[~up, 0], coords[~up, 1], s=8, c="cyan", label="down (dark)")
    plt.legend(loc="upper right"); plt.axis("off")
    plt.title("green=satisfied(up-down)  red=frustrated(same)")
    plt.savefig(os.path.join(OUTDIR, "bond_map.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nfigures: {OUTDIR}\\bond_asymmetry_hist.png , bond_map.png")


if __name__ == "__main__":
    main()
