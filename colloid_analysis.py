#!/usr/bin/env python3
"""colloid_analysis.py — shared, Qt-free structural & dynamical analysis.

Single source of truth for order parameters and diffusion, used by BOTH:
  * colloid_app.py (the PyQt6 tracking GUI) — imports these functions instead of
    defining them, so there is exactly one implementation, and
  * the TempSim simulation — imports the same code so simulated crystals and real
    microscope data are analysed identically.

This module has **zero PyQt / cv2 / trackpy dependencies** so it imports on headless
Oscar compute nodes. It depends only on numpy, scipy, pandas, the stdlib, and the
optional compiled `colloid_kernels` (with a pure-Python fallback).

STRUCTURAL functions were extracted verbatim from colloid_app.py (Delaunay ψ6,
5-7 dislocation pairs, LAGB/HAGB grain boundaries, g(r)/g6(r), defect series, affine
drift correction, cage-relative coordinates). The DYNAMICAL layer (msd / diffusion_D /
stokes_einstein / effective_temperature) is new — the GUI app never computed diffusion.

Units convention: the simulation exports a tracks DataFrame with the same columns the
app uses (particle, frame, x_px, y_px, x_um, y_um, t); px_um and dt_eff are repurposed
as the reduced-unit -> experiment-unit map so this code runs unchanged on both.
"""

from __future__ import annotations

import os
import concurrent.futures as _cf
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree, QhullError
from scipy.sparse import csr_matrix as _csr
from scipy.sparse.csgraph import connected_components as _cc

# Optional compiled kernel for ψ6 accumulation; pure-Python fallback otherwise.
try:
    from colloid_kernels import psi6_accumulate as _psi6_accumulate
except Exception:
    def _psi6_accumulate(i_all, j_all, dx, dy, N):
        """Pure-Python fallback when colloid_kernels is unavailable."""
        psi6 = np.zeros(N, dtype=complex)
        np.add.at(psi6, i_all, np.exp(6j * np.arctan2(dy, dx)))
        coord = np.bincount(i_all, minlength=N)
        return psi6, coord


# ════════════════════════════════════════════════════════════════════
# Result container
# ════════════════════════════════════════════════════════════════════

@dataclass
class FrameResult:
    pts_px: np.ndarray; coord: np.ndarray; psi6: np.ndarray
    pairs57: list; boundaries: list
    edges: np.ndarray = None   # unique undirected edge pairs (E,2) from Delaunay
    _dp: dict = None           # grain-boundary params; None until first LAGB/HAGB render


# ════════════════════════════════════════════════════════════════════
# Structural analysis  (extracted verbatim from colloid_app.py)
# ════════════════════════════════════════════════════════════════════

def _delaunay(pts, max_dist_px: "float | None" = None):
    """Compute Delaunay coordination, ψ₆, and unique edge array.

    Fully vectorised: psi6 via numpy arctan2/exp/add.at, edges via numpy
    sort+unique. No per-particle Python loop — _57_pairs and the grain-overlay
    rendering consume the edge array directly rather than neighbour sets.

    max_dist_px:
        Optional maximum edge length (px). Delaunay triangulation is known to
        produce long, physically-meaningless edges at the convex-hull boundary
        or in sparse/dilute regions — connecting particles that aren't really
        neighbours. When set (> 0), edges longer than this are dropped before
        ψ₆/coordination-number accumulation, so downstream defect/grain-
        boundary logic (which consumes this same edge array) never sees them
        either. None/0 (default) disables the filter — legacy behaviour.
    """
    N = len(pts)
    _empty = (np.full(N, 6, int), np.zeros(N, complex), np.zeros((0, 2), int))
    if N < 4:
        return _empty

    try:
        tri = Delaunay(pts)
    except QhullError:
        # Degenerate input (collinear / coincident points) — qhull can't
        # form a simplex. No well-defined triangulation, so report no
        # structure rather than crashing the analysis/live-overlay thread.
        return _empty
    s   = tri.simplices                        # (M, 3)

    # Unique undirected edges — encode each edge (i<j) as scalar i*N+j and
    # dedup with a flat 1-D sort: identical output to np.unique(rows, axis=0)
    # (col0 < col1 < N makes scalar-key order == lexicographic row order) but
    # 7-9x faster on thousands of particles.
    s64 = s.astype(np.int64)                   # int32*N would overflow
    lo  = np.concatenate([np.minimum(s64[:, 0], s64[:, 1]),
                          np.minimum(s64[:, 1], s64[:, 2]),
                          np.minimum(s64[:, 0], s64[:, 2])])
    hi  = np.concatenate([np.maximum(s64[:, 0], s64[:, 1]),
                          np.maximum(s64[:, 1], s64[:, 2]),
                          np.maximum(s64[:, 0], s64[:, 2])])
    uk  = np.unique(lo * N + hi)
    uniq = np.empty((len(uk), 2), dtype=s.dtype)   # preserve simplices dtype
    uniq[:, 0], uniq[:, 1] = uk // N, uk % N       # (E, 2)
    ui, uj = uniq[:, 0], uniq[:, 1]

    if max_dist_px is not None and max_dist_px > 0 and len(uniq):
        elen = np.hypot(pts[uj, 0] - pts[ui, 0], pts[uj, 1] - pts[ui, 1])
        short = elen <= float(max_dist_px)
        if not short.all():
            uniq = uniq[short]
            ui, uj = uniq[:, 0], uniq[:, 1]

    # Directed edges both ways for coordination count and ψ₆ accumulation
    i_all = np.concatenate([ui, uj])
    j_all = np.concatenate([uj, ui])

    dx    = pts[j_all, 0] - pts[i_all, 0]
    dy    = pts[j_all, 1] - pts[i_all, 1]
    psi6, coord = _psi6_accumulate(i_all, j_all, dx, dy, N)
    psi6 /= np.maximum(coord, 1)

    return coord, psi6, uniq


def _57_pairs(pts, coord, edges, dist):
    """Find Delaunay-neighbour 5-7 dislocation pairs.

    Vectorised over the unique edge array instead of nested Python loops.
    """
    if edges is None or len(edges) == 0:
        return []
    ui, uj = edges[:, 0], edges[:, 1]
    ci, cj = coord[ui], coord[uj]

    # Edges with one 5-fold and one 7-fold endpoint
    mask57 = ((ci == 5) & (cj == 7))
    mask75 = ((ci == 7) & (cj == 5))

    i5 = np.concatenate([ui[mask57], uj[mask75]])
    i7 = np.concatenate([uj[mask57], ui[mask75]])

    if len(i5) == 0:
        return []

    ok = np.hypot(pts[i7, 0] - pts[i5, 0], pts[i7, 1] - pts[i5, 1]) <= dist
    return list(zip(i5[ok].tolist(), i7[ok].tolist()))


def _dbscan_via_kdtree(X, eps, min_samples):
    """DBSCAN-equivalent clustering built from cKDTree.query_pairs +
    connected_components — both scipy C-extensions that release the GIL
    cleanly, unlike sklearn's DBSCAN (measured at only 1.20x speedup across
    8 threads vs. 4.68x for Delaunay/QHull, capping the structural-analysis
    thread pool's real-world efficiency at ~2.5x). Reuses the _csr/_cc
    imports already used by _draw_grain_overlay's grain-adjacency clustering.

    Replicates sklearn's exact DBSCAN semantics (not a naive "connect
    everything within eps" simplification):
      - A point is a *core point* if it has >= min_samples neighbours
        within eps (including itself, matching sklearn's convention).
      - Clusters are formed by connecting CORE points that are mutually
        within eps of each other (core-to-core edges only).
      - Non-core points that are within eps of at least one core point are
        *border points*: assigned to that core point's cluster (sklearn
        assigns a border point to the first core neighbour's cluster found;
        since all core points in one connected blob share the same cluster
        id after connected_components, any of its core neighbours gives the
        same answer).
      - Points that are neither core nor within eps of any core point are
        noise (label -1).

    Returns an int label array, same contract as DBSCAN(...).fit_predict:
    non-negative cluster ids, -1 for noise.
    """
    n = len(X)
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels

    tree = cKDTree(X)
    pairs = tree.query_pairs(r=float(eps), output_type="ndarray")  # (P, 2), i<j

    # Neighbour counts within eps, including self (sklearn's min_samples
    # convention counts the point itself as one of its own neighbours).
    neighbor_count = np.ones(n, dtype=np.int64)
    if len(pairs):
        np.add.at(neighbor_count, pairs[:, 0], 1)
        np.add.at(neighbor_count, pairs[:, 1], 1)
    is_core = neighbor_count >= min_samples

    if not is_core.any():
        return labels  # everything is noise

    if len(pairs):
        # Core-to-core edges only decide cluster connectivity.
        core_pair_mask = is_core[pairs[:, 0]] & is_core[pairs[:, 1]]
        core_rows = pairs[core_pair_mask, 0]
        core_cols = pairs[core_pair_mask, 1]
    else:
        core_rows = core_cols = np.zeros(0, dtype=np.int64)

    adj = _csr((np.ones(len(core_rows)), (core_rows, core_cols)), shape=(n, n))
    n_comp, comp_labels = _cc(adj, directed=False)

    core_idx = np.nonzero(is_core)[0]
    core_comp_ids = comp_labels[core_idx]
    # Relabel to a dense 0..K-1 range (every core point's connected_components
    # id qualifies — isolated core points form their own singleton component).
    # `dense` from np.unique(..., return_inverse=True) is already the
    # per-element (same length as core_comp_ids) mapping into 0..K-1 — no
    # separate dict lookup needed/correct here.
    _uniq_comp, dense = np.unique(core_comp_ids, return_inverse=True)
    dense = np.asarray(dense).reshape(-1)  # numpy>=2.0 may return column-shape

    labels[core_idx] = dense

    # Border points: non-core points within eps of >=1 core point get that
    # core point's cluster id. When a border point is adjacent to core
    # points from *different* clusters, sklearn's DBSCAN resolves this
    # ambiguity deterministically: dbscan_inner.pyx scans core points in
    # ascending index order and DFS-labels each new cluster with the next
    # sequential label_num the first time an unvisited core point is
    # reached, so a shared border point ends up with whichever cluster's
    # *lowest-index core point* is scanned first. Since connected-components
    # cluster identity doesn't depend on traversal order, that "first
    # discovered" cluster is exactly the one containing the smallest core
    # point index overall — replicate that tie-break here (not just
    # "first listed neighbour pair", which does not reliably match sklearn).
    if len(pairs):
        noncore_core_mask_a = (~is_core[pairs[:, 0]]) & is_core[pairs[:, 1]]
        noncore_core_mask_b = is_core[pairs[:, 0]] & (~is_core[pairs[:, 1]])
        border_pt = np.concatenate([pairs[noncore_core_mask_a, 0],
                                     pairs[noncore_core_mask_b, 1]])
        border_core = np.concatenate([pairs[noncore_core_mask_a, 1],
                                       pairs[noncore_core_mask_b, 0]])
        if len(border_pt):
            border_cluster = labels[border_core]
            # Rank clusters by the smallest core-point index they contain —
            # this is the order sklearn's scan discovers/labels them in.
            min_core_idx_per_cluster = np.full(int(labels[core_idx].max()) + 1,
                                                n, dtype=np.int64)
            np.minimum.at(min_core_idx_per_cluster, labels[core_idx], core_idx)
            # For each (border_pt, border_core) candidate, sort so that for a
            # given border point the candidate with the numerically-smallest
            # min_core_idx_per_cluster[cluster] (i.e. earliest-discovered
            # cluster) comes first, then keep only that first entry per point.
            priority = min_core_idx_per_cluster[border_cluster]
            order = np.lexsort((priority, border_pt))
            bp_sorted = border_pt[order]
            bc_sorted = border_cluster[order]
            first_mask = np.ones(len(bp_sorted), dtype=bool)
            first_mask[1:] = bp_sorted[1:] != bp_sorted[:-1]
            labels[bp_sorted[first_mask]] = bc_sorted[first_mask]

    return labels


def _grain_bds(pts, pairs57, psi6, eps, minn, asp_thr, ang_thr):
    if len(pairs57) < minn: return []
    pairs_arr = np.asarray(pairs57, dtype=int)             # (P, 2)
    i5, i7    = pairs_arr[:, 0], pairs_arr[:, 1]
    mids      = 0.5 * (pts[i5] + pts[i7])
    # Vectorised misorientation per pair (replaces a per-pair Python call
    # inside the per-cluster loop below): NaN where either endpoint is
    # disordered (|ψ₆| < 0.1).
    a, b   = psi6[i5], psi6[i7]
    mis_all = np.degrees(np.abs(np.angle(a * np.conj(b)))) / 6.0
    mis_all = np.where((np.abs(a) < 0.1) | (np.abs(b) < 0.1), np.nan, mis_all)

    labels = _dbscan_via_kdtree(mids, eps=float(eps), min_samples=minn)
    out    = []
    for lbl in set(labels):
        if lbl==-1: continue
        mask = labels==lbl; cl = mids[mask]
        if len(cl)<minn: continue
        c0 = cl-cl.mean(0); eigs,vecs = np.linalg.eigh(c0.T@c0)
        asp = float(np.sqrt(eigs[-1]/eigs[-2])) if eigs[-2]>1e-9 else 999.
        if asp<asp_thr: continue
        axis = vecs[:,-1]; proj = c0@axis; ctr = cl.mean(0)
        p1,p2 = ctr+proj.min()*axis, ctr+proj.max()*axis
        mis_cl = mis_all[mask]
        mis_cl = mis_cl[np.isfinite(mis_cl)]
        mm     = float(mis_cl.mean()) if len(mis_cl) else 0.
        out.append({"kind":"LAGB" if mm<ang_thr else "HAGB",
                    "p1":p1,"p2":p2,"n":len(cl),"aspect":asp,"misorientation_deg":mm})
    return out


def _pair_correlation_hist(pts_um: np.ndarray, psi6: np.ndarray, r_max_um: float,
                            dr_um: float, exclude_boundary: bool = False,
                            bounds_um: "tuple | None" = None):
    """Raw (unnormalized) pairwise-distance histogram accumulation for g(r)/g6(r).

    Single cKDTree.query_pairs() call enumerates every pair with separation
    < r_max_um once; the same pairs are binned for both the plain pair count
    (-> g(r)) and the psi6-correlation sum (-> g6(r)). This is the reusable
    building block for both a single-frame computation and a multi-frame
    trajectory-average accumulator (call this once per frame, sum the raw
    arrays, then normalize once at the end — do NOT average already-
    normalized per-frame g(r) curves, that is not statistically correct).

    exclude_boundary:
        If True, drop any pair where EITHER particle is within r_max_um of
        the bounding-box edge given by bounds_um=(xmin,xmax,ymin,ymax) — an
        opt-in edge correction. query_pairs returns unordered pairs and each
        particle acts as the "reference" for the other, so the exclusion is
        applied symmetrically to both members of a pair.

    Returns
    -------
    n_pairs_per_bin : (nbins,) int array — raw pair counts per distance bin
    sum_g6_per_bin  : (nbins,) float array — sum of Re(psi6[i]*conj(psi6[j]))
                       per distance bin
    N               : int — number of particles actually eligible as
                       reference points (all particles, or boundary-excluded
                       count when exclude_boundary is set — used only for
                       bookkeeping; density normalization always uses the
                       full particle count, per the caller's convention)
    area_um2        : float — bounding-box area of pts_um (µm^2)
    bin_edges_um    : (nbins+1,) array of bin edges (µm)
    """
    n = len(pts_um)
    nbins = max(1, int(r_max_um / dr_um)) if dr_um > 0 else 1
    bin_edges = np.linspace(0.0, r_max_um, nbins + 1)
    n_pairs_per_bin = np.zeros(nbins, dtype=np.int64)
    sum_g6_per_bin = np.zeros(nbins, dtype=np.float64)

    if n < 2:
        area = 0.0
        return n_pairs_per_bin, sum_g6_per_bin, n, area, bin_edges

    xmin, xmax = pts_um[:, 0].min(), pts_um[:, 0].max()
    ymin, ymax = pts_um[:, 1].min(), pts_um[:, 1].max()
    area = max(float((xmax - xmin) * (ymax - ymin)), 1e-12)

    tree = cKDTree(pts_um)
    pairs = tree.query_pairs(r=float(r_max_um), output_type="ndarray")
    if len(pairs) == 0:
        return n_pairs_per_bin, sum_g6_per_bin, n, area, bin_edges

    i_idx, j_idx = pairs[:, 0], pairs[:, 1]

    if exclude_boundary:
        if bounds_um is None:
            bxmin, bxmax, bymin, bymax = xmin, xmax, ymin, ymax
        else:
            bxmin, bxmax, bymin, bymax = bounds_um
        # A particle is "interior" if it is farther than r_max_um from every
        # edge of the frame bounding box.
        interior = ((pts_um[:, 0] - bxmin) > r_max_um) & ((bxmax - pts_um[:, 0]) > r_max_um) & \
                   ((pts_um[:, 1] - bymin) > r_max_um) & ((bymax - pts_um[:, 1]) > r_max_um)
        keep = interior[i_idx] & interior[j_idx]
        i_idx, j_idx = i_idx[keep], j_idx[keep]
        if len(i_idx) == 0:
            return n_pairs_per_bin, sum_g6_per_bin, n, area, bin_edges

    dx = pts_um[j_idx, 0] - pts_um[i_idx, 0]
    dy = pts_um[j_idx, 1] - pts_um[i_idx, 1]
    d = np.hypot(dx, dy)

    bin_idx = np.clip(np.searchsorted(bin_edges, d, side="right") - 1, 0, nbins - 1)
    valid = d < r_max_um
    bin_idx = bin_idx[valid]
    i_v, j_v = i_idx[valid], j_idx[valid]

    np.add.at(n_pairs_per_bin, bin_idx, 1)
    g6_terms = np.real(psi6[i_v] * np.conj(psi6[j_v]))
    np.add.at(sum_g6_per_bin, bin_idx, g6_terms)

    return n_pairs_per_bin, sum_g6_per_bin, n, area, bin_edges


def _normalize_pair_correlations(n_pairs_per_bin: np.ndarray, sum_g6_per_bin: np.ndarray,
                                  N: int, area_um2: float, bin_edges_um: np.ndarray,
                                  mean_psi6_sq: float, n_frames: int = 1):
    """Normalize raw pair-count/psi6-sum histograms into g(r), g6(r), g6(r)/g(r).

    N, area_um2 are totals (summed across frames for a trajectory average, or
    single-frame values); n_frames divides the density so rho is a per-frame
    average number density, consistent with n_pairs_per_bin/sum_g6_per_bin
    also being sums across n_frames frames.
    """
    dr_um = bin_edges_um[1] - bin_edges_um[0] if len(bin_edges_um) > 1 else 1.0
    r_centers = 0.5 * (bin_edges_um[:-1] + bin_edges_um[1:])

    mean_N = N / max(n_frames, 1)
    mean_area = area_um2 / max(n_frames, 1)
    rho = mean_N / mean_area if mean_area > 0 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        denom_g = n_frames * mean_N * rho * 2.0 * np.pi * r_centers * dr_um
        g_r = np.where(denom_g > 0, (2.0 * n_pairs_per_bin) / np.where(denom_g > 0, denom_g, 1.0), np.nan)

        g6_r = np.where(n_pairs_per_bin > 0,
                         sum_g6_per_bin / np.where(n_pairs_per_bin > 0, n_pairs_per_bin, 1.0) / max(mean_psi6_sq, 1e-12),
                         np.nan)

        g6_over_g_r = np.where((g_r > 0) & np.isfinite(g_r), g6_r / np.where(g_r > 0, g_r, 1.0), np.nan)

    return r_centers, g_r, g6_r, g6_over_g_r


def _pair_correlations(pts_um: np.ndarray, psi6: np.ndarray, r_max_um: float,
                        dr_um: float, exclude_boundary: bool = False,
                        bounds_um: "tuple | None" = None):
    """Single-frame g(r) and g6(r) via one cKDTree.query_pairs call.

    Returns (r_centers, g_r, g6_r, g6_over_g_r), each a (nbins,) array (µm
    for r_centers; dimensionless for the others; NaN where a bin has no
    pairs / normalization would divide by zero).
    """
    n_pairs_per_bin, sum_g6_per_bin, N, area_um2, bin_edges = _pair_correlation_hist(
        pts_um, psi6, r_max_um, dr_um, exclude_boundary=exclude_boundary, bounds_um=bounds_um)
    mean_psi6_sq = float(np.mean(np.abs(psi6) ** 2)) if len(psi6) else 0.0
    return _normalize_pair_correlations(n_pairs_per_bin, sum_g6_per_bin, N, area_um2,
                                         bin_edges, mean_psi6_sq, n_frames=1)


def _pair_correlations_trajectory(frame_results: dict, frame_range: "range | list",
                                   px_um: float, r_max_um: float, dr_um: float,
                                   exclude_boundary: bool = False,
                                   progress_cb=None):
    """Trajectory-averaged g(r)/g6(r): accumulates raw histograms across all
    frames in frame_range, then normalizes once at the end (statistically
    correct — averaging already-normalized per-frame curves is not).

    progress_cb, if given, is called as progress_cb(n_done, n_total) after
    each processed frame (used to drive a worker-thread progress signal).
    """
    nbins = max(1, int(r_max_um / dr_um)) if dr_um > 0 else 1
    total_n_pairs = np.zeros(nbins, dtype=np.int64)
    total_sum_g6 = np.zeros(nbins, dtype=np.float64)
    total_N = 0
    total_area = 0.0
    total_psi6_sq_sum = 0.0
    total_psi6_count = 0
    n_frames_used = 0
    bin_edges = np.linspace(0.0, r_max_um, nbins + 1)

    frames = list(frame_range)
    n_total = len(frames)

    def _one(fr):
        res = frame_results.get(fr)
        if res is None or not len(res.pts_px):
            return None
        h = _pair_correlation_hist(res.pts_px * px_um, res.psi6, r_max_um, dr_um,
                                   exclude_boundary=exclude_boundary, bounds_um=None)
        return h, float(np.sum(np.abs(res.psi6) ** 2)), len(res.psi6)

    # Frames are independent; _pair_correlation_hist's cKDTree work releases
    # the GIL, so a small thread pool gives ~4x here (measured). Accumulators
    # are updated in exact serial frame order via ordered pool.map consumption
    # -> bit-identical sums; progress_cb stays monotonic and on this thread.
    with _cf.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as pool:
        for k, item in enumerate(pool.map(_one, frames)):
            if item is not None:
                (n_pairs_per_bin, sum_g6_per_bin, N, area_um2, _be), psq, pcnt = item
                total_n_pairs += n_pairs_per_bin
                total_sum_g6 += sum_g6_per_bin
                total_N += N
                total_area += area_um2
                total_psi6_sq_sum += psq
                total_psi6_count += pcnt
                n_frames_used += 1
            if progress_cb is not None:
                progress_cb(k + 1, n_total)

    mean_psi6_sq = (total_psi6_sq_sum / total_psi6_count) if total_psi6_count else 0.0
    return _normalize_pair_correlations(total_n_pairs, total_sum_g6, total_N, total_area,
                                         bin_edges, mean_psi6_sq, n_frames=max(n_frames_used, 1))


def _defect_concentration_series(frame_results: dict) -> pd.DataFrame:
    """Per-frame defect-concentration reduction — pure reduction over the
    already-computed `coord` array, no new per-frame computation needed.

    Returns a DataFrame with columns: frame, frac_defect, frac_5fold, frac_7fold,
    sorted by frame.
    """
    rows = []
    for fr, res in frame_results.items():
        coord = res.coord
        if coord is None or len(coord) == 0:
            frac_defect = frac_5 = frac_7 = float("nan")
        else:
            frac_defect = float(np.mean(coord != 6))
            frac_5 = float(np.mean(coord == 5))
            frac_7 = float(np.mean(coord == 7))
        rows.append((fr, frac_defect, frac_5, frac_7))
    df = pd.DataFrame(rows, columns=["frame", "frac_defect", "frac_5fold", "frac_7fold"])
    return df.sort_values("frame").reset_index(drop=True)


def analyze_frame(pts: np.ndarray, dp: dict) -> FrameResult:
    # Optional max-neighbour-distance cutoff (µm, converted to px via px_um)
    # to reject long spurious Delaunay edges at cluster boundaries / dilute
    # regions. 0/absent disables the filter (legacy behaviour).
    max_dist_um = float(dp.get("max_neighbor_dist_um", 0.0) or 0.0)
    px_um       = float(dp.get("px_um", 0.0) or 0.0)
    max_dist_px = (max_dist_um / px_um) if (max_dist_um > 0 and px_um > 0) else None
    coord, psi6, edges = _delaunay(pts, max_dist_px=max_dist_px)
    pairs57 = _57_pairs(pts, coord, edges, dp["pair_dist_px"])
    bds     = _grain_bds(pts, pairs57, psi6, dp["lagb_eps_px"], dp["lagb_min_n"],
                          dp["lagb_aspect"], dp["lagb_angle_deg"])
    return FrameResult(pts, coord, psi6, pairs57, bds, edges)


def _affine(trk):
    """Affine drift correction.

    This is a genuinely sequential recurrence (frame N's correction depends
    on frame N-1's already-corrected coordinates) and cannot be parallelized
    across frames. What CAN be cut is the per-frame pandas overhead: this
    pre-extracts particle/x_um/y_um/t into flat numpy arrays once, matches
    particles between consecutive frames with np.intersect1d + searchsorted
    instead of DataFrame.merge(), and writes corrections directly into those
    arrays (only at the very end is the DataFrame touched again) instead of
    per-frame .loc[...] label writes.

    Pre-builds a frame→row-position mapping so each frame is accessed in
    O(1) instead of scanning the full DataFrame per frame.
    """
    df  = trk.sort_values(["frame", "particle"]).copy().reset_index(drop=True)
    n   = len(df)
    particle = df["particle"].values
    x_um     = df["x_um"].values.astype(float).copy()
    y_um     = df["y_um"].values.astype(float).copy()
    t_val    = df["t"].values.astype(float)
    x_aff    = x_um.copy()
    y_aff    = y_um.copy()

    # Build frame→row-position mapping once — O(N) total instead of O(F·N).
    # Positions within each frame are already particle-sorted (stable sort
    # above), matching the previous groupby(...).index.values order.
    frame_pos: dict = {}
    for fr_val, grp in df.groupby("frame", sort=False):
        frame_pos[int(fr_val)] = grp.index.values  # positional, since df was reset_index'd

    frs = np.sort(df["frame"].unique()).astype(int)
    for fr, frn in zip(frs[:-1], frs[1:]):
        pos0 = frame_pos.get(fr)
        pos1 = frame_pos.get(frn)
        if pos0 is None or pos1 is None:
            continue

        p0 = particle[pos0]
        p1 = particle[pos1]
        # particle ids are unique within a frame (sorted-and-grouped above),
        # so intersect1d + searchsorted reproduces the inner join on
        # "particle" that DataFrame.merge(g0, g1, on="particle") performed —
        # the fit below only depends on the *set* of matched pairs, not row
        # order, so this is exactly equivalent.
        common = np.intersect1d(p0, p1, assume_unique=True)
        if len(common) < 20:
            continue
        m0 = pos0[np.searchsorted(p0, common)]
        m1 = pos1[np.searchsorted(p1, common)]

        dt = float(np.median(t_val[m1] - t_val[m0]))
        if not (np.isfinite(dt) and dt > 0):
            continue

        X  = np.c_[x_um[m0], y_um[m0], np.ones(len(common))]
        a  = np.linalg.lstsq(X, (x_um[m1] - x_um[m0]) / dt, rcond=None)[0]
        b  = np.linalg.lstsq(X, (y_um[m1] - y_um[m0]) / dt, rcond=None)[0]

        X1 = np.c_[x_um[pos1], y_um[pos1], np.ones(len(pos1))]
        x_aff[pos1] = x_um[pos1] - (X1 @ a) * dt
        y_aff[pos1] = y_um[pos1] - (X1 @ b) * dt
        # Subsequent frames must see the corrected coordinates (sequential
        # recurrence) — update the working arrays in place, mirroring the
        # original df.loc[idx1, "x_um"] = ... overwrite.
        x_um[pos1] = x_aff[pos1]
        y_um[pos1] = y_aff[pos1]

    df["x_um_aff"] = x_aff
    df["y_um_aff"] = y_aff
    df["x_um"]     = x_um
    df["y_um"]     = y_um
    return df


def _cage(trk, k=8, max_neighbor_dist_um: "float | None" = None):
    """Cage-relative coordinates.

    Pre-allocates output arrays and writes by row position instead of
    copying each per-frame group and concat-ing at the end.

    max_neighbor_dist_um:
        Optional cutoff (µm, same unit as x_um_aff/y_um_aff — no px_um
        conversion needed here). The k-nearest-neighbour query below has the
        same "spurious distant neighbour" problem as Delaunay triangulation:
        nearest-8 always returns 8 points regardless of how far away they
        actually are, which is wrong at the edge of a cluster or in dilute
        regions. When set (> 0), any of the k nearest neighbours farther than
        this distance is excluded from the cage-centroid average instead of
        being treated as a real neighbour. None/0 (default) disables the
        filter — legacy behaviour (always averages exactly k neighbours).
    """
    df = trk.reset_index(drop=True)   # ensure 0-based iloc
    n  = len(df)
    x_cr = df["x_um_aff"].values.copy()
    y_cr = df["y_um_aff"].values.copy()
    cutoff = float(max_neighbor_dist_um) if max_neighbor_dist_um else None

    # Precompute frame -> (pos, size, P) offsets in a single sequential pass
    # (group order doesn't matter — writes below are position-keyed via pos),
    # then dispatch each frame's KDTree build/query to a thread pool. cKDTree
    # releases the GIL (same justification as the structural-analysis stage),
    # so this parallelizes cleanly; each task only writes its own pos:pos+size
    # slice of x_cr/y_cr, so there's no cross-task contention.
    jobs = []
    pos = 0
    for _, g in df.groupby("frame", sort=False):
        size = len(g)
        P    = g[["x_um_aff", "y_um_aff"]].values
        jobs.append((pos, size, P))
        pos += size

    def _cage_one(job):
        pos, size, P = job
        kk = min(k + 1, size)
        if kk >= 2:
            dist, idx = cKDTree(P).query(P, k=kk)
            nbr_idx  = idx[:, 1:]
            nbr_dist = dist[:, 1:]
            if cutoff is not None and cutoff > 0:
                valid = nbr_dist <= cutoff
                cnt = valid.sum(axis=1)
                # Rows with zero valid neighbours (all farther than cutoff)
                # fall back to the unfiltered mean rather than producing NaN —
                # keeps behaviour graceful in very dilute/edge regions.
                neighbor_pts = np.where(valid[:, :, None], P[nbr_idx], 0.0)
                safe_cnt = np.maximum(cnt, 1)[:, None]
                mean_valid = neighbor_pts.sum(axis=1) / safe_cnt
                mean_all   = P[nbr_idx].mean(axis=1)
                mean_nbr = np.where((cnt > 0)[:, None], mean_valid, mean_all)
                Q = P - mean_nbr
            else:
                Q = P - P[nbr_idx].mean(axis=1)
            return pos, size, Q[:, 0], Q[:, 1]
        return pos, size, None, None

    n_workers = min(8, max(1, os.cpu_count() or 2))
    with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for pos, size, qx, qy in pool.map(_cage_one, jobs):
            if qx is not None:
                x_cr[pos:pos + size] = qx
                y_cr[pos:pos + size] = qy

    df["x_um_cr"] = x_cr
    df["y_um_cr"] = y_cr
    return df


# ════════════════════════════════════════════════════════════════════
# Dynamical analysis — NEW (Stokes-Einstein-Sutherland layer).
# The GUI app never computed diffusion; TempSim needs it (criterion 3).
#
# NOTE: diffusion_D auto-detects the long-time linear window AND classifies caging /
# sub-diffusion (crystal rattling plateau); for a 'caged' result use plateau_msd as the
# thermal proxy rather than D (see PLAN.md §2 risk 3).
# ════════════════════════════════════════════════════════════════════

K_B = 1.380649e-23   # J/K. In reduced simulation units pass kB=1.0 explicitly.


def msd(tracks: pd.DataFrame, pos_cols=("x_um", "y_um"), max_lag: "int | None" = None,
        drift_subtract: bool = True):
    """Ensemble-and-time-averaged mean-squared displacement vs lag time.

    Uses all overlapping frame pairs per particle (trackpy `emsd` semantics).
    2D expectation in the diffusive regime: MSD = 4 D t.

    Parameters
    ----------
    tracks : DataFrame with columns 'particle', 'frame', 't', and `pos_cols`.
    pos_cols : the two position columns to use (default microns; pass the
               drift/cage-corrected columns for consistency with the experiment).
    max_lag : maximum lag in FRAMES (default: half the frame span).
    drift_subtract : subtract the per-lag mean displacement vector (net drift)
                     before squaring, i.e. MSD = <|Δr - <Δr>|²>. Mirrors the
                     app's _affine intent for datasets without pre-correction.

    Returns
    -------
    lag_t : (L,) lag times in seconds (uses the median frame->t spacing).
    msd   : (L,) MSD in the squared units of `pos_cols`.
    """
    xcol, ycol = pos_cols
    df = tracks[["particle", "frame", "t", xcol, ycol]].dropna()
    frames_all = np.sort(df["frame"].unique())
    if len(frames_all) < 2:
        return np.zeros(0), np.zeros(0)
    # median frame->seconds spacing (robust to gaps)
    t_by_frame = df.groupby("frame")["t"].median()
    dt = float(np.median(np.diff(t_by_frame.reindex(frames_all).values)))
    span = int(frames_all.max() - frames_all.min())
    if max_lag is None:
        max_lag = max(1, span // 2)
    max_lag = int(min(max_lag, span))

    # Per-particle frame->position lookup (dict of frame->(x,y)).
    part_pos: dict = {}
    for pid, g in df.groupby("particle", sort=False):
        fr = g["frame"].to_numpy()
        xy = g[[xcol, ycol]].to_numpy(dtype=float)
        part_pos[pid] = dict(zip(fr.tolist(), xy))

    lags = np.arange(1, max_lag + 1)
    msd_vals = np.full(len(lags), np.nan)
    for li, lag in enumerate(lags):
        dxs = []
        dys = []
        for pos in part_pos.values():
            for f, xy0 in pos.items():
                xy1 = pos.get(f + lag)
                if xy1 is not None:
                    dxs.append(xy1[0] - xy0[0])
                    dys.append(xy1[1] - xy0[1])
        if not dxs:
            continue
        dxs = np.asarray(dxs); dys = np.asarray(dys)
        if drift_subtract:
            dxs = dxs - dxs.mean()
            dys = dys - dys.mean()
        msd_vals[li] = float(np.mean(dxs * dxs + dys * dys))

    return lags * dt, msd_vals


def diffusion_D(lag_t: np.ndarray, msd_vals: np.ndarray,
                fit_window: "tuple | None" = None, dim: int = 2,
                slope_tol: float = 0.18, cage_slope: float = 0.5) -> dict:
    """Extract D from the long-time linear MSD slope (2D: MSD = 4 D t; general 2*dim*D*t),
    with explicit caging / sub-diffusion handling.

    Caging in a crystal makes the "long-time linear" window subtle (PLAN.md §2, risk 3):
    the MSD rises ballistically/diffusively, then PLATEAUS at the cage (rattling) amplitude,
    and only rare hops give a terminal linear regime whose slope may be << 1. This routine:

      1. classifies the regime from the terminal (late-time) log-log slope:
           terminal_slope >= cage_slope  -> 'diffusive'   (a real long-time D)
           terminal_slope <  cage_slope  -> 'caged'        (report the plateau; D unreliable)
      2. picks the fit window (unless `fit_window=(lo,hi)` indices given) as the LONGEST
         contiguous run where the log-log slope is within slope_tol of 1; if no such run
         exists (fully caged), falls back to the late third and flags the regime.

    Returns dict(D, D_err, slope, intercept, window, r2, n_points, regime,
                 terminal_loglog_slope, plateau_msd). For a 'caged' result, prefer
        plateau_msd (the Debye-Waller / Lindemann rattling amplitude) as the thermal proxy.
    """
    lag_t = np.asarray(lag_t, dtype=float)
    msd_vals = np.asarray(msd_vals, dtype=float)
    good = np.isfinite(lag_t) & np.isfinite(msd_vals) & (lag_t > 0) & (msd_vals > 0)
    idx = np.nonzero(good)[0]
    nan = float("nan")
    if len(idx) < 2:
        return dict(D=nan, D_err=nan, slope=nan, intercept=nan, window=None, r2=nan,
                    n_points=0, regime="insufficient", terminal_loglog_slope=nan, plateau_msd=nan)

    lt, lm = np.log(lag_t[idx]), np.log(msd_vals[idx])
    loglog = np.gradient(lm, lt)                       # local log-log slope
    tail = max(2, len(idx) // 3)
    terminal_slope = float(np.median(loglog[-tail:]))
    plateau = float(np.median(msd_vals[idx][-tail:]))
    regime = "diffusive" if terminal_slope >= cage_slope else "caged"

    if fit_window is not None:
        lo, hi = fit_window
        lo = max(lo, int(idx.min())); hi = min(hi, int(idx.max()))
        w = np.arange(lo, hi + 1)
    else:
        # longest contiguous run (in the good-index sequence) with |loglog - 1| <= slope_tol
        near1 = np.abs(loglog - 1.0) <= slope_tol
        best_a = best_b = -1; run_start = None
        for k, v in enumerate(list(near1) + [False]):
            if v and run_start is None:
                run_start = k
            elif not v and run_start is not None:
                if (k - 1 - run_start) > (best_b - best_a):
                    best_a, best_b = run_start, k - 1
                run_start = None
        if best_b >= best_a and best_b > best_a:
            w = idx[best_a:best_b + 1]
        else:
            w = idx[len(idx) * 2 // 3:]     # no clean diffusive window (caged) -> late third
    w = w[np.isin(w, idx)]
    if len(w) < 2:
        w = idx

    x = lag_t[w]; y = msd_vals[w]
    A = np.c_[x, np.ones_like(x)]
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([slope, intercept])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else nan
    n = len(x)
    resid_var = ss_res / (n - 2) if n > 2 else nan
    sxx = float(np.sum((x - x.mean()) ** 2))
    slope_err = np.sqrt(resid_var / sxx) if (sxx > 0 and np.isfinite(resid_var)) else nan

    denom = 2.0 * dim
    return dict(D=slope / denom, D_err=slope_err / denom if np.isfinite(slope_err) else nan,
                slope=slope, intercept=intercept, window=(int(w[0]), int(w[-1])),
                r2=r2, n_points=n, regime=regime,
                terminal_loglog_slope=terminal_slope, plateau_msd=plateau)


def stokes_einstein(D: float, T: float, eta: float, dim: int = 3, kB: float = K_B) -> float:
    """Solve D = kB*T / (6*pi*eta*a) for hydrodynamic radius a (3D Stokes drag).

    For quasi-2D colloids at a wall the sphere still feels ~3D solvent drag, so the
    6*pi form is the usual operational choice; document if a different drag is used.
    """
    return kB * T / (6.0 * np.pi * eta * D)


def effective_temperature(D: float, a: float, eta: float, dim: int = 3, kB: float = K_B) -> float:
    """Invert SES for an effective temperature: T_eff = 6*pi*eta*a*D / kB.

    The key observable for the research question. Prefer the dimensionless ratio
    D(GB)/D(bulk) (see local_D_map) which cancels eta and a.
    """
    return 6.0 * np.pi * eta * a * D / kB


def local_D_map(tracks: pd.DataFrame, boundary_geom, n_bins: int = 20, short_window=None):
    """Spatially resolved diffusivity: bin particles by distance-to-nearest-GB, compute
    per-bin short-time D, return the D(x) (hence T_eff(x)) profile across the boundary.

    Report D(GB)/D(bulk) as the headline, confound-robust observable.

    Deferred: needs the grain-boundary geometry object from the simulation
    (sim.bicrystal) to define distance-to-GB. Implement alongside Phase-1 runs.
    """
    raise NotImplementedError("Implement with sim.bicrystal boundary geometry (Phase 1).")
