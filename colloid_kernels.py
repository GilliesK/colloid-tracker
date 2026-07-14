"""Optional Numba-JIT kernels for colloid_app.py's particle-detection hot path.

Side module so the main app stays a single file for everything else; this is
imported lazily and colloid_app.py falls back to a pure-numpy/Python
implementation if numba isn't installed or JIT compilation fails for any
reason (e.g. unsupported CPU, broken install) — detection must never hard-fail
for lack of this optional speedup.
"""
import numpy as np

try:
    import numba as _nb
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False


if NUMBA_OK:
    @_nb.njit(cache=True, fastmath=True, nogil=True)
    def greedy_separation_keep(pairs_i, pairs_j, masses, n):
        """Faithful JIT port of the original Python greedy-NMS loop:
        for each (i, j) pair (already sorted by descending pair max-mass),
        if both are still 'kept', drop whichever has lower mass.

        Pure-Python this loop is O(P) with per-iteration interpreter overhead;
        for dense colloidal packings P (close-pair count) can reach the tens
        of thousands per frame. JIT removes the interpreter overhead entirely
        with identical semantics/output to the numpy/Python version.
        """
        keep = np.ones(n, dtype=np.bool_)
        for k in range(len(pairs_i)):
            i = pairs_i[k]; j = pairs_j[k]
            if keep[i] and keep[j]:
                if masses[i] < masses[j]:
                    keep[i] = False
                else:
                    keep[j] = False
        return keep
else:
    def greedy_separation_keep(pairs_i, pairs_j, masses, n):
        keep = np.ones(n, dtype=bool)
        for k in range(len(pairs_i)):
            i = pairs_i[k]; j = pairs_j[k]
            if keep[i] and keep[j]:
                keep[i if masses[i] < masses[j] else j] = False
        return keep


# ── ψ₆ order-parameter scatter-add ──────────────────────────────────────────
# The bottleneck in _delaunay is np.add.at(psi6, i_all, np.exp(6j*angle)) —
# a sequential scatter-add that NumPy executes element-by-element.  The JIT
# version splits into real/imag parts (Numba handles complex poorly) and uses
# a plain indexed loop which the JIT compiles to a tight native loop with no
# interpreter overhead.

if NUMBA_OK:
    @_nb.njit(cache=True, fastmath=True, nogil=True)
    def psi6_accumulate(i_all, j_all, dx, dy, N):
        """ψ₆ order-parameter scatter-add, Numba JIT.

        Computes psi6[i] += exp(6j * arctan2(dy, dx)) for each directed edge
        (i→j), and coord[i] = number of neighbours of i.  Split into real/imag
        because Numba's complex scatter-add is less efficient than two float
        arrays.

        Returns (psi6_complex_array, coord_int_array) with the same semantics
        as the numpy np.add.at + np.bincount pattern in _delaunay.
        """
        psi6_r = np.zeros(N, dtype=np.float64)
        psi6_i = np.zeros(N, dtype=np.float64)
        coord  = np.zeros(N, dtype=np.int64)
        for k in range(len(i_all)):
            angle = 6.0 * np.arctan2(dy[k], dx[k])
            psi6_r[i_all[k]] += np.cos(angle)
            psi6_i[i_all[k]] += np.sin(angle)
            coord[i_all[k]]  += 1
        return psi6_r + 1j * psi6_i, coord
else:
    def psi6_accumulate(i_all, j_all, dx, dy, N):
        """Pure-Python fallback when Numba is absent."""
        psi6  = np.zeros(N, dtype=complex)
        coord = np.zeros(N, dtype=np.int64)
        np.add.at(psi6, i_all, np.exp(6j * np.arctan2(dy, dx)))
        np.add.at(coord, i_all, 1)
        return psi6, coord


# ── Nearest-neighbor linker per-frame matching kernel ──────────────────────
# _link_nn (colloid_app.py) queries a cKDTree of the current frame's
# detections for each live particle's k nearest candidates, then greedily
# assigns nearest-first while resolving conflicts (a detection or live
# particle already claimed loses out and its partner falls back to the next
# candidate). Both the candidate-gathering double loop and the greedy
# assignment loop are pure Python, run once per frame, and for dense
# colloidal-crystal frames (hundreds-to-low-thousands of particles) this is a
# real per-frame interpreter-overhead cost across an entire video. JIT
# compiling removes that overhead with identical semantics/output.
#
# The cKDTree build/query itself is NOT ported (scipy isn't numba-compatible)
# — only the post-query candidate-collection + greedy-assignment loops move
# into the kernel. Inputs are the tree's raw `dists`/`idxs` (k), n_live,
# n_det and search_range; outputs are the same assigned_live/assigned_det
# arrays (-1 sentinel for unmatched) the pure-Python version produced.

if NUMBA_OK:
    @_nb.njit(cache=True, fastmath=True, nogil=True)
    def link_nn_match(dists, idxs, n_live, n_det, search_range):
        """JIT port of the candidate-gather + greedy-assign loops in _link_nn.

        dists, idxs: (n_live, k) arrays from cKDTree.query(..., k=k,
            distance_upper_bound=search_range). idxs entries equal to n_det
            mean "no neighbour found" (scipy's sentinel for an under-filled
            query) and are skipped, matching the original Python loop.

        Returns (assigned_live, assigned_det): int64 arrays, -1 where
        unmatched, identical semantics to the pure-Python greedy loop
        (nearest-first assignment, skip already-claimed live/det, natural
        fallback to next-nearest candidate on conflict).
        """
        k = dists.shape[1]
        assigned_live = np.full(n_live, -1, dtype=np.int64)
        assigned_det  = np.full(n_det, -1, dtype=np.int64)

        # Gather valid (live_idx, det_idx, dist) candidates, preallocating to
        # the worst case (n_live * k) and tracking the actual count used.
        max_cand = n_live * k
        cand_live = np.empty(max_cand, dtype=np.int64)
        cand_det  = np.empty(max_cand, dtype=np.int64)
        cand_dist = np.empty(max_cand, dtype=np.float64)
        n_cand = 0
        for li in range(n_live):
            for kk in range(k):
                dj = idxs[li, kk]
                dd = dists[li, kk]
                if dj == n_det or not np.isfinite(dd) or dd > search_range:
                    continue
                cand_live[n_cand] = li
                cand_det[n_cand]  = dj
                cand_dist[n_cand] = dd
                n_cand += 1

        if n_cand == 0:
            return assigned_live, assigned_det

        order = np.argsort(cand_dist[:n_cand])
        claimed_live = np.zeros(n_live, dtype=np.bool_)
        claimed_det  = np.zeros(n_det, dtype=np.bool_)
        for oi in range(n_cand):
            ci = order[oi]
            li = cand_live[ci]; dj = cand_det[ci]
            if claimed_live[li] or claimed_det[dj]:
                continue
            claimed_live[li] = True
            claimed_det[dj]  = True
            assigned_live[li] = dj
            assigned_det[dj]  = li

        return assigned_live, assigned_det
else:
    def link_nn_match(dists, idxs, n_live, n_det, search_range):
        """Pure-Python fallback when Numba is absent (identical semantics)."""
        k = dists.shape[1]
        assigned_live = np.full(n_live, -1, dtype=np.int64)
        assigned_det  = np.full(n_det, -1, dtype=np.int64)

        cand_live, cand_det, cand_dist = [], [], []
        for li in range(n_live):
            for kk in range(k):
                dj = idxs[li, kk]
                dd = dists[li, kk]
                if dj == n_det or not np.isfinite(dd) or dd > search_range:
                    continue
                cand_live.append(li); cand_det.append(dj); cand_dist.append(dd)

        if not cand_live:
            return assigned_live, assigned_det

        order = np.argsort(cand_dist, kind="mergesort")
        claimed_live = np.zeros(n_live, dtype=bool)
        claimed_det  = np.zeros(n_det, dtype=bool)
        for oi in order:
            li = cand_live[oi]; dj = cand_det[oi]
            if claimed_live[li] or claimed_det[dj]:
                continue
            claimed_live[li] = True
            claimed_det[dj]  = True
            assigned_live[li] = dj
            assigned_det[dj]  = li

        return assigned_live, assigned_det
