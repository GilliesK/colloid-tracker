#!/usr/bin/env python3
"""
colloid_detect.py — Qt-free image-processing / particle-detection core.

Extracted verbatim from colloid_app.py so the detection pipeline can run
headless on an HPC cluster (no PyQt6 / no Qt event loop). colloid_app.py
imports every public name back from here, so GUI behaviour is unchanged and
detection results are bit-identical before and after the split.

Nothing in this module imports Qt. Only the numeric/vision stack is pulled in:
numpy, pandas, OpenCV, scipy (cKDTree only), pims (@pipeline), and the optional
GPU backends (cupy for FFT ring/dark-disk convolution, cv2.cuda for
bilateral/NLM/bandpass), each guarded so a missing backend only slows detection
down, never breaks it.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import cv2
from scipy.spatial import cKDTree
from numpy.lib.stride_tricks import sliding_window_view as _swv
# pims only provides the @pipeline decorator for the legacy _to_gray helper
# (unused off-GUI). Make it optional so the headless cluster worker doesn't need
# pims (and its slicerator dep) installed — see oscar/README.md.
try:
    from pims import pipeline
except Exception:                       # pragma: no cover - cluster/headless
    def pipeline(func):                 # no-op stand-in; _to_gray is never called here
        return func

# JIT-accelerated greedy separation keep/drop loop (used by _fast_locate);
# falls back to an identical pure-Python loop if colloid_kernels is absent.
try:
    from colloid_kernels import greedy_separation_keep as _greedy_separation_keep
except Exception:
    def _greedy_separation_keep(pairs_i, pairs_j, masses, n):
        keep = np.ones(n, dtype=bool)
        for k in range(len(pairs_i)):
            i = pairs_i[k]; j = pairs_j[k]
            if keep[i] and keep[j]:
                keep[i if masses[i] < masses[j] else j] = False
        return keep

# ── optional cupy (GPU-accelerated ring-filter convolution) ────────────────
# Benchmarked on a GTX 1660 SUPER: a plain GPU port of the bandpass filter
# only nets ~1.5-2x after PCIe transfer overhead (OpenCV's CPU separable
# filters are already near-optimal), so that one stays CPU-only. The ring
# filter's non-separable, large-kernel (up to ~300px) FFT convolution is
# where GPU genuinely wins — measured 3-5x including transfer, with FFT
# results matching cv2.filter2D to ~1e-5 once reflect-padded to match its
# border handling.
try:
    import cupy as _cp
    CUPY_OK = True
except Exception:
    CUPY_OK = False

# ── optional cv2.cuda (GPU-accelerated bilateral/NLM denoise) ──────────────
# Requires a CUDA-enabled cv2 build (opencv-contrib-python from
# cudawarped/opencv-python-cuda-wheels) plus matching CUDA 13.x runtime DLLs.
# Falls back to the CPU cv2 implementations transparently if either the
# build lacks cv2.cuda or no CUDA device is present.
try:
    CV2_CUDA_OK = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
except Exception:
    CV2_CUDA_OK = False

# Module-level CLAHE object cache keyed by (clipLimit, tileGridSize).
# cv2.createCLAHE is called on every preprocessed frame when CLAHE is enabled;
# the CLAHE object itself is stateless between apply() calls so it is safe to
# reuse across frames and threads.
_CLAHE_CACHE: dict = {}


# ════════════════════════════════════════════════════════════════════
# Analysis backend  (unchanged from v1)
# ════════════════════════════════════════════════════════════════════

_INVERT_FLAG = False


@pipeline
def _to_gray(img):
    if getattr(img, "ndim", 2) == 3: img = img.mean(axis=2)
    if _INVERT_FLAG: img = img.max() - img
    return img.astype(np.float32)


_BP_TLS = threading.local()


def _bp_buf(key: str, shape) -> np.ndarray:
    """Per-thread reusable float32 scratch buffer (keyed by name + shape)."""
    d = _BP_TLS.__dict__.setdefault("bufs", {})
    b = d.get(key)
    if b is None or b.shape != shape:
        b = d[key] = np.empty(shape, np.float32)
    return b


def _fast_bandpass(img: np.ndarray, lshort: float, llong: float,
                   reuse_buffers: bool = False) -> np.ndarray:
    """cv2-based Crocker-Grier bandpass: Gaussian(lshort) - Uniform(llong).
    Stays in float32 and uses SIMD C++ — 3-5× faster than tp.bandpass (scipy/float64).

    reuse_buffers=True routes the intermediate/output arrays through
    per-thread scratch buffers, eliminating ~4 full-frame allocations per
    call (~1.3 GB/s of allocation traffic at 2840² across a 10-worker pool;
    measured 1.15-1.25× detection-stage throughput). Outputs are
    bit-identical (cv2 dst= runs identical kernel code; copyto/subtract/
    maximum are elementwise). WARNING: the returned array then aliases a
    per-thread buffer valid only until this thread's next call — callers
    must fully consume it within the same task and never retain it
    (TrackingWorker._process_gray's plain path qualifies; display/preview
    paths that keep the result must use the default False)."""
    if img.dtype == np.float32:
        f = img
    elif reuse_buffers:
        f = _bp_buf("f32", img.shape)
        np.copyto(f, img, casting="unsafe")
    else:
        f = img.astype(np.float32)
    smooth = cv2.GaussianBlur(f, (0, 0), sigmaX=float(lshort),
                              borderType=cv2.BORDER_REFLECT,
                              dst=_bp_buf("g", f.shape) if reuse_buffers else None)
    sz = max(1, int(round(float(llong))))
    bg = cv2.blur(f, (sz, sz), borderType=cv2.BORDER_REFLECT,
                  dst=_bp_buf("b", f.shape) if reuse_buffers else None)
    np.subtract(smooth, bg, out=smooth)
    np.maximum(smooth, 0, out=smooth)
    return smooth


_RING_KERNEL_CACHE: dict = {}      # diameter -> numpy kernel
_RING_KERNEL_CACHE_GPU: dict = {}  # diameter -> cupy kernel (only populated if CUPY_OK)

# Persistent GPU resources for bilateral / NLM denoise.
# Keyed by (shape, dtype) so the GpuMat is reused across frames of the same
# size without re-allocating device memory on every call.  The stream is
# shared across both functions; initialised lazily on first use.
_GPU_BUF_CACHE: dict = {}               # (shape, dtype) → cv2.cuda_GpuMat
_CUDA_STREAM: "cv2.cuda.Stream | None" = None

def _ring_kernel(diameter: int) -> np.ndarray:
    if diameter not in _RING_KERNEL_CACHE:
        r = max(2, diameter // 2)
        d = 2 * r + 1
        y, x = np.ogrid[-r:r+1, -r:r+1]
        dist2 = (x * x + y * y).astype(np.float32)

        # Inner disk: 0 – 55 % of r (where the bright centre lives)
        # Annular rim: 55 – 100 % of r (where the dark rim lives)
        inner_r2 = (r * 0.55) ** 2
        centre_mask = dist2 <= inner_r2
        rim_mask    = (dist2 > inner_r2) & (dist2 <= float(r * r))

        n_c = int(centre_mask.sum())
        n_r = int(rim_mask.sum())

        kernel = np.zeros((d, d), np.float32)
        if n_c > 0 and n_r > 0:
            # Zero-mean: sum_centre(+1/n_c) + sum_rim(-1/n_r) = 1 - 1 = 0
            kernel[centre_mask] =  1.0 / n_c
            kernel[rim_mask]    = -1.0 / n_r
        _RING_KERNEL_CACHE[diameter] = kernel
    return _RING_KERNEL_CACHE[diameter]


def _ring_to_spot_gpu(img: np.ndarray, diameter: int) -> "np.ndarray | None":
    """GPU FFT-convolution path for _ring_to_spot. Returns None (triggering
    the CPU fallback) on any failure — GPU acquisition/driver issues must
    never break detection, only slow it down.

    FFT convolution is O(N log N) regardless of kernel size, vs cv2.filter2D's
    direct convolution for this non-separable annular kernel — the GPU win
    grows with diameter (measured 3-5x including PCIe transfer on a GTX 1660
    SUPER at diameter 19-299px). Reflect-pads by the kernel radius before the
    FFT so the border matches cv2.filter2D's default BORDER_REFLECT_101
    (verified to ~1e-5 max difference against the CPU path away from edges,
    and after padding, including edges).
    """
    return _matched_filter_gpu(img, _ring_kernel(diameter),
                               _RING_KERNEL_CACHE_GPU, diameter)


def _matched_filter_gpu(img: np.ndarray, kernel_np: np.ndarray,
                        gpu_cache: dict, key) -> "np.ndarray | None":
    """Shared GPU FFT-convolution core for _ring_to_spot / _dark_disk_to_spot
    (same padding/border semantics as documented on _ring_to_spot_gpu)."""
    try:
        if key not in gpu_cache:
            gpu_cache[key] = _cp.asarray(kernel_np)
        kernel = gpu_cache[key]
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2

        padded = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
        hp, wp = padded.shape
        H, W = hp + kh - 1, wp + kw - 1

        g = _cp.zeros((H, W), dtype=_cp.float32); g[:hp, :wp] = _cp.asarray(padded)
        k = _cp.zeros((H, W), dtype=_cp.float32); k[:kh, :kw] = kernel
        full = _cp.fft.irfft2(_cp.fft.rfft2(g) * _cp.fft.rfft2(k), s=(H, W))

        h, w = img.shape
        out = full[ph + kh // 2 : ph + kh // 2 + h, pw + kw // 2 : pw + kw // 2 + w]
        return _cp.asnumpy(_cp.clip(out, 0, None))
    except Exception:
        return None


def _ring_to_spot(img: np.ndarray, diameter: int) -> np.ndarray:
    """Convert dark-rim / bright-center ring particles into bright spots.

    Applies a zero-mean annular matched filter sized to *diameter*.
    Response at a ring center = H_center - L_rim (positive for dark rims).
    Response on uniform background = 0 (kernel is zero-mean).
    Negative responses are clipped to 0.

    Use this when particles have a bright (or transparent) center surrounded
    by a dark rim — the standard bandpass finds bright-blob peaks and will
    miss such particles, whereas this filter creates a sharp peak at each
    ring centre that _fast_locate can find directly.
    """
    if CUPY_OK:
        gpu_result = _ring_to_spot_gpu(img, diameter)
        if gpu_result is not None:
            return gpu_result
    kernel = _ring_kernel(diameter)
    response = cv2.filter2D(img, cv2.CV_32F, kernel)
    np.maximum(response, 0, out=response)
    return response


_DARK_DISK_KERNEL_CACHE: dict = {}       # diameter -> numpy kernel
_DARK_DISK_KERNEL_CACHE_GPU: dict = {}   # diameter -> cupy kernel (CUPY_OK only)


def _dark_disk_kernel(diameter: int) -> np.ndarray:
    """3-zone zero-mean matched filter for particles that are entirely DARKER
    than the background: dark rim, dimmer-than-background center.

    Zones (r = diameter//2): center disk (ρ ≤ 0.55r, weight −30/T, dim
    center), rim annulus (0.55r < ρ ≤ r, weight −70/T, darkest — T normalizes
    the negative lobe to −1), background annulus (r < ρ ≤ 1.30r, +1/n_bg,
    sums to +1). Response at a particle center ≈ (bg − particle) weighted by
    the expected rim/center contrast ratio; exactly 0 on uniform background
    and immune to linear illumination gradients (zero-mean, radially
    symmetric). Spec chosen by the design debate (advocate/skeptic/mediator),
    validated: recall ≥ 0.93, precision ≥ 0.97 on dense hex synthetic frames
    where invert+bandpass scores ~0.34."""
    if diameter not in _DARK_DISK_KERNEL_CACHE:
        r = max(2, diameter // 2)
        ro = int(round(1.30 * r))
        y, x = np.ogrid[-ro:ro + 1, -ro:ro + 1]
        d2 = (x * x + y * y).astype(np.float32)
        cen = d2 <= (0.55 * r) ** 2
        rim = (d2 > (0.55 * r) ** 2) & (d2 <= float(r * r))
        bg  = (d2 > float(r * r)) & (d2 <= float(ro * ro))
        n_c, n_r, n_b = int(cen.sum()), int(rim.sum()), int(bg.sum())
        kernel = np.zeros((2 * ro + 1, 2 * ro + 1), np.float32)
        if n_c > 0 and n_r > 0 and n_b > 0:
            T = 70.0 * n_r + 30.0 * n_c
            kernel[cen] = -30.0 / T
            kernel[rim] = -70.0 / T
            kernel[bg]  = 1.0 / n_b
        _DARK_DISK_KERNEL_CACHE[diameter] = kernel
    return _DARK_DISK_KERNEL_CACHE[diameter]


def _dark_disk_to_spot(img: np.ndarray, diameter: int) -> np.ndarray:
    """Convert all-dark (dark rim + darker-than-background center) particles
    into bright spots consumable by _fast_locate.

    Pipeline (per the design-debate ruling): matched filter → clip ≥ 0 →
    subtract local box mean (suppresses the interstitial \"dark web\" between
    touching particles in dense packing, whose integrated mass otherwise
    rivals true peaks) → clip ≥ 0 → absolute noise floor (percentile
    thresholds alone flood sparse frames with noise candidates) → 4th power
    (makes disk-integrated mass peak-dominated so _fast_locate's mass-greedy
    separation logic keeps true centers).

    NOTE: this transform must see the frame's real intensity structure — the
    callers bypass invert/gamma/CLAHE/sharpen/bandpass in this mode (only
    illumination flattening and denoise run first). Being a matched filter it
    is sensitive to the Diameter parameter (±20% mis-set collapses recall)."""
    d = max(3, int(diameter) | 1)
    kernel = _dark_disk_kernel(d)
    resp = None
    if CUPY_OK:
        resp = _matched_filter_gpu(img, kernel, _DARK_DISK_KERNEL_CACHE_GPU, d)
    if resp is None:
        f = img if img.dtype == np.float32 else img.astype(np.float32)
        resp = cv2.filter2D(f, cv2.CV_32F, kernel)
        np.maximum(resp, 0, out=resp)
    bgm = cv2.blur(resp, (d + 1, d + 1), borderType=cv2.BORDER_REFLECT)
    resp -= bgm
    np.maximum(resp, 0, out=resp)
    # Noise floor: estimate the background-response scale from a strided
    # sample of positive pixels (peaks occupy only a tiny pixel fraction, so
    # the median tracks the noise web, not the particles).
    pos = resp[::4, ::4]
    pos = pos[pos > 0]
    if pos.size > 64:
        sigma = float(np.median(pos)) / 0.6745
        resp[resp < 4.0 * sigma] = 0.0
    resp *= resp
    resp *= resp          # ^4
    return resp


def _dark_disk_minmass_filter(feats):
    """Adaptive minmass for dark-disk mode (design-debate guardrail): the ^4
    response rescales mass semantics with contrast⁴, so a fixed minmass
    silently breaks under focus/illumination drift. True-particle masses sit
    orders of magnitude above the residual noise after the transform's noise
    floor, so a small fraction of the median candidate mass separates them
    robustly per frame."""
    if feats is None or len(feats) < 8:
        return feats
    med = float(feats["mass"].median())
    if med <= 0:
        return feats
    return feats[feats["mass"] >= 0.05 * med].reset_index(drop=True)


def _ring_minmass_filter(feats):
    """Adaptive minmass for ring mode — the fix for 'ring detection dies when
    the live feed dims'.

    The annular matched-filter response scales with local RING CONTRAST, so a
    FIXED minmass (the user's value, calibrated at one brightness) silently
    drops every particle once the feed is a little dimmer than when it was
    tuned — measured: at 0.5x contrast detection fell from 416 to 25 particles,
    and to 0 below 0.35x. True ring centres sit far above the residual
    background response, so keeping candidates above a small fraction of the
    per-frame median mass is contrast-invariant (422 particles held flat from
    1.0x down to 0.1x contrast in the same test) while matching the full-
    contrast quality (frac6 0.71, psi6 0.74). Mirrors _dark_disk_minmass_filter;
    both matched-filter modes therefore run with minmass=0 at _fast_locate and
    threshold adaptively here instead."""
    if feats is None or len(feats) < 8:
        return feats
    med = float(feats["mass"].median())
    if med <= 0:
        return feats
    return feats[feats["mass"] >= 0.05 * med].reset_index(drop=True)


def _get_gpu_buf(shape: tuple, dtype) -> "cv2.cuda_GpuMat":
    """Return a cached GpuMat for the given shape/dtype, creating one if needed.
    GpuMat objects are reusable: uploading new data into the same object avoids
    a device-memory allocation on every call."""
    global _GPU_BUF_CACHE
    key = (shape, dtype)
    if key not in _GPU_BUF_CACHE:
        _GPU_BUF_CACHE[key] = cv2.cuda_GpuMat()
    return _GPU_BUF_CACHE[key]


def _get_cuda_stream() -> "cv2.cuda.Stream":
    """Return the module-level persistent CUDA stream, creating it on first call."""
    global _CUDA_STREAM
    if _CUDA_STREAM is None:
        _CUDA_STREAM = cv2.cuda.Stream()
    return _CUDA_STREAM


def _bilateral_gpu(img: np.ndarray, strength: float) -> "np.ndarray | None":
    """GPU bilateral filter via cv2.cuda. Returns None (triggering the CPU
    fallback) on any failure — same defensive pattern as _ring_to_spot_gpu.
    Operates on float32 directly (the build's cudaimgproc bilateralFilter
    supports CV_32FC1), so output matches the CPU path's precision exactly.

    Uses a cached GpuMat (keyed by shape/dtype) and a persistent CUDA stream
    to avoid per-call device-memory allocation and stream creation overhead.
    """
    try:
        stream = _get_cuda_stream()
        gmat = _get_gpu_buf(img.shape, img.dtype)
        gmat.upload(img, stream)
        out = cv2.cuda.bilateralFilter(gmat, -1, float(strength), float(strength), stream=stream)
        stream.waitForCompletion()
        return out.download()
    except Exception:
        return None


def _nlm_gpu(img_u8: np.ndarray, strength: float) -> "np.ndarray | None":
    """GPU fastNlMeansDenoising via cv2.cuda. Returns None on any failure.
    Requires uint8 input (matches the existing CPU NLM branch's normalization).

    Uses a cached GpuMat (keyed by shape/dtype) and a persistent CUDA stream
    to avoid per-call device-memory allocation and stream creation overhead.
    """
    try:
        stream = _get_cuda_stream()
        gmat = _get_gpu_buf(img_u8.shape, img_u8.dtype)
        gmat.upload(img_u8, stream)
        out = cv2.cuda.fastNlMeansDenoising(gmat, h=float(strength), stream=stream)
        stream.waitForCompletion()
        return out.download()
    except Exception:
        return None


# Cached cv2.cuda.Filter objects for the chained bandpass+dilate GPU path,
# keyed by the parameters that determine the filter (Gaussian/box/morphology
# filter objects are relatively expensive to construct and are safe to reuse
# across frames as long as their construction parameters — sigma/kernel size/
# structuring element — don't change, which they don't for a fixed set of
# detection params). Separate from _GPU_BUF_CACHE (that one caches GpuMat
# data buffers, not filter objects).
_GPU_BP_FILTER_CACHE: dict = {}   # (lshort, sz) -> (gaussian_filter, box_filter)
_GPU_DILATE_FILTER_CACHE: dict = {}  # se.tobytes()+shape -> morphology_filter


def _gpu_bandpass_dilate_mask(gray_frame: np.ndarray, lshort: float, llong: float,
                               percentile: float, se: np.ndarray,
                               thresh: "float | None" = None
                               ) -> "tuple[np.ndarray, np.ndarray] | None":
    """Chained GPU pipeline: bandpass (Gaussian - box) -> local-maxima dilate
    -> threshold compare, all GPU-resident between stages. Returns
    (bandpassed_frame, local_maxima_mask) as CPU numpy arrays, or None on any
    failure (triggering the caller's CPU fallback) — same defensive pattern as
    _bilateral_gpu/_nlm_gpu. The bandpassed frame is returned too (not just the
    mask) because _fast_locate needs it downstream for patch extraction /
    mass / centroid / eccentricity, exactly as it uses its `proc` argument
    today — this keeps that downstream logic completely unchanged.

    Rationale (see module-level GPU-utilization investigation notes): doing
    bandpass and dilate as separate isolated GPU calls each pays ~2.5ms one-way
    PCIe transfer for a full float32 frame, which roughly cancels out the
    compute saving. Keeping the frame GPU-resident across both stages and only
    downloading the (much smaller) boolean mask — plus the bandpassed frame,
    still needed downstream — avoids repeated transfer cost while saving real
    compute time on the (identical) Gaussian/box/dilate arithmetic.

    Mirrors _fast_bandpass's exact logic (Gaussian(lshort) - Uniform(llong),
    clamped to >= 0) and _fast_locate's exact local-maxima test
    (img == dilate(img, se)) & (img > thresh), using the SAME rectangular
    structuring element _fast_locate uses (from _locate_masks) so the result
    is numerically equivalent to the CPU path (verified to float32-rounding
    precision, ~1e-4 max abs diff, against the CPU reference).

    thresh:
        Percentile threshold. _fast_locate computes its percentile threshold
        from the POST-bandpass image, so callers should pass a thresh already
        computed appropriately, or leave it None to let this function compute
        the threshold itself from the GPU bandpass result (downloaded once,
        as a small strided sample, same convention as _fast_locate).
    """
    try:
        stream = _get_cuda_stream()
        f = gray_frame if gray_frame.dtype == np.float32 else gray_frame.astype(np.float32)
        gmat = _get_gpu_buf(f.shape, f.dtype)
        gmat.upload(f, stream)

        sz = max(1, int(round(float(llong))))
        bp_key = (float(lshort), sz)
        if bp_key not in _GPU_BP_FILTER_CACHE:
            gauss_f = cv2.cuda.createGaussianFilter(
                cv2.CV_32FC1, cv2.CV_32FC1, (0, 0), float(lshort), 0,
                cv2.BORDER_REFLECT, cv2.BORDER_REFLECT)
            box_f = cv2.cuda.createBoxFilter(
                cv2.CV_32FC1, cv2.CV_32FC1, (sz, sz), (-1, -1), cv2.BORDER_REFLECT)
            _GPU_BP_FILTER_CACHE[bp_key] = (gauss_f, box_f)
        gauss_f, box_f = _GPU_BP_FILTER_CACHE[bp_key]

        smooth_g = gauss_f.apply(gmat, stream=stream)
        bg_g = box_f.apply(gmat, stream=stream)
        bp_g = cv2.cuda.subtract(smooth_g, bg_g, stream=stream)
        # clamp-to-zero: max(bp, 0) via compareWithScalar isn't a single op;
        # cv2.cuda.threshold with THRESH_TOZERO does exactly this in-place
        # on the GPU (values <= 0 -> 0, values > 0 unchanged).
        bp_g = cv2.cuda.threshold(bp_g, 0.0, 0.0, cv2.THRESH_TOZERO)[1]

        se_key = (se.shape, se.tobytes())
        if se_key not in _GPU_DILATE_FILTER_CACHE:
            _GPU_DILATE_FILTER_CACHE[se_key] = cv2.cuda.createMorphologyFilter(
                cv2.MORPH_DILATE, cv2.CV_32FC1, se)
        morph_f = _GPU_DILATE_FILTER_CACHE[se_key]
        dil_g = morph_f.apply(bp_g, stream=stream)

        if thresh is None:
            stream.waitForCompletion()
            bp_sample = bp_g.download()[::4, ::4]
            thresh = float(np.percentile(bp_sample, percentile))

        eq_g = cv2.cuda.compare(bp_g, dil_g, cv2.CMP_EQ, stream=stream)
        gt_g = cv2.cuda.threshold(bp_g, float(thresh), 1.0, cv2.THRESH_BINARY)[1]
        stream.waitForCompletion()

        eq = eq_g.download().astype(bool)
        gt = gt_g.download().astype(bool)
        bp_cpu = bp_g.download()
        return bp_cpu, (eq & gt)
    except Exception:
        return None


def _flatten_illumination(img: np.ndarray, sigma_frac: float = 0.15) -> np.ndarray:
    """Flatten large-scale illumination gradients (vignetting, wall shadows,
    uneven backlight) by dividing the frame by a heavily-blurred estimate of
    its own local background.

    The background estimate uses a Gaussian blur with sigma proportional to
    the image size (sigma_frac × min(H, W), clamped to a sane range) — large
    enough to average out individual particles (which occupy only a small
    fraction of the frame) while still tracking slow spatial gradients like a
    dimmer region near a physical wall. This is distinct from the existing
    bandpass filter (_fast_bandpass), whose Llong is tuned to roughly the
    particle spacing (tens of px) specifically to preserve local contrast
    between neighbouring particles — far too short a scale to characterise a
    true wall-shadow/vignetting gradient spanning a large fraction of the FOV,
    and not intended to (raising Llong to that scale would blur the fine
    background texture bandpass is meant to reject).

    Division (rather than subtraction) is used because brightness gradients
    from vignetting/shadowing are multiplicative in nature (attenuation of
    illumination), matching the float32 "brightness value" convention used
    throughout the rest of this pipeline (gamma, CLAHE, bandpass all operate
    on intensity-like values, not zero-mean signals).
    """
    h, w = img.shape[:2]
    sigma = max(8.0, float(sigma_frac) * min(h, w))
    bg = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    mean_bg = float(bg.mean()) + 1e-6
    # Normalise so overall brightness is preserved (divide by bg/mean_bg
    # rather than by bg directly) — keeps the output on the same intensity
    # scale as the input so downstream gamma/CLAHE/minmass thresholds tuned
    # against the un-flattened pipeline remain roughly valid.
    flat = img / (bg / mean_bg + 1e-6)
    return flat.astype(np.float32)


def _preprocess(img, use_bp, lshort, llong, use_clahe, clip,
                 denoise="off", denoise_strength=10.0,
                 gamma=1.0, sharpen=0.0,
                 stop_after_sharpen: bool = False,
                 preprocessed_prefix: "np.ndarray | None" = None,
                 flatten_illum: bool = False,
                 flatten_sigma_frac: float = 0.15,
                 skip_bandpass: bool = False):
    """Shared pre-detection pipeline — used identically for live camera frames
    and recorded-video frames, so every enhancement below applies to both.

    Order: flatten illumination (optional, large-scale background removal) ->
    denoise (remove sensor noise first) -> gamma (brightness curve) ->
    CLAHE (existing local-contrast step) -> unsharp-mask sharpen (edge
    contrast for semi-opaque particles) -> bandpass (final detection input).

    flatten_illum:
        Opt-in local/large-scale illumination correction. Estimates a smooth
        background via a very large-sigma Gaussian blur (sigma = flatten_sigma_frac
        × min(H, W), i.e. a good chunk of the frame — much larger than the
        bandpass Llong scale, which is tuned to particle spacing, not to
        vignetting/wall-shadow scale gradients) and divides the frame by that
        background estimate (renormalised so the mean brightness is preserved).
        This flattens slow spatial gradients (e.g. dimmer particles near a
        wall/vignette) while leaving particle-scale signal intact, since the
        blur sigma is far larger than any single particle. Runs BEFORE
        denoise/gamma/CLAHE/bandpass so every downstream step sees an already
        illumination-flattened frame. Off by default — purely additive.

    Denoise uses OpenCV's built-in bilateral/NLM filters rather than a
    separate denoising package: both are already vectorised C++, well-tested,
    and avoid integration/licensing overhead. Bilateral is fast enough for
    live 40+ fps frames; NLM is stronger but slower, better suited to
    retroactive (recorded-video) enhancement than live preview.

    stop_after_sharpen:
        When True, return immediately after the sharpen step (before CLAHE and
        bandpass). Used by CameraPane._on_frame to produce a shared prefix that
        is both displayed and forwarded to AnalysisWorker, so the expensive
        denoise/gamma/sharpen steps run only once per frame in Compare mode.

    preprocessed_prefix:
        When supplied, skip the denoise/gamma/sharpen steps entirely and start
        from this already-processed intermediate. Used by AnalysisWorker.run()
        to resume the pipeline from the shared prefix produced above, applying
        only CLAHE and bandpass (the steps that differ between display and
        detection paths).

    skip_bandpass:
        When True (and use_bp is also True), every other step runs normally
        but the final bandpass step itself is skipped, returning the post-
        CLAHE/sharpen frame with bandpass NOT applied. Used by the chained-GPU
        detection path (see _fast_locate's `gpu_bandpass` parameter): the
        caller still wants all of _preprocess's other optional steps (denoise,
        gamma, CLAHE, sharpen, illumination flattening) applied CPU-side as
        usual, but wants bandpass to run GPU-resident together with the
        local-maxima dilate inside _fast_locate instead of here, to avoid an
        extra full-frame PCIe round-trip. False (default): identical
        behaviour to before this parameter existed.
    """
    # If a shared prefix was supplied, jump straight to the post-sharpen steps.
    if preprocessed_prefix is not None:
        out = preprocessed_prefix
        mx = None
    else:
        out = img
        mx = None  # cached out.max(), computed lazily and reused across branches

        if flatten_illum:
            out = _flatten_illumination(out, flatten_sigma_frac)

        if denoise == "bilateral":
            gpu_out = _bilateral_gpu(out, denoise_strength) if CV2_CUDA_OK else None
            if gpu_out is not None:
                out = gpu_out
            else:
                out = cv2.bilateralFilter(out, d=9,
                                           sigmaColor=float(denoise_strength),
                                           sigmaSpace=float(denoise_strength))
        elif denoise == "nlm":
            mx = out.max() + 1e-9
            u8 = np.clip(out / mx * 255, 0, 255).astype(np.uint8)
            gpu_u8 = _nlm_gpu(u8, denoise_strength) if CV2_CUDA_OK else None
            u8 = gpu_u8 if gpu_u8 is not None else cv2.fastNlMeansDenoising(u8, h=float(denoise_strength))
            out = (u8.astype(np.float32) / 255.0) * mx
            mx = None  # out changed; invalidate cached max

        if gamma != 1.0:
            if mx is None:
                mx = out.max() + 1e-9
            norm = np.clip(out / mx, 0, 1)
            out  = (np.power(norm, 1.0 / float(gamma)) * mx).astype(np.float32)

        if sharpen > 0:
            blur = cv2.GaussianBlur(out, (7, 7), 1.5)
            out  = out + float(sharpen) * (out - blur)

        # Return the intermediate result (denoise+gamma+sharpen done, no CLAHE/bp)
        # so the caller can share it between the display and analysis paths.
        if stop_after_sharpen:
            return out if out is not img else img.copy()

        mx = None  # mx may be stale relative to sharpen-modified `out`

    if use_clahe:
        if mx is None:
            mx = out.max() + 1e-9
        u8  = np.clip(np.multiply(out, np.float32(255.0 / mx)), 0, 255).astype(np.uint8)
        out = _CLAHE_CACHE.setdefault(
                (float(clip), (8, 8)),
                cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
              ).apply(u8).astype(np.float32)

    if use_bp and not skip_bandpass:
        out = _fast_bandpass(out, lshort, llong)
    elif out is img:
        out = img.copy()
    return out


# Cached disk masks and coordinate grids keyed by radius.
# Allocated once per unique diameter and reused across all frames/threads.
_LOCATE_CACHE: dict = {}

# Sentinel empty DataFrame returned by _fast_locate when no particles are found.
# Callers only read its .empty / len() attributes or copy() before adding columns
# so sharing a single instance is safe.
_EMPTY_FEATS = pd.DataFrame(columns=["x", "y", "mass", "ecc", "signal"])

def _locate_masks(R: int):
    if R not in _LOCATE_CACHE:
        gy, gx = np.mgrid[-R:R + 1, -R:R + 1]
        disk = (gx ** 2 + gy ** 2 <= R ** 2).astype(np.float32)
        d = 2 * R + 1
        # Rectangular SE for the local-maxima dilate: OpenCV implements rect
        # dilate with the separable van Herk/Gil-Werman algorithm (O(N) in
        # kernel size) vs. the elliptical SE's O(N*k^2) generic path — a large
        # win at big diameters. The disk mask above (used for mass/centroid/
        # eccentricity) is unaffected and stays exact; the rectangular SE only
        # widens the candidate-maxima search to the bounding square, and the
        # mass/separation filters downstream reject any spurious corner maxima.
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (d, d))
        _LOCATE_CACHE[R] = (disk,
                            gx[np.newaxis].astype(np.float32),
                            gy[np.newaxis].astype(np.float32),
                            se)
    return _LOCATE_CACHE[R]


def _fast_locate(proc: np.ndarray, diameter: int, separation: int,
                 minmass: float, percentile: float,
                 invert: bool = False,
                 ecc_max: "float | None" = None,
                 reject_size_outliers: bool = False,
                 size_outlier_mad_mult: float = 2.5,
                 search_mask: "np.ndarray | None" = None,
                 gpu_bandpass: "tuple[float, float] | None" = None) -> pd.DataFrame:
    """Vectorised Crocker-Grier particle detection.
    Uses scipy C extensions + numpy BLAS; ~10-20× faster than tp.locate (Python engine).
    Masks are cached per radius so repeated calls at the same diameter avoid re-allocation.

    Optional dirt/debris rejection (both OFF by default — additive, opt-in):

    ecc_max:
        If set, candidates with eccentricity (already computed below from the
        2nd central moments) above this threshold are rejected. Real
        colloidal particles are close to circular; irregular debris/dirt is
        often elongated. None (default) disables this filter, preserving
        existing behaviour where `ecc` is reported but never used to reject.

    reject_size_outliers:
        If True, rejects candidates whose `mass` deviates too far from the
        frame's own median mass, using a robust (median + MAD) statistic
        computed from THIS frame's own detections rather than a fixed
        universal constant — adapts automatically to different videos/
        magnifications/exposure settings instead of requiring per-video
        tuning. Threshold is median_mass ± size_outlier_mad_mult × MAD
        (MAD rescaled by 1.4826 to be a consistent estimator of the standard
        deviation for a Gaussian-like mass distribution). Requires at least
        a handful of detections in the frame to be statistically meaningful;
        skipped (no-op) on frames with too few candidates.

    search_mask:
        Optional bool array, same (h, w) shape as `proc`. If given, restricts
        the STAGE-1 candidate search (local-maxima detection below) to only
        the True region — used by the opt-in crystal-lattice-predicted sparse
        search (TrackingWorker, use_lattice_prediction) to skip the full-frame
        dilate/threshold scan for particles whose position was already
        predicted with high confidence from the previous frame. This affects
        ONLY where candidates are looked for; every candidate that IS found
        (inside or outside the mask) still goes through the exact same
        full-resolution patch-extraction / disk-weighted mass / sub-pixel
        centroid / eccentricity computation below — final precision of
        reported positions is completely unaffected by this parameter.
        None (default) disables it: identical behaviour to before this
        parameter existed.

    gpu_bandpass:
        Optional (lshort, llong) tuple. When given AND cv2 CUDA is available,
        `proc` is treated as the PRE-bandpass frame (post-denoise/gamma/CLAHE/
        sharpen/flatten, but bandpass NOT yet applied — see _preprocess's
        `use_bp` skip path) and the bandpass + local-maxima dilate + threshold
        (stage 1 below) run as a single GPU-resident chain via
        _gpu_bandpass_dilate_mask, uploading `proc` once and downloading only
        the bandpassed frame and boolean mask instead of paying separate
        upload/download costs for isolated bandpass and dilate calls. Falls
        back to the exact existing CPU bandpass+dilate path unchanged if CUDA
        is unavailable or the GPU call fails for any reason (never a
        correctness risk — see _gpu_bandpass_dilate_mask's try/except).
        None (default): `proc` is used as-is (already bandpassed by the
        caller), identical behaviour to before this parameter existed.
    """
    R    = diameter // 2
    d    = 2 * R + 1
    disk, gxf, gyf, se = _locate_masks(R)
    _disk_count = float(disk.sum())

    img = None
    lmax = None
    if gpu_bandpass is not None and CV2_CUDA_OK and not invert:
        # invert is handled on the raw frame before bandpass in the CPU path
        # (img = proc.max() - proc BEFORE bandpass would be wrong ordering —
        # rather than risk getting invert+GPU-bandpass ordering subtly wrong,
        # simply skip the GPU path when invert is requested and fall through
        # to the well-tested CPU path below; invert is an uncommon opt-in flag).
        lshort, llong = gpu_bandpass
        gpu_result = _gpu_bandpass_dilate_mask(proc, lshort, llong, percentile, se)
        if gpu_result is not None:
            img, lmax = gpu_result

    if img is None:
        # ── 1. Local maxima above percentile threshold (CPU path) ──────
        # cv2.dilate with a disk SE is 2× faster than scipy maximum_filter.
        img = (proc.max() - proc) if invert else proc
        if gpu_bandpass is not None:
            lshort, llong = gpu_bandpass
            img = _fast_bandpass(img, lshort, llong)
        # Threshold only needs to be a statistical estimate, not an exact percentile —
        # a strided sample (1/16 of pixels) gives the same value within noise at a
        # fraction of the cost of sorting every pixel in a multi-megapixel frame.
        sample = img[::4, ::4]
        thresh = float(np.percentile(sample, percentile))
        if img.dtype == np.float32:
            # Reuse a per-thread dst for the dilated image — one fewer
            # full-frame allocation per frame; result is bit-identical.
            dil  = cv2.dilate(img, se, dst=_bp_buf("dil", img.shape))
            lmax = (img == dil) & (img > thresh)
        else:
            lmax = (img == cv2.dilate(img, se)) & (img > thresh)

    h, w = img.shape
    if search_mask is not None:
        # Restrict candidate search to the predicted ROI. Correctness note:
        # this can only ever DROP candidates that would have been found
        # outside the mask — callers are responsible for ensuring the masked
        # region only excludes particles they are separately confident about
        # (e.g. the previous frame's well-ordered particles, expected not to
        # have moved far) and/or for merging in a full-frame pass for the
        # complement region. _fast_locate itself makes no such distinction;
        # it just honours the mask it's given.
        lmax &= search_mask
    # flatnonzero+divmod == np.argwhere (same C-order scan, values, dtype) but
    # ~10x faster on multi-megapixel masks (measured 12.9 -> 1.3 ms at 2840^2).
    flat   = np.flatnonzero(lmax)
    yy_, xx_ = np.divmod(flat, w)
    yx     = np.column_stack([yy_, xx_])
    if len(yx) == 0:
        return _EMPTY_FEATS

    # ── 2. Boundary guard ─────────────────────────────────────────────
    mask_in = (yx[:, 0] >= R) & (yx[:, 0] < h - R) & \
              (yx[:, 1] >= R) & (yx[:, 1] < w - R)
    yx = yx[mask_in]
    if len(yx) == 0:
        return _EMPTY_FEATS

    # ── 2b. Cheap upper-bound pre-filter (skip expensive per-candidate work) ─
    # Raw local-maxima counts can run 3x+ the final kept-particle count in
    # dense frames (measured ~3.5x on a synthetic 2840x2840/3000-particle
    # benchmark), and stages 3-4 below (patch extraction + disk-weighted mass)
    # are O(candidate count) — so most of that work is wasted on candidates
    # that will fail the minmass filter anyway.
    #
    # `lmax` (step 1) already guarantees each candidate's centre pixel is a
    # local max over the full (d,d) rectangular dilate window, which is a
    # strict superset of the disk mask used for mass — so no pixel inside the
    # disk can exceed the centre pixel's value. That makes
    # `peak_value * disk_pixel_count` a mathematically guaranteed upper bound
    # on the true disk-weighted mass computed in step 4 (mass = sum of
    # disk-masked pixels, each <= peak). Any candidate whose upper bound is
    # already below minmass is certain to fail the real filter, so it is safe
    # to drop here — this can only ever be conservative (over-count), never
    # under-count, so it cannot cause a false rejection.
    peak_vals = img[yx[:, 0], yx[:, 1]]
    prefilter_keep = (peak_vals * _disk_count) >= minmass
    yx = yx[prefilter_keep]
    if len(yx) == 0:
        return _EMPTY_FEATS

    # ── 3. Patch extraction via stride-trick zero-copy view ───────────
    try:
        patches = _swv(img, (d, d))[yx[:, 0] - R, yx[:, 1] - R].astype(np.float32)
    except Exception:
        patches = np.array([img[y - R:y + R + 1, x - R:x + R + 1]
                            for y, x in yx], dtype=np.float32)

    # ── 4. Disk mask + mass (cached masks: no re-allocation per frame) ─
    masked = patches * disk                          # (N, d, d)
    masses = masked.sum(axis=(1, 2))

    valid = masses >= minmass
    if not valid.any():
        return _EMPTY_FEATS
    yx = yx[valid];  masked = masked[valid];  masses = masses[valid]

    # ── 5. Sub-pixel centroid ─────────────────────────────────────────
    mi   = np.float32(1.0) / np.maximum(masses, np.float32(1e-12))
    cx_r = (gxf * masked).sum((1, 2)) * mi
    cy_r = (gyf * masked).sum((1, 2)) * mi
    cx   = yx[:, 1].astype(np.float32) + cx_r
    cy   = yx[:, 0].astype(np.float32) + cy_r

    # ── 6. Eccentricity from 2nd central moments ──────────────────────
    dx   = gxf - cx_r[:, None, None]
    dy   = gyf - cy_r[:, None, None]
    mi2  = mi                                        # same normalisation
    Ixx  = (dx ** 2 * masked).sum((1, 2)) * mi2
    Iyy  = (dy ** 2 * masked).sum((1, 2)) * mi2
    Ixy  = (dx * dy  * masked).sum((1, 2)) * mi2
    disc = np.sqrt(np.maximum(np.float32(0.),
                              ((Ixx - Iyy) * np.float32(0.5)) ** 2 + Ixy ** 2))
    l1   = (Ixx + Iyy) * np.float32(0.5) + disc
    l2   = (Ixx + Iyy) * np.float32(0.5) - disc
    ecc  = np.sqrt(np.maximum(np.float32(0.),
                              np.float32(1.) - l2 / np.maximum(l1, np.float32(1e-12)))).clip(0, 1)
    signals = img[yx[:, 0], yx[:, 1]]

    # ── 7. Separation filter — greedy, keep brighter of each close pair ─
    # Build KDTree from numpy arrays directly (avoids a DataFrame round-trip).
    # Sort pairs by descending max-mass with numpy argsort (no Python sorted()).
    # The greedy keep/drop loop itself is JIT-compiled (colloid_kernels) since
    # it's an inherently sequential Python loop whose pair count grows with
    # particle density — JIT removes interpreter overhead with identical
    # semantics; falls back to the same loop in pure Python if numba is absent.
    if separation > 1 and len(cx) > 1:
        xy   = np.column_stack([cx, cy])
        tree = cKDTree(xy)
        pairs = tree.query_pairs(r=float(separation), output_type="ndarray")  # (M,2)
        if len(pairs):
            order  = np.argsort(-np.maximum(masses[pairs[:, 0]], masses[pairs[:, 1]]))
            pairs  = pairs[order]
            keep = _greedy_separation_keep(
                np.ascontiguousarray(pairs[:, 0]), np.ascontiguousarray(pairs[:, 1]),
                masses, len(cx))
            cx = cx[keep];  cy = cy[keep];  masses = masses[keep]
            ecc = ecc[keep];  signals = signals[keep]

    # ── 8. Optional dirt/debris rejection (opt-in, off by default) ────────
    if ecc_max is not None and len(cx) > 0:
        keep_ecc = ecc <= float(ecc_max)
        if not keep_ecc.all():
            cx = cx[keep_ecc]; cy = cy[keep_ecc]; masses = masses[keep_ecc]
            ecc = ecc[keep_ecc]; signals = signals[keep_ecc]

    if reject_size_outliers and len(cx) >= 5:
        med = float(np.median(masses))
        mad = float(np.median(np.abs(masses - med))) * 1.4826  # ~std-equivalent
        if mad > 1e-9:
            lo = med - size_outlier_mad_mult * mad
            hi = med + size_outlier_mad_mult * mad
            keep_size = (masses >= lo) & (masses <= hi)
            if not keep_size.all() and keep_size.any():
                cx = cx[keep_size]; cy = cy[keep_size]; masses = masses[keep_size]
                ecc = ecc[keep_size]; signals = signals[keep_size]

    return pd.DataFrame({
        "x": cx.astype(np.float32), "y": cy.astype(np.float32),
        "mass": masses.astype(np.float32), "ecc": ecc.astype(np.float32),
        "signal": signals.astype(np.float32),
    })


# ════════════════════════════════════════════════════════════════════
# Crystal-lattice-predicted sparse search (opt-in, TrackingWorker only)
# ════════════════════════════════════════════════════════════════════
#
# Motivation: for dense, near-crystalline colloidal packings, the vast
# majority of particles sit in locally well-ordered regions where their
# position barely changes frame-to-frame relative to their neighbours.
# The full-frame cv2.dilate + threshold local-maxima search in step 1 of
# _fast_locate is one of the most expensive parts of detection, and most
# of the frame it scans is redundant: well-ordered particles could be
# found by searching only a small disk around a predicted position.
#
# Design note (why this does NOT use the full psi6/Delaunay structural-
# analysis pipeline): that pipeline (analyze_frame/_delaunay) runs as a
# SEPARATE stage, after ALL frames have already been detected (see
# TrackingWorker.run — _run_analysis_thread starts only once pts_by_frame
# is fully populated). Depending on it here would mean detection could
# never benefit from it on a first (and only) pass. Instead we compute a
# cheap, local, in-detection order proxy from the PREVIOUS FRAME'S OWN
# raw detected positions (which detection already produces) — a
# neighbour-count/spacing-regularity heuristic on a cKDTree, not the
# expensive Delaunay+psi6 accumulation. This has no dependency on the
# separate structural-analysis stage or its timing.
#
# Ordering hazard: detection frames are submitted to a ThreadPoolExecutor
# and are not guaranteed to complete in strict order. Rather than forcing
# sequential processing (which would sacrifice the pool's parallelism),
# each frame looks up state for EXACTLY frame (fr-1) in a shared dict; if
# that state is not yet available (e.g. frame fr-1 hasn't completed yet,
# or an ordering hiccup), the worker safely falls back to a normal
# full-frame search for that frame — never a correctness risk, only a
# missed speedup opportunity for that one frame.

def _local_order_proxy(xy: np.ndarray, k: int = 6,
                       cutoff_mult: float = 1.6) -> np.ndarray:
    """Cheap per-particle "well-orderedness" score in [0, 1], analogous in
    spirit to |psi6| but far cheaper: based on the regularity of the
    distances to each particle's k nearest neighbours (low relative spread
    ⇒ locally crystalline; high spread ⇒ defect/boundary/isolated particle).

    Not a physical order parameter — purely a heuristic used to decide
    which particles are safe to predict-and-narrow-search for. Always
    conservative in downstream use: only particles ABOVE a high threshold
    are treated as predictable; everything else gets full-frame search.
    """
    n = len(xy)
    if n < k + 1:
        return np.zeros(n, dtype=np.float32)
    tree = cKDTree(xy)
    dist, _ = tree.query(xy, k=k + 1)          # column 0 is self (dist 0)
    dist = dist[:, 1:]                          # (n, k) neighbour distances
    med = np.median(dist, axis=1)
    # Relative spread of neighbour distances around the local median spacing.
    spread = np.std(dist, axis=1) / np.maximum(med, 1e-9)
    # Map spread -> [0,1] score; spread ~0 (perfect lattice) -> 1,
    # spread >= ~0.5*median (typical for defects/edges) -> ~0.
    score = np.clip(1.0 - spread / 0.5, 0.0, 1.0).astype(np.float32)
    return score


def _predict_roi_mask(shape: "tuple[int,int]", prev_xy: np.ndarray,
                      order_score: np.ndarray, drift: "tuple[float,float]",
                      order_thresh: float, radius: int) -> "np.ndarray | None":
    """Build a boolean mask (True = search here) covering small disks around
    predicted positions for well-ordered particles only. Returns None if too
    few particles qualify (caller should then just do a full-frame search).

    prev_xy:      (N,2) array of previous-frame (x,y) pixel positions.
    order_score:  (N,) local-order proxy for those same particles (from
                  _local_order_proxy on the previous frame's positions).
    drift:        (dx, dy) global median displacement estimate (px) to
                  shift predictions by — accounts for stage/sample drift
                  between frames; (0,0) if unknown.
    order_thresh: only particles with order_score >= this are predicted.
    radius:       disk radius (px) around each prediction, in the same
                  spirit as the ~1.5-2x particle-radius the design calls for.
    """
    h, w = shape
    good = order_score >= order_thresh
    n_good = int(good.sum())
    # Require a substantial majority to be predictable before bothering —
    # otherwise the masked search wouldn't save meaningful time and the
    # bookkeeping/mask-union cost isn't worth it.
    if n_good < 0.5 * len(order_score):
        return None
    px = prev_xy[good, 0] + drift[0]
    py = prev_xy[good, 1] + drift[1]

    mask = np.zeros((h, w), dtype=bool)
    r = int(max(1, radius))
    # Clip predicted centres into frame bounds before drawing disks; points
    # far outside are simply skipped (their disk would be empty anyway).
    xi = np.clip(np.round(px).astype(np.int64), 0, w - 1)
    yi = np.clip(np.round(py).astype(np.int64), 0, h - 1)
    x0 = np.clip(xi - r, 0, w); x1 = np.clip(xi + r + 1, 0, w)
    y0 = np.clip(yi - r, 0, h); y1 = np.clip(yi + r + 1, 0, h)
    # Disk (not just square) mask via cv2.circle is much faster than a
    # Python loop at particle counts in the thousands; draw filled circles
    # directly onto a uint8 canvas then view as bool.
    canvas = mask.view(np.uint8)
    for cx_, cy_ in zip(xi.tolist(), yi.tolist()):
        cv2.circle(canvas, (int(cx_), int(cy_)), r, 1, thickness=-1)
    return mask
