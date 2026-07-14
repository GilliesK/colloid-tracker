# Colloid Tracker — Handoff Document

This document describes the structure and design conventions of `colloid_app.py`
(and its companion module `colloid_kernels.py`) as they currently exist. It is
a factual description of what is present in the code, not an assessment of
it — no recommendations, no quality judgments, no "should" statements.

## 1. Purpose and scope

The application is a PyQt6 desktop GUI for tracking colloidal particles
(microspheres, typically single-digit micrometers) in microscopy video, both
from live camera feeds and from previously recorded video files. It performs:

- Image preprocessing (denoise, bandpass filtering, gamma, CLAHE, sharpening,
  illumination flattening).
- Particle detection (Crocker-Grier-style: local-maxima finding, sub-pixel
  centroid/mass/eccentricity via masked image moments).
- Frame-to-frame particle linking (assigning a persistent identity to each
  detected particle across frames).
- Structural analysis (Delaunay triangulation, the ψ₆ hexatic order
  parameter, grain-boundary/defect detection, cage-relative coordinates,
  drift correction).
- Live camera control (Basler/pypylon and generic OpenCV/DirectShow/MSMF
  cameras), video recording (FFmpeg/NVENC, libx264, or OpenCV `VideoWriter`
  fallback), and lossless TIFF-sequence capture.
- Visualization: annotated video playback, matplotlib-based diagnostic
  plots, structural/defect overlays.

## 2. File organization

The application is a single file, `colloid_app.py` (~6,700 lines), plus one
small companion module, `colloid_kernels.py` (~190 lines), that holds
optional Numba-JIT-compiled versions of three specific hot-path functions.
`colloid_kernels.py` exists as a separate module specifically so that
`colloid_app.py` can remain a single file for everything else; it is
imported inside a `try/except` block, and if the import fails for any
reason (Numba not installed, incompatible CPU, broken install), pure-Python
equivalents defined inline in `colloid_app.py` are used instead. The
call signatures and return semantics of the JIT and non-JIT versions are
identical by design.

Within `colloid_app.py`, code is organized top-to-bottom as:

1. Imports, optional-dependency capability detection (see §4), module-level
   constants and default parameter dictionaries (see §5).
2. `SettingsManager` — JSON-file persistence of parameters/colors/window
   state.
3. Free (module-level) functions implementing the core numerical pipeline:
   preprocessing (`_preprocess`, `_fast_bandpass`, `_flatten_illumination`,
   ring-mode transforms), detection (`_fast_locate`, `_locate_masks`,
   `_local_order_proxy`, `_predict_roi_mask`), linking (`_link_nn`), drift
   correction (`_affine`), structural analysis (`_delaunay`, `_57_pairs`,
   `_grain_bds`, `_dbscan_via_kdtree`, `analyze_frame`), cage coordinates
   (`_cage`), event flagging (`_flag_events`), and drawing/overlay functions
   (`annotate`, `_draw_linking_overlay`, `_draw_grain_overlay`,
   `_draw_ecc_overlay`, `_legend`).
4. Small data-holder classes: `LayerConfig`, `FrameResult`, `FlagEvent`,
   `AnalysisData`, `RecordingBuffer`.
5. `QThread`-based worker classes, each responsible for one long-running or
   concurrent operation: `TrackingWorker` (offline video-file analysis),
   `PreviewWorker` (single-frame preview detection), `SweepWorker`
   (parameter-sweep diameter search), `CameraWorker` (live camera frame
   acquisition), `RecorderWorker` (video encoding/writing),imageSequenceRecorder
   / `TiffSequenceRecorder` (lossless frame-sequence capture),
   `TimelapseWorker`, `AnalysisWorker` (live-camera detection/analysis),
   `GrainSizeWorker`, `GrainConsistencyWorker`.
6. `QWidget`/`QDialog`-based GUI classes: `ParamPanel`, `LayerPanel`,
   `HelpDialog`, `TrackingProgressDialog`, `LogTable`, `DefectPlot`,
   `GrainWarningBar`, `DiagnosticsPanel`, `VideoPane`, `CameraPane`,
   `FrameGLWidget` (OpenGL-backed frame display), small helpers
   (`ColourBtn`, `_SpinWheelFilter`).
7. `MainWindow` — top-level `QMainWindow` assembling all panes/docks/menus
   and owning the top-level worker lifecycle (starting/stopping
   `TrackingWorker` runs, wiring signals to GUI slots).
8. `main()` — application entry point.

## 3. GUI/threading architecture

The GUI follows a producer/consumer pattern built on Qt's signal/slot
mechanism:

- Long-running or blocking work (video decode, detection, camera frame
  acquisition, video encoding) is always done on a `QThread` subclass, never
  on the GUI thread.
- Each worker class defines a fixed set of `pyqtSignal`s used to communicate
  back to the GUI thread — the recurring convention is `progress(int, str)`
  for progress-bar updates, `finished(object)` carrying the final result
  payload, `error(str)` for failure reporting, and, where applicable,
  `frame_done(int, object)` for per-frame incremental results (e.g.
  `TrackingWorker` emits this so the GUI can update a live preview during an
  offline analysis run).
- Workers are constructed with a `params: dict` (the same parameter
  dictionary described in §5) and read whatever keys they need via
  `params.get(key, default)`, rather than a typed configuration object.
- Cancellation is cooperative: workers expose an `abort()` method that sets
  an internal flag (or a `threading.Event`), and their `run()` loops check
  that flag periodically rather than being force-terminated.
- Where a worker itself needs internal concurrency (e.g. `TrackingWorker`
  overlapping video decode with detection, or overlapping frame linking
  with structural analysis), it uses `concurrent.futures.ThreadPoolExecutor`
  and/or a dedicated `threading.Thread` internally, plus `queue.Queue` for
  bounded producer/consumer handoff between a reader thread and a worker
  pool. This second-level concurrency exists inside a single `QThread.run()`
  call, distinct from the QThread-per-worker-class pattern used at the
  outer level.
- Video/image data crosses thread boundaries as plain `numpy` arrays or
  `pandas.DataFrame`s carried inside signal payloads (via `object`-typed
  Qt signal arguments), not as Qt-native image types until the point where a
  frame is actually about to be rendered to a widget.

## 4. Optional-dependency and capability detection

A recurring pattern in the module-level setup (top of `colloid_app.py`,
before any class definitions) is: attempt an import or capability probe
inside `try/except`, set a module-level boolean flag recording whether it
succeeded, and have every downstream consumer check that flag before using
the corresponding feature, falling back to a pure-CPU or pure-Python
equivalent if the flag is `False`. This pattern is applied to:

- `pypylon` (Basler camera SDK) → `PYPYLON_OK`.
- `pyserial` (Sutter Lambda SC shutter controller) → `PYSERIAL_OK`.
- `cupy` (GPU array library, used for FFT-based ring-mode filtering) →
  `CUPY_OK`.
- `cv2.cuda` (CUDA-enabled OpenCV build, used for bilateral/NLM denoise and
  an optional chained bandpass+dilate detection path) → `CV2_CUDA_OK`,
  determined by checking both that the attribute exists on the installed
  `cv2` build and that `cv2.cuda.getCudaEnabledDeviceCount() > 0`.
- `numba` (JIT compiler, used for three specific hot-path kernels in
  `colloid_kernels.py`) → `NUMBA_OK`, checked inside `colloid_kernels.py`
  itself.
- FFmpeg binary presence and NVENC (GPU H.264 encoder) availability →
  `_FFMPEG_PATH`, `_NVENC_OK`. The NVENC probe specifically is deferred to a
  background daemon thread started after the main window is shown (rather
  than run synchronously at import time), because it involves a
  subprocess call; the only consumer of `_NVENC_OK` (`RecorderWorker`)
  does not run until a recording is actually started, well after the probe
  has had time to complete, and falls back to CPU encoding if the probe
  result is still `False`/not-yet-resolved at that point.

Each capability flag is checked at the point of use (e.g. `if CV2_CUDA_OK:`
before calling a `cv2.cuda.*` function), and each GPU/JIT-accelerated
function that has one is paired with a plain CPU/Python fallback that is
functionally equivalent (same inputs, same outputs, same semantics),
reached either by the capability flag being `False` or by the accelerated
path raising an exception at runtime (wrapped in its own `try/except`
returning `None`, with the caller checking for `None` and falling through).

## 5. Parameter representation

All tunable analysis/detection/structural-analysis parameters live in a
single flat dictionary, `_DEFAULT_PARAMS` (module level), with a parallel
`_DEFAULT_LAYER` dictionary for display/overlay settings (colors, line
thicknesses, which overlay categories are shown). Both are plain
`dict[str, ...]` — not dataclasses, not typed config objects.

Consequences of this representation, as implemented:

- Every worker function/class reads parameters via `params.get(key,
  default)` (or `params[key]` where the key is considered always-present),
  so a missing key falls back to a locally-specified default at each call
  site rather than a single centrally-enforced default.
- New parameters are added by (a) adding a key to `_DEFAULT_PARAMS` with
  its default value and an explanatory comment describing what it does and
  why the chosen default was picked, (b) adding a corresponding widget
  (checkbox, spin box, combo box, etc.) inside `ParamPanel`'s construction,
  (c) wiring that widget into `ParamPanel.get()` (which serializes all
  widget states back into a flat dict) and `ParamPanel.apply()` (which
  restores widget states from a loaded dict), and (d) reading the new key
  from `params` at whatever worker call site(s) need it. This four-step
  pattern recurs for every parameter added this session (illumination
  flattening, eccentricity/size-outlier rejection, max-neighbor-distance
  cutoff, lattice-prediction toggle, GPU-bandpass toggle).
- Parameters that represent optional/additive features consistently default
  to values that reproduce prior behavior exactly (e.g. `flatten_illum:
  False`, `use_ecc_filter: False`, `max_neighbor_dist_um: 0.0` meaning "no
  cutoff", `use_lattice_prediction: False`, `use_gpu_bandpass_dilate:
  False`) — enabling a new feature is an explicit opt-in action, not a
  change to the default analysis path.
- `SettingsManager` persists this dictionary (plus the layer dictionary,
  window geometry, and named presets) to a JSON file next to the
  executable/script.

## 6. Detection pipeline structure (`_preprocess` → `_fast_locate`)

`_preprocess(img, use_bp, lshort, llong, use_clahe, clip, denoise=...,
denoise_strength=..., gamma=..., sharpen=..., flatten_illum=...,
flatten_sigma_frac=..., skip_bandpass=...)` applies, in a fixed order:
optional illumination flattening (division by a heavily-blurred background
estimate), optional denoise (bilateral or non-local-means, each with a GPU
path gated by `CV2_CUDA_OK` and a CPU fallback), optional gamma correction,
optional CLAHE, optional sharpening (unsharp mask via Gaussian blur
subtraction), and optional bandpass filtering (`_fast_bandpass`: short-sigma
Gaussian minus long-kernel box filter, clamped to non-negative). Each step
is gated by its own boolean/parameter and is a no-op when disabled. The
`skip_bandpass` parameter lets a caller (specifically `_fast_locate`'s GPU
chained-pipeline path) opt out of `_preprocess`'s own CPU bandpass step when
it intends to perform bandpass itself as part of a GPU-resident chain.

`_fast_locate(proc, diameter, separation, minmass, percentile, invert,
ecc_max=None, reject_size_outliers=False, size_outlier_mad_mult=2.5,
search_mask=None, gpu_bandpass=None)` performs, per frame: local-maxima
detection (rectangular-structuring-element dilation, equality test against
the dilated image, threshold against a percentile computed from a strided
pixel sample), an optional `search_mask` AND-in restricting which pixels are
eligible as candidates (used by the lattice-prediction feature, see below),
an upper-bound pre-filter that rejects candidates whose peak pixel value
times the disk-mask pixel count cannot possibly reach `minmass` (a provably
safe bound, since the local-maxima test already establishes the candidate
pixel dominates every pixel in its window, which is a superset of the disk
mask used for the true mass computation), sub-pixel centroid/mass/
eccentricity computation via masked image moments over small patches
extracted with a zero-copy sliding-window view, optional eccentricity-based
and mass-outlier-based rejection (the latter using a per-frame adaptive
median+MAD threshold rather than a fixed constant), and a minimum-separation
enforcement step (cKDTree pair query plus a greedy keep/drop resolution,
implemented as `greedy_separation_keep` — JIT-compiled when Numba is
available). The `gpu_bandpass` parameter, when provided together with
`CV2_CUDA_OK`, routes the local-maxima-finding stage through
`_gpu_bandpass_dilate_mask`, which performs bandpass, dilation, and
threshold comparison as a single GPU-resident operation chain (uploading
the input frame once and downloading only the resulting boolean mask and
bandpassed frame), falling back to the standard CPU path on any failure.

`_local_order_proxy` and `_predict_roi_mask` implement the
lattice-prediction feature: for a set of particle positions, `
_local_order_proxy` computes a per-particle score based on the relative
spread of distances to each particle's k nearest neighbors (low spread
scored near 1, indicating locally regular/crystalline spacing); `
_predict_roi_mask` uses that score plus a set of previous-frame positions to
build a boolean mask covering small disks around particles above a
score threshold, for use as `_fast_locate`'s `search_mask` argument. This
mechanism is wired into `TrackingWorker.run()`'s frame-processing closure
(gated by the `use_lattice_prediction` parameter): each frame's own raw
detections are used to compute a local-order proxy and stored (keyed by
frame number, under a lock, since frames may be processed out of submission
order by the worker thread pool) for lookup by whichever frame is processed
next; a periodic cadence (every 15 frames, and any frame lacking usable
prior-frame state) always performs the unrestricted full-frame search
instead.

## 7. Linking (`_link_nn`)

`_link_nn(feats_df, search_range, memory, adaptive_stop=None,
adaptive_step=None)` assigns persistent particle IDs across frames using a
frame-by-frame nearest-neighbor matching scheme, as an alternative to
`trackpy.link_df`. Per frame, it builds a `cKDTree` over the current
frame's detections, queries each currently-tracked ("live") particle's last
known position for its k nearest candidates within `search_range`, and
resolves the candidate list via a greedy nearest-distance-first assignment
(implemented in `link_nn_match` in `colloid_kernels.py`, JIT-compiled when
available) that skips any live particle or detection already claimed by a
closer pair. Live particles unmatched in a given frame have a
`frames_missing` counter incremented and are dropped from tracking once
that counter exceeds `memory`; unmatched detections spawn new particle IDs.
Per-frame detections are looked up via a `groupby("frame")` built once
before the per-frame loop begins, rather than re-filtering the full
DataFrame inside the loop. The `adaptive_stop`/`adaptive_step` parameters
are accepted for call-site compatibility with the alternative `trackpy`
code path but are not used internally, since this algorithm has no
combinatorial subnet-solving step for them to bound. `TrackingWorker`
selects between this linker and `trackpy.link_df` via a `linker` parameter
(`"nn"` default, `"trackpy"` alternative), with the `trackpy` path retaining
its original `SubnetOversize`-triggered fallback to a recursive link
strategy.

## 8. Structural analysis

`_delaunay(pts, max_dist_px=None)` computes a Delaunay triangulation over a
frame's particle positions, extracts the unique undirected edge set, and —
if `max_dist_px` is given — filters out edges longer than that distance
before computing the ψ₆ hexatic order parameter (via `psi6_accumulate`,
JIT-compiled when available) and per-particle coordination number.
`_57_pairs` identifies particles with 5-fold/7-fold coordination (as
opposed to the 6-fold coordination expected in a perfect hexagonal lattice)
and pairs them by proximity. `_grain_bds` identifies grain boundaries by
clustering 5-7 pair midpoints; its clustering step, `_dbscan_via_kdtree`, is
a from-scratch reimplementation of DBSCAN's core/border/noise-point
semantics (including the `min_samples` parameter) built from
`scipy.spatial.cKDTree.query_pairs` (for radius-based neighbor pairs) and
`scipy.sparse.csgraph.connected_components` (for core-point cluster
assignment over core-to-core edges), with border-point assignment tie-broken
by ascending core-point index to match `sklearn.cluster.DBSCAN`'s internal
scan order on ambiguous cases. `analyze_frame(pts, dp)` is the per-frame
entry point tying these together into a `FrameResult`. `_affine` performs
sequential frame-to-frame drift correction (a genuine sequential recurrence,
since each frame's correction depends on the previous frame's already-
corrected positions, implemented with pre-extracted numpy arrays and
`np.intersect1d`/`searchsorted`-based particle matching rather than
per-frame `pandas.merge`/`.loc` operations). `_cage` computes cage-relative
coordinates via a per-frame `cKDTree` k-nearest-neighbor query (parallelized
across frames via `ThreadPoolExecutor`), with the same optional
`max_neighbor_dist_um` cutoff available to `_delaunay`. `_flag_events`
identifies particles that appear or disappear away from the frame edge
(potential detection artifacts) using a single vectorized groupby-aggregate
rather than a per-particle loop.

In `TrackingWorker.run()`, structural analysis (per-frame, parallelized
across an 8-worker `ThreadPoolExecutor`) is started as soon as detection
completes and raw per-frame positions (`pts_by_frame`) are available, on a
background `threading.Thread`, running concurrently with linking
(`_link_nn`/`tp.link_df`) and `_affine` on the main thread — these two
stages operate on independent `AnalysisData` attributes
(`data.frame_results`/`data.n_pairs_series`/`data.n_lagb_series` written by
the analysis thread; `data.tracks`/`data.flag_events` written by the main
thread) with no shared mutable state between them. `_cage`'s own
`ThreadPoolExecutor` is started only after the analysis thread has been
joined, so the two 8-worker pools are never active at the same time.

## 9. Video I/O and recording

`RecorderWorker` drains a `RecordingBuffer` (a thread-safe circular RAM
buffer with high/low watermark hysteresis, used to decouple camera-frame
arrival from encoder throughput) to a video file, preferring an FFmpeg
subprocess pipeline with `h264_nvenc` (GPU) as the primary codec, falling
back to `libx264` (CPU) if NVENC is unavailable or fails to initialize
(detected via a short post-launch health check reading the FFmpeg process's
own exit status and stderr), and falling back further to
`cv2.VideoWriter`/`mp4v` if FFmpeg itself is not present on the system. If
NVENC fails partway through an already-started recording (rather than at
startup), the recording is stopped with an error identifying how many
frames were captured, and the partial file is left in place unmodified,
rather than attempting to start a second encoder process against the same
output path (which would silently truncate/overwrite the already-encoded
portion). `TiffSequenceRecorder` provides an independent, lossless
frame-sequence recording path (used for 12-bit/Mono12 camera output),
running alongside `RecorderWorker` rather than replacing it, gated by its
own checkbox.

`CameraWorker` supports both Basler cameras (via `pypylon`, gated by
`PYPYLON_OK`) and generic OpenCV-backed cameras, each in its own grab-loop
method. File-based `cv2.VideoCapture` calls throughout the codebase
explicitly specify `cv2.CAP_FFMPEG` as the backend.

## 10. Live-camera analysis path vs. offline analysis path

Live-camera preprocessing/detection (`CameraPane`, `AnalysisWorker`) and
offline video-file analysis (`VideoPane`, `TrackingWorker`) are separate
code paths that both call the same shared `_preprocess`/`_fast_locate`
functions, but are wired up independently — each constructs its own worker
instance, its own parameter snapshot, and its own signal connections.
`CameraPane._on_frame` (the live per-frame handler) and `AnalysisWorker`
(the live "Compare" detection thread) share a common preprocessing prefix:
`_preprocess` accepts a `stop_after_sharpen` flag to return the result after
denoise/gamma/sharpen but before CLAHE/bandpass, and a `preprocessed_prefix`
parameter to accept and continue from that intermediate result — used so
the display path and the analysis path do not each independently repeat
the denoise step when both are active simultaneously.

## 11. Extension points as currently structured

Based on the patterns above, the file's own conventions for adding
capability are:

- **New optional native/GPU/JIT dependency**: add a `try/except` import
  block near the top of `colloid_app.py` (or, for JIT kernels specifically,
  inside `colloid_kernels.py`) setting a module-level `*_OK` boolean; gate
  all use sites on that flag; provide a functionally-equivalent fallback.
- **New tunable parameter**: add a key/default/comment to `_DEFAULT_PARAMS`
  (or `_DEFAULT_LAYER` for display settings); add a widget in `ParamPanel`
  (or `LayerPanel`); wire it into that panel's `get()`/`apply()` methods;
  read it via `params.get(key, default)` at the relevant worker call
  site(s).
- **New long-running operation**: add a `QThread` subclass with `progress`/
  `finished`/`error` signals (and `frame_done` if incremental per-frame
  results are needed), constructed with a `params` dict, exposing an
  `abort()` method checked cooperatively inside `run()`.
- **New per-frame numerical hot-path function**: implemented as a
  module-level free function operating on plain numpy arrays/DataFrames;
  if profiling identifies it as a bottleneck with a tight, GIL-releasing-
  incompatible inner loop, a JIT-compiled version may be added to
  `colloid_kernels.py` with a matching pure-Python fallback of identical
  signature/semantics defined inline in `colloid_app.py`'s import
  `try/except` block.

## 12. Testing and validation approach as currently practiced

The codebase does not contain an automated test suite. Validation of
individual changes has been performed via standalone scripts (not part of
the shipped application) that import the relevant functions from
`colloid_app.py`/`colloid_kernels.py`, construct synthetic input data (e.g.
a generated hexagonal-lattice particle arrangement with configurable noise/
density/motion), and compare a new code path's output against either a
prior/reference implementation or an independent library (e.g. comparing
`_dbscan_via_kdtree` against `sklearn.cluster.DBSCAN` on the same input) for
equivalence, alongside wall-clock timing measurements. `python -c "import
ast; ast.parse(...)"` is used as a syntax-validity check after edits.
