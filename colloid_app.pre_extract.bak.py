#!/usr/bin/env python3
"""
colloid_app.py  —  Colloid Tracker GUI  (v2)

New in v2:
  • QDockWidget panes — float/pop-out, hide, or resize every panel
  • View menu  — show/hide any dock; Reset Layout
  • Persistent settings  — JSON file saves all parameters, colours, window state
  • USB 3.0 / GigE camera detection  — enumerates Basler + OpenCV cameras
  • Auto-install notification for optional dependencies (source mode)
  • Savable / loadable named presets

Run:
    python colloid_app.py [video.mp4]

Build Windows .exe:
    See build_windows.bat  (run on a Windows machine)
"""

from __future__ import annotations

import concurrent.futures as _cf
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import queue as _queue
import threading
import time
import warnings
import traceback as _tb
from collections import deque as _deque
from dataclasses import dataclass, asdict
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
from scipy.spatial import Delaunay, cKDTree, QhullError
from scipy.sparse import csr_matrix as _csr
from scipy.sparse.csgraph import connected_components as _cc
from scipy.ndimage import maximum_filter as _mf
from numpy.lib.stride_tricks import sliding_window_view as _swv
from sklearn.cluster import DBSCAN
import trackpy as tp
import pims
from pims import pipeline

try:
    from colloid_kernels import greedy_separation_keep as _greedy_separation_keep
    from colloid_kernels import psi6_accumulate as _psi6_accumulate
    from colloid_kernels import link_nn_match as _link_nn_match
except Exception:
    def _greedy_separation_keep(pairs_i, pairs_j, masses, n):
        keep = np.ones(n, dtype=bool)
        for k in range(len(pairs_i)):
            i = pairs_i[k]; j = pairs_j[k]
            if keep[i] and keep[j]:
                keep[i if masses[i] < masses[j] else j] = False
        return keep

    def _psi6_accumulate(i_all, j_all, dx, dy, N):
        """Pure-Python fallback when colloid_kernels is unavailable."""
        psi6 = np.zeros(N, dtype=complex)
        np.add.at(psi6, i_all, np.exp(6j * np.arctan2(dy, dx)))
        coord = np.bincount(i_all, minlength=N)
        return psi6, coord

    def _link_nn_match(dists, idxs, n_live, n_det, search_range):
        """Pure-Python fallback when colloid_kernels is unavailable — identical
        semantics to _link_nn's original candidate-gather + greedy-assign loops."""
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

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDockWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QProgressBar, QColorDialog, QFileDialog,
    QTableWidget, QTableWidgetItem, QTabWidget, QMessageBox,
    QSizePolicy, QHeaderView, QStatusBar, QToolBar, QComboBox,
    QDialog, QDialogButtonBox, QScrollArea, QFrame, QMenu, QTextEdit,
    QInputDialog, QToolButton,
)
from PyQt6.QtCore import (
    Qt, QObject, QThread, pyqtSignal, QTimer, QSettings, QByteArray,
    QSize, QRect, QEvent,
)
from PyQt6.QtGui import QImage, QPixmap, QColor, QAction, QIcon, QFont, QPainter

try:
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget as _QOpenGLWidget
    _GL_WIDGET_BASE = _QOpenGLWidget
except ImportError:
    _GL_WIDGET_BASE = QWidget  # type: ignore[assignment]


# ── mouse-wheel-safe widgets ────────────────────────────────────────────────
# Classic Qt bug: hovering over a QSpinBox/QDoubleSpinBox/QComboBox/QSlider
# while scrolling the mouse wheel (e.g. while scrolling a whole parameter
# panel) silently changes that widget's value instead of scrolling the panel.
# These subclasses ignore wheel events unless the widget already has keyboard
# focus (i.e. the user deliberately clicked/tabbed into it), so an incidental
# scroll-past never mutates a value. Used app-wide in place of the raw Qt
# widgets — see the NoScroll* instantiations throughout this file.
class NoScrollSpinBox(QSpinBox):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
        else:
            super().wheelEvent(event)


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
        else:
            super().wheelEvent(event)


class NoScrollComboBox(QComboBox):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
        else:
            super().wheelEvent(event)


class NoScrollSlider(QSlider):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
        else:
            super().wheelEvent(event)


# ── collapsible section (Task: collapsible parameter groups) ───────────────
class CollapsibleGroupBox(QWidget):
    """
    Drop-in replacement for `QGroupBox(title)` that adds a clickable header
    with a disclosure arrow, letting the user collapse/expand the whole
    section. Usage mirrors QGroupBox: construct with a title, then use
    `.layout()`-style access via the returned content widget — but for a
    mechanical swap of existing `QGroupBox`/`QGridLayout(box)` call sites,
    prefer the `content_layout` attribute (already installed as the group's
    QVBoxLayout) and add a QGridLayout/QVBoxLayout into it, OR just take the
    `.body` QWidget and construct your own layout on it exactly as before on
    the old QGroupBox.

    All instances are tracked in `_ALL_SECTIONS` (title -> instance) so the
    app can persist collapsed/expanded state without each call site having
    to manage that separately.
    """
    _ALL_SECTIONS: dict = {}

    def __init__(self, title: str, parent=None, expanded: bool = True):
        super().__init__(parent)
        self._title = title

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header: a checkable QToolButton with an arrow, native/DPI-correct.
        # Styled to match the app's QGroupBox title colour/weight (see the
        # dark-theme stylesheet's "QGroupBox::title" rule) so a collapsible
        # section reads as the same visual category as a plain group box.
        self.toggle_btn = QToolButton(self)
        self.toggle_btn.setObjectName("CollapsibleHeader")
        self.toggle_btn.setText(title)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(expanded)
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.setStyleSheet(
            "#CollapsibleHeader {border:none;font-weight:bold;color:#88aacc;"
            "padding:2px 0px;background:transparent;}"
            "#CollapsibleHeader:hover {color:#a0c0e6;}"
        )
        self.toggle_btn.clicked.connect(self._on_toggled)
        outer.addWidget(self.toggle_btn)

        # Body: a bordered container that visually mimics QGroupBox's frame,
        # holding whatever layout/children the caller adds — this is what
        # gets hidden/shown, never the header, so the section can always be
        # re-expanded.
        self.body = QFrame(self)
        self.body.setFrameShape(QFrame.Shape.StyledPanel)
        self.body.setObjectName("CollapsibleBody")
        self.body.setStyleSheet(
            "#CollapsibleBody {border:1px solid #3a3a3a; border-radius:5px;"
            " background:transparent;}"
        )
        self.body.setVisible(expanded)
        outer.addWidget(self.body)

        CollapsibleGroupBox._ALL_SECTIONS[title] = self

    def _on_toggled(self, checked: bool):
        self.body.setVisible(checked)
        self.toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    # ── convenience API mirroring QGroupBox enough for mechanical swaps ──
    def setLayout(self, layout):
        """Install `layout` onto the body frame (mirrors QGroupBox(...))."""
        self.body.setLayout(layout)

    def is_expanded(self) -> bool:
        return self.toggle_btn.isChecked()

    def set_expanded(self, expanded: bool):
        self.toggle_btn.setChecked(expanded)
        self._on_toggled(expanded)

    @classmethod
    def apply_saved_states(cls, states: dict):
        """Apply a {title: bool} dict (e.g. loaded from settings) to any
        currently-registered sections that match by title."""
        for title, expanded in states.items():
            sec = cls._ALL_SECTIONS.get(title)
            if sec is not None:
                sec.set_expanded(bool(expanded))

    @classmethod
    def collect_states(cls) -> dict:
        """Return {title: expanded_bool} for all currently-registered sections."""
        return {title: sec.is_expanded() for title, sec in cls._ALL_SECTIONS.items()}


tp.quiet()

# ── optional pypylon ────────────────────────────────────────────────────────
try:
    from pypylon import pylon as _pylon
    PYPYLON_OK = True
except ImportError:
    PYPYLON_OK = False

# ── optional pyserial (Sutter Lambda SC shutter controller) ────────────────
try:
    import serial as _serial
    import serial.tools.list_ports as _serial_list_ports
    PYSERIAL_OK = True
except ImportError:
    PYSERIAL_OK = False

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

# ── FFmpeg / NVENC detection ─────────────────────────────────────────────────
# RecorderWorker prefers h264_nvenc (GPU) → libx264 (CPU, ultrafast) → mp4v
# (cv2 fallback when ffmpeg is absent). Detection runs once at import time.
_FFMPEG_PATH: "str | None" = shutil.which("ffmpeg")

def _ffmpeg_has_nvenc() -> bool:
    if not _FFMPEG_PATH:
        return False
    try:
        out = subprocess.check_output(
            [_FFMPEG_PATH, "-hide_banner", "-encoders"],
            stderr=subprocess.STDOUT, timeout=8,
        ).decode("utf-8", errors="ignore")
        return "h264_nvenc" in out
    except Exception:
        return False

# Starts False/"unknown" and is updated once _start_nvenc_probe()'s background
# thread completes (kicked off after the main window is shown — see main()).
# This used to run synchronously at import time, blocking app launch on a
# subprocess call before the Qt event loop even existed. The only consumer,
# RecorderWorker._drain_ffmpeg, only runs once a recording actually starts
# (well after the window is up), and safely falls back to libx264 if the
# probe hasn't finished yet — so an unknown/False value here is harmless.
_NVENC_OK: bool = False

def _start_nvenc_probe():
    """Run the (blocking) NVENC capability probe on a daemon thread so it
    never delays app launch. Updates the module-level _NVENC_OK in place."""
    def _probe():
        global _NVENC_OK
        _NVENC_OK = _ffmpeg_has_nvenc()
    threading.Thread(target=_probe, daemon=True).start()

# Module-level CLAHE object cache keyed by (clipLimit, tileGridSize).
# cv2.createCLAHE is called on every preprocessed frame when CLAHE is enabled;
# the CLAHE object itself is stateless between apply() calls so it is safe to
# reuse across frames and threads.
_CLAHE_CACHE: dict = {}

IS_FROZEN = getattr(sys, "frozen", False)   # True when running as PyInstaller exe

APP_NAME    = "ColloidTracker"
APP_VERSION = "2.0"

# Settings file lives next to the exe (portable) or in the source directory
if IS_FROZEN:
    _SETTINGS_DIR = Path(sys.executable).parent
else:
    _SETTINGS_DIR = Path(__file__).parent
SETTINGS_PATH = _SETTINGS_DIR / "colloid_settings.json"


# ════════════════════════════════════════════════════════════════════
# Settings manager
# ════════════════════════════════════════════════════════════════════

_DEFAULT_PARAMS = {
    "diameter": 19, "separation": 19, "minmass": 3000, "percentile": 80,
    "use_bandpass": True, "lshort": 1, "llong": 53,
    "invert": False, "ring_mode": False,
    # Illumination flattening (opt-in): divide by a heavily-blurred background
    # estimate before denoise/gamma/CLAHE/bandpass, to correct large-scale
    # gradients (vignetting, dimmer regions near a physical wall) that global
    # bandpass/CLAHE/gamma don't address. Off by default — additive quality
    # improvement, not a replacement for the existing bandpass.
    "flatten_illum": False, "flatten_sigma_frac": 0.15,
    # Dirt/debris rejection (opt-in): reject candidates that are too
    # elongated (ecc_max) or whose mass is a frame-relative outlier
    # (reject_size_outliers, adaptive via median+MAD so it works across
    # different videos/magnifications without a fixed universal constant).
    "ecc_max": 0.8, "use_ecc_filter": False,
    "reject_size_outliers": False, "size_outlier_mad_mult": 2.5,
    "max_step_um": 3.0, "memory": 3, "min_len": 15,
    "edge_px": 38, "appear_thresh": 5,
    "pair_dist_px": 48, "lagb_eps_px": 76, "lagb_min_n": 3,
    "lagb_aspect": 2.0, "lagb_angle_deg": 15.0,
    # Structural-analysis neighbour-distance cutoff (opt-in, µm). When > 0,
    # Delaunay/k-NN "neighbour" edges longer than this are dropped before
    # psi6/coordination/grain-boundary/cage calculations, avoiding the
    # well-known Delaunay/k-NN artifact of long spurious edges at cluster
    # boundaries or in dilute regions. 0 = disabled (no cutoff, legacy behaviour).
    "max_neighbor_dist_um": 0.0,
    # Crystal-lattice-predicted sparse search (opt-in, offline TrackingWorker
    # only — see _fast_locate's `search_mask` param and TrackingWorker.run).
    # When True, well-ordered particles (by a cheap in-detection local-order
    # proxy, not the full psi6 pipeline) have their candidate search
    # restricted to a small disk around a position predicted from the
    # previous frame, instead of scanning the full frame. Off by default:
    # purely a speed optimisation, never required for correct results, and
    # every particle's FINAL centroid/mass/eccentricity is still computed
    # from full-resolution pixel data exactly as before — this only narrows
    # WHERE candidates are searched for, never the precision of what's
    # measured. A full-frame reconciliation pass runs periodically (and for
    # any frame where the previous frame's state isn't available) as a
    # safety net against missed particles / drift accumulation.
    "use_lattice_prediction": False,
    "px_um": 0.11, "dt": 0.0, "start_fr": 0, "end_fr": 500,
    "chunk": 50,
    # Raised from 256: a Basler ace2 Pro at 2840x2840/44fps fills a 256MB
    # buffer in well under a second, triggering premature recording downscale.
    "rec_buf_mb": 1024,
    # Image enhancement — applied identically to live camera and recorded-video
    # analysis (both paths share _preprocess), for low-contrast/semi-opaque particles.
    "gamma": 1.0, "sharpen_amount": 0.0,
    "denoise_method": "off", "denoise_strength": 10.0,
}

_DEFAULT_LAYER = {
    "show_6fold": True,  "show_5fold": True,  "show_7fold": True,
    "show_bonds": False, "show_pairs": True,  "show_lagb": True,
    "show_hagb": True,   "show_flags": True,  "show_hud": True,
    "col_6fold": [160,160,160], "col_5fold": [220,60,20],
    "col_7fold": [20,60,220],   "col_bonds": [100,100,100],
    "col_pairs": [220,20,220],  "col_lagb": [0,220,220],
    "col_hagb": [200,180,0],    "col_flags": [0,220,80],
    "radius": 6, "bond_thick": 1, "pair_thick": 2,
    "gb_thick": 3, "flag_thick": 3, "flag_frames": 8,
}

# Wong (2011) colorblind-safe palette — no red/green dependency.
# Stored as BGR tuples (OpenCV convention).
_CB_COLORS = {
    "col_6fold": (200, 200, 200),   # light grey  (unchanged)
    "col_5fold": (  0, 159, 230),   # orange      #E69F00
    "col_7fold": (233, 180,  86),   # sky blue    #56B4E9
    "col_bonds": (100, 100, 100),   # dark grey   (unchanged)
    "col_pairs": (167, 121, 204),   # mauve       #CC79A7
    "col_lagb":  (115, 158,   0),   # teal green  #009E73
    "col_hagb":  ( 66, 228, 240),   # yellow      #F0E442
    "col_flags": (178, 114,   0),   # blue        #0072B2
}


class SettingsManager:
    """
    Loads/saves all app settings to a JSON file beside the executable.
    Also persists window geometry/state via QSettings.
    """
    def __init__(self):
        self._path = SETTINGS_PATH
        self._data: dict = {}
        self.load()

    def load(self):
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            print(f"[settings] Could not save: {e}")

    def get_params(self) -> dict:
        return {**_DEFAULT_PARAMS, **self._data.get("params", {})}

    def set_params(self, p: dict):
        self._data["params"] = p

    def get_layer(self) -> dict:
        return {**_DEFAULT_LAYER, **self._data.get("layer", {})}

    def set_layer(self, l: dict):
        self._data["layer"] = l

    def get_last_video(self) -> str:
        return self._data.get("last_video", "")

    def set_last_video(self, p: str):
        self._data["last_video"] = p

    def get_export_dir(self) -> str:
        return self._data.get("export_dir", "")

    def set_export_dir(self, d: str):
        self._data["export_dir"] = d

    def get_presets(self) -> dict:
        return self._data.get("presets", {})

    def save_preset(self, name: str, params: dict, layer: dict):
        self._data.setdefault("presets", {})[name] = {"params": params, "layer": layer}

    def delete_preset(self, name: str):
        self._data.get("presets", {}).pop(name, None)

    def save_window(self, geometry: QByteArray, state: QByteArray):
        self._data["window_geometry"] = geometry.toBase64().data().decode()
        self._data["window_state"]    = state.toBase64().data().decode()

    def get_window(self):
        g = self._data.get("window_geometry")
        s = self._data.get("window_state")
        if g and s:
            return (QByteArray.fromBase64(g.encode()),
                    QByteArray.fromBase64(s.encode()))
        return None, None

    def get_section_states(self) -> dict:
        """Collapsed/expanded state of CollapsibleGroupBox sections, keyed by title."""
        return self._data.get("section_states", {})

    def set_section_states(self, s: dict):
        self._data["section_states"] = s


# ════════════════════════════════════════════════════════════════════
# Dependency check (source-mode only)
# ════════════════════════════════════════════════════════════════════

class DepBar(QWidget):
    """
    Non-blocking notification bar shown when optional packages are missing.
    Only used in source mode (not in frozen exe).
    """
    def __init__(self, packages: list[str], parent=None):
        super().__init__(parent)
        self._packages = packages
        self.setStyleSheet("background:#3a2a00;border-bottom:1px solid #aa6600;")
        row = QHBoxLayout(self); row.setContentsMargins(8,4,8,4)

        names = ", ".join(packages)
        row.addWidget(QLabel(f"⚠  Optional packages not found: {names}"))
        row.addStretch()

        if not IS_FROZEN:
            btn = QPushButton("Install now")
            btn.setStyleSheet("background:#aa6600;color:white;font-weight:bold;")
            btn.clicked.connect(self._install)
            row.addWidget(btn)

        close_btn = QPushButton("✕"); close_btn.setFixedWidth(24)
        close_btn.clicked.connect(self.hide)
        row.addWidget(close_btn)

    def _install(self):
        self.setEnabled(False)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install"] + self._packages
            )
            QMessageBox.information(
                self.parent(), "Installed",
                f"Installed: {', '.join(self._packages)}\n\nPlease restart the app."
            )
        except Exception as e:
            QMessageBox.warning(self.parent(), "Install failed", str(e))
        self.setEnabled(True)


# ════════════════════════════════════════════════════════════════════
# Camera detection
# ════════════════════════════════════════════════════════════════════

def detect_cameras() -> list[dict]:
    """
    Enumerate available cameras.
    Returns list of dicts: {type, index/serial, label, device_info}.
    """
    cameras = []

    # 1. Basler cameras via pypylon (USB3 Vision + GigE Vision)
    if PYPYLON_OK:
        try:
            tl  = _pylon.TlFactory.GetInstance()
            devs = tl.EnumerateDevices()
            for i, dev in enumerate(devs):
                model  = dev.GetModelName()
                serial = dev.GetSerialNumber()
                iface  = dev.GetDeviceClass()     # e.g. "BaslerUsb", "BaslerGigE"
                label  = f"Basler {iface.replace('Basler','')} — {model}  [{serial}]"
                cameras.append({
                    "type": "basler", "index": i,
                    "serial": serial, "model": model,
                    "interface": iface, "label": label,
                })
        except Exception as e:
            print(f"[camera] pypylon enumerate error: {e}")

    # 2. Generic cameras via OpenCV (handles most USB cameras)
    # Suppress OpenCV's verbose stderr during probing
    devnull = open(os.devnull, "w")
    old_stderr_fd = os.dup(2)
    os.dup2(devnull.fileno(), 2)
    try:
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        for idx in range(4):   # probe 0–3 only; rare to have >3 USB cameras
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    # skip cameras already found via pypylon (same device)
                    already = any(c["type"] == "opencv" and c["index"] == idx
                                  for c in cameras)
                    if not already:
                        cameras.append({
                            "type": "opencv", "index": idx,
                            "label": f"USB Camera {idx}  ({w}×{h})",
                        })
            except Exception:
                pass
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        devnull.close()

    return cameras


# ════════════════════════════════════════════════════════════════════
# RAM recording buffer
# ════════════════════════════════════════════════════════════════════

class RecordingBuffer:
    """
    Thread-safe circular frame buffer for live-camera recording.

    Behaviour
    ---------
    * put()  is called from the Qt GUI thread (frame arrives via signal).
    * get()  is called from RecorderWorker (background thread).
    * When fill >= HIGH_WATERMARK the caller should halve the frame
      resolution before put()-ing to relieve memory pressure.
    * When fill drops back to <= LOW_WATERMARK the caller restores
      full resolution.
    """
    HIGH_WATERMARK = 0.72
    LOW_WATERMARK  = 0.35

    def __init__(self, max_mb: int = 256):
        self._max   = int(max_mb) * 1024 * 1024
        self._buf   = _deque()
        self._bytes = 0
        self._lock  = threading.Lock()
        # Wakes the draining RecorderWorker immediately on put() instead of
        # making it poll on a fixed msleep interval (same pattern as
        # AnalysisWorker's _event) — RecorderWorker.run() waits on this.
        self._event = threading.Event()

    @property
    def fill(self) -> float:
        return self._bytes / self._max if self._max else 0.

    def put(self, frame: np.ndarray) -> bool:
        """Return True if the frame was accepted, False if buffer is full."""
        nb = int(frame.nbytes)
        with self._lock:
            if self._bytes + nb > self._max:
                return False
            self._buf.append((frame, nb))
            self._bytes += nb
        self._event.set()
        return True

    def wait(self, timeout: float = 0.1):
        """Block until a frame is put() or the timeout elapses, then clear
        the wake flag. Used by RecorderWorker to avoid busy-polling."""
        self._event.wait(timeout=timeout)
        self._event.clear()

    def get(self) -> "np.ndarray | None":
        with self._lock:
            if not self._buf:
                return None
            frame, nb = self._buf.popleft()
            self._bytes -= nb
            return frame

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def clear(self):
        with self._lock:
            self._buf.clear()
            self._bytes = 0


# ════════════════════════════════════════════════════════════════════
# Data structures  (unchanged from v1)
# ════════════════════════════════════════════════════════════════════

@dataclass
class LayerConfig:
    show_6fold:  bool = True;   show_5fold:  bool = True
    show_7fold:  bool = True;   show_bonds:  bool = False
    show_pairs:  bool = True;   show_lagb:   bool = True
    show_hagb:   bool = True;   show_flags:  bool = True
    show_hud:    bool = True

    col_6fold:  tuple = (160, 160, 160);  col_5fold:  tuple = (220,  60,  20)
    col_7fold:  tuple = ( 20,  60, 220);  col_bonds:  tuple = (100, 100, 100)
    col_pairs:  tuple = (220,  20, 220);  col_lagb:   tuple = (  0, 220, 220)
    col_hagb:   tuple = (200, 180,   0);  col_flags:  tuple = (  0, 220,  80)

    radius:      int = 6;   bond_thick: int = 1;  pair_thick: int = 2
    gb_thick:    int = 3;   flag_thick: int = 3;  flag_frames: int = 8

    @classmethod
    def from_dict(cls, d: dict) -> LayerConfig:
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, tuple(v) if isinstance(v, list) else v)
        return obj

    def to_dict(self) -> dict:
        out = {}
        for k, v in asdict(self).items():
            out[k] = list(v) if isinstance(v, tuple) else v
        return out


@dataclass
class FrameResult:
    pts_px: np.ndarray; coord: np.ndarray; psi6: np.ndarray
    pairs57: list; boundaries: list
    edges: np.ndarray = None   # unique undirected edge pairs (E,2) from Delaunay
    _dp: dict = None           # grain-boundary params; None until first LAGB/HAGB render


@dataclass
class FlagEvent:
    frame: int; x_px: float; y_px: float; kind: str; track_id: int


class AnalysisData:
    def __init__(self):
        self.video_path = None; self.dt_eff = 1.0
        self.H = 0; self.W = 0
        self.start_fr = 0; self.end_fr = 0
        self.feats = None; self.tracks = None
        self.frame_results: dict = {}
        self.flag_events: list = []
        self.n_pairs_series: list = []
        self.n_lagb_series: list = []


# ════════════════════════════════════════════════════════════════════
# Analysis backend  (unchanged from v1)
# ════════════════════════════════════════════════════════════════════

_INVERT_FLAG = False


@pipeline
def _to_gray(img):
    if getattr(img, "ndim", 2) == 3: img = img.mean(axis=2)
    if _INVERT_FLAG: img = img.max() - img
    return img.astype(np.float32)


def _fast_bandpass(img: np.ndarray, lshort: float, llong: float) -> np.ndarray:
    """cv2-based Crocker-Grier bandpass: Gaussian(lshort) - Uniform(llong).
    Stays in float32 and uses SIMD C++ — 3-5× faster than tp.bandpass (scipy/float64).
    """
    f = img if img.dtype == np.float32 else img.astype(np.float32)
    smooth = cv2.GaussianBlur(f, (0, 0), sigmaX=float(lshort),
                               borderType=cv2.BORDER_REFLECT)
    sz = max(1, int(round(float(llong))))
    bg = cv2.blur(f, (sz, sz), borderType=cv2.BORDER_REFLECT)
    result = smooth - bg
    np.maximum(result, 0, out=result)
    return result


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
    try:
        if diameter not in _RING_KERNEL_CACHE_GPU:
            _RING_KERNEL_CACHE_GPU[diameter] = _cp.asarray(_ring_kernel(diameter))
        kernel = _RING_KERNEL_CACHE_GPU[diameter]
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
        lmax   = (img == cv2.dilate(img, se)) & (img > thresh)

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
    yx     = np.argwhere(lmax)
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


def detect_grain_diameter(bgr: np.ndarray, params: dict,
                          min_diam: int = 5, max_diam: int = 51,
                          min_particles: int = 3) -> "tuple[int,int] | None":
    """Return (diameter, n_detected) for the odd diameter whose detected
    particles have the lowest mean eccentricity (most round), or None.

    Roundness is preferred over quantity: given two diameters where one finds
    fewer but rounder particles and another finds more but more elongated ones,
    the rounder result wins.  A minimum of *min_particles* detections is
    required before a diameter is considered.

    Uses more permissive thresholds than the regular detection so that it still
    works on slightly out-of-focus or dim frames.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if params.get("invert", False): gray = gray.max() - gray
    proc = _preprocess(gray,
                       params.get("use_bandpass", True),
                       params.get("lshort", 1),
                       params.get("llong",  53),
                       False, 2.0,
                       denoise=params.get("denoise_method", "off"),
                       denoise_strength=params.get("denoise_strength", 10.0),
                       gamma=params.get("gamma", 1.0),
                       sharpen=params.get("sharpen_amount", 0.0),
                       flatten_illum=bool(params.get("flatten_illum", False)),
                       flatten_sigma_frac=params.get("flatten_sigma_frac", 0.15))
    best_diam, best_ecc, best_n = None, 1.0, 0
    diameters = list(range(min_diam | 1, max_diam + 1, 2))
    _mm   = float(params.get("minmass", 1000))
    _pct  = int(params.get("percentile", 64))
    _ring = bool(params.get("ring_mode", False))

    def _check(diam):
        try:
            p_detect = _ring_to_spot(proc, diam) if _ring else proc
            feats = _fast_locate(p_detect, diameter=diam, separation=diam,
                                 minmass=_mm, percentile=_pct, invert=False)
            if feats is None or len(feats) < min_particles:
                return None
            return (diam, float(feats["ecc"].mean()), len(feats))
        except Exception:
            return None

    n_workers = min(len(diameters), max(1, (os.cpu_count() or 2)))
    with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for res in pool.map(_check, diameters):
            if res is None:
                continue
            diam, mean_ecc, n = res
            if mean_ecc < best_ecc or (abs(mean_ecc - best_ecc) < 0.01 and n > best_n):
                best_ecc, best_n, best_diam = mean_ecc, n, diam

    return (best_diam, best_n) if best_diam is not None else None


# ── Diagnostic / auto-tune helpers ──────────────────────────────────────────

def _auto_min_mass(masses: np.ndarray) -> "int | None":
    """1-D Otsu on log(1+mass) to find the noise/particle boundary.

    Vectorised cumsum formulation — no Python loop over thresholds.
    """
    if len(masses) < 10:
        return None
    log_m = np.log1p(masses.astype(np.float64))
    lo, hi = np.percentile(log_m, 2), np.percentile(log_m, 98)
    n_bins = 200
    bins = np.linspace(lo, hi, n_bins + 1)
    hist, _ = np.histogram(log_m, bins=bins)
    bin_mids = 0.5 * (bins[:-1] + bins[1:])

    # Cumulative sums for vectorised between-class variance
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * bin_mids)
    s1 = s0[-1] - s0
    valid = (w0 > 2) & (w1 > 2)
    if not valid.any():
        return None
    mu0 = np.where(valid, s0 / np.maximum(w0, 1), 0.0)
    mu1 = np.where(valid, s1 / np.maximum(w1, 1), 0.0)
    n = w0[-1]
    var = np.where(valid, (w0 / n) * (w1 / n) * (mu0 - mu1) ** 2, 0.0)
    best_t = bin_mids[np.argmax(var)]
    return int(np.expm1(best_t)) if best_t > lo else None


def _auto_max_step(tracks) -> "float | None":
    """Suggest max_step_um = 99th-percentile observed inter-frame displacement × 1.5."""
    if tracks is None or tracks.empty:
        return None
    disps: list[float] = []
    for _, grp in tracks.sort_values("frame").groupby("particle"):
        dx = grp["x_um"].diff().dropna()
        dy = grp["y_um"].diff().dropna()
        disps.extend(np.sqrt(dx ** 2 + dy ** 2).tolist())
    if len(disps) < 5:
        return None
    return round(float(np.percentile(disps, 99)) * 1.5, 2)


def _auto_min_length(tracks) -> "int | None":
    """Suggest min track length from 20th-percentile of the distribution."""
    if tracks is None or tracks.empty:
        return None
    lengths = tracks.groupby("particle").size()
    return max(3, int(np.percentile(lengths, 20)))


def _check_pixel_locking(feats) -> "tuple[bool, float]":
    """Return (is_locked, uniformity_ratio).  ratio < 0.5 → pixel locking."""
    if feats is None or len(feats) < 20:
        return False, 1.0
    x_col = "x_px" if "x_px" in feats.columns else ("x" if "x" in feats.columns else None)
    y_col = "y_px" if "y_px" in feats.columns else ("y" if "y" in feats.columns else None)
    if x_col is None or y_col is None:
        return False, 1.0
    fx = feats[x_col].values % 1.0
    fy = feats[y_col].values % 1.0
    ratio = (float(np.var(fx)) + float(np.var(fy))) / (2.0 * (1.0 / 12.0))
    return ratio < 0.5, round(ratio, 3)


def _draw_ecc_overlay(bgr: np.ndarray, feats, radius: int = 8) -> np.ndarray:
    """Draw detection circles coloured green(round)→red(elongated) by eccentricity."""
    out = bgr.copy()
    if feats is None or feats.empty:
        return out
    x_col = "x_px" if "x_px" in feats.columns else "x"
    y_col = "y_px" if "y_px" in feats.columns else "y"
    xs   = feats[x_col].values
    ys   = feats[y_col].values
    eccs = feats["ecc"].values if "ecc" in feats.columns else np.full(len(xs), 0.5)
    for i in range(len(xs)):
        ecc = float(eccs[i])
        rv  = int(255 * min(1.0, ecc * 2.0))
        gv  = int(255 * max(0.0, 1.0 - ecc * 2.0))
        cv2.circle(out, (int(round(xs[i])), int(round(ys[i]))), radius,
                   (0, gv, rv), 2, cv2.LINE_AA)
    return out


def _draw_linking_overlay(bgr: np.ndarray, tracks, fr: int,
                           max_step_px: float) -> np.ndarray:
    """Draw displacement arrows from frame fr to fr+1, coloured green→red by step fraction."""
    out = bgr.copy()
    if tracks is None or tracks.empty:
        return out
    cur = tracks[tracks.frame == fr][["particle", "x_px", "y_px"]]
    nxt = tracks[tracks.frame == fr + 1][["particle", "x_px", "y_px"]]
    if cur.empty or nxt.empty:
        return out
    merged = cur.merge(nxt, on="particle", suffixes=("", "_n"))
    if not merged.empty:
        x0 = merged["x_px"].values
        y0 = merged["y_px"].values
        x1 = merged["x_px_n"].values
        y1 = merged["y_px_n"].values
        for k in range(len(merged)):
            p1   = (int(round(x0[k])), int(round(y0[k])))
            p2   = (int(round(x1[k])), int(round(y1[k])))
            dist = float(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))
            frac = min(1.0, dist / max(1.0, max_step_px))
            gv   = int(255 * (1.0 - frac))
            rv   = int(255 * frac)
            tip  = max(0.05, min(0.5, 6.0 / max(1.0, dist)))
            cv2.arrowedLine(out, p1, p2, (0, gv, rv), 1, tipLength=tip)
    return out


def _draw_grain_overlay(bgr: np.ndarray, fr_result,
                         radius: int = 8,
                         order_thr: float = 0.6,
                         mis_thr_deg: float = 15.0) -> np.ndarray:
    """Colour particles by crystallographic grain using ψ₆ orientation clustering.

    Ordered particles (|ψ₆| > order_thr) with misorientation < mis_thr_deg are
    grouped into the same grain and drawn in a consistent HSV colour derived from
    their mean lattice orientation.  Defect / boundary particles are drawn gray.
    Edges in the Delaunay graph that cross grain boundaries are drawn white.
    """
    out = bgr.copy()
    pts   = fr_result.pts_px
    psi6  = fr_result.psi6
    edges = fr_result.edges
    N = len(pts)
    if N == 0:
        return out
    if edges is None:
        edges = np.zeros((0, 2), int)

    mag = np.abs(psi6)
    ordered = mag > order_thr

    # Grain orientation angle: arg(ψ₆)/6, period π/3 (60° due to hex symmetry)
    grain_angle = np.angle(psi6) / 6.0          # radians in [-π/6, π/6]
    grain_angle = grain_angle % (np.pi / 3)     # normalise to [0, π/3)

    mis_thr_rad = np.radians(mis_thr_deg) / 6.0

    # Build adjacency for same-grain ordered neighbour pairs — fully vectorised
    # over the unique Delaunay edge array (replaces a Python double-loop over
    # the per-particle neighbour sets, which scales O(particles * neighbours)).
    if len(edges):
        ei, ej   = edges[:, 0], edges[:, 1]
        both_ord = ordered[ei] & ordered[ej]
        da       = np.abs(grain_angle[ei] - grain_angle[ej])
        da       = np.minimum(da, np.pi / 3 - da)        # fold into [0, π/6)
        same     = both_ord & (da < mis_thr_rad)
        rows, cols = ei[same], ej[same]
    else:
        rows = cols = np.zeros(0, int)

    if len(rows):
        adj = _csr((np.ones(len(rows)), (rows, cols)), shape=(N, N))
        _, labels = _cc(adj, directed=False)
        grain_id = np.where(ordered, labels, -1)
    else:
        grain_id = np.full(N, -1)

    # Mean orientation angle per grain → consistent HSV hue (vectorised via
    # bincount instead of a per-particle dict-accumulation loop).
    valid = grain_id >= 0
    grain_hue: dict[int, int] = {}
    if valid.any():
        gids  = grain_id[valid]
        angs  = grain_angle[valid]
        ngr   = int(gids.max()) + 1
        sums  = np.bincount(gids, weights=angs, minlength=ngr)
        cnts  = np.bincount(gids, minlength=ngr)
        means = sums / np.maximum(cnts, 1)
        for gid in np.unique(gids).tolist():
            grain_hue[gid] = int(means[gid] / (np.pi / 3) * 179)

    # Draw grain boundary edges first (behind circles). Filter to boundary
    # edges with numpy first so the cv2.line loop only runs over the (usually
    # small) subset that actually crosses a grain — edges are already unique
    # and undirected so no i/j dedup check is needed.
    if len(edges):
        gi_all = grain_id[edges[:, 0]]; gj_all = grain_id[edges[:, 1]]
        boundary = (gi_all >= 0) & (gj_all >= 0) & (gi_all != gj_all)
        for i, j in edges[boundary].tolist():
            p1 = (int(round(pts[i, 0])), int(round(pts[i, 1])))
            p2 = (int(round(pts[j, 0])), int(round(pts[j, 1])))
            cv2.line(out, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)

    # Pre-build a BGR lookup table for all unique grain hues in a single
    # cv2.cvtColor call instead of one call per particle per frame.
    # grain_hue maps grain_id -> hue (int 0-179); build a (G,1,3) HSV array,
    # convert once, then index by grain_id inside the draw loop.
    _DEFECT_COLOR = (90, 90, 90)
    if grain_hue:
        _gids_sorted = sorted(grain_hue.keys())
        _max_gid     = max(_gids_sorted) + 1
        _hsv_lut     = np.zeros((_max_gid, 1, 3), dtype=np.uint8)
        for _gid in _gids_sorted:
            _hsv_lut[_gid, 0] = (grain_hue[_gid], 210, 245)
        _bgr_lut = cv2.cvtColor(_hsv_lut, cv2.COLOR_HSV2BGR)  # (_max_gid, 1, 3)
    else:
        _max_gid = 0
        _bgr_lut = np.zeros((0, 1, 3), dtype=np.uint8)

    # Draw per-particle circles coloured by grain
    for i in range(N):
        cx, cy = int(round(pts[i, 0])), int(round(pts[i, 1]))
        gid = int(grain_id[i])
        if gid < 0:
            color = _DEFECT_COLOR
        else:
            bgr_c = _bgr_lut[gid, 0]
            color = (int(bgr_c[0]), int(bgr_c[1]), int(bgr_c[2]))
        cv2.circle(out, (cx, cy), radius, color, 2, cv2.LINE_AA)

    return out


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

    # Unique undirected edges — sort each row so col0 < col1, then deduplicate
    e0  = np.sort(s[:, [0, 1]], axis=1)
    e1  = np.sort(s[:, [1, 2]], axis=1)
    e2  = np.sort(s[:, [0, 2]], axis=1)
    uniq = np.unique(np.vstack([e0, e1, e2]), axis=0)  # (E, 2)
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
    for k, fr in enumerate(frames):
        res = frame_results.get(fr)
        if res is not None and len(res.pts_px):
            pts_um = res.pts_px * px_um
            n_pairs_per_bin, sum_g6_per_bin, N, area_um2, _be = _pair_correlation_hist(
                pts_um, res.psi6, r_max_um, dr_um,
                exclude_boundary=exclude_boundary, bounds_um=None)
            total_n_pairs += n_pairs_per_bin
            total_sum_g6 += sum_g6_per_bin
            total_N += N
            total_area += area_um2
            total_psi6_sq_sum += float(np.sum(np.abs(res.psi6) ** 2))
            total_psi6_count += len(res.psi6)
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


def _link_nn(feats_df: pd.DataFrame, search_range: float, memory: int,
             adaptive_stop=None, adaptive_step=None) -> pd.DataFrame:
    """Fast nearest-neighbor linker — replaces tp.link_df for the dense,
    near-monodisperse colloid regime this app targets (search_range /
    separation ~ 1.4, density ratio ~0.004 — firmly in the "greedy NN is
    provably near-optimal" regime; see architectural audit Recommendation 1).

    trackpy's tp.link_df builds a general subnet solver (bipartite matching
    over clusters of ambiguous links) because it has to handle arbitrary
    motion/density. That generality is pure overhead here: for each frame we
    just need the nearest live particle within search_range, resolved
    greedily. This is O(n log n) per frame via cKDTree — no combinatorial
    subnet-solving step exists in this algorithm, so it has no analog to
    trackpy's SubnetOversize failure mode and needs no adaptive_stop/
    adaptive_step safety net (accepted for signature compatibility with the
    tp.link_df call site but otherwise unused/ignored).

    Parameters mirror tp.link_df's semantics:
      search_range — max frame-to-frame displacement (px) to consider a match.
      memory       — max consecutive frames a particle may go unmatched
                     before its track ends.

    Returns a DataFrame with the same columns as feats_df plus an integer
    "particle" column, matching tp.link_df's output contract.
    """
    frames = np.sort(feats_df["frame"].unique())
    cols = list(feats_df.columns)
    xi, yi = cols.index("x_px"), cols.index("y_px")

    next_pid = 0
    # live: particle_id -> (x, y, frames_missing)
    live_ids: list = []
    live_xy: np.ndarray = np.zeros((0, 2), dtype=np.float64)
    live_missing: list = []

    out_records = []  # list of (row_values..., particle_id)

    # Build per-frame lookup once — O(N) — instead of re-scanning the whole
    # multi-frame DataFrame with a boolean mask on every frame iteration
    # (same pattern as pts_by_frame in TrackingWorker.run / _cage's jobs).
    frame_groups: dict = {}
    for fr_val, grp in feats_df.groupby("frame", sort=False):
        frame_groups[fr_val] = (
            grp[["x_px", "y_px"]].to_numpy(dtype=np.float64),
            grp.to_numpy(),
        )
    _empty_xy = np.zeros((0, 2), dtype=np.float64)
    _empty_rows = np.zeros((0, len(cols)), dtype=object)

    for fr in frames:
        det_xy, det_rows = frame_groups.get(fr, (_empty_xy, _empty_rows))
        n_det = len(det_xy)
        n_live = len(live_ids)

        assigned_live = np.full(n_live, -1, dtype=np.int64)   # live idx -> det idx
        assigned_det  = np.full(n_det, -1, dtype=np.int64)    # det idx -> live idx

        if n_live > 0 and n_det > 0:
            # Query each live particle's last-known position against the new
            # frame's detections (natural direction: live positions are the
            # "queries", new detections are the "reference" tree — a live
            # particle either finds its next position among this frame's
            # detections or it doesn't; building the tree over detections
            # also naturally supports "new particle" detections that no live
            # particle claims).
            tree = cKDTree(det_xy)
            # Ask for a handful of nearest neighbors per live particle so a
            # loser in a conflict can fall back to its next-nearest
            # candidate within range, rather than going unmatched outright.
            k = min(5, n_det)
            dists, idxs = tree.query(live_xy, k=k, distance_upper_bound=search_range)
            if k == 1:
                dists = dists[:, None]; idxs = idxs[:, None]

            # Candidate-gather + greedy-assign: JIT-compiled (colloid_kernels)
            # since this pure-Python double loop + sort/assign loop ran once
            # per frame for the whole video — a real cost for dense colloidal-
            # crystal frames (hundreds-to-low-thousands of particles/frame).
            # Only the post-cKDTree-query processing moves into the kernel;
            # the tree build/query itself stays here (scipy isn't numba-
            # compatible). Semantics are unchanged: nearest-first greedy
            # assignment, skip already-claimed live/det, fallback to next-
            # nearest candidate on conflict.
            assigned_live, assigned_det = _link_nn_match(
                np.ascontiguousarray(dists, dtype=np.float64),
                np.ascontiguousarray(idxs, dtype=np.int64),
                n_live, n_det, float(search_range),
            )

        # Update live particles that were matched. Vectorized via boolean
        # masking / fancy indexing over assigned_live/assigned_det (numpy
        # int arrays with -1 sentinels) instead of per-particle Python loops
        # — this bookkeeping was measured as the single largest per-frame
        # cost in _link_nn once the candidate-gather/greedy-assign step
        # moved into the JIT kernel above. Semantics are identical to the
        # original loop: same ID continuity, memory-gap handling, and
        # output row order (matched live particles in original live order,
        # then new particles in detection-index order).
        live_ids_arr = np.asarray(live_ids, dtype=np.int64)
        live_missing_arr = np.asarray(live_missing, dtype=np.int64)

        matched_mask = assigned_live != -1          # (n_live,)
        unmatched_mask = ~matched_mask

        # -- Matched live particles: emit output rows + carry forward with
        #    frames_missing reset to 0, new position = matched detection.
        matched_li = np.nonzero(matched_mask)[0]
        matched_dj = assigned_live[matched_li]
        for li, dj in zip(matched_li.tolist(), matched_dj.tolist()):
            row = det_rows[dj]
            out_records.append(tuple(row) + (live_ids_arr[li],))
        matched_ids = live_ids_arr[matched_li]
        matched_xy = det_xy[matched_dj] if len(matched_dj) else np.zeros((0, 2), np.float64)
        matched_missing = np.zeros(len(matched_li), dtype=np.int64)

        # -- Unmatched live particles: increment frames_missing, keep only
        #    those still within `memory`; drop (track ends) otherwise.
        unmatched_li = np.nonzero(unmatched_mask)[0]
        new_missing_vals = live_missing_arr[unmatched_li] + 1
        keep_mask = new_missing_vals <= memory
        kept_li = unmatched_li[keep_mask]
        kept_ids = live_ids_arr[kept_li]
        kept_xy = live_xy[kept_li] if len(kept_li) else np.zeros((0, 2), np.float64)
        kept_missing = new_missing_vals[keep_mask]

        # -- Unmatched detections become new particles (new sequential IDs).
        new_det_idx = np.nonzero(assigned_det == -1)[0]
        n_new = len(new_det_idx)
        new_ids = np.arange(next_pid, next_pid + n_new, dtype=np.int64)
        next_pid += n_new
        for dj, pid in zip(new_det_idx.tolist(), new_ids.tolist()):
            row = det_rows[dj]
            out_records.append(tuple(row) + (pid,))
        new_xy = det_xy[new_det_idx] if n_new else np.zeros((0, 2), np.float64)
        new_missing = np.zeros(n_new, dtype=np.int64)

        live_ids = np.concatenate([matched_ids, kept_ids, new_ids]).tolist()
        live_xy = np.concatenate([matched_xy, kept_xy, new_xy], axis=0).reshape(-1, 2)
        live_missing = np.concatenate([matched_missing, kept_missing, new_missing]).tolist()

    out_cols = cols + ["particle"]
    if not out_records:
        return pd.DataFrame(columns=out_cols)
    out = pd.DataFrame.from_records(out_records, columns=out_cols)
    # Match tp.link_df's dtypes where practical.
    out["particle"] = out["particle"].astype(np.int64)
    out["frame"] = out["frame"].astype(feats_df["frame"].dtype)
    return out


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


def _flag_events(trk, s, e, H, W, edge, thresh=5):
    evs=[]
    agg = trk.groupby("particle", sort=False).agg(
        f0=("frame", "min"), fn=("frame", "max"),
        xm=("x_px", "mean"), ym=("y_px", "mean"))
    for pid, f0, fn, xm, ym in agg[["f0","fn","xm","ym"]].itertuples(name=None):
        f0=int(f0); fn=int(fn); xm=float(xm); ym=float(ym)
        if xm<edge or xm>W-edge or ym<edge or ym>H-edge: continue
        if f0>s+thresh: evs.append(FlagEvent(f0,xm,ym,"appeared",int(pid)))
        if fn<e-thresh: evs.append(FlagEvent(fn,xm,ym,"disappeared",int(pid)))
    return evs


# ════════════════════════════════════════════════════════════════════
# Annotation engine  (unchanged from v1)
# ════════════════════════════════════════════════════════════════════

def annotate(raw, result, layer, flag_events, fr):
    out = raw.copy()
    if result is None: return out
    pts=result.pts_px; coord=result.coord; r=layer.radius

    if layer.show_bonds and result.edges is not None:
        # edges is already the unique undirected Delaunay edge list, so no
        # i<j / seen-set dedup is needed (was previously a double loop over
        # per-particle neighbour sets with a Python-level seen-pair check).
        for i, j in result.edges.tolist():
            cv2.line(out,(int(round(pts[i,0])),int(round(pts[i,1]))),
                     (int(round(pts[j,0])),int(round(pts[j,1]))),
                     layer.col_bonds,layer.bond_thick,cv2.LINE_AA)

    for gb in result.boundaries:
        is_l=gb["kind"]=="LAGB"
        if is_l and not layer.show_lagb: continue
        if not is_l and not layer.show_hagb: continue
        clr=layer.col_lagb if is_l else layer.col_hagb
        cv2.line(out,(int(round(gb["p1"][0])),int(round(gb["p1"][1]))),
                 (int(round(gb["p2"][0])),int(round(gb["p2"][1]))),
                 clr,layer.gb_thick,cv2.LINE_AA)
        mx=int(round(0.5*(gb["p1"][0]+gb["p2"][0])))
        my=int(round(0.5*(gb["p1"][1]+gb["p2"][1])))
        cv2.putText(out,f"{gb['misorientation_deg']:.1f}°",(mx+4,my-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,clr,1,cv2.LINE_AA)

    if layer.show_pairs:
        for i5,i7 in result.pairs57:
            cv2.line(out,(int(round(pts[i5,0])),int(round(pts[i5,1]))),
                     (int(round(pts[i7,0])),int(round(pts[i7,1]))),
                     layer.col_pairs,layer.pair_thick,cv2.LINE_AA)

    for i in range(len(pts)):
        cx,cy=int(round(pts[i,0])),int(round(pts[i,1]))
        c=int(coord[i]) if i<len(coord) else 6
        if   c==5 and layer.show_5fold: cv2.circle(out,(cx,cy),r,layer.col_5fold,2,cv2.LINE_AA)
        elif c==7 and layer.show_7fold: cv2.circle(out,(cx,cy),r,layer.col_7fold,2,cv2.LINE_AA)
        elif layer.show_6fold:          cv2.circle(out,(cx,cy),r,layer.col_6fold,1,cv2.LINE_AA)

    if layer.show_flags:
        for ev in flag_events:
            if abs(ev.frame-fr)<=layer.flag_frames:
                cv2.circle(out,(int(round(ev.x_px)),int(round(ev.y_px))),
                           r+6,layer.col_flags,layer.flag_thick,cv2.LINE_AA)
                cv2.putText(out,"▲" if ev.kind=="appeared" else "▼",
                            (int(round(ev.x_px))+r+4,int(round(ev.y_px))+4),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,layer.col_flags,1,cv2.LINE_AA)

    if layer.show_hud:
        cv2.putText(out,f"5-7:{len(result.pairs57)}  GBs:{len(result.boundaries)}  fr:{fr}",
                    (10,out.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1,cv2.LINE_AA)
    return out


def _legend(img, layer):
    items=[]
    if layer.show_5fold: items.append((layer.col_5fold,"5-fold"))
    if layer.show_7fold: items.append((layer.col_7fold,"7-fold"))
    if layer.show_pairs: items.append((layer.col_pairs,"5-7 disloc."))
    if layer.show_lagb:  items.append((layer.col_lagb, "LAGB"))
    if layer.show_hagb:  items.append((layer.col_hagb, "HAGB"))
    if layer.show_flags: items.append((layer.col_flags,"Flag"))
    for k,(clr,txt) in enumerate(items):
        y=12+k*18; cv2.circle(img,(12,y+4),4,clr,-1)
        cv2.putText(img,txt,(22,y+9),cv2.FONT_HERSHEY_SIMPLEX,0.38,clr,1,cv2.LINE_AA)
    return img


# Detect whether Qt was built with BGR888 support (Qt ≥ 5.14 / Qt 6).
# If available we can wrap the raw camera buffer without any copy at all.
_FMT_BGR888 = getattr(QImage.Format, "Format_BGR888", None)


class FrameGLWidget(_GL_WIDGET_BASE):
    """GPU-accelerated camera frame display.

    Each call to set_frame() wraps the BGR numpy array in a QImage using
    Format_BGR888 (zero CPU copy) then schedules a repaint.  In the paint
    handler Qt's OpenGL compositor uploads the image as a GPU texture and
    performs aspect-ratio scaling on the GPU — eliminating cvtColor,
    QPixmap.fromImage, and Qt's SmoothTransformation (~5 full-frame CPU
    copies reduced to one DMA upload).
    """

    def __init__(self, placeholder: str = "", parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._buf = None          # keep numpy array alive while QImage references it
        self._placeholder = placeholder
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 150)

    def set_frame(self, bgr: np.ndarray):
        """Wrap bgr array as a QImage (zero-copy when Format_BGR888 available)
        and schedule a repaint.  Must be called from the main thread."""
        if not bgr.flags["C_CONTIGUOUS"]:
            bgr = np.ascontiguousarray(bgr)
        h, w = bgr.shape[:2]
        if _FMT_BGR888 is not None:
            self._buf   = bgr
            self._image = QImage(bgr.data, w, h, w * 3, _FMT_BGR888)
        else:
            # Fallback: one BGR→RGB copy
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            self._buf   = rgb
            self._image = QImage(rgb.data, w, h, w * 3,
                                 QImage.Format.Format_RGB888)
        self.update()

    def clear(self, text: str = ""):
        self._image       = None
        self._buf         = None
        self._placeholder = text
        self.update()

    def _paint(self):
        painter = QPainter(self)
        rect = self.rect()
        if self._image is None:
            painter.fillRect(rect, QColor(4, 4, 14))
            painter.setPen(QColor(0x44, 0xaa, 0xff))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             self._placeholder)
        else:
            painter.fillRect(rect, QColor(0, 0, 0))
            iw, ih = self._image.width(), self._image.height()
            rw, rh = rect.width(), rect.height()
            if iw > 0 and ih > 0 and rw > 0 and rh > 0:
                scale = min(rw / iw, rh / ih)
                dw    = int(iw * scale)
                dh    = int(ih * scale)
                ox    = (rw - dw) // 2
                oy    = (rh - dh) // 2
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                painter.drawImage(QRect(ox, oy, dw, dh), self._image)
        painter.end()

    # QOpenGLWidget calls paintGL; plain QWidget calls paintEvent.
    if _GL_WIDGET_BASE is not QWidget:
        def paintGL(self):           # type: ignore[misc]
            self._paint()
    else:
        def paintEvent(self, event): # type: ignore[misc]
            self._paint()


# ════════════════════════════════════════════════════════════════════
# Workers  (unchanged from v1, CameraWorker updated for camera index)
# ════════════════════════════════════════════════════════════════════

class TrackingWorker(QThread):
    progress   = pyqtSignal(int, str)
    frame_done = pyqtSignal(int, object)
    finished   = pyqtSignal(object)
    error      = pyqtSignal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent); self.params=params; self._abort=False
    def abort(self): self._abort=True

    def run(self):
        p = self.params; data = AnalysisData()
        try:
            vpath = Path(p["video_path"]); data.video_path = vpath

            # Open video with cv2 (faster I/O than pims)
            cap = cv2.VideoCapture(str(vpath), cv2.CAP_FFMPEG)
            if not cap.isOpened():
                raise RuntimeError("Cannot open video — check the file path.")
            N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            data.dt_eff = (1. / float(fps)) if fps and fps > 0 else 1.
            if float(p.get("dt", 0)) > 0: data.dt_eff = float(p["dt"])

            # Dimension from first frame
            ok, _fr0 = cap.read()
            if not ok: raise RuntimeError("Cannot read first frame from video.")
            data.H, data.W = _fr0.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            data.start_fr = int(p.get("start_fr", 0))
            data.end_fr   = min(int(p["end_fr"]) if int(p.get("end_fr", 0)) > 0 else N - 1, N - 1)
            s, e = data.start_fr, data.end_fr
            chunk = int(p.get("chunk", 50))

            diam   = max(3, int(p["diameter"]) | 1)
            sep    = max(diam, int(p["separation"]))
            invert = bool(p.get("invert", False))

            self.progress.emit(2, f"Opened {data.W}×{data.H} | {N} frames (full res)")

            # Preprocessing + detection (runs in parallel worker threads).
            # tp.bandpass / _fast_locate / cv2 all release the GIL → real speedup.
            _bp    = bool(p.get("use_bandpass", True))
            _ls    = p.get("lshort", 1); _ll = p.get("llong", 53)
            _mm    = float(p["minmass"]); _pct = int(p["percentile"])
            _dn    = p.get("denoise_method", "off"); _dns = p.get("denoise_strength", 10.0)
            _gam   = p.get("gamma", 1.0); _shrp = p.get("sharpen_amount", 0.0)
            _ring  = bool(p.get("ring_mode", False))
            _flat  = bool(p.get("flatten_illum", False)); _flat_sf = p.get("flatten_sigma_frac", 0.15)
            _ecc_max = (p.get("ecc_max", 0.8) if p.get("use_ecc_filter", False) else None)
            _reject_size = bool(p.get("reject_size_outliers", False))
            _size_mad    = p.get("size_outlier_mad_mult", 2.5)

            # ── Crystal-lattice-predicted sparse search (opt-in) ───────────
            # See _fast_locate's `search_mask` docstring and the module-level
            # comment above _local_order_proxy/_predict_roi_mask for the full
            # design rationale. Summary: for each frame fr (after warm-up),
            # if frame fr-1's raw detected positions are already available
            # (stored by whichever worker processed fr-1 — frames may
            # complete out of order in the thread pool, so this is looked up
            # by exact frame number, not "most recent"), restrict the
            # candidate search to small disks around well-ordered particles'
            # predicted positions, unioned with a full-frame search on a
            # periodic reconciliation cadence. Any frame lacking usable
            # fr-1 state (not yet computed, aborted, empty, or too few
            # particles) transparently falls back to the existing unrestricted
            # full-frame search — never a correctness risk.
            _use_pred   = bool(p.get("use_lattice_prediction", False))
            _pred_state: dict = {}          # frame_idx -> (xy ndarray, order_score ndarray)
            _pred_lock  = threading.Lock()
            _RECON_EVERY = 15               # periodic full-frame safety-net cadence
            _PRED_RADIUS_MULT = 1.75        # disk radius = this * particle radius
            _ORDER_THRESH = 0.75            # min local-order-proxy score to predict

            # Chained-GPU bandpass+dilate (_gpu_bandpass_dilate_mask / the
            # `gpu_bandpass` param on _fast_locate): DEFAULT-DISABLED.
            #
            # This was implemented and measured on this exact machine
            # (i7-12700F / GTX 1660 Super) using a real, opt-in
            # "use_gpu_bandpass_dilate" params flag rather than being wired on
            # unconditionally, because direct measurement contradicted the
            # ~15-30ms/frame saving estimated by the prior investigation:
            # instead it costs an extra ~40ms/frame (measured via
            # validate_gpu_bandpass_dilate.py in scratchpad; see also the
            # per-stage breakdown showing cv2.cuda's createBoxFilter on
            # CV_32FC1 is O(kernel_size^2) / non-separable in this build,
            # unlike cv2.blur's O(1)-per-pixel CPU box filter -- at the
            # default llong=53 the GPU box step alone (~50ms) is ~3x slower
            # than the entire CPU bandpass+dilate combined). Correctness is
            # exact (verified: identical detections, sub-millipixel match)
            # but performance is a net regression, so it must not be the
            # default. Left in place (opt-in only) in case a future OpenCV
            # CUDA build fixes the box-filter performance, or for experimentation.
            _gpu_chain_eligible = (bool(p.get("use_gpu_bandpass_dilate", False))
                                    and CV2_CUDA_OK and _bp and not _ring and not invert)

            def _process_gray(g: np.ndarray, fr: int = -1):
                g = g.astype(np.float32)
                if invert: g = g.max() - g
                proc = _preprocess(g, _bp, _ls, _ll, False, 2.0,
                                   denoise=_dn, denoise_strength=_dns,
                                   gamma=_gam, sharpen=_shrp,
                                   flatten_illum=_flat, flatten_sigma_frac=_flat_sf,
                                   skip_bandpass=_gpu_chain_eligible)
                if _ring: proc = _ring_to_spot(proc, diam)

                search_mask = None
                prev = None
                if _use_pred and fr >= 0 and (fr % _RECON_EVERY) != 0:
                    with _pred_lock:
                        prev = _pred_state.get(fr - 1)
                    if prev is not None:
                        prev_xy, prev_order = prev
                        if len(prev_xy) >= 8:
                            try:
                                mask = _predict_roi_mask(
                                    proc.shape, prev_xy, prev_order,
                                    drift=(0.0, 0.0),
                                    order_thresh=_ORDER_THRESH,
                                    radius=int(round(R_particle * _PRED_RADIUS_MULT)))
                            except Exception:
                                mask = None
                            if mask is not None:
                                # Always still search the (presumably minority)
                                # low-order/defect/boundary region in full —
                                # only the well-ordered majority's search is
                                # narrowed. This guarantees the masked pass
                                # never removes coverage that the unrestricted
                                # pass would have provided; worst case, this
                                # frame is no faster than the baseline.
                                search_mask = mask

                feats = _fast_locate(proc, diameter=diam, separation=sep,
                                    minmass=_mm, percentile=_pct, invert=False,
                                    ecc_max=_ecc_max, reject_size_outliers=_reject_size,
                                    size_outlier_mad_mult=_size_mad,
                                    search_mask=search_mask,
                                    gpu_bandpass=(_ls, _ll) if _gpu_chain_eligible else None)

                if _use_pred and fr >= 0:
                    # Update state for the NEXT frame using THIS frame's own
                    # raw detections — a cheap local-order proxy computed
                    # directly here, with no dependency on the separate
                    # structural-analysis stage (which runs later, on the
                    # full linked dataset, and would not be available yet on
                    # a first pass — see design note above _local_order_proxy).
                    try:
                        if feats is not None and len(feats) >= 8:
                            xy = feats[["x", "y"]].to_numpy(dtype=np.float64)
                            order = _local_order_proxy(xy)
                        else:
                            xy = np.zeros((0, 2)); order = np.zeros(0, dtype=np.float32)
                    except Exception:
                        xy = np.zeros((0, 2)); order = np.zeros(0, dtype=np.float32)
                    with _pred_lock:
                        _pred_state[fr] = (xy, order)
                        # Bound memory: only ever need the immediately-prior
                        # frame's state, so drop anything older.
                        for old_fr in [k for k in _pred_state if k < fr - 1]:
                            del _pred_state[old_fr]

                return feats

            n_workers = min(10, max(1, (os.cpu_count() or 2)))
            R_particle = max(1, diam // 2)
            all_c = []

            # ── Pipelined decode + detect ──────────────────────────────
            # Decode (VideoCapture.read, single-threaded, not thread-safe) and
            # detection (4 worker threads) previously ran strictly sequentially
            # per chunk — T_read + T_detect. A dedicated reader thread feeding a
            # bounded queue overlaps the two, so wall time becomes
            # max(T_read, T_detect). Seek once up front instead of per-chunk
            # (sequential .read() calls already advance the position; a
            # repeated cap.set() per chunk forces redundant keyframe/GOP
            # re-decodes on inter-coded video).
            cap.set(cv2.CAP_PROP_POS_FRAMES, s)
            qsize = max(8, min(chunk, 64))
            read_q: "_queue.Queue" = _queue.Queue(maxsize=qsize)
            stop_evt = threading.Event()

            def _reader_loop():
                for fr in range(s, e + 1):
                    if stop_evt.is_set():
                        break
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    # Plain put() blocks forever on a full queue if the consumer
                    # stops draining (abort) — recheck stop_evt on a timed put so
                    # this thread can never be stuck past stop_evt.set(), which
                    # would otherwise let cap.release() run concurrently with a
                    # still-pending cap.read() below.
                    while not stop_evt.is_set():
                        try:
                            read_q.put((fr, gray), timeout=0.1); break
                        except _queue.Full:
                            continue
                    if stop_evt.is_set():
                        break
                # Retry until accepted — a timed put that silently discards
                # on Full would leave the consumer waiting for an end-of-stream
                # signal it never receives, causing a permanent hang.
                while not stop_evt.is_set():
                    try:
                        read_q.put(None, timeout=0.1); break
                    except _queue.Full:
                        continue

            reader_thread = threading.Thread(target=_reader_loop, daemon=True)
            reader_thread.start()

            total = max(1, e - s + 1)
            processed = 0
            try:
                with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
                    pending: dict = {}
                    finished_reading = False
                    while not finished_reading or pending:
                        if self._abort:
                            self.error.emit("Aborted."); return
                        # Keep up to qsize detections in flight; pull more from
                        # the queue as workers free up, without blocking past
                        # what's already buffered.
                        while len(pending) < qsize and not finished_reading:
                            try:
                                item = read_q.get(timeout=0.05)
                            except _queue.Empty:
                                break
                            if item is None:
                                finished_reading = True
                                break
                            fr, gray = item
                            fut = pool.submit(_process_gray, gray, fr)
                            pending[fut] = fr

                        if not pending:
                            continue

                        done, _ = _cf.wait(list(pending.keys()), timeout=0.2,
                                           return_when=_cf.FIRST_COMPLETED)
                        for fut in done:
                            fr = pending.pop(fut)
                            feats = fut.result()
                            if feats is not None and not feats.empty:
                                feats = feats.copy()
                                feats["frame"] = fr
                                all_c.append(feats)
                            processed += 1
                        if done:
                            self.progress.emit(5 + int(35 * processed / total),
                                               f"Detecting… {min(s + processed - 1, e)}/{e}")
            finally:
                stop_evt.set()
                # Blocking join (not timed): the reader now always notices
                # stop_evt within ~0.1s even while blocked on a full queue, so
                # this can't hang — and cap.release() must never run while the
                # reader might still be inside cap.read().
                reader_thread.join()
                cap.release()

            if not all_c: raise RuntimeError("No particles detected. Adjust thresholds.")
            feats=pd.concat(all_c,ignore_index=True)
            edge=int(p.get("edge_px",38))
            ok=(feats.x>=edge)&(feats.x<=data.W-1-edge)&(feats.y>=edge)&(feats.y<=data.H-1-edge)
            feats=feats.loc[ok].rename(columns={"x":"x_px","y":"y_px"}).copy()
            px=float(p.get("px_um",0.11))
            feats["x_um"]=feats.x_px*px; feats["y_um"]=feats.y_px*px
            data.feats=feats; self.progress.emit(42,f"Detected {len(feats):,}")

            dp={k:p[k] for k in("pair_dist_px","lagb_eps_px","lagb_min_n","lagb_aspect","lagb_angle_deg")}
            dp["px_um"] = p.get("px_um", 0.11)
            dp["max_neighbor_dist_um"] = p.get("max_neighbor_dist_um", 0.0)
            nfr=e-s+1

            # Build per-frame pts dict once — O(N), avoids O(F·N) repeated scans.
            # Uses only raw detected positions from `feats` — does NOT depend on
            # lnk/trk (the linked tracks), so structural analysis can start
            # immediately, before linking runs.
            pts_by_frame: dict = {}
            for fr_val, grp in feats.groupby("frame"):
                pts_by_frame[int(fr_val)] = grp[["x_px","y_px"]].values

            def _analyze_one(fr):
                if self._abort: return fr, None
                pts = pts_by_frame.get(fr, np.zeros((0,2), float))
                if len(pts) >= 4:
                    return fr, analyze_frame(pts, dp)
                else:
                    return fr, FrameResult(pts,
                        np.zeros(len(pts),int), np.zeros(len(pts),complex),
                        [], [], np.zeros((0,2),int))

            # ── Overlap structural analysis with linking + drift-correction/cage
            # (Proposal 6) ─────────────────────────────────────────────────────
            # Both stages only need `feats` (raw per-frame detections): linking
            # (_link_nn/tp.link_df) resolves frame-to-frame identity, structural
            # analysis (_analyze_one) computes per-frame pair/LAGB/psi6 stats —
            # neither depends on the other's output. Previously analysis started
            # only after linking (and _affine/_cage) had already run in full, so
            # the (expensive) linking step was purely additive on top of the
            # analysis wall-clock instead of overlapping with it. Starting the
            # analysis thread here, before linking begins in the main thread,
            # lets the two run concurrently.
            #
            # The background thread writes only to data.frame_results /
            # data.n_pairs_series / data.n_lagb_series; the main thread (below)
            # writes only to data.tracks / data.flag_events — independent
            # attributes of `data`, so there is no shared mutable state between
            # the two threads. Qt signal emissions from a non-QThread background
            # thread go through queued connections and are therefore thread-safe.
            _ana_aborted = threading.Event()

            def _run_analysis_thread():
                n_ana = min(8, max(1, os.cpu_count() or 2))
                completed = 0
                with _cf.ThreadPoolExecutor(max_workers=n_ana) as ana_pool:
                    futs = {ana_pool.submit(_analyze_one, fr): fr for fr in range(s, e+1)}
                    for fut in _cf.as_completed(futs):
                        if self._abort:
                            _ana_aborted.set()
                            return
                        fr, res = fut.result()
                        if res is None:
                            _ana_aborted.set()
                            return
                        data.frame_results[fr]=res
                        data.n_pairs_series.append((fr,len(res.pairs57)))
                        data.n_lagb_series.append((fr,len(res.boundaries)))
                        completed += 1
                        # Throttled: emitting every frame floods the Qt queued-
                        # connection with thousands of FrameResult objects, each
                        # triggering a GUI-thread plot/render/label update. All
                        # data is already in `data` (sent in full via `finished`),
                        # so the live preview only needs periodic updates.
                        if completed%5==0 or completed==nfr:
                            self.frame_done.emit(fr,res)
                        if completed%20==0:
                            self.progress.emit(63+int(36*completed/max(1,nfr)),
                                               f"Analysing {completed}/{nfr}")

            analyze_thread = threading.Thread(target=_run_analysis_thread, daemon=True)
            analyze_thread.start()

            # ── While analysis runs (background thread), do linking + affine +
            # cage + flag_events in the main thread — these depend on the
            # linked tracks, so they can't start until linking finishes, but
            # linking itself now overlaps with the analysis thread instead of
            # running before it starts.
            self.progress.emit(52,"Linking & analysing…")
            _lnk_kw = dict(
                search_range=float(p.get("max_step_um",3.))/px,
                memory=int(p.get("memory",3)),
                pos_columns=["x_px","y_px"], t_column="frame",
                # adaptive_stop was previously 2.5/px ≈ 22.7 px — barely below
                # the default search_range of 27 px, leaving the linker only one
                # reduction step before hitting the floor. On dense crystal
                # lattices that caused subnet explosion and a permanent hang.
                # Floor at 35% of diameter keeps it well inside inter-particle
                # spacing so the adaptive linker can actually escape dense subnets.
                adaptive_stop=max(diam * 0.35, 2.0), adaptive_step=0.7,
            )
            _lnk_df = feats[["frame","x_px","y_px","x_um","y_um"]].copy()
            _lnk_df = _lnk_df.sort_values("frame", kind="mergesort")
            _lnk_df["x_px"] = _lnk_df["x_px"].astype(np.float32)
            _lnk_df["y_px"] = _lnk_df["y_px"].astype(np.float32)
            # Linker selection (architectural audit Recommendation 1): the
            # default "nn" path uses a cKDTree-based greedy nearest-neighbor
            # linker (_link_nn) tailored to this app's dense, near-
            # monodisperse colloid regime — 5-20x faster than trackpy's
            # general subnet solver for this use case, with no GPU needed.
            # "trackpy" keeps the original path available for fallback/
            # comparison if the NN linker ever mis-tracks on some edge case.
            _linker = p.get("linker", "nn")
            if _linker == "trackpy":
                try:
                    lnk = tp.link_df(_lnk_df, **_lnk_kw)
                except Exception as _le:
                    if "SubnetOversize" in type(_le).__name__ or "SubnetOversize" in str(_le):
                        # Numba linker refuses dense subnets; fall back to the
                        # recursive solver which handles any subnet size (slower
                        # for very dense frames but always produces a result).
                        self.progress.emit(50, "Dense packing — switching to recursive linker…")
                        lnk = tp.link_df(_lnk_df, link_strategy="recursive", **_lnk_kw)
                    else:
                        raise
            else:
                lnk = _link_nn(_lnk_df,
                                search_range=_lnk_kw["search_range"],
                                memory=_lnk_kw["memory"])
            lnk["t"]=(lnk.frame-lnk.frame.min())*data.dt_eff
            _min_len = int(p.get("min_len", 15))
            _counts  = lnk.groupby("particle").size()
            _keep_ids = _counts.index[_counts >= _min_len]
            trk = lnk[lnk["particle"].isin(_keep_ids)].reset_index(drop=True)
            data.tracks=trk; self.progress.emit(52,f"Linked {trk.particle.nunique()} tracks")

            # ── Finish the linked-tracks pipeline (main thread), while the
            # structural-analysis background thread (started above, right
            # after `feats`/pts_by_frame were ready) runs concurrently. ──────
            data.tracks=_affine(data.tracks)

            # Wait for structural analysis to finish before starting _cage.
            # _cage runs its own 8-worker ThreadPoolExecutor (CPU-bound
            # cKDTree queries); if it started while the analysis thread's
            # own 8-worker pool (_run_analysis_thread, above) is still
            # running, up to 16 concurrently-runnable threads would contend
            # for the available physical cores. _cage only reads
            # data.tracks (x_um_aff/y_um_aff), which is independent of the
            # analysis thread's outputs (data.frame_results/n_pairs_series/
            # n_lagb_series), so joining here doesn't introduce a data
            # dependency — it only avoids oversubscribing the CPU. Structural
            # analysis is the longer-running of the two stages, so it has
            # typically already finished by the time linking + _affine
            # complete, making this join a no-op in practice.
            analyze_thread.join()
            if _ana_aborted.is_set() or self._abort:
                self.error.emit("Aborted."); return

            data.tracks=_cage(data.tracks, max_neighbor_dist_um=p.get("max_neighbor_dist_um", 0.0))
            data.flag_events=_flag_events(data.tracks,s,e,data.H,data.W,edge,int(p.get("appear_thresh",5)))
            self.progress.emit(62,f"{len(data.flag_events)} flag events")

            self.progress.emit(100,"Done."); self.finished.emit(data)
        except Exception as exc:
            self.error.emit(f"{exc}\n{_tb.format_exc()}")


class PreviewWorker(QThread):
    done        = pyqtSignal(object)
    feats_ready = pyqtSignal(object)   # normalized DataFrame with x_px/y_px/mass/ecc
    error       = pyqtSignal(str)
    def __init__(self, frame_bgr, params, parent=None):
        super().__init__(parent); self.frame_bgr=frame_bgr; self.params=params
    def run(self):
        p=self.params
        try:
            gray=cv2.cvtColor(self.frame_bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
            if bool(p.get("invert",False)): gray=gray.max()-gray
            proc=_preprocess(gray,p.get("use_bandpass",True),p.get("lshort",1),p.get("llong",53),
                             False, 2.0,
                             denoise=p.get("denoise_method","off"),
                             denoise_strength=p.get("denoise_strength",10.0),
                             gamma=p.get("gamma",1.0),sharpen=p.get("sharpen_amount",0.0),
                             flatten_illum=bool(p.get("flatten_illum",False)),
                             flatten_sigma_frac=p.get("flatten_sigma_frac",0.15))
            diam=int(p["diameter"])|1
            if p.get("ring_mode",False): proc=_ring_to_spot(proc,diam)
            feats=_fast_locate(proc,diameter=diam,separation=int(p["separation"]),
                               minmass=float(p["minmass"]),percentile=int(p["percentile"]),
                               invert=False,
                               ecc_max=(p.get("ecc_max",0.8) if p.get("use_ecc_filter",False) else None),
                               reject_size_outliers=bool(p.get("reject_size_outliers",False)),
                               size_outlier_mad_mult=p.get("size_outlier_mad_mult",2.5))
            if feats is None or feats.empty or len(feats)<4: self.done.emit(None); return
            # Emit feats with x_px/y_px aliases for DiagnosticsPanel
            feats_px = feats.copy()
            feats_px["x_px"] = feats["x"]; feats_px["y_px"] = feats["y"]
            self.feats_ready.emit(feats_px)
            pts=feats[["x","y"]].values
            dp={k:p[k] for k in("pair_dist_px","lagb_eps_px","lagb_min_n","lagb_aspect","lagb_angle_deg")}
            dp["px_um"] = p.get("px_um", 0.11)
            dp["max_neighbor_dist_um"] = p.get("max_neighbor_dist_um", 0.0)
            self.done.emit(analyze_frame(pts,dp))
        except Exception as exc: self.error.emit(str(exc))


class SweepWorker(QThread):
    """Sweep one detection parameter on a single frame and count particles."""
    result   = pyqtSignal(str, list)   # (param_name, [(value, count), ...])
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, frame_bgr, params, sweep_param="minmass",
                 n_steps=20, parent=None):
        super().__init__(parent)
        self.frame_bgr   = frame_bgr
        self.params      = dict(params)
        self.sweep_param = sweep_param
        self.n_steps     = n_steps

    def run(self):
        p = self.params
        try:
            gray = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if bool(p.get("invert", False)):
                gray = gray.max() - gray
            proc = _preprocess(gray, p.get("use_bandpass", True),
                               p.get("lshort", 1), p.get("llong", 53),
                               False, 2.0,
                               denoise=p.get("denoise_method", "off"),
                               denoise_strength=p.get("denoise_strength", 10.0),
                               gamma=p.get("gamma", 1.0), sharpen=p.get("sharpen_amount", 0.0),
                               flatten_illum=bool(p.get("flatten_illum", False)),
                               flatten_sigma_frac=p.get("flatten_sigma_frac", 0.15))
            diam     = int(p["diameter"]) | 1
            if p.get("ring_mode", False): proc = _ring_to_spot(proc, diam)
            base_mm  = float(p.get("minmass",   1000))
            base_pct = int(p.get("percentile",  64))
            if self.sweep_param == "minmass":
                vals = np.linspace(max(10., base_mm * 0.1), base_mm * 4.0, self.n_steps)
            else:
                vals = np.linspace(40., 95., self.n_steps)
            results: list[tuple[float, int]] = []
            for v in vals:
                mm  = float(v)             if self.sweep_param == "minmass"   else base_mm
                pct = int(round(float(v))) if self.sweep_param == "percentile" else base_pct
                try:
                    f = _fast_locate(proc, diameter=diam,
                                     separation=int(p.get("separation", diam)),
                                     minmass=mm, percentile=pct,
                                     invert=False)
                    count = 0 if f is None else len(f)
                except Exception:
                    count = 0
                results.append((float(v), count))
            self.result.emit(self.sweep_param, results)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


# Named live-resolution options: label → max pixel count (longer dimension).
# The camera frame is downsampled to at most this size before being sent to
# the main thread and to AnalysisWorker, keeping signal payloads small.
_LIVE_RES_OPTIONS = [
    ("400 px (fast)",  400),
    ("600 px",         600),
    ("800 px",         800),
    ("1024 px",       1024),
    ("1200 px",       1200),
    ("Full res",      99999),
]
_LIVE_RES_DEFAULT = 99999


class CameraWorker(QThread):
    # Emits (downsampled_bgr, cam_scale) where cam_scale = live_px / original_px.
    frame_ready      = pyqtSignal(np.ndarray, float)
    # Emits raw full-resolution uint16 (12-bit values in a 16-bit container)
    # ONLY when PixelFormat==Mono12 — used for lossless TIFF-sequence recording.
    # Standard MP4/cv2.VideoWriter cannot hold >8 bits/channel (verified: even
    # FFV1 gets silently clamped to 8-bit by OpenCV's Python videoio bindings),
    # so this is a separate path rather than overloading frame_ready's contract.
    frame16_ready    = pyqtSignal(np.ndarray)
    error            = pyqtSignal(str)
    ranges_ready     = pyqtSignal(dict)    # GenICam node ranges, emitted once after Open()
    temperature_ready= pyqtSignal(float)   # DeviceTemperature, polled every ~2s (Basler only)

    def __init__(self, cam_info: dict, live_max_px: int = _LIVE_RES_DEFAULT, parent=None):
        super().__init__(parent)
        self._info        = cam_info
        self._running     = False
        self._live_max_px = live_max_px
        self.fps          = 30.0   # set when camera opens; read by CameraPane
        self._settings_lock = threading.Lock()
        self._pending: dict = {}              # GenICam node name -> value, applied live
        self._pending_pixfmt: "str | None" = None   # requires a stop/restart of grabbing
        # frame_ready is a cross-thread queued signal — if the GUI thread's
        # _on_frame (image enhancement, drawing) falls behind the grab rate,
        # Qt queues every emission with a full-resolution frame attached and
        # nothing ever drains them, growing memory without bound until the
        # process dies with an ArrayMemoryError. Skipping the emit while a
        # previous frame is still in flight caps the queue at one frame.
        self._frame_in_flight = False

    def mark_frame_done(self):
        """Called by CameraPane._on_frame (GUI thread) once it has finished
        with the previous frame, allowing the next one to be emitted."""
        self._frame_in_flight = False

    def set_live_max_px(self, px: int):
        self._live_max_px = px   # Python int write is atomic

    def set_param(self, name: str, value):
        """Queue a GenICam parameter change (ExposureTime, Gain, BlackLevel,
        ExposureAuto, GainAuto, AcquisitionFrameRate, AcquisitionFrameRateEnable),
        applied on the next grab-loop tick. Thread-safe — called from the GUI thread."""
        with self._settings_lock:
            self._pending[name] = value

    def set_pixel_format(self, fmt: str):
        """Queue a PixelFormat change. Applied on the next loop tick by briefly
        stopping/restarting grabbing, since PixelFormat cannot change mid-acquisition."""
        with self._settings_lock:
            self._pending_pixfmt = fmt

    def stop(self): self._running = False

    def run(self):
        if self._info.get("type") == "basler":
            self._run_basler()
        else:
            self._run_opencv()

    @staticmethod
    def _resize_frame(arr: np.ndarray, live_max_px: int):
        """Return (small_frame, scale). scale=1.0 if no resize needed."""
        h, w  = arr.shape[:2]
        scale = min(1.0, live_max_px / max(w, h))
        if scale < 1.0:
            sw = max(1, int(w * scale))
            sh = max(1, int(h * scale))
            return cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA), scale
        return arr, 1.0

    def _apply_pending(self, camera):
        with self._settings_lock:
            pending, self._pending = self._pending, {}
        for name, value in pending.items():
            try:
                node = getattr(camera, name, None)
                if node is None:
                    continue
                if name in ("ExposureAuto", "GainAuto"):
                    node.SetValue(str(value))
                elif name == "AcquisitionFrameRateEnable":
                    node.SetValue(bool(value))
                else:
                    node.SetValue(float(value))
            except Exception as e:
                self.error.emit(f"Camera setting '{name}'={value} rejected: {e}")

    def _maybe_change_pixel_format(self, camera, conv, cur_fmt: str) -> str:
        with self._settings_lock:
            fmt, self._pending_pixfmt = self._pending_pixfmt, None
        if fmt is None:
            return cur_fmt
        try:
            camera.StopGrabbing()
            camera.PixelFormat.SetValue(fmt)
            is_mono = "Mono" in fmt
            conv.OutputPixelFormat = (_pylon.PixelType_Mono8 if is_mono
                                       else _pylon.PixelType_BGR8packed)
            cur_fmt = fmt
        except Exception as e:
            self.error.emit(f"Pixel format change to '{fmt}' failed: {e}")
        finally:
            try: camera.StartGrabbing(_pylon.GrabStrategy_LatestImageOnly)
            except Exception: pass
        return cur_fmt

    def _emit_ranges(self, camera, is_mono: bool):
        ranges: dict = {"is_mono": is_mono}
        for name in ("ExposureTime", "Gain", "BlackLevel"):
            try:
                node = getattr(camera, name, None)
                if node is not None and node.IsReadable():
                    ranges[name] = (float(node.Min), float(node.Max), float(node.Value))
            except Exception:
                pass
        try:
            afr = camera.AcquisitionFrameRate
            ranges["AcquisitionFrameRate"] = (float(afr.Min), float(afr.Max), float(afr.Value))
        except Exception:
            pass
        try:
            pf = camera.PixelFormat
            ranges["PixelFormat_options"] = [str(s) for s in pf.GetSymbolics()]
            ranges["PixelFormat_current"] = str(pf.Value)
        except Exception:
            pass
        self.ranges_ready.emit(ranges)

    def _run_basler(self):
        if not PYPYLON_OK:
            self.error.emit("pypylon not installed.\npip install pypylon"); return
        try:
            import time as _time
            tl   = _pylon.TlFactory.GetInstance()
            devs = tl.EnumerateDevices()
            idx  = self._info.get("index", 0)
            camera = _pylon.InstantCamera(tl.CreateDevice(devs[idx]))
            camera.Open()

            # Prefer ResultingFrameRate — accounts for actual exposure/bandwidth
            # limits, unlike the configured target which may not reflect reality.
            # Never fail silently: a wrong fallback here mis-times every recording.
            try:
                self.fps = float(camera.ResultingFrameRate.Value)
            except Exception:
                try:
                    self.fps = float(camera.AcquisitionFrameRate.Value)
                except Exception:
                    self.fps = 30.0
                    self.error.emit(
                        "Could not read camera frame rate — defaulting to 30 fps. "
                        "Recorded video timing may be inaccurate; verify "
                        "AcquisitionFrameRate in the camera settings.")

            try:
                pixfmt = str(camera.PixelFormat.Value)
            except Exception:
                pixfmt = "BGR8"
            is_mono = "Mono" in pixfmt
            self._emit_ranges(camera, is_mono)
            self._apply_pending(camera)   # apply anything queued before grabbing started

            camera.StartGrabbing(_pylon.GrabStrategy_LatestImageOnly)
            conv = _pylon.ImageFormatConverter()
            conv.OutputPixelFormat = (_pylon.PixelType_Mono8 if is_mono
                                       else _pylon.PixelType_BGR8packed)
            # Second converter, only ever invoked when PixelFormat==Mono12 —
            # preserves the full 12-bit sensor data in a uint16 container for
            # the lossless recording path (frame16_ready).
            conv16 = _pylon.ImageFormatConverter()
            conv16.OutputPixelFormat = _pylon.PixelType_Mono16

            self._running = True
            _temp_every = 2.0
            _last_temp  = 0.0
            while self._running and camera.IsGrabbing():
                pixfmt = self._maybe_change_pixel_format(camera, conv, pixfmt)
                is_mono   = "Mono" in pixfmt
                is_mono12 = pixfmt == "Mono12"
                self._apply_pending(camera)
                # Short timeout (re-checked via TimeoutHandling_Return rather than
                # an exception) so self._running is re-polled every ~300 ms instead
                # of blocking up to 2 s — improves shutdown responsiveness without
                # risking a spurious error on slow-exposure frames.
                res = camera.RetrieveResult(300, _pylon.TimeoutHandling_Return)
                if not res.IsValid():
                    continue
                if res.GrabSucceeded():
                    now = _time.monotonic()
                    # Lossless recording must see every frame regardless of
                    # display load, so it is never gated by _frame_in_flight.
                    if is_mono12:
                        self.frame16_ready.emit(conv16.Convert(res).GetArray().copy())
                    # Display/live-analysis path: no artificial fps ceiling —
                    # emit as fast as the camera delivers, throttled only by
                    # backpressure (skip while the GUI is still on the
                    # previous frame) so the queue can never grow unbounded.
                    if not self._frame_in_flight:
                        arr = conv.Convert(res).GetArray().copy()
                        if is_mono:
                            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                        small, scale = self._resize_frame(arr, self._live_max_px)
                        self._frame_in_flight = True
                        self.frame_ready.emit(small, scale)
                    if now - _last_temp >= _temp_every:
                        try:
                            self.temperature_ready.emit(float(camera.DeviceTemperature.Value))
                        except Exception:
                            pass
                        _last_temp = now
                res.Release()
            camera.StopGrabbing(); camera.Close()
        except Exception as exc: self.error.emit(str(exc))

    def _run_opencv(self):
        import time as _time
        idx     = self._info.get("index", 0)
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap     = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            self.error.emit(f"Cannot open camera {idx}"); return
        v_fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(v_fps) if v_fps > 0 else 30.0
        self._running = True
        while self._running:
            ok, frame = cap.read()
            if ok:
                if not self._frame_in_flight:
                    small, scale = self._resize_frame(frame, self._live_max_px)
                    self._frame_in_flight = True
                    self.frame_ready.emit(small.copy(), scale)
            else:
                QThread.msleep(33)
        cap.release()


class RecorderWorker(QThread):
    """
    Drains RecordingBuffer to a video file on a background thread.

    Encoding priority:
      1. FFmpeg + h264_nvenc  — GPU encoding via NVENC; handles hundreds of
         fps at full resolution without touching the CPU encoder at all.
      2. FFmpeg + libx264 ultrafast  — CPU software encode but much faster
         than mp4v because libx264 is multi-threaded.
      3. cv2.VideoWriter (mp4v)  — fallback when FFmpeg is not on PATH.

    Frames downscaled by the backpressure logic are upscaled to the original
    resolution before writing so the output is always a consistent size.
    """
    progress = pyqtSignal(int, str)   # (fill_pct, status_text)
    finished = pyqtSignal(str)         # output path when fully flushed
    error    = pyqtSignal(str)

    def __init__(self, buf: RecordingBuffer, path: str,
                 fps: float, out_size: tuple, parent=None):
        super().__init__(parent)
        self._buf    = buf
        self._path   = path
        self._fps    = max(float(fps), 1.0)
        self._out_w  = int(out_size[0])
        self._out_h  = int(out_size[1])
        self._running = True

    def stop(self): self._running = False

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        fh, fw = frame.shape[:2]
        if fw != self._out_w or fh != self._out_h:
            frame = cv2.resize(frame, (self._out_w, self._out_h),
                               interpolation=cv2.INTER_LINEAR)
        return frame

    def run(self):
        try:
            if _FFMPEG_PATH:
                self._drain_ffmpeg()
            else:
                self._drain_cv2()
            self.finished.emit(self._path)
        except Exception as exc:
            self.error.emit(f"{exc}\n{_tb.format_exc()}")

    def _build_ffmpeg_cmd(self, use_nvenc: bool) -> list:
        if use_nvenc:
            enc_args = ["-c:v", "h264_nvenc", "-preset", "p4",
                        "-rc:v", "vbr", "-cq:v", "23"]
        else:
            enc_args = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"]

        return [
            _FFMPEG_PATH, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self._out_w}x{self._out_h}",
            "-r", str(self._fps),
            "-i", "pipe:0",
            *enc_args,
            "-pix_fmt", "yuv420p",
            self._path,
        ]

    def _spawn_ffmpeg(self, use_nvenc: bool):
        """Start the ffmpeg subprocess and do a brief startup-health check.

        NVENC can fail to initialize for reasons that only show up once ffmpeg
        actually tries to open the encoder (GPU busy, driver mismatch,
        unsupported resolution, etc.) — those failures make ffmpeg exit almost
        immediately, which is exactly what turns the *next* stdin.write() into
        a broken-pipe/EINVAL crash. Polling proc.poll() shortly after Popen
        lets us catch that here, with ffmpeg's real stderr message, instead of
        discovering it later as an opaque pipe error.
        """
        cmd = self._build_ffmpeg_cmd(use_nvenc)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
        time.sleep(0.15)
        if proc.poll() is not None:
            # Died before a single frame was written - almost always a bad
            # encoder/arg combination rather than anything frame-data related.
            try:
                stderr_b = proc.stderr.read()
            except Exception:
                stderr_b = b""
            msg = stderr_b.decode("utf-8", errors="ignore")[-600:]
            return None, msg
        return proc, None

    def _drain_ffmpeg(self):
        use_nvenc = _NVENC_OK
        proc, startup_err = self._spawn_ffmpeg(use_nvenc)
        if proc is None and use_nvenc:
            # NVENC failed to start (busy GPU, driver mismatch, unsupported
            # resolution, ...) - fall back to the CPU encoder rather than
            # aborting the whole recording.
            self.progress.emit(0, "NVENC unavailable, falling back to CPU encoder (libx264)")
            use_nvenc = False
            proc, startup_err = self._spawn_ffmpeg(use_nvenc)
        if proc is None:
            raise RuntimeError(
                f"FFmpeg failed to start (exit before first frame):\n{startup_err}"
            )

        n = 0
        broken = False
        try:
            while self._running or len(self._buf) > 0:
                frame = self._buf.get()
                if frame is None:
                    self._buf.wait(timeout=0.1); continue
                try:
                    proc.stdin.write(self._prep(frame).tobytes())
                except OSError:
                    # BrokenPipeError is an OSError subclass, but on Windows a
                    # write to a pipe whose reader already exited can surface
                    # as a plain OSError (errno 22 / EINVAL) instead of the
                    # POSIX-style BrokenPipeError, so both must be caught here.
                    broken = True
                    break
                n += 1
                if n % 30 == 0:
                    pct = int(self._buf.fill * 100)
                    self.progress.emit(pct, f"Rec {n} fr  buf {pct}%")
        finally:
            try: proc.stdin.close()
            except Exception: pass

        # NOTE: we deliberately do NOT attempt a mid-stream NVENC->libx264
        # fallback into the same output file. A prior version of this code
        # tried that: on a mid-stream NVENC death it re-ran _spawn_ffmpeg(),
        # whose command (see _build_ffmpeg_cmd) always targets self._path
        # with "-y" (unconditional overwrite). That second ffmpeg process
        # truncated/overwrote whatever the dead NVENC process had already
        # written, silently discarding every frame encoded before the crash
        # while still reporting the full frame count `n` - producing an
        # output file that is shorter than (and inconsistent with) what the
        # UI claims was recorded. Two independently-encoded MP4 streams also
        # cannot be spliced together by simply writing a second one to the
        # same path (MP4's moov/mdat atom structure doesn't support naive
        # concatenation), so "resuming" into the same file is unsound
        # regardless of the overwrite issue. If NVENC dies partway through,
        # fail the recording cleanly instead: keep whatever partial file
        # ffmpeg already finalized and tell the user how many frames made it
        # in, rather than risk silently corrupting/truncating the output.
        _, stderr_b = proc.communicate(timeout=120)
        if broken or proc.returncode != 0:
            msg = stderr_b.decode("utf-8", errors="ignore")[-600:]
            reason = "pipe closed unexpectedly" if broken else f"exit {proc.returncode}"
            partial_note = (
                f" ({n} frame(s) were captured and finalized to {self._path} "
                "before the failure; that partial file has been left in place.)"
                if n > 0 else ""
            )
            raise RuntimeError(
                f"FFmpeg {reason} after {n} frame(s):\n{msg}{partial_note}"
            )

    def _drain_cv2(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            self._path, fourcc, self._fps, (self._out_w, self._out_h)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open output file:\n{self._path}")
        n = 0
        try:
            while self._running or len(self._buf) > 0:
                frame = self._buf.get()
                if frame is None:
                    self._buf.wait(timeout=0.1); continue
                writer.write(self._prep(frame))
                n += 1
                if n % 30 == 0:
                    pct = int(self._buf.fill * 100)
                    self.progress.emit(pct, f"Rec {n} fr  buf {pct}%")
        finally:
            writer.release()


class TiffSequenceRecorder(QThread):
    """Writes lossless 16-bit TIFF frames to an output folder.

    Used for Mono12 capture: standard MP4 (and even FFV1 through OpenCV's
    Python videoio bindings — verified empirically) cannot hold more than
    8 bits/channel, so genuine 12-bit data can only be preserved losslessly
    as individual frames rather than a single video file. cv2.imwrite/imread
    DO round-trip 16-bit single-channel TIFF exactly (unlike VideoWriter).
    """
    progress = pyqtSignal(int, str)   # (frame_count, status_text)
    finished = pyqtSignal(str)         # output folder path
    error    = pyqtSignal(str)

    def __init__(self, out_dir: str, max_queued: int = 256, parent=None):
        super().__init__(parent)
        self._out_dir    = Path(out_dir)
        self._queue: _deque = _deque()
        self._max_queued = max_queued
        self._lock        = threading.Lock()
        self._event       = threading.Event()   # wakes run() on push(), like AnalysisWorker
        self._running     = True
        self._count        = 0
        self._dropped       = 0

    def push(self, frame16: np.ndarray) -> bool:
        """Called from the GUI thread on every Mono12 frame. Returns False
        (and drops the frame) if the write-out queue is saturated."""
        with self._lock:
            if len(self._queue) >= self._max_queued:
                self._dropped += 1
                return False
            self._queue.append(frame16)
        self._event.set()
        return True

    def stop(self):
        self._running = False
        self._event.set()   # wake run() immediately so it can observe _running

    def run(self):
        try:
            self._out_dir.mkdir(parents=True, exist_ok=True)
            while self._running or self._queue:
                with self._lock:
                    frame = self._queue.popleft() if self._queue else None
                if frame is None:
                    self._event.wait(timeout=0.1); self._event.clear(); continue
                fname = self._out_dir / f"frame_{self._count:06d}.tiff"
                cv2.imwrite(str(fname), frame, [cv2.IMWRITE_TIFF_COMPRESSION, 5])  # LZW lossless
                self._count += 1
                if self._count % 20 == 0:
                    msg = f"{self._count} frames"
                    if self._dropped:
                        msg += f"  ({self._dropped} dropped — disk I/O too slow)"
                    self.progress.emit(self._count, msg)
        except Exception as exc:
            self.error.emit(f"{exc}\n{_tb.format_exc()}")
        finally:
            self.finished.emit(str(self._out_dir))


# ════════════════════════════════════════════════════════════════════
# Sutter Lambda SC shutter controller (USB/serial, long-duration timelapse)
# ════════════════════════════════════════════════════════════════════

class LambdaSCController:
    """Serial (USB-CDC virtual COM port) controller for a Sutter Lambda SC
    SmartShutter controller.

    Protocol per the Lambda SC Operation Manual (Rev. 1.20D): 9600 baud, 8
    data bits, no parity, 1 stop bit, no flow control. Every command is a
    single byte; the controller echoes that byte back immediately, then
    transmits 0x0D (ASCII CR) once the operation completes. The Status
    command's reply embeds the echo as its first byte followed by status
    data, 13 bytes total, also CR-terminated.

    Command bytes: Open Shutter A = 0xAA, Close Shutter A = 0xAC,
    Fast mode = 0xDC, Soft mode = 0xDD, Status = 0xCC.
    """
    OPEN   = 0xAA
    CLOSE  = 0xAC
    FAST   = 0xDC
    SOFT   = 0xDD
    STATUS = 0xCC
    BAUD   = 9600

    _MODE_NAMES = {0xDB: "not_connected", 0xDC: "fast", 0xDD: "soft", 0xDE: "neutral_density"}

    def __init__(self):
        self._ser: "_serial.Serial | None" = None
        self._lock = threading.Lock()

    @staticmethod
    def list_ports() -> list:
        if not PYSERIAL_OK:
            return []
        return [p.device for p in _serial_list_ports.comports()]

    @property
    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def connect(self, port: str, timeout: float = 2.0):
        if not PYSERIAL_OK:
            raise RuntimeError("pyserial not installed.\npip install pyserial")
        ser = _serial.Serial(port, self.BAUD, bytesize=_serial.EIGHTBITS,
                              parity=_serial.PARITY_NONE, stopbits=_serial.STOPBITS_ONE,
                              timeout=timeout)
        self._ser = ser
        try:
            self.get_status()   # verifies the controller actually answers
        except Exception:
            ser.close(); self._ser = None
            raise

    def disconnect(self):
        # Locked so this can't close the handle out from under a concurrent
        # _send() that's blocked inside a read() on another thread (e.g. the
        # GUI thread disconnecting while a TimelapseWorker cycle is in flight).
        with self._lock:
            if self._ser is not None:
                try: self._ser.close()
                except Exception: pass
                self._ser = None

    def _send(self, cmd: int, expect_total: int = 2) -> bytes:
        """Write one command byte and read back *expect_total* bytes (echo
        included). Raises if the echo doesn't match or no CR arrives within
        the serial timeout."""
        with self._lock:
            ser = self._ser
            if ser is None or not ser.is_open:
                raise RuntimeError("Lambda SC not connected.")
            ser.reset_input_buffer()
            ser.write(bytes([cmd]))
            reply = ser.read(expect_total)
        if len(reply) < 1 or reply[0] != cmd:
            raise RuntimeError(f"Lambda SC did not echo command 0x{cmd:02X} (got {reply!r}).")
        if len(reply) < expect_total or reply[-1] != 0x0D:
            raise RuntimeError(f"Lambda SC did not confirm command 0x{cmd:02X} (timed out).")
        return reply

    def open_shutter(self):  self._send(self.OPEN)
    def close_shutter(self): self._send(self.CLOSE)
    def set_fast_mode(self): self._send(self.FAST)
    def set_soft_mode(self): self._send(self.SOFT)

    def get_status(self) -> dict:
        reply = self._send(self.STATUS, expect_total=13)
        state = reply[1] if len(reply) > 1 else None
        mode  = reply[2] if len(reply) > 2 else None
        return {"open": state == self.OPEN,
                "mode": self._MODE_NAMES.get(mode, "unknown")}


class TimelapseWorker(QThread):
    """Drives a long-duration timelapse via the Lambda SC shutter: opens the
    shutter, waits a short settle time, grabs the current live frame and
    saves it, closes the shutter again, then sleeps the remaining interval.

    Keeping the shutter closed between captures is the point of doing this
    at all — it protects light-sensitive colloidal samples from continuous
    illumination/heating over runs that can span hours, opening only for the
    brief moment each frame is actually captured.
    """
    captured = pyqtSignal(int, str)     # (capture_index, saved_path)
    progress = pyqtSignal(int, float)   # (capture_index, seconds_until_next)
    finished = pyqtSignal(int)          # total captures taken
    error    = pyqtSignal(str)

    def __init__(self, shutter: LambdaSCController, get_frame, out_dir: str,
                 interval_s: float, settle_s: float, max_captures: int, parent=None):
        super().__init__(parent)
        self._shutter   = shutter
        self._get_frame = get_frame     # callable() -> latest BGR frame or None
        self._out_dir   = Path(out_dir)
        self._interval   = max(0.0, float(interval_s))
        self._settle     = max(0.0, float(settle_s))
        self._max        = int(max_captures)   # 0 = unlimited, run until stop()
        self._running    = False

    def stop(self): self._running = False

    def run(self):
        self._running = True
        try:
            self._out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.error.emit(f"Cannot create output folder: {exc}"); self.finished.emit(0); return

        n = 0
        try:
            while self._running and (self._max <= 0 or n < self._max):
                t_cycle = time.monotonic()
                try:
                    self._shutter.open_shutter()
                except Exception as exc:
                    self.error.emit(f"Shutter open failed: {exc}"); break

                # Sleep the settle time in short chunks so stop() lands promptly.
                remaining = self._settle
                while remaining > 0 and self._running:
                    chunk = min(0.05, remaining); time.sleep(chunk); remaining -= chunk

                # .copy() — some capture backends reuse a single buffer in
                # place; without copying, the array this cycle grabbed could
                # be overwritten by the next live frame before imwrite runs.
                frame = self._get_frame()
                frame = frame.copy() if frame is not None else None
                ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = self._out_dir / f"tl_{n:05d}_{ts}.tiff"
                if frame is not None:
                    cv2.imwrite(str(fname), frame, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
                else:
                    self.error.emit("No frame available (camera disconnected?) — capture skipped.")

                try:
                    self._shutter.close_shutter()
                except Exception as exc:
                    self.error.emit(f"Shutter close failed: {exc}"); break

                n += 1
                self.captured.emit(n, str(fname) if frame is not None else "(skipped — no frame)")

                elapsed   = time.monotonic() - t_cycle
                remaining = self._interval - elapsed
                while remaining > 0 and self._running:
                    chunk = min(0.2, remaining)
                    time.sleep(chunk)
                    remaining -= chunk
                    self.progress.emit(n, max(0.0, remaining))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit(n)


# ════════════════════════════════════════════════════════════════════
# Live-camera analysis worker (background thread)
# ════════════════════════════════════════════════════════════════════

class AnalysisWorker(QThread):
    """Runs tp.locate + defect analysis on a background thread.

    The camera worker already downsamples frames to the chosen live resolution
    before emitting them, so this worker receives small frames and the
    cam_scale factor that was applied.  All pixel-space parameters (diameter,
    pair_dist_px, etc.) are scaled by cam_scale before use so that the user's
    UI values always refer to original-camera pixels.

    Only the latest pending frame is kept — stale frames are dropped.
    """
    analysis_ready = pyqtSignal(np.ndarray)   # annotated BGR at live resolution

    def __init__(self, layer: LayerConfig, params: dict, parent=None):
        super().__init__(parent)
        self._lock           = threading.Lock()
        self._event          = threading.Event()   # wakes worker immediately on new frame
        self._pending        : np.ndarray | None = None
        self._pending_prefix : np.ndarray | None = None  # shared preprocess prefix from _on_frame
        self._cam_scale      : float = 1.0
        self._layer          = layer
        self._params         = dict(params)
        self._running        = False
        self._min_interval   : float = 0.20   # seconds; 1/fps cap

    def update_frame(self, bgr: np.ndarray, cam_scale: float = 1.0,
                     preprocessed_prefix: "np.ndarray | None" = None):
        """Called from main thread on every camera frame — keeps only the latest.

        preprocessed_prefix: optional float32 grayscale image that has already
        had denoise/gamma/sharpen applied (produced by CameraPane._on_frame so
        those steps run only once per frame in Compare mode). When supplied,
        run() resumes the pipeline from CLAHE+bandpass instead of repeating the
        full preprocess.
        """
        with self._lock:
            self._pending            = bgr
            self._cam_scale          = cam_scale
            self._pending_prefix     = preprocessed_prefix
        self._event.set()   # wake worker immediately; no more 10 ms poll latency

    def set_fps_cap(self, fps: float):
        self._min_interval = 1.0 / max(0.5, fps)

    def update_config(self, layer: LayerConfig, params: dict):
        with self._lock:
            self._layer  = layer
            self._params = dict(params)

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            self._event.wait(timeout=0.1)   # wake on new frame or 100 ms watchdog
            self._event.clear()
            with self._lock:
                bgr              = self._pending
                self._pending    = None
                prefix           = self._pending_prefix
                self._pending_prefix = None
                cam_scale        = self._cam_scale
                layer            = self._layer
                p                = dict(self._params)
            if bgr is None:
                continue

            # Frame is already at live resolution from CameraWorker.
            # Scale all pixel-space params by cam_scale so they match the frame.
            t_start = time.monotonic()
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if p.get("invert", False): gray = gray.max() - gray
            # When _on_frame already computed the shared denoise/gamma/sharpen
            # prefix, pass it in so _preprocess can skip those expensive steps.
            proc = _preprocess(gray,
                               p.get("use_bandpass", True), p.get("lshort", 1),
                               p.get("llong", 53), False, 2.0,
                               denoise=p.get("denoise_method", "off"),
                               denoise_strength=p.get("denoise_strength", 10.0),
                               gamma=p.get("gamma", 1.0), sharpen=p.get("sharpen_amount", 0.0),
                               flatten_illum=bool(p.get("flatten_illum", False)),
                               flatten_sigma_frac=p.get("flatten_sigma_frac", 0.15),
                               preprocessed_prefix=prefix)
            diam = max(3, int(p.get("diameter", 19)   * cam_scale) | 1)
            sep  = max(diam, int(p.get("separation", 19) * cam_scale))
            if p.get("ring_mode", False): proc = _ring_to_spot(proc, diam)
            try:
                feats = _fast_locate(proc, diameter=diam, separation=sep,
                                     minmass=float(p.get("minmass", 3000)),
                                     percentile=int(p.get("percentile", 80)),
                                     invert=False,
                                     ecc_max=(p.get("ecc_max", 0.8) if p.get("use_ecc_filter", False) else None),
                                     reject_size_outliers=bool(p.get("reject_size_outliers", False)),
                                     size_outlier_mad_mult=p.get("size_outlier_mad_mult", 2.5))
            except Exception:
                feats = None

            if feats is not None and not feats.empty and len(feats) >= 4:
                pts = feats[["x", "y"]].values
                dp  = {
                    "pair_dist_px":  p.get("pair_dist_px",  48)  * cam_scale,
                    "lagb_eps_px":   p.get("lagb_eps_px",   96)  * cam_scale,
                    "lagb_min_n":    p.get("lagb_min_n",     3),
                    "lagb_aspect":   p.get("lagb_aspect",   2.0),
                    "lagb_angle_deg":p.get("lagb_angle_deg",15.0),
                    # px_um is calibrated against original-resolution pixels;
                    # `pts` here are in this (possibly downscaled) live-preview
                    # frame's pixel space, so divide by cam_scale to convert
                    # µm -> px consistently in analyze_frame.
                    "px_um":         p.get("px_um", 0.11) / max(cam_scale, 1e-9),
                    "max_neighbor_dist_um": p.get("max_neighbor_dist_um", 0.0),
                }
                res = analyze_frame(pts, dp)
                ann = annotate(bgr, res, layer, [], 0)
                _legend(ann, layer)
            else:
                ann = bgr.copy()

            self.analysis_ready.emit(ann)
            # Throttle to the configured fps cap in short chunks so stop() is
            # noticed within ~20 ms rather than sleeping through the full interval.
            elapsed = time.monotonic() - t_start
            remaining = self._min_interval - elapsed
            while remaining > 0 and self._running:
                time.sleep(min(0.02, remaining))
                remaining -= 0.02


class GrainSizeWorker(QThread):
    """Scans a range of particle diameters and finds the one that detects the
    greatest number of near-circular (low-eccentricity) particles.

    Runs entirely in a background thread; emits progress so the UI can update
    a status label or progress bar without blocking.
    """
    detected = pyqtSignal(int, int)   # (best_diameter_px, n_round_particles)
    progress = pyqtSignal(int, str)   # (percent 0-100, message)
    failed   = pyqtSignal(str)        # reason — no particles found

    _MIN_DIAM = 5
    _MAX_DIAM = 51

    def __init__(self, bgr: np.ndarray, params: dict, parent=None):
        super().__init__(parent)
        self._bgr    = bgr
        self._params = dict(params)
        self._abort  = False

    def abort(self): self._abort = True

    def run(self):
        diameters = list(range(self._MIN_DIAM | 1, self._MAX_DIAM + 1, 2))
        p = self._params
        gray = cv2.cvtColor(self._bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if p.get("invert", False): gray = gray.max() - gray
        proc = _preprocess(gray, p.get("use_bandpass", True),
                           p.get("lshort", 1), p.get("llong", 53),
                           False, 2.0,
                           denoise=p.get("denoise_method", "off"),
                           denoise_strength=p.get("denoise_strength", 10.0),
                           gamma=p.get("gamma", 1.0), sharpen=p.get("sharpen_amount", 0.0),
                           flatten_illum=bool(p.get("flatten_illum", False)),
                           flatten_sigma_frac=p.get("flatten_sigma_frac", 0.15))
        best_diam, best_ecc, best_n = None, 1.0, 0
        _MIN_P = 3
        _mm    = float(p.get("minmass", 1000))
        _pct   = int(p.get("percentile", 64))
        _ring  = bool(p.get("ring_mode", False))

        def _check(diam):
            try:
                p_detect = _ring_to_spot(proc, diam) if _ring else proc
                feats = _fast_locate(p_detect, diameter=diam, separation=diam,
                                     minmass=_mm, percentile=_pct, invert=False)
                if feats is None or len(feats) < _MIN_P:
                    return None
                return (diam, float(feats["ecc"].mean()), len(feats))
            except Exception:
                return None

        # All diameters share the same read-only proc array — safe to parallelise.
        n_workers = min(len(diameters), max(1, (os.cpu_count() or 2)))
        done = 0
        with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_check, d): d for d in diameters}
            for fut in _cf.as_completed(futs):
                if self._abort:
                    return
                done += 1
                self.progress.emit(int(100 * done / len(diameters)),
                                   "Scanning diameters…")
                res = fut.result()
                if res is None:
                    continue
                diam, mean_ecc, n = res
                if mean_ecc < best_ecc or (abs(mean_ecc - best_ecc) < 0.01 and n > best_n):
                    best_ecc, best_n, best_diam = mean_ecc, n, diam

        self.progress.emit(100, "Done.")
        if best_diam is not None:
            self.detected.emit(best_diam, best_n)
        else:
            self.failed.emit("No round particles found — check focus, exposure, "
                             "and bandpass settings.")


class GrainConsistencyWorker(QThread):
    """Samples N evenly-spaced frames from a video and checks whether the
    auto-detected diameter is consistent with the diameter used in analysis.

    Opens its own VideoCapture so it does not interfere with the main capture.
    """
    inconsistent = pyqtSignal(int, int, int)  # (frame_no, detected_diam, used_diam)
    consistent   = pyqtSignal()

    def __init__(self, video_path: str, frame_nums: "list[int]",
                 used_diam: int, params: dict, parent=None):
        super().__init__(parent)
        self._path   = video_path
        self._frames = frame_nums
        self._used   = used_diam
        self._params = dict(params)
        self._abort  = False

    def abort(self): self._abort = True

    def run(self):
        cap = cv2.VideoCapture(self._path, cv2.CAP_FFMPEG)
        try:
            for fr_num in self._frames:
                if self._abort:
                    return
                cap.set(cv2.CAP_PROP_POS_FRAMES, fr_num)
                ok, frame = cap.read()
                if not ok:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                result = detect_grain_diameter(bgr, self._params)
                if result is not None:
                    det = result[0]
                    if abs(det - self._used) > 2:
                        self.inconsistent.emit(fr_num, det, self._used)
                        return
            self.consistent.emit()
        finally:
            cap.release()


class StructureFunctionWorker(QThread):
    """Computes trajectory-averaged (or custom-range) g(r)/g6(r) in the
    background — mirrors GrainConsistencyWorker's simple sampled-frame-loop
    pattern. Used for "Full trajectory (averaged)" and "Custom range" modes;
    "Current frame" mode is fast enough to run synchronously on the GUI
    thread instead (see MainWindow._on_compute_structure_functions)."""
    progress = pyqtSignal(int, str)   # (pct 0-100, message)
    finished_ok = pyqtSignal(object, object, object, object)  # r_centers, g_r, g6_r, g6_over_g_r
    error = pyqtSignal(str)

    def __init__(self, frame_results: dict, frame_range, px_um: float,
                 r_max_um: float, dr_um: float, exclude_boundary: bool, parent=None):
        super().__init__(parent)
        self._frame_results = frame_results
        self._frame_range   = list(frame_range)
        self._px_um         = px_um
        self._r_max_um      = r_max_um
        self._dr_um         = dr_um
        self._exclude_boundary = exclude_boundary
        self._abort = False

    def abort(self): self._abort = True

    def run(self):
        try:
            n_total = max(1, len(self._frame_range))

            def _progress_cb(done, total):
                if done % 5 == 0 or done == total:
                    pct = int(100 * done / max(total, 1))
                    self.progress.emit(pct, f"Accumulating pair correlations ({done}/{total})")

            result = _pair_correlations_trajectory(
                self._frame_results, self._frame_range, self._px_um,
                self._r_max_um, self._dr_um,
                exclude_boundary=self._exclude_boundary,
                progress_cb=_progress_cb)
            if self._abort:
                return
            if result is None:
                self.error.emit("No usable frames in the selected range.")
                return
            r_centers, g_r, g6_r, g6_over_g_r = result
            self.finished_ok.emit(r_centers, g_r, g6_r, g6_over_g_r)
        except Exception as e:
            self.error.emit(str(e))


# ════════════════════════════════════════════════════════════════════
# GUI helpers
# ════════════════════════════════════════════════════════════════════

class ColourBtn(QPushButton):
    colour_changed = pyqtSignal(tuple)
    def __init__(self, bgr, parent=None):
        super().__init__(parent); self._bgr=bgr; self.setFixedSize(26,20); self._refresh(); self.clicked.connect(self._pick)
    def _refresh(self):
        b,g,r=self._bgr; self.setStyleSheet(f"background:rgb({r},{g},{b});border:1px solid #666;")
    def _pick(self):
        qc=QColorDialog.getColor(QColor(self._bgr[2],self._bgr[1],self._bgr[0]),self)
        if qc.isValid():
            self._bgr=(qc.blue(),qc.green(),qc.red()); self._refresh(); self.colour_changed.emit(self._bgr)
    def get_bgr(self): return self._bgr
    def set_bgr(self,b): self._bgr=b; self._refresh()


# ════════════════════════════════════════════════════════════════════
# Parameter panel
# ════════════════════════════════════════════════════════════════════

class ParamPanel(QScrollArea):
    params_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        # Long labels ("Max neighbour dist [µm]") plus a spin box need more
        # room than Qt's default dock-splitter guess gives on first launch —
        # without this the dock starts too narrow and labels/tooltips get
        # squeezed. Saved window geometry (restoreState) overrides this on
        # subsequent launches once the user has resized it themselves.
        self.setMinimumWidth(280)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.viewport().setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # NOTE: previously installed a wheel-event filter here that redirected
        # scroll-wheel events over spinboxes/combos to change their value
        # instead of scrolling the panel — the opposite of what users want.
        # Removed; NoScrollSpinBox/NoScrollDoubleSpinBox/NoScrollComboBox now
        # correctly ignore wheel events unless focused, so plain panel
        # scrolling works everywhere and deliberate in-widget scrolling still
        # works once a widget is clicked/tabbed into.
        inner = QWidget(); self.setWidget(inner)
        root = QVBoxLayout(inner)
        root.setContentsMargins(4,4,4,4); root.setSpacing(6)

        def grp(t): g=CollapsibleGroupBox(t); l=QGridLayout(); l.setSpacing(3); g.setLayout(l); return g,l

        def sp(gl, row, label, attr, lo, hi, val, dbl=False, step=1, dec=1, tip=""):
            lw = QLabel(label)
            if tip: lw.setToolTip(tip)
            gl.addWidget(lw, row, 0)
            w = NoScrollDoubleSpinBox() if dbl else NoScrollSpinBox()
            if dbl: w.setDecimals(dec); w.setSingleStep(step)
            else:   w.setSingleStep(int(step))
            w.setRange(lo,hi); w.setValue(val)
            if tip: w.setToolTip(tip)
            gl.addWidget(w,row,1)
            setattr(self,attr,w); w.valueChanged.connect(lambda _:self.params_changed.emit())
            return row+1

        g,gl=grp("Detection"); row=0
        row=sp(gl,row,"Diameter [px]","sp_diam",3,299,19,step=2,
               tip="Expected particle diameter in pixels (rounded to next odd number).\n"
                   "This is the most critical parameter — set it as close as possible\n"
                   "to the true particle size. Wrong values cause missed or spurious detections.")
        row=sp(gl,row,"Separation [px]","sp_sep",3,299,19,step=2,
               tip="Minimum allowed centre-to-centre distance between two detected\n"
                   "particles (px). Set to approximately the particle diameter to prevent\n"
                   "nearby particles from being merged into one detection.")
        row=sp(gl,row,"Min mass","sp_mass",100,1_000_000,3000,step=500,
               tip="Minimum integrated brightness of a candidate particle (arbitrary units).\n"
                   "Raise to reject background noise and artefacts.\n"
                   "LOWER THIS to detect dim particles at grain boundaries / defect regions —\n"
                   "they are often 20-50% dimmer than particles in the bulk lattice.")
        row=sp(gl,row,"Percentile","sp_pct",1,99,80,
               tip="Pixels dimmer than this brightness percentile are zeroed before\n"
                   "feature-finding. Lower this value if particles near defects are\n"
                   "being missed — they typically sit in locally darker regions.")
        self.cb_bp=QCheckBox("Bandpass"); self.cb_bp.setChecked(True)
        self.cb_bp.setToolTip("Apply a bandpass filter before detection to suppress\n"
                              "high-frequency noise (< Lshort) and low-frequency illumination\n"
                              "gradients (> Llong). Disable for already-processed images.")
        gl.addWidget(self.cb_bp,row,0,1,2); row+=1
        row=sp(gl,row,"  Lshort","sp_ls",1,20,1,
               tip="Short-wavelength bandpass cutoff (px). Noise at scales smaller\n"
                   "than this is suppressed. Typically 1–2 px.")
        row=sp(gl,row,"  Llong","sp_ll",5,200,53,
               tip="Long-wavelength bandpass cutoff (px). Illumination gradients at\n"
                   "scales larger than this are suppressed. Should be somewhat larger\n"
                   "than the particle-to-particle spacing.")
        self.cb_inv=QCheckBox("Invert (dark on bright)")
        self.cb_inv.setToolTip("Tick if your particles appear dark on a bright background\n"
                               "(e.g. transmitted-light or phase-contrast microscopy).\n"
                               "Untick for bright particles on a dark background (fluorescence).")
        gl.addWidget(self.cb_inv,row,0,1,2); row+=1
        self.cb_ring=QCheckBox("Ring particles (dark rim, bright centre)")
        self.cb_ring.setToolTip(
            "Enable for particles whose centre matches the background but have a\n"
            "dark surrounding rim (e.g. semi-opaque colloidal particles in transmitted\n"
            "light).  Applies a zero-mean annular matched filter sized to Diameter before\n"
            "detection — creates a sharp intensity peak at each ring centre so that\n"
            "trackpy can find particles the standard bandpass alone would miss.\n"
            "Can be combined with Invert for particles with bright rims and dark centres.")
        gl.addWidget(self.cb_ring,row,0,1,2); row+=1
        self.cb_flatten=QCheckBox("Flatten illumination (correct wall/vignette dimming)")
        self.cb_flatten.setToolTip(
            "Divide the frame by a heavily-blurred estimate of its own background\n"
            "before denoise/gamma/CLAHE/bandpass, to correct large-scale illumination\n"
            "gradients — e.g. particles that are dimmer near a wall/edge of the field\n"
            "of view due to vignetting or shadowing. This is a DIFFERENT scale than\n"
            "the Llong bandpass cutoff above (which is tuned to particle spacing, not\n"
            "to FOV-scale gradients). Off by default — enable only if you see a real\n"
            "systematic brightness gradient across the frame, not just noise.")
        gl.addWidget(self.cb_flatten,row,0,1,2); row+=1
        row=sp(gl,row,"  Flatten sigma frac","sp_flatten_sf",0.02,0.5,0.15,dbl=True,step=0.01,dec=2,
               tip="Gaussian blur sigma used to estimate the illumination background,\n"
                   "as a fraction of the smaller image dimension. Larger = smoother/more\n"
                   "large-scale background estimate; too small will start removing real\n"
                   "particle-scale contrast rather than just the illumination gradient.")
        for w in(self.cb_bp,self.cb_inv,self.cb_ring,self.cb_flatten): w.toggled.connect(lambda _:self.params_changed.emit())
        root.addWidget(g)

        g_enh,gl_enh=grp("Image enhancement"); er=0
        er=sp(gl_enh,er,"Gamma","sp_gamma",0.2,3.0,1.0,dbl=True,step=0.05,dec=2,
              tip="Gamma correction applied before detection — to both live camera\n"
                  "frames and recorded-video analysis. >1 brightens shadows/midtones,\n"
                  "useful for semi-opaque particles with low contrast. 1.0 = off.")
        er=sp(gl_enh,er,"Sharpen","sp_sharpen",0.0,3.0,0.0,dbl=True,step=0.1,dec=1,
              tip="Unsharp-mask edge enhancement strength. Boosts particle-edge contrast\n"
                  "for semi-opaque/low-contrast particles. 0 = off; useful range 0.5–1.5.")
        gl_enh.addWidget(QLabel("Denoise"),er,0)
        self.combo_denoise=NoScrollComboBox()
        self.combo_denoise.addItem("Off","off")
        self.combo_denoise.addItem("Bilateral (fast, live-safe)","bilateral")
        self.combo_denoise.addItem("NLM (strong, slower)","nlm")
        self.combo_denoise.setToolTip(
            "Noise reduction applied before gamma/contrast/sharpen.\n"
            "Bilateral: edge-preserving, fast enough for live high-fps streams.\n"
            "NLM: stronger but much slower — best for retroactive re-analysis\n"
            "of recorded video rather than live tracking.")
        gl_enh.addWidget(self.combo_denoise,er,1)
        self.combo_denoise.currentIndexChanged.connect(lambda _:self.params_changed.emit())
        er+=1
        er=sp(gl_enh,er,"  Strength","sp_denoise",1.0,50.0,10.0,dbl=True,step=1.0,dec=0,
              tip="Denoise filter strength (bilateral: sigma; NLM: h parameter).\n"
                  "Higher = smoother but risks blurring small/dim particles.")
        root.addWidget(g_enh)

        self.cb_live_preview = QCheckBox("Live preview during tracking")
        self.cb_live_preview.setChecked(True)
        self.cb_live_preview.setToolTip(
            "Update the annotated preview frame as tracking progresses.\n"
            "Disable to skip per-frame preview rendering (only the final result\n"
            "is shown) — reduces GUI-thread overhead and can speed up tracking\n"
            "on long videos.")
        self.cb_live_preview.toggled.connect(lambda _:self.params_changed.emit())
        root.addWidget(self.cb_live_preview)

        g2,gl2=grp("Linking"); r2=0
        r2=sp(gl2,r2,"Max step [µm]","sp_step",0.1,20.,3.,dbl=True,step=0.1,dec=2,
               tip="Maximum distance (µm) a particle may travel between consecutive\n"
                   "frames and still be linked to the same track.")
        r2=sp(gl2,r2,"Memory [fr]","sp_mem",0,20,3,
               tip="Number of consecutive frames a particle can be absent (missing\n"
                   "detection) before its track is terminated. Useful for particles\n"
                   "that occasionally blink or are occluded.")
        r2=sp(gl2,r2,"Min track len","sp_mlen",2,200,15,
               tip="Tracks shorter than this many frames are discarded.\n"
                   "Raise to eliminate transient noise detections; lower if real\n"
                   "particles near boundaries have short lifetimes in the field of view.")
        r2=sp(gl2,r2,"Edge margin [px]","sp_edge",0,100,38,
               tip="Particles within this many pixels of the image border are excluded\n"
                   "(they are partially out of frame and skew statistics).")
        r2=sp(gl2,r2,"Appear thresh","sp_athr",1,50,5,
               tip="A particle that first appears or last disappears more than this many\n"
                   "frames from the analysis start/end is flagged as an event (ring overlay).")
        root.addWidget(g2)

        g3,gl3=grp("Defect detection"); r3=0
        r3=sp(gl3,r3,"Pair dist [px]","sp_pd",5,200,48,step=2,
               tip="Maximum centre-to-centre distance (px) between a 5-fold and a 7-fold\n"
                   "coordinated particle to count as a dislocation pair.\n"
                   "Set to roughly 1–1.5× the particle spacing.")
        r3=sp(gl3,r3,"LAGB eps [px]","sp_le",10,500,76,step=4,
               tip="DBSCAN neighbourhood radius (px) for clustering dislocation pairs into\n"
                   "grain boundaries. Larger values merge more pairs; useful for curved\n"
                   "or diffuse boundaries.")
        r3=sp(gl3,r3,"LAGB min","sp_lm",2,20,3,
               tip="Minimum number of dislocation pairs that must cluster together to be\n"
                   "classified as a grain boundary line.")
        r3=sp(gl3,r3,"LAGB aspect","sp_la",1.,10.,2.,dbl=True,step=0.1,dec=1,
               tip="Minimum aspect ratio of the dislocation cluster to be accepted as a\n"
                   "line-like boundary. Higher values reject compact blobs.")
        r3=sp(gl3,r3,"LAGB angle [°]","sp_lang",1.,30.,15.,dbl=True,step=0.5,dec=1,
               tip="Misorientation angle threshold (°). Boundaries with mean misorientation\n"
                   "below this are classified as Low-Angle (LAGB); above as High-Angle (HAGB).")
        r3=sp(gl3,r3,"Max neighbour dist [µm]","sp_maxnbr",0.0,50.,0.0,dbl=True,step=0.1,dec=2,
               tip="Maximum physical distance (µm) between two particles for them to be\n"
                   "counted as structural neighbours (ψ₆, coordination number, grain\n"
                   "boundaries, cage-relative coordinates). Delaunay triangulation and\n"
                   "k-nearest-neighbour queries both create long, physically-meaningless\n"
                   "edges at cluster boundaries or in dilute regions — this filters them\n"
                   "out after the fact. 0 = disabled (no cutoff, legacy behaviour).")
        root.addWidget(g3)

        # ── Structural correlations: g(r) / g6(r) ───────────────────────
        # Separate concern from "Defect detection" above: that section's
        # Max neighbour dist filters the Delaunay nearest-neighbour graph
        # (coordination number / grain boundaries); this section's Max r /
        # Bin width instead control an independent pairwise-distance
        # histogram (full pair population, not just nearest neighbours).
        g7,gl7=grp("Structural correlations"); r7=0
        gl7.addWidget(QLabel("Frame range"),r7,0)
        self.combo_gr_range=NoScrollComboBox()
        self.combo_gr_range.addItem("Current frame","current")
        self.combo_gr_range.addItem("Full trajectory (averaged)","full")
        self.combo_gr_range.addItem("Custom range","custom")
        self.combo_gr_range.setToolTip(
            "Which frame(s) to compute g(r)/g6(r) over.\n"
            "Current frame: fast, runs immediately on the GUI thread.\n"
            "Full trajectory (averaged): statistically accumulates raw pair\n"
            "histograms across every analysed frame before normalising once —\n"
            "more representative but can be slow on long videos, so it runs in\n"
            "a background thread.\n"
            "Custom range: same background-thread averaging, restricted to the\n"
            "start/end frames below.")
        gl7.addWidget(self.combo_gr_range,r7,1); r7+=1
        self.combo_gr_range.currentIndexChanged.connect(self._on_gr_range_mode_changed)

        r7=sp(gl7,r7,"Start frame","sp_gr_start",0,999999,0,
               tip="First frame included when Frame range = Custom range.")
        r7=sp(gl7,r7,"End frame","sp_gr_end",0,999999,100,
               tip="Last frame included when Frame range = Custom range.")

        r7=sp(gl7,r7,"Max r [µm]","sp_gr_rmax",0.5,100.0,10.0,dbl=True,step=0.5,dec=2,
               tip="Maximum pair separation (µm) included in g(r)/g6(r). Auto-filled once\n"
                   "from ~10× the particle spacing when this section is first opened —\n"
                   "edit freely afterwards, it will not be recomputed automatically.\n"
                   "Independent from the Max neighbour dist filter above (that only\n"
                   "affects the Delaunay nearest-neighbour graph, not this pairwise\n"
                   "histogram).")
        r7=sp(gl7,r7,"Bin width [µm]","sp_gr_dr",0.01,5.0,1.0,dbl=True,step=0.01,dec=3,
               tip="Histogram bin width (µm) for g(r)/g6(r). Auto-filled once from\n"
                   "Diameter/10 when this section is first opened — edit freely\n"
                   "afterwards. Smaller = finer radial resolution but noisier bins.")

        self.cb_gr_exclude_boundary=QCheckBox("Exclude boundary particles as pair centres")
        self.cb_gr_exclude_boundary.setToolTip(
            "Edge correction: when checked, only particles farther than Max r from\n"
            "the detected-point bounding box edge are used as pair centres — removes\n"
            "the bias from particles near the field-of-view edge having artificially\n"
            "fewer neighbours counted. Off by default (simplest, fastest; the bias is\n"
            "usually small unless Max r is a large fraction of the field of view).")
        gl7.addWidget(self.cb_gr_exclude_boundary,r7,0,1,2); r7+=1
        self.cb_gr_exclude_boundary.toggled.connect(lambda _:self.params_changed.emit())

        self.btn_gr_compute=QPushButton("Compute structure functions")
        self.btn_gr_compute.setToolTip(
            "Compute g(r) and g6(r) over the selected frame range and plot them in\n"
            "the Diagnostics panel. Requires a completed tracking/analysis run.")
        self.btn_gr_compute.setEnabled(False)
        gl7.addWidget(self.btn_gr_compute,r7,0,1,2); r7+=1

        self.btn_gr_export=QPushButton("Export CSV…")
        self.btn_gr_export.setToolTip(
            "Save the currently displayed g(r)/g6(r) curve to a CSV file\n"
            "(columns: r_um, g_r, g6_r, g6_over_g_r).")
        self.btn_gr_export.setEnabled(False)
        gl7.addWidget(self.btn_gr_export,r7,0,1,2); r7+=1

        for w in(self.sp_gr_start,self.sp_gr_end):
            w.setEnabled(False)   # only meaningful for "Custom range"
        root.addWidget(g7)
        self._gr_defaults_set = False   # auto-fill Max r/Bin width once, on first show
        # Auto-fill Max r / Bin width the first time this section is expanded
        # (Decision 4: compute once when first shown/opened, not on every open).
        g7.toggle_btn.clicked.connect(lambda _: self.maybe_autofill_gr_defaults())

        g5,gl5=grp("Dirt / debris rejection"); r5=0
        self.cb_ecc_filter=QCheckBox("Reject elongated candidates (eccentricity)")
        self.cb_ecc_filter.setToolTip(
            "Reject detected candidates whose eccentricity exceeds the threshold\n"
            "below. Real colloidal particles are close to circular; irregular\n"
            "debris/dirt is often elongated. Off by default since ecc is already\n"
            "reported (Diagnostics tab) without being used to reject anything.")
        gl5.addWidget(self.cb_ecc_filter,r5,0,1,2); r5+=1
        r5=sp(gl5,r5,"  Max eccentricity","sp_eccmax",0.0,1.0,0.8,dbl=True,step=0.05,dec=2,
               tip="Candidates with eccentricity above this value are rejected when the\n"
                   "checkbox above is enabled. 0 = perfectly circular, 1 = a line.")
        self.cb_size_outlier=QCheckBox("Reject size outliers (adaptive, per-frame)")
        self.cb_size_outlier.setToolTip(
            "Reject candidates whose mass is a statistical outlier relative to the\n"
            "OTHER detections in the same frame (median ± N×MAD, robust to a few\n"
            "outliers) rather than a fixed universal threshold — adapts automatically\n"
            "across videos/magnifications. Targets polydisperse dirt/debris that is\n"
            "much bigger or smaller than the real (near-monodisperse) particles.")
        gl5.addWidget(self.cb_size_outlier,r5,0,1,2); r5+=1
        r5=sp(gl5,r5,"  Outlier MAD mult","sp_madmult",1.0,10.0,2.5,dbl=True,step=0.1,dec=1,
               tip="Number of MADs (median absolute deviations, rescaled to be\n"
                   "std-equivalent) away from the frame's median mass before a\n"
                   "candidate is rejected as a size outlier. Lower = stricter.")
        for w in(self.cb_ecc_filter,self.cb_size_outlier): w.toggled.connect(lambda _:self.params_changed.emit())
        root.addWidget(g5)

        g6,gl6=grp("Performance"); r6=0
        self.cb_lattice_pred=QCheckBox("Crystal-lattice-predicted sparse search")
        self.cb_lattice_pred.setToolTip(
            "Speeds up detection on dense, near-crystalline packings: particles\n"
            "whose local neighbourhood looks well-ordered (spacing regularity, not\n"
            "the full ψ6 pipeline) have their candidate search restricted to a\n"
            "small region around a position predicted from the previous frame,\n"
            "instead of scanning the whole frame. Defect/boundary particles and a\n"
            "periodic full-frame reconciliation pass are always still searched in\n"
            "full, so new/lost particles are still found. Does NOT reduce the\n"
            "precision of reported positions — every candidate found is still\n"
            "measured from full-resolution pixel data exactly as before. Offline\n"
            "video analysis only (batch/tracking tab); off by default.")
        gl6.addWidget(self.cb_lattice_pred,r6,0,1,2); r6+=1
        self.cb_lattice_pred.toggled.connect(lambda _:self.params_changed.emit())
        root.addWidget(g6)

        g4,gl4=grp("Physical / frame window"); r4=0
        r4=sp(gl4,r4,"µm/px","sp_pxum",0.0001,10.,0.11,dbl=True,step=0.001,dec=4,
               tip="Physical pixel size in micrometres. Used to convert pixel coordinates\n"
                   "to real-space units for MSD and diffusion calculations.\n"
                   "⚠ CAMERA-DEPENDENT — recalculate after any camera/optics change:\n"
                   "px_um = sensor pixel pitch (µm) ÷ total system magnification.\n"
                   "(a2A2840-48umPRO pixel pitch = 2.74 µm.) Left unchanged after a\n"
                   "camera swap, this silently produces wrong physical units.")
        r4=sp(gl4,r4,"s/frame","sp_dt",0.,100.,0.,dbl=True,step=0.001,dec=5,
               tip="Frame interval in seconds. Set to 0 to read automatically from the\n"
                   "video frame rate.")
        r4=sp(gl4,r4,"Start frame","sp_sf",0,999999,0,
               tip="Index of the first video frame to include in the analysis.")
        r4=sp(gl4,r4,"End frame","sp_ef",-1,999999,500,
               tip="Index of the last video frame to include (-1 = last frame in the file).")
        root.addWidget(g4)

        root.addStretch()

    # ── Structural correlations helpers ─────────────────────────────────

    def _on_gr_range_mode_changed(self, _idx=None):
        mode = self.combo_gr_range.currentData()
        custom = (mode == "custom")
        self.sp_gr_start.setEnabled(custom)
        self.sp_gr_end.setEnabled(custom)
        self.params_changed.emit()

    def maybe_autofill_gr_defaults(self):
        """Auto-scale Max r / Bin width from the current Diameter parameter,
        but only the FIRST time this is called (e.g. when the Structural
        correlations section is first shown) — never overwrites a value the
        user has since edited themselves."""
        if self._gr_defaults_set:
            return
        self._gr_defaults_set = True
        diam_px = float(self.sp_diam.value())
        px_um   = float(self.sp_pxum.value())
        diam_um = diam_px * px_um if px_um > 0 else diam_px
        if diam_um <= 0:
            return
        self.sp_gr_rmax.blockSignals(True)
        self.sp_gr_rmax.setValue(min(max(10.0 * diam_um, self.sp_gr_rmax.minimum()),
                                      self.sp_gr_rmax.maximum()))
        self.sp_gr_rmax.blockSignals(False)
        self.sp_gr_dr.blockSignals(True)
        self.sp_gr_dr.setValue(min(max(diam_um / 10.0, self.sp_gr_dr.minimum()),
                                    self.sp_gr_dr.maximum()))
        self.sp_gr_dr.blockSignals(False)

    def get(self) -> dict:
        return {
            "diameter":self.sp_diam.value(),"separation":self.sp_sep.value(),
            "minmass":self.sp_mass.value(),"percentile":self.sp_pct.value(),
            "use_bandpass":self.cb_bp.isChecked(),"lshort":self.sp_ls.value(),"llong":self.sp_ll.value(),
            "invert":self.cb_inv.isChecked(),"ring_mode":self.cb_ring.isChecked(),
            "flatten_illum":self.cb_flatten.isChecked(),"flatten_sigma_frac":self.sp_flatten_sf.value(),
            "max_step_um":self.sp_step.value(),"memory":self.sp_mem.value(),"min_len":self.sp_mlen.value(),
            "edge_px":self.sp_edge.value(),"appear_thresh":self.sp_athr.value(),
            "pair_dist_px":self.sp_pd.value(),"lagb_eps_px":self.sp_le.value(),
            "lagb_min_n":self.sp_lm.value(),"lagb_aspect":self.sp_la.value(),"lagb_angle_deg":self.sp_lang.value(),
            "max_neighbor_dist_um":self.sp_maxnbr.value(),
            "use_ecc_filter":self.cb_ecc_filter.isChecked(),"ecc_max":self.sp_eccmax.value(),
            "reject_size_outliers":self.cb_size_outlier.isChecked(),
            "use_lattice_prediction":self.cb_lattice_pred.isChecked(),
            "size_outlier_mad_mult":self.sp_madmult.value(),
            "px_um":self.sp_pxum.value(),"dt":self.sp_dt.value(),
            "start_fr":self.sp_sf.value(),"end_fr":self.sp_ef.value(),"chunk":50,
            "gamma":self.sp_gamma.value(),"sharpen_amount":self.sp_sharpen.value(),
            "denoise_method":self.combo_denoise.currentData(),
            "denoise_strength":self.sp_denoise.value(),
            "live_preview":self.cb_live_preview.isChecked(),
            "gr_frame_range_mode":self.combo_gr_range.currentData(),
            "gr_start_fr":self.sp_gr_start.value(),"gr_end_fr":self.sp_gr_end.value(),
            "gr_rmax_um":self.sp_gr_rmax.value(),"gr_dr_um":self.sp_gr_dr.value(),
            "gr_exclude_boundary":self.cb_gr_exclude_boundary.isChecked(),
        }

    def apply(self, d: dict):
        """Load values from a dict (e.g. loaded settings)."""
        _set = lambda w, v: (w.blockSignals(True), w.setValue(v), w.blockSignals(False))
        _cb  = lambda w, v: (w.blockSignals(True), w.setChecked(v), w.blockSignals(False))
        if "diameter"    in d: _set(self.sp_diam,  d["diameter"])
        if "separation"  in d: _set(self.sp_sep,   d["separation"])
        if "minmass"     in d: _set(self.sp_mass,  d["minmass"])
        if "percentile"  in d: _set(self.sp_pct,   d["percentile"])
        if "use_bandpass"in d: _cb(self.cb_bp,      d["use_bandpass"])
        if "lshort"      in d: _set(self.sp_ls,    d["lshort"])
        if "llong"       in d: _set(self.sp_ll,    d["llong"])
        if "invert"      in d: _cb(self.cb_inv,     d["invert"])
        if "ring_mode"   in d: _cb(self.cb_ring,    d["ring_mode"])
        if "flatten_illum"in d:_cb(self.cb_flatten, d["flatten_illum"])
        if "flatten_sigma_frac"in d:_set(self.sp_flatten_sf,d["flatten_sigma_frac"])
        if "max_step_um" in d: _set(self.sp_step,  d["max_step_um"])
        if "memory"      in d: _set(self.sp_mem,   d["memory"])
        if "min_len"     in d: _set(self.sp_mlen,  d["min_len"])
        if "edge_px"     in d: _set(self.sp_edge,  d["edge_px"])
        if "appear_thresh"in d:_set(self.sp_athr,  d["appear_thresh"])
        if "pair_dist_px"in d: _set(self.sp_pd,    d["pair_dist_px"])
        if "lagb_eps_px" in d: _set(self.sp_le,    d["lagb_eps_px"])
        if "lagb_min_n"  in d: _set(self.sp_lm,    d["lagb_min_n"])
        if "lagb_aspect" in d: _set(self.sp_la,    d["lagb_aspect"])
        if "lagb_angle_deg"in d:_set(self.sp_lang, d["lagb_angle_deg"])
        if "max_neighbor_dist_um"in d:_set(self.sp_maxnbr,d["max_neighbor_dist_um"])
        if "use_ecc_filter"in d:_cb(self.cb_ecc_filter,d["use_ecc_filter"])
        if "ecc_max"     in d: _set(self.sp_eccmax,d["ecc_max"])
        if "reject_size_outliers"in d:_cb(self.cb_size_outlier,d["reject_size_outliers"])
        if "use_lattice_prediction"in d:_cb(self.cb_lattice_pred,d["use_lattice_prediction"])
        if "size_outlier_mad_mult"in d:_set(self.sp_madmult,d["size_outlier_mad_mult"])
        if "px_um"       in d: _set(self.sp_pxum,  d["px_um"])
        if "dt"          in d: _set(self.sp_dt,    d["dt"])
        if "start_fr"    in d: _set(self.sp_sf,    d["start_fr"])
        if "end_fr"      in d: _set(self.sp_ef,    d["end_fr"])
        if "gamma"       in d: _set(self.sp_gamma,  d["gamma"])
        if "sharpen_amount"in d:_set(self.sp_sharpen,d["sharpen_amount"])
        if "denoise_strength"in d:_set(self.sp_denoise,d["denoise_strength"])
        if "denoise_method" in d:
            self.combo_denoise.blockSignals(True)
            idx = self.combo_denoise.findData(d["denoise_method"])
            if idx >= 0: self.combo_denoise.setCurrentIndex(idx)
            self.combo_denoise.blockSignals(False)
        if "live_preview" in d: _cb(self.cb_live_preview, d["live_preview"])
        if "gr_frame_range_mode" in d:
            self.combo_gr_range.blockSignals(True)
            idx = self.combo_gr_range.findData(d["gr_frame_range_mode"])
            if idx >= 0: self.combo_gr_range.setCurrentIndex(idx)
            self.combo_gr_range.blockSignals(False)
            self._on_gr_range_mode_changed()
        if "gr_start_fr" in d: _set(self.sp_gr_start, d["gr_start_fr"])
        if "gr_end_fr"   in d: _set(self.sp_gr_end,   d["gr_end_fr"])
        if "gr_rmax_um"  in d:
            _set(self.sp_gr_rmax, d["gr_rmax_um"]); self._gr_defaults_set = True
        if "gr_dr_um"    in d:
            _set(self.sp_gr_dr,   d["gr_dr_um"]);   self._gr_defaults_set = True
        if "gr_exclude_boundary" in d: _cb(self.cb_gr_exclude_boundary, d["gr_exclude_boundary"])


# ════════════════════════════════════════════════════════════════════
# Layer panel
# ════════════════════════════════════════════════════════════════════

class LayerPanel(QScrollArea):
    layer_changed = pyqtSignal()

    _ITEMS = [
        ("6-fold circles",   "show_6fold","col_6fold"),
        ("5-fold circles",   "show_5fold","col_5fold"),
        ("7-fold circles",   "show_7fold","col_7fold"),
        ("Delaunay bonds",   "show_bonds","col_bonds"),
        ("5-7 dislocations", "show_pairs","col_pairs"),
        ("Low-angle GB",     "show_lagb", "col_lagb"),
        ("High-angle GB",    "show_hagb", "col_hagb"),
        ("Appear/disappear", "show_flags","col_flags"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        inner=QWidget(); self.setWidget(inner)
        self._layer=LayerConfig()
        self._cb_mode=False
        root=QVBoxLayout(inner); root.setContentsMargins(4,4,4,4); root.setSpacing(6)

        vis=CollapsibleGroupBox("Layers & colours"); vl=QGridLayout(); vl.setSpacing(3); vis.setLayout(vl)
        for row,(label,va,ca) in enumerate(self._ITEMS):
            cb=QCheckBox(label); cb.setChecked(getattr(self._layer,va))
            btn=ColourBtn(getattr(self._layer,ca))
            vl.addWidget(cb,row,0); vl.addWidget(btn,row,1)
            def _cv(v,_va=va): setattr(self._layer,_va,v); self.layer_changed.emit()
            def _cc(b,_ca=ca): setattr(self._layer,_ca,b); self.layer_changed.emit()
            cb.toggled.connect(_cv); btn.colour_changed.connect(_cc)
        root.addWidget(vis)

        geo=CollapsibleGroupBox("Geometry"); gl=QGridLayout(); gl.setSpacing(3); geo.setLayout(gl)
        for grow,(lbl,attr,lo,hi,val,tip) in enumerate([
            ("Circle radius","radius",2,30,6,""),
            ("Appear/disappear marker duration [fr]","flag_frames",1,30,8,
             "How many frames the appear/disappear ring marker stays visible\n"
             "around a particle after it appears or before it disappears."),
            ("Bond thickness","bond_thick",1,5,1,""),
        ]):
            lw=QLabel(lbl)
            if tip: lw.setToolTip(tip)
            gl.addWidget(lw,grow,0)
            w=NoScrollSpinBox(); w.setRange(lo,hi); w.setValue(val)
            if tip: w.setToolTip(tip)
            gl.addWidget(w,grow,1)
            def _g(v,a=attr): setattr(self._layer,a,v); self.layer_changed.emit()
            w.valueChanged.connect(_g)
        root.addWidget(geo)

        self.cb_hud=QCheckBox("Show defect-count overlay"); self.cb_hud.setChecked(True)
        self.cb_hud.setToolTip(
            "Show a text overlay in the corner of the video with the current\n"
            "5-7 dislocation-pair count, grain-boundary count, and frame number.")
        self.cb_hud.toggled.connect(lambda v:(setattr(self._layer,"show_hud",v),self.layer_changed.emit()))
        root.addWidget(self.cb_hud)

        self.cb_cbmode=QCheckBox("Colorblind-friendly colours (no red/green)")
        self.cb_cbmode.setToolTip(
            "Replaces red/green colours with the Wong (2011) deuteranopia-safe palette:\n"
            "  5-fold  → orange   (#E69F00)\n"
            "  7-fold  → sky blue (#56B4E9)\n"
            "  pairs   → mauve    (#CC79A7)\n"
            "  LAGB    → teal     (#009E73)\n"
            "  HAGB    → yellow   (#F0E442)\n"
            "  flags   → blue     (#0072B2)"
        )
        self.cb_cbmode.toggled.connect(
            lambda v: (setattr(self,"_cb_mode",v), self.layer_changed.emit())
        )
        root.addWidget(self.cb_cbmode)
        root.addStretch()

    def cfg(self) -> LayerConfig:
        if self._cb_mode:
            return LayerConfig.from_dict({**self._layer.to_dict(), **_CB_COLORS})
        return self._layer

    def apply(self, d: dict):
        lc = LayerConfig.from_dict(d)
        self._layer = lc
        # rebuild checkboxes / colours from the new config
        # (simpler: just update backing data; widgets sync on next repaint via signal)
        self.layer_changed.emit()


# ════════════════════════════════════════════════════════════════════
# Tracking progress dialog
# ════════════════════════════════════════════════════════════════════

_HELP_HTML = """<html>
<body style="background-color:#1a1a2e; color:#ddd; font-family:Segoe UI, Arial, sans-serif; font-size:13px; line-height:1.5; padding:12px;">

<h1 style="color:#44aaff; border-bottom:1px solid #44aaff; padding-bottom:4px;">Parameter Tuning Guide</h1>
<p style="color:#bbb;">Work top-to-bottom: get <b>Diameter</b> right first, then refine detection, then tune linking. Use <b>Preview frame</b> after every change so you see the effect immediately without running a full analysis.</p>

<h2 style="color:#44aaff;">Recommended Workflow</h2>
<ol>
<li><b>Load the video.</b> Auto grain-size detection runs on the first frame and suggests a <b>Diameter [px]</b>. Trust it as a starting point.</li>
<li><b>Check focus first</b> (see Focus section). No parameter fixes a badly focused image.</li>
<li><b>Set Diameter.</b> Confirm with the auto suggestion, the <b>Auto size</b> button (live camera), or by eye. This is the single most important parameter.</li>
<li><b>Press Preview frame.</b> Are roughly the right particles marked? Adjust detection params and re-preview.</li>
<li><b>Tune detection</b> in this order: <code style="background:#333; padding:1px 4px;">Invert</code> &rarr; <code style="background:#333; padding:1px 4px;">Diameter</code> &rarr; <code style="background:#333; padding:1px 4px;">Min mass</code> &rarr; <code style="background:#333; padding:1px 4px;">Percentile</code> &rarr; bandpass.</li>
<li><b>Tune linking</b> (<b>Max step</b>, <b>Memory</b>, <b>Min length</b>) only after detection looks clean.</li>
<li><b>Run full analysis.</b> If the grain-size consistency check warns you, your <b>Diameter</b> is probably wrong — go back to step 3.</li>
</ol>
<p style="background-color:#2a2a1a; border-left:3px solid #ffcc44; padding:6px;"><span style="color:#ffcc44;">Rule of thumb:</span> change <b>one parameter at a time</b> and re-Preview. Detection problems are almost always solved in detection, not in linking.</p>

<h2 style="color:#44aaff;">Microscope Focus</h2>
<p>In bright-field, well-focused monodisperse colloids look like crisp discs with a consistent dark ring. Focus directly controls how well every parameter works.</p>
<ul>
<li><b>Sharp, even contrast:</b> particles have a clear edge and uniform appearance. This is the target.</li>
<li><b>Out of focus:</b> particles blur into the background &rarr; low contrast &rarr; <b>Min mass</b> and <b>Percentile</b> become hard to set, and detection drops particles. Refocus before touching parameters.</li>
<li><b>Over/under-focus flips contrast:</b> particles can appear bright-on-dark or dark-on-bright. In bright-field, particles are usually <b>darker</b> than the background &rarr; enable <b>Invert</b>.</li>
<li><b>Halos / diffraction rings:</b> strong focus rings inflate the apparent particle size. Focus so the core disc dominates the ring, then re-run Auto size.</li>
<li><b>Uneven illumination</b> &rarr; use <b>Bandpass L long</b> or <b>Flatten illumination</b> rather than fighting it with <b>Percentile</b>.</li>
</ul>

<h2 style="color:#44aaff;">Diagnosing &amp; Fixing Problems</h2>

<h3 style="color:#66bbff;">Too few particles detected</h3>
<p><b>Diagnose</b> (use Preview frame): many obvious particles have no marker; markers cluster only on the brightest.</p>
<ul>
<li>Wrong <code style="background:#333; padding:1px 4px;">Invert</code> — if particles are dark on a bright field, enable <b>Invert</b>. This is the most common cause.</li>
<li><b>Min mass</b> too high &rarr; lower it until dim particles appear.</li>
<li><b>Percentile</b> too high &rarr; lower it (e.g. 80 &rarr; 64) so dimmer pixels survive thresholding.</li>
<li><b>Diameter</b> too large &rarr; trackpy merges/misses real particles. Reduce toward the true size.</li>
<li>Dim particles &rarr; raise <b>Gamma</b> (brightens midtones) and/or <b>Sharpen</b> to boost local contrast.</li>
<li>Illumination gradient hiding particles &rarr; enable <b>Bandpass</b> and set <b>L long</b> to roughly the gradient scale.</li>
</ul>

<h3 style="color:#66bbff;">Too many spurious detections (noise/artefacts)</h3>
<p><b>Diagnose:</b> markers on empty background, on speckle, or several markers on one particle.</p>
<ul>
<li><b>Min mass</b> too low &rarr; raise it until noise specks drop out but real particles stay.</li>
<li><b>Percentile</b> too low &rarr; raise it to zero out low-brightness noise before detection.</li>
<li>Pixel noise &rarr; enable <b>Bandpass</b> and set <b>L short</b> to ~1–2 px to smooth sub-particle noise.</li>
<li>Multiple markers per particle &rarr; <b>Diameter</b> too small, or <b>Separation</b> too small. Increase <b>Separation</b> toward the true centre-to-centre spacing.</li>
<li>Edge/border artefacts &rarr; increase <b>Edge margin</b> to exclude near-border detections.</li>
</ul>

<h3 style="color:#66bbff;">Particles detected at the wrong size</h3>
<p><b>Diagnose:</b> the consistency-check banner fires; auto grain-size disagrees with your value by &gt;2 px; markers look too tight or too loose on the discs.</p>
<ul>
<li><b>Diameter</b> must be an <b>odd integer</b> approximating the full particle diameter including its dark ring.</li>
<li>On live camera, press <b>Auto size</b> — it scans 5–51 px and picks the diameter giving the roundest particles.</li>
<li>If diffraction halos inflate the size, refocus before trusting any auto value.</li>
</ul>

<h3 style="color:#66bbff;">Tracking dropouts (tracks terminating early)</h3>
<p><b>Diagnose:</b> many short tracks; particles you can follow by eye get split into several track IDs.</p>
<ul>
<li><b>Max step</b> too small &rarr; fast particles jump farther than allowed. Increase <b>Max step [&micro;m]</b>.</li>
<li>Particles blink out for a frame &rarr; increase <b>Memory</b> so a brief disappearance does not end the track.</li>
<li>Underlying cause is detection flicker &rarr; fix detection first. A particle dropped on some frames cannot be linked.</li>
<li><b>Min length</b> set too high &rarr; you may be discarding valid short tracks; lower it if appropriate.</li>
</ul>

<h3 style="color:#66bbff;">Particles near grain boundaries being missed</h3>
<p><b>Diagnose:</b> gaps in detection along dislocation lines; the <b>Defect plot</b> looks noisier than the structure warrants.</p>
<ul>
<li>Particles are closer together at boundaries &rarr; <b>Separation</b> too large merges them. Reduce <b>Separation</b> toward the true minimum spacing.</li>
<li>Local distortion changes apparent brightness &rarr; lower <b>Min mass</b> slightly and/or raise <b>Gamma</b>.</li>
<li><b>Diameter</b> slightly too large blurs over closely-packed defect cores &rarr; nudge it down by 2 px and Preview.</li>
</ul>

<h3 style="color:#66bbff;">Noisy / inconsistent tracks</h3>
<p><b>Diagnose:</b> positions jitter frame-to-frame beyond real motion; track IDs swap between neighbouring particles.</p>
<ul>
<li>Sub-pixel jitter from poor contrast &rarr; refocus and/or enable <b>Bandpass</b> (L short ~1 px) to stabilise centroids.</li>
<li>ID swapping in dense regions &rarr; <b>Max step</b> too large relative to spacing; reduce it so neighbours are not candidate matches.</li>
<li>Spurious detections feeding the linker &rarr; clean detection first (raise <b>Min mass</b> / <b>Percentile</b>).</li>
</ul>

<h3 style="color:#66bbff;">Slow analysis</h3>
<ul>
<li>Detection runs at full resolution using parallel threads — no quality/speed trade-off needed.</li>
<li>Keep <b>Denoise</b> set to Off or Bilateral unless needed — NLM is comparatively expensive.</li>
<li>Larger <b>Diameter</b> and <b>Bandpass</b> kernels cost more — don&apos;t oversize them.</li>
</ul>

<h2 style="color:#44aaff;">What&apos;s Already Automated</h2>
<ul>
<li><b>Auto grain-size on load</b> — suggests Diameter from the first frame automatically.</li>
<li><b>Auto size button</b> (live camera) — scans 5–51 px, picks the diameter giving the roundest particles.</li>
<li><b>Grain-size consistency check</b> — after tracking, warns if detected size differs from the used Diameter by &gt;2 px.</li>
<li><b>Preview frame &amp; Compare mode</b> — immediate visual feedback so you can tune without full runs.</li>
</ul>

</body>
</html>"""


class HelpDialog(QDialog):
    """Scrollable parameter-tuning guide."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parameter Tuning Guide")
        self.resize(760, 620)
        lay = QVBoxLayout(self)
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setHtml(_HELP_HTML)
        lay.addWidget(txt)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.accept)
        lay.addWidget(bb)


class TrackingProgressDialog(QDialog):
    """
    Non-modal floating window that shows real-time tracking progress.
    Opened automatically when analysis starts; Close button enables on completion.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tracking in progress…")
        self.setMinimumWidth(500)
        self.setModal(False)
        lay = QVBoxLayout(self)

        self.prog = QProgressBar()
        self.prog.setRange(0, 100); self.prog.setValue(0)
        self.prog.setTextVisible(True); self.prog.setFormat("0%  Initialising…")
        lay.addWidget(self.prog)

        self.log_lbl = QLabel("Starting…")
        self.log_lbl.setWordWrap(True)
        self.log_lbl.setMinimumHeight(60)
        self.log_lbl.setStyleSheet("font-family:monospace;color:#aaa;")
        lay.addWidget(self.log_lbl)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._close_btn = bb.button(QDialogButtonBox.StandardButton.Close)
        self._close_btn.setEnabled(False)
        bb.rejected.connect(self.accept)
        lay.addWidget(bb)

    def set_progress(self, pct: int, msg: str):
        self.prog.setValue(pct)
        self.prog.setFormat(f"{pct}%  {msg}")
        self.log_lbl.setText(msg)

    def mark_done(self, msg: str = "Complete."):
        self.prog.setValue(100)
        self.prog.setFormat("100%  Done")
        self.log_lbl.setText(msg)
        self._close_btn.setEnabled(True)
        self.setWindowTitle("Tracking complete")

    def mark_error(self, msg: str):
        self.log_lbl.setText(f"Error:\n{msg[:500]}")
        self.log_lbl.setStyleSheet("font-family:monospace;color:#ff6666;")
        self._close_btn.setEnabled(True)
        self.setWindowTitle("Tracking failed")


# ════════════════════════════════════════════════════════════════════
# Log table + Defect plot  (unchanged from v1)
# ════════════════════════════════════════════════════════════════════

class LogTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay=QVBoxLayout(self); lay.setContentsMargins(2,2,2,2)
        lay.addWidget(QLabel("<b>Appear / Disappear Events</b>"))
        self.tbl=QTableWidget(0,5)
        self.tbl.setHorizontalHeaderLabels(["Frame","X [px]","Y [px]","Event","Track ID"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.tbl)
    def populate(self, evs):
        self.tbl.setRowCount(0)
        for ev in evs:
            r=self.tbl.rowCount(); self.tbl.insertRow(r)
            for c,v in enumerate([ev.frame,f"{ev.x_px:.1f}",f"{ev.y_px:.1f}",ev.kind,str(ev.track_id)]):
                self.tbl.setItem(r,c,QTableWidgetItem(str(v)))


class DefectPlot(QWidget):
    export_csv_requested = pyqtSignal()   # "Export CSV…" clicked; MainWindow wires the file dialog

    def __init__(self, parent=None):
        super().__init__(parent)
        lay=QVBoxLayout(self); lay.setContentsMargins(2,2,2,2)
        self.fig=Figure(figsize=(4,2.),facecolor="#1e1e1e",tight_layout=True)
        self.canvas=FigureCanvas(self.fig); lay.addWidget(self.canvas)
        self.ax=self.fig.add_subplot(111,facecolor="#1e1e1e")
        # Second y-axis: dislocation-pair COUNT (left, existing) vs. defect
        # FRACTION (right, new) are different physical quantities with very
        # different scales — sharing one axis would flatten one or the other.
        self.ax2=self.ax.twinx()
        self.ax.tick_params(colors="#aaa"); [s.set_color("#444") for s in self.ax.spines.values()]
        self.ax2.tick_params(colors="#aaa"); [s.set_color("#444") for s in self.ax2.spines.values()]
        self._b57=[]; self._blagb=[]; self._vl=None
        self._defect_df: pd.DataFrame | None = None

        self.btn_export=QPushButton("Export CSV…")
        self.btn_export.setToolTip(
            "Save the defect-concentration series (frame, frac_defect, frac_5fold,\n"
            "frac_7fold) to a CSV file. Enabled once tracking/analysis has completed.")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_csv_requested.emit)
        lay.addWidget(self.btn_export)

    def add_point(self,fr,n57,nl):
        self._b57.append((fr,n57)); self._blagb.append((fr,nl))
        if len(self._b57)%10==0: self._draw()

    def set_series(self,p,l,defect_df: "pd.DataFrame | None" = None):
        self._b57=list(p); self._blagb=list(l)
        if defect_df is not None:
            self._defect_df = defect_df
            self.btn_export.setEnabled(not defect_df.empty)
        self._draw()

    def mark_frame(self,fr):
        if self._vl:
            try: self._vl.remove()
            except: pass
        self._vl=self.ax.axvline(fr,color="#888",lw=0.8,ls="--")
        self.canvas.draw_idle()

    def _draw(self):
        self.ax.cla(); self.ax2.cla()
        self.ax.set_facecolor("#1e1e1e")
        self.ax.tick_params(colors="#aaa"); [s.set_color("#444") for s in self.ax.spines.values()]
        self.ax2.tick_params(colors="#aaa"); [s.set_color("#444") for s in self.ax2.spines.values()]
        # Sort by frame before plotting — points can arrive out of frame order
        # since structural analysis now runs in parallel across frames.
        lines, labels = [], []
        if self._b57:
            fr,n=zip(*sorted(self._b57))
            l1,=self.ax.plot(fr,n,lw=1,color="#dd44dd",label="5-7 pairs")
            lines.append(l1); labels.append("5-7 pairs")
        if self._blagb:
            fr,n=zip(*sorted(self._blagb))
            l2,=self.ax.plot(fr,n,lw=1,color="#44dddd",label="LAGBs")
            lines.append(l2); labels.append("LAGBs")
        if self._defect_df is not None and not self._defect_df.empty:
            df = self._defect_df.sort_values("frame")
            l3,=self.ax2.plot(df["frame"],df["frac_defect"],lw=1,color="#ffaa33",ls="-",label="Defect frac")
            l4,=self.ax2.plot(df["frame"],df["frac_5fold"],lw=1,color="#ff6644",ls="--",label="5-fold frac")
            l5,=self.ax2.plot(df["frame"],df["frac_7fold"],lw=1,color="#4488ff",ls="--",label="7-fold frac")
            lines += [l3,l4,l5]; labels += ["Defect frac","5-fold frac","7-fold frac"]
        self.ax.set_xlabel("Frame",color="#aaa")
        self.ax.set_ylabel("Count",color="#aaa")
        self.ax2.set_ylabel("Fraction",color="#aaa")
        if lines:
            self.ax.legend(lines,labels,fontsize=6,frameon=False,labelcolor="#aaa",loc="upper left")
        # Layout is computed once at Figure construction (tight_layout=True
        # above), matching DiagnosticsPanel's other figures — no per-draw
        # tight_layout() call needed.
        self.canvas.draw_idle()


class GrainWarningBar(QFrame):
    """Persistent warning banner shown when grain-size inconsistency is detected.

    Stays visible until the user explicitly clicks Dismiss.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame{background:#5a2800;border-bottom:2px solid #ff6600;}"
            "QLabel{color:#ffcc44;font-size:11px;background:transparent;}"
        )
        lay  = QHBoxLayout(self)
        lay.setContentsMargins(8, 5, 8, 5)
        icon = QLabel("⚠")
        icon.setStyleSheet("font-size:18px;color:#ff6600;background:transparent;")
        self._msg = QLabel("")
        self._msg.setWordWrap(True)
        btn = QPushButton("✕  Dismiss")
        btn.setFixedWidth(90)
        btn.setStyleSheet(
            "color:#ffcc44;background:#7a3800;border:1px solid #ff6600;"
            "padding:3px 6px;border-radius:3px;"
        )
        btn.clicked.connect(self.hide)
        lay.addWidget(icon, 0)
        lay.addWidget(self._msg, 1)
        lay.addWidget(btn, 0)
        self.hide()

    def warn(self, msg: str):
        self._msg.setText(msg)
        self.show()


# ════════════════════════════════════════════════════════════════════
# Diagnostics panel  (7 diagnostic plots + auto-tune buttons)
# ════════════════════════════════════════════════════════════════════

class DiagnosticsPanel(QWidget):
    """Dock panel with 6 matplotlib-based diagnostic tabs and auto-tune buttons."""
    apply_params = pyqtSignal(dict)   # emitted by auto-tune buttons

    # ── shared dark axes style ────────────────────────────────────
    _BG  = "#111122"
    _FG  = "#aaaacc"
    _SPL = "#2a2a4a"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._feats       = None   # latest detection DataFrame (x_px/y_px/mass/ecc)
        self._tracks      = None   # latest tracks DataFrame
        self._px_um       = 0.11
        self._params: dict = {}
        self._frame_bgr   = None   # BGR frame corresponding to _feats
        self._frame_counts: list[tuple[int,int]] = []
        self._sweep_worker: SweepWorker | None = None
        # Lazy-redraw bookkeeping (M3): only the tab currently visible in the
        # QTabWidget is redrawn immediately on update_detections/update_tracks;
        # the rest are marked dirty and flushed when the user switches to them,
        # since 5 of 6 tabs are hidden behind the active tab at any given time.
        self._dirty: set[str] = set()
        self._tabs: QTabWidget | None = None
        self._build()

    # ── build ─────────────────────────────────────────────────────

    def _build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(2, 2, 2, 2)
        tabs = QTabWidget(); tabs.setDocumentMode(True)

        # ── Tab 1: Mass distribution ──────────────────────────────
        self._fig_mass = Figure(figsize=(4, 2.8), facecolor=self._BG, tight_layout=True)
        self._ax_mass  = self._fig_mass.add_subplot(111, facecolor=self._BG)
        self._cv_mass  = FigureCanvas(self._fig_mass)
        btn_am = QPushButton("Auto Min mass")
        btn_am.setToolTip("Set Min mass to the Otsu threshold of the mass distribution")
        btn_am.clicked.connect(self._on_auto_min_mass)
        w1 = QWidget(); v1 = QVBoxLayout(w1); v1.setContentsMargins(2,2,2,2)
        v1.addWidget(self._cv_mass, stretch=1); v1.addWidget(btn_am)
        tabs.addTab(w1, "Mass distribution")

        # ── Tab 2: Eccentricity distribution ─────────────────────
        self._fig_ecc = Figure(figsize=(4, 2.8), facecolor=self._BG, tight_layout=True)
        self._ax_ecc  = self._fig_ecc.add_subplot(111, facecolor=self._BG)
        self._cv_ecc  = FigureCanvas(self._fig_ecc)
        tabs.addTab(self._cv_ecc, "Eccentricity")

        # ── Tab 3: Sub-pixel bias ─────────────────────────────────
        self._fig_spx = Figure(figsize=(4, 2.8), facecolor=self._BG, tight_layout=True)
        self._ax_spx_x, self._ax_spx_y = self._fig_spx.subplots(1, 2)
        self._cv_spx  = FigureCanvas(self._fig_spx)
        self._lbl_spx = QLabel("Run Preview frame to populate.")
        self._lbl_spx.setWordWrap(True)
        self._lbl_spx.setStyleSheet("color:#aaa;font-size:10px;padding:2px;")
        w3 = QWidget(); v3 = QVBoxLayout(w3); v3.setContentsMargins(2,2,2,2)
        v3.addWidget(self._cv_spx, stretch=1); v3.addWidget(self._lbl_spx)
        tabs.addTab(w3, "Sub-pixel bias")

        # ── Tab 4: Track lengths ──────────────────────────────────
        self._fig_tlen = Figure(figsize=(4, 2.8), facecolor=self._BG, tight_layout=True)
        self._ax_tlen  = self._fig_tlen.add_subplot(111, facecolor=self._BG)
        self._cv_tlen  = FigureCanvas(self._fig_tlen)
        btn_row4 = QHBoxLayout()
        btn_al = QPushButton("Auto Min length")
        btn_al.setToolTip("Set Min length to the 20th percentile of track lengths")
        btn_al.clicked.connect(self._on_auto_min_length)
        btn_as = QPushButton("Auto Max step")
        btn_as.setToolTip("Set Max step to 1.5 × 99th-percentile observed displacement")
        btn_as.clicked.connect(self._on_auto_max_step)
        btn_row4.addWidget(btn_al); btn_row4.addWidget(btn_as)
        w4 = QWidget(); v4 = QVBoxLayout(w4); v4.setContentsMargins(2,2,2,2)
        v4.addWidget(self._cv_tlen, stretch=1); v4.addLayout(btn_row4)
        tabs.addTab(w4, "Track lengths")

        # ── Tab 5: Per-frame detection count ─────────────────────
        self._fig_cnt = Figure(figsize=(4, 2.8), facecolor=self._BG, tight_layout=True)
        self._ax_cnt  = self._fig_cnt.add_subplot(111, facecolor=self._BG)
        self._cv_cnt  = FigureCanvas(self._fig_cnt)
        tabs.addTab(self._cv_cnt, "Frame counts")

        # ── Tab 6: Parameter sweep ────────────────────────────────
        self._fig_sw = Figure(figsize=(4, 2.8), facecolor=self._BG, tight_layout=True)
        self._ax_sw  = self._fig_sw.add_subplot(111, facecolor=self._BG)
        self._cv_sw  = FigureCanvas(self._fig_sw)
        sw_row = QHBoxLayout()
        self._combo_sw = NoScrollComboBox()
        self._combo_sw.addItems(["Min mass", "Percentile"])
        self._combo_sw.setFixedWidth(100)
        self._btn_sw = QPushButton("Run sweep")
        self._btn_sw.setToolTip(
            "Re-run detection on the current frame across a range of values for the\n"
            "chosen parameter, and plot how the detected particle count changes —\n"
            "helps you see how sensitive detection is to that parameter's setting.")
        self._combo_sw.setToolTip("Parameter to vary across a range of values for the sweep.")
        self._btn_sw.clicked.connect(self._on_run_sweep)
        sw_row.addWidget(QLabel("Sweep:")); sw_row.addWidget(self._combo_sw)
        sw_row.addWidget(self._btn_sw); sw_row.addStretch()
        w6 = QWidget(); v6 = QVBoxLayout(w6); v6.setContentsMargins(2,2,2,2)
        v6.addLayout(sw_row); v6.addWidget(self._cv_sw, stretch=1)
        tabs.addTab(w6, "Param sweep")

        # ── Tab 7: g(r) ────────────────────────────────────────────
        self._fig_gr = Figure(figsize=(4, 2.8), facecolor=self._BG, tight_layout=True)
        self._ax_gr  = self._fig_gr.add_subplot(111, facecolor=self._BG)
        self._cv_gr  = FigureCanvas(self._fig_gr)
        tabs.addTab(self._cv_gr, "g(r)")

        # ── Tab 8: g6(r) ───────────────────────────────────────────
        self._fig_g6 = Figure(figsize=(4, 2.8), facecolor=self._BG, tight_layout=True)
        self._ax_g6  = self._fig_g6.add_subplot(111, facecolor=self._BG)
        self._cv_g6  = FigureCanvas(self._fig_g6)
        tabs.addTab(self._cv_g6, "g6(r)")

        # ── Tab 9: g6(r)/g(r) ──────────────────────────────────────
        self._fig_g6g = Figure(figsize=(4, 2.8), facecolor=self._BG, tight_layout=True)
        self._ax_g6g  = self._fig_g6g.add_subplot(111, facecolor=self._BG)
        self._cv_g6g  = FigureCanvas(self._fig_g6g)
        tabs.addTab(self._cv_g6g, "g6(r)/g(r)")

        lay.addWidget(tabs)
        self._tabs = tabs
        # Tab index → plot name, used by _mark_dirty/_flush_dirty (M3).
        self._tab_plot_names = ["mass", "ecc", "spx", "tlen", "cnt", "sw", "gr", "g6", "g6g"]
        tabs.currentChanged.connect(self._on_tab_changed)
        self._pair_corr = None   # (r_centers, g_r, g6_r, g6_over_g_r), set by update_pair_correlations

    # ── style helper ──────────────────────────────────────────────

    def _sa(self, ax):
        ax.set_facecolor(self._BG)
        for sp in ax.spines.values(): sp.set_edgecolor(self._SPL)
        ax.tick_params(colors=self._FG, labelsize=7)
        ax.xaxis.label.set_color(self._FG)
        ax.yaxis.label.set_color(self._FG)

    # ── Lazy-redraw plumbing (M3) ────────────────────────────────────
    # Each entry is (tab_index, plot_method_name). "sw" (Param sweep) is
    # driven by its own button, not by update_detections/update_tracks, so
    # it's excluded here.

    _PLOT_TAB_INDEX = {"mass": 0, "ecc": 1, "spx": 2, "tlen": 3, "cnt": 4,
                        "gr": 6, "g6": 7, "g6g": 8}

    def _redraw_or_mark(self, name: str):
        """Redraw immediately if `name`'s tab is the active one; otherwise
        just flag it dirty so _on_tab_changed flushes it lazily when the
        user actually switches to that tab."""
        idx = self._PLOT_TAB_INDEX[name]
        if self._tabs is not None and self._tabs.currentIndex() == idx:
            getattr(self, f"_plot_{name}")()
        else:
            self._dirty.add(name)

    def _on_tab_changed(self, idx: int):
        if idx < 0 or idx >= len(self._tab_plot_names):
            return
        name = self._tab_plot_names[idx]
        if name in self._dirty:
            self._dirty.discard(name)
            getattr(self, f"_plot_{name}")()

    # ── Public update API ─────────────────────────────────────────

    def update_detections(self, feats, frame_bgr, params: dict):
        """Call after Preview Worker finishes — populates per-frame diagnostic tabs."""
        self._feats     = feats
        self._frame_bgr = frame_bgr
        self._params    = params
        self._redraw_or_mark("mass")
        self._redraw_or_mark("ecc")
        self._redraw_or_mark("spx")

    def update_frame_count(self, fr: int, count: int):
        """Call per-frame during tracking to build the sparkline.
        Throttled to every 10th point, mirroring DefectPlot.add_point —
        redrawing the full history on every single frame is wasted work."""
        self._frame_counts.append((fr, count))
        if len(self._frame_counts) % 10 == 0:
            self._redraw_or_mark("cnt")

    def clear_frame_counts(self):
        self._frame_counts = []
        self._dirty.discard("cnt")
        self._ax_cnt.clear(); self._cv_cnt.draw_idle()

    def update_tracks(self, tracks, px_um: float, params: dict, feats=None):
        """Call when analysis completes — populates track-based tabs."""
        self._tracks  = tracks
        self._px_um   = px_um
        self._params  = params
        if feats is not None:
            self._feats = feats
        self._redraw_or_mark("tlen")
        self._redraw_or_mark("mass")   # refresh with full-video feats
        self._redraw_or_mark("ecc")
        self._redraw_or_mark("spx")

    def update_pair_correlations(self, r_centers, g_r, g6_r, g6_over_g_r):
        """Call after a structure-function computation (g(r)/g6(r)) completes
        — populates all 3 pair-correlation tabs at once."""
        self._pair_corr = (r_centers, g_r, g6_r, g6_over_g_r)
        self._redraw_or_mark("gr")
        self._redraw_or_mark("g6")
        self._redraw_or_mark("g6g")

    # ── Plotting ──────────────────────────────────────────────────

    def _plot_mass(self):
        ax = self._ax_mass; ax.clear(); self._sa(ax)
        if self._feats is None or self._feats.empty:
            ax.text(0.5, 0.5, "Run Preview frame to populate",
                    ha="center", va="center", transform=ax.transAxes, color="#555")
        else:
            masses = self._feats["mass"].values
            ax.hist(masses, bins=50, color="#4488cc", edgecolor="none", alpha=0.85)
            if self._params:
                mm = float(self._params.get("minmass", 0))
                ax.axvline(mm, color="#ff6600", lw=1.5, label=f"Min mass={mm:.0f}")
            auto = _auto_min_mass(masses)
            if auto is not None:
                ax.axvline(auto, color="#44ff88", lw=1.2, ls="--",
                           label=f"Auto={auto}")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=6, labelcolor=self._FG,
                          facecolor="#1a1a2e", edgecolor="none")
            ax.set_xlabel("Mass"); ax.set_ylabel("Count")
            ax.set_title(f"{len(masses)} particles", color=self._FG, fontsize=8)
        self._cv_mass.draw_idle()

    def _plot_ecc(self):
        ax = self._ax_ecc; ax.clear(); self._sa(ax)
        if self._feats is None or self._feats.empty or "ecc" not in self._feats.columns:
            ax.text(0.5, 0.5, "Run Preview frame to populate",
                    ha="center", va="center", transform=ax.transAxes, color="#555")
        else:
            ecc = self._feats["ecc"].values
            ax.hist(ecc, bins=30, color="#44cc88", edgecolor="none", alpha=0.85)
            ax.axvline(0.1, color="#ffaa00", lw=1.2, ls="--", label="0.1 (round)")
            ax.axvline(0.3, color="#ff4444", lw=1.2, ls="--", label="0.3 (limit)")
            ax.legend(fontsize=6, labelcolor=self._FG,
                      facecolor="#1a1a2e", edgecolor="none")
            ax.set_xlabel("Eccentricity (0=circle, 1=rod)")
            ax.set_ylabel("Count")
            ax.set_title(f"Mean ecc = {ecc.mean():.3f}", color=self._FG, fontsize=8)
        self._cv_ecc.draw_idle()

    def _plot_spx(self):
        for ax in (self._ax_spx_x, self._ax_spx_y):
            ax.clear(); self._sa(ax)
        locked, ratio = False, 1.0
        if self._feats is not None and not self._feats.empty:
            x_col = "x_px" if "x_px" in self._feats.columns else "x"
            y_col = "y_px" if "y_px" in self._feats.columns else "y"
            if x_col in self._feats.columns:
                fx = self._feats[x_col].values % 1.0
                fy = self._feats[y_col].values % 1.0
                self._ax_spx_x.hist(fx, bins=20, color="#44aacc", edgecolor="none", alpha=0.85)
                self._ax_spx_y.hist(fy, bins=20, color="#cc8844", edgecolor="none", alpha=0.85)
                locked, ratio = _check_pixel_locking(self._feats)
        else:
            for ax in (self._ax_spx_x, self._ax_spx_y):
                ax.text(0.5, 0.5, "Run Preview", ha="center", va="center",
                        transform=ax.transAxes, color="#555")
        self._ax_spx_x.set_title("frac(x)", color=self._FG, fontsize=8)
        self._ax_spx_y.set_title("frac(y)", color=self._FG, fontsize=8)
        self._cv_spx.draw_idle()
        if locked:
            self._lbl_spx.setText(
                f"⚠ Pixel locking detected (ratio={ratio:.2f} < 0.5). "
                "Increase Diameter by 2 px or improve focus.")
            self._lbl_spx.setStyleSheet("color:#ff8844;font-size:10px;padding:2px;")
        else:
            self._lbl_spx.setText(f"Sub-pixel OK (uniformity ratio={ratio:.2f}).")
            self._lbl_spx.setStyleSheet("color:#44cc88;font-size:10px;padding:2px;")

    def _plot_tlen(self):
        ax = self._ax_tlen; ax.clear(); self._sa(ax)
        if self._tracks is None or self._tracks.empty:
            ax.text(0.5, 0.5, "Run tracking to populate",
                    ha="center", va="center", transform=ax.transAxes, color="#555")
        else:
            lengths = self._tracks.groupby("particle").size()
            ax.hist(lengths, bins=30, color="#8844cc", edgecolor="none", alpha=0.85)
            ml = float(self._params.get("min_len", 15))
            ax.axvline(ml, color="#ff6600", lw=1.5, label=f"Min len={ml:.0f}")
            auto = _auto_min_length(self._tracks)
            if auto:
                ax.axvline(auto, color="#44ff88", lw=1.2, ls="--", label=f"Auto={auto}")
            ax.legend(fontsize=6, labelcolor=self._FG,
                      facecolor="#1a1a2e", edgecolor="none")
            ax.set_xlabel("Track length (frames)"); ax.set_ylabel("Count")
            ax.set_title(f"{len(lengths)} tracks", color=self._FG, fontsize=8)
        self._cv_tlen.draw_idle()

    def _plot_cnt(self):
        ax = self._ax_cnt; ax.clear(); self._sa(ax)
        if not self._frame_counts:
            ax.text(0.5, 0.5, "Run tracking to populate",
                    ha="center", va="center", transform=ax.transAxes, color="#555")
        else:
            frs, cnts = zip(*self._frame_counts)
            med = float(np.median(cnts))
            ax.plot(frs, cnts, color="#44aacc", lw=0.7)
            ax.axhline(med, color="#666", lw=0.8, ls="--")
            thresh = med * 0.5
            dip_x = [f for f, c in self._frame_counts if c < thresh]
            dip_y = [c for f, c in self._frame_counts if c < thresh]
            if dip_x:
                ax.scatter(dip_x, dip_y, color="#ff4444", s=5, zorder=5)
            ax.set_xlabel("Frame"); ax.set_ylabel("Particles")
            ax.set_title(f"median={med:.0f}  dips={len(dip_x)}",
                         color=self._FG, fontsize=8)
        self._cv_cnt.draw_idle()

    def _plot_gr(self):
        ax = self._ax_gr; ax.clear(); self._sa(ax)
        if self._pair_corr is None:
            ax.text(0.5, 0.5, "Click 'Compute structure functions' to populate",
                    ha="center", va="center", transform=ax.transAxes, color="#555")
        else:
            r, g_r, _, _ = self._pair_corr
            ax.plot(r, g_r, color="#44cc88", lw=1.2)
            ax.axhline(1.0, color="#888", lw=0.8, ls="--", label="ideal gas (g=1)")
            ax.legend(fontsize=6, labelcolor=self._FG, facecolor="#1a1a2e", edgecolor="none")
            ax.set_xlabel("r [µm]"); ax.set_ylabel("g(r)")
            ax.set_title("Radial pair-distribution function", color=self._FG, fontsize=8)
        self._cv_gr.draw_idle()

    def _plot_g6(self):
        ax = self._ax_g6; ax.clear(); self._sa(ax)
        if self._pair_corr is None:
            ax.text(0.5, 0.5, "Click 'Compute structure functions' to populate",
                    ha="center", va="center", transform=ax.transAxes, color="#555")
        else:
            r, _, g6_r, _ = self._pair_corr
            ax.plot(r, g6_r, color="#cc8844", lw=1.2)
            ax.set_yscale("log")
            ax.set_xlabel("r [µm]"); ax.set_ylabel("g6(r)  (log)")
            ax.set_title("Hexatic bond-orientational correlation", color=self._FG, fontsize=8)
        self._cv_g6.draw_idle()

    def _plot_g6g(self):
        ax = self._ax_g6g; ax.clear(); self._sa(ax)
        if self._pair_corr is None:
            ax.text(0.5, 0.5, "Click 'Compute structure functions' to populate",
                    ha="center", va="center", transform=ax.transAxes, color="#555")
        else:
            r, _, _, g6g = self._pair_corr
            ax.plot(r, g6g, color="#cc44aa", lw=1.2)
            ax.set_yscale("log")
            ax.set_xlabel("r [µm]"); ax.set_ylabel("g6(r)/g(r)  (log)")
            ax.set_title("Density-normalised hexatic correlation", color=self._FG, fontsize=8)
        self._cv_g6g.draw_idle()

    def _plot_sweep(self, param_name: str, results: list):
        ax = self._ax_sw; ax.clear(); self._sa(ax)
        if not results:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="#555")
        else:
            vals, cnts = zip(*results)
            ax.plot(vals, cnts, color="#44aacc", marker="o", ms=3, lw=1.2)
            if self._params:
                cur_key = "minmass" if param_name == "minmass" else "percentile"
                cur = float(self._params.get(cur_key, 0))
                ax.axvline(cur, color="#ff6600", lw=1.5, label=f"current={cur:.0f}")
                ax.legend(fontsize=6, labelcolor=self._FG,
                          facecolor="#1a1a2e", edgecolor="none")
            ax.set_xlabel(param_name); ax.set_ylabel("Count")
            ax.set_title("Parameter sweep", color=self._FG, fontsize=8)
        self._btn_sw.setEnabled(True); self._btn_sw.setText("Run sweep")
        self._cv_sw.draw_idle()

    # ── Auto-tune buttons ─────────────────────────────────────────

    def _on_auto_min_mass(self):
        if self._feats is None or self._feats.empty: return
        auto = _auto_min_mass(self._feats["mass"].values)
        if auto is not None:
            self.apply_params.emit({"minmass": auto})

    def _on_auto_min_length(self):
        if self._tracks is None or self._tracks.empty: return
        auto = _auto_min_length(self._tracks)
        if auto is not None:
            self.apply_params.emit({"min_len": auto})

    def _on_auto_max_step(self):
        if self._tracks is None or self._tracks.empty: return
        auto = _auto_max_step(self._tracks)
        if auto is not None:
            self.apply_params.emit({"max_step_um": auto})

    def _on_run_sweep(self):
        if self._frame_bgr is None or not self._params: return
        if self._sweep_worker and self._sweep_worker.isRunning(): return
        param = "minmass" if self._combo_sw.currentIndex() == 0 else "percentile"
        self._btn_sw.setEnabled(False); self._btn_sw.setText("Sweeping…")
        self._sweep_worker = SweepWorker(self._frame_bgr, self._params,
                                         param, parent=self)
        self._sweep_worker.result.connect(self._plot_sweep)
        self._sweep_worker.error.connect(
            lambda _: (self._btn_sw.setEnabled(True),
                       self._btn_sw.setText("Run sweep")))
        self._sweep_worker.start()


# ════════════════════════════════════════════════════════════════════
# Video pane  (updated: preview + params passed externally)
# ════════════════════════════════════════════════════════════════════

class VideoPane(QWidget):
    frame_changed        = pyqtSignal(int)
    grain_size_suggested = pyqtSignal(int)
    preview_feats_ready  = pyqtSignal(object, object)   # (feats_df, frame_bgr)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap=None; self._data=None; self._layer=LayerConfig()
        self._fr=0; self._total=0; self._playing=False; self._fps_d=10.
        self._params=dict(_DEFAULT_PARAMS); self._prev_w=None
        self._vpath: str | None = None
        self._gs_worker: GrainSizeWorker | None = None
        self._consistency_worker: GrainConsistencyWorker | None = None
        self._detected_diam: int | None = None
        self._ecc_overlay   = False   # colour detections by eccentricity
        self._link_overlay  = False   # show inter-frame displacement arrows
        self._grain_overlay = False   # colour particles by crystallographic grain (ψ₆)
        self._hide_raw      = False   # suppress raw-video panel during analysis
        self._ann_cache: dict = {}    # bounded cache: key→bgr annotated frame
        self._preview_feats = None   # feats from most-recent PreviewWorker run
        self._preview_raw   = None   # raw frame backing _preview_feats (avoids a 2nd re-read)
        self._cap_pos      = -1      # track cap read position to skip redundant seeks
        self._timer=QTimer(self); self._timer.timeout.connect(self._tick)
        self._build()

    def _build(self):
        lay=QVBoxLayout(self); lay.setContentsMargins(2,2,2,2); lay.setSpacing(0)
        self._warn_bar = GrainWarningBar()
        lay.addWidget(self._warn_bar)
        imgs=QHBoxLayout()
        # Use GPU-accelerated FrameGLWidget for fast playback — eliminates the
        # QPixmap copy chain (~5 CPU copies/frame) that made VideoPane sluggish.
        self.lbl_raw = FrameGLWidget("Raw")
        self.lbl_ann = FrameGLWidget("Annotated")
        imgs.addWidget(self.lbl_raw); imgs.addWidget(self.lbl_ann)
        lay.addLayout(imgs,stretch=1)
        self.slider=NoScrollSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0,0); self.slider.valueChanged.connect(self._on_sl)
        lay.addWidget(self.slider)
        ctrl=QHBoxLayout()
        self.btn_play=QPushButton("▶"); self.btn_play.setCheckable(True); self.btn_play.setFixedWidth(36)
        self.btn_play.toggled.connect(self._on_play)
        self.btn_p=QPushButton("◀"); self.btn_p.setFixedWidth(28); self.btn_p.clicked.connect(lambda:self.goto(self._fr-1))
        self.btn_n=QPushButton("▶"); self.btn_n.setFixedWidth(28); self.btn_n.clicked.connect(lambda:self.goto(self._fr+1))
        self.btn_prev=QPushButton("Preview frame")
        self.btn_prev.setToolTip("Run detection on this frame only — fast parameter check")
        self.btn_prev.clicked.connect(self._run_preview)
        self.btn_ecc=QPushButton("Eccentricity")
        self.btn_ecc.setCheckable(True); self.btn_ecc.setFixedWidth(78)
        self.btn_ecc.setToolTip("Colour-code detected particles by eccentricity\n(green = round, red = elongated)")
        self.btn_ecc.toggled.connect(self._on_ecc_toggle)
        self.btn_link=QPushButton("Motion")
        self.btn_link.setCheckable(True); self.btn_link.setFixedWidth(50)
        self.btn_link.setToolTip("Show displacement arrows from each particle to its position\nin the next frame (requires tracking data)")
        self.btn_link.toggled.connect(self._on_link_toggle)
        self.btn_grain=QPushButton("Grains")
        self.btn_grain.setCheckable(True)
        self.btn_grain.setToolTip("Colour particles by crystallographic grain (ψ₆ orientation clustering)")
        self.btn_grain.toggled.connect(self._on_grain_toggle)
        self.btn_hide_raw=QPushButton("Hide raw")
        self.btn_hide_raw.setCheckable(True)
        self.btn_hide_raw.setToolTip("Hide the unannotated raw video panel")
        self.btn_hide_raw.toggled.connect(self._on_hide_raw_toggle)
        self.lbl_info=QLabel("—")
        self.sp_fps=NoScrollSpinBox(); self.sp_fps.setRange(1,120); self.sp_fps.setValue(10)
        self.sp_fps.setPrefix("fps: "); self.sp_fps.setMinimumWidth(80)
        self.sp_fps.valueChanged.connect(lambda v:setattr(self,"_fps_d",float(v)))
        for w in(self.btn_p,self.btn_play,self.btn_n,self.btn_prev,
                 self.btn_ecc,self.btn_link,self.btn_grain,self.btn_hide_raw,self.lbl_info,self.sp_fps): ctrl.addWidget(w)
        ctrl.addStretch(); lay.addLayout(ctrl)

    def load_video(self, path):
        if self._cap: self._cap.release()
        self._cap   = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._vpath = str(path)
        self._cap_pos = 0
        self._detected_diam = None
        self._warn_bar.hide()
        self.slider.setRange(0, max(0, self._total - 1))
        self.goto(0)
        # Auto-detect grain size on first frame after a brief delay so the
        # frame is rendered and the UI is responsive first.
        QTimer.singleShot(400, self._auto_detect_grain_size)

    def _auto_detect_grain_size(self):
        if not self._cap:
            return
        bgr = self._read_bgr(0)
        if bgr is None:
            return
        if self._gs_worker and self._gs_worker.isRunning():
            self._gs_worker.abort()
        self._gs_worker = GrainSizeWorker(bgr, self._params, self)
        self._gs_worker.detected.connect(self._on_grain_video)
        self._gs_worker.failed.connect(lambda _: None)   # silent fail
        self._gs_worker.start()

    def _on_grain_video(self, diam: int, count: int):
        self._detected_diam = diam
        self.grain_size_suggested.emit(diam)

    def check_grain_consistency(self, video_path: str, used_diam: int,
                                n_samples: int = 5):
        """Sample evenly-spaced frames and check that auto-detected diameter
        is consistent with *used_diam* (the diameter that was used in tracking).
        Shows GrainWarningBar if a significant discrepancy is found."""
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else 0
        if total < 2:
            return
        start = self.slider.minimum()
        end   = self.slider.maximum()
        if end <= start:
            return
        step   = max(1, (end - start) // max(1, n_samples - 1))
        frames = list(range(start, end + 1, step))[:n_samples]
        if self._consistency_worker and self._consistency_worker.isRunning():
            self._consistency_worker.abort()
        self._consistency_worker = GrainConsistencyWorker(
            video_path, frames, used_diam, self._params, self
        )
        self._consistency_worker.inconsistent.connect(self._on_inconsistency)
        self._consistency_worker.start()

    def _on_inconsistency(self, fr: int, detected: int, expected: int):
        self._warn_bar.warn(
            f"Grain size inconsistency detected at frame {fr}: "
            f"auto-detection found {detected} px but tracking used {expected} px.  "
            f"Re-run tracking with the corrected diameter for accurate results."
        )

    def set_analysis(self,data):
        self._data=data; self._ann_cache.clear()
        if data: self.slider.setRange(data.start_fr,data.end_fr); self.goto(data.start_fr)

    def set_layer(self,l): self._layer=l; self._ann_cache.clear(); self._render(self._fr)
    def set_params(self,p): self._params=p; self._ann_cache.clear()

    def goto(self,fr):
        if self._cap is None: return
        fr=max(self.slider.minimum(),min(self.slider.maximum(),fr))
        self._fr=fr; self.slider.blockSignals(True); self.slider.setValue(fr); self.slider.blockSignals(False)
        self._render(fr); self.frame_changed.emit(fr)

    def _tick(self):
        self.goto(self._fr+1)
        if self._fr>=self.slider.maximum(): self._playing=False; self.btn_play.setChecked(False)

    def _on_play(self,on):
        self._playing=on
        if on: self._timer.start(max(1,int(1000/self._fps_d)))
        else:  self._timer.stop()

    def _on_sl(self,v): self._fr=v; self._render(v); self.frame_changed.emit(v)

    def _read_bgr(self,fr):
        if not self._cap: return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES,fr); ok,f=self._cap.read()
        if not ok: return None
        gray = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) if f.ndim==3 else f
        u8   = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(u8,cv2.COLOR_GRAY2BGR)

    _ANN_CACHE_MAX = 12   # max annotated BGR frames to hold in memory (~180 MB at 4k)

    def _render(self,fr,override=None):
        raw=self._read_display_bgr(fr)
        if raw is None: return
        raw=self._apply_enhancement_for_display(raw)
        if not self._hide_raw:
            self.lbl_raw.set_frame(raw)
        res=override
        if res is None and self._data and fr in self._data.frame_results:
            res=self._data.frame_results[fr]
        fevs=self._data.flag_events if self._data else []

        # Cache key: frame + overlay flags + layer digest + particle radius
        r = max(4, int(self._params.get("diameter", 19)) // 2 + 3)
        lyr = self._layer
        _lkey = (lyr.show_6fold, lyr.show_5fold, lyr.show_7fold, lyr.show_bonds,
                 lyr.show_pairs, lyr.show_lagb, lyr.show_hagb, lyr.show_flags,
                 lyr.show_hud, lyr.radius, lyr.bond_thick, lyr.pair_thick,
                 lyr.gb_thick, lyr.flag_thick, lyr.flag_frames,
                 lyr.col_6fold, lyr.col_5fold, lyr.col_7fold)
        cache_key = (fr, id(self._data), _lkey,
                     self._ecc_overlay, self._link_overlay, self._grain_overlay, r)

        if cache_key in self._ann_cache and override is None:
            ann = self._ann_cache[cache_key]
        else:
            if res is not None:
                ann=annotate(raw,res,lyr,fevs,fr); _legend(ann,lyr)
            else:
                ann=raw.copy()
                cv2.putText(ann,"Run tracking or press 'Preview frame'",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(60,160,60),1)
            # ── eccentricity heat overlay ──────────────────────────
            if self._ecc_overlay:
                feats_fr = None
                if self._data is not None and self._data.feats is not None:
                    feats_fr = self._data.feats[self._data.feats.frame == fr]
                if (feats_fr is None or feats_fr.empty) and self._preview_feats is not None:
                    feats_fr = self._preview_feats
                if feats_fr is not None and not feats_fr.empty:
                    ann = _draw_ecc_overlay(ann, feats_fr, radius=r)
            # ── linking displacement overlay ───────────────────────
            if self._link_overlay and self._data is not None and self._data.tracks is not None:
                px_um      = float(self._params.get("px_um", 0.11))
                max_step_u = float(self._params.get("max_step_um", 3.0))
                ann = _draw_linking_overlay(ann, self._data.tracks, fr,
                                            max_step_u / max(px_um, 1e-9))
            # ── ψ₆ grain overlay ──────────────────────────────────
            if self._grain_overlay and self._data and fr in self._data.frame_results:
                ann = _draw_grain_overlay(ann, self._data.frame_results[fr], radius=r)
            # Store in bounded cache (evict oldest entry when full)
            if override is None:
                if len(self._ann_cache) >= self._ANN_CACHE_MAX:
                    self._ann_cache.pop(next(iter(self._ann_cache)))
                self._ann_cache[cache_key] = ann

        self.lbl_ann.set_frame(ann)
        self.lbl_info.setText(f"fr {fr}/{self._total-1}")

    def _on_ecc_toggle(self, checked: bool):
        self._ecc_overlay = checked; self._render(self._fr)

    def _on_link_toggle(self, checked: bool):
        self._link_overlay = checked; self._render(self._fr)

    def _on_grain_toggle(self, checked: bool):
        self._grain_overlay = checked; self._render(self._fr)

    def _on_hide_raw_toggle(self, checked: bool):
        self._hide_raw = checked
        self.lbl_raw.setVisible(not checked)
        self._render(self._fr)

    def _run_preview(self):
        if not self._cap: return
        raw=self._read_bgr(self._fr)
        if raw is None: return
        # Stash the frame we just read so _on_preview_feats can forward it
        # without a second synchronous seek+decode of the same frame index.
        self._preview_raw = raw
        self.btn_prev.setEnabled(False); self.btn_prev.setText("…")
        self._prev_w=PreviewWorker(raw,self._params,self)
        self._prev_w.done.connect(self._on_preview)
        self._prev_w.feats_ready.connect(self._on_preview_feats)
        self._prev_w.error.connect(lambda _:(self.btn_prev.setEnabled(True),self.btn_prev.setText("Preview frame")))
        self._prev_w.start()

    def _on_preview(self,res):
        self.btn_prev.setEnabled(True); self.btn_prev.setText("Preview frame")
        if res: self._render(self._fr,override=res)

    def _on_preview_feats(self, feats):
        """Store feats and forward to MainWindow → DiagnosticsPanel."""
        self._preview_feats = feats
        raw = self._preview_raw
        self.preview_feats_ready.emit(feats, raw)

    @staticmethod
    def _show(bgr, widget):
        widget.set_frame(bgr)

    def _read_display_bgr(self, fr):
        """Fast frame read for display only — skips seek when playing sequentially."""
        if not self._cap: return None
        if self._cap_pos != fr:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, f = self._cap.read()
        self._cap_pos = fr + 1 if ok else -1
        if not ok: return None
        if f.ndim == 2:
            return cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        return f

    def _apply_enhancement_for_display(self, bgr):
        """Apply the same image-enhancement params (gamma/sharpen/denoise) used
        for detection so the preview actually reflects what tracking sees —
        mirrors CameraPane._on_frame's display-side enhancement. Detection
        itself already applies these via _preprocess inside TrackingWorker;
        this only affects what's rendered on screen."""
        p = self._params
        gamma   = p.get("gamma", 1.0)
        sharpen = p.get("sharpen_amount", 0.0)
        denoise = p.get("denoise_method", "off")
        dns     = p.get("denoise_strength", 10.0)
        flatten = bool(p.get("flatten_illum", False))
        if gamma == 1.0 and sharpen <= 0 and denoise == "off" and not flatten:
            return bgr
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        enh  = _preprocess(gray, False, 1, 53, False, 2.0,
                           denoise=denoise, denoise_strength=dns,
                           gamma=gamma, sharpen=sharpen,
                           flatten_illum=flatten,
                           flatten_sigma_frac=p.get("flatten_sigma_frac", 0.15))
        u8   = cv2.normalize(enh, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


# ════════════════════════════════════════════════════════════════════
# Camera pane  (updated with camera selector)
# ════════════════════════════════════════════════════════════════════

class CameraPane(QWidget):
    grain_size_detected = pyqtSignal(int)   # auto-detected diameter in original-camera px

    def __init__(self, layer, params, parent=None):
        super().__init__(parent)
        self._layer       = layer
        self._params      = params
        self._worker      = None
        self._cameras     = []
        # recording state
        self._recording   = False
        self._rec_buf     = None
        self._recorder    = None
        self._tiff_recorder: "TiffSequenceRecorder | None" = None   # lossless Mono12 path
        self._rec_start   = None
        self._last_bgr    = None
        self._downscaling   = False
        self._rec_out_dir   = Path(_SETTINGS_DIR)
        self._last_ann      = None   # most-recently annotated frame (BGR)
        self._ana_worker    = None   # AnalysisWorker background thread
        self._compare_mode  = False
        self._gs_worker     : GrainSizeWorker | None = None
        self._last_cam_scale: float = 1.0   # scale of most-recent frame (live_px/orig_px)
        # shutter / timelapse state
        self._shutter       = LambdaSCController()
        self._tl_worker      : "TimelapseWorker | None" = None
        self._tl_out_dir     = Path(_SETTINGS_DIR) / "timelapse"
        self._build()

    def _build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(2,2,2,2)

        # ── caption row (compare mode only) ───────────────────────
        self._cap_row = QWidget()
        cap_h = QHBoxLayout(self._cap_row); cap_h.setContentsMargins(0, 0, 0, 0)
        for txt in ("Raw feed", "With tracking"):
            cl = QLabel(txt)
            cl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cl.setStyleSheet("color:#666;font-size:10px;")
            cap_h.addWidget(cl, stretch=1)
        self._cap_row.setVisible(False)
        lay.addWidget(self._cap_row)

        # ── image area (GPU-accelerated via FrameGLWidget) ────────
        img_h = QHBoxLayout()
        self.lbl     = FrameGLWidget("Camera feed",     self)
        self.lbl_ann = FrameGLWidget("Tracking overlay", self)
        self.lbl_ann.setVisible(False)
        img_h.addWidget(self.lbl,     stretch=1)
        img_h.addWidget(self.lbl_ann, stretch=1)
        lay.addLayout(img_h, stretch=1)

        # ── camera row ────────────────────────────────────────────
        row = QHBoxLayout()
        self.combo = NoScrollComboBox(); self.combo.setMinimumWidth(200)
        self.combo.addItem("(no cameras detected)")
        self.btn_scan = QPushButton("Scan"); self.btn_scan.setFixedWidth(50)
        self.btn_scan.clicked.connect(self._scan)
        self.btn_conn = QPushButton("Connect")
        self.btn_conn.clicked.connect(self._toggle)
        self.lbl_st = QLabel("Disconnected"); self.lbl_st.setStyleSheet("color:#888;")

        # Live resolution selector
        self._live_max_px = _LIVE_RES_DEFAULT
        res_lbl = QLabel("Live res:")
        self.combo_res = NoScrollComboBox()
        self.combo_res.setFixedWidth(110)
        _default_idx = 0
        for i, (lbl, px) in enumerate(_LIVE_RES_OPTIONS):
            self.combo_res.addItem(lbl, px)
            if px == _LIVE_RES_DEFAULT:
                _default_idx = i
        self.combo_res.setCurrentIndex(_default_idx)
        self.combo_res.setToolTip(
            "Maximum pixel size (longer dimension) of frames sent to display and analysis.\n"
            "Lower = faster display and faster particle detection.\n"
            "Higher = more detail but slower. Full res = original camera resolution."
        )
        self.combo_res.currentIndexChanged.connect(self._on_res_changed)

        # Compare toggle — shows a live overlay of real-time particle detection
        # (annotated markers/rings) next to the raw feed, so parameters can be
        # tuned while watching detection respond live. Renamed from "Compare"
        # (Task 4) to describe what turning it ON actually shows.
        self.btn_compare = QPushButton("Live detection overlay")
        self.btn_compare.setCheckable(True)
        self.btn_compare.setFixedWidth(150)
        self.btn_compare.setToolTip(
            "Show a live side-by-side view: raw camera feed on the left,\n"
            "annotated particle-detection overlay on the right — lets you tune\n"
            "detection parameters (diameter, min mass, etc.) while watching the\n"
            "camera live. Off by default to avoid the extra analysis overhead."
        )
        self.btn_compare.clicked.connect(self._toggle_compare)

        # Auto grain-size button — only meaningful while the live detection
        # overlay is active (Task 3); hidden otherwise.
        self.btn_auto_size = QPushButton("Auto diameter")
        self.btn_auto_size.setFixedWidth(95)
        self.btn_auto_size.setToolTip(
            "Scan the current frame for the particle Diameter (Detection panel)\n"
            "that detects the most round (near-circular) particles, and apply it.\n"
            "Takes a few seconds — progress shown in the status label."
        )
        self.btn_auto_size.clicked.connect(self._on_auto_size)
        self.btn_auto_size.setVisible(False)

        # Analysis fps limiter — only relevant while the live detection
        # overlay is active (it throttles AnalysisWorker); hidden otherwise.
        self._ana_fps_lbl = QLabel("Analysis:")
        self.sp_ana_fps = NoScrollDoubleSpinBox()
        self.sp_ana_fps.setRange(0.5, 240.0); self.sp_ana_fps.setDecimals(1)
        self.sp_ana_fps.setValue(5.0); self.sp_ana_fps.setSuffix(" fps")
        self.sp_ana_fps.setFixedWidth(80)
        self.sp_ana_fps.setToolTip(
            "Maximum frame rate for live analysis while the detection overlay is\n"
            "on. Lower values keep the UI responsive; raise if your CPU/GPU is fast\n"
            "enough and you need tighter tracking feedback. AnalysisWorker always\n"
            "keeps only the latest pending frame (older ones are dropped, never\n"
            "queued), so raising this has no risk of unbounded memory growth —\n"
            "it only affects how often the overlay updates.")
        self.sp_ana_fps.valueChanged.connect(self._on_ana_fps_changed)
        self._ana_fps_lbl.setVisible(False)
        self.sp_ana_fps.setVisible(False)

        # Sub-controls that only make sense while the live detection overlay
        # is active — hidden/shown together in _toggle_compare (Task 3/4).
        self._compare_sub_widgets = [self.btn_auto_size, self._ana_fps_lbl, self.sp_ana_fps]

        for w in (self.btn_scan, self.combo, self.btn_conn, self.lbl_st,
                  res_lbl, self.combo_res, self.btn_compare, self.btn_auto_size,
                  self._ana_fps_lbl, self.sp_ana_fps):
            row.addWidget(w)
        row.addStretch(); lay.addLayout(row)

        # ── camera acquisition controls (exposure / gain / pixel format) ──
        # Ranges are placeholders until ranges_ready reports the real GenICam
        # min/max for whatever camera is connected; widgets stay disabled
        # until then since they have no effect without a running camera.
        # Grouped under one "Image quality" box (with Exposure/Gain/Black level
        # together) so it visually matches the "Image enhancement" group in the
        # Parameters dock (gamma/sharpen/denoise) — different panels (this one
        # drives hardware sensor settings, that one drives software post-
        # processing), but both are "image appearance" controls and the user
        # wants them presented as one conceptual category.
        iq_group = CollapsibleGroupBox("Image quality (sensor)")
        iq_lay = QVBoxLayout()
        iq_group.setLayout(iq_lay)
        cc_row = QHBoxLayout()

        self.cb_exp_auto = NoScrollComboBox(); self.cb_exp_auto.addItems(["Off","Once","Continuous"])
        self.cb_exp_auto.setToolTip(
            "Auto-exposure mode. 'Off' (recommended) keeps exposure fixed so "
            "particle mass/brightness thresholds stay consistent frame-to-frame — "
            "continuous auto-exposure will fight the detection thresholds.")
        self.sp_exposure = NoScrollDoubleSpinBox()
        self.sp_exposure.setRange(20, 1_000_000); self.sp_exposure.setDecimals(0)
        self.sp_exposure.setSuffix(" µs"); self.sp_exposure.setSingleStep(100)
        self.sp_exposure.setToolTip("Sensor exposure time. Longer = brighter but more "
                                     "motion blur and a lower achievable frame rate.")

        self.cb_gain_auto = NoScrollComboBox(); self.cb_gain_auto.addItems(["Off","Once","Continuous"])
        self.cb_gain_auto.setToolTip("Auto-gain mode. 'Off' (recommended) — see the Exposure auto-mode note above.")
        self.sp_gain = NoScrollDoubleSpinBox()
        self.sp_gain.setRange(0, 48); self.sp_gain.setDecimals(1)
        self.sp_gain.setSuffix(" dB"); self.sp_gain.setSingleStep(0.5)
        self.sp_gain.setToolTip("Sensor gain. Raises brightness but amplifies noise — "
                                 "prefer raising exposure time first.")

        self.sp_blacklevel = NoScrollDoubleSpinBox()
        self.sp_blacklevel.setRange(0, 64); self.sp_blacklevel.setDecimals(1)
        self.sp_blacklevel.setToolTip("Sensor black-level offset (zero-photon floor). "
                                       "Affects the noise floor the auto-threshold "
                                       "(min-mass Otsu) logic assumes is clean.")

        self.combo_pixfmt = NoScrollComboBox()
        self.combo_pixfmt.setToolTip(
            "Sensor pixel format. Mono8 is the simplest/fastest. Mono12 captures more\n"
            "dynamic range at the sensor — tone-mapped to 8-bit for live display/detection,\n"
            "but can be recorded losslessly at full 12-bit via 'Lossless (Mono12)' below.")

        self.cb_lossless = QCheckBox("Lossless (Mono12)")
        self.cb_lossless.setEnabled(False)
        self.cb_lossless.setToolTip(
            "Record the full 12-bit sensor data as a lossless 16-bit TIFF sequence\n"
            "instead of an 8-bit MP4. Standard MP4 (even FFV1) cannot hold more than\n"
            "8 bits/channel through OpenCV — verified empirically — so this writes one\n"
            ".tiff file per frame to a folder instead. Much larger on disk; only enabled\n"
            "when PixelFormat = Mono12.")
        self.combo_pixfmt.currentTextChanged.connect(self._on_pixfmt_for_lossless)

        self.cb_fps_cap = QCheckBox("Cap fps")
        self.sp_fps_cap = NoScrollDoubleSpinBox()
        self.sp_fps_cap.setRange(1, 200); self.sp_fps_cap.setDecimals(1)
        self.sp_fps_cap.setSuffix(" fps"); self.sp_fps_cap.setEnabled(False)
        self.cb_fps_cap.toggled.connect(self.sp_fps_cap.setEnabled)
        self.cb_fps_cap.setToolTip("Cap the acquisition frame rate below the sensor's max "
                                    "(e.g. to pair with a longer exposure time).")

        self.lbl_temp = QLabel("Temp: —"); self.lbl_temp.setStyleSheet("color:#888;")

        # Split across two rows with addSpacing between groups so each label
        # stays visually attached to its own field instead of all labels
        # reading as a run-on list ahead of all the value widgets.
        cc_row.addWidget(QLabel("Exposure:")); cc_row.addWidget(self.cb_exp_auto)
        cc_row.addWidget(self.sp_exposure)
        cc_row.addSpacing(14)
        cc_row.addWidget(QLabel("Gain:")); cc_row.addWidget(self.cb_gain_auto)
        cc_row.addWidget(self.sp_gain)
        cc_row.addSpacing(14)
        cc_row.addWidget(QLabel("Black level:")); cc_row.addWidget(self.sp_blacklevel)
        cc_row.addStretch(); iq_lay.addLayout(cc_row)

        cc_row2 = QHBoxLayout()
        cc_row2.addWidget(QLabel("Format:")); cc_row2.addWidget(self.combo_pixfmt)
        cc_row2.addWidget(self.cb_lossless)
        cc_row2.addSpacing(14)
        cc_row2.addWidget(self.cb_fps_cap); cc_row2.addWidget(self.sp_fps_cap)
        cc_row2.addSpacing(14)
        cc_row2.addWidget(self.lbl_temp)
        cc_row2.addStretch(); iq_lay.addLayout(cc_row2)
        lay.addWidget(iq_group)

        self._cam_ctrl_widgets = [self.cb_exp_auto, self.sp_exposure, self.cb_gain_auto,
                                   self.sp_gain, self.sp_blacklevel, self.combo_pixfmt,
                                   self.cb_fps_cap, self.sp_fps_cap]
        for w in self._cam_ctrl_widgets:
            w.setEnabled(False)

        self.sp_exposure.valueChanged.connect(lambda v: self._push_cam_param("ExposureTime", v))
        self.cb_exp_auto.currentTextChanged.connect(lambda v: self._push_cam_param("ExposureAuto", v))
        self.sp_gain.valueChanged.connect(lambda v: self._push_cam_param("Gain", v))
        self.cb_gain_auto.currentTextChanged.connect(lambda v: self._push_cam_param("GainAuto", v))
        self.sp_blacklevel.valueChanged.connect(lambda v: self._push_cam_param("BlackLevel", v))
        self.combo_pixfmt.currentTextChanged.connect(self._on_pixfmt_changed)
        self.cb_fps_cap.toggled.connect(lambda v: self._push_cam_param("AcquisitionFrameRateEnable", v))
        self.sp_fps_cap.valueChanged.connect(lambda v: self._push_cam_param("AcquisitionFrameRate", v))

        # ── recording row ─────────────────────────────────────────
        rec_row = QHBoxLayout()

        self.btn_rec = QPushButton("● Record"); self.btn_rec.setFixedWidth(90)
        self.btn_rec.setToolTip("Start / stop recording the live feed to MP4")
        self.btn_rec.clicked.connect(self._toggle_recording)

        # RAM-buffer spinbox
        rec_row.addWidget(QLabel("RAM:"))
        self.sp_buf = NoScrollSpinBox()
        self.sp_buf.setRange(64, 8192); self.sp_buf.setValue(1024)
        self.sp_buf.setSuffix(" MB"); self.sp_buf.setFixedWidth(80)
        self.sp_buf.setToolTip("Maximum RAM to use for the recording buffer.\n"
                                "Raised default for high-resolution/high-fps cameras — "
                                "a 256MB buffer fills in well under a second at "
                                "2840x2840/44fps and triggers premature downscaling.")

        # Fill bar: green → amber → red
        self.buf_bar = QProgressBar()
        self.buf_bar.setRange(0, 100); self.buf_bar.setValue(0)
        self.buf_bar.setFormat("buf %p%"); self.buf_bar.setMaximumHeight(14)
        self.buf_bar.setFixedWidth(110)
        self.buf_bar.setToolTip(
            f"Buffer fill level.\n"
            f"Above {int(RecordingBuffer.HIGH_WATERMARK*100)}%: "
            f"incoming frames are halved in resolution to relieve pressure.\n"
            f"Below {int(RecordingBuffer.LOW_WATERMARK*100)}%: full resolution restored."
        )

        self.lbl_rec = QLabel("—"); self.lbl_rec.setStyleSheet("color:#888;")

        self.btn_rec_dir = QPushButton("Save to…"); self.btn_rec_dir.setFixedWidth(70)
        self.btn_rec_dir.setToolTip(f"Output folder (currently: {self._rec_out_dir})")
        self.btn_rec_dir.clicked.connect(self._choose_rec_dir)

        for w in (self.btn_rec, self.sp_buf, self.buf_bar, self.lbl_rec, self.btn_rec_dir):
            rec_row.addWidget(w)
        rec_row.addStretch(); lay.addLayout(rec_row)

        # Timer to keep the elapsed-time label alive during recording
        self._rec_tick = QTimer(self)
        self._rec_tick.timeout.connect(self._refresh_rec_label)
        self._rec_tick.start(500)

        # ── shutter / timelapse group (Sutter Lambda SC) ───────────
        sh_group = CollapsibleGroupBox("Shutter & timelapse (Lambda SC)")
        sh_group.toggle_btn.setToolTip(
            "Controls for a Sutter Instrument Lambda SC shutter controller — the\n"
            "hardware that opens/closes an external light-path shutter, used here\n"
            "to automate timelapse image capture. Not needed unless you have this\n"
            "specific hardware connected.")
        sh_lay = QVBoxLayout()
        sh_group.setLayout(sh_lay)

        # Enable checkbox: gates visibility of all the detailed sub-controls
        # below so the group takes up almost no space for users who never use
        # timelapse/shutter hardware — just grayed-out was still visual clutter.
        self.cb_shutter_enable = QCheckBox("Enable time-lapse shutter")
        self.cb_shutter_enable.setToolTip(
            "Show/hide the Sutter Lambda SC shutter and timelapse controls.\n"
            "Leave unchecked if you don't have this hardware — keeps the panel compact.")
        self.cb_shutter_enable.toggled.connect(self._on_shutter_enable_toggled)
        sh_lay.addWidget(self.cb_shutter_enable)

        # Sub-controls container — shown/hidden as a whole via the checkbox above.
        self._shutter_sub = QWidget()
        sh_sub_lay = QVBoxLayout(self._shutter_sub)
        sh_sub_lay.setContentsMargins(0, 0, 0, 0)
        self._shutter_sub.setVisible(False)

        sh_row1 = QHBoxLayout()
        sh_row1.addWidget(QLabel("Port:"))
        self.combo_shutter_port = NoScrollComboBox(); self.combo_shutter_port.setMinimumWidth(110)
        sh_row1.addWidget(self.combo_shutter_port)
        self.btn_shutter_scan = QPushButton("Scan"); self.btn_shutter_scan.setFixedWidth(50)
        self.btn_shutter_scan.clicked.connect(self._scan_shutter_ports)
        sh_row1.addWidget(self.btn_shutter_scan)
        self.btn_shutter_conn = QPushButton("Connect"); self.btn_shutter_conn.setFixedWidth(80)
        self.btn_shutter_conn.clicked.connect(self._toggle_shutter)
        sh_row1.addWidget(self.btn_shutter_conn)
        self.btn_shutter_open  = QPushButton("Open");  self.btn_shutter_open.setFixedWidth(55)
        self.btn_shutter_close = QPushButton("Close"); self.btn_shutter_close.setFixedWidth(55)
        self.btn_shutter_open.clicked.connect(self._manual_shutter_open)
        self.btn_shutter_close.clicked.connect(self._manual_shutter_close)
        for w in (self.btn_shutter_open, self.btn_shutter_close): w.setEnabled(False)
        sh_row1.addWidget(self.btn_shutter_open); sh_row1.addWidget(self.btn_shutter_close)
        self.lbl_shutter_st = QLabel("Not connected"); self.lbl_shutter_st.setStyleSheet("color:#888;")
        sh_row1.addWidget(self.lbl_shutter_st)
        sh_row1.addStretch()
        sh_sub_lay.addLayout(sh_row1)

        sh_row2 = QHBoxLayout()
        sh_row2.addWidget(QLabel("Interval:"))
        self.sp_tl_interval = NoScrollDoubleSpinBox()
        self.sp_tl_interval.setRange(1.0, 86400.0); self.sp_tl_interval.setValue(60.0)
        self.sp_tl_interval.setSuffix(" s"); self.sp_tl_interval.setDecimals(1)
        self.sp_tl_interval.setFixedWidth(90)
        self.sp_tl_interval.setToolTip("Time between successive timelapse captures, in seconds.")
        sh_row2.addWidget(self.sp_tl_interval)

        sh_row2.addWidget(QLabel("Settle:"))
        self.sp_tl_settle = NoScrollDoubleSpinBox()
        self.sp_tl_settle.setRange(0.0, 30.0); self.sp_tl_settle.setValue(0.2)
        self.sp_tl_settle.setSuffix(" s"); self.sp_tl_settle.setDecimals(2)
        self.sp_tl_settle.setFixedWidth(80)
        self.sp_tl_settle.setToolTip("Delay after opening the shutter before the frame is\n"
                                       "captured — lets illumination/exposure settle.")
        sh_row2.addWidget(self.sp_tl_settle)

        sh_row2.addWidget(QLabel("Count:"))
        self.sp_tl_count = NoScrollSpinBox()
        self.sp_tl_count.setRange(0, 1_000_000); self.sp_tl_count.setValue(0)
        self.sp_tl_count.setFixedWidth(80)
        self.sp_tl_count.setToolTip("Total number of captures to take. 0 = run until Stop is pressed.")
        sh_row2.addWidget(self.sp_tl_count)

        self.btn_tl_dir = QPushButton("Save to…"); self.btn_tl_dir.setFixedWidth(70)
        self.btn_tl_dir.setToolTip(f"Timelapse output folder (currently: {self._tl_out_dir})")
        self.btn_tl_dir.clicked.connect(self._choose_tl_dir)
        sh_row2.addWidget(self.btn_tl_dir)

        self.btn_tl_start = QPushButton("▶ Start timelapse")
        self.btn_tl_start.setEnabled(False)
        self.btn_tl_start.clicked.connect(self._toggle_timelapse)
        sh_row2.addWidget(self.btn_tl_start)
        sh_row2.addStretch()
        sh_sub_lay.addLayout(sh_row2)

        self.lbl_tl_st = QLabel("—"); self.lbl_tl_st.setStyleSheet("color:#888;")
        sh_sub_lay.addWidget(self.lbl_tl_st)

        sh_lay.addWidget(self._shutter_sub)

        lay.addWidget(sh_group)
        self._scan_shutter_ports()

        self._scan()

    # ── camera control ─────────────────────────────────────────────

    def _scan(self):
        self.combo.blockSignals(True); self.combo.clear()
        self._cameras = detect_cameras()
        if self._cameras:
            for c in self._cameras: self.combo.addItem(c["label"])
        else:
            self.combo.addItem("(no cameras found)")
        self.combo.blockSignals(False)

    def _start_ana_worker(self):
        """Start the AnalysisWorker (live analysis) thread on demand — only
        while Compare mode is actually active, so it isn't left running idle
        (or, worse, still receiving frames) once the user turns it off."""
        if self._ana_worker and self._ana_worker.isRunning():
            return
        self._ana_worker = AnalysisWorker(self._layer, self._params, self)
        self._ana_worker.set_fps_cap(self.sp_ana_fps.value())
        self._ana_worker.analysis_ready.connect(self._on_analysis)
        self._ana_worker.start()

    def _stop_ana_worker(self):
        """Stop and release the AnalysisWorker thread. Waits briefly so the
        thread object isn't deleted while run() may still be executing."""
        if self._ana_worker:
            self._ana_worker.stop()
            self._ana_worker.wait(500)
            self._ana_worker.deleteLater()
            self._ana_worker = None

    def _toggle(self):
        if self._worker and self._worker.isRunning():
            if self._recording: self._stop_recording()
            self._stop_ana_worker()
            self._worker.stop()
            self._worker.finished.connect(self._on_cam_stopped)
            self.btn_conn.setEnabled(False)
            self.lbl_st.setText("Stopping…")
        else:
            if not self._cameras:
                QMessageBox.information(self,"No cameras","No cameras found. Click Scan."); return
            idx = self.combo.currentIndex()
            if idx < 0 or idx >= len(self._cameras): return
            cam_info = self._cameras[idx]
            if cam_info["type"] == "basler" and not PYPYLON_OK:
                QMessageBox.warning(self,"pypylon missing","pip install pypylon"); return
            # AnalysisWorker (live analysis) is only needed in Compare mode —
            # starting it here unconditionally meant it kept running (idle,
            # but never stopped) even with Compare off, and toggling Compare
            # off didn't tear it down either. Start it only if Compare is
            # already checked; _toggle_compare() now starts/stops it on demand.
            if self._compare_mode:
                self._start_ana_worker()
            self._worker = CameraWorker(cam_info, self._live_max_px, self)
            self._worker.frame_ready.connect(self._on_frame)
            self._worker.frame16_ready.connect(self._on_frame16)
            self._worker.error.connect(lambda e: self.lbl_st.setText(f"Err: {e[:60]}"))
            self._worker.ranges_ready.connect(self._on_cam_ranges)
            self._worker.temperature_ready.connect(self._on_cam_temp)
            self._worker.start()
            self.btn_conn.setText("Disconnect")
            self.btn_conn.setEnabled(True)
            self.lbl_st.setText(f"Live — {cam_info['label'][:40]}")

    def _push_cam_param(self, name: str, value):
        if self._worker and self._worker.isRunning():
            self._worker.set_param(name, value)

    def _on_pixfmt_changed(self, fmt: str):
        if fmt and self._worker and self._worker.isRunning():
            self._worker.set_pixel_format(fmt)

    def _on_pixfmt_for_lossless(self, fmt: str):
        """Enable the Lossless checkbox only when Mono12 is selected."""
        self.cb_lossless.setEnabled("12" in fmt)

    def _on_ana_fps_changed(self, fps: float):
        if self._ana_worker:
            self._ana_worker.set_fps_cap(fps)

    def _on_cam_ranges(self, ranges: dict):
        """Populate control ranges/current values once the camera reports them
        (emitted once after Open()); only then do the controls become meaningful."""
        def _cfg(spin, key, lo_fb, hi_fb):
            lo, hi, cur = ranges.get(key, (lo_fb, hi_fb, None))
            spin.blockSignals(True)
            spin.setRange(lo, hi)
            if cur is not None: spin.setValue(cur)
            spin.blockSignals(False)

        _cfg(self.sp_exposure,   "ExposureTime", 20, 1_000_000)
        _cfg(self.sp_gain,       "Gain", 0, 48)
        _cfg(self.sp_blacklevel, "BlackLevel", 0, 64)
        if "AcquisitionFrameRate" in ranges:
            lo, hi, cur = ranges["AcquisitionFrameRate"]
            self.sp_fps_cap.blockSignals(True)
            self.sp_fps_cap.setRange(lo, hi); self.sp_fps_cap.setValue(cur)
            self.sp_fps_cap.blockSignals(False)

        opts = ranges.get("PixelFormat_options")
        if opts:
            mono_opts = [o for o in opts if "Mono" in o] or opts
            self.combo_pixfmt.blockSignals(True)
            self.combo_pixfmt.clear()
            self.combo_pixfmt.addItems(mono_opts)
            cur = ranges.get("PixelFormat_current")
            if cur in mono_opts:
                self.combo_pixfmt.setCurrentText(cur)
            self.combo_pixfmt.blockSignals(False)

        for w in self._cam_ctrl_widgets:
            w.setEnabled(True)

    def _on_cam_temp(self, temp_c: float):
        self.lbl_temp.setText(f"Temp: {temp_c:.1f}°C")

    def _on_res_changed(self, idx: int):
        self._live_max_px = self.combo_res.itemData(idx)
        if self._worker:
            self._worker.set_live_max_px(self._live_max_px)

    def _toggle_compare(self, checked: bool):
        self._compare_mode = checked
        self.lbl_ann.setVisible(checked)
        self._cap_row.setVisible(checked)
        for w in self._compare_sub_widgets:
            w.setVisible(checked)
        if checked:
            # Entering compare mode: (re)start live analysis and ensure raw is on left
            if self._worker and self._worker.isRunning():
                self._start_ana_worker()
            if self._last_bgr is not None:
                self.lbl.set_frame(self._last_bgr)
            if self._last_ann is not None:
                self.lbl_ann.set_frame(self._last_ann)
        else:
            # Leaving compare mode: stop live analysis (it was the source of
            # the lingering CPU/thread overhead — Issue 4) and clear stale
            # annotation, show raw.
            self._stop_ana_worker()
            self._last_ann = None
            if self._last_bgr is not None:
                self.lbl.set_frame(self._last_bgr)

    def _on_cam_stopped(self):
        """Called when the camera worker thread actually finishes (no main-thread wait)."""
        if self._worker:
            self._worker.deleteLater()
            self._worker = None
        if self._ana_worker:
            # Must wait for the thread to finish before deleteLater(); deleting a
            # QThread whose run() is still executing causes a C-level crash.
            self._ana_worker.wait(500)
            self._ana_worker.deleteLater()
            self._ana_worker = None
        self._last_bgr = None
        self._last_ann = None
        self.btn_conn.setText("Connect")
        self.btn_conn.setEnabled(True)
        self.lbl_st.setText("Disconnected")
        self.lbl.clear("Camera feed")
        if self._compare_mode:
            self.lbl_ann.clear("Tracking overlay")
        for w in self._cam_ctrl_widgets:
            w.setEnabled(False)
        self.lbl_temp.setText("Temp: —")

    # ── frame handler (display only — fast path) ───────────────────

    def _on_frame(self, bgr: np.ndarray, cam_scale: float):
        """Store the latest frame, show raw feed, and forward to AnalysisWorker
        only when compare mode is active."""
        self._last_bgr       = bgr
        self._last_cam_scale = cam_scale
        # Apply image enhancement to display (not to recording buffer or analysis feed).
        # When Compare mode is active, compute the shared denoise/gamma/sharpen prefix
        # once and pass it to AnalysisWorker so those expensive steps are not repeated.
        p = self._params
        gamma      = p.get("gamma", 1.0)
        sharpen    = p.get("sharpen_amount", 0.0)
        denoise    = p.get("denoise_method", "off")
        dns        = p.get("denoise_strength", 10.0)
        flatten    = bool(p.get("flatten_illum", False))
        flatten_sf = p.get("flatten_sigma_frac", 0.15)
        has_enh    = gamma != 1.0 or sharpen > 0 or denoise != "off" or flatten
        if has_enh:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            # stop_after_sharpen=True → returns after denoise+gamma+sharpen, before
            # CLAHE/bandpass. This is exactly what the display needs AND forms the
            # shared prefix for AnalysisWorker (which adds CLAHE/bandpass separately).
            # flatten_illum runs first (inside _preprocess, before denoise), so it
            # must be included here too — AnalysisWorker's preprocessed_prefix path
            # skips straight past that step.
            enh  = _preprocess(gray, False, 1, 53, False, 2.0,
                               denoise=denoise, denoise_strength=dns,
                               gamma=gamma, sharpen=sharpen,
                               flatten_illum=flatten, flatten_sigma_frac=flatten_sf,
                               stop_after_sharpen=True)
            u8   = cv2.normalize(enh, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disp = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
            prefix = enh   # float32 grayscale shared with AnalysisWorker
        else:
            disp   = bgr
            prefix = None  # no preprocessing — AnalysisWorker does full pipeline
        self.lbl.set_frame(disp)
        # Analysis overlay only in compare mode (right label)
        if self._compare_mode:
            if self._last_ann is not None:
                self.lbl_ann.set_frame(self._last_ann)
            if self._ana_worker and self._ana_worker.isRunning():
                self._ana_worker.update_frame(bgr, cam_scale, preprocessed_prefix=prefix)

        # ── buffer recording ───────────────────────────────────────
        if self._recording and self._rec_buf is not None:
            fill = self._rec_buf.fill
            # Hysteresis: engage downscale above HIGH, release below LOW
            if fill >= RecordingBuffer.HIGH_WATERMARK:
                self._downscaling = True
            elif fill <= RecordingBuffer.LOW_WATERMARK:
                self._downscaling = False

            if self._downscaling:
                h, w = bgr.shape[:2]
                store = cv2.resize(bgr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            else:
                store = bgr

            self._rec_buf.put(store)   # silently drops if truly full

            # Colour-code the fill bar
            pct = int(fill * 100)
            self.buf_bar.setValue(pct)
            if fill < 0.40:
                chunk_clr = "#2a8a2a"
            elif fill < RecordingBuffer.HIGH_WATERMARK:
                chunk_clr = "#aa6600"
            else:
                chunk_clr = "#aa2222"
            self.buf_bar.setStyleSheet(f"QProgressBar::chunk{{background:{chunk_clr};}}")

        # Allow the camera thread to emit its next frame now that this one is
        # fully handled — without this, a slow enhancement step (NLM, CLAHE)
        # lets queued frame_ready events pile up unbounded and exhaust memory.
        if self._worker:
            self._worker.mark_frame_done()

    # ── Mono12 lossless frame handler ──────────────────────────────

    def _on_frame16(self, frame16: np.ndarray):
        """Receive raw uint16 Mono12 frame; forward to TiffSequenceRecorder if active."""
        if self._recording and self._tiff_recorder is not None:
            self._tiff_recorder.push(frame16)

    # ── analysis result from AnalysisWorker ────────────────────────

    def _on_analysis(self, ann: np.ndarray):
        """Receive annotated frame from AnalysisWorker.
        Only update the display if compare mode is currently active."""
        self._last_ann = ann
        if self._compare_mode:
            self.lbl_ann.set_frame(ann)

    # ── recording control ──────────────────────────────────────────

    def _toggle_recording(self):
        if self._recording: self._stop_recording()
        else:               self._start_recording()

    def _start_recording(self):
        if self._worker is None or not self._worker.isRunning():
            QMessageBox.warning(self, "Not connected", "Connect to a camera first."); return
        if self._last_bgr is None:
            QMessageBox.warning(self, "No frame", "Waiting for the first frame…"); return

        # Build auto-filename:  <DeviceName>_YYYY-MM-DD_HH-MM-SS.mp4
        idx = self.combo.currentIndex()
        cam_info  = self._cameras[idx] if 0 <= idx < len(self._cameras) else {}
        raw_label = cam_info.get("label", "camera")
        safe      = re.sub(r"[^\w.-]", "_", raw_label)
        safe      = re.sub(r"_+", "_", safe).strip("_")[:40]
        ts        = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname     = f"{safe}_{ts}.mp4"
        out_path  = str(self._rec_out_dir / fname)

        fps      = getattr(self._worker, "fps", 30.0)
        h, w     = self._last_bgr.shape[:2]
        max_mb   = int(self.sp_buf.value())

        # Disconnect any stale signals from a previous recorder that is still
        # draining (e.g. user stopped and quickly started again).  This prevents
        # the old recorder's finished/error from corrupting the new recording's
        # state when it eventually arrives in the GUI thread.
        if self._recorder is not None:
            try:
                self._recorder.finished.disconnect(self._on_rec_finished)
                self._recorder.error.disconnect(self._on_rec_error)
            except Exception:
                pass

        self._rec_buf  = RecordingBuffer(max_mb)
        self._recorder = RecorderWorker(self._rec_buf, out_path, fps, (w, h), self)
        self._recorder.progress.connect(lambda pct, _: self.buf_bar.setValue(pct))
        self._recorder.finished.connect(self._on_rec_finished)
        self._recorder.error.connect(self._on_rec_error)
        self._recorder.start()

        # Lossless Mono12 path: writes the full 12-bit sensor data as a TIFF
        # sequence (via _on_frame16/frame16_ready) alongside the tone-mapped
        # 8-bit MP4 above. Only meaningful when PixelFormat==Mono12, which is
        # exactly when cb_lossless is enabled (see _on_pixfmt_for_lossless).
        if self.cb_lossless.isChecked():
            tiff_dir = self._rec_out_dir / f"{Path(fname).stem}_lossless"
            self._tiff_recorder = TiffSequenceRecorder(str(tiff_dir), parent=self)
            self._tiff_recorder.error.connect(self._on_tiff_error)
            self._tiff_recorder.start()
        else:
            self._tiff_recorder = None

        self._recording   = True
        self._rec_start   = datetime.datetime.now()
        self._downscaling = False

        self.btn_rec.setText("■ Stop Rec")
        self.btn_rec.setStyleSheet("background:#882222;color:white;font-weight:bold;")
        self.lbl_rec.setStyleSheet("color:#ff6666;")
        self.lbl_rec.setText(f"REC → {fname}")

    def _stop_recording(self):
        self._recording   = False
        self._downscaling = False
        if self._recorder:
            self._recorder.stop()     # drains buffer then emits finished
        if self._tiff_recorder:
            self._tiff_recorder.stop()   # drains queue then emits finished
            self._tiff_recorder.wait(5000)
            self._tiff_recorder.deleteLater()
            self._tiff_recorder = None
        self.btn_rec.setText("● Record")
        self.btn_rec.setStyleSheet("")
        self.lbl_rec.setStyleSheet("color:#888;")
        self.lbl_rec.setText("Saving…")

    def _on_rec_finished(self, path):
        self._recorder = None; self._rec_buf = None
        self.buf_bar.setValue(0)
        self.buf_bar.setStyleSheet("")
        self.lbl_rec.setText(f"Saved: {Path(path).name}")
        self.lbl_rec.setStyleSheet("color:#66cc66;")
        QMessageBox.information(self, "Recording saved",
                                f"Recording saved to:\n{path}")

    def _on_rec_error(self, msg):
        self._recording = False
        self._recorder  = None; self._rec_buf = None
        self.btn_rec.setText("● Record"); self.btn_rec.setStyleSheet("")
        self.buf_bar.setValue(0); self.buf_bar.setStyleSheet("")
        self.lbl_rec.setText("Rec error"); self.lbl_rec.setStyleSheet("color:#ff4444;")
        QMessageBox.critical(self, "Recording error", msg[:800])

    def _on_tiff_error(self, msg):
        self._tiff_recorder = None
        QMessageBox.critical(self, "Lossless recording error", msg[:800])

    def _refresh_rec_label(self):
        """Update elapsed time + buffer stats every 500 ms while recording."""
        if not self._recording or self._rec_start is None: return
        elapsed = int((datetime.datetime.now() - self._rec_start).total_seconds())
        m, s    = divmod(elapsed, 60)
        buf_pct = int(self._rec_buf.fill * 100) if self._rec_buf else 0
        ds_flag = " ↓res" if self._downscaling else ""
        self.lbl_rec.setText(f"REC {m:02d}:{s:02d}  {buf_pct}%{ds_flag}")

    def _choose_rec_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Choose recording output folder", str(self._rec_out_dir)
        )
        if d:
            self._rec_out_dir = Path(d)
            self.btn_rec_dir.setToolTip(f"Output folder: {self._rec_out_dir}")

    # ── shutter control / timelapse (Lambda SC) ─────────────────────

    def _on_shutter_enable_toggled(self, checked: bool):
        """Show/hide the shutter & timelapse sub-controls (Task 2: removable
        shutter settings). Disabling also stops any running timelapse and
        disconnects the shutter hardware, rather than leaving it silently
        connected behind a hidden panel."""
        self._shutter_sub.setVisible(checked)
        if not checked:
            if self._tl_worker and self._tl_worker.isRunning():
                self._stop_timelapse()
            if self._shutter.is_connected:
                self._toggle_shutter()

    def _scan_shutter_ports(self):
        self.combo_shutter_port.blockSignals(True)
        self.combo_shutter_port.clear()
        ports = LambdaSCController.list_ports()
        if ports:
            self.combo_shutter_port.addItems(ports)
        else:
            self.combo_shutter_port.addItem("(no serial ports found)")
        self.combo_shutter_port.blockSignals(False)

    def _toggle_shutter(self):
        if self._shutter.is_connected:
            if self._tl_worker and self._tl_worker.isRunning():
                self._stop_timelapse()
                # Must actually wait for the cycle to exit before disconnecting —
                # otherwise disconnect() can close the serial handle while the
                # worker is still blocked inside a read(), or (pre-fix) racing
                # the handle close outright. Generous timeout: worst case is one
                # full serial round-trip timeout (2s) plus in-flight TIFF write.
                self._tl_worker.wait(5000)
            self._shutter.disconnect()
            self.btn_shutter_conn.setText("Connect")
            self.lbl_shutter_st.setText("Not connected"); self.lbl_shutter_st.setStyleSheet("color:#888;")
            for w in (self.btn_shutter_open, self.btn_shutter_close, self.btn_tl_start):
                w.setEnabled(False)
            return
        if not PYSERIAL_OK:
            QMessageBox.warning(self, "pyserial missing", "pip install pyserial"); return
        port = self.combo_shutter_port.currentText()
        if not port or port.startswith("("):
            QMessageBox.information(self, "No port", "Click Scan and select a serial port first."); return
        try:
            self._shutter.connect(port)
        except Exception as exc:
            QMessageBox.critical(self, "Shutter connection failed", str(exc)); return
        self.btn_shutter_conn.setText("Disconnect")
        self.lbl_shutter_st.setText("Connected"); self.lbl_shutter_st.setStyleSheet("color:#66cc66;")
        for w in (self.btn_shutter_open, self.btn_shutter_close, self.btn_tl_start):
            w.setEnabled(True)

    def _manual_shutter_open(self):
        try: self._shutter.open_shutter()
        except Exception as exc: QMessageBox.warning(self, "Shutter error", str(exc))

    def _manual_shutter_close(self):
        try: self._shutter.close_shutter()
        except Exception as exc: QMessageBox.warning(self, "Shutter error", str(exc))

    def _choose_tl_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Choose timelapse output folder", str(self._tl_out_dir)
        )
        if d:
            self._tl_out_dir = Path(d)
            self.btn_tl_dir.setToolTip(f"Timelapse output folder: {self._tl_out_dir}")

    def _toggle_timelapse(self):
        if self._tl_worker and self._tl_worker.isRunning():
            self._stop_timelapse()
        else:
            self._start_timelapse()

    def _start_timelapse(self):
        if not self._shutter.is_connected:
            QMessageBox.warning(self, "Not connected", "Connect the Lambda SC first."); return
        self._tl_worker = TimelapseWorker(
            self._shutter, lambda: self._last_bgr, str(self._tl_out_dir),
            interval_s=self.sp_tl_interval.value(), settle_s=self.sp_tl_settle.value(),
            max_captures=self.sp_tl_count.value(), parent=self)
        self._tl_worker.captured.connect(self._on_tl_captured)
        self._tl_worker.progress.connect(self._on_tl_progress)
        self._tl_worker.finished.connect(self._on_tl_finished)
        self._tl_worker.error.connect(self._on_tl_error)
        self._tl_worker.start()
        self.btn_tl_start.setText("■ Stop timelapse")
        self.btn_tl_start.setStyleSheet("background:#882222;color:white;font-weight:bold;")
        for w in (self.btn_shutter_open, self.btn_shutter_close,
                  self.sp_tl_interval, self.sp_tl_settle, self.sp_tl_count):
            w.setEnabled(False)
        self.lbl_tl_st.setText("Starting…")

    def _stop_timelapse(self):
        if self._tl_worker:
            self._tl_worker.stop()

    def _on_tl_captured(self, n: int, path: str):
        self.lbl_tl_st.setText(f"Captured {n}: {Path(path).name}")

    def _on_tl_progress(self, n: int, secs_left: float):
        self.lbl_tl_st.setText(f"Captured {n} — next in {secs_left:.0f}s")

    def _on_tl_finished(self, n: int):
        # finished is emitted from the worker thread's run()-ending finally
        # block, just before the thread actually terminates — wait() here
        # blocks until it's truly done before deleteLater(), same pattern as
        # AnalysisWorker (deleting a still-running QThread is a C-level crash).
        if self._tl_worker:
            self._tl_worker.wait(5000)
            self._tl_worker.deleteLater()
        self._tl_worker = None
        self.btn_tl_start.setText("▶ Start timelapse")
        self.btn_tl_start.setStyleSheet("")
        for w in (self.btn_shutter_open, self.btn_shutter_close,
                  self.sp_tl_interval, self.sp_tl_settle, self.sp_tl_count):
            w.setEnabled(True)
        self.lbl_tl_st.setText(f"Timelapse finished — {n} frames captured")

    def _on_tl_error(self, msg: str):
        QMessageBox.warning(self, "Timelapse error", msg[:600])

    # ── auto grain-size detection ──────────────────────────────────

    def _on_auto_size(self):
        if self._last_bgr is None:
            self.lbl_st.setText("No frame — connect camera first")
            return
        if self._gs_worker and self._gs_worker.isRunning():
            return
        self.btn_auto_size.setEnabled(False)
        self.btn_auto_size.setText("Scanning…")
        self._gs_worker = GrainSizeWorker(self._last_bgr.copy(), self._params, self)
        self._gs_worker.detected.connect(self._on_grain_detected)
        self._gs_worker.failed.connect(self._on_grain_failed)
        self._gs_worker.progress.connect(
            lambda _pct, msg: self.lbl_st.setText(msg)
        )
        self._gs_worker.start()

    def _on_grain_detected(self, diam: int, count: int):
        self.btn_auto_size.setEnabled(True)
        self.btn_auto_size.setText("Auto size")
        # The frame was captured at live resolution; scale back to original px.
        orig = max(3, round(diam / max(self._last_cam_scale, 0.01))) | 1
        self._params = dict(self._params)
        self._params["diameter"]   = orig
        self._params["separation"] = orig
        if self._ana_worker:
            self._ana_worker.update_config(self._layer, self._params)
        self.lbl_st.setText(f"Auto size: {orig} px  ({count} round particles)")
        self.grain_size_detected.emit(orig)

    def _on_grain_failed(self, msg: str):
        self.btn_auto_size.setEnabled(True)
        self.btn_auto_size.setText("Auto size")
        self.lbl_st.setText(f"Auto size: {msg[:70]}")

    # ── layer / params update ──────────────────────────────────────

    def update_layer(self, l):
        self._layer = l
        if self._ana_worker:
            self._ana_worker.update_config(l, self._params)

    def update_params(self, p):
        self._params = p
        if self._ana_worker:
            self._ana_worker.update_config(self._layer, p)




# ════════════════════════════════════════════════════════════════════
# Main window  (QDockWidget layout + menus + settings)
# ════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self, init_video=None):
        super().__init__()
        self.setWindowTitle(f"Colloid Tracker  {APP_VERSION}")
        self.resize(1700, 960)
        self.setMinimumSize(1200, 700)
        self._data=None; self._worker=None; self._vpath=None; self._progress_dlg=None
        self._track_start_t=None
        self._sf_worker=None   # StructureFunctionWorker (g(r)/g6(r) trajectory/custom-range compute)
        self._settings=SettingsManager()
        self._build(); self._connect(); self._theme()
        self._load_settings()
        if init_video: self._open(Path(init_video))
        self._check_deps()

    # ── build ──────────────────────────────────────────────────────

    def _build(self):
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks |
            QMainWindow.DockOption.AllowNestedDocks |
            QMainWindow.DockOption.AllowTabbedDocks
        )

        # ── status bar ────────────────────────────────────────────
        self.status_bar=QStatusBar(); self.setStatusBar(self.status_bar)

        # ── dep bar (hidden until shown) ──────────────────────────
        self._dep_bar=None   # created in _check_deps if needed

        # ── central widget: tab widget with video panes ────────────
        self.tabs=QTabWidget()

        self.vid_pane=VideoPane()
        self.tabs.addTab(self.vid_pane,"Video Analysis")

        self.cam_pane=CameraPane(LayerConfig(),{**_DEFAULT_PARAMS})
        self.tabs.addTab(self.cam_pane,"Live Camera")

        # wrap in vbox so we can slot in the dep bar above
        central=QWidget(); cl=QVBoxLayout(central); cl.setContentsMargins(0,0,0,0); cl.setSpacing(0)
        cl.addWidget(self.tabs,stretch=1)
        self.setCentralWidget(central)

        # ── dock: Parameters ──────────────────────────────────────
        self.param_panel=ParamPanel()
        self._dock_params=self._make_dock("Parameters",self.param_panel,
                                          Qt.DockWidgetArea.LeftDockWidgetArea)

        # ── dock: Layers ──────────────────────────────────────────
        self.layer_panel=LayerPanel()
        self._dock_layers=self._make_dock("Layers",self.layer_panel,
                                          Qt.DockWidgetArea.LeftDockWidgetArea)
        self.tabifyDockWidget(self._dock_params,self._dock_layers)
        self._dock_params.raise_()   # show Parameters tab first

        # ── dock: Diagnostics ─────────────────────────────────────
        self.diag_panel=DiagnosticsPanel()
        self._dock_diag=self._make_dock("Diagnostics",self.diag_panel,
                                        Qt.DockWidgetArea.RightDockWidgetArea)

        # ── dock: Defect plot ─────────────────────────────────────
        self.defect_plot=DefectPlot()
        self._dock_plot=self._make_dock("Defect Count Plot",self.defect_plot,
                                        Qt.DockWidgetArea.BottomDockWidgetArea)

        # ── dock: Event log ───────────────────────────────────────
        self.log_table=LogTable()
        self._dock_log=self._make_dock("Event Log",self.log_table,
                                       Qt.DockWidgetArea.BottomDockWidgetArea)
        self.tabifyDockWidget(self._dock_plot,self._dock_log)
        self._dock_plot.raise_()

        # ── progress + toolbar ────────────────────────────────────
        self.prog=QProgressBar(); self.prog.setVisible(False); self.prog.setTextVisible(True)
        self.prog.setMaximumHeight(18)

        tb=self.addToolBar("Main")
        tb.setMovable(False); tb.setIconSize(QSize(16,16))

        self.btn_open=QPushButton("Open video…")
        self.btn_track=QPushButton("▶  Track & Analyse")
        self.btn_track.setStyleSheet("background:#2a5a2a;color:white;font-weight:bold;padding:3px 10px;")
        self.btn_track.setEnabled(False)
        self.btn_stop=QPushButton("■  Stop"); self.btn_stop.setEnabled(False)
        self.btn_export=QPushButton("Export MP4…"); self.btn_export.setEnabled(False)
        self.btn_save_set=QPushButton("Save settings")
        self.btn_save_preset=QPushButton("Save preset…")
        self.btn_load_preset=QPushButton("Load preset…")

        for w in(self.btn_open,self.btn_track,self.btn_stop,self.btn_export):
            tb.addWidget(w)
        tb.addSeparator()
        for w in(self.btn_save_set,self.btn_save_preset,self.btn_load_preset):
            tb.addWidget(w)
        tb.addSeparator()
        tb.addWidget(self.prog)
        # give the progress bar some width
        self.prog.setMinimumWidth(200)

        # ── menu bar ──────────────────────────────────────────────
        mb=self.menuBar()

        def _act(menu, label, slot, shortcut=None):
            a = QAction(label, self); a.triggered.connect(slot)
            if shortcut: a.setShortcut(shortcut)
            menu.addAction(a); return a

        # File
        fm=mb.addMenu("File")
        self._a_open=_act(fm,"Open video…",self._browse,"Ctrl+O")
        fm.addSeparator()
        self._a_export=_act(fm,"Export annotated MP4…",self._export)
        self._a_export.setEnabled(False)
        fm.addSeparator()
        _act(fm,"Quit",self.close,"Ctrl+Q")

        # View
        vm=mb.addMenu("View")
        vm.addAction(self._dock_params.toggleViewAction())
        vm.addAction(self._dock_layers.toggleViewAction())
        vm.addAction(self._dock_diag.toggleViewAction())
        vm.addAction(self._dock_plot.toggleViewAction())
        vm.addAction(self._dock_log.toggleViewAction())
        vm.addSeparator()
        _act(vm,"Reset layout",self._reset_layout)

        # Settings
        sm=mb.addMenu("Settings")
        _act(sm,"Save settings",self._save_settings_now,"Ctrl+S")
        _act(sm,"Save preset…",self._save_preset)
        _act(sm,"Load preset…",self._load_preset)
        sm.addSeparator()
        _act(sm,"Open settings file…",self._open_settings_file)

        # Camera
        cm=mb.addMenu("Camera")
        _act(cm,"Scan for cameras",self.cam_pane._scan)

        # Help
        hm=mb.addMenu("Help")
        _act(hm,"Parameter Tuning Guide…", self._show_help)
        hm.addSeparator()
        _act(hm,"About",self._about)
        hm.addAction("Install pypylon…",self._pypylon_help)

    def _make_dock(self, title, widget, area):
        dock=QDockWidget(title,self)
        dock.setObjectName(title.replace(" ","_"))
        dock.setWidget(widget)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(area,dock)
        return dock

    # ── signals ────────────────────────────────────────────────────

    def _connect(self):
        self.btn_open.clicked.connect(self._browse)
        self.btn_track.clicked.connect(self._start_tracking)
        self.btn_stop.clicked.connect(self._stop_tracking)
        self.btn_export.clicked.connect(self._export)
        self.btn_save_set.clicked.connect(self._save_settings_now)
        self.btn_save_preset.clicked.connect(self._save_preset)
        self.btn_load_preset.clicked.connect(self._load_preset)
        # _a_open and _a_export already wired via _act() in _build()
        self.layer_panel.layer_changed.connect(self._on_layer)
        self.param_panel.params_changed.connect(self._on_params)
        self.vid_pane.frame_changed.connect(self.defect_plot.mark_frame)
        # Auto grain-size detection → update diameter spinbox
        self.cam_pane.grain_size_detected.connect(self._on_grain_size)
        self.vid_pane.grain_size_suggested.connect(self._on_grain_size)
        # Diagnostics panel
        self.vid_pane.preview_feats_ready.connect(self._on_preview_feats)
        self.diag_panel.apply_params.connect(self._on_diag_apply)
        # Structural correlations (g(r)/g6(r)) + defect-concentration export
        self.param_panel.btn_gr_compute.clicked.connect(self._on_compute_structure_functions)
        self.param_panel.btn_gr_export.clicked.connect(self._on_export_gr_csv)
        self.defect_plot.export_csv_requested.connect(self._on_export_defect_csv)

    # ── video ──────────────────────────────────────────────────────

    def _browse(self):
        start=self._settings.get_last_video() or str(Path.cwd())
        p,_=QFileDialog.getOpenFileName(self,"Open video",start,
                "Video files (*.mp4 *.avi *.mov *.mkv);;All (*)")
        if p: self._open(Path(p))

    def _open(self,path):
        self._vpath=path; self.setWindowTitle(f"Colloid Tracker {APP_VERSION} — {path.name}")
        self.vid_pane.load_video(path)
        self.btn_track.setEnabled(True); self._settings.set_last_video(str(path))
        self.status_bar.showMessage(f"Loaded: {path}")

    # ── tracking ───────────────────────────────────────────────────

    def _start_tracking(self):
        if not self._vpath: QMessageBox.warning(self,"No video","Open a video first."); return
        if self._worker and self._worker.isRunning(): return
        params=self.param_panel.get(); params["video_path"]=str(self._vpath)
        self._worker=TrackingWorker(params,self)
        self._worker.progress.connect(self._on_prog)
        self._worker.frame_done.connect(self._on_fr_done)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self.defect_plot._b57=[]; self.defect_plot._blagb=[]
        self.diag_panel.clear_frame_counts()
        self.prog.setVisible(True); self.prog.setValue(0)
        self.btn_track.setEnabled(False); self.btn_stop.setEnabled(True); self.btn_export.setEnabled(False)
        # Stale structural-correlation results from a previous run — gate
        # both until this run's frame_results are available (Task 6).
        self.param_panel.btn_gr_compute.setEnabled(False)
        self.param_panel.btn_gr_export.setEnabled(False)
        self._track_start_t = time.monotonic()
        self._progress_dlg = TrackingProgressDialog(self)
        self._progress_dlg.show()
        self._worker.start(); self.status_bar.showMessage("Tracking…")

    def _stop_tracking(self):
        if self._worker: self._worker.abort()
        self.btn_stop.setEnabled(False); self.status_bar.showMessage("Stopping…")

    def _eta_str(self, pct: int) -> str:
        """Simple linear ETA from elapsed wall-clock time and current progress %."""
        if not self._track_start_t or pct <= 0 or pct >= 100:
            return ""
        elapsed = time.monotonic() - self._track_start_t
        remaining = elapsed / pct * (100 - pct)
        if remaining < 1:
            return ""
        m, s = divmod(int(remaining + 0.5), 60)
        return f"  (approx {m}m {s}s remaining)" if m else f"  (approx {s}s remaining)"

    def _on_prog(self,pct,msg):
        eta = self._eta_str(pct)
        self.prog.setValue(pct); self.prog.setFormat(f"{pct}%  {msg}{eta}"); self.status_bar.showMessage(msg+eta)
        if self._progress_dlg: self._progress_dlg.set_progress(pct, msg+eta)

    def _on_preview_feats(self, feats, frame_bgr):
        self.diag_panel.update_detections(feats, frame_bgr, self.param_panel.get())

    def _on_diag_apply(self, d: dict):
        """Apply auto-tuned parameter values from DiagnosticsPanel."""
        p = self.param_panel
        mapping = {
            "minmass":     (p.sp_mass,  lambda v: int(v)),
            "min_len":     (p.sp_mlen,  lambda v: int(v)),
            "max_step_um": (p.sp_step,  lambda v: float(v)),
        }
        for key, val in d.items():
            if key in mapping:
                widget, conv = mapping[key]
                widget.blockSignals(True); widget.setValue(conv(val)); widget.blockSignals(False)
        self._on_params()
        self.status_bar.showMessage("Auto-tuned: " + ", ".join(f"{k}={v}" for k,v in d.items()))

    # ── structural correlations (g(r) / g6(r)) ───────────────────────

    def _on_compute_structure_functions(self):
        if not self._data or not self._data.frame_results:
            QMessageBox.information(self, "No data", "Run Track & Analyse first.")
            return
        p = self.param_panel.get()
        mode = p.get("gr_frame_range_mode", "current")
        r_max_um = float(p.get("gr_rmax_um", 10.0))
        dr_um = float(p.get("gr_dr_um", 0.1))
        exclude_boundary = bool(p.get("gr_exclude_boundary", False))
        px_um = float(p.get("px_um", 0.11))

        if mode == "current":
            fr = self.vid_pane._fr
            res = self._data.frame_results.get(fr)
            if res is None or len(res.pts_px) == 0:
                QMessageBox.information(self, "No data",
                    f"No structural results for frame {fr}. Try a different frame.")
                return
            pts_um = res.pts_px * px_um
            r_centers, g_r, g6_r, g6_over_g_r = _pair_correlations(
                pts_um, res.psi6, r_max_um, dr_um, exclude_boundary=exclude_boundary)
            self.diag_panel.update_pair_correlations(r_centers, g_r, g6_r, g6_over_g_r)
            self.param_panel.btn_gr_export.setEnabled(True)
            self.status_bar.showMessage(f"g(r)/g6(r) computed for frame {fr}.")
            return

        # "full" / "custom" — background thread (can be slow on long videos)
        if self._sf_worker and self._sf_worker.isRunning():
            return
        if mode == "custom":
            start_fr = int(p.get("gr_start_fr", 0))
            end_fr   = int(p.get("gr_end_fr", 0))
            frame_range = range(min(start_fr, end_fr), max(start_fr, end_fr) + 1)
        else:
            frames = sorted(self._data.frame_results.keys())
            frame_range = range(frames[0], frames[-1] + 1) if frames else range(0)

        self.param_panel.btn_gr_compute.setEnabled(False)
        self.param_panel.btn_gr_compute.setText("Computing…")
        self.status_bar.showMessage("Computing structure functions…")
        self._sf_worker = StructureFunctionWorker(
            self._data.frame_results, frame_range, px_um,
            r_max_um, dr_um, exclude_boundary, self)
        self._sf_worker.progress.connect(self._on_sf_progress)
        self._sf_worker.finished_ok.connect(self._on_sf_finished)
        self._sf_worker.error.connect(self._on_sf_error)
        self._sf_worker.start()

    def _on_sf_progress(self, pct: int, msg: str):
        self.status_bar.showMessage(f"{msg} ({pct}%)")

    def _on_sf_finished(self, r_centers, g_r, g6_r, g6_over_g_r):
        self.diag_panel.update_pair_correlations(r_centers, g_r, g6_r, g6_over_g_r)
        self.param_panel.btn_gr_compute.setEnabled(True)
        self.param_panel.btn_gr_compute.setText("Compute structure functions")
        self.param_panel.btn_gr_export.setEnabled(True)
        self.status_bar.showMessage("g(r)/g6(r) computed.")

    def _on_sf_error(self, msg: str):
        self.param_panel.btn_gr_compute.setEnabled(True)
        self.param_panel.btn_gr_compute.setText("Compute structure functions")
        QMessageBox.warning(self, "Structure function error", msg[:800])
        self.status_bar.showMessage("Structure function computation failed.")

    def _on_export_gr_csv(self):
        pc = self.diag_panel._pair_corr
        if pc is None:
            QMessageBox.information(self, "No data", "Compute structure functions first.")
            return
        r_centers, g_r, g6_r, g6_over_g_r = pc
        df = pd.DataFrame({"r_um": r_centers, "g_r": g_r, "g6_r": g6_r,
                            "g6_over_g_r": g6_over_g_r})
        saved_dir = self._settings.get_export_dir()
        default = str((Path(saved_dir) if saved_dir else Path.cwd()) / "pair_correlations.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Save g(r)/g6(r) CSV", default, "CSV (*.csv)")
        if not path: return
        self._settings.set_export_dir(str(Path(path).parent)); self._settings.save()
        df.to_csv(path, index=False)
        self.status_bar.showMessage(f"Saved g(r)/g6(r) → {path}")

    def _on_export_defect_csv(self):
        if not self._data:
            QMessageBox.information(self, "No data", "Run Track & Analyse first.")
            return
        df = _defect_concentration_series(self._data.frame_results)
        if df.empty:
            QMessageBox.information(self, "No data", "No structural results to export.")
            return
        saved_dir = self._settings.get_export_dir()
        default = str((Path(saved_dir) if saved_dir else Path.cwd()) / "defect_concentration.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Save defect-concentration CSV", default, "CSV (*.csv)")
        if not path: return
        self._settings.set_export_dir(str(Path(path).parent)); self._settings.save()
        df.to_csv(path, index=False)
        self.status_bar.showMessage(f"Saved defect concentration → {path}")

    def _on_fr_done(self,fr,res):
        self.defect_plot.add_point(fr,len(res.pairs57),len(res.boundaries))
        if self._data is None: self._data=AnalysisData()
        self._data.frame_results[fr]=res
        # Live preview is an explicit opt-in (Issue 3): when off, skip the
        # per-frame annotated-render — it's GUI-thread work that isn't needed
        # until the final result, and skipping it reduces overhead during
        # tracking. Use the params the worker actually started with, not the
        # live panel state, so a mid-run toggle can't race the worker thread.
        live_preview = bool(self._worker.params.get("live_preview", True)) if self._worker else True
        if live_preview and self.vid_pane._fr==fr: self.vid_pane._render(fr)
        self.diag_panel.update_frame_count(fr, len(res.pts_px))

    def _on_grain_size(self, diam: int):
        """Apply auto-detected diameter to the Detection parameter panel."""
        for sp, val in ((self.param_panel.sp_diam, diam),
                        (self.param_panel.sp_sep,  diam)):
            sp.blockSignals(True); sp.setValue(val); sp.blockSignals(False)
        self._on_params()
        self.status_bar.showMessage(f"Grain size auto-detected: diameter = {diam} px")

    def _on_done(self,data):
        self._track_start_t=None
        # Thread has emitted `finished`, so run() has returned — safe to
        # release the worker now (mirrors CameraPane._on_cam_stopped).
        if self._worker:
            self._worker.wait(500)
            self._worker.deleteLater()
            self._worker = None
        self._data=data
        self.vid_pane.set_analysis(data)
        self.log_table.populate(data.flag_events)
        defect_df = _defect_concentration_series(data.frame_results)
        self.defect_plot.set_series(data.n_pairs_series,data.n_lagb_series,defect_df)
        self.prog.setValue(100)
        self.btn_track.setEnabled(True); self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(True); self._a_export.setEnabled(True)
        # Structural correlations become available once frame_results exist.
        self.param_panel.btn_gr_compute.setEnabled(bool(data.frame_results))
        self.param_panel.maybe_autofill_gr_defaults()
        nev=len(data.flag_events)
        ntr=data.tracks.particle.nunique() if data.tracks is not None else 0
        done_msg = f"Done — {ntr} tracks  |  {nev} flag events"
        if self._progress_dlg: self._progress_dlg.mark_done(done_msg)
        # Feed track diagnostics panel
        px_um  = float(self.param_panel.get().get("px_um", 0.11))
        params = self.param_panel.get()
        self.diag_panel.update_tracks(data.tracks, px_um, params, feats=data.feats)
        # Optionally append auto-step tip to done message
        auto_step = _auto_max_step(data.tracks)
        if auto_step is not None:
            cur_step = float(self.param_panel.sp_step.value())
            if abs(auto_step - cur_step) / max(cur_step, 0.01) > 0.3:
                done_msg += (f"  |  Max step auto={auto_step} µm "
                             f"(current={cur_step}) — use Auto in Diagnostics")
        # Kick off consistency check in background
        if self._vpath:
            used_diam = int(self.param_panel.sp_diam.value())
            self.vid_pane.check_grain_consistency(str(self._vpath), used_diam)
        if nev>10:
            # Deferred via singleShot: a modal QMessageBox opened synchronously
            # inside this slot stalls the Qt queue for any late-arriving signal
            # from the just-finished worker thread (e.g. a straggling
            # frame_done) until the user dismisses it — looks like a freeze.
            QTimer.singleShot(0, lambda: QMessageBox.warning(self,"⚠  Detection quality warning",
                f"{nev} particles appeared/disappeared unexpectedly away from the edge.\n\n"
                f"Suggestions:\n"
                f"  • Lower Min mass (currently {self.param_panel.sp_mass.value():,})\n"
                f"  • Raise Gamma or Sharpen (dim-particle enhancement)\n"
                f"  • Increase Memory\n\n"
                f"Flag events are shown as rings in the annotated view and listed in Event Log."))
        self.status_bar.showMessage(done_msg)

    def _on_error(self,msg):
        self._track_start_t=None
        if self._worker:
            self._worker.wait(500)
            self._worker.deleteLater()
            self._worker = None
        self.prog.setVisible(False); self.btn_track.setEnabled(True); self.btn_stop.setEnabled(False)
        if self._progress_dlg: self._progress_dlg.mark_error(msg[:500])
        QMessageBox.critical(self,"Tracking error",msg[:1200])

    # ── layer / param ──────────────────────────────────────────────

    def _on_layer(self):
        l=self.layer_panel.cfg()
        self.vid_pane.set_layer(l)
        self.cam_pane.update_layer(l)

    def _on_params(self):
        p=self.param_panel.get()
        self.vid_pane.set_params(p)
        self.cam_pane.update_params(p)

    # ── export ─────────────────────────────────────────────────────

    def _export(self):
        if not self._data: return
        saved_dir = self._settings.get_export_dir()
        default = str((Path(saved_dir) if saved_dir else Path.cwd()) / "colloid_tracked.mp4")
        path,_=QFileDialog.getSaveFileName(self,"Save annotated video",default,"MP4 (*.mp4)")
        if not path: return
        self._settings.set_export_dir(str(Path(path).parent))
        self._settings.save()
        layer=self.layer_panel.cfg(); data=self._data
        cap=cv2.VideoCapture(str(data.video_path), cv2.CAP_FFMPEG)
        fps=cap.get(cv2.CAP_PROP_FPS) or (1./data.dt_eff)
        W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))
        ntot=data.end_fr-data.start_fr+1
        self.prog.setVisible(True); self.prog.setValue(0); self.btn_export.setEnabled(False)
        cap.set(cv2.CAP_PROP_POS_FRAMES, data.start_fr)   # single initial seek
        for idx,fr in enumerate(range(data.start_fr,data.end_fr+1)):
            ok,frame=cap.read()                            # sequential — no per-frame seek
            if not ok: break
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim==3 else frame
            u8   = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            bgr  = cv2.cvtColor(u8,cv2.COLOR_GRAY2BGR)
            if fr in data.frame_results:
                ann=annotate(bgr,data.frame_results[fr],layer,data.flag_events,fr); _legend(ann,layer)
            else: ann=bgr
            writer.write(ann)
            if idx%20==0:
                p2=int(100*idx/ntot); self.prog.setValue(p2)
                self.prog.setFormat(f"{p2}%  Exporting fr {fr}"); QApplication.processEvents()
        writer.release(); cap.release()
        self.prog.setValue(100); self.btn_export.setEnabled(True); self._a_export.setEnabled(True)
        QMessageBox.information(self,"Export done",f"Saved:\n{path}")

    # ── settings ───────────────────────────────────────────────────

    def _load_settings(self):
        """Apply saved settings to panels and restore window geometry."""
        self.param_panel.apply(self._settings.get_params())
        self.layer_panel.apply(self._settings.get_layer())
        # Sync cam_pane / vid_pane with the just-loaded widget values.
        # param_panel.apply() uses blockSignals so params_changed is not emitted;
        # without this explicit call, cam_pane._params stays at _DEFAULT_PARAMS
        # until the user first touches a widget — causing the display to ignore
        # any saved enhancement settings (and making "turn off" appear to have
        # no effect because the saved non-zero values were never applied).
        self._on_params()
        self._on_layer()
        # window geometry
        geo,state=self._settings.get_window()
        if geo: self.restoreGeometry(geo)
        if state: self.restoreState(state)
        # collapsible-section collapsed/expanded state, keyed by title
        CollapsibleGroupBox.apply_saved_states(self._settings.get_section_states())

    def _save_settings_now(self):
        self._settings.set_params(self.param_panel.get())
        self._settings.set_layer(self.layer_panel.cfg().to_dict())
        self._settings.save_window(self.saveGeometry(),self.saveState())
        self._settings.set_section_states(CollapsibleGroupBox.collect_states())
        self._settings.save()
        self.status_bar.showMessage(f"Settings saved → {SETTINGS_PATH}")

    def _save_preset(self):
        name,ok=QInputDialog.getText(self,"Save preset","Preset name:")
        if not ok or not name.strip(): return
        self._settings.save_preset(name.strip(),self.param_panel.get(),
                                   self.layer_panel.cfg().to_dict())
        self._settings.save(); self.status_bar.showMessage(f"Preset '{name}' saved.")

    def _load_preset(self):
        presets=self._settings.get_presets()
        if not presets: QMessageBox.information(self,"No presets","No saved presets yet."); return
        names=list(presets.keys())
        name,ok=QInputDialog.getItem(self,"Load preset","Select preset:",names,0,False)
        if not ok: return
        p=presets[name]
        self.param_panel.apply(p.get("params",{}))
        self.layer_panel.apply(p.get("layer",{}))
        self._on_layer(); self._on_params()
        self.status_bar.showMessage(f"Loaded preset '{name}'.")

    def _open_settings_file(self):
        if sys.platform=="win32": os.startfile(str(SETTINGS_PATH))
        elif sys.platform=="darwin": subprocess.run(["open",str(SETTINGS_PATH)])
        else: subprocess.run(["xdg-open",str(SETTINGS_PATH)])

    def _reset_layout(self):
        """Restore all docks to their default positions."""
        for dock in (self._dock_params,self._dock_layers,self._dock_plot,self._dock_log):
            dock.setFloating(False); dock.show()
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,   self._dock_params)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,   self._dock_layers)
        self.tabifyDockWidget(self._dock_params,self._dock_layers); self._dock_params.raise_()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._dock_plot)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._dock_log)
        self.tabifyDockWidget(self._dock_plot,self._dock_log);      self._dock_plot.raise_()

    # ── dep check ─────────────────────────────────────────────────

    def _check_deps(self):
        missing=[]
        if not PYPYLON_OK:          missing.append("pypylon")
        if not IS_FROZEN:
            try: import imageio_ffmpeg  # noqa
            except ImportError: missing.append("imageio-ffmpeg")
        if missing and not IS_FROZEN:
            bar=DepBar(missing,self)
            # insert into the central widget's layout (above tabs)
            self.centralWidget().layout().insertWidget(0,bar)

    # ── misc ───────────────────────────────────────────────────────

    def _show_help(self):
        HelpDialog(self).exec()

    def _about(self):
        QMessageBox.about(self,f"Colloid Tracker {APP_VERSION}",
            f"<b>Colloid Tracker {APP_VERSION}</b><br><br>"
            "Particle tracking and structural defect analysis.<br>"
            "Based on Owen Tower's PhD_Code/Colloid_Analysis.py<br>(explicit permission granted).<br><br>"
            "Detects: 5-fold / 7-fold coordination, 5-7 edge dislocations,<br>"
            "low-angle and high-angle grain boundaries.<br><br>"
            f"Settings: <tt>{SETTINGS_PATH}</tt>")

    def _pypylon_help(self):
        QMessageBox.information(self,"Install pypylon",
            "To use a live Basler camera:\n\n"
            "1. Download and install the Basler Pylon SDK:\n"
            "   https://www.baslerweb.com/en/downloads/software-downloads/\n\n"
            "2. Then install the Python bindings:\n"
            "   pip install pypylon\n\n"
            "3. Restart the application.\n\n"
            "USB 3.0 Vision cameras are auto-detected via the Scan button.")

    # ── window close ──────────────────────────────────────────────

    def closeEvent(self,event):
        self._save_settings_now()

        def _stop_wait(worker, ms, label, stop_method="stop"):
            """Stop a QThread worker and wait, warning (not blocking forever)
            if it doesn't finish in time — avoids silently pretending a
            thread/subprocess (e.g. an open FFmpeg pipe) shut down cleanly."""
            if worker is None or not worker.isRunning():
                return
            getattr(worker, stop_method)()
            if not worker.wait(ms):
                print(f"[closeEvent] {label} did not stop within {ms} ms")

        if self._worker and self._worker.isRunning():
            _stop_wait(self._worker, 3000, "TrackingWorker", stop_method="abort")
        if self._sf_worker and self._sf_worker.isRunning():
            _stop_wait(self._sf_worker, 3000, "StructureFunctionWorker", stop_method="abort")
        cam = getattr(self, "cam_pane", None)
        if cam is not None:
            # Recorder must be told to stop first (drains its buffer to the
            # output file / closes the FFmpeg pipe) before the camera worker
            # that feeds it is torn down.
            if getattr(cam, "_recorder", None) is not None:
                _stop_wait(cam._recorder, 5000, "RecorderWorker")
            if getattr(cam, "_tiff_recorder", None) is not None:
                _stop_wait(cam._tiff_recorder, 5000, "TiffSequenceRecorder")
            if getattr(cam, "_ana_worker", None) is not None:
                _stop_wait(cam._ana_worker, 1000, "AnalysisWorker")
            if getattr(cam, "_worker", None) is not None:
                _stop_wait(cam._worker, 2000, "CameraWorker")
            if cam._tl_worker and cam._tl_worker.isRunning():
                _stop_wait(cam._tl_worker, 5000, "TimelapseWorker")
            if cam._shutter.is_connected:
                cam._shutter.disconnect()
        event.accept()

    # ── dark theme ────────────────────────────────────────────────

    def _theme(self):
        self.setStyleSheet("""
        QMainWindow,QWidget,QScrollArea,QAbstractScrollArea
                                    {background:#1e1e1e;color:#ddd;}

        /* ── Group boxes ───────────────────────────── */
        QGroupBox                   {border:1px solid #3a3a3a;margin-top:8px;
                                     border-radius:5px;padding-top:4px;}
        QGroupBox::title            {color:#88aacc;subcontrol-origin:margin;
                                     left:8px;padding:0 4px;}

        /* ── Labels ────────────────────────────────── */
        QLabel                      {color:#c8c8c8;}

        /* ── Spin boxes ─────────────────────────────
           Explicit button styling prevents invisible dark arrows in dark theme */
        QSpinBox, QDoubleSpinBox    {background:#252525;color:#e0e0e0;
                                     border:1px solid #4a4a4a;border-radius:3px;
                                     padding-right:18px;}
        QSpinBox:focus, QDoubleSpinBox:focus
                                    {border-color:#6688cc;}
        QSpinBox::up-button, QDoubleSpinBox::up-button {
                                     subcontrol-origin:border;
                                     subcontrol-position:top right;
                                     width:17px;
                                     background:#323232;
                                     border-left:1px solid #4a4a4a;
                                     border-bottom:1px solid #4a4a4a;
                                     border-top-right-radius:3px;}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover
                                    {background:#424242;}
        QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed
                                    {background:#555;}
        QSpinBox::down-button, QDoubleSpinBox::down-button {
                                     subcontrol-origin:border;
                                     subcontrol-position:bottom right;
                                     width:17px;
                                     background:#323232;
                                     border-left:1px solid #4a4a4a;
                                     border-bottom-right-radius:3px;}
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover
                                    {background:#424242;}
        QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed
                                    {background:#555;}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow
                                    {width:6px;height:6px;
                                     border-left:3px solid transparent;
                                     border-right:3px solid transparent;
                                     border-bottom:5px solid #aaa;}
        QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled
                                    {border-bottom-color:#555;}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow
                                    {width:6px;height:6px;
                                     border-left:3px solid transparent;
                                     border-right:3px solid transparent;
                                     border-top:5px solid #aaa;}
        QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled
                                    {border-top-color:#555;}

        /* ── Combo boxes ────────────────────────────── */
        QComboBox                   {background:#252525;color:#e0e0e0;
                                     border:1px solid #4a4a4a;border-radius:3px;
                                     padding:2px 6px;}
        QComboBox:focus             {border-color:#6688cc;}
        QComboBox::drop-down        {width:18px;border-left:1px solid #4a4a4a;
                                     background:#323232;border-top-right-radius:3px;
                                     border-bottom-right-radius:3px;}
        QComboBox::down-arrow       {width:6px;height:6px;
                                     border-left:3px solid transparent;
                                     border-right:3px solid transparent;
                                     border-top:5px solid #aaa;}
        QComboBox QAbstractItemView {background:#252525;color:#e0e0e0;
                                     border:1px solid #555;
                                     selection-background-color:#3a5a7a;}

        /* ── Checkboxes ─────────────────────────────── */
        QCheckBox                   {color:#c8c8c8;spacing:6px;}
        QCheckBox::indicator        {width:14px;height:14px;border:1px solid #555;
                                     background:#252525;border-radius:2px;}
        QCheckBox::indicator:hover  {border-color:#88aacc;}
        QCheckBox::indicator:checked{background:#3a6a3a;border-color:#5aaa5a;}
        QCheckBox::indicator:checked:hover
                                    {background:#4a7a4a;border-color:#6abb6a;}

        /* ── Buttons ────────────────────────────────── */
        QPushButton                 {background:#2e2e2e;color:#ddd;
                                     border:1px solid #4a4a4a;
                                     padding:4px 10px;border-radius:4px;}
        QPushButton:hover           {background:#3a3a3a;border-color:#6688cc;}
        QPushButton:pressed         {background:#484848;}
        QPushButton:checked         {background:#2a4a6a;color:#cce4ff;
                                     border-color:#5588bb;}
        QPushButton:checked:hover   {background:#3a5a7a;border-color:#66aadd;}
        QPushButton:disabled        {color:#666;border-color:#333;background:#262626;}

        /* ── Toolbar ────────────────────────────────── */
        QToolBar                    {background:#222;border-bottom:1px solid #333;
                                     spacing:3px;}
        QToolBar QPushButton        {margin:2px;padding:3px 7px;}

        /* ── Dock widgets ───────────────────────────── */
        QDockWidget                 {color:#ccc;}
        QDockWidget::title          {background:#242424;padding:4px 8px;
                                     border-bottom:1px solid #383838;
                                     font-weight:bold;color:#aac;}

        /* ── Tabs ───────────────────────────────────── */
        QTabWidget::pane            {border:1px solid #383838;border-top:none;}
        QTabBar::tab                {background:#252525;color:#999;
                                     padding:5px 12px;border:1px solid #333;
                                     border-bottom:none;margin-right:1px;
                                     border-top-left-radius:3px;
                                     border-top-right-radius:3px;}
        QTabBar::tab:selected       {background:#2e2e2e;color:#ddd;
                                     border-color:#4a4a4a;}
        QTabBar::tab:hover:!selected{background:#2a2a2a;color:#bbb;}

        /* ── Progress bar ───────────────────────────── */
        QProgressBar                {background:#232323;border:1px solid #444;
                                     color:#ccc;text-align:center;
                                     max-height:16px;border-radius:3px;}
        QProgressBar::chunk         {background:#2a6a2a;border-radius:2px;}

        /* ── Table / tree ───────────────────────────── */
        QTableWidget                {background:#212121;gridline-color:#2e2e2e;}
        QHeaderView::section        {background:#252525;color:#999;
                                     border:none;border-right:1px solid #333;
                                     padding:3px 6px;}

        /* ── Slider ─────────────────────────────────── */
        QSlider::groove:horizontal  {background:#2a2a2a;height:4px;border-radius:2px;}
        QSlider::handle:horizontal  {background:#5577bb;width:12px;height:12px;
                                     margin:-4px 0;border-radius:6px;}
        QSlider::handle:horizontal:hover
                                    {background:#7799cc;}

        /* ── Scrollbars ─────────────────────────────── */
        QScrollBar:vertical         {background:#1e1e1e;width:8px;margin:0;}
        QScrollBar::handle:vertical {background:#404040;border-radius:4px;min-height:20px;}
        QScrollBar::handle:vertical:hover
                                    {background:#555;}
        QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical
                                    {height:0;}
        QScrollBar:horizontal       {background:#1e1e1e;height:8px;margin:0;}
        QScrollBar::handle:horizontal
                                    {background:#404040;border-radius:4px;min-width:20px;}
        QScrollBar::handle:horizontal:hover
                                    {background:#555;}
        QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal
                                    {width:0;}

        /* ── Menus ──────────────────────────────────── */
        QMenuBar                    {background:#202020;color:#ccc;
                                     border-bottom:1px solid #333;}
        QMenuBar::item              {padding:4px 10px;}
        QMenuBar::item:selected     {background:#2e2e2e;color:#eee;}
        QMenu                       {background:#242424;color:#ccc;
                                     border:1px solid #444;padding:2px;}
        QMenu::item                 {padding:5px 24px 5px 14px;}
        QMenu::item:selected        {background:#2a4a6a;color:#eee;}
        QMenu::separator            {height:1px;background:#383838;margin:3px 6px;}
        """)


# ════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════

def main():
    # Write unhandled Python exceptions to a crash log beside the script,
    # so slot-level crashes (which Qt swallows from stderr) are captured.
    _log_path = Path(__file__).parent / "crash_log.txt"
    _orig_hook = sys.excepthook
    def _crash_hook(exc_type, exc_val, exc_tb):
        import datetime as _dt
        with open(_log_path, "a", encoding="utf-8") as _f:
            _f.write(f"\n=== {_dt.datetime.now()} ===\n")
            _tb.print_exception(exc_type, exc_val, exc_tb, file=_f)
        _orig_hook(exc_type, exc_val, exc_tb)
    sys.excepthook = _crash_hook

    # High-DPI support
    if hasattr(Qt.ApplicationAttribute, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    app=QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    video=sys.argv[1] if len(sys.argv)>1 else None
    win=MainWindow(init_video=video)
    win.show()
    _start_nvenc_probe()   # deferred — see _start_nvenc_probe() docstring
    sys.exit(app.exec())


if __name__=="__main__":
    main()
