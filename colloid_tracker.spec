# colloid_tracker.spec  —  PyInstaller spec for Windows one-file exe
# Run on Windows:   pyinstaller colloid_tracker.spec

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, collect_all

block_cipher = None

# ── Collect scipy + numpy Python files / extension modules ────────────────────
_sp_datas, _sp_bins, _sp_hidden = collect_all("scipy")
_np_datas, _np_bins, _np_hidden = collect_all("numpy")

EXTRA_BINS  = _sp_bins  + _np_bins
EXTRA_DATAS = _sp_datas + _np_datas

# ── Explicitly add numpy.libs/ and scipy.libs/ DLLs ──────────────────────────
# collect_all() does NOT reach these sibling directories; they hold the
# bundled OpenBLAS (libscipy_openblas64_*.dll) that _fblas depends on.
#
# Destination must mirror the installed layout: scipy/__init__.py does
#   os.add_dll_directory(os.path.join(os.path.dirname(__file__), '..', 'scipy.libs'))
# so in _MEIPASS the DLL must land at scipy.libs/ (not the bare root).
_site_pkgs = Path(sys.prefix) / "Lib" / "site-packages"
for _libs_dir in ("numpy.libs", "scipy.libs"):
    _libs_path = _site_pkgs / _libs_dir
    if _libs_path.is_dir():
        for _dll in _libs_path.glob("*.dll"):
            EXTRA_BINS.append((str(_dll), _libs_dir))   # preserve subdir name

# ── Exclude the conda BLAS wrappers that depend on the missing mkl_rt.dll ────
# These are conda-environment DLLs (libblas.dll, liblapack.dll, libcblas.dll)
# that PyInstaller picks up via DLL-dependency scanning of conda-built numpy.
# The pip-installed scipy/numpy uses its own bundled OpenBLAS (above) and does
# NOT need these wrappers, so excluding them silences the mkl_rt.dll warnings
# and prevents dead DLLs from bloating the bundle.
_CONDA_BIN = str(Path(sys.prefix) / "Library" / "bin")
EXCLUDE_BINARIES = [
    os.path.join(_CONDA_BIN, "libblas.dll"),
    os.path.join(_CONDA_BIN, "libcblas.dll"),
    os.path.join(_CONDA_BIN, "liblapack.dll"),
]

# ── Hidden imports ────────────────────────────────────────────────────────────
HIDDEN = [
    # scipy extras (beyond collect_all)
    "scipy._lib.array_api_compat.numpy.fft",
    "scipy.special._ufuncs",
    "scipy.spatial._ckdtree",
    "scipy.spatial._qhull",
    "scipy.sparse.csgraph._min_spanning_tree",
    "scipy.sparse.csgraph._tools",
    "scipy.sparse.csgraph._shortest_path",
    "scipy.sparse.csgraph._traversal",
    "scipy.sparse.csgraph._flow",
    "scipy.sparse.csgraph._matching",
    # sklearn
    "sklearn.utils._cython_blas",
    "sklearn.neighbors._partition_nodes",
    "sklearn.cluster._kmeans",
    "sklearn.cluster._dbscan_inner",
    "sklearn.tree._utils",
    "sklearn.utils._weight_vector",
    # trackpy
    "trackpy.linking.subnet",
    "trackpy.linking.utils",
    "trackpy.feature",
    "trackpy.refine",
    # pims
    "pims",
    "pims.ffmpeg_reader",
    "pims.moviepy_reader",
    # imageio
    "imageio",
    "imageio.plugins.ffmpeg",
    "imageio_ffmpeg",
    # matplotlib
    "matplotlib.backends.backend_qtagg",
    "matplotlib.figure",
    # PyQt6
    "PyQt6.QtWidgets",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.sip",
] + _sp_hidden + _np_hidden

a = Analysis(
    ["colloid_app.py"],
    pathex=["."],
    binaries=[b for b in EXTRA_BINS
              if b[0] not in EXCLUDE_BINARIES],
    datas=EXTRA_DATAS,
    hiddenimports=HIDDEN,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "wx", "gi", "PySide6", "PySide2", "PyQt5"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ColloidTracker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no terminal window
    icon=None,              # add "icon.ico" here if desired
    onefile=True,
)
