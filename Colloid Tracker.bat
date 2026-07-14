@echo off
REM Always launches colloid_app.py with the miniforge3 interpreter, which is
REM the only environment on this machine with numba, cupy, and a CUDA-enabled
REM OpenCV build all present and working. Double-click this file to run the
REM app with full acceleration, instead of relying on whatever "python" or a
REM file association happens to resolve to.
"C:\Users\Ling_\miniforge3\python.exe" "%~dp0colloid_app.py"
pause
