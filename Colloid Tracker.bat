@echo off
REM Launches Colloid Tracker with the miniforge3 interpreter, which is the
REM environment on this machine with numba, cupy, and a CUDA-enabled OpenCV
REM build all present and working.
REM
REM Normal double-click: starts the app via pythonw.exe, fully detached —
REM no console window lingers behind the GUI and nothing waits for a
REM keypress when you close the app.
REM
REM Troubleshooting: run   "Colloid Tracker.bat" debug   from a terminal
REM (or make a shortcut with the debug argument) to launch with a visible
REM console that stays open and shows any startup errors.
setlocal
set "APPDIR=%~dp0"
set "PYW=C:\Users\Ling_\miniforge3\pythonw.exe"
set "PY=C:\Users\Ling_\miniforge3\python.exe"

if not exist "%PY%" (
    echo Could not find the miniforge3 Python interpreter at:
    echo   %PY%
    echo Install miniforge3 or edit this .bat to point at your Python.
    pause
    exit /b 1
)

if /I "%~1"=="debug" goto :debug

if exist "%PYW%" (
    start "Colloid Tracker" /D "%APPDIR%" "%PYW%" "%APPDIR%colloid_app.py"
) else (
    start "Colloid Tracker" /D "%APPDIR%" "%PY%" "%APPDIR%colloid_app.py"
)
exit /b 0

:debug
cd /d "%APPDIR%"
"%PY%" "%APPDIR%colloid_app.py"
echo.
echo Colloid Tracker exited with code %ERRORLEVEL%.
pause
