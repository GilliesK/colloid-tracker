#!/bin/bash
# macOS launcher for Colloid Tracker (double-clickable from Finder).
# Counterpart of "Colloid Tracker.bat"; run `chmod +x "Colloid Tracker.command"` once.
cd "$(dirname "$0")"
exec python3 colloid_app.py "$@"
