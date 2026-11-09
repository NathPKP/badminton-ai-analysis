#!/usr/bin/env sh
# One-command install on Linux / macOS: Python 3.12 virtual environment, packages, models, check.
# Run from the project folder:  sh setup.sh      (PYTHON=python3.12 by default)
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3.12}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "$PY not found: install Python 3.12, or run PYTHON=/path/to/python3.12 sh setup.sh"
    exit 1
fi
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_models.py
command -v ffmpeg >/dev/null 2>&1 || echo "ffmpeg is not on the PATH: install it (apt install ffmpeg / brew install ffmpeg)"
echo
.venv/bin/python run.py doctor
