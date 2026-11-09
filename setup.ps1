# One-command install on Windows: Python 3.12 virtual environment, packages, models, check.
# Run from the project folder:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host "Python launcher 'py' not found. Install Python 3.12: winget install Python.Python.3.12"
    exit 1
}
& py -3.12 -c "import sys" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.12 not found. Install it: winget install Python.Python.3.12  (then open a new terminal)"
    exit 1
}
if (-not (Test-Path .venv)) {
    Write-Host "Creating .venv (Python 3.12)..."
    & py -3.12 -m venv .venv
}
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed: see the message above."; exit 1 }
& $py download_models.py
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "ffmpeg is not on the PATH. Install it: winget install Gyan.FFmpeg  (then open a new terminal)"
}
Write-Host ""
& $py run.py doctor
