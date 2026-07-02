<#
.SYNOPSIS
    Local (Windows) environment setup for devagi Stage 0.
    本地 Windows 环境安装脚本（Stage 0）。

.DESCRIPTION
    Creates .venv at D:\karbon\.venv using the currently-active `python`
    (must be 3.10.x), then installs cpu.txt + dev.txt.

.EXAMPLE
    .\scripts\local\setup_env.ps1
#>

[CmdletBinding()]
param(
    [switch]$Force,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$VenvPath = Join-Path $ProjectRoot ".venv"

Write-Host "==> devagi Stage 0 · local environment setup" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host "Venv path:    $VenvPath"

# --- Verify Python ---
$pyver = & $PythonExe --version 2>&1
Write-Host "Detected Python: $pyver"
if ($pyver -notmatch "3\.10\.") {
    Write-Warning "Expected Python 3.10.x, got: $pyver"
    Write-Warning "You can proceed but the venv may fail Torch install."
    if (-not $Force) {
        Write-Host "Aborting. Re-run with -Force to override." -ForegroundColor Yellow
        exit 1
    }
}

# --- Create venv ---
if (Test-Path $VenvPath) {
    if ($Force) {
        Write-Host "Removing existing venv (Force)" -ForegroundColor Yellow
        Remove-Item -LiteralPath $VenvPath -Recurse -Force
    } else {
        Write-Host "Venv already exists. Use -Force to recreate. Skipping creation." -ForegroundColor Yellow
    }
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "==> Creating venv..." -ForegroundColor Cyan
    & $PythonExe -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}

# --- Activate ---
$activate = Join-Path $VenvPath "Scripts\Activate.ps1"
Write-Host "==> Activating venv..." -ForegroundColor Cyan
. $activate

# --- Upgrade pip ---
Write-Host "==> Upgrading pip / wheel / setuptools..." -ForegroundColor Cyan
python -m pip install --upgrade pip wheel setuptools

# --- Install deps ---
Write-Host "==> Installing base + cpu + dev requirements..." -ForegroundColor Cyan
pip install -r (Join-Path $ProjectRoot "requirements\base.txt") `
            -r (Join-Path $ProjectRoot "requirements\cpu.txt") `
            -r (Join-Path $ProjectRoot "requirements\dev.txt")

# --- Verify ---
Write-Host "==> Verifying install..." -ForegroundColor Cyan
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"
python -c "import minigrid, gymnasium; print('minigrid + gymnasium ok')"

Write-Host ""
Write-Host "==> Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Activate venv:   .\.venv\Scripts\Activate.ps1"
Write-Host "  2. Run smoke test:  .\scripts\local\smoke_test.ps1"
Write-Host "  3. Run unit tests:  pytest -x tests\"
