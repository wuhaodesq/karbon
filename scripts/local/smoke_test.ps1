<#
.SYNOPSIS
    5-minute local smoke test for devagi Stage 0.
    devagi Stage 0 五分钟本地冒烟测试。

.DESCRIPTION
    1. Activates .venv
    2. Runs pytest (unit tests)
    3. Runs `python -m src.train --stage 0 --preset local_smoke --smoke-only`
    Both must succeed for Stage 0 exit criterion to hold.

.EXAMPLE
    .\scripts\local\smoke_test.ps1
#>

[CmdletBinding()]
param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"

Write-Host "==> devagi Stage 0 · smoke test" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"

if (-not (Test-Path $VenvActivate)) {
    Write-Host "ERROR: venv not found at $VenvActivate" -ForegroundColor Red
    Write-Host "Run .\scripts\local\setup_env.ps1 first." -ForegroundColor Yellow
    exit 1
}

. $VenvActivate

Push-Location $ProjectRoot
try {
    if (-not $SkipTests) {
        Write-Host "==> Running unit tests (pytest)..." -ForegroundColor Cyan
        python -m pytest -x tests/
        if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
    }

    Write-Host ""
    Write-Host "==> Running Stage 0 baseline (smoke-only)..." -ForegroundColor Cyan
    python -m src.train --stage 0 --preset local_smoke --smoke-only
    if ($LASTEXITCODE -ne 0) { throw "smoke training failed" }

    Write-Host ""
    Write-Host "==> Smoke test PASSED." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Inspect logs under logs/stage0/<run-id>/"
    Write-Host "  2. Initialize git and tag v0.0.0-stage0-local:"
    Write-Host "       git init"
    Write-Host "       git add ."
    Write-Host "       git commit -m 'Stage 0 local skeleton'"
    Write-Host "       git tag v0.0.0-stage0-local"
    Write-Host "  3. Create a private remote and push."
}
finally {
    Pop-Location
}
