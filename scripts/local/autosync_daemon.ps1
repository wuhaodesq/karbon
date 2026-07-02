<#
.SYNOPSIS
    Autosync daemon (local Windows edition).
    本地 Windows 定时同步守护，把训练产物推到 GitHub / TOS 镜像。

.DESCRIPTION
    Windows PowerShell equivalent of scripts/cloud/autosync_daemon.sh.
    Recommended for the laptop when developing locally; the cloud version
    is a bash script for Linux cloud/home rigs.

.PARAMETER Interval
    Seconds between sync cycles. Default 3600 (1h).

.PARAMETER Stage
    Stage number for commit-message tag.

.PARAMETER NoGit
    Skip git sync.

.EXAMPLE
    .\scripts\local\autosync_daemon.ps1
    .\scripts\local\autosync_daemon.ps1 -Interval 1800 -Stage 0

.NOTES
    Runs indefinitely. Press Ctrl-C to stop.
    All actions best-effort; failures logged but never crash the daemon.
#>

[CmdletBinding()]
param(
    [int]$Interval = 3600,
    [string]$Stage = "",
    [string]$Branch = "main",
    [switch]$NoGit
)

$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$LogDir = Join-Path $ProjectRoot "logs\autosync"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$today = Get-Date -Format "yyyyMMdd"
$LogFile = Join-Path $LogDir "autosync_$today.log"

function Write-SyncLog {
    param([string]$Message)
    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $line = "[$ts] $Message"
    Write-Host $line
    Add-Content -LiteralPath $LogFile -Value $line
}

Write-SyncLog "==> devagi autosync daemon (Windows) starting"
Write-SyncLog "    project_root: $ProjectRoot"
Write-SyncLog "    interval:     ${Interval}s"
Write-SyncLog "    stage:        $Stage"
Write-SyncLog "    branch:       $Branch"

Push-Location $ProjectRoot

function Sync-Git {
    if ($NoGit) { return }

    if (-not (Test-Path (Join-Path $ProjectRoot ".git"))) {
        Write-SyncLog "  git: skip (not a git repo)"
        return
    }

    # Stage small text artefacts only
    git add -A docs/ configs/ CHANGELOG.md 2>&1 | Out-Null

    # If nothing staged, exit
    $staged = git diff --cached --name-only 2>&1
    if (-not $staged) {
        Write-SyncLog "  git: nothing to commit"
        return
    }

    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $subject = "autosync: stage $Stage @ $ts"
    git commit -m $subject 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-SyncLog "  git: commit failed"
        return
    }

    git push origin $Branch 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-SyncLog "  git: pushed to origin/$Branch"
    } else {
        Write-SyncLog "  git: push failed (will retry next cycle)"
    }
}

$cycle = 0
try {
    while ($true) {
        $cycle++
        Write-SyncLog "-- cycle $cycle --"

        try {
            Sync-Git
        } catch {
            Write-SyncLog "  git: exception: $_"
        }

        Write-SyncLog "  sleeping ${Interval}s..."
        Start-Sleep -Seconds $Interval
    }
} finally {
    Pop-Location
    Write-SyncLog "==> daemon stopped"
}
