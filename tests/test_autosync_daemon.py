"""Tests for autosync daemon scripts (syntactic / structural only)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cloud_autosync_bash_is_valid():
    """The autosync bash daemon must have proper LF + shebang + termination trap."""
    p = REPO_ROOT / "scripts" / "cloud" / "autosync_daemon.sh"
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash"), "missing shebang"
    assert "\r" not in text, "has CRLF endings"
    # Must handle SIGTERM / SIGINT for graceful shutdown
    assert "trap" in text and ("INT" in text or "SIGINT" in text)
    assert "SIGTERM" in text or "TERM" in text
    # Must never bare-`set -e` (autosync should not crash on partial failure)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("set ")]
    for ln in lines:
        # allow `set -uo pipefail` or `set +e`, but NOT `set -e` (would kill the loop)
        assert "-e" not in ln or "+e" in ln, f"autosync must not use 'set -e': {ln}"


def test_cloud_autosync_uses_helpers_correctly():
    p = REPO_ROOT / "scripts" / "cloud" / "autosync_daemon.sh"
    text = p.read_text(encoding="utf-8")
    # Each helper must exist and be a function
    for fn in ("sync_git", "sync_rsync", "sync_export", "log"):
        assert f"{fn}()" in text or f"{fn} ()" in text, f"missing function: {fn}"
    # The loop must exist
    assert "while true" in text or "while :" in text


def test_local_autosync_ps1_is_valid():
    p = REPO_ROOT / "scripts" / "local" / "autosync_daemon.ps1"
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "\r" not in text, "has CRLF endings (should be LF per .gitattributes)"
    assert "[CmdletBinding()]" in text
    # Must have the main loop
    assert "while ($true)" in text


def test_no_daemon_uses_hardcoded_git_credentials():
    """Neither daemon may embed a PAT or username in URLs."""
    for p in [
        REPO_ROOT / "scripts" / "cloud" / "autosync_daemon.sh",
        REPO_ROOT / "scripts" / "local" / "autosync_daemon.ps1",
    ]:
        text = p.read_text(encoding="utf-8")
        assert "ghp_" not in text, f"leaked PAT in {p}"
        assert "@github.com" not in text, f"credential-in-url in {p}"
