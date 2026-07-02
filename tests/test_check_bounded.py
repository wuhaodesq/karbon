"""Tests for the static bounded-axiom checker."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKER = REPO_ROOT / "scripts" / "ci" / "check_bounded.py"


def _run_checker(root: Path) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(root)],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout + result.stderr


def test_checker_passes_on_project_src():
    """The current codebase must be clean."""
    code, out = _run_checker(REPO_ROOT / "src")
    assert code == 0, f"check_bounded found issues:\n{out}"


def test_checker_flags_bare_deque(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        textwrap.dedent(
            """
            from collections import deque

            def make():
                q = deque()
                return q
            """
        ).strip(),
        encoding="utf-8",
    )
    code, out = _run_checker(tmp_path)
    assert code == 1
    assert "unbounded-deque" in out


def test_checker_allows_maxlen_deque(tmp_path):
    good = tmp_path / "good.py"
    good.write_text(
        textwrap.dedent(
            """
            from collections import deque

            def make():
                return deque(maxlen=32)
            """
        ).strip(),
        encoding="utf-8",
    )
    code, out = _run_checker(tmp_path)
    assert code == 0, out


def test_checker_flags_direct_cuda(tmp_path):
    bad = tmp_path / "bad_cuda.py"
    bad.write_text(
        textwrap.dedent(
            """
            import torch

            def move(x):
                return x.cuda()
            """
        ).strip(),
        encoding="utf-8",
    )
    code, out = _run_checker(tmp_path)
    assert code == 1
    assert "no-direct-cuda" in out


def test_checker_allows_torch_cuda_module_access(tmp_path):
    """`torch.cuda.is_available()` is a module namespace, not a method — should not trigger."""
    good = tmp_path / "good_cuda.py"
    good.write_text(
        textwrap.dedent(
            """
            import torch

            def probe():
                return torch.cuda.is_available()
            """
        ).strip(),
        encoding="utf-8",
    )
    code, out = _run_checker(tmp_path)
    assert code == 0, out


def test_checker_respects_suppression(tmp_path):
    bad = tmp_path / "silenced.py"
    bad.write_text(
        textwrap.dedent(
            """
            from collections import deque

            def make():
                return deque()  # BOUNDS-OK: this is a demo
            """
        ).strip(),
        encoding="utf-8",
    )
    code, out = _run_checker(tmp_path)
    assert code == 0, out


def test_checker_flags_infinite_while(tmp_path):
    bad = tmp_path / "loop.py"
    bad.write_text(
        textwrap.dedent(
            """
            def run():
                while True:
                    x = 1
                    y = 2
            """
        ).strip(),
        encoding="utf-8",
    )
    code, out = _run_checker(tmp_path)
    assert code == 1
    assert "no-unbounded-loop" in out


def test_checker_allows_while_true_with_break(tmp_path):
    good = tmp_path / "loop2.py"
    good.write_text(
        textwrap.dedent(
            """
            def run(n):
                while True:
                    n -= 1
                    if n <= 0:
                        break
            """
        ).strip(),
        encoding="utf-8",
    )
    code, out = _run_checker(tmp_path)
    assert code == 0, out
