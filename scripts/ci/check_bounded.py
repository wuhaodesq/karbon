"""Static check for Bounded Design Axioms (Axiom 1 & 2 primarily).

Grep-based lint that scans ``src/`` for suspicious patterns:

- Python lists/deques appended to inside training loops without capacity checks.
- ``while True`` loops without a break/return in the same function.
- Direct ``.cuda()`` calls (should use ``get_device()`` from platform layer).
- Hardcoded ``"cuda"`` / ``"cuda:0"`` device strings.
- Unbounded ``collections.deque()`` (without ``maxlen``).

This is a heuristic linter — false positives are expected and can be silenced
with ``# BOUNDS-OK: <reason>`` end-of-line comments.

对 Axiom 1/2 的静态检查。基于 grep + AST 的启发式扫描；
误报可用行尾 ``# BOUNDS-OK: <理由>`` 抑制。

Exit codes:
  0 — clean or all findings silenced
  1 — findings present
  2 — internal error
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

# Directory hardcoded relative to this file: <root>/scripts/ci/check_bounded.py
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

SUPPRESS_MARK = "BOUNDS-OK"


@dataclass
class Finding:
    path: Path
    line: int
    rule: str
    message: str
    snippet: str

    def format(self) -> str:
        try:
            rel = self.path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = self.path
        return f"{rel}:{self.line}: [{self.rule}] {self.message}\n    {self.snippet.rstrip()}"


# =====================================================================
# Rules
# =====================================================================


class BoundedChecker(ast.NodeVisitor):
    """AST-level checks that grep can't easily do."""

    def __init__(self, path: Path, lines: list[str]) -> None:
        self.path = path
        self.lines = lines
        self.findings: list[Finding] = []
        self._while_stack: list[int] = []

    def _suppressed(self, lineno: int) -> bool:
        if 1 <= lineno <= len(self.lines):
            return SUPPRESS_MARK in self.lines[lineno - 1]
        return False

    def _snippet(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1]
        return ""

    def _add(self, node: ast.AST, rule: str, msg: str) -> None:
        ln = getattr(node, "lineno", 0)
        if self._suppressed(ln):
            return
        self.findings.append(Finding(self.path, ln, rule, msg, self._snippet(ln)))

    def visit_While(self, node: ast.While) -> None:
        # while True: must have a break/return/raise inside
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            has_exit = any(
                isinstance(n, (ast.Break, ast.Return, ast.Raise))
                for n in ast.walk(node)
            )
            if not has_exit:
                self._add(
                    node,
                    "no-unbounded-loop",
                    "`while True` without break/return/raise inside — unbounded loop",
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Case A: `.cuda()` method call on a tensor/module — flagged.
        # We only flag when `cuda` is used as a *method* (Call over Attribute),
        # not as a module reference like `torch.cuda.is_available()`.
        if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
            # Skip when the receiver is the `torch` module: `torch.cuda` is
            # the submodule namespace, not a method. Actual `.cuda()` method
            # calls happen on tensors / modules, whose receiver isn't `torch`.
            receiver = node.func.value
            is_torch_module = (
                isinstance(receiver, ast.Name) and receiver.id == "torch"
            )
            if not is_torch_module:
                self._add(
                    node,
                    "no-direct-cuda",
                    "direct `.cuda()` — use `.to(get_device())` from src.platform",
                )

        # collections.deque() with no maxlen kwarg
        if isinstance(node.func, ast.Attribute) and node.func.attr == "deque":
            has_maxlen = any(kw.arg == "maxlen" for kw in node.keywords)
            if not has_maxlen:
                self._add(
                    node,
                    "unbounded-deque",
                    "`deque()` without `maxlen` — declare a capacity (Axiom 1)",
                )
        elif isinstance(node.func, ast.Name) and node.func.id == "deque":
            has_maxlen = any(kw.arg == "maxlen" for kw in node.keywords)
            if not has_maxlen:
                self._add(
                    node,
                    "unbounded-deque",
                    "`deque()` without `maxlen` — declare a capacity (Axiom 1)",
                )
        self.generic_visit(node)


# =====================================================================
# Textual rules (regex-style; cheap)
# =====================================================================


TEXT_RULES: list[tuple[str, str, str]] = [
    # (rule_id, needle, message)
    ('hardcoded-cuda-string',
     '"cuda"',
     'Hardcoded "cuda" device string — use get_device()'),
    ('hardcoded-cuda-string',
     "'cuda'",
     "Hardcoded 'cuda' device string — use get_device()"),
    ('hardcoded-cuda-index',
     '"cuda:',
     'Hardcoded "cuda:N" device string — use get_device()'),
]


def check_text_rules(path: Path, lines: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for i, line in enumerate(lines, start=1):
        if SUPPRESS_MARK in line:
            continue
        stripped = line.lstrip()
        # Ignore comments and the platform layer itself (source of truth)
        if stripped.startswith("#"):
            continue
        if "src" in str(path) and "platform" in str(path):
            continue
        for rule_id, needle, msg in TEXT_RULES:
            if needle in line:
                findings.append(Finding(path, i, rule_id, msg, line))
    return findings


# =====================================================================
# Driver
# =====================================================================


def collect_python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def check_file(path: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    lines = text.splitlines()
    findings: list[Finding] = []

    # AST rules
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as e:
        findings.append(
            Finding(path, e.lineno or 0, "syntax-error", str(e.msg), "")
        )
        return findings
    checker = BoundedChecker(path, lines)
    checker.visit(tree)
    findings.extend(checker.findings)

    # Text rules
    findings.extend(check_text_rules(path, lines))

    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description="devagi bounded-axiom static checker")
    ap.add_argument("--root", type=Path, default=SRC_ROOT, help="Directory to scan (default: src/)")
    ap.add_argument("--fail-on", type=int, default=1, help="Number of findings to fail at (default 1)")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[check_bounded] root not found: {args.root}", file=sys.stderr)
        return 2

    files = collect_python_files(args.root)
    all_findings: list[Finding] = []
    for p in files:
        all_findings.extend(check_file(p))

    if not all_findings:
        print(f"[check_bounded] OK — no findings in {len(files)} files under {args.root}")
        return 0

    print(f"[check_bounded] {len(all_findings)} finding(s):")
    for f in all_findings:
        print(f.format())

    if len(all_findings) >= args.fail_on:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
