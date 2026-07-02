# AGENTS.md — Instructions for Automated Agents

This file gives coding assistants (opencode, Claude Code, Codex, etc.) the
minimum operating protocol for working inside this repo.

本文档给 AI 编码助理提供在本仓库工作的最小协议。

---

## 1. Quick Facts / 快速事实

- **Project root**: `D:\karbon\` (Windows) / repo root on Linux.
- **Python**: 3.10 in a project-local `.venv`.
- **PyTorch**: `2.5.1+cpu` locally / `2.5.1+cu121` on cloud & home rigs.
- **Docs are bilingual**: Chinese-friendly first, English follows.
- **Everything must pass `make test` + `make check-bounds`** before commit.

## 2. Cardinal Rules / 铁律

1. **Read `PLAN.md` and `DESIGN_PRINCIPLES.md` first** — the six bounded
   axioms are non-negotiable.
2. **Never commit secrets, PATs, or `.env`.**
3. **Never commit `logs/`, `checkpoints/`, `data/` contents** (only `.gitkeep`).
4. **Never use `.cuda()` directly** — use `get_device()` from `src.platform`.
5. **Never create unbounded data structures on GPU** — declare a capacity.
6. **Never commit without running:** `make test && make check-bounds`.

## 3. Where to look / 定位速查

- Big picture / plan → `PLAN.md`
- Hardware phases → `HARDWARE_TOPOLOGY.md`
- How to move project → `MIGRATION.md`
- Stage-by-stage plan → `ROADMAP.md`
- Terminology (中英对照) → `GLOSSARY.md`
- Bilingual reports → `docs/stage*.md`
- Six axioms enforced → `DESIGN_PRINCIPLES.md`
- Static enforcement of axioms → `scripts/ci/check_bounded.py`

## 4. Adding a new module / 添加新模块的步骤

1. Add source under `src/<subpkg>/<mod>.py`.
2. Add `__all__` entries in the subpkg's `__init__.py`.
3. Add tests under `tests/test_<mod>.py` (pytest style).
4. If bounded storage: expose `.capacity` and `__len__` so `HealthChecker` picks it up.
5. Run:
   ```bash
   make test
   make check-bounds
   ```
6. Update `CHANGELOG.md` under the current `[Unreleased]` section.
7. If adds a public concept: add a bilingual entry to `GLOSSARY.md`.

## 5. Suppressing static checks / 抑制静态检查

If `scripts/ci/check_bounded.py` flags something you know is safe, add an
end-of-line comment:

```python
q = deque(maxlen=32)   # BOUNDS-OK: capacity declared inline
```

The token `BOUNDS-OK` silences that specific line. Do not silence more than
you must; and always justify with a comment.

## 6. Preset-aware code / 三档预设编码原则

Any new module that consumes budget-sensitive resources must:

- Read its limits from `configs/_presets/*.yaml` via `load_config`.
- Support all three presets: `local_smoke`, `cloud_24g`, `home_64g`.
- Not hardcode batch sizes, sequence lengths, or capacity numbers.

## 7. Git conventions / Git 惯例

- Commits: imperative, describe *why* not *what*.
- Tag Stage exits: `vMAJOR.MINOR.PATCH-stageN[-variant]`
  (e.g., `v0.2.0-stage2-pytorch`).
- Never force-push `main`.
- Never amend after push unless coordinating.
- Never commit hooks; instead, extend `scripts/ci/`.

## 8. Cross-platform code hygiene / 跨平台代码规范

- Use `pathlib.Path` everywhere. Never write string paths with `\` or `/`.
- Prefer environment variables (`DEVAGI_DATA_DIR`, `DEVAGI_LOGS_DIR`,
  `DEVAGI_CKPT_DIR`) over hardcoded absolute paths.
- Bash scripts must have `#!/usr/bin/env bash` and LF line endings.
- PowerShell scripts live under `scripts/local/*.ps1`. Bash under
  `scripts/cloud/*.sh` and `scripts/home/*.sh`.

## 9. When touching the training loop

The main training entry is `src/train.py`. It is intentionally minimal —
new features layer *on top* via composable modules. Do not silently blow up
the loop's shape; keep changes additive so that:

- Stage-0 preset (`local_smoke`) still finishes in <5 min on CPU.
- Existing checkpoints load with `--resume` unchanged.

## 10. When you are unsure / 不确定时

- **Grep first**: this repo has explicit `GLOSSARY.md`, docstrings, and a
  static checker. Prefer reading over guessing.
- **Do not invent structure**: if a concept doesn't fit an existing folder,
  ask before creating a new top-level package.
- **Do not disable failing tests**: fix them or open a follow-up TODO in
  `CHANGELOG.md`.

## 11. What "done" means / 完成标准

For any code change:
- [ ] Unit tests added / updated and passing.
- [ ] `make check-bounds` clean.
- [ ] `CHANGELOG.md` updated.
- [ ] Docstrings + bilingual note where relevant.
- [ ] No secrets, no logs, no bulk artifacts committed.

For a Stage exit:
- [ ] Everything above, plus
- [ ] `docs/stageN_report.md` filled in.
- [ ] Git tag pushed.
- [ ] Longevity test (where applicable) recorded.
