# Contributing / 贡献指南

Thank you for your interest. This is a research-scale project with strong
architectural discipline. Please read `PLAN.md`, `DESIGN_PRINCIPLES.md`, and
`ROADMAP.md` before opening a change.

感谢参与。本项目是研究性质的，架构纪律很硬。提交任何改动之前请先读
`PLAN.md`、`DESIGN_PRINCIPLES.md`、`ROADMAP.md`。

---

## 1. Environment setup / 环境安装

```powershell
# Windows / local CPU
.\scripts\local\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux cloud (Phase 2)
bash scripts/cloud/setup_env.sh
source .venv/bin/activate

# Linux home 64G (Phase 3)
bash scripts/home/setup_env.sh
source .venv/bin/activate
```

## 2. Pre-commit checks / 提交前必跑

```bash
make test           # pytest -x tests/
make check-bounds   # static enforcement of the six axioms
make lint           # optional ruff check
```

All three must be green. Do not skip.

## 3. Branch & commit style / 分支与提交

- Base branch: `main` (always deployable).
- Feature branches: `feat/<short-slug>`.
- Fix branches: `fix/<short-slug>`.
- Stage branches: `stage{N}-<feature>` (e.g. `stage2-triton-parity`).

Commit messages: imperative, ≤ 72 chars subject, blank line, then body.

```
Add BoundedReplayBuffer three-tier eviction

- GPU ring stores 10k transitions, demotes to CPU when full
- CPU ring caps at 100k, demotes to SSD shard archive
- SSD shard count bounded (Axiom 1)
- Tests cover all three eviction paths
```

## 4. Adding a new module / 新模块清单

1. `src/<subpkg>/<mod>.py`
2. Update `src/<subpkg>/__init__.py` exports
3. `tests/test_<mod>.py` with unit tests
4. If bounded storage: implement `capacity` property + `__len__`
5. `CHANGELOG.md` entry under `[Unreleased]`
6. If public: add bilingual entry to `GLOSSARY.md`
7. Confirm `make check-bounds` still clean

## 5. Six Bounded Design Axioms (recap) / 六大公理速览

| # | Axiom | Meaning |
|---|---|---|
| 1 | Zero unbounded GPU structs | Every device buffer declares capacity |
| 2 | Eviction before learning | Learn nothing new without deciding what to forget |
| 3 | Enforced hierarchical storage | GPU / CPU / SSD / archive four tiers |
| 4 | Periodic consolidation | Transient state distilled into slow weights |
| 5 | Fragmentation governance | Slope-based VRAM monitor + empty_cache |
| 6 | State is serializable | Every component supports checkpoint round-trip |

Full text: `DESIGN_PRINCIPLES.md`.

## 6. Testing conventions / 测试规范

- Use `pytest` (already configured in `pyproject.toml`).
- No network in tests.
- No fixed GPU assumption — mark CUDA-only tests with `@pytest.mark.cuda`.
- Prefer parametrization over duplicated tests.
- Every bounded module must include a test asserting `len(x) ≤ x.capacity`
  after synthetic overload.

## 7. Documentation style / 文档规范

- Chinese-friendly first. Add an English rephrase after.
- Module docstrings begin with a one-line summary, blank line, then details.
- Public functions: type-annotated signature + docstring with Args/Returns.
- No emojis in code unless requested.

## 8. What NOT to do / 禁忌

- ❌ `torch.tensor(...).cuda()` — use `.to(get_device())`.
- ❌ `list.append(...)` on a GPU-resident collection.
- ❌ `while True:` without an exit condition.
- ❌ Committing `.env`, `logs/*`, `checkpoints/*`, `data/*` contents.
- ❌ Silently rewriting checkpoint format (bump `CKPT_FORMAT_VERSION`).
- ❌ Adding a training-loop shortcut that breaks `local_smoke` finishing in <5 min.

## 9. Reporting bugs / 报告 bug

Please include:

1. Preset used (`local_smoke` / `cloud_24g` / `home_64g`)
2. Git SHA (`git rev-parse --short HEAD`)
3. Command line invocation
4. Relevant log excerpt (redact secrets)
5. If applicable: `logs/stage*/…/memory.csv`

## 10. Getting help / 求助

- Design questions → `DESIGN_PRINCIPLES.md` + `PLAN.md`.
- Term confusion → `GLOSSARY.md`.
- Migration issues → `MIGRATION.md`.
- Nothing else works → open an issue with the run info from §9.
