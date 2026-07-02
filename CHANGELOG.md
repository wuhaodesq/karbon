# Changelog

All notable changes to this project are documented here.
本项目所有值得记录的变更。

Format loosely follows [Keep a Changelog](https://keepachangelog.com/).
遵循 Keep a Changelog 惯例。

---

## [Unreleased]

### Added — Full-journey planning doc

- `FULL_JOURNEY.md` — Bilingual end-to-end Stage 0–6 timeline, cost matrix,
  hardware purchase guide, risk register, and copy-paste kickoff commands.
  Recommends Route C (hybrid: cloud vGPU/5090 for Stage 0–4, home 64G rig
  for Stage 5–6 perpetual).
- README: link to FULL_JOURNEY.md.

### Route summary (from FULL_JOURNEY.md)
- Route A · full cloud (vGPU):    ~85 days, ¥2900–3500
- Route B · full cloud (5090):    ~60 days, ¥3900–4550
- **Route C · hybrid (recommended): ~90 days, ¥1200–1600 + hardware (¥16k–60k)**

---

## [Unreleased]

### Added — Cloud-training operational scripts

- `scripts/preflight.py` — 10-step pre-training checklist (Python version, torch/CUDA, GPU inventory, Triton, disk, env vars, project imports, bounded check, preset load, 20-step smoke). Exits nonzero on any critical failure.
- `scripts/cloud/sync_to_git.sh` — pushes small stage artefacts (reports, figures, config snapshots, CHANGELOG) + optional git tag to remote. Explicitly documents the split: text → Git, binaries (`*.pt`, replay data) → TOS/rsync mirror.

### Guidance
- Reports (`docs/stage*_report.md`) and preset snapshots ARE synced to Git.
- Checkpoints, replay cold data, exports/ are NOT — they go to TOS or an rsync mirror.

---

## [Unreleased]

### Added — RTX 5090 / Blackwell support + platform-image (PyTorch 2.8 / CUDA 12.8) support

Target scenario: cloud VMs with pre-installed **PyTorch 2.8.0 / Python 3.12 /
Ubuntu 22.04 / CUDA 12.8** running on RTX 5090 (Blackwell / sm_120).

- `requirements/cuda128.txt` — new; `torch>=2.8,<2.9` + `triton>=3` on cu128 wheels.
  Ships sm_120 kernels required by RTX 5090.
- `configs/_presets/cloud_5090.yaml` — new preset:
  22 GB VRAM budget, batch=16, seq=96, 16 parallel envs, 6-layer × 384-dim model.
  Sits between `cloud_24g` and `home_64g`.
- `scripts/cloud/setup_env.sh`:
  * Auto-detects Python 3.10/3.11/3.12 (was 3.10-only).
  * Detects RTX 50-series → auto-uses `cuda128.txt`.
  * New `--skip-torch` flag to reuse pre-installed torch on platform images.
- `scripts/home/setup_env.sh`: same auto-detect + `--skip-torch` support.
- `pyproject.toml`: `requires-python = ">=3.10,<3.13"`; black targets py310/11/12.
- `HARDWARE_TOPOLOGY.md`: Phase-2 section now documents 5090 wheel selection.
- Preset test coverage extended to `cloud_5090` (211 tests total).

### Test coverage after this batch
- **211 passed** (was 208), 10 skipped, 0 failing.
- `check_bounded`: OK across 37 source files.

---

## [Unreleased]

### Added — HuggingFace-format export path (this session)

**Export tooling for TOS / HuggingFace Hub / ARK custom-model upload:**
- `scripts/export_hf.py` — converts `src.utils.ckpt.save_ckpt` payloads to HF
  layout: `config.json` + sharded `model.safetensors` (+ `model.safetensors.index.json`
  when >5 GB) + bilingual `README.md`.
- Supports architectures: `hybrid_backbone`, `rssm`, `rnd`, `ttt_linear`.
- Supports dtype cast: `float32` / `float16` / `bfloat16`.
- `scripts/build_demo_export.py` — one-shot generator for demo `exports/demo-hybrid-{fp32,fp16}/`
  directly uploadable to TOS.
- `requirements/base.txt`: adds `safetensors>=0.4`.
- `.gitignore`: ignores `exports/` (upload artefacts, not for Git).
- `.gitattributes`: marks `*.safetensors` as binary.
- `MIGRATION.md` §8: bilingual export + TOS upload guide.
- `tests/test_export_hf.py`: 8 unit tests covering flatten / shard / dtype cast
  / roundtrip / architecture whitelist.

### Test coverage after this batch
- **208 tests passing**, 10 skipped, 0 failing.
- `check_bounded`: OK across 37 source files.

---

## [Unreleased]

### Added — Full local pre-work batch (A–N)

**Models:**
- `src/models/ttt_mlp.py` — TTT-MLP with 2-layer inner MLP, analytic GELU derivative, mini-batch dual form.
- `src/models/world_model.py` — Dreamer-style RSSM: encoder / decoder / GRU / prior / posterior heads, bounded rollouts.

**Memory:**
- `src/memory/skill_library.py` — Bounded 3-tier LoRA-based skill library with LRU × usefulness × reward eviction and cosine-similarity merging.
- `src/memory/generative_replay.py` — Small MLP VAE for anti-forgetting rehearsal.

**Intrinsic / Curriculum / Continual:**
- `src/intrinsic/learning_progress.py` — Per-task ring buffer + LP metric, smoothing, priority normalization.
- `src/curriculum/auto_curriculum.py` — LP-driven task sampling with FIFO eviction and ε-exploration.
- `src/continual/online_ewc.py` — Single-Fisher exponentially-decayed EWC with penalty and gradient integration.
- `src/continual/consolidation.py` — Periodic sleep-consolidation loop with warmup gate and disabled-task support.

**Envs:**
- `src/envs/crafter_wrapper.py` — Stage-3 Crafter wrapper with lazy import, auto-reset, bounded episode-return history.

**Utils:**
- `src/utils/config_schema.py` — Dataclass-based config validation catching typos, wrong types, out-of-range values.

**Scripts:**
- `scripts/home/setup_env.sh` — Phase-3 home 64G rig setup with VRAM ≥40 GB sanity check.
- `scripts/home/run_perpetual.sh` — tmux-wrapped perpetual training launcher.
- `scripts/home/health_daemon.sh` — External CSV-logging health monitor with VRAM slope alarm.

**Docs & governance:**
- `AGENTS.md` — Operating protocol for automated coding assistants.
- `CONTRIBUTING.md` — Human contributor guide.
- `notebooks/memory_profiling.ipynb` — MemoryWatcher CSV visualization.
- `notebooks/skill_visualization.ipynb` — Skill library usage / weight-heatmap / similarity analysis.
- `notebooks/ttt_state_inspection.ipynb` — TTT-Linear inner-state per-segment norm plot.

**Tests:**
- `tests/test_ttt_mlp.py` (7)
- `tests/test_skill_library.py` (13)
- `tests/test_world_model.py` (10)
- `tests/test_learning_progress.py` (13)
- `tests/test_auto_curriculum.py` (10)
- `tests/test_online_ewc.py` (11)
- `tests/test_consolidation.py` (9)
- `tests/test_generative_replay.py` (9)
- `tests/test_config_schema.py` (16)
- `tests/test_integration_stage0.py` (2) — End-to-end wire-up test using a DummyEnv.
- `tests/test_crafter_wrapper.py` (7 + 1 skipped) — Fake-crafter-based mechanics tests.

### Test coverage after this batch
- **200 tests passing**, 10 skipped (Triton parity + Crafter install), 0 failing.
- `check_bounded`: OK across 37 source files.

### Stage readiness after this batch
- Stage 1 ready: `RND` + `BoundedReplayBuffer` complete.
- Stage 2 ready: `TTT-Linear` + `TTT-MLP` + `SlidingWindowAttention` + `HybridBackbone` complete.
- Stage 3 ready: `RSSM` world model + `CrafterWrapper` complete.
- Stage 4 ready: `BoundedSkillLibrary` complete.
- Stage 5 ready: `LearningProgressTracker` + `AutoCurriculum` complete.
- Stage 6 ready: `OnlineEWC` + `SleepConsolidationLoop` + `GenerativeReplayVAE` complete.
- Static enforcement of six axioms operational; `make check-bounds` integrated.

---



## [v0.0.0-stage0-local] — planned

### Added
- Full project skeleton at `D:\karbon\`.
- Documents: `PLAN.md`, `README.md`, `HARDWARE_TOPOLOGY.md`, `MIGRATION.md`, `DESIGN_PRINCIPLES.md`, `GLOSSARY.md`, `ROADMAP.md`.
- Requirements split: `base.txt`, `cpu.txt`, `cuda121.txt`, `dev.txt`.
- Platform abstraction: `src/platform/device.py`, `paths.py`, `memory_probe.py`.
- Monitoring: `src/monitoring/memory_watcher.py`, `longevity_test.py`, `health_check.py`.
- MiniGrid wrapper: `src/envs/minigrid_wrapper.py`.
- Minimal PPO baseline: `src/train.py`.
- Three-tier preset system: `configs/_presets/{local_smoke,cloud_24g,home_64g}.yaml`.
- Stage 0 config: `configs/stage0_baseline.yaml`.
- Unit tests for platform/presets/memory.
- Scripts: `scripts/local/setup_env.ps1`, `smoke_test.ps1`.

### Bootstrap decisions
- Project root: `D:\karbon\` (no sub-directory).
- Python: 3.10 + project-local venv.
- PyTorch: 2.5.1 (+cpu locally, +cu121 on cloud/home).
- Docs: bilingual, Chinese-friendly.
- Cloud platform: user-choice, platform-agnostic project code.
- Persistence: rsync after each training run.
- Longevity: perpetual target (Stage 6: 30 days uninterrupted).

---

## Template for future stages / 后续 Stage 模板

```
## [vX.Y.0-stageN] — YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...

### Deprecated / Removed
- ...

### Bounded-axiom review
- ...

### Longevity result
- 24h VRAM slope: ... GB/day
- ...
```
