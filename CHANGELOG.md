# Changelog

All notable changes to this project are documented here.
本项目所有值得记录的变更。

Format loosely follows [Keep a Changelog](https://keepachangelog.com/).
遵循 Keep a Changelog 惯例。

---

## [Unreleased]

### Added — Stage 1/2 pre-work on local CPU (this session, batch A+B)
- `src/memory/bounded_replay.py` — three-tier BoundedReplayBuffer (GPU ring + CPU ring + SSD shard archive). Enforces Axioms 1, 2, 3, 6.
- `scripts/ci/check_bounded.py` — static AST+text linter for the bounded axioms; `make check-bounds` target added.
- `src/models/ttt_backend.py` — backend abstraction protocol; supports `pytorch`/`triton`/env-var/auto selection with graceful fallback.
- `src/models/ttt_linear.py` — pure-PyTorch mini-batch TTT-Linear (dual-form-equivalent). Guarded by parity tests (skipped until Triton env available).
- `src/models/sliding_attn.py` — causal Sliding-Window Attention with cached masks and causality proofs by perturbation tests.
- `src/models/hybrid_backbone.py` — `HybridBlock` (TTT-Linear + SWA + FFN, pre-norm) and `HybridBackbone` (stackable, sinusoidal PE, optional token embedding).
- `src/intrinsic/rnd.py` — Random Network Distillation with frozen target, trainable predictor, `RunningMeanStd` reward normalization, full state-dict serialization.
- `scripts/cloud/{setup_env,run_stage,longevity_24h,pull_logs}.sh` — LF-only bash scripts for Phase-2 cloud deployment.
- `docs/stage0_report.md` — bilingual template for Stage-0 exit report.

### Test coverage after this batch
- **93 tests passing**, 9 skipped (Triton parity, activates on Linux+CUDA).
- Suites: platform (7), presets (9), memory-watcher (6), bounded-replay (16), check-bounded (8), TTT-Linear (11+9), sliding-attn (11), hybrid-backbone (14), RND (11).
- Bounded-axiom static check: OK across 27 source files.

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
