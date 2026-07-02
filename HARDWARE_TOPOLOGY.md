# Hardware Topology / 硬件拓扑

Three-stage evolution of hardware backing this project.
项目对应的三阶段硬件演化。

---

## Phase 1 · Local Windows Laptop (current) / 阶段 1：本地 Windows 笔记本（当前）

**Machine**: HP 240R 14" G9 Notebook PC
**CPU**: Intel i5-1335U (10 cores / 12 threads, U-series)
**RAM**: 16 GB total (≈11 GB usable for training after OS)
**GPU**: Intel UHD Graphics (integrated, **no CUDA**)
**Storage**: KIOXIA 512 GB NVMe SSD (C:/D: partitions)
**OS**: Windows 11 + WSL2 Ubuntu (Stopped)
**Role**:
- All code writing and design
- Skeleton, docs, unit tests, static checks
- 5-minute smoke tests with `local_smoke` preset (batch=1, seq=32, ≤1M params)

**Not for**:
- Any real training beyond Stage 0 smoke
- Anything requiring CUDA or Triton

---

## Phase 2 · Cloud Linux 24GB / 32GB GPU / 阶段 2：云端 Linux 24GB / 32GB 显存

**Target hardware**: RTX 4090 24 GB, RTX 5090 32 GB, or A100 40 GB
**OS**: Linux (Ubuntu 22.04 typical)
**Python**: 3.10 / 3.11 / 3.12 + venv
**PyTorch wheels** (auto-selected by `scripts/cloud/setup_env.sh`):
- RTX 5090 (Blackwell, sm_120) → `torch>=2.8+cu128` (`requirements/cuda128.txt`)
- Ampere / Hopper / Ada         → `torch==2.5.1+cu121` (`requirements/cuda121.txt`)
- If the image already ships PyTorch, pass `--skip-torch` to reuse it.
**Role**:
- Stage 0 first cloud run + 24 h longevity
- Stages 1–4 formal training
- Triton kernel development for TTT (Stage 2b)
- After every run: `rsync` checkpoints & logs back to local (or object storage)

**Preset picking**:
| GPU | Preset |
|---|---|
| 4090 24 GB / A100-40 | `cloud_24g` |
| **RTX 5090 32 GB** | **`cloud_5090`** (best fit — 22 GB budget, batch 16) |
| ≥48 GB | `home_64g` |

**Platform selection**: **user's choice** — see `MIGRATION.md` §"Platform Selection Checklist".

**Data lifecycle**:
- Cloud persistent disk **not required** — every training result is `rsync`'d back
- Cold data (replay archive) stays on cloud SSD during a run, then transferred out
- After a run, cloud instance can be destroyed without data loss

---

## Phase 3 · Home 64GB Rig (future, planned) / 阶段 3：家用 64GB 显存机（未来规划）

**Target VRAM**: 64 GB (or more)
**Candidate configurations** (evaluate at purchase time):

| Config | VRAM | Notes |
|---|---|---|
| Dual RTX 5090 | 32 + 32 = 64 GB | Available late 2025+; NVLink absent but DDP works |
| Single RTX 6000 Ada | 48 GB | Under 64G but very comfortable |
| Single A100 80 GB (used) | 80 GB | Premium price, best headroom |
| Modded 4090 48G | 48 GB | Grey-market; verify vendor |
| Mac Studio M3 Ultra 192G unified | 192 GB unified | ⚠ MPS backend, **Triton unsupported** — likely blocker for Stage 2b |

**Recommendation**: **evaluate at purchase time** based on availability and Triton support. NVIDIA + CUDA is strongly preferred to keep Triton working.

**OS**: Linux (Ubuntu 22.04 LTS or 24.04 LTS)
**Python**: same as cloud — `3.10 + venv + torch==2.5.1+cu121 + triton`

**Role**:
- Stages 5–6 perpetual training
- The "home" of the long-lived developmental agent
- Runs continuously — the agent lives here permanently

**Migration in**: `scripts/home/setup_env.sh` provisions env; `git pull` gets code; `rsync` from cloud pulls the last cloud checkpoint; then `bash scripts/home/run_perpetual.sh` starts perpetual loop.

---

## Data Flow / 数据流

```
                Code (Git)               Ckpt/Logs (rsync)
Local  <------------------->  Cloud  ------------------------>  Object Storage
                                 |
                                 | rsync (Phase 3+)
                                 v
                              Home 64G  <-------- Object Storage
```

- **Code**: single source of truth = Git repo. All three locations are Git clients.
- **Configs**: in Git.
- **Checkpoints**: born on the training machine, `rsync`'d off, retrievable from Object Storage or local.
- **Replay cold data**: too large for Git; either regenerated per run or synced via rsync.
- **Small logs / metrics / figures**: pulled back to local for analysis.

---

## Preset ↔ Hardware Mapping / 预算档位 ↔ 硬件映射

| Preset | Phase | Typical params | Batch × Env × Seq |
|---|---|---|---|
| `local_smoke` | Phase 1 (laptop CPU) | ≤1 M | 1 × 2 × 32 |
| `cloud_24g` | Phase 2 (cloud 4090 / A100-40) | 5–20 M | 8 × 8 × 64 |
| `cloud_5090` | Phase 2 (cloud RTX 5090 32 GB) | 15–50 M | 16 × 16 × 96 |
| `home_64g` | Phase 3 (home rig) | 20–200 M | 32 × 32 × 128 |

Switch: `--preset {local_smoke|cloud_24g|cloud_5090|home_64g}`.

---

## Transition Triggers / 阶段迁移触发条件

**Phase 1 → Phase 2**: Stage 0 local smoke passes AND `v0.0.0-stage0-local` tag committed.
**Phase 2 → Phase 3**: home rig operational (has NVIDIA GPU + CUDA 12.x + Triton), Stage 4 completed on cloud.

Do not migrate between phases mid-Stage; wait for a clean Stage boundary.

---

## No-Go List / 硬件红线

- ❌ AMD ROCm — Triton on ROCm is nascent; not planned.
- ❌ Apple MPS as primary — no Triton, no `torch.cuda`, breaks Stage 2b.
- ❌ Multi-node distributed — out of scope; single-node all the way.
- ❌ TPU — PyTorch/XLA overhead not worth it for a small research project.
