# Migration Guide / 迁移指南

Bilingual. How to move the project between Phase 1 (local), Phase 2 (cloud), and Phase 3 (home 64G).
双语。项目在三阶段硬件之间迁移的完整指南。

---

## 1. Platform Selection Checklist / 平台选型清单

**We do not recommend a specific vendor.** Use this checklist when choosing.
**本文档不推荐具体云平台家。** 你自己挑，看这 9 个维度：

| # | Dimension | What to check |
|---|---|---|
| 1 | GPU model | 3090 / 4090 / A100-40 / A100-80 — dictates ceiling stage |
| 2 | Billing | per-second / per-hour / spot / monthly. This project favors per-second or spot |
| 3 | Persistent disk | is data kept after stop? capacity? cost? |
| 4 | Network | can it reach HuggingFace / GitHub / PyPI reliably? |
| 5 | SSH & rsync | port 22 open? can we `rsync` in/out? if not, must use object storage |
| 6 | Base image | official PyTorch + CUDA 12.1 image available? |
| 7 | Preemption | spot-eviction probability, auto-checkpoint support |
| 8 | Cost transparency | does stopping stop billing? data-disk fees? balance alerts? |
| 9 | Compliance | invoices needed? data can leave the country? |

**Common candidates** (informational, not endorsement): AutoDL, RunPod, Lambda Labs, Vast.ai, Tencent Cloud, Alibaba Cloud.

---

## 2. Phase 1 → Phase 2: Local Laptop → Cloud / 本地笔记本 → 云端

### 2.1 Prerequisites / 前置条件

- Stage 0 local smoke passed locally.
- `v0.0.0-stage0-local` tag committed and pushed to a Git remote (private repo).
- Cloud instance has: Ubuntu 22.04, Python 3.10, NVIDIA driver + CUDA 12.1, git.

### 2.2 On the cloud machine / 在云端机器上

```bash
# 1. Clone
git clone <your-remote-repo-url> ~/karbon
cd ~/karbon

# 2. Setup environment (Linux)
bash scripts/cloud/setup_env.sh

# 3. Activate
source .venv/bin/activate

# 4. Sanity check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.5.1+cu121  True

# 5. First cloud run — Stage 0 with cloud preset
bash scripts/cloud/run_stage.sh 0 cloud_24g

# 6. 24h longevity test
bash scripts/cloud/longevity_24h.sh
```

### 2.3 After training / 训练结束

```bash
# From cloud, push ckpt/logs back to local (or object storage)
# Configure a target env-var first
export DEVAGI_REMOTE_TARGET=<user>@<local-or-storage>:/path/to/karbon-mirror

rsync -avz --partial --append-verify --progress \
  checkpoints/ logs/ docs/ \
  "$DEVAGI_REMOTE_TARGET/"
```

Then destroy the cloud instance safely — no state loss.

### 2.4 Verification / 验证

- `git status` on cloud shows only untracked logs/ckpt (expected, gitignored)
- Local Windows can open the returned `docs/stage{N}_report.md` and view figures
- Local `git pull` gets the latest tag pushed from cloud (typically `v0.X.0-stage{N}`)

---

## 3. Phase 2 → Phase 3: Cloud → Home 64G Rig / 云端 → 家用 64G

### 3.1 Prerequisites / 前置条件

- Home rig: NVIDIA GPU, 64 GB total VRAM, Ubuntu 22.04 LTS or 24.04 LTS, CUDA 12.1+
- SSH access from your dev machine
- Stage 4 done on cloud, tagged `v0.4.0-stage4`

### 3.2 One-time setup / 一次性配置

```bash
git clone <your-remote-repo-url> ~/karbon
cd ~/karbon
bash scripts/home/setup_env.sh          # equivalent to cloud setup
source .venv/bin/activate

# Pull latest ckpt from wherever it lives (local mirror / object storage / directly from cloud if still alive)
rsync -avz --partial --progress \
  <source>:/path/to/karbon-mirror/checkpoints/ \
  checkpoints/
```

### 3.3 Kickoff Stage 5 / 启动 Stage 5

```bash
# Resume from Stage 4 checkpoint, use home_64g preset
bash scripts/home/run_perpetual.sh \
  --resume checkpoints/ckpt_stage4_000XXXXXX.pt \
  --stage 5 \
  --preset home_64g
```

The `run_perpetual.sh` wraps `nohup` / `tmux` and enables the health daemon.

---

## 4. Common Gotchas / 常见坑

### 4.1 CRLF vs LF
- `.gitattributes` enforces `text eol=lf`. Do **not** edit shell scripts with Windows Notepad.
- Use VSCode / Notepad++ with LF explicitly.

### 4.2 Path separators
- Never write `"data\\replay"` in code. Always `Path(paths.data_dir()) / "replay"`.
- The `src/platform/paths.py` module is the single source of paths.

### 4.3 Device detection
- Never call `.cuda()` directly. Always `.to(device)` where `device = get_device()` from `src/platform/device.py`.

### 4.4 Torch version mismatch
- Both sides pin `torch==2.5.1`. Different wheel variant (`+cpu` vs `+cu121`) is fine because the Python API is identical.
- If you must upgrade, upgrade both sides at once and re-run parity tests.

### 4.5 Triton on Windows
- Not supported. `src/models/ttt_linear_triton.py` is guarded by:
  ```python
  try:
      import triton
      TRITON_AVAILABLE = True
  except ImportError:
      TRITON_AVAILABLE = False
  ```
- On Windows, the code silently falls back to `ttt_linear.py` PyTorch backend.

### 4.6 Environment variables
- Copy `.env.example` to `.env` and adjust. `.env` is gitignored.
- Never hardcode absolute paths in configs or scripts.

---

## 5. rsync Recipes / rsync 参考命令

**Cloud → Local mirror** (run from cloud):
```bash
rsync -avz --partial --append-verify --progress \
  checkpoints/ logs/ docs/figures/ \
  ${DEVAGI_REMOTE_TARGET}/
```

**Local mirror → Home rig** (run from home rig):
```bash
rsync -avz --partial --progress \
  ${DEVAGI_MIRROR_SOURCE}/checkpoints/ \
  checkpoints/
```

**Selective (only latest ckpt)**:
```bash
rsync -avz --include="ckpt_stage4_*.pt" --exclude="*" \
  ${DEVAGI_REMOTE_TARGET}/checkpoints/ \
  checkpoints/
```

---

## 6. Verification Matrix / 迁移后验证矩阵

After any phase transition, run:

```bash
# On the new machine
source .venv/bin/activate
pytest -x tests/test_platform_device.py
pytest -x tests/test_config_presets.py
python -m src.train --stage 0 --preset local_smoke --smoke-only
```

All three must pass. If any fail, do not proceed to real training — debug the environment first.

---

## 7. Emergency Rollback / 应急回滚

If a stage regresses on new hardware:

1. `git checkout` the last good tag.
2. `rsync` back the last known-good checkpoint from local mirror.
3. Do not retrain from scratch — first bisect what changed.
4. Log the incident in `CHANGELOG.md` under an `## Incident` section.
