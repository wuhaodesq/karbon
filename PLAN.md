# devagi · Developmental AI (Bounded, Perpetual, Migration-Ready)

> Final consolidated plan. Written on entry to Build Mode.
> 最终定稿方案。构建模式启动时首个写入。

---

## 1. Vision / 愿景

Build a **long-lived, developmental AI agent** whose learning happens *while it runs* — not merely during a training phase. It grows the way humans do: through curiosity, self-generated curriculum, and hierarchical memory that is **bounded in space** but **unbounded in time**.

以"发育式"（developmental）范式打造一个长期在线、边跑边学的智能体。它不像 LLM 那样一次性训练后冻结部署，而是像人类一样——**在有限空间内、无限时间地生长**。

**Backbone**: TTT-Hybrid (TTT-Linear + Sliding-Window Attention + TTT-MLP)

---

## 1.5 The long-term goal & the path to it / 长期目标与通向它的路径

> Revised after goal clarification. This section defines the North Star and
> the developmental ladder that leads to it.
> 目标澄清后修订。本节界定北极星终点与通往它的发育阶梯。

**Long-term North Star / 长期北极星终点：**

- 🌟 **Human 8–15-year-old level intelligence, reached *from scratch* through autonomous developmental growth.** 长期终点是**达到人类 8–15 岁少年的智力水平**——但**关键在路径**:必须像人一样**从零、自主、经验驱动地发育成长**到那里,而**不是**靠大规模预训练一次性堆出一个高分模型。
- This is a **serious, honest attempt**, not a guarantee. No team has yet reached general human-child-level intelligence via from-scratch autonomous learning; we pursue it as a direction, measuring real progress by the ladder below. 这是一次**严肃而诚实的尝试**,不是承诺一定达到;我们以下方阶梯衡量真实进展。
- The **method is the constraint**: growth must be developmental (grasp → crawl → recognize → use tools → transfer → compose), under the six bounded axioms. 方法本身即约束——成长必须是发育式的、在六条有界公理下进行。

**The developmental ladder to the North Star / 通往北极星的发育阶梯：**

These milestones are **steps toward** the 8–15 goal, not substitutes for it. 以下里程碑是**通往** 8–15 岁终点的台阶,而非终点本身。Each is independently verifiable so we never fool ourselves about progress.

| # | Milestone / 里程碑 | Verifiable signal / 可验证信号 | Stage |
|---|---|---|---|
| M1 | **Bounded memory** 有界记忆 | GPU skill/replay count stays ≤ capacity under long runs | 1–4 ✅ |
| M2 | **Skill reuse** 技能复用 | a learned skill is retrieved & re-applied later (`usage_count > 1`) | 4 ⛔ *(current gap)* |
| M3 | **Cross-task transfer** 跨任务迁移 | skill learned in task A is invoked in task B ≥1× | 5 |
| M4 | **Autonomous curriculum** 自主课程 | agent raises its own difficulty via learning-progress signal | 5 |
| M5 | **Perpetual retention** 永续保持 | ≥10 tasks retained across many days without catastrophic forgetting | 6 |
| M6 | **Compositional growth** 组合式成长 | new skills built by composing older skills (toward juvenile-level reasoning) | post-6 |

**Honest current status / 当前诚实状态：** M1 达成;**M2 尚未达成**——技能库目前只"存"不"取"(`usage_count` 恒为 1)。M2 是通往终点的第一级真实台阶,在它闭合前后续里程碑无根基。这是回到主线后的第一优先级。

**Analogy / 类比：** at its current frontier, a from-scratch autonomous learner like this is closer to *an insect / simple animal that keeps learning* than to a child. That is not failure — it is the honest research frontier. The value is the **learning paradigm** (grows without collapsing), not the intelligence altitude.
**Learning driver**: Intrinsic motivation (RND → Learning Progress)
**Memory**: Hierarchical, bounded, evicting (GPU/CPU/SSD/archive)
**Long-term stability**: Online EWC + Generative Replay + Sleep Consolidation

---

## 2. Locked Decisions / 已锁定决策

| Item | Value |
|---|---|
| Project root / 项目根 | `D:\karbon\` (no sub-directory / 不建子目录) |
| Python | 3.10, project-local `venv` at `D:\karbon\.venv\` |
| PyTorch | `torch==2.5.1+cpu` locally, `torch==2.5.1+cu121` on cloud/home-64G |
| Docs language | Bilingual, Chinese-friendly (中文优先，英文并存) |
| Git hosting | Private repo (default GitHub; user may replace with Gitee / self-hosted) |
| Cloud platform | **Platform-agnostic**. Selection criteria in `MIGRATION.md`; user picks. |
| Data persistence | rsync back to local / object storage after every training run |
| Longevity target | Truly perpetual — never restart as a fallback (Stage 6 target: 30 days uninterrupted) |
| Hardware topology | 3-stage: local laptop CPU → cloud 24G GPU → home-built 64G rig |
| Timeline | 6–12+ months, no hard deadline |

---

## 3. Six Bounded Design Axioms / 六大有界设计公理

Non-negotiable. Every PR is reviewed against these six.

1. **Zero Unbounded Structures on GPU** — no append-only structures on device memory; every buffer declares a capacity.
   任何 GPU 上驻留的数据结构必须显式声明容量上限。
2. **Eviction Before Learning** — deciding what to forget precedes deciding what to learn.
   学新东西前先决定淘汰什么。
3. **Enforced Hierarchical Storage** — hot on GPU, warm on CPU, cold on SSD, archived off-machine.
   分层存储强制执行。
4. **Periodic Consolidation** — transient state (TTT inner W, replay entries, skill candidates) is periodically distilled into fixed weights (aka "sleep").
   周期性固化：流动状态定期蒸馏进固定权重。
5. **Fragmentation Governance** — `expandable_segments`, active `empty_cache`, and monitored VRAM/RAM slope.
   显存/内存碎片主动治理。
6. **State is Serializable** — every component's state must be checkpointable (for *upgrade*, not for *restart-as-workaround*).
   任何组件状态必须可序列化——用于升级迁移，不是重启兜底。

---

## 4. Three-Stage Hardware Topology / 三阶段硬件拓扑

```
Phase 1 (now, 1–2 weeks)   Local Windows laptop (i5-1335U + 16GB RAM, no dGPU)
                           ├─ Skeleton, bounded primitives, unit tests
                           └─ Stage 0 local smoke test

Phase 2 (cloud period)     Cloud Linux + single 24GB GPU (4090 / A100 40G)
                           ├─ Stage 0 cloud first-run + 24h longevity
                           ├─ Stages 1 → 4 formal training
                           └─ rsync ckpt/logs back after each run

Phase 3 (future home rig)  Local Linux + 64GB VRAM (planned)
                           ├─ Stages 5 → 6 perpetual (>30 days)
                           ├─ Sleep consolidation, Online EWC, generative replay
                           └─ Long-lived agent lives here permanently
```

Details: see `HARDWARE_TOPOLOGY.md`.

---

## 5. Six-Stage Roadmap / 六 Stage 里程碑

| Stage | Focus | Duration | Exit criterion | Git tag |
|---|---|---|---|---|
| 0 | Skeleton + PPO baseline + memory watcher + longevity harness | 1–2 wk | 24h VRAM drift ≤ 0.2GB | `v0.0.0-stage0-local` → `-cloud` |
| 1 | + RND curiosity + Bounded Replay (3-tier) | 2 wk | state coverage ≥ 2× baseline | `v0.1.0-stage1` |
| 2 | + TTT-Hybrid backbone (PyTorch → Triton hot-swap) | 3–4 wk | long-context ↑ vs GRU; parity ≤1e-4 | `v0.2.0` → `v0.2.1-triton` |
| 3 | + Dreamer-style world model (bounded rollouts) | 3–4 wk | sample efficiency ↑ 3× | `v0.3.0-stage3` |
| 4 | + Bounded Skill Library (LoRA, LRU eviction) | 4–6 wk | observed skill reuse under cap | `v0.4.0-stage4` |
| 5 | + Auto Curriculum (Learning Progress) | 4–6 wk | autonomous difficulty ramp | `v0.5.0-stage5` |
| 6 | + Online EWC + Generative Replay + Sleep | long-term | 30 days perpetual, 10+ tasks retained | `v1.0.0-stage6` |

Details: see `ROADMAP.md`.

---

## 6. Preset System / 显存预算三档预设

Every `configs/stage*.yaml` inherits one of these presets. Switch by CLI flag.

| Preset | Target hardware | Budget | Batch × Env × Seq | Purpose |
|---|---|---|---|---|
| `local_smoke` | Laptop CPU / 16GB RAM | ≤4 GB RAM | 1 × 2 × 32 | Local smoke test |
| `cloud_24g` | Cloud 4090 24GB | ≤14 GB VRAM | 8 × 8 × 64 | Stages 0–4 formal |
| `home_64g` | Home 64GB rig | ≤48 GB VRAM | 32 × 32 × 128 | Stages 5–6 comfortable |

CLI: `python -m src.train --stage 0 --preset local_smoke`.

---

## 7. Migration Readiness / 迁移就绪原则

1. All paths via `pathlib.Path`; env-var overrides (`DEVAGI_DATA_DIR`, `DEVAGI_LOGS_DIR`, `DEVAGI_CKPT_DIR`).
2. Device detection at exactly one place: `src/platform/device.py` (`cuda → xpu → mps → cpu`).
3. Requirements split: `base.txt` (platform-agnostic), `cpu.txt`, `cuda121.txt`, `dev.txt`.
4. `.gitattributes` enforces `text eol=lf` (no CRLF surprises on Linux).
5. rsync-based data sync (no vendor-specific CLI).
6. `Makefile` provides same targets on Windows-Make and Linux (`make train`, `make test`, `make smoke`).
7. See `MIGRATION.md` for the platform-selection checklist and the local↔cloud↔home-64G choreography.

---

## 8. Stage 0 Execution Checklist (this session) / 本轮执行清单

Following steps are performed by the AI, in order:

1. ✅ Docs: `PLAN.md`, `README.md`, `HARDWARE_TOPOLOGY.md`, `MIGRATION.md`, `DESIGN_PRINCIPLES.md`, `GLOSSARY.md`, `ROADMAP.md`, `CHANGELOG.md`
2. ✅ Directory skeleton with `.gitkeep` markers
3. ✅ Project meta: `.gitignore`, `.gitattributes`, `.editorconfig`, `.env.example`, `pyproject.toml`, `Makefile`
4. ✅ `requirements/{base,cpu,cuda121,dev}.txt`
5. ✅ `scripts/local/setup_env.ps1`
6. ✅ `src/platform/{device,paths,memory_probe}.py`
7. ✅ `src/monitoring/{memory_watcher,longevity_test,health_check}.py`
8. ✅ `src/envs/minigrid_wrapper.py`
9. ✅ `configs/_presets/*.yaml` + `configs/stage0_baseline.yaml`
10. ✅ `src/train.py` (minimal PPO baseline with `--preset` / `--resume`)
11. ✅ `tests/test_*.py`
12. ✅ `scripts/local/smoke_test.ps1`

Handed to user (Step 16):
- Run `.\scripts\local\setup_env.ps1` (creates `.venv`, installs `cpu.txt + dev.txt`)
- Run `.\scripts\local\smoke_test.ps1` (5-min local smoke)
- `git init && git add . && git commit -m "Stage 0 local skeleton"`
- `git tag v0.0.0-stage0-local`
- Create private remote (GitHub / Gitee / self-hosted) and push

---

## 9. Risk Register / 风险登记

| Risk | Trigger | Downgrade action (requires approval) |
|---|---|---|
| torch 2.5.1 install fails | version conflict | fall back to 2.4.x, sync both sides |
| Windows Triton unavailable | local attempt | do not install locally; cloud/home-only |
| rsync stalls on large ckpt | flaky link | `--partial --append-verify` or object storage |
| Home 64G rig delayed | hardware not ready by Stage 5 | continue on cloud, accept extra spend |
| Stage 6 perpetuity fails | memory leak / drift | negotiate downgrade to weekly-restart tolerance |

---

## 10. Non-Goals / 不做的事

- 不追求"训练一个能对话的 chatbot"
- 不做多机分布式（single-node all the way）
- 不追前沿 LLM scale（本方案上限在 ~200M 参数级）
- 不做 RLHF / SFT / DPO 等对齐范式（本方案是无监督 developmental 路径）

---

## 11. Definition of Done for This Session / 本轮完成定义

- All files listed in §8 exist under `D:\karbon\`.
- `python -c "import src"` works after `setup_env.ps1`.
- Unit tests pass under `pytest` (offline, CPU-only, no MiniGrid required for platform tests).
- Smoke test script exists and prints a next-step banner (actual run is user's step 16).
- No dependency on any specific cloud provider or account.
