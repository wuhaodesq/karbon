# devagi

**Developmental AI · Bounded · Perpetual · Migration-Ready**

一个从零开始、边跑边学、内存有界、可无限期运行的智能体研究平台。
A developmental AI research platform: learns while running, bounded memory, perpetually runnable.

> **状态**：Stage 0 · Skeleton
> **项目根**：`D:\karbon\`（无子目录 / no sub-directory）
> **语言**：中文优先，英文并存 / Bilingual, Chinese-friendly

---

## Quick Start / 快速开始

```powershell
# 1. Create venv and install deps (local CPU)
.\scripts\local\setup_env.ps1

# 2. Activate venv
.\.venv\Scripts\Activate.ps1

# 3. Run 5-minute smoke test
.\scripts\local\smoke_test.ps1
```

---

## What is this? / 这是什么？

An implementation of a **long-lived developmental agent** whose architecture combines:

- **TTT-Hybrid backbone** — Test-Time Training (Linear + MLP) + Sliding-Window Attention. Provides in-context adaptation while running.
- **Intrinsic motivation** — RND-driven curiosity, later upgraded to Learning Progress.
- **Bounded hierarchical memory** — GPU/CPU/SSD/archive four-tier storage; every buffer has an eviction policy.
- **Sleep consolidation** — periodic offline distillation of transient state into fixed weights.
- **Online lifelong learning** — Online EWC + Generative Replay against catastrophic forgetting.

一个"发育式"智能体的完整实现：TTT 骨干 + 内在动机 + 有界分层记忆 + 睡眠固化 + 在线终身学习。

---

## Read First / 必读

| Doc | Purpose |
|---|---|
| [`PLAN.md`](PLAN.md) | Final consolidated project plan / 最终方案 |
| [`FULL_JOURNEY.md`](FULL_JOURNEY.md) | End-to-end Stage 0–6 timeline & cost / 全流程时间成本 |
| [`DESIGN_PRINCIPLES.md`](DESIGN_PRINCIPLES.md) | Six bounded design axioms / 六大有界公理 |
| [`ROADMAP.md`](ROADMAP.md) | Stage 0–6 milestones / 六 Stage 里程碑 |
| [`HARDWARE_TOPOLOGY.md`](HARDWARE_TOPOLOGY.md) | 3-stage hardware evolution / 三阶段硬件路线 |
| [`MIGRATION.md`](MIGRATION.md) | Local ↔ cloud ↔ home-64G migration / 迁移指南 |
| [`GLOSSARY.md`](GLOSSARY.md) | Chinese-English term table / 中英术语 |
| [`AGENTS.md`](AGENTS.md) | Instructions for automated coding assistants |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Human contributor guide |
| [`CHANGELOG.md`](CHANGELOG.md) | Stage-by-stage changelog |

---

## Project Layout / 项目结构

```
D:\karbon\
├── configs/            # YAML configs per stage + preset system
├── requirements/       # Split by platform: base/cpu/cuda121/dev
├── src/
│   ├── envs/           # MiniGrid, Crafter wrappers
│   ├── models/         # TTT-Linear, TTT-MLP, Sliding Attn, Hybrid
│   ├── memory/         # Bounded replay, skill library, generative replay
│   ├── intrinsic/      # RND, Learning Progress
│   ├── curriculum/     # Auto curriculum
│   ├── continual/      # Online EWC, sleep consolidation
│   ├── monitoring/     # Memory watcher, longevity harness, health check
│   ├── platform/       # Device/paths/memory-probe abstraction
│   └── train.py        # Main entry
├── scripts/
│   ├── local/          # Windows PowerShell
│   ├── cloud/          # Linux bash (platform-agnostic)
│   └── home/           # Future home 64G rig
├── tests/
├── notebooks/
├── docs/
└── logs, checkpoints, data (gitignored)
```

---

## Non-Goals / 不做的事

- No chatbot / LLM alignment (RLHF/SFT/DPO out of scope)
- No multi-node distributed training
- Frontier LLM scale not targeted (~200M param ceiling in this plan)
- Not aiming to compete on Atari/MuJoCo benchmarks — we're chasing *developmental phenomena*

---

## License

TBD — decide at first public release.
