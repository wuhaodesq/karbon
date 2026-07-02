# Glossary / 术语表

中英对照 + 缩写。项目内所有变量/类命名应与此表一致。

| 中文 | English | Abbrev | 说明 |
|---|---|---|---|
| 测试时训练 | Test-Time Training | TTT | 推理时也在做 SGD 的架构范式 |
| 线性 TTT | TTT-Linear | TTT-L | 内层是无激活线性层的 TTT 实例，等价于 DeltaNet |
| MLP 型 TTT | TTT-MLP | TTT-M | 内层是两层 MLP 的 TTT 实例 |
| 混合骨干 | Hybrid Backbone | — | TTT-Linear + Sliding Attention + TTT-MLP 三合一 |
| 滑窗注意力 | Sliding-Window Attention | SWA | 固定窗口的因果注意力 |
| 双形式 | Dual Form | — | Mini-batch TTT 的对偶计算形式，避免显式 d×d 矩阵 |
| 快权重 | Fast Weights | — | 每步更新的内层参数 W |
| 慢权重 | Slow Weights | — | 反向传播训练的外层参数 (θ_K/V/Q, η) |
| 有界回放缓冲 | Bounded Replay Buffer | BRB | 三层（GPU/CPU/SSD）容量固定的经验回放 |
| 分层存储 | Hierarchical Storage | HS | GPU/CPU/SSD/归档 四级 |
| 淘汰策略 | Eviction Policy | — | LRU / 优先级 / 加权综合 |
| 睡眠固化 | Sleep Consolidation | SC | 周期性蒸馏流动状态到固定权重 |
| 内在动机 | Intrinsic Motivation | IM | 非外部 reward 驱动的探索信号 |
| 随机网络蒸馏 | Random Network Distillation | RND | 一种内在动机实现 |
| 内在好奇心模块 | Intrinsic Curiosity Module | ICM | 预测误差型好奇心 |
| 学习进度 | Learning Progress | LP | 预测误差的下降速度作为动机 |
| 自动课程学习 | Automatic Curriculum Learning | ACL | 由 agent 自主选择任务 |
| 世界模型 | World Model | WM | 内部环境模拟器 |
| 技能库 | Skill Library | SL | 存储可复用技能的有界集合 |
| 低秩适配 | Low-Rank Adaptation | LoRA | 用于压缩技能表示 |
| 在线弹性权重固化 | Online Elastic Weight Consolidation | oEWC | 单份累积的 Fisher 保护 |
| 生成式回放 | Generative Replay | GR | 用生成模型合成旧经验 |
| 灾难性遗忘 | Catastrophic Forgetting | CF | 学新任务时旧任务性能骤降 |
| 永续训练 | Perpetual Training | PT | 长期不重启运行 |
| 存活性测试 | Longevity Test | — | 24h / 7d / 30d 显存漂移与状态健康检查 |
| 显存监视器 | Memory Watcher | — | 监控 VRAM/RAM 使用与斜率 |
| 组件健康检查 | Health Check | — | 校验每个组件是否在容量上限内 |
| 预设 | Preset | — | 硬件档位（local_smoke / cloud_24g / home_64g）配置模板 |
| 三阶段拓扑 | Three-Stage Topology | — | 本地笔记本 → 云端 24G → 家用 64G |
| 平台抽象层 | Platform Abstraction Layer | PAL | `src/platform/*` 下的 device/path/memory 抽象 |
| 有界公理 | Bounded Design Axiom | — | 见 `DESIGN_PRINCIPLES.md` |
| 冒烟测试 | Smoke Test | — | 5 分钟端到端最小验证 |
| 数值对齐 | Numerical Parity | — | PyTorch 与 Triton 后端输出差异 ≤ 1e-4 |
| Fisher 信息矩阵 | Fisher Information Matrix | FIM | 用于 EWC 加权参数保护 |
| 检查点 | Checkpoint | ckpt | 组件状态序列化文件 |
| 断点续训 | Resume Training | — | 从 ckpt 继续训练 |

## Environment Variables / 环境变量

| Variable | Default | Purpose |
|---|---|---|
| `DEVAGI_DATA_DIR` | `./data` | Replay cold tier + dataset root |
| `DEVAGI_LOGS_DIR` | `./logs` | Training logs |
| `DEVAGI_CKPT_DIR` | `./checkpoints` | Checkpoints |
| `DEVAGI_PRESET` | `local_smoke` | Default preset if `--preset` not given |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Fragmentation mitigation |

## Naming Conventions / 命名规范

- Directories / files: `snake_case`
- Classes: `PascalCase` (e.g., `BoundedReplayBuffer`, `TTTLinearBackend`)
- Functions / variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: prefix with `_`
- Checkpoint files: `ckpt_stage{N}_{step:09d}.pt`
- Log runs: `logs/stage{N}/{YYYYMMDD_HHMMSS}_{shortsha}/`
- Git tags: `v{MAJOR}.{MINOR}.{PATCH}-stage{N}[-{variant}]`
