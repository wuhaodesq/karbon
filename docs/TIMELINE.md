# Developmental Stage Timeline / 发育阶段时序图

```mermaid
timeline
    title devagi 发育式智能体 · Stage 0 → 19+

    section 感知 (Perception) : Stage 0-3
        Stage 0 : PPO 骨架 + 内存监控 + 耐久测试
                : CNN 编码器 + RND 好奇心
        Stage 1 : 有界分层回放 (GPU/CPU/SSD)
                : 状态覆盖度 ≥2× baseline
        Stage 2 : TTT-Hybrid 骨干 (PyTorch → Triton)
                : 长上下文优势 vs GRU
        Stage 3 : Dreamer 风格世界模型 (RSSM)
                : 样本效率 ↑3×

    section 动作 (Action) : Stage 4-6
        Stage 4 : 有界技能库 (LoRA, LRU 淘汰)
                : M1 达成 ✓ · M2 技能复用
        Stage 5 : 自动课程 (学习进度 AutoCurriculum)
                : 自主难度爬升 · 课程切换
        Stage 6 : Online EWC + 生成式回放 + Sleep
                : 30 天永续 · 10+ 任务保持

    section 元认知 (Metacognition) : Stage 7-8
        Stage 7 : 自我模型 + 神经符号层
                : 反思循环 · 假设检验 · 规则归纳
        Stage 8 : 语言扎根 + LLM 融合
                : 内在驱动力 · 情绪系统 · 创造力

    section 分层决策 (Hierarchical) : Stage 9-11
        Stage 9  : 环境迁移 (PhysicsSandbox → MiniGrid)
                 : 探索替代规划
        Stage 10 : 分层架构 (Manager + Worker)
                 : 子目标策略 · 技能注入
        Stage 11 : 闭合 M2 技能复用 (嵌入检索)
                 : doorkey-5x5 SR=0.985 ✓

    section 外部记忆 (External Memory) : Stage 13 ← 当前
        Stage 13 : 情景/语义/程序分层记忆
                 : Surprise Detector 关键事件存档
                 : 长期保留 > 短期回放

    section 因果推理 (Causal Reasoning) : Stage 14
        Stage 14 : 因果发现 + 反事实想象
                 : 工具使用链 · 多步因果

    section 想象规划 (Imagination) : Stage 12
        Stage 12 : Dreamer 风格想象驱动训练
                 : 基于高质量记忆的想象
                 : 样本效率 ↑ (拥有因果图后的想象)

    section Core Knowledge : Stage 15
        Stage 15 : 物体永存 · 直觉物理
                 : 数感 · 空间推理

    section 神经符号 (Neuro-symbolic) : Stage 16
        Stage 16 : 规则抽取 + 逻辑推理
                 : 神经符号桥梁

    section 元反思 (Meta-reflection) : Stage 17
        Stage 17 : 学习过程自省
                 : 策略元认知 · 知识缺口检测

    section 组合式成长 (Compositional) : Stage 18
        Stage 18 : 技能组合创新
                 : 变换式创造性

    section 永续学习 (Perpetual) : Stage 19+
        Stage 19+ : 无限时间持续学习
                  : 开放式探索 · 自主目标设定
```

## Current Status / 当前状态

- **Completed**: Stage 0–14 (Stage 14 closed at 1,000,000 steps, report in `docs/stage14_report.md`)
- **Running**: (none - Stage 14 sealed, deciding next stage)
- **Next**: Stage 12 (Imagination) blocked - needs working WM (proven S14: encode cost > recon gain on MiniGrid). Recommend 3D world migration or Stage 15 (Core Knowledge) as bridge.

> **时序修正说明**: 原定 Stage 12（想象训练）→ Stage 13（外部记忆）→ Stage 14（因果推理）。
> 经 Stage 13 训练发现世界模型 (RSSM) 未有效学习（wm=0.506, near chance），导致依赖 world model 的
> Stage 12 想象训练在此时启动可能效果不佳。Stage 14 的因果发现 + 反事实想象可以先在无想象训练的条件下
> 建立因果图，为后续 Stage 12 提供更高质的记忆基础。顺序调整为：
> **Stage 13（外部记忆）→ Stage 14（因果推理）→ Stage 12（想象训练）**。

## Stage Descriptions / 阶段说明

| Stage | 中文名 | 核心交付 | 验证指标 |
|-------|--------|----------|----------|
| 0 | 骨架基线 | PPO + RND + Memory Watcher | 24h VRAM 漂移 ≤0.2GB |
| 1 | 有界回放 | 三阶回放 (GPU/CPU/SSD) | 覆盖度 ≥2× baseline |
| 2 | TTT 骨干 | TTT-Linear + SWA + FFN | 长上下文 vs GRU 持平 |
| 3 | 世界模型 | RSSM 预测 + 重建 | 样本效率 ↑3× |
| 4 | 技能库 | LoRA 技能 + LRU 淘汰 | usage_count > 1 |
| 5 | 自动课程 | LP 驱动课程切换 | 自主难度爬升 |
| 6 | 永续机制 | EWC + GR + Sleep | 30 天 10+ 任务 |
| 7 | 元认知 | SelfModel + Symbolic | 规则归纳 |
| 8 | 语言融合 | LLM Fusion + 创造力 | 指令跟随 |
| 9 | 环境迁移 | Physics→MiniGrid | 跨环境适应 |
| 10 | 分层决策 | Manager/Worker 架构 | 子目标达成率 |
| 11 | 技能闭合 | M2 嵌入检索 | doorkey SR 0.985 |
| **13** | **外部记忆** | **情景/语义/程序记忆** | **长期关键事件保留** |
| **14** | **因果推理** | **因果图 + 反事实** | **多步因果链完成** |
| 12 | 想象规划 | Dreamer 想象训练 | 样本效率 vs S11 (拥有因果图后) |
| 15 | 核心知识 | 物体永存/物理/数感 | 物理直觉任务 |
| 16 | 神经符号 | 规则 + 逻辑推理 | 符号推理准确率 |
| 17 | 元反思 | 学习自省 | 知识缺口检测 |
| 18 | 组合成长 | 技能组合 + 创造 | 新技能组合数 |
| 19+ | 永续学习 | 开放探索 | 无限时间稳定 |
