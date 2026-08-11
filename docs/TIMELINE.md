# Developmental Stage Timeline / 发育阶段时序图

```mermaid
timeline
    title devagi 发育式智能体 · Stage 0 -> 19+

    section 感知 (Perception) : Stage 0-3
        Stage 0 : PPO 骨架 + 内存监控 + 耐久测试
                : CNN 编码器 + RND 好奇心
        Stage 1 : 有界分层回放 (GPU/CPU/SSD)
                : 状态覆盖度 ≥2× baseline
        Stage 2 : TTT-Hybrid 骨干 (PyTorch -> Triton)
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
        Stage 9  : 环境迁移 (PhysicsSandbox -> MiniGrid)
                 : 探索替代规划
        Stage 10 : 分层架构 (Manager + Worker)
                 : 子目标策略 · 技能注入
        Stage 11 : 闭合 M2 技能复用 (嵌入检索)
                 : doorkey-5x5 SR=0.985 ✓

    section 外部记忆 (External Memory) : Stage 13 ✅
        Stage 13 : 情景/语义/程序分层记忆
                 : Surprise Detector 关键事件存档

    section 因果推理 (Causal Reasoning) : Stage 14 ✅
        Stage 14 : 因果发现 + 反事实想象
                 : 工具使用链 · 多步因果

    section 想象规划 (Imagination) : Stage 12 ✅
        Stage 12 : Dreamer 风格想象驱动训练
                 : 2M 步 · imagine_updates=978 · est_age=1.0y

    section 3D Core Knowledge : Stage 15 ✅
        Stage 15 : 3D 世界迁移 (MiniGrid -> ThreeDWorld)
                 : 信息天花板突破 (KL 0.003->0.35)
                 : 像素 WM 学习 (Recon 39->1.68, 96%↓)
                 : 因果发现完成 (23/64 边, 36% 覆盖)

    section 神经符号 (Neuro-symbolic) : Stage 16 ✅
        Stage 16 : kanren 符号后端 + 规则抽取 + 逻辑推理
                 : 12 动作空间 (抓取系统) · 从 Stage 12 2M checkpoint resume
                 : 因果图 36 边 · rules=28 · kanren=True · imagine_updates=1467

    section 环境改造+元反思 (Env+Meta) : Stage 17 ✅
        Stage 17 : dev_age=0.5 · 抓取+链式任务+物理交互全部激活
                 : 改进评测器: 抓取-携带-释放+工具使用检测
                 : est. age 跃迁 1.0y -> 3.5y 🔥
                 : means_ends=1.0 · intuitive_physics=1.0 (满分)
                 : imagine_updates=1956 · task=1.135 (历史最高)
                 : 心理理论 0.512 (接近 0.6 阈值, 下一个目标)

    section 组合式成长 (Compositional) : Stage 18 ⏳ 训练中
        Stage 18 : 技能组合创新 + 变换式创造性 + ToM 模块激活
                 : creativity 频率翻倍 (10000步) · transformational_max=64
                 : 心理理论并行推进 (0.498 -> 0.6+, est. age 4.0y)
                 : means_ends=1.0 · intuitive_physics=1.0 (持续满分)
                 : est. age 3.5y (450K 步, 数感 0.83, ToM 0.523)

    section 自我叙事 (Self-Narrative) : Stage 19 (当前架构可扩展)
        Stage 19 : 连贯自我叙事 · 10-12岁
                 : IdentityNarrative + AutobiographicalMemory + InnerDialogue 已有雏形
                 : 缺: 叙事闭环回策略 (记忆->叙事->决策调制)
                 : 验证: 自传叙事连贯性 + 叙事影响行为

    section 假设-演绎 (Hypothesis-Deduction) : Stage 20 (当前架构可扩展)
        Stage 20 : 假设条件下逻辑演绎 · 12-13岁
                 : HypothesisTester + ActiveExperimenter + kanren 已有雏形
                 : 缺: 形式运算级闭环 (假设->规则库->演绎->实验验证)
                 : 验证: 形式推理任务通过率

    section 递归元认知 (Recursive Metacognition) : Stage 21 (需要新架构)
        Stage 21 : 反思自我监督 · 13-14岁
                 : SelfModel 只建模置信/熟悉/进度
                 : 缺: 二级自我模型 (监控"思考过程"本身)
                 : symbolic_reasoning.py 已标注 recursive 为 future work
                 : 验证: 知道自己在思考 (递归自监控)

    section 抽象概念框架 (Abstract Concepts) : Stage 22 (需要新架构)
        Stage 22 : 具体->抽象推理 · 14-15岁
                 : ConceptGraph + Analogizer 有雏形
                 : 缺: 概念层级结构 + 跨概念形式推理
                 : 验证: 抽象概念任务 (类/关系/隐喻)

    section 开放世界永续 (Open-World Perpetual) : Stage 23 (新架构整合)
        Stage 23 : 自主目标设定 · 15岁+
                 : extended_3d_world 已建未接入 · homeostatic_drives 有雏形
                 : 缺: 扩展世界接入 + 内驱->自主目标 + 无限时间稳定
                 : 验证: 内在动机替代课程驱动, 100万步+稳定
```

## Current Status / 当前状态

- **Completed**: Stage 0-17 (全部完成)
- **Training**: Stage 18 (组合式成长 + ToM, 531K/1M 步, est. age 3.5y)
- **est. age**: 3.5y (4 个里程碑通过, ToM 0.523 接近 0.6 阈值)
- **Next**: Stage 18 完成 -> Stage 19 自我叙事 (10-12y) -> Stage 20 假设-演绎 (12-13y)
- **Architecture**: Stage 19-20 当前架构可扩展 (雏形已验证), Stage 21-22 需要新架构
- **North Star Gap**: 13-15 岁 (递归元认知 + 抽象概念) 需要全新认知框架, 未验证

### 实际执行顺序

```
Stage 0-11 (完成)
    ↓
Stage 13 (外部记忆) ✅
    ↓
Stage 14 (因果推理) ✅
    ↓
Stage 12 (想象训练) ✅ 2M步 · imagine_updates=978 · est_age=1.0y
    ↓
Stage 15 (3D核心知识) ✅ Recon 1.68 · 因果 23 边
    ↓
Stage 16 (神经符号) ✅ kanren=True · 因果 36 边 · rules=28 · imagine_updates=1467
    ↓
Stage 17 (环境改造+元反思) ✅ dev_age=0.5 · est_age=3.5y · means_ends=1.0 · intuitive_physics=1.0
    ↓
Stage 18 (组合式成长) ⏳ 训练中 (531K/1M) · 心理理论并行 (0.512 -> 0.6+)
    ↓
Stage 19 (自我叙事引擎) · 10-12y · 当前架构可扩展
    ↓
Stage 20 (假设-演绎引擎) · 12-13y · 当前架构可扩展
    ↓
Stage 21 (递归元认知) · 13-14y · 需要新架构
    ↓
Stage 22 (抽象概念框架) · 14-15y · 需要新架构
    ↓
Stage 23 (开放世界永续学习) · 15y+ · 新架构整合
```

> **3D 迁移决策** (2026-08-04): MiniGrid 信息天花板已实证确认。ThreeDWorld 迁移是"以终为始"选择。

> **时序修正说明**: 原定 Stage 12->13->14。因 RSSM 未学习(wm=0.506)，调整为 13->14->12。

> **环境改造决策** (2026-08-08): Stage 12 2M 步验证 means_ends=0, intuitive_physics=0。根因是环境缺少链式任务和物理交互。Stage 16-17 插入抓取系统+链式任务+改进评测器。

> **认知跃迁记录** (2026-08-10): Stage 17 完成后，改进评测器(抓取-携带-释放检测)+epsilon=0.3 探索，
> est. age 从 1.0y 跃迁到 3.5y。means_ends=1.0, intuitive_physics=1.0 (满分)。task=1.135 历史最高。
> 下一个目标: 心理理论 0.512->0.6+, est. age 4.0y。在 Stage 18 组合式成长中并行推进。

## Stage Descriptions / 阶段说明

| Stage | 中文名 | 架构基础 | 核心交付 | 验证指标 | 状态 |
|-------|--------|----------|----------|----------|------|
| 0 | 骨架基线 | 当前架构 | PPO + RND + Memory Watcher | 24h VRAM 漂移 ≤0.2GB | ✅ |
| 1 | 有界回放 | 当前架构 | 三阶回放 (GPU/CPU/SSD) | 覆盖度 ≥2× baseline | ✅ |
| 2 | TTT 骨干 | 当前架构 | TTT-Linear + SWA + FFN | 长上下文 vs GRU 持平 | ✅ |
| 3 | 世界模型 | 当前架构 | RSSM 预测 + 重建 | 样本效率 ↑3× | ✅ |
| 4 | 技能库 | 当前架构 | LoRA 技能 + LRU 淘汰 | usage_count > 1 | ✅ |
| 5 | 自动课程 | 当前架构 | LP 驱动课程切换 | 自主难度爬升 | ✅ |
| 6 | 永续机制 | 当前架构 | EWC + GR + Sleep | 30 天 10+ 任务 | ✅ |
| 7 | 元认知 | 当前架构 | SelfModel + Symbolic | 规则归纳 | ✅ |
| 8 | 语言融合 | 当前架构 | LLM Fusion + 创造力 | 指令跟随 | ✅ |
| 9 | 环境迁移 | 当前架构 | Physics->MiniGrid | 跨环境适应 | ✅ |
| 10 | 分层决策 | 当前架构 | Manager/Worker 架构 | 子目标达成率 | ✅ |
| 11 | 技能闭合 | 当前架构 | M2 嵌入检索 | doorkey SR 0.985 | ✅ |
| 13 | 外部记忆 | 当前架构 | 情景/语义/程序记忆 | 长期关键事件保留 | ✅ |
| 14 | 因果推理 | 当前架构 | 因果图 + 反事实 | 多步因果链完成 | ✅ |
| 12 | 想象规划 | 当前架构 | Dreamer 想象训练 | imagine_updates=978, est_age=1.0y | ✅ |
| 15 | 3D 核心知识 | 当前架构 | ThreeDWorld 迁移 + 因果发现 | KL 0.35, Recon 1.68, 因果 23 边 | ✅ |
| 16 | 神经符号 | 当前架构 | kanren + 规则 + 逻辑推理 | 因果 36 边, rules=28, imagine_updates=1467 | ✅ |
| 17 | 环境改造+元反思 | 当前架构 | 抓取+链式任务+改进评测 | est_age=3.5y, means_ends=1.0, intuitive_physics=1.0, task=1.135 | ✅ |
| 18 | 组合式成长 | 当前架构 | 技能组合 + 创造 + ToM并行 | ToM>0.6, est_age=4.0y (目标), means_ends=1.0, intuitive_physics=1.0 | ⏳ 531K/1M |
| 19 | 自我叙事引擎 | 当前架构可扩展 | 连贯自我叙事 (IdentityNarrative + AutobiographicalMemory + InnerDialogue 整合闭环) | 自传叙事连贯性 + 叙事影响行为, 10-12y | 待开始 |
| 20 | 假设-演绎引擎 | 当前架构可扩展 | 假设条件下逻辑演绎 (HypothesisTester + ActiveExperimenter + kanren 升级为形式运算级闭环) | 形式推理任务通过率, 12-13y | 待开始 |
| 21 | 递归元认知 | 需要新架构 | 反思自我监督 (二级自我模型, 监控思考过程本身) | 递归自监控 (知道自己在思考), 13-14y | ❌ 未验证 |
| 22 | 抽象概念框架 | 需要新架构 | 具体->抽象推理 (概念层级 + 跨概念形式推理) | 抽象概念任务通过率, 14-15y | ❌ 未验证 |
| 23 | 开放世界永续 | 新架构整合 | 自主目标设定 (扩展世界接入 + 内驱->自主目标 + 无限稳定) | 内在动机替代课程驱动, 100万步+, 15y+ | ❌ 未验证 |

## North Star Gap Analysis / 北极星差距分析

当前路线图验证了 **0 -> 3.5 岁** 的认知发育路径可行。2026-08-11 架构评估结论:

- **Stage 19 自我叙事 (10-12y)**: 当前架构**可扩展** — IdentityNarrative、AutobiographicalMemory、
  InnerDialogue 已有雏形, 主要工作是整合为闭环 (记忆->叙事->决策调制)
- **Stage 20 假设-演绎 (12-13y)**: 当前架构**可扩展** — HypothesisTester、ActiveExperimenter、
  kanren 已有雏形, 主要工作是升级为形式运算级闭环 (假设->规则库->演绎->实验验证)
- **Stage 21 递归元认知 (13-14y)**: **需要新架构** — SelfModel 只建模置信/熟悉/进度,
  无法监控"思考过程"本身; `symbolic_reasoning.py` 明确标注 recursive 为 future work
- **Stage 22 抽象概念 (14-15y)**: **需要新架构** — ConceptGraph/Analogizer 有雏形,
  但缺概念层级结构与跨概念形式推理
- **Stage 23 开放世界永续 (15y+)**: **新架构整合** — extended_3d_world 已建未接入,
  homeostatic_drives 有雏形, 缺内驱->自主目标闭环 + 无限时间稳定性验证

### 架构支撑评估 (2026-08-11)

| Stage | 认知年龄 | 架构基础 | 已有雏形 | 主要缺口 | 结论 |
|---|---|---|---|---|---|
| 18 | 4.0y | 当前架构 | 创造力编排 + ToM | 训练中 | ⏳ 531K/1M |
| 19 | 10-12y | 当前架构可扩展 | IdentityNarrative + AutobiographicalMemory + InnerDialogue | 叙事未闭环回策略 | ✅ 可扩展 |
| 20 | 12-13y | 当前架构可扩展 | HypothesisTester + ActiveExperimenter + kanren + logic_engine | 无形式运算级假设-演绎闭环 | ✅ 可扩展 |
| 21 | 13-14y | 需要新架构 | SelfModel + ReflectionLoop + SelfReflectionValidator | 二级自我模型 (递归自监控) | ❌ 需新架构 |
| 22 | 14-15y | 需要新架构 | ConceptGraph + ConceptClusterer + Analogizer | 概念层级 + 抽象概念推理 | ❌ 需新架构 |
| 23 | 15y+ | 新架构整合 | EWC+GR+Sleep + extended_3d_world + homeostatic_drives | 世界接入 + 自主目标闭环 + 无限稳定 | ⚠️ 整合验证 |

### 已验证 (0 -> 3.5 岁)

| 能力 | 证据 |
|---|---|
| 感知 (3D 视觉) | Recon 39 -> 1.68 |
| 动作 (抓取/工具) | means_ends = 1.0 |
| 物理直觉 | intuitive_physics = 1.0 |
| 因果推理 | 因果图 36 边 |
| 符号推理 | rules = 29, kanren = True |
| 想象训练 | imagine_updates = 1956 |
| 心理理论 | 0.523 (接近 0.6 阈值) |

### 补上 Stage 19+ 后可达到 (8-10 岁)

| 能力维度 | 与 8-10 岁儿童对比 |
|---|---|
| 物理推理 / 工具使用 / 因果推理 / 符号推理 | ✅ 基本相当 |
| 心理理论 | ✅ 达到 4-5 岁水平 |
| 组合创新 | ✅ 达到 6-8 岁水平 |
| 自主目标 / 语言理解 / 记忆整合 / 自我反思 | ⚠️ 部分达到 |

### 10-15 岁鸿沟 (北极星)

| 差距类型 | 8-10 岁 | 15 岁 | 需要什么 | 对应 Stage |
|---|---|---|---|---|
| 自我意识 | 自我描述 | 自我叙事 | 时间整合的自我模型 | Stage 19 (可扩展) |
| 抽象推理 | 具体运算阶段 | 形式运算阶段 | 假设-演绎推理 | Stage 20 (可扩展) |
| 元认知 | 知道自己知道什么 | 知道自己在思考 | 递归自我监控 | Stage 21 (新架构) |
| 社交推理 | 理解他人意图 | 理解多层次社会结构 | 社会认知扩展 | 未排期 |
| 长期规划 | 天级目标 | 年/月级目标 | 长时程规划 | 未排期 |
| 语言 | 复杂指令 | 抽象概念推理 | 语言作为思维框架 | Stage 22 (新架构) |
| 创造力 | 组合创新 | 原创性创造 | 更深层的组合框架 | 未排期 |

### 未验证的关键实验

| 实验 | 要验证什么 | 状态 |
|---|---|---|
| 训练到 5M-10M 步 | 认知能力继续增长还是饱和 | 未做 |
| 3D 环境增加复杂度 | 能否处理更复杂的物理/社交场景 | ThreeDWorld 有限 |
| 自主目标设定 | 内在动机能否替代课程驱动 | 未验证 |
| 长期记忆整合 | 记忆能否作为认知核心 | 未验证 |
| 模块深度整合 | 感知/符号/因果/想象能否真正协同 | 初步整合 |
| 永续稳定性 | EWC+GR 能否支撑数百万步 | 仅 Stage 6 验证 30 天 |

### 建议优先级

| 优先级 | 任务 | 原因 |
|---|---|---|
| 1 | 完成 Stage 18 (组合式成长 + ToM 突破) | 当前最近的里程碑 |
| 2 | Stage 19 自我叙事闭环 (整合现有模块) | 当前架构可扩展, 成本最低, 验证 10-12y 路径 |
| 3 | Stage 20 假设-演绎闭环 (升级 HypothesisTester + kanren) | 当前架构可扩展, 验证形式推理起点 |
| 4 | Stage 21 递归元认知 (新架构设计) | 13-14y 真正的架构分水岭 |
| 5 | Stage 23 开放世界整合 (扩展 3D 世界接入) | 为永续学习铺路, 验证自主目标 |

> 核心判断 (2026-08-11 更新): 架构评估显示 10-12y (Stage 19-20) 可基于当前架构
> 实现 — 自我叙事、假设-演绎的雏形模块已存在, 主要工作是整合闭环。13-14y 开始
> (Stage 21 递归元认知) 才是真正的架构分水岭, 需要设计二级自我模型。
> 在没看到明确饱和信号之前先继续推进 (ToM 还在上升、规则还在增长、est. age 还在提升)。

