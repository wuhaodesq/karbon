# Stage 21 Design — 递归元认知 (Recursive Metacognition, 13-14y)

> Status: **design draft** (2026-10-08). Stage 20 sealed
> (`docs/stage20_report.md`); this stage inherits its carried gaps.
> 状态：设计草案。承接 Stage 20 封存时的携带缺口。

---

## 0. Carried gaps from Stage 20 (must be addressed here)

| # | Gap | Stage 21 treatment |
|---|---|---|
| G1 | **systematic_reasoning 诚实度量 ≈ 0** (阶段命名能力; 两轮巩固失败) | M1: 度量+机制重设计 (本阶段的核心科研任务) |
| G2 | 浅振荡残余 (op 0.36-0.62 带, 无深坍缩) | M4: 诊断或显式接受 (记录为动力学特性) |
| G3 | num_sense 单 seed 差 0.0125 | M4: 廉价复验 |

## 1. Stage goal (per TIMELINE)

**递归元认知**: 二级自我模型 — 监控"思考过程"本身（不只是结果好坏）。
**验证口径 (operationalized)**: "知道自己在思考" =
1. **元预测 (meta-prediction)**: 监控自身认知过程流, 对自身推理结果做
   held-out 预测, skill_vs_majority 显著 > 0。
2. **自纠错 (self-contradiction)**: 在自身规则库/信念中检测矛盾, 命中率
   显著高于随机。
3. **审慎控制 (deliberation control)**: 基于二级状态决定"何时多想想"，
   在**等计算预算** A/B 下带来可测收益 (vs 固定预算基线)。

**关键区别 (为什么不是再训练一遍)**: Stage 20 失败的两轮巩固都在预测
**环境**结局 (raw hidden → arrival)。Stage 21 预测的是**自己的认知过程**
(假设/探针/规则/反思的事件流 → 我自己接下来会成功还是失败) ——
输入是过程特征而非感官表征。

## 2. Existing assets (复用而非重造)

| Asset | Location | Role in Stage 21 |
|---|---|---|
| HypothesisTester (propose/probe/verify 事件流) | `src/models/advanced_cognition.py` | **二级监控的事件源** (每事件: 条件、动作、结局、时长) |
| SelfModel (置信/熟悉/进度) | `src/models/metacognition.py` | 一级自我模型 (被监控对象) |
| ReflectionLoop + SelfReflectionValidator + ReflectionRecord | `metacognition.py`, `metacognition_v2.py` | 反思记录→过程特征 |
| ThoughtActionLoop (FiLM) | `thought_action_loop.py` | 思考→决策调制的现有通道 |
| RuleMemory / logic engine / symbol backend | `neural_symbolic.py`, `logic_engine.py` | 自纠错的对象 (规则库) |
| systematic_probe_v2 / formal_reasoning_probe | `scripts/eval/` | 诚实仪器 (扩充 G1 的量) |

## 3. Architecture sketch (bounded, preset-aware)

```
认知事件流 (hypothesis/probe/verify, rule match, reflection, plan)
        │  (有界环形缓冲, capacity 256)
        ▼
二级状态编码器 (MLP: 过程特征 → d_meta=64)
        │
        ├─► 元预测头   p(下一次验证成功 | 过程状态)      [G1 的能力核心]
        ├─► 矛盾检测头 p(当前规则库存在矛盾对)           [自纠错]
        └─► 审慎门控   g(是否加大思考预算)               [deliberation control]
```

- **过程特征 (每事件, 全部可得且廉价)**: 事件类型、假设 id/年龄、探针次数、
  最近 K 次验证成败序列、规则匹配相似度、覆盖率、episode 内时间、一级
  SelfModel 输出。
- **训练信号 (全部自有, 无手标注)**:
  - 元预测: 事件结局的 BCE (下一事件真值), **按事件切分 train/test**;
  - 矛盾检测: 规则库注入/自然涌现的矛盾对 (两规则同前提异结论) 为正例,
    随机对为负例;
  - 审慎门控: 与固定预算的 A/B (同 seed、同步数, 预算重分配)。
- **有界性 (Axioms)**: 所有缓冲声明容量; 头参数 <100K; 不新建 GPU 无界结构。

## 4. Milestones (small-step, each with honest measurable)

| M | Deliverable | Verification (honest) | Est. |
|---|---|---|---|
| **M1** | `meta_monitor.py`: 事件流缓冲 + 元预测头 + 离线评测 (用 Stage 20 的已有事件日志先做离线训练/测试) | held-out meta-prediction **skill_vs_majority > 0.05** (n≥1000) — 判定能力真实存在 | 1-2 天 |
| **M2** | 在线接入: 事件流实时进入 monitor, 元预测 BCE 参与训练 (小系数) | 12-24h 训练后 skill 保持 >0; op/ToM 不回退 | 2-3 天 |
| **M3** | 矛盾检测头 + 自纠错回路 (检出→降权/标注矛盾规则) | 注入矛盾对的检出率 > 0.8, 误报 < 0.1 | 2 天 |
| **M4** | 审慎门控 + G2/G3 收尾 (振荡诊断/接受、num 复验) | 等预算 A/B 收益 > 0; 记录结论 | 2-3 天 |
| **Exit** | 报告 + tag `v0.21.0-stage21` | M1-M4 全绿 | — |

## 5. G1 note — why this measures systematic_reasoning

Stage 20 的诚实缺口本质是: 智能体**没有把"预测→验证"作为自己的认知操作**。
Stage 21 的元预测头正是把这个操作显式化 (monitor 预测→事件验证→反馈),
且其度量 (held-out skill) 与规范科学方法同构 (先验预测, 后验检验)。
若 M1-M2 成功, systematic_reasoning 将以**新定义**达标:
"对自身推理过程的预测准确率显著高于基线"。

## 6. Risks / anti-patterns

- **泄漏**: 过程特征里不得混入结局信息 (事件结局只在监督端)。评测按事件切分。
- **自欺**: 全部指标带 majority_baseline 与 skill; 样本量 n 写进报告。
- **预算**: 所有训练复用 chain 自动门控 (op 不退化即续); 预置三档不变。
- **老路重走**: 若 M1 离线即失败 (skill≤0), 先停手改特征设计, 不上在线。
