# Stage 22 Design — 抽象概念框架 (Abstract Concepts, 14-15y)

> Status: **design draft** (2026-10-10). Stage 21 M1-M3 done, M4 honest
> negative recorded; seal pending the soak5 long-run data.
> 状态：设计草案（承接 Stage 21 的携带缺口）。

---

## 0. Carried context

| 来源 | 内容 |
|---|---|
| Stage 21 | 过程级元监控 (meta skill +0.11~0.27); 矛盾自纠错; 审慎门控 (择时无独立价值, 如实记录) |
| G1 残余 | systematic 的"环境结局预测"仍≈随机 (过程级预测已成立 — 这是重要区分) |
| G2 | op 浅振荡 (0.25-0.74) 已被彻底刻画 (value/规则/物理三类根因已修; 残余为内在动力学) |
| G3 | num_sense 单 seed 差 0.0125 (随封存段复验) |
| 语言 | `lang=False` — 14-15 岁的跨概念推理大概率需要语言锚点 (本阶段评估激活时机) |

## 1. Stage goal (per TIMELINE)

**抽象概念框架**: 具体→抽象推理 (14-15y)。
**验证口径 (operationalized)**:
1. **层级抽象**: 从实例/类别构造有界概念层级; 未见实例可经层级归类 (novel-instance
   classification via hierarchy), 显著高于平铺基线。
2. **跨概念形式推理**: 以概念为量词域的规则 (如 "所有 C 类物体具备 P") 可应用于
   新实例并支持多跳链 — 复用 Stage 20 已打通的 Horn 演绎引擎。
3. **类比/隐喻**: 关系迁移 (A:B :: C:?) 在 held-out 关系上显著高于随机。

## 2. Existing assets (复用而非重造)

| Asset | Location | Role |
|---|---|---|
| ConceptGraph (nodes/edges) | `src/models/concept_graph.py` | 层级化的基底 (需加层级结构) |
| ConceptClusterer (类别发现) | `src/models/metacognition_v2.py` | 生成候选类别 |
| Analogizer (隐喻/关系) | `src/models/tier2_cognitive.py` | 类比雏形 |
| VisualAnalyzer (属性分类) | `src/models/visual_analyzer.py` | 实例属性 → 概念谓词 |
| 演绎引擎 (多跳前向链) | `symbol_backend.py` `_derive_facts` | 跨概念推理的执行器 |
| MetaMonitor (过程状态) | `src/models/meta_monitor.py` | 抽象阶段的元监控复用 |
| 形式探针 | `scripts/eval/formal_reasoning_probe.py` | 扩为概念级批次 |

## 3. Architecture sketch (bounded)

```
实例 (slot 属性: 形状/颜色/大小/运动模式)
   │  VisualAnalyzer + ConceptClusterer
   ▼
概念节点 (有界 ≤256): 属性签名 + 原型嵌入
   │  层级构建: 自底向上合并 (类别→超类), 深度 ≤3
   ▼
概念层级 (instance → category → super-category)
   │
   ├─► 抽象映射: 新实例 → 最近原型链 (层级归类头)
   ├─► 概念规则: LogicEngine 变量域扩为"概念量词"
   │     (∀x∈C: P(x)) → 新实例经多跳链推导 P
   └─► 类比: 关系三元组 (A,B,R) 的嵌入迁移 (Analogizer 升级)
```

- **有界 (Axioms)**: 概念 ≤256、层级深度 ≤3、每节点属性签名长度上限;
  规则库沿用既有上限。
- **训练信号 (自监督为主)**:
  - 层级: 对比学习 (同类别实例嵌入拉近) + 聚类稳定性;
  - 概念规则: 由 Stage 20 的假设-验证回路产生 (规则命中→验证→反馈, 已接线);
  - 类比: held-out 关系三元组预测 (自生成)。
- **语言锚点评估**: 用冻结的本地文本编码器 (LLMFusion 的 TinyTextEncoder
  先例) 给概念节点挂"可读标签"仅作评测/诊断, 不参与策略 — 语言策略互作用的
  正式评估放 Stage 23 (开放世界) 再定。

## 4. Milestones (small-step, honest)

| M | Deliverable | Verification (honest) | Est. |
|---|---|---|---|
| **M1** | `concept_hierarchy.py`: 有界层级构建 + 层级归类头; 离线评测 (用现有 ConceptGraph 数据 + VisualAnalyzer 输出) | novel-instance 归类 skill vs 平铺基线 > 0.05 (held-out 实例) | 1-2 天 |
| **M2** | 概念规则接入: LogicEngine 量词域扩展 + 假设回路产出概念级规则 | 概念级多跳推理批次 (探针扩条) 通过率 vs 实例级基线提升 | 2-3 天 |
| **M3** | 类比升级 (关系迁移) | held-out 关系三元组准确率显著 > 随机 (+0.1) | 2 天 |
| **M4** | 训练集成 + G3 复验 + 长程浸泡 | 训练中概念指标不退化; num_sense 复验过门; op 不崩 | 2-3 天 |
| **Exit** | 报告 + tag `v0.22.0-stage22` | M1-M4 全绿 | — |

## 5. Risks / honesty guards

- **"抽象"的空心化风险**: 概念节点若只是嵌入簇的重命名, 无推理价值 — 因此
  验证必须含**行为化应用** (跨概念推理批次), 而非只有聚类指标。
- **泄漏**: 类比/归类评测按实例切分; 关系迁移用 held-out 关系类型。
- **规模**: 全部复用有界范式; 不引入无界结构。
- **与 Stage 21 的关系**: meta monitor 继续运行 (过程级预测是抽象推理的自监控面)。
