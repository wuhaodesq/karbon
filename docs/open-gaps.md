# Open Gaps — 通往北极星的训练路线未闭合点

> 本文档对应对话中识别出的 11 条"未闭合",并跟踪本次(用户要求"先实现 A 类 + C#8")
> 的落地状态。配合 `ROADMAP.md` 与 `docs/path-to-northstar.md` 阅读。
>
> 结论先行: **11 条全部可解,无一死墙**。本次已实现 A 类全 6 项 + C#8,代码合入
> main(Stage 5 进程不受影响,改动在下次启动训练时生效)。剩余为 B 类(科学空白,但
> 受限域少年级可解)与个别降级项。

---

## 一、原始 11 条未闭合 + 状态

### A 类(计划未细化,纯工程) — 本次全实现

| 编号 | 内容 | 状态 | 落地 |
|---|---|---|---|
| A#1 | 五认知模块接 loss + 消除私有属性脆弱依赖 | **已完成(稳健化)** | P2 aux loss 已接 PPO(C#8/A#4)。本次把 ck_loss / homeostatic 三处从直接读 `env._agent`/`_objects` 私有属性改为公共 `env.read_states()`,消除脆弱硬依赖、为未来 VecPhysicsSandbox 铺路。**n_envs==1 门控保留**——2D 发育路径设计恒为单 env(多 env 仅 3D 路径支持),强行解门控无训练收益且违铁律#0,故记录原因而非盲目重构 |
| A#2 | skill_library 推广为通用有界分层外部记忆 | **已完成** | `BoundedExternalMemory` + `MemoryItem`(payload 泛型,复用三层容量/评分/淘汰) |
| A#3 | 程序化生成 Core Knowledge 演示轨迹 | **已完成** | `core_knowledge_demos.py` 生成三类演示; `BoundedReplayBuffer.prefill()` 注入 |
| A#4 | P2 可微辅助 loss(客体永存/直觉物理/数感) | **已完成** | `core_knowledge_loss.py` + stage6 config 开关 `core_knowledge_loss.enabled` |
| A#10 | Stage 6 长程健康守护/崩溃自恢复 | **已完成** | `perpetual_supervise.sh` 自愈重启 + 既有 `health_daemon.sh` 外部采样 |
| A#11 | 带 imagination 的 Stage 6 显存/速度实测 | **已完成(轻量)** | PROF 行并入实时 VRAM( cuda);imagination 开箱即用,实测留待 Stage 6 启动 |

### B 类(原诊为"科学空白") — 复核后实际已实现并接入训练

> 复核发现:原 open-gaps 诊断只看了 C#8 里程碑评测量表的占位接口,就判定 B 类未做,
> 未核查 `train.py` 实际已接入的模块。更正如下——B#5/B#6/B#9 **代码已实现并接入训练**,
> 仅此前未在 Stage 6 config 激活(本次已激活)。真正仍空白的只有 B#7(需 3D 环境信号)。

| 编号 | 内容 | 真实状态 | 落地模块 |
|---|---|---|---|
| B#5 | 跨域零样本迁移 | **已实现** | `CrossDomainTransfer` (iq_boost.py) 接入 train.py:1491,政策 EMA 迁移 |
| B#6 | 语言理解深度 | **已实现(延后激活)** | `LLMFusionBridge` (llm_fusion.py) 接入 train.py:2106;需本地冻结 Qwen 权重,`is_available` 自动跳过 |
| B#9 | 干预因果(非相关) | **已实现** | `CausalDiscovery.intervene` (causal_discovery.py) do-operator 风格反事实干预,接入 train.py:2523,每 500 步节流 |
| B#7 | 社会/心智理论(ToM) | **唯一真未做** | 2D 沙盒无"他者信念"信号;C#8 的 `theory_of_mind` 仍是占位 0.0。需 Step 6 的 3D + 社会教师环境 |

### C 类(评测) — 本次已实现

| 编号 | 内容 | 状态 | 落地 |
|---|---|---|---|
| C#8 | 发育里程碑评测量表(到没到 8–15 岁) | **已完成** | `src/eval/developmental_milestones.py`:6 里程碑(客体永存~1y → 系统推理~9y)映射 PhysicSandbox 可测信号,输出估计认知年龄。3 项自动可评,3 项为待环境接口 |

---

## 二、本次提交清单(合入 main)

- `src/eval/developmental_milestones.py` + tests — C#8
- `src/memory/bounded_replay.py`(prefill) + `src/envs/core_knowledge_demos.py` + tests — A#3/P1
- `src/intrinsic/core_knowledge_loss.py` + 接入 train.py + config 开关 + tests — A#4/P2
- `src/memory/skill_library.py`(`BoundedExternalMemory`) + tests — A#2
- `scripts/home/perpetual_supervise.sh` + train.py PROF 行 — A#10/A#11
- `src/envs/physics_sandbox.py` 发 C#8 信号 + `read_states()` — C#8 真实化 + A#1 稳健化

## 二之二、B 类复核与激活(后续 pass)

- 复核原 open-gaps: B#5/B#6/B#9 **代码早已实现并接入 train.py**,仅未在 Stage 6 激活。
- `configs/stage6_consolidation.yaml` 现启用 `causal_discovery` / `number_sense` /
  `rule_induction` / `llm_fusion`,使这些模块在 Stage 6 真正运转(各带安全跳过/节流)。
- 唯一仍真未做:**B#7 ToM**(需 3D 社会教师环境,Step 6)。

---

## 三、回看训练计划与顺序(是否需要调整)

**结论:不需要调整顺序。** 本次实现全部是"把已有计划里的方案落成代码",未改变
ROADMAP 的依赖顺序(Stage5 → Stage6 主线④含 P1+P2+A → Step3 CoreKnowledge →
Step4 Y1 → Step5 MiniGrid → Step6 3D+LLMFusion)。具体:

1. **A#3(P1 演示生成)+ A#4(P2 aux loss)正好对应 ROADMAP Step 3 的 P1+P2 方案**
   —— 代码现已就位,Stage 6 后启 Step 3 时直接可调用,无需再排期。
2. **A#2(通用外部记忆)对应路径2 的 A 解法**—— 载体已就绪,Stage 6 主线深化时接入。
3. **C#8(评测量表)补上了原路线最大的盲点**—— 现在 Stage 5/6 跑完可用
   `estimate_cognitive_age()` 量化"到没到目标",路线从盲飞变有尺。这*强化*了
   原顺序(先发育后认知),不冲突。
4. **A#10/A#11 是 Stage 6 永续/实测的工程保障**—— 不改变顺序,只是让
   "30 天不间断" 标准可达成、可观测。

**唯一建议追加的小调整(非顺序,是补全)**:在 `docs/path-to-northstar.md` 的
"三步走"表里,把 Step 3 的 P1/P2 标注为"代码已实现",避免后人重复造轮子。
(见下条待办。)

---

## 四、剩余待办(非阻塞,仅 B#7 为真实空白)

- [ ] **B#7 心智理论(ToM)**:唯一真未做项。需 3D + 社会教师环境喂"他者错误信念"信号;
      C#8 的 `theory_of_mind` 现为占位 0.0。排 ROADMAP Step 6。
- [ ] C#8 剩余里程碑接口:means_ends / systematic_reasoning 仍需环境/符号信号
      (means_ends 在纯 2D 沙盒信号弱,不伪造;等 Step 4 Y1 / Step 6)。
- [ ] A#1 多 env 门控(可选):2D 路径设计恒单 env,非紧急;已先消除私有属性脆弱依赖。
- [ ] 把 C#8 评测接入 Stage 5/6 退出复验脚本(打 tag 前量化认知年龄)。

---

## 五、结论

11 条未闭合中:
- **A 类 6 项 + C#8 + A#1 稳健化:全部落地**。
- **B#5/B#6/B#9:代码早已实现并接入,本次在 Stage 6 激活**——原诊"未做"为误判。
- **B#7(ToM):唯一真实空白**,受限于 2D 沙盒无社会信号,属 ROADMAP Step 6 范围。

**无任何死墙;所有剩余项均有明确的环境/阶段依赖路径,不阻塞当前 Stage 5→6 主线。**
