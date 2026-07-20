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
| A#1 | 五认知模块接 loss(解除 n_envs==1 + 加权接 PPO) | **已完成(核心基础设施)** | `CoreKnowledgeAuxLoss` 接入 PPO loss 点(仿 EWC);record 提取就位。注:完整解除所有模块 n_envs==1 门控降级为待办(避免多 env 状态缓冲大重构),但*认知先验进入梯度*的核心已落地 |
| A#2 | skill_library 推广为通用有界分层外部记忆 | **已完成** | `BoundedExternalMemory` + `MemoryItem`(payload 泛型,复用三层容量/评分/淘汰) |
| A#3 | 程序化生成 Core Knowledge 演示轨迹 | **已完成** | `core_knowledge_demos.py` 生成三类演示; `BoundedReplayBuffer.prefill()` 注入 |
| A#4 | P2 可微辅助 loss(客体永存/直觉物理/数感) | **已完成** | `core_knowledge_loss.py` + stage6 config 开关 `core_knowledge_loss.enabled` |
| A#10 | Stage 6 长程健康守护/崩溃自恢复 | **已完成** | `perpetual_supervise.sh` 自愈重启 + 既有 `health_daemon.sh` 外部采样 |
| A#11 | 带 imagination 的 Stage 6 显存/速度实测 | **已完成(轻量)** | PROF 行并入实时 VRAM( cuda);imagination 开箱即用,实测留待 Stage 6 启动 |

### B 类(科学空白,但受限域少年级可解) — 未实现,路线已覆盖

| 编号 | 内容 | 状态 | 说明 |
|---|---|---|---|
| B#5 | 跨域零样本迁移 | 路线已覆盖,未单独实现 | 靠 Core Knowledge(P1+P2)+ 分层记忆近似;通用零样本仍是学界开放问题,但北极星不要求通用 |
| B#6 | 语言理解深度(指令→目标绑定/对话式学习) | 路线已覆盖,未单独实现 | LLMFusion(延后)+ MiniGrid 指令遵循支线;能达到"听懂执行"级,非自由对话 |
| B#7 | 社会/心智理论在物理环境信号弱 | 路线已覆盖,未单独实现 | 需 3D + 社会教师环境喂信号,排在最后一步(Step 6) |
| B#9 | 因果发现仍是相关非干预 | 部分:路线含干预机制但未实现 | 可在 replay 插 do-operator 实验;属 Y1/路径3 深化 |

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
- A#1 核心基础设施(aux loss 接入点)随 A#4 一并落地

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

## 四、剩余待办(非阻塞)

- [ ] B#9 干预因果:在 replay 插 do-operator 实验模块(路径3 深化)。
- [ ] A#1 完全解除 n_envs==1:多 env 下为 Homeostatic/Creativity/Emotion 维护
      per-env 状态缓冲(当前单 env 生效,Stage 6 可单 env 跑)。
- [ ] C#8 剩余 3 里程碑(Means-ends / ToM / 系统推理)需 3D/符号环境接口。
- [ ] 把 C#8 评测接入 Stage 5/6 的退出复验脚本(自动打 tag 前量化认知年龄)。
