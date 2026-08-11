# Stage 19 Design - 自我叙事引擎 (Self-Narrative Engine)

> **目标**: 10-12 岁认知水平 -- 连贯自我叙事 + 叙事影响行为
>
> **Target**: 10-12y cognitive level -- coherent self-narrative that
> modulates behavior, not just logged text.
>
> **架构基础**: 当前架构可扩展 (TIMELINE.md 评估); 三个模块已有雏形
> (IdentityNarrative / AutobiographicalMemory / InnerDialogue), 缺整合闭环。
>
> **前置 4/4 修复**: 全部完成 (见 stage18_report.md §7.1) -- 反思维度、
> except->log、SelfModel 训练、kanren 查询。

---

## 1. 问题定义

Stage 18 证明: 行为层 (means_ends/physics) 已固化, 但认知/符号层完全空转
(reflection=0 条, kanren=0 查询, SelfModel 未训练)。est. age=0.0y。

Stage 19 的核心命题: **让叙事真正影响行为**。不是多一个日志行, 而是让
智能体的"自我认知"通过 FiLM 调制 + 符号偏置改变它的动作选择。

闭环:
```
经验 -> 自传记忆 -> 身份叙事 -> 反思 -> 内心独白
                                          |
                              CLIP 编码 -> FiLM 调制视觉特征
                                          |
                              kanren 预测 -> 动作 logit 偏置
                                          |
                                        动作 -> 经验 (下一轮)
```

---

## 2. 现有组件盘点

| 组件 | 位置 | 成熟度 | 缺口 |
|---|---|---|---|
| IdentityNarrative | `abstract_reasoning.py:266` | 关键词 trait + 模板叙事; trait_projector MLP 未用 | 每 50K 步只打日志, 不回策略; trait 提取未学习 |
| AutobiographicalMemory | `developmental_memory.py:377` | 可用: 存储/晋升/驱逐/查询 LifeEvent | 不实时喂 IdentityNarrative; 缺"重要性"学习 |
| InnerDialogue | `metacognition.py:371` | 可用: template/LLM 双模式 | 生成的 lessons 不回行为 |
| ThoughtActionLoop | `thought_action_loop.py:52` | **闭环模板**: FiLM 调制 | 不含叙事层; 需扩展 |
| SymbolBackend | `symbol_backend.py:49` | 可用: predict_action + feedback | 已接查询 (Stage 18 修), 缺动作 logit 注入 |
| ReflectionLoop | `metacognition.py:226` | 可用 (Bug E 已修) | 反思内容不喂叙事 |
| SelfModel | `metacognition.py:80` | 可用 (auxiliary_loss 已接) | 输出不喂叙事 |

---

## 3. 架构设计

### 3.1 NarrativeLoopController (新模块)

**文件**: `src/models/narrative_loop.py`

**角色**: 扩展 ThoughtActionLoop, 在反思链上增加自传记忆 + 身份叙事。

**数据流**:
```
每步:
  1. ThoughtActionLoop.maybe_think(hidden, ep_ret, ep_done)  # FiLM 调制
  2. NarrativeLoopController.maybe_narrate(ep_done)           # 叙事周期

每 episode 结束:
  A. AutobiographicalMemory.add_event(step, description, importance, ep_id, lesson)
  B. ReflectionLoop.end_episode(ep_ret) -> reflection
  C. 每 N episodes:
     - IdentityNarrative(life_events) -> traits + narrative_text
     - InnerDialogue.generate(reflection + narrative_text) -> lessons
     - lessons -> CLIP encode -> ThoughtActionLoop 缓存 FiLM 调制
  D. kanren.predict_action(predicates) -> symbol_action
     -> NarrativeLoopController._cached_symbol_bias (action logit 偏置)
```

**类签名**:
```python
class NarrativeLoopController(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        thought_loop: ThoughtActionLoop | None = None,
        autobiographical: AutobiographicalMemory | None = None,
        identity_narrative: IdentityNarrative | None = None,
        symbol_backend: SymbolBackend | None = None,
        narrative_every_episodes: int = 10,
        symbol_bias_weight: float = 0.1,
    ): ...

    def step_hook(self, hidden_state, ep_ret, ep_done) -> str | None:
        """每步调用: 委托 thought_loop + 检查叙事周期."""

    def get_symbol_bias(self) -> torch.Tensor | None:
        """返回当前的动作 logit 偏置 (num_actions,) 或 None."""

    def episode_end_hook(self, step, ep_ret, ep_id, description, lesson):
        """Episode 结束: 存记忆 + 触发叙事."""

    def modulate(self, vision_feats) -> torch.Tensor:
        """委托 ThoughtActionLoop.modulate (FiLM)."""
```

**有界保证**: 
- AutobiographicalMemory 已有 max_events (100)
- IdentityNarrative capacity=1 (单叙事)
- 缓存: 1 个 FiLM embedding (d_model,) + 1 个 symbol_bias (num_actions,)
- 无增长, Axiom 1 满足

### 3.2 Symbol-to-Action Bias (新)

在 `HierarchicalActorCritic.forward()` 中, worker 的 action logits 计算后,
加入 symbol 偏置:

```python
# In HierarchicalActorCritic.forward(), after worker logits:
if self._narrative_loop is not None:
    symbol_bias = self._narrative_loop.get_symbol_bias()
    if symbol_bias is not None:
        logits = logits + symbol_bias  # (B, num_actions)
```

**梯度**: symbol_bias 是 detach 的 (kanren 不可微), 不影响 PPO 梯度。
这符合 SymbolBackend 的 REINFORCE 设计 (learning-back, 非 end-to-end)。

### 3.3 IdentityNarrative 增强: 学习式 trait 提取

当前 `extract_traits` 用关键词匹配。Stage 19 启用已有的 `trait_projector` MLP:

```python
# 在 extract_traits 中, 当有 hidden_states 时:
if hidden_states is not None:
    traits = self.trait_projector(hidden_states.mean(dim=0))  # (5,)
    traits = torch.sigmoid(traits)  # -> [0, 1]^5
```

训练信号: traits 的 auxiliary loss (与 SelfModel 类似, episode-end 监督)。
Targets: 从 episode 行为统计推算 (探索次数 -> openness, 成功率 -> conscientiousness)。

### 3.4 True Occlusion Probe (环境修改)

在 `ThreeDWorld` 中增加遮挡墙几何体:

```python
# 在 _build_scene 中, 随机放置 1-2 面 occluder walls
# 在 _track_3d_developmental_signals 中, 真正检查视线遮挡:
#   agent -> occluder -> object: 如果 occluder 挡住, 记录 occlusion event
#   事件结构: {last_known, agent_traj_during_occ, truly_occluded: True}
```

这让 object_permanence 评测从"距离事件"变为"真遮挡事件"。

### 3.5 Curriculum 调整: 打破 task-3 垄断

```yaml
curriculum:
  mode: sequential
  switch_every_steps: 20000       # 50K -> 20K (更快轮转)
  exploration_epsilon: 0.3        # 保持
  interleave_review: true          # 新: 每 3 个 task 切换插入 1 个 review task
```

---

## 4. 配置 (stage19_self_narrative.yaml)

```yaml
stage: 19
name: "stage19_self_narrative"

model:
  hidden_size: 128
  use_hierarchical: true
  # ... (继承 Stage 18)

# --- Stage 19: Narrative Loop ---
narrative:
  enabled: true
  narrative_every_episodes: 10     # 每 10 episodes 生成一次叙事
  symbol_bias_weight: 0.1          # kanren 动作偏置权重
  trait_learning_enabled: true     # 启用学习式 trait 提取

cognitive:
  symbolic_enabled: true
  self_model_enabled: true
  self_model_d_model: 128          # = hidden_size (Bug E 修复)
  self_model_lr: 1.0e-4
  logic_engine_enabled: true
  reflection_enabled: true
  reflection_max: 256
  reflection_every_episodes: 10    # 与叙事同步

# --- Env: add occluders ---
env:
  id: "ThreeDWorld"
  num_objects: 8
  max_episode_steps: 300
  render_size: 128
  action_force: 50.0
  developmental_age: 0.5
  num_occluders: 2                 # 新: 2 面遮挡墙

# --- Curriculum: faster rotation ---
curriculum:
  max_tasks: 5
  mode: sequential
  switch_every_steps: 20000        # 50K -> 20K
  exploration_epsilon: 0.3
  tasks:
    - {id: 0, num_objects: 3,  action_force: 70.0, difficulty: 0.1, tag: "3d-sparse"}
    - {id: 1, num_objects: 5,  action_force: 60.0, difficulty: 0.3, tag: "3d-few"}
    - {id: 2, num_objects: 8,  action_force: 50.0, difficulty: 0.5, tag: "3d-base"}
    - {id: 3, num_objects: 12, action_force: 45.0, difficulty: 0.7, tag: "3d-many"}
    - {id: 4, num_objects: 16, action_force: 40.0, difficulty: 0.9, tag: "3d-crowded"}

# ... (其余继承 Stage 18: continual, replay, imagination, etc.)
```

---

## 5. 训练计划

| 项目 | 值 |
|---|---|
| Resume from | `ckpt_stage18_001000000.pt` |
| Total steps | 1,000,000 (第一阶段) |
| 预期 VRAM | ~7 GB (与 Stage 18 相同, NarrativeLoop 很小) |
| 关键里程碑检查 | 每 50K 步跑 v2 评测 |
| 新日志行 | `[narrative]`, `[symbol_bias]`, `[identity]` |

**分阶段验证**:
1. 0-50K: 确认 reflection 非空 (Bug E 修复生效)、kanren queries > 0
2. 50K-200K: 确认 IdentityNarrative 产出非中性 traits (>0.5 方差)
3. 200K-500K: 确认 symbol_bias 影响动作分布 (action entropy 下降)
4. 500K-1M: v2 评测 est. age > 0.0y (object_permanence 突破 0.6)

---

## 6. 评测标准

### 6.1 量化 (v2 评测脚本)

| 里程碑 | Stage 18 基线 | Stage 19 目标 |
|---|---|---|
| object_permanence | 0.524 | > 0.6 (真遮挡探针) |
| means_ends | 1.0 | 维持 |
| intuitive_physics | 1.0 | 维持 |
| number_sense | 0.550 | > 0.6 |
| theory_of_mind | 0.443 | > 0.5 |
| systematic_reasoning | 0.374 | > 0.5 |
| **estimated_age** | **0.0y** | **> 3.5y** |

### 6.2 叙事质量 (新指标)

1. **Trait 稳定性**: 连续 5 次叙事的 trait 向量方差 < 0.1 (身份收敛)
2. **Lesson-action 相关性**: InnerDialogue 生成的 lesson 与下一 episode 的动作变化方向一致 (>60%)
3. **Symbol accuracy**: kanren predict_action 正确率 > 30% (优于随机 1/12=8.3%)
4. **Reflection 非空**: reflections 列表非空 (Bug E 修复验证)

### 6.3 退出标准

- [ ] est. age > 3.5y (object_permanence + number_sense 通过)
- [ ] reflection 非空, episode_count 与 reflections 数量一致
- [ ] kanren queries > 1000, accuracy > 30%
- [ ] IdentityNarrative traits 方差 > 0.01 (非全 0.5)
- [ ] 30 天不间断训练稳定 (无 OOM / 崩溃)
- [ ] v2 评测脚本通过, make test + make check-bounds 干净

---

## 7. 实现步骤

| 步骤 | 内容 | 文件 |
|---|---|---|
| 1 | 创建 `NarrativeLoopController` | `src/models/narrative_loop.py` |
| 2 | 在 `HierarchicalActorCritic` 中接入 symbol bias | `src/models/hierarchical_policy.py` |
| 3 | IdentityNarrative: 启用 trait_projector + auxiliary loss | `src/models/abstract_reasoning.py` |
| 4 | ThreeDWorld: 添加 occluder 几何 + 真遮挡检测 | `src/envs/three_d_world.py` |
| 5 | 创建 `stage19_self_narrative.yaml` | `configs/` |
| 6 | train.py: 接入 NarrativeLoopController | `src/train.py` |
| 7 | 测试: `tests/test_narrative_loop.py` | `tests/` |
| 8 | 推送云端 + 启动训练 | 云端 |
| 9 | 50K 步验证: reflection 非空 + kanren queries > 0 | - |
| 10 | 1M 步全量评测 + stage19_report.md | - |

---

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| FiLM 调制破坏已固化的 means_ends/physics | symbol_bias_weight=0.1 起步; 50K 步检查行为分数不降 |
| IdentityNarrative trait 学习信号太弱 | 用 episode 行为统计 (探索率/成功率) 做硬 target |
| occluder 几何影响 MuJoCo 物理 | 用 kinematic body (不参与动力学, 只遮挡视线) |
| 叙事周期 (每 10 episodes) 太慢 | 可调; 先保守后加速 |
| kanren predict_action 全错 (随机规则) | accuracy 从随机 8.3% 开始, feedback 逐步修正 |
