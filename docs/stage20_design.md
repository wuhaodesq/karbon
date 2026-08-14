# Stage 20 设计 — 假设-演绎引擎 + 物体追踪专项 (12-13y)

## 1. 定位 / Positioning

Stage 19 收官 (1.3M sealed) 确认:
- ✅ means_ends / intuitive_physics / number_sense 通过
- ❌ object_permanence 真遮挡 = 0.10 (瓶颈) — 引导奖励 (0.3) 因事件稀疏无效
- ❌ ToM 0.09-0.43 / systematic 0.37 (双稳态摆动)

Stage 20 目标 (以终为始):
1. **早期 (≤200K 步) 专项解决 object_permanence** — 把"物体被遮挡后推断位置"实现为
   假设-演绎推理的**原生任务** (不是旁路奖励, 而是推理闭环本身)
2. 升级 HypothesisTester + ActiveExperimenter + kanren 为形式运算级闭环
   (假设 → 实验 → 反馈 → 置信度 → 规则库)
3. 双稳态稳定化 (means_ends 连贯性)

## 2. 核心机制: 遮挡追踪作为假设-演绎闭环

人类儿童 12 岁的形式运算: "物体被墙挡住 → 它没有消失 → 它在 last_known →
我可以走过去验证"。这正是假设-演绎: **假设 + 实验 + 验证**。

```
遮挡事件 (env 检测, 墙挡住物体)
  → HypothesisTester.propose_hypothesis(
        condition=场景嵌入, predicted_action=走向 last_known,
        description="obj_{i} occluded at ({x},{y})")
  → should_probe() → 主动走向 last_known (探针动作)
  → 到达/未到达 → env 反馈 (物体可见 = 验证成功)
  → feedback() 更新置信度
  → 高置信 → kanren 规则 "IF obj_occluded THEN go_last_known"
  → 行为涌现 → 评测 (真遮挡) 通过
```

**与 Stage 19 引导奖励的本质区别**:
| 维度 | Stage 19 (失败) | Stage 20 (本设计) |
|---|---|---|
| 信号 | 稀疏奖励 (0.3) | **推理闭环** (假设-验证) |
| 练习密度 | 2 墙随机, 事件稀疏 | **4 墙 + 物体穿越** (密集) |
| 认知 | 奖励-反应关联 | **假设-演绎推理** (形式运算) |
| 泛化 | 无 | 规则进 kanren, 可泛化 |

## 3. 环境改造 (three_d_world.py)

### 3.1 密集遮挡
- `num_occluders: 2 → 4` (config)
- occluder 半径环 0.6-1.4 保持, 角度均匀分布 (不再完全随机, 保证覆盖)
- **物体穿越**: 每 ~50 步, 随机选一个物体移动到 occluder 另一侧
  (模拟"物体经过墙后"), 产生确定性遮挡事件
  - 实现: `_occ_mover_step` 计数器, 每 50 步 `obj_x = -obj_x` (镜像翻越)
  - 有界: 无新存储

### 3.2 遮挡事件显式化 (供推理闭环消费)
- 遮挡开始事件 → 暴露给 HypothesisTester:
  ```
  env.get_occlusion_signal() -> {
      "active": [(obj_id, last_known_x, last_known_y), ...],
      "just_occluded": bool  # 本 step 新遮挡
  }
  ```
- 遮挡结束 → `just_revealed` (验证反馈源)

### 3.3 保留
- `occluder_target_reward: 0.3` (保留, 密集事件下可学)
- 评测 make_env: 真墙一致 (num_occluders=4), 不传 reward/trace
- occluder_trace: false (痕迹已弃)

## 4. 推理闭环实现 (train.py + HypothesisTester)

### 4.1 遮挡 → 假设 (episode 内, 每 step 检查)
```python
sig = env.get_occlusion_signal()
if sig["just_occluded"] and hypothesis_tester is not None:
    for obj_id, x, y in sig["active"]:
        hypothesis_tester.propose_hypothesis(
            condition_embedding=hidden.detach(),
            predicted_action=_action_toward(x, y),   # 最近动作方向
            description=f"obj_{obj_id} occluded at ({x:.2f},{y:.2f})",
        )
```

### 4.2 探针 (should_probe 命中时)
```python
if hypothesis_tester.should_probe(hidden):
    a = hypothesis_tester.get_probe_action()
    if a is not None:
        action = a  # 覆盖策略动作
```

### 4.3 反馈 (物体重现/到达时)
```python
if sig["just_revealed"] and hypothesis_tester._active_hypothesis_id is not None:
    ok = 1.0 if reached_last_known else 0.0
    hypothesis_tester.feedback(ok)
```

### 4.4 高置信 → kanren 规则 (hook)
```python
# HypothesisTester.feedback 内: confidence > 0.75 -> 回调写入 kanren
# (通过 symbol_backend.add_rule, 需加接口)
```

## 5. 双稳态稳定化

- means_ends 连贯性: 保持 chain-task 奖励 + 引入"连续工具使用"计数
  (task_progress 连续 3+ 步上升给 bonus) — 暂缓, 观察评测

## 6. Stage 20 配置 (stage20_hypothesis_deduction.yaml)

```yaml
preset: cloud_24g
stage: 20
env:
  num_objects: 8
  num_occluders: 4            # 密集遮挡
  occluder_trace: false
  occluder_target_reward: 0.3 # 保留 (密集事件下可学)
  object_crossing_every: 50   # 物体穿越墙周期
  max_episode_steps: 300
advanced:
  hypothesis_tester_enabled: true
  hypothesis_max: 64          # 更多假设容量 (遮挡事件多)
  hypothesis_probe_epsilon: 0.15
train:
  total_steps: 1000000
  ckpt_every_steps: 50000
```

## 7. 评测 (与训练一致)

- make_env: num_occluders=4 (真墙一致), 无 reward/trace
- **新增评测信号**: 真遮挡 op (已有, 用 4 墙配置重测)
- 里程碑阈值: op ≥ 0.6 / ToM ≥ 0.55 / systematic ≥ 0.6
- est. age: 4 里程碑通过 → 12-13y

## 8. 退出标准 (Stage 20 完成)

- [ ] object_permanence (真遮挡) ≥ 0.6 — **早期优先 (200K 内专项)**
- [ ] ToM ≥ 0.55
- [ ] hypothesis_tester: 假设数 > 0, 验证过的规则进入 kanren
- [ ] kanren 规则含遮挡-追踪规则 ("IF occluded THEN track")
- [ ] 全量评测 est. age ≥ 12y

## 9. 风险

| 风险 | 缓解 |
|---|---|
| 探针动作覆盖策略 → 其他能力崩 | probe_epsilon 低 (0.15), 只在假设活跃时探 |
| 物体穿越改动破坏物理 | 穿越 = 镜像位置, 不引入新物理 (mujoco 直接改 body_pos) |
| 4 墙评测更难 (其他指标降) | 训练/评测一致 (都 4 墙), agent 适应墙环境 |
| 假设泛滥 (每次遮挡都 propose) | hypothesis_max=64 有界 + 同物体去重 |

## 10. 里程碑 (本次实现)

1. env: num_occluders=4 + 物体穿越 + occlusion_signal 接口
2. HypothesisTester 接线 (train.py: propose/probe/feedback)
3. kanren 规则 hook (高置信 → 规则)
4. stage20_hypothesis_deduction.yaml
5. 评测: 真遮挡 4 墙配置
6. 早期专项验证: 200K 步后 op 评测
