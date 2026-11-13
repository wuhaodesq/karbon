# Stage 27 M1 — 目标级执行层 (Executive Goal Layer) 设计（预注册）

**状态**: 已实现 + 烟测通过 → 三臂实验运行中 (B 臂 2026-11-13 07:30 起)
**日期**: 2026-11-13
**前提**: Stage 26 R1/R2 完成；M2b (goal conditioning) 负结果；M-L1 负结果；
P-A 负结果（序列信用分配是瓶颈）；OP 卡在 0.20-0.30 (trained world)。
**目标**: 从"反应式策略"升级为"有目标策略"——执行功能 (executive function)：
工作记忆 + 注意选择 + 目标维持 + 完成切换。此为 3-6 岁核心发育里程碑。

---

## 1. 动机与归因

### 1.1 为什么是"目标级执行"而不是继续调奖励

修复尝试的历史：

| 尝试 | 结论 | 教训 |
|------|------|------|
| M2b goal conditioning | 负（chosen 38.2% vs control 40.6%，op 崩塌） | 目标向量若无执行路径 = 噪声偏置 |
| M-L1 指令通道 | 负（win_rate 0.53/0.47 ≈ 随机） | 无结算窗口信号，仪器迭代 4 次仍无区分 |
| R3-b object slots | 负（proj_norm 增长，OP 0.252→0.186） | 无执行路径的输入通道会吸收梯度 |
| P-A 链任务 | 负（door 0.05→0.10，grasp 0.85→0.60） | ~200 步链的信用分配是架构瓶颈 |

**共同点**: 所有失败的通道都只做"条件偏置"，没有"选择—执行—结算"的因果直连。

### 1.2 B10 窗口假设：已检验，被证伪（2026-11-13）

怀疑 `occluder_reach_hold_max_steps=15` (B10) 截断了远处事件的奖励信号。
**代码验证结论: 不成立**。`_occluder_only_reward` 中远处接近的稠密项
（`occluder_target_reward=0.3` 距离 progress、`occluder_shaping_weight=0.3`
方向点积、`occluder_orient_weight=0.1`）按事件生命周期支付，**不受 15 步
cap 限制**——cap 只限圈内驻留（防 camp 刷分）。`far_start` 门只有
`occluder_search_min_start=1.2m`，近乎全部事件通过。

**因此 OP 卡点不是奖励截断，而是行为层：多事件场中没有目标承诺
(commitment)——策略在事件间摇摆/漂移，远处事件追不上。** 本设计直接针对此点。

### 1.3 发育对应（3-6 岁 executive function）

```
工作记忆 (working memory):   有界目标缓冲 K=3（与观测槽同容量）
注意选择 (attention):        从自身感知事件中学习选哪个目标（PPO 学习）
目标维持 (maintenance):      承诺窗口 = 事件生命周期，结算前不换
完成切换 (switching):        结算 → 一次性兑现 → 释放
```
关键区别于 M2b：目标是**可直接执行的对象**（槽位位置），选择有因果路径；
执行塑形指向被选目标，结算奖励在结算时刻到达，信用沿 GAE 回传选择步。

---

## 2. 机制（全部有界）

### 2.1 候选集 (bounded K=3 + none)

每步从观测槽构造（与 `_proprio` 同一排序：最近优先）：

- 候选 0..2: 活跃遮挡事件的 `last_known`（dx/4, dy/4, dist/4 已在观测中）
- 候选 3: `none`（不承诺任何目标）

**无新观测输入**——候选就是策略已经看到的槽位信号。选择头的输入：
`[h (trunk hidden), slot_feats (3×3), valid_mask (3)]`，输出 4 logits。

### 2.2 选择头 `GoalExecutive`（src/models/goal_executive.py）

- MLP: `hidden → 64 → 4`，尾层零初始化（初始 ≈ 均匀探索）
- 每步采样（Categorical），logp 与动作 logp **联合**进入 PPO 损失：
  `logp_joint = logp_action + logp_exec`
- 采样-行为一致性：每步的采样即为行为选择（无 hysteresis 硬规则；
  稳定性由学习与 GAE 决定——抖动会被优势惩罚）

### 2.3 环境侧承诺状态机 (src/envs/three_d_world.py)

- `set_exec_goal(idx: int)`：trainer 每步调用（默认 -1 = 关闭）。
  env 用**当步**活跃集合的距离排序把 idx 映射为事件 key `_exec_goal_key`
  （与观测槽同一排序，避免槽序漂移歧义）。
- idx == K (none) 或槽空 → `_exec_goal_key = None`。

### 2.4 执行塑形（新增，默认权重 0）

`_occluder_only_reward` 中追加：若 `_exec_goal_key` 仍活跃，
`r += max(0, dot(v_toward_lk)) * exec_goal_commit_weight`（每步有界，
不对称：只奖励朝向，不惩罚背离——沿用 L1 教训）。

### 2.5 结算奖励（一次性，B10 安全）

`_maybe_reveal_bonus(key)` 是唯一结算点（到达或揭示时调用，随后 pop）。
钩子：`key == _exec_goal_key` 且通过 `_approach_ok` 时，
`_reveal_bonus_pending = max(pending, exec_goal_settle_bonus)`，
并清空 `_exec_goal_key`。一次性，无驻留项 → 不可能产生 camp 策略。
（pending 是单槽 max 语义：同一步揭示归因与结算奖励取较大者一次兑现。）

### 2.6 B10 / 有界性核对

| 项 | 界 |
|----|----|
| 候选数 K | 3（= `occluder_obs_slots`，槽接口限制） |
| 承诺窗口 | 事件生命周期（事件本身有 deadline，`_arrival_deadline_steps`） |
| 结算奖励 | 一次性 per 选择 |
| 塑形 | 每步 ≤ weight（0.15） |
| buffer | `GoalExecutive.capacity = 3`、`__len__` 供 HealthChecker |

---

## 3. 配置（`env:` 新键，默认关 = 与现状逐位一致）

```yaml
exec_goal_enabled: false          # 总开关（模型头 + env 侧逻辑）
exec_goal_commit_weight: 0.15     # 执行塑形权重
exec_goal_settle_bonus: 10.0      # 结算一次性奖励
```

---

## 4. PPO 接入点（明确的编辑面，A15 纪律）

1. `src/train.py` RolloutBuffer: 新增 `exec_logp`、`exec_action` 两列
2. 采样循环 (≈L3130): `sel = model.exec_sample(hidden)`；`env.set_exec_goal(sel)`
3. 存储 (≈L3834/L3858): 写入两列（禁用时 0）
4. PPO 更新 (≈L4706): 重算 `exec_logp_new`；**联合** ratio
5. 熵: `exec_entropy × entropy_coef`（可选，flag 控制）
6. 评测脚本: `run_stage18_full_eval.py`、`op_failure_diag.py` 增加
   "argmax 选择 + set_exec_goal"（train/eval 同构；其它电池不测 OP，标注即可）
7. checkpoint: 新头进 state_dict（旧 ckpt 加载时缺失键用零初始化默认——与本仓库
   既有可选模块同一路径，需在上载前验证 `--resume` 兼容）

---

## 5. 预注册判据 (Pre-registration)

**三臂对比**（同一固定起点 checkpoint；修订案 2026-11-13，开工前, 零数据）：

| 臂 | 机制 | 选择头 | 成本 |
|----|------|--------|------|
| A (baseline) | 关 | — | 0（复用主链同起点窗口的 full eval：机制关时代码路径逐位一致） |
| B (control) | 开 | 冻结均匀随机（`exec_goal_freeze: true`，仍采样不训练） | 1.2M 微调 |
| C (treatment) | 开 | PPO 训练 | 1.2M 微调 |

**预算修订（开工前）**: 每臂 3M → **1.2M 步**（选择头仅 ~4.7K 参数，
稠密目标级奖励；参考 M-L1 教训：大预算低信息量）。评估点
0.4M / 0.8M / 1.2M 各一次 full eval + op_failure_diag；B/C 顺序执行，
与主链共享 GPU（不中断发育主线；步数计量不受并发影响）。

**成功 (C 专属)**:
- 主指标: trained-world OP ≥ 0.35（连续 3 次 full eval），1.2M 步内
- 必须同时: 小世界 OP ≥ 0.5（技能保留）；physics 相对回归 ≥ 0.85；
  est-age 规则不变（OP 未到 0.5 不开龄）
- 选择头区分度: 实际选择分布显著偏离均匀（且与事件距离/年龄相关）

**归因判定**:
- B ≈ A 且 C > A → 学习的选择是杠杆（M1 成立）
- B > A 且 C ≈ B → 塑形是杠杆，选择学习未兑现（能力声明降级，诚实记录）
- C ≈ A → 负结果，关闭 `exec_goal_enabled`，记录负结果与归因

**Kill 条件**: 700K 步时 C 的 OP 无提升（vs A 同窗口 < +0.03）且
选择分布 ≈ 均匀 → 停止 C（B 继续到 1.2M 作为对照记录）。

**诚实边界**:
- 能力声明 = "执行级目标选择与维持"（executive goal selection/maintenance），
  **不是**深思熟虑规划 (deliberation)
- chain 任务（key_door 阶段目标）→ M2，不在本里程碑
- 数感只补注意半边；渲染分辨率问题仍独立

---

## 6. 风险与回滚

- **风险 1**: 联合 logp 改变 PPO 比率动态 → 先跑 A/B 短烟测（200K 步，
  approx_kl、entropy、op 指标无异常）再上 3M
- **风险 2**: 槽序漂移导致映射歧义 → 映射只在 `set_exec_goal` 调用瞬间做，
  之后 key 持久（不随槽序变）
- **风险 3**: 头输出全 none → 结算分布监测；若 none > 80% 持续 500K 步，
  加均匀探索 warmup（记录为仪器修订）
- **回滚**: `exec_goal_enabled: false` 即完全恢复（无状态污染；新头只是旁路）

---

## 7. 复现命令

```bash
# 单元测试
pytest tests/test_goal_executive.py tests/test_exec_goal_env.py -q
# 短烟测（服务器）
python -m src.train --config stage20_hypothesis_v2.yaml --preset cloud_24g \
    --resume .../ckpt_stage20_latest.pt --steps 200000
# 评测
python scripts/eval/run_stage18_full_eval.py --ckpt ... --config ... --preset cloud_24g
python scripts/eval/op_failure_diag.py --ckpt ... --episodes 12 --env-type extended
```

---

## 8. 修订记录

- 2026-11-13: 初稿。B10 窗口假设证伪并记录（§1.2）。
- 2026-11-13: 实现完成。接入点落地为: `GoalExecutive` (src/models/
  goal_executive.py) + `set_exec_goal`/commit/settle (three_d_world.py) +
  RolloutBuffer `exec_actions`/`exec_logps` 两列 + 联合 logp (train.py) +
  eval 脚本 argmax 选择同构 (run_stage18_full_eval / op_failure_diag)。
  配置键置于 `env:`。单测: tests/test_goal_executive.py (5) +
  tests/test_exec_goal_env.py (6, 需 mujoco → 本地跳过, 服务器跑)。
