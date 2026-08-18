# Changelog

All notable changes to this project are documented here.
本项目所有值得记录的变更。

## [Unreleased]

### Stage 20h#4 · BC NLL 无界 -> NaN 级联修复 + 训练自愈 (2026-08-18) 🔥

- **崩溃实录**: 20h#3 修复 gather bug 后 bc 2.49→0.89 真实下降 (模仿
  发生, entropy 2.47→1.07, 策略变尖), 但 06:15 训练死于
  `Categorical(logits)` ValueError — logits 全 NaN (256,12):
  `-log π(teacher)` 在确定性模仿下无界 → 单个 teacher 步的 BC 梯度
  级联污染整个 PPO mini-batch
- **修复** (train.py):
  1. `teach_loss = -_lp_t.clamp(min=-6.0).mean()` — NLL 上限 6,
     单样本梯度有界 (~6/logit)
  2. **NaN 守护**: backward 前 `torch.isfinite(loss)` 检查, NaN 时
     跳过该 update 并打 WARNING — 训练永不静默毒化权重
- **配置**: entropy_coef 0.01→0.05 (熵缓冲), bc_teacher_coef 1.0→0.5
- **验证**: py_compile/test/check_bounds 全绿; 从 3401408 ckpt
  (崩前最后完好) 重启, total 3.5M, 重启后 100K 步内无 NaN

### Stage 20h#5 · teacher 满血重训 + 静止教学 (2026-08-18) 🔥

- **3400K 行为诊断实锤** (3401408): `move_steps 154 → 2429` (冻结
  解除!), `hit 501 vs random 108` (命中帧 5x), `rew_total 1226 vs
  random 1056` — 策略活了! 但 `op=0.08`: 窗口内游荡命中高,
  **reveal 时刻不在 last_known 旁** (cos≈0 的方向感 + 到位不驻留)
- **根因**: 教师只教"走"不教"停" — teacher 在 dist<0.8m 时放弃
  接管, BC 从未见过"到位=静止"的监督; agent 学会走近但继续游走
- **修复** (three_d_world.py): dist<0.8m 时 teacher 改教
  `action=8` (dev_age>0.15 时零力抓取=原地不动) — 把"驻留等待
  reveal"直接教进策略, 方向动作与静止动作同帧监督
- **调度修复** (train.py): teacher ramp 原用**绝对步数** —
  3.4M+ resume 时立即打到 0.15 地板, 模仿已死; 改为
  `(state.step - _resume_step_at)/ramp` 从 resume 点起算, 重启即
  满血 0.9, 4M 终点才降到地板
- **配置**: total_steps 3.5M → **4M**, ramp 600K (从 resume 起算);
  评测链 p1 (2100-3500K) + p2 (3600-4000K)
- **验证**: 从 3401408 重启 (08:47), ~57ms/步, 单实例确认

### Stage 20h · BC 教学脚手架: 模仿打破"冻结策略"死结 (2026-08-17) 🔥

- **根因** (20a-20g 七轮奖励干预全部失败的共同死结):
  - 2800/2900/3000K op = 0.08/0.00/0.04, 且行为诊断实锤:
    `move_steps=154` 在 20g 前后**一字不差** — 策略完全冻结,
    entropy≈2.45 (≈均匀), PPO 梯度 ≈ 0
  - 非对称塑形 (20c, 只奖趋近) 与对称塑形 (20g, 趋近+/远离−)
    对均匀策略的**期望梯度都是 0** (随机方向 dot 期望为 0);
    窗口 60 步 (3s ≈ 走 3-6m) 对 last_known 常距 5-10m
    **物理上走不到** → reveal 因果链永远连不上 → 无任何可学信号
  - conclusion: 停止调奖励, 改成"教"
- **方案: BC 教学脚手架** (发育学: 婴幼儿通过模仿学习):
  1. `ThreeDWorld.occluder_teacher_force` (默认 0): 遮挡窗口内
     以该概率由 teacher 接管动作, 朝 last_known 方向行走 (4 基本
     方向 + 远距双倍力档), 让 agent 反复亲历"搜索→找到→reveal"
  2. `env.last_teacher_action` 暴露动作标签; `RolloutBuffer` 新增
     `teach` 通道; PPO mini-batch 对 teacher 样本加
     `-bc_teacher_coef * log π(teacher_action)` 模仿损失 (默认 0.5),
     step 日志新增 `bc=` 字段
  3. `occluder_teacher_ramp` (300K 步): 接管率 0.9 → 0.15 线性衰减
     (脚手架逐步拆除, BC 样本随之消失 — 内化过渡)
  4. **纯净性**: eval 构造不转发该参数 (默认 0); 行为诊断脚本
     显式清零; 训练/评测观测形状不变 (12 动作 / 27 proprio)
- **配套**: total_steps 3M → 3.5M; 评测链扩展到 3500K
- **验证**: v/loss 瞬态 (同 20f) 回落, 首个 PPO 周期 `bc=2.55`
  (≈ log(12), 均匀初始, 信号链路通)
- **状态**: 已从 3000K ckpt 重启, 3200/3300/3400/3500K 为 20h
  首考

### Stage 20f · proprio 进策略 + yaw 观测: 槽终于可见 (2026-08-17) 🔥

- **根因** (20e 的继承根因): 20e 把 last_known 槽注入 proprio, 但
  **策略输入只有渲染图像 — proprio 从不进模型** (train.py rollout
  `model(obs_t)` 只传 image; proprio 仅被 cross-modal touch bridge
  使用)。槽信息从未到达策略 → 2500/2600/2700K op = 0.11/0.05/0.11
  纹丝不动, 行为诊断实锤: policy 遮挡窗口内 cos=-0.60 (反向离开
  last_known), 而 random ≈ 0; policy 总奖励 44 vs random 408
  (reveal/塑形信号被"不动"策略系统性错过)
- **方案**:
  1. `HierarchicalActorCritic.proprio_mlp` (Linear→GELU→Linear,
     proprio_dim→d_model), forward 里残差注入 h (manager 子目标 +
     worker 动作都看到), proprio 可为 None (无注入, 兼容旧路径)
  2. rollout 每步读 env 实时 proprio (含槽) 传入模型;
     `RolloutBuffer` 新增 `propr` 通道, PPO mini-batch 更新时同
     步传入 (否则 proprio_mlp 无梯度)
  3. `_proprio` 新增 yaw (cos, sin) 2 维 — 没有朝向, 全局 (dx,dy)
     槽无法映射成局部转弯/前进动作; 16→18 维, 含槽 27 维
  4. eval 同步: `_prop_to_tensor` + build_model(proprio_dim),
     measure_milestones 每步传 proprio (评测动作与训练同输入)
  5. resume 改 strict=False: 新 proprio_mlp 随机初始化, 其余 2.7M
     步权重完整保留 (20e 之前是 strict 加载, obs 形状未变所以其实
     权重从未重置 — 这也是 20e 无效的旁证)
- **验证**: 本地 smoke (梯度可达 proprio_mlp, None 路径安全, 旧模型
  无 proprio 层兼容), 远端 proprio_dim=27 + yaw 值实测
- **状态**: 重启训练, 3000K 之前看 2800/2900K 评测首考

### Stage 20e · 遮挡记忆注入观测: 打破盲走 (2026-08-17) 🔥

- **根因** (比 import math 更深): PPO 策略是 proprio-only
  (vision encoder off), `_proprio` 只有 pos/vel/touch/joints/grasp —
  **观测里从来没有任何 last_known/遮挡信息**。奖励给了 (距离差/
  塑形/归因), 但 agent 从不知道"该往哪走" → 任务对策略不可解 →
  op 天花板 0.11 = 随机接近底座噪声 (2.4M 步五次干预全部封顶于此)
- **方案**: `occluder_obs_slots=3` — 观测追加最多 3 个活动遮挡槽,
  每槽 (dx, dy, dist)/4.0 归一化相对偏移 (指向 agent 自己记忆中的
  last_known)。训练与评测同 env 代码 → 评测同样提供该向量
  (last_known = agent 自身记忆, 非外部注入, 评测度量 end<0.7*start
  测的正是这个行为)。观测 16 → 25 维
- **兼容**: resume 时 load_state_dict 不匹配 → 自动 fresh model
  (replay/记忆/技能全保留, 丢的只是没学会的旧策略)
- **验证**: 远端实测 proprio_dim=25, 遮挡激活时槽位携带 last_known
  偏移且值与 expect 一致
- **状态**: 从 2400K ckpt 重启, 2500K 评测验证 (新观测首考)

### Stage 20d P1.1 · 窗口物理对齐: hold 8→30 步 (2026-08-16) 🔧

- **背景**: P1 固定目标后 2100/2200/2300K op = 0.11/0.04/0.11,
  与 20d 随机目标区间重叠 → 固定映射未带来爬升。诊断: 
  **训练与评测的时间预算物理不匹配** — 训练 hold 8 步 = 0.4s,
  agent 最快走 ~0.14m, 而评测要求 end_d < 0.7*start_d, start_d
  平均 1-4m 的 30% = 0.3-1.2m, 8 步物理上走不完 → 归因奖励
  几乎永远发不出 → 又回稀疏陷阱 (评测用自然遮挡, 窗口天然长;
  训练却造了更苛刻的 8 步窗口)
- **方案**: `object_crossing_hold_steps: 8 → 30` (1.5s, 可走 ~0.5m,
  0.7 阈值物理可达)。评测不读此参数 → 评测侧零变化, 容器测泛化。
  发育类比: 藏物游戏给孩子充分翻找时间, 非降低目标
- **状态**: 配置改动重启, P1 固定目标保持, 2400K 评测验证

### Stage 20d P1 · 固定遮挡课程: 恒定可学映射 (2026-08-16) 🎯

- **背景**: 20d 归因奖励成绩单 — 1600K=0.00, 1700K=0.11 (峰值),
  1800K=0.09, 1900K=0.04 → 有信号但脆弱回落; 且 1900K 伴随
  means_ends/physics 同步崩坏 (1.00/1.00 → 0.02/0.00), 疑似被
  3d-crowded 场景噪声主导
- **方案**: `object_crossing_fixed_object=0` + `object_crossing_fixed_wall=0`
  — crossing 永远只让物体 0 穿越墙 0 (镜射点在两固定位置间振荡),
  "哪个消失→去哪找"成为恒定映射而非每 50 步随机新目标。评测侧
  make_env 不读这些参数 → 评测仍是随机遮挡 = **测泛化**, 不是测
  记忆单个位置
- **预算**: `total_steps: 2000000 → 3000000` (2M 终点续训 1M 步)
- **验证**: 远端 eposide 实测 crossing 恒为物体 0 / reveal 恒为 [0]
- **状态**: 从最新 ckpt 重启, 新评测链 2100K→3000K 每 100K 验证

### Stage 20d · reveal 归因奖励: 给"成功追踪"显式因果 (2026-08-15) 🎯

- **背景**: 1500K 评测 op=0.05 与 1400K 一模一样 → 奖励通路已通
  (diag 0.30 实测) 但 200K 步零增长。诊断: 随机遮挡目标 (20 选 1)
  + 随机行走奖励陷阱 (dist/dot 触发率 ~50%, 正负对称) + 缺归因帧
  (8 步 hold 揭示时无"找到"奖励) → 稠密随机信号淹没过因果信号
- **方案**: `occluder_reveal_bonus=1.0` — 遮挡揭示瞬间若
  `end_d < 0.7*start_d` 且轨迹 ≥3 点 → 发大额归因奖励。度量子与
  评测 op 指标完全同构 (同一 quantity), 纯训练信号, 评测 make_env
  不读任何 occluder 参数 (已验证只转发 num_occluders)
- **架构**: `_maybe_reveal_bonus(key)` 挂在两条 reveal 路径 (hold
  结束 / 物体变可见); 通过 `_reveal_bonus_pending` (容量 1 浮点)
  延迟 1 帧交给 `_occluder_only_reward` 消费 (读取后清零, 一次);
  与距离差/塑形并存, focus 锁不变
- **验证**: 远端 pytest 4/4 (归因/不归因/短轨迹/关闭开关); 本地
  py_compile; check-bounds 无新结构
- **状态**: 从最新 ckpt 重启, 1600K 评测验证归因是否破对称性

### Stage 20c · 关键路径异常暴露改造 (2026-08-15) 🛡️

- **背景**: import math 灾祸证明"裸 except 静默吞错"= 慢性毒药 ——
  检查点丢失/奖励失效可以默默存在三个 Stage。
  回填: 关键路径 (奖励计算/遮挡事件生成/观测构造/穿越 teleport)
  全部改为 `_expose_exc(where)` (打印完整 traceback + 安全回退,
  训练不崩但日志可见); per-object 循环容错保留但必须带 `# legit:`
  注释说明为何安全
- **改造范围** (three_d_world.py): `_occluder_only_reward`,
  `occluder_target_reward`, contact/caregiver/approach 奖励, crossing
  teleport, `_proprio`, `_sync_held_object`, `_contact_reach`,
  `_update_chain_task`, count_finalize, `_use_held_as_tool`
- **验证**: 远程 diag OFFICIAL=0.3003 依旧 (异常暴露改造不影响正常路径);
  pytest env 相关通过
- **规则已固化**: 写进 AGENTS.md §14 (铁律: 关键路径绝不静默吞异常;
  新 try/except 必须回答 "异常是谁预期的, 出事后谁看见")

### Stage 20c · 根因修复: occluder 奖励从未生效 (import math) 🔥 (2026-08-15)

- **根因** (op 瓶颈的真正症结): `three_d_world.py` 从未 `import math`,
  而 `_occluder_only_reward` 与 occluder_target_reward 用 `math.hypot`
  → 每次调用抛 `NameError: name 'math' is not defined`, 被裸 `except`
  吞掉恒定返回 0.0。**该奖励自 Stage 20 起从未给过任何梯度**
  (0.3→1.0 的强化/20b 聚焦/20c 塑形全部默默无效)
- **证据链**: 模块顶部字节码/源码一致; 诊断脚本自行 import math 才"成功"
  (掩盖真相); 把 except 改成打印 traceback 后立刻暴露 NameError。
  800K op=0.11 是 means_ends 接近行为恰好经过 last_known 的巧合,
  不是奖励驱动的学习
- **修复**: 模块顶部 `import math` (1 行). 远端验证: `_occluder_only_reward`
  OFFICIAL 由 0.0000 → 0.3003 (塑形真给梯度)
- **状态**: 已 push (`8b6c818`). 训练从 1301K ckpt 重启 (pid 882822),
  这是 focus 锁 + 距离差(1.5) + 塑形(0.3) 三者首次真正生效的训练

### Stage 20c · 塑形解锁 op 冷启动 (2026-08-15) 🧭

- **背景**: 20b 纯 focus (只留距离差 occluder 奖励) 训练 300K 步 (1100K-1300K)
  评测 op 恒为 0 —— 机制验证: agent 从不朝 last_known 移动 (random walk 单调
  远离), 距离差奖励 `dist<prev` 永不成立 → 零梯度陷阱; 且 occluder_trace=false
  (评测一致性) 使 agent 纯靠 RNN 记忆 8 物体位置, 冷启动失败
- **方案**: `occluder_shaping_weight=0.3` — 遮挡窗口内奖励朝 last_known 方向的
  速度分量 (dot>0)。密集即时, 不依赖 agent 已靠近, 教 "物体消失→走向它最后
  位置"; 只用 pre-occlusion 记忆 (last_known), 无评测实体信息泄露
- **架构**: `_occluder_only_reward()` 内新增塑形分量, 与距离差并存; focus 锁
  不变; 评测 make_env 不读塑造参数, 评测纯净
- **config**: 同一 `stage20b_focus_op.yaml` 增 `occluder_shaping_weight`
- **验证**: pytest (含 env 相关) + check-bounds 通过
- **状态**: 从 1300K ckpt 续训, 1400K 评测验证塑形是否催生 op 梯度

### Stage 20b · 课程固化: 专训物体恒存 (2026-08-15) 🎯

- **背景**: S18/S19/S20 连续三 stage 卡 op 瓶颈 800K op=0.11 -> 1M op=0
  (means_ends 0.01 -> 1.00) —— 目标是奖励竞争而非训练技巧
- **方案**: `focus_op_only` env 开关 — 训练时封闭 means_ends/物体移动/
  接触/caregiver 等全部非-op extrinsic 回报, 唯一剩余 extrinsic 梯度 =
  occluder_target_reward (1.0 -> 1.5)。intrinsic 好奇保留 (探索燃料),
  PPO 对 intrinsic 的梯度是探索性的, 不构成目标竞争
- **架构**: `_occluder_only_reward()` 抽取遮挡追踪奖励; 评测侧不受影响
  (make_env 不读 focus 参数, 评测纯净)
- **config**: `stage20b_focus_op.yaml` (新, stage20 派生, 1M 独立预算)
- **验证**: pytest + check-bounds 通过
- **状态**: 从 1M ckpt 恢复, op 到 0.6 后再开放多目标 (Stage 20c)

### Stage 20 · 评测有效性修复: 遮挡持留 + 奖励强化 (2026-08-15) 🔧

- **根因**: op 600K=0.0 非退化——评测度量需要多步遮挡轨迹
  (end<0.7*start), 但 crossing 把物体瞬间 teleport 过墙 → 事件 1-2 步
  结束 → 轨迹 <3 点被丢弃 → op 测 ~0 (400K 的 0.115 是 chance 波动)
- **fix1**: `object_crossing_hold_steps=8` — 穿越后物体停在墙后 8 步,
  期间强制 truly_occluded, 遮挡持续可测; reveal 信号在 hold 结束时发
- **fix2**: `occluder_target_reward` 0.3 → 1.0 — 遮挡期间朝 last_known
  的 PPO 梯度显著化 (0.3 相对 intrinsic 太弱)
- **验证**: 修复后 651K ckpt 12 eps 事件 6→19 (全有效轨迹), 
  op 测得真实能力 ~0.05 (修复前真能力被噪声掩盖)
- **状态**: 训练从 651K 恢复 (train_s20b.log), 评测链 v2 只跑 800K/1M
- 附带: 远端 reflection device-mismatch 警告为既有问题 (与本次无关)

### Stage 20 · 推理闭环修复: 探针激活 + 接近验证 + 超时保护 (2026-08-14) 🔧

- **bug1**: probe_net 从头训练 (~0.5 < 0.85 阈值) -> 从不探针 ->
  active hypothesis 永远 None -> 验证永不跑。修复: propose 后显式激活
  (有 active 遮挡时取 least-tested 假设的动作)
- **bug2**: 验证依赖 just_revealed (物体重现), 但 agent 不动 -> 遮挡不解除
  -> 验证卡死。修复: 接近 last_known (<1.2) 即验证成功 (贴合评测指标:
  追踪 = 遮挡期间朝 last_known 靠近) + 30 步超时重置 (防死锁)
- **bug3**: 无可见性 (成功才打日志)。修复: stats 计数器 + 每 5000 步日志
  (proposed/probed/verified/timeout)
- **验证**: 闭环全链打通 — proposed 14K / probed 13K / verified 13K
  (93% 成功率) / timeout 0
- **效果**: 400K 评测 (4 墙真遮挡) op 0.045 -> 0.115 (2.6x),
  physics 0.0 -> 0.8 ✅ (4 墙环境适应); 方向正确但距 0.6 闸门仍远
- 附带: 物体穿越 (crossing) 触发 just_revealed 信号

### Stage 20 · 假设-演绎引擎: 物体追踪推理闭环 (2026-08-14) 📋

- **核心**: 物体追踪 = 假设-演绎原生任务 (非旁路奖励):
  - `three_d_world.py`: `get_occlusion_signal()` 接口 (just_occluded/just_revealed/
    active) + 物体穿越墙 (`object_crossing_every`, 镜像翻越) + occluder 4
  - `train.py`: HypothesisTester 接线 — 遮挡 -> propose_hypothesis("obj 在
    last_known") -> should_probe 走向 last_known -> 重现时 feedback(1.0) ->
    验证成功 -> logic_engine.add_rule("IF occluded THEN track")
  - `_action_toward()`: last_known 方向 -> 8 向动作映射
- **env 验证**: 300 步内 25 个遮挡事件 (Stage 19 稀疏 <5) — 密集化 8 倍,
  引导奖励 (0.3) 可学
- **config**: `stage20_hypothesis_deduction.yaml` (num_occluders=4,
  object_crossing_every=50, hypothesis_tester_enabled, total_steps=1M)
- **跨 stage resume**: stage19 1.3M ckpt 权重起步, step 重置 0
- **评测一致**: 评测 make_env 从 config 读 num_occluders=4 (真遮挡, 不传
  reward/trace)
- 训练已启动: 早期专项目标 op(真遮挡) >= 0.6 @ <=200K

### Stage 19 · 收尾: 引导奖励无效, op=0.1 记录为已知瓶颈 (2026-08-14) 📋

- **1.1M/1.2M/1.3M 真遮挡评测 (occluder_target_reward=0.3 引导 300K 步)**:
  object_permanence 0.119 -> 0.089 -> 0.089 -> 0.100 —— **引导奖励完全无效**
  (真遮挡 op 全程稳定 ~0.1); number_sense 升到 0.713 ✅
- **根因 (已验证)**: 遮挡事件稀疏 (2 随机墙 × 8 物体, agent 很少处于
  "刚目睹物体被墙挡"状态) + 0.3 权重稀疏奖励学不动
- **双稳态确认**: 距离遮挡 op 在 0.51-0.56 (好态) 与 0.33 (坏态) 间摆动,
  真遮挡 op 稳定 0.1 —— 之前 0.51 是评测口径宽松的假象
- **停止 Stage 19 训练** (1.3M sealed), 记录:
  - 通过: means_ends / intuitive_physics / number_sense (多次, 0.61-0.71)
  - 未过: object_permanence (真遮挡 0.1) / ToM (0.09-0.43) / systematic (0.37)
- **Stage 20 解决路径 (已确认)**:
  1. 遮挡事件密集化 (occluder 2->4 + 物体频繁穿越墙后)
  2. 物体追踪作为假设-演绎推理子任务 (推理任务自然提供练习)
  3. 训练-评测完全一致 (真墙一致, 评测不传 reward)
  4. 引导奖励保留 (0.3 + 密集事件 -> 可学得动)
  5. 双稳态稳定化 (工具使用连贯性任务)
- **基础设施修复 (本次)**: 数据重定向到数据盘 `/root/autodl-tmp/karbon/data`
  (DEVAGI_DATA_DIR, /dev/sdb 125G 可写) + replay cold_max_shards 32->6
  + 训练内部磁盘守卫 (>70% 清 shards) —— 30G 系统盘不再被 replay 撑爆

### Stage 19 · occluder_trace 视觉痕迹反馈 (发育式干预, 2026-08-13) 🔧

- **多 seed 评测确认坏态**: 800K 在 seed 42/7/123 下综合分数完全相同
  (op 0.333 / means_ends 0.333 / physics 1.0 / number 0.625 / ToM 0.291) ——
  非评测噪声, 策略稳定锁定"无目标行为"态; 600K 好态 (0.514/1.0) 未再现
- **发育式干预 (B 方案)**: `three_d_world.py` 加 `occluder_trace` 参数 —
  遮挡发生时在 last_known 地面显示黄色标记 (预埋 geom, 初始隐藏),
  解除时隐藏; 这是"环境反馈增强" (类比物体掉落有声音), 非行为奖励植入
- **训练/评测分离**: 训练 config 开 `occluder_trace: true`, 评测脚本
  不传 (默认 false) —— 评测验证真实记忆追踪, 不被痕迹污染
- **发现**: train.py env 构造从未传 num_occluders (默认 0) —— 训练环境
  一直无真墙, config 的 occluder 数从未生效; 已修复 (传 num_occluders +
  occluder_trace)
- **教训**: 600K 好态 ckpt 被滚动覆盖删除 (ckpt 只保留最近几个);
  误杀 900K 健康训练想换 600K, 结果 600K 已不在 -> 从 901K 重启
- 验证: 标记创建/隐藏 OK; 训练 resume 901K 运行中

### Stage 19 · 600K 评测: 修复验证成功, 回到健康轨迹 (2026-08-13) ✅

- **600K 评测** (device bug 修复 + occluder=2 后 50K 步):
  - means_ends 0.333->**1.0 ✅** (恢复), object_permanence 0.333->**0.514**,
    ToM 0.291->**0.429** 全面回升
  - **600K 与 200K 分数完全一致** (0.5135/1.0/1.0/0.35/0.4285/0.3736) ——
    策略回到 approach 污染前的健康目标导向状态, 修复彻底
- **number_sense 头本身完美** (acc 1.0/MAE 0.0), 评测任务内子项低是
  环境变化 (occluder 3->2) 引起的行为不匹配, 非真实退化
- object_permanence 0.514 是唯一活跃闸门 (差 0.086, task0 0.60 已过线);
  决策 A: 继续观察 700K/800K 自然恢复 (参考 200K->250K 曾 0.513->0.556)

### Stage 19 · 修复叙事 device bug + 训练环境回归 occluder=2 (2026-08-13) 🔧

- **叙事 device bug** (cloud_24g 切换引入): `rollout_hidden_states` 存 CPU
  tensor (2291 行 .cpu()), 而 trait_projector 在 GPU (cloud_24g) ->
  300K->550K 全程 `narrative generation failed` (device 不匹配), 叙事投影
  混合从未生效; 修复: `n_last_hidden = ... .to(device)`; 验证 #65 正常
  (extraversion 0.38 投影真实生效)
- **训练环境回归**: `num_occluders` 3->2 (250K/300K 时代环境)。550K 评测
  object_permanence 0.333 远低于 300K 0.556 —— 嫌疑: 训练 3 墙环境与评测
  无墙不匹配, agent 学会绕墙行为在无墙评测下退化
- **磁盘满事故**: replay SSD shards 14GB 写满 30G 盘 -> torch.save 失败
  训练崩溃 (600K ckpt 损坏 128B); 清理 shards 后从 550K 恢复;
  教训: replay 冷层 shards 是孤儿数据 (ckpt 不序列化), 可安全删除

### Stage 19 · 回滚 approach reward (400K 回归确认有害) (2026-08-12) ⚠️

- **400K 评测 (approach 训练 51K 步后)**: object_permanence 0.556->0.17,
  means_ends 1.0->0.02, ToM 0.46->0.20 全面崩溃; number_sense 反而升到 0.78
  -> approach 动机让 agent 变成"到处接近物体"的无头苍蝇, 淹没目标导向信号
- **回滚**: `three_d_world.py` 加 `approach_reward_weight` 参数 (默认 0.0),
  从 300K ckpt (健康基线) 恢复训练; 叙事修复保留 (250K 证明有效)
- **教训**: 通用动机奖励 (approach) 在物体多的 3D 场景里每步累积多物体
  奖励, 淹没任务信号 —— 需去重/上限机制才可能再试
- **最终判断 (Stage 19 收尾记录)**:
  - approach_reward_weight 能用, 但不能用现有 0.2 泛化版本 (对场景内
    所有物体每步累积, 8-16 物体时信号淹没)
  - 当前保持 0.0; 代码保留门控 (weight>0 才启用), 便于后续情境化改造
  - **去重版方向 (Stage 20 假设-演绎阶段候选)**: 情境限定为 occluder
    遮挡事件中的 last_known 目标 —— 仅当物体刚被遮挡时才给 approach
    信号, 且每次遮挡只对单个目标生效 (去重 + 上限)
  - 发育路线判定: 定向/情境版 approach 是"内在动机 + 任务感知"的自然
    延伸, 符合发育路线 (非外部干预, 非特异性奖励植入)

### Stage 19 · 云端训练必须 `--preset cloud_24g` — 默认 local_smoke 强制 CPU (2026-08-12) ⚠️

- **事故**: 云端启动命令未传 `--preset`, 落默认 `local_smoke`
  (`device_preferred: cpu`) -> 模型建在 CPU, 3.3 步/s 慢速训练,
  日志间隔 90s-10min 被误判为"卡死" (4 次误判 + 数小时排查)
- **铁律**: 云端跑训练必须显式 `--preset cloud_24g` (GPU);
  评测脚本 `run_stage18_full_eval.py` 默认已是 cloud_24g, 不受影响
- **佐证**: 切换后 3.3 -> 6.8 步/s, 显存 4MiB -> 2.2GB, GPU 正常占用
- 附带: `DEVAGI_NO_COMPILE=1` 避免 torch.compile recompile 抖动
  (5 维训练 batch 触发 slot_attention permute 编译失败, 每次卡 2-4 分钟)

### Stage 19 · approach reward: 通用接近动机打通 遮挡→追踪→重现 因果链 (2026-08-12) 🔧

- **根因 (reward 结构审查)**: `ThreeDWorld._compute_reward` 无 approach 组件
  (只有推物/接触/caregiver 接近), 而 `physics_sandbox` 有 (prev-dist)*0.2。
  遮挡期间 agent 走向 last_known 无任何内在回报 -> 永不学"跟踪被遮挡物"
- **修复 (环境反馈增强, 非特异性植入)**: `_compute_reward` 增加 approach
  reward (与 physics_sandbox 完全同款); 这是环境对"接近物体"的自然反馈,
  非针对 object_permanence 探针的奖励 —— 符合发育路线 (不对行为特异性奖励,
  只恢复环境该有的反馈)。遮挡→追踪→重现→接触 的因果链得以训练中自然涌现
- 配套: `_prev_obj_dist` 有界 (num_objects 固定, reset 精确重建);
  config `num_occluders` 2->3 (更多遮挡练习情境)
- 评测无影响 (评测 rollout 不看 reward)

### Stage 19 · 叙事去模板化: 事件分型 + trait 投影接入 + 分档文案 (2026-08-12) 🔧

- **根因 (200K dump 确诊)**: 100 条 life events 全为成功事件 (失败/探索
  return≈0.03 被 importance 淘汰) -> traits 恒 (0,1,0,0,0) -> 叙事恒同
  "persists through challenges to achieve goals." -> symbol bias 恒定 act=2 ->
  策略不被调制, object_permanence 100K->200K 停滞/回落
- **修复 1 (事件分型)**: `developmental_memory.py` LifeEvent 增加 `event_type`
  (success/failure/exploration) 字段 + add_event/promote_to_life_event 透传 +
  state_dict 序列化; `train.py` 事件构造分档 (success: importance=return /
  failure: 保底 8.0 / exploration: 保底 4.0), 失败与探索事件不再被挤出记忆
- **修复 2 (投影接入)**: `abstract_reasoning.py` `extract_traits` 优先按
  event_type 结构计数 (关键词降级为 fallback, 兼容旧事件);
  `forward(life_events, hidden_state, blend)` 混合统计 traits 与
  trait_projector 投影 (blend=0.5) — 叙事随行为演化, projector 不再只训练不消费;
  narrative_loop 与 train.py 透传 episode 末 hidden_state
- **修复 3 (分档文案)**: `generate_narrative` 从 5 个固定 if 升级为连续分数
  分档变体 (开放/尽责/外向/宜人/神经质 各有 2 档句式)
- 测试: test_narrative_loop.py +2 (importance/event_type 透传, hidden 传递),
  test_cognitive_landing.py +2 (event_type 计数, projector 混合)

### Stage 19 · 运行修复: developmental_memory 段缺失 + 评测脚本读取 narrative (2026-08-11) 🔧

- **Bug F (配置接线)**: `memory_manager` 创建条件需要 `developmental_memory.enabled`,
  但 Stage 18/19 配置都缺该段 -> memory_manager 从未创建 -> NarrativeLoopController
  创建条件静默失败 -> AutobiographicalMemory 不存在 -> 叙事从未产出
  (也解释了 Stage 18 `[identity]` 计数 0 的谜底: IdentityNarrative 依赖 memory_manager)
- 修复: `stage19_self_narrative.yaml` 增加 `developmental_memory` 段 (episodic=10000,
  semantic=1000, autobiographical=100); 重启训练 (resume 自 51.2K ckpt)
- 验证: 训练日志出现 `MemoryManager enabled` + `NarrativeLoopController enabled`
- 评测脚本: summary 白名单加入 `narrative_loop_state` (读取叙事状态)
- Stage 19 @ 51.2K 首次全量评测 (v2 脚本, 5 tasks × 20 eps):
  - object_permanence 0.583 (↑ vs S18 1M 0.533, 最接近闸门 0.6)
  - reflection len=6 ✅ / kanren queries=21,760 ✅ (闭环修复验证)
  - means_ends 0.50 / physics 0.78 / number 0.51 (跨阶段 resume 回退, 符合预期)
  - est. age=0.0y

### Stage 19 · 自我叙事引擎 设计 + 核心实现 (2026-08-11) 📋

- 设计文档: `docs/stage19_design.md` — 记忆->叙事->策略调制闭环
- 新模块: `src/models/narrative_loop.py` — NarrativeLoopController:
  - episode_end_hook: 自传记忆存储 + 周期身份叙事 + kanren symbol bias 刷新
  - step_hook: 委托 ThoughtActionLoop (FiLM 调制, 可选)
  - get_symbol_bias: 由 kanren 最高置信规则推导 (num_actions,) logit 偏置
  - 有界: 单叙事字符串 + 5 维 trait + (num_actions,) 张量
- `hierarchical_policy.py`: `set_symbol_bias_fn` 回调 + forward 中注入
  detached logit bias (模型与叙事模块解耦)
- `abstract_reasoning.py`: IdentityNarrative 新增 `trait_auxiliary_loss`
  (启用未用的 trait_projector, 用行为统计做监督)
- `three_d_world.py`: 新增 occluder 遮挡墙 (num_occluders) + 真遮挡
  线段-AABB 检测 (`_line_of_sight_blocked`) — object_permanence 从
  "距离事件" 变为 "真遮挡事件"
- `train.py`: 接入 NarrativeLoopController (创建/回调/hook/训练/状态),
  trait projector 优化器 + episode-end 训练
- 配置: `configs/stage19_self_narrative.yaml` (narrative 段 + num_occluders=2
  + curriculum 20K 加速轮转)
- 测试: `tests/test_narrative_loop.py` 7 例 (事件存储/周期叙事/symbol bias/
  step hook/有界/降级/状态往返)

### Stage 18 · 1M 训练完成 + 评测缺陷修复 + 评测报告 (2026-08-11) 📋

- Stage 18 训练 1M 步完成并封存 (`ckpt_stage18_001000000.pt`):
  - 最终状态: rules=30, 因果图 36/512 边 (8,300 干预), skills 满 10,496,
    coverage 100%, replay 672K/688K, EWC consolidated, 想象更新 2,445 次
- 1M 全量评测 (watcher 自动触发): number_sense 0.325->0.600 **新通过** ✅
  (头精度 76.7%->90%, MAE 1.633->0.7); object_permanence / ToM / systematic 未过
- 50-ep 定基重测确认: number_sense 0.550 (20-ep 0.613 是小样本运气, 实际未过),
  object_permanence 0.524 (真实差距 0.076, 非方差), systematic 0.374 (稳定)
- **评测缺陷修复** (`src/eval/developmental_milestones.py` +
  `scripts/eval/run_stage18_full_eval.py`):
  1. systematic_reasoning 熵归一化错配: 硬编码 ln(8) 改为按实际动作空间 (12) -
     修复前该里程碑被天花板锁死在 ~0.04, 修复后测出真实水平 **0.373**
  2. `rule_count` 未接线: 评测从不传规则数 -> rule 项恒 0; 现从 ckpt
     symbolic_state 传递 (30)
  3. epsilon 0.3->0.1 + 分 curriculum task (0-4) 逐任务评测 -
     object_permanence 0.339->0.533, ToM 0.281->0.443
  - 新增 3 个回归测试 (带规则数可过 / 无规则数不过 / 12 动作均匀分布低分)
- **Bug E: 反思维度不匹配** (`configs/stage16_neuro_symbolic.yaml` + `src/train.py`):
  - 根因: `self_model_d_model: 384` 但 backbone `hidden_size: 128` ->
    ReflectionLoop.end_episode 喂 128 维给 GRU(384) -> RuntimeError ->
    被 `except: pass` 静默吞掉 -> 16,599 episodes 0 条反思 (count-only 空转)
  - 附: SelfModel.auxiliary_loss 从未在 train.py 中调用 -> 权重随机初始化未训练
  - 修复: config 384->128; 裸 except -> logger.warning (不再静默吞异常)
- **前置 4/4 修复完成**:
  3. SelfModel.auxiliary_loss 接线: 创建 self_model_optimizer + episode-end
     训练块 (targets: confidence=成功/失败, familiarity=coverage_ratio,
     progress=ep_ret vs running mean) -> SelfModel 权重不再随机
  4. kanren 后端消费: episode-end query+feedback 循环 -> 每条规则 predict_action
     被查询, 与实际动作比对, feedback(correct) 接入 learning-back 路径
     -> queries > 0, accuracy 可测
- 根因分析: task 3 独占训练 (99.95% 优先级) 导致稀疏场景 (task 0/1)
  means_ends 崩至 0.05; kanren 后端 0 查询 / reflection 空转 / SelfModel 未训练 -
  符号-元认知栈从未闭环; object_permanence 评测是"距离事件"非真遮挡
- 报告: `docs/stage18_report.md` (500K vs 1M, 修复前后, 分 task 分解,
  Bug E 根因, Stage 19 建议)

### Stage 18 · 全量评测脚本 (2026-08-11) 📋

- 新增 `scripts/eval/run_stage18_full_eval.py` — Stage 18 全量 3D 评测脚本:
  - 与训练配置精确匹配: ThreeDWorld 8 物体 / 128 渲染 / 12 动作 / dev_age=0.5
  - 从 ckpt 推断层数 (7 层), HierarchicalActorCritic 完整加载 (0 missing/0 unexpected)
  - 评测项: 3D 发育里程碑 (est. age) / ToM 模块直测 / NumberSense 头 / 场景 slot 利用率 / 训练状态摘要
  - `MUJOCO_GL=egl` 无头渲染
- 500K 步全量评测结果:
  - means_ends=1.0, intuitive_physics=1.0 (满分保留)
  - object_permanence=0.444 (差 0.156 到阈值)
  - ToM 行为指标 0.362, ToM 模块直测: 视角✅ 错误信念✅ 惊讶预测❌
  - number_sense=0.325 (头精度 76.7%), systematic=0.042
  - est. age=0.0y (object_permanence 未过阻塞年龄递增; 与之前 64x64 环境评测的 2.5y 虚高对比)
  - 训练状态: rules=29, 因果 36 边, curiosity=1.0 drive=1.0 task=0.7567
- 教训: 评测环境必须与训练精确匹配 (渲染尺寸/物体数/动作空间), 否则 est. age 虚高

### Timeline 更新 · Stage 19-23 架构可行性评估 (2026-08-11) 📋

- `docs/TIMELINE.md` 采用新 Stage 19-23 规划 (自我叙事->假设演绎->递归元认知->抽象概念->开放世界永续)
- 完成架构可行性评估 (基于代码库实测):
  - Stage 19 自我叙事 (10-12y): **当前架构可扩展** — IdentityNarrative + AutobiographicalMemory + InnerDialogue 已有雏形, 缺整合闭环
  - Stage 20 假设-演绎 (12-13y): **当前架构可扩展** — HypothesisTester + ActiveExperimenter + kanren 已有雏形, 缺形式运算级闭环
  - Stage 21 递归元认知 (13-14y): **需要新架构** — SelfModel 无法监控"思考过程"本身
  - Stage 22 抽象概念 (14-15y): **需要新架构** — 缺概念层级结构
  - Stage 23 开放世界永续 (15y+): **新架构整合** — extended_3d_world 已建未接入
- 北极星鸿沟收敛: 10-12y 已从"未验证"降级为"可扩展", 13y+ 才是真正的架构分水岭

### Timeline 更新 · Stage 19+ 机制展开 + 北极星差距分析 (2026-08-10) 📋

- `docs/TIMELINE.md` 展开 Stage 19+ 为五个子阶段:
  - 19A 自主目标设定 (内在动机驱动)
  - 19B 深度语言扎根 (语言作为推理工具)
  - 19C 抽象推理 (类比+反事实+多步逻辑链)
  - 19D 跨领域迁移 (技能组合+零样本适应)
  - 19E 持续自我进化 (记忆核心+架构扩展+无限稳定)
- 新增 "北极星差距分析" 章节, 明确诚实评估:
  - 已验证: 0 -> 3.5 岁路径可行 (感知/动作/物理/因果/符号/想象)
  - 补上机制后预期: 8-10 岁认知水平
  - 10-15 岁鸿沟: 需要形式运算/递归元认知/自我叙事, 未验证
  - 6 个关键实验待跑: 5M-10M 步饱和测试 / 环境复杂度 / 自主目标 /
    记忆整合 / 模块深度整合 / 永续稳定性

### Stage 15 · 3D 世界迁移 (2026-08-04 ~ 2026-08-05) ✅ 完成

#### 核心成果：信息天花板突破
- **KL 跃升**: 0.003 (MiniGrid 坍缩) → 0.35 (3D 活跃编码)
- **Recon 下降**: 39.09 → 1.68 (96% 降幅)
- **Mean Return**: 9.7 → 56.3 (5.8x 提升)
- **结论**: 高信息量 3D 环境是像素 WM 学习的前提，已被实证验证

#### 1M 步训练最终指标
| 指标 | 初始 | 最终 | 变化 |
|------|------|------|------|
| Recon | 39.09 | 1.68 | ↓ 96% |
| KL | 0.23 | 0.35 | 活跃编码 |
| 因果图 | 0/512 | **23/64** | ✅ 完成 |
| 覆盖率 | — | 100% | 满仓 |
| 技能 | — | 10496 | 满仓 |
| 规则 | 0 | 19 | 积累 |
| Mean Return | — | 56.3 | 行为学习 |
| EWC Fisher | — | 215.25 | 旧知识保护 |

#### 因果发现：✅ 完成
- **环境结构**: 8 动作 × 7 物体，理论上限 64 条因果边
- **实际发现**: 23 条有效因果边（覆盖率 36%）
- **验证方法**: EMA 收敛确认，无新 (source, target) 对涌现
- **Effect 强度**: 0.32/0.23/0.10，远超阈值 0.005
- **探索充分性**: Action 分布均匀（覆盖 6/8 动作）
- **结论**: 因果图谱已达当前环境完整结构，因果发现阶段正式结束

#### 因果图完整性声明
> 在 8 动作 × 7 物体的环境空间中，经 1M 步充分探索，确认 23 条有效因果关系，EMA 强度收敛，无新边涌现。因果发现阶段正式结束。

#### 渲染修复 (three_d_world.py)
- **camera 缺失修复**: 场景 XML 无 `<camera>` 定义 → free camera 在视野外 → 渲染全白
  (mean=255, std=0)。添加 `targetbody` camera 跟随 learner
- **像素缩放 bug**: `Renderer.render()` 在 mujoco 3.x 返回 uint8(0-255)，
  代码误用 `clip(0,1)*255` → 任何非零像素变纯白。修正为 dtype 检查
- **camera 参数化**: SceneBuilder + ThreeDWorld 新增 `camera_pos`/`camera_fovy`
  参数，支持多视角对比实验
- **默认视角优化**: close-up (0,-1,0.8) fovy60 → edge=4.98/unique=3431/frame_diff=6.19
  (vs 旧高远视角 edge=0.008/unique=9)，画面信息量提升 57 倍

#### 接触检测修复
- **固定阈值 bug**: contact/force_motion 用固定距离(0.25/0.4)，大物体
  (halfsize=0.5) 接触时中心距≈0.62 永远检测不到。改为 `_contact_reach()`
  基于 agent 半径 + 物体半径 + margin 的动态阈值
- **agent size 缓存**: `_agent_size` 存储在 `_build_scene` 中供检测复用

#### 验证结果
- ✅ 渲染: unique=3431, edge=4.98 (17x vs 旧视角)
- ✅ agent 移动: 30 步移动 2.19 单位
- ✅ 接触检测: contact_reach 动态阈值生效
- ✅ 物体推动: position change verified (真实物理)
- ✅ 环境接口兼容: env_id="ThreeDWorld", action_space=8, obs=(128,128,3)

#### 文档
- **3D 迁移规划** (`docs/stage15_3d_migration_plan.md`): 迁移动机、验证结果、
  保留/迁移/新建清单、配置设计、训练计划、风险分析、成功标准

### Stage 14 修正 — WM 验证结论 + 因果发现 experience 模式 (2026-08-03)

#### 世界模型实验结论（已实证，84K 步）
- **像素 WM 在 MiniGrid 上经济上不可行**：`wm_diag` 验证 + free_nats=0.0 消融实验
  （kl_raw 单调跌向 0.0028，recon 恒为平均帧地板 ~4-5e-5）确认根因是
  **编码成本（~0.1-0.5 KL nats）> 重建收益（~0.003 loss）**，优化器理性地选择
  "不感知"。外部评审的"环境承载上限"风险被实证坐实。
- **发育路线修正**：想象训练（Stage 12 式）推迟到 3D 世界（Step 6，高信息量像素 +
  真实物理动态），类比人类 7-11 岁具体运算期之后。MiniGrid 阶段聚焦分层决策 +
  真实轨迹因果发现。

#### 新增
- **CausalDiscovery experience 模式** (`src/models/causal_discovery.py`):
  `observe()` 每步收集真实 (s, a, s', r) 转移（有界 deque，Axiom 1），
  `intervene_from_experience()` 按动作分组统计 E[||s'-s|| | a] 与全局基线之差，
  即随机探索策略下的观测等价 do(a)。不依赖世界模型；effect 信号直接来自真实交互。
- **配置** (`configs/stage14_causal_reasoning.yaml`): `causal_discovery.mode:
  "experience"`（默认；"wm" 模式保留给 3D 世界）、`buffer_capacity: 4096`
- **WM latent-only 切换**: `world_model.recon_loss_weight: 0.0`（像素重建关闭，
  保留 next 预测 + KL + reward 头），算力释放给因果主线；recon 权重 100x/
  像素加权 200x 的调优记录在此前的 Stage 14 配置中，供 3D 世界复用
- **测试** (`tests/test_causal_discovery.py`): 8 项（experience 边记录/最小样本/
  有界性/wm 模式隔离/legacy 干预路径/序列化往返/图容量/autograd 释放）

#### 修复
- **train.py 因果干预死循环修复**：intervene 调用点原要求 `wm is not None`，
  wm 无可用隐空间动态时 effect 恒 0（Stage 13-14 全程）；现按 mode 分流，
  experience 模式不再依赖 wm
- **experience 模式量纲修正**：effect 改为与全局基线的**相对提升**
  `(E[||Δs|||a] − baseline) / baseline`，替换原绝对差；记录阈值 rel>0.05、
  reward 缩放 ×100。因为 embedding 距离量纲是任意缩放的，绝对差（~0.01）
  × EMA(0.1) 收敛后 strength≈0.1，低于 wm 时代遗留的 get_causes(≥0.3)
  查询阈值，导致边界记了却查不出
- **CausalGraph 查询阈值** `min_strength=0.3→0.1`：wm 模式同样受益
  （MSE 量纲 ×10 scaling 原本也达不到 0.3）；`creativity_orchestrator`
  已显式传 0.1，不受影响
- **EWC 发育式强化（anti-forgetting）**: `ewc_lambda` 3.0->50.0、
  `ewc_gamma` 0.95->0.99、`ewc_consolidate_every_steps` 100K->25K +
  课程切换钩子（task-end Fisher，`src/train.py`）。修复
  `OnlineEWC.load_state_dict` 无条件从 ckpt 恢复 config 的 bug（每次 resume
  静默回退 yaml 调参 -> lambda=3.0）。回归测试 `test_load_state_dict_keeps_current_config`
- **eval 动作映射 bug**: 8-action 模型 argmax 直接喂 5-action MiniGrid 环境
  报 `Unknown action: 7`；`independent_evaluator.py` + `run_minigrid_eval.py`
  clip 到 `action_space_n-1`

### Stage 14 完成 - 最终结果 (2026-08-03)

> 1,000,000 步，RTX 3080 Ti (12 GB)，Stage sealed。
> 报告见 `docs/stage14_report.md`。

| 指标 | 值 |
|---|---|
| 独立评估 total | **0.648** (curiosity=0.21 drive=1.00 task=0.91) |
| empty-5x5 SR | 20% (GRR 43%) |
| empty-8x8 SR | 7% (GRR 10%) |
| doorkey-5x5 SR | **30%** (GRR 47%) ← 手段-目的推理核心目标 |
| doorkey-6x6 SR | 7% (GRR 17%) |
| Navigation / Means-Ends / Systematic | 0.20 / **0.30** / 0.22 |
| 因果图 | 58/512 边，experience 模式，effect 非零，query_why 可返回 |
| EWC | lambda=50, gamma=0.99, 23× consolidated, task-end 钩子触发 |
| Coverage | 49.9% |
| WM (latent-only) | loss ~0.0007, kl_raw ~0.0002 |

**关键发现**:
1. **像素 WM 在 MiniGrid 经济上不可行**（84K 步实证：encode cost > recon gain）->
   因果发现改用真实轨迹干预统计（experience 模式），想象训练推迟到 3D 世界
2. **EWC 强化验证发育式能力堆叠**：lambda 3->50 后 empty-5x5 SR 从 3%(强化前)
   回升至 37%(强化后)，doorkey-5x5 从 10% 升至 30%（最终）；对比强化前
   全任务崩至 0-3% 的覆盖式遗忘，是质的改变
3. **empty-8x8 遗留遗忘**：训练时间最久远的任务，EWC gamma=0.99 仍有衰减，
   SR 波动大（37%->0%）。后续可考虑任务级 Fisher 重放或 interleaved curriculum

### Stage 14 — Causal Reasoning (因果发现 + 反事实想象) (2026-07-31)

#### 新增
- **Stage 14 配置** (`configs/stage14_causal_reasoning.yaml`): 继承 Stage 13 外部记忆，
  启用 CounterfactualImagination + 增强 CausalDiscovery（intervene_every=250, max_edges=512），
  工具使用链课程（RedBlueDoors-6x6 多步推理），eval_epsilon=0.05 避免 eval 低估能力
- **Eval epsilon noise** (`src/eval/independent_evaluator.py`): 新增 `eval_epsilon` 配置项，
  在独立评估中以 epsilon 概率随机探索而非纯 argmax，修复 Stage 13 sr=0.00 在贪婪评估下被低估的问题
- **时序调整** (`docs/TIMELINE.md`): 将 Stage 14（因果推理）提到 Stage 12（想象训练）之前，
  发育逻辑：先建立因果图再发展基于因果图的想象

#### 修复
- **`causal_discovery.py` 干预永久休眠 bug**: `intervene()` 用 `intervention_count >= max_edges`
  作为开关——计数达到 512 后因果发现永远空转（Stage 13 遗留 39 条边后 `[causal]` 日志消失）。
  修复：计数与图容量解耦，干预永远运行；`_trim_graph()` 在边数达 `max_edges` 时淘汰最弱边
  （Axiom 1 有界性仍成立，Axiom 3 图不再冻结）
- **`logic_engine.py` checkpoint 恢复设备错位**: `state_dict()` 把 `category_embedding` 存到 CPU
  （`.cpu()`），但 `load_state_dict()` 直接赋回——恢复后的变量在 CPU、新定义的变量在 CUDA，
  `forward_chain()` 的 `cosine_similarity` 抛 "Expected all tensors to be on the same device"，
  `[logic] engine population failed` 导致该步符号推理空转。修复：`load_state_dict` 恢复时
  `.to(self._device)`（补充 2e6e19c 仅覆盖 init 路径的遗漏），附 meta-device 回归测试
- **`independent_evaluator.py` eval 样本量硬编码**: `min(episodes_per_task, 5)` 把每任务
  episode 数锁死在 5（即使 yaml 配置更高），加上 seed=0 固定布局 + rng(42) 固定序列，
  eval 分数呈 ±40% 噪声（102400 的 0.60 vs 126976/151552 的 0.00 均为 3/5 与 0/5 的抖动）。
  修复：解除 5 上限（配置 25），每 episode 用 `seed=ep+random` 换布局，`evaluate()` 用
  `eval_seed = (eval_seed*31 + step) % 2^31` 让每次 eval 序列不同
- **`world_model.py` + `train.py` + `bounded_replay.py`: wm 退化根因修复（T=1 序列近似）**:
  wm 更新一直把每个 transition 当 T=1 独立样本训练（train.py 注释 "proper sequential
  replay comes later"），GRU 递归从未在训练中展开，多步想象从未被要求正确 → posterior
  塌缩（kl 卡在 free-nats 下限 0.500）、recon 退化为常数帧、counterfactual reward 恒 0，
  Stage 11-14 全程 wm=0.500。修复三件套：
  1. `compute_loss()` 新增 `next_obs_seq` 参数 + one-step-ahead 预测项（posterior 状态
     imagine 下一帧必须重建 next_obs，给 z 前向预测压力）
  2. `HotRingTier.sample_sequences()`: 热层按时间序写入，连续索引即真实轨迹段；按
     起点滚动窗口，跳过跨越 done 边界的段（Axiom 1 有界）
  3. `train.py`: wm 更新改调 `replay.sample_sequences(wm_bsz, T=max_rollout_steps)`，
     日志新增 `nx=` 显示 next 预测 loss
  附 6 个回归测试（连续性断言、done 边界跳过、短缓冲拒绝、next_loss 梯度流、向后兼容）

### Stage 13 — External Memory (SurpriseDetector + EpisodicReplay) (2026-07-31)

#### 新增
- **SurpriseDetector** (`src/memory/surprise_detector.py`): 集合 4 个 surprise 信号（RND + RSSM reconstruction + Coverage novelty + TD error），用 running avg/std 自适应归一化
- **EpisodicReplayMemory** (`src/memory/episodic_replay.py`): 在 `EpisodicMemory`（in-memory embedding）之上叠加 `ColdShardTier`（SSD 分片全量 Transition），支持 surprise-gated cold-tier 归档 + 基于 TD 损失的 replay 采样
- **Config** (`configs/stage13_external_memory.yaml`): 三档预设 `local_smoke`/`home_64g`/`cloud_24g`，`developmental_memory.enabled=true`
- **Train loop**: SurpriseDetector 集成到 `memory_manager.store_experience()`；EpisodicReplayMemory 冷层存储 + TD-loss 采样；extras 行 `episodic=N/capacity` 日志

#### 训练结果
- **远端 RTX 3080 Ti 完整训练**: 641K steps (of 1M budget)，~11.5 h，5 轮完整 curriculum cycle
- **Doorkey-5x5 峰值 mean_ret**: 0.665→0.779→0.943→1.030→0.966（5 轮逐轮提升）
- **技能库饱和**: skills=10496/10496（~250K 步后写满），无淘汰策略
- **Eval sr=0.00 全程**: mean_ret 虽升至 0.966，但独立评估从未报告 door-open 成功
- **Coverage 仅 2.3%**: 行为多样性坍缩于最优策略附近
- **World Model 未学习**: wm=0.506 (r=0.000, kl=0.500, rew=0.006)
- **最终 VRAM**: 6.90 GB（Qwen-7B 4-bit 约 5 GB 占用）
- **详细报告**: `docs/stage13_report.md`

#### 修复
- **`hierarchical_policy.py:301`**: `skill_delta` 在权重被降级到 CPU 后缺少 `.to(device)`，导致设备不匹配崩溃
- **`train.py`**: 添加 `torch._dynamo.config.suppress_errors = True` 作为 TorchDynamo compile 回退
- **`stage13_external_memory.yaml`**: `developmental_memory` 从 `external_memory:` 嵌套下移到顶层 YAML key

### M2 技能复用闭合 + Stage 11 训练验证 (2026-07-30)

#### 新增
- **M2 skill-reuse 检索回路** (`src/memory/skill_library.py`, `src/train.py`):
  - `SkillEntry.key_embedding`: 存储技能创建时首步隐状态作为检索键
  - `BoundedSkillLibrary.retrieve_by_embedding()`: 用当前观察 embedding 在 GPU 层找最相似技能（余弦相似度 ≥0.6）
  - 每 episode 首步：模型前向拿 hidden state → 检索匹配技能 → 注入 LoRA residual
  - 成功 episode 结束时：将首步 key_embedding 存入新技能
  - `[skills] retrieved skill #N (sim=0.xxx)` 日志
  - key_embedding 随 checkpoint 序列化/恢复
- **Stage 11 远端训练验证**: doorkey-5x5 SR 峰值 0.985，课程循环 7 次完美切换

#### 修复
- **课程切换不生效** (`src/train.py:1018-1024`): `AutoCurriculumConfig` 缺 `mode` 参数，默认 `"lp"` 导致 `peek_next()` 返回 None——任务永远不切换。添加 `mode=str(curriculum_cfg.get("mode", "lp"))`

#### 新增
- **MiniGrid SR 评测** (`src/eval/independent_evaluator.py`): `_measure_minigrid_sr` 用 MiniGrid-DoorKey-5x5 评测通关率，写入 `EvalReport` 和 JSONL
- **`_force_next` 强制评测**: Evaluator 构建后第一次检查立即触发评测（代码更新重启后不需等待边界）
- **Catch-up 逻辑**: `should_evaluate` 在代码更新重启后自动补齐漏掉的评测
- **配置**: `eval_every_steps` 从 50000 降至 25000，更细粒度监控
- **`HierarchicalActorCritic`** (`src/models/hierarchical_policy.py`): 双层结构，Manager（高层）每 K=10 步生成子目标，Worker（底层）每步输出动作 + FiLM 子目标条件化
- **`ManagerHead`**: 子目标生成 + Manager value head
- **`compute_intrinsic_reward`**: Worker 内在奖励 = -||h_t - g||²（隐空间距离），用于训练导航能力
- **`compute_sub_goal_loss`**: 自监督子目标预测损失（预测 K 步后的 hidden state）
- **配置文件** `configs/stage11_hierarchical.yaml`: 启用 `use_hierarchical: true`，`sub_goal_every: 10`
- **train.py**: 分层 rollout 跟踪（manager_buffer 存每周期转换）、子目标辅助损失、Worker 内在奖赏

#### 动机
解决 Stage 10 的"拉锯"问题（doorkey 上升 → 导航归零），通过结构上将规划（Manager，env reward）与导航（Worker，goal-progress reward）分离。

#### 文件改动
- `src/models/hierarchical_policy.py`: 重写，新增 ManagerHead + 完整双层 forward
- `src/models/__init__.py`: 导出 HierarchicalActorCritic
- `src/train.py`: 分层模型构造 + rollout 周期跟踪 + 子目标辅助损失
- `configs/stage11_hierarchical.yaml`: 新配置

### 踩坑记录: Stage 9 符号/逻辑引擎修复 (2026-07-26)

#### 现象
`rules=0 logic=0` 持续 200k 步。符号层和逻辑引擎虽已激活但从未产出。

#### 根因 (5 个 bug)

**Bug 1 — 符号层设备不匹配 (neural_symbolic.py + train.py)**
- `train.py:2069` 将 hidden states 移到 CPU (`.cpu()`) 用于 episode 收集
- `extract_rules()` 调用时未移回 GPU, 导致 `self.rule_projection(h)` 抛 RuntimeError
- 异常被 `train.py:2414` 的 `except Exception: pass` 静默吞掉
- **修复**: `extract_rules` 入口统一 `hidden_states = [h.to(_device) for h in hidden_states]`

**Bug 2 — buffer.rewards 类型错误 (train.py)**
- `buffer.rewards` 是 (T, N) 批次格式, `.tolist()` 得到 `[[r0],[r1],...]` 而非 `[r0,r1,...]`
- `step_advantages = [r - mean_ret_val for r in buf_rewards]` 对嵌套列表报 `unsupported operand type(s)`
- **修复**: 展平: `buf_rewards = [r[0] if isinstance(r, list) else r for r in buf_rewards]`

**Bug 3 — _rule_matrix 设备不一致 (neural_symbolic.py)**
- 修复 Bug 1 后新增规则存 GPU embedding, 旧规则仍为 CPU
- `_rebuild_matrices` 用 `r.condition_embedding` 直接赋值, 混合 CPU/GPU
- `F.cosine_similarity` 在不同设备张量间报错
- **修复**: `_rebuild_matrices` 中统一 `.to(_dev)`, `add()` 中 `_matrix.to(embedding.device)`

**Bug 4 — logic_engine.add_rule() 参数错误 (train.py)**
- 实际签名: `add_rule(quantifier, variable_name, condition, action)`
- 错误调用: `add_rule(condition=..., conclusion=...)`
- **修复**: 传入正确的 `Quantifier.EXISTENTIAL` + `variable_name` + `action`

**Bug 5 — logic_engine.define_variable() 参数错误 (train.py)**
- 实际签名: `define_variable(name, var_type, category_embedding)`
- 错误调用: `define_variable(name=...)` (缺 `var_type` 和 `category_embedding`)
- **修复**: 传入 `VariableType.STATE` + `rule.condition_embedding`

#### 教训
1. **不要 `except Exception: pass` 静默吞错**——改为 `logger.warning` 至少能看见
2. **跨模块调用先读签名**——不要假设参数名, 直接看源码
3. **`except: pass` 是项目最大的藏 bug 窝点**——建议全量 grep 并替换为 `logger.warning`

#### 相关 Bug (本日)
- **EWC 设备不匹配**: `OnlineEWC.state_dict()` 存 CPU, `load_state_dict()` 只 clone 不移回 GPU → consolidate 时 CPU/GPU 张量混合。修复: `load_state_dict` 加 `device` 参数; `consolidate` 用 `model.parameters` 推断设备
- **Stage 6 ckpt_dir 未导入**: train.py 退出段 `backup_stage` 调用 `ckpt_dir()` 但 import 漏引 → `NameError`, `[fossil]` 封印行丢失

### 修复: Stage 6 退出段 `ckpt_dir` 未导入导致备份+化石封印失败 (2026-07-24)
- `train.py:101` 补 `ckpt_dir` 导入, 原 `from src.platform import ...` 漏引,
  导致退出段 auto-backup `shutil.copy2` 时抛 `NameError`, `[fossil]` 封印行
  也未写入日志。Stage 6 5M 步训练正常结束, 但最终 ckpt 备份未生成。
- `src/train.py`: 第 101 行 import 加入 `ckpt_dir`(与已有 `stage_ckpt_path`
  同源 `src.platform.paths`)。

### 编码纪律 / Coding discipline (2026-07-22)
- **缩进对齐即编译安全**: J-Space timeline 代码中 `try:` 缩进 16 格
  而 `except:` 缩进 12 格,Python 在 `import` 阶段即抛 `IndentationError`,
  任何重启都会直接崩溃。教训:**编辑 Python 块后立即用 `py_compile`
  或 `python -c "import ast"` 验证语法**(尤其缩进敏感的 try/except/with 链)。
- **跨模块参数边界必须交叉校验**: NumberSense 默认 `max_count=10`,
  curriculum 任务 `crowded-heavy-weak` 有 18 个物体——两者从未对齐。
  因为 NumberSense 训练是死代码(未修的 Bug 2),冲突未暴露。修复后
  NLL loss 的 `t >= 0 && t < n_classes` CUDA assert 立刻炸。教训:
  **添加/修改模块时,回查它依赖和被依赖的上下游参数范围。**
- **写代码前先读接口,不假设方法存在**: `IndependentEvaluator._measure_drive`
  假设 `HomeostaticDrives.update()` 和 `.all_satisfied()` 存在,实际接口是
  `tick(novelty, success, ...)` 和 `is_homeostatic()`。修改为按真实接口调用,
  参数从 `env.read_states()` 推导。
- **周期性触发用边界穿越,不用取模**: `step` 以 `rollout_capacity` 批量跳跃,
  `step % X == 0` 几乎永远不触发。所有周期检查(ckpt/log/eval/wm/…)必须用
  `(step // X) > (prev_step // X)` 形式。本次 evaluator 触发即重复此错。

### 发育评估体系设计决策 (2026-07-22)
- **里程碑触发,非步数触发**: 内驱力/好奇心系数的衰减不由 Stage 进度百分比决定,
  而是由 agent 实际跨越的发育里程碑触发。当前 agent 六项里程碑(客体永存 1y /
  直觉物理 2.5y / 数感 3.5y / 手段-目的 1.5y / 心智理论 4y / 系统推理 9y)
  **全部未跨过**(estimated_age=0.0y),因此保持**婴儿期全额内驱力**,不做任何
  衰减。里程碑达标(评分稳定 >0.7)→触发对应的系数衰减。
- **独立评测(观察不干预)**: `IndependentEvaluator` 以好奇/内驱/任务三维度
  评分,权重 2:1:2(可配置),完全关掉内驱力加分只看纯环境信号。评测按参考值跑,
  只输出结果不修改训练参数。评测结果存 `eval_scores.jsonl` 供趋势分析。
- **发育阶段映射(能力驱动,非步数驱动)**:
  - ~3–4M 步 + 客体永存稳 → ≈1 岁幼儿
  - ~4–6M 步 + 直觉物理稳 → ≈2.5 岁
  - ~6–8M 步 + 数感稳 → ≈3.5 岁
  - 8M+ + 进入 3D 环境 → 少年期
- **6 轮发育评测结果总结**: Stage 5(2M 步)和 Stage 6(350k/500k/1.2M/1.35M/1.5M)
  共 6 次,所有分数在 0.0–0.5 间随机振荡,无一次跨过 0.7 门槛。发育仍在早期
  探索阶段,为正常发育过程,不标记为失败。

### Performance: fix 11x `% X < rollout_capacity` → `== 0` + double rollout window
- **11 处 `state.step % X < rollout_capacity` 全改为 `state.step % X == 0`**
  (ckpt 已于上版修复,此次修复 wm / imagination / GR / replay / planner /
  diagnostic / logging / compositional / identity / j-space)。原条件因
  `rollout_capacity=512` 对很多 `X < 512` 恒成立,导致这些模块几乎每步触发,
  产生 5-8× 冗余开销(非质量相关,纯 "多算了")。
- **`rollout_capacity` 512 → 2048**: 每批攒 4 倍经验再 PPO 更新,GPU 一次吃
  更多 → 利用率上升。
- 综合预期 **~2.5–3× 墙钟加速**(ETA 10 天 → 3–4 天),训练质量无损(各模块
  语义本就是 "每 X 步触发一次",非 "脉冲触发")。

### C#8 eval harness + Stage 6 launch fixes / 评测接入与启动修复
- `scripts/eval/run_developmental_eval.py`: 加载 ckpt → 在 PhysicsSandbox 跑
  rollout 收集发育信号 → `DevelopmentalEvaluator` 输出 `estimated_age`。用于
  Stage 5/6 退出复验的 *认知能力* 维度(补现有 exit 标准只衡量系统韧性的盲点)。
- `tests/test_run_developmental_eval.py`: 验证评测脚本的逐步收集→聚合→打分
  数据契约(不需 torch,纯单测)。
- Stage 6 启动连修三处卡点: (1) LLMFusion 改魔搭源 + 严格离线/不阻塞
  (未缓存即 template 模式, 绝不触网); (2) stage6 `hidden_size:128` 对齐
  Stage 5 ckpt; (3) stage6 `env.id: PhysicsSandbox` 对齐 Stage 5 obs 形状
  (64×64×3), 否则 resume 维度冲突。现已 `HomeostaticDrives/EmotionSystem/
  CoreKnowledgeAuxLoss/ImaginationTrainer` 全激活, 跨阶段 resume 成功。
- 打 tag: `v0.5.0-stage5`(fc895cb) + `v0.6.0-stage6`(a59fa5a), 不碰现有
  `-cloud` 占位 tag。
- **存盘 bug 修复 (training throughput + 防数据盘爆)**: `train.py:3119` 的
  `state.step % ckpt_every < rollout_capacity` 因 `rollout_capacity=512 > ckpt_every=200`
  恒成立, 导致**每一步都存一次 ckpt**, 把步速从算力上限 ~60步/秒 压到 ~5.5步/秒,
  并向数据盘狂写数万 ~95MB 文件。改为 `state.step % ckpt_every == 0`,
  `stage6 ckpt_every_steps: 50000`(全程约 100 个 ckpt ≈ 9.5GB)。ETA 由 ~10.5 天
  降至 ~1–1.5 天。
- `llm_fusion.py`: `_resolve_local_path` 修正 modelscope 缓存布局判定
  (`models/` 而非 `hub/`), 离线解析更稳。
- `scripts/eval/run_developmental_eval.py`: 数感里程碑改接**真实 NumberSense 头**
  (从 `ckpt.extra.number_sense_state` 加载权重, 用 `model._last_slots` 调
  `predict_count()`), 无权重时回退 env 行为代理; 新增信号样本计数诊断输出。

### A#1 hardened: public read_states() replaces private-attr access / 稳健化

- `PhysicsSandbox.read_states()` public snapshot (agent/object pos+vel, world
  half-width) added; `train.py` ck_loss + homeostatic drives now call it instead
  of reaching into `_agent`/`_objects`. Vectorization-safe, removes a fragile
  coupling.
- `n_envs==1` gating on cognitive modules intentionally kept: 2D developmental
  path is single-env by design (multi-env only in 3D path). Fully lifting the
  gate would add risk with no training benefit (violates "no green-metric
  refactors"), so the reason is documented in open-gaps, not silently dropped.
- Tests: ck_loss integration + sandbox-signal + milestone suites still green.

### PhysicsSandbox emits C#8 developmental signals / 沙盒产发育信号

- `src/envs/physics_sandbox.py` step() now tracks and exposes three milestone
  signals in `info`: `occlusion_events` (object permanence), `force_motion_pairs`
  (intuitive physics), `count_trials` (number sense, behavioral lower-bound proxy
  from distinct objects contacted per episode). Buffers are episode-scoped and
  reset on auto-reset (bounded, Axiom 1). No env contract break for existing keys.
- C#8 evaluator now scores on *real* env output instead of only synthetic data;
  a random policy reads ~0.25 intuitive-physics, correctly flagging the gap.
- Added `tests/test_physics_sandbox_signals.py` (4 tests, all pass).

### External review risk note archived / 外部评审风险提示入档

- A independent code/roadmap review (2026-07-20) praised engineering discipline,
  doc honesty, and the verifiable milestone ladder; flagged the top risk as
  "whether PhysicsSandbox can承载 higher cognitive eval" — which elevates Step 6's
  3D world + social_teacher from optional to necessary for B#7 / C#8 interfaces.
- Archived in `docs/open-gaps.md` §5 (外部独立评审风险提示) and ROADMAP Step 6.
- Review also noted: M2 reuse lacks usage-count实证 (to be filled when Stage 6
  skill_library has data), train.py is oversized (known tech debt, not touched
  during main line), LLMFusion effect unverified (deferred per ROADMAP:258).
- Review agreed with open-gaps: no new omissions; corrected its stale claim that
  cognitive modules were "only active single-env" — they are now enabled in Stage 6
  config (constrained by 2D single-env architecture, not unactivated).

### Cognitive-module activation timing locked / 认知模块激活时机已定

- Per user decision (2026-07-20): enable the five lightweight cognitive modules
  at their *developmentally appropriate* stage, not all at once. Batched plan
  written into ROADMAP "Stage 6 认知模块分批激活时机":
  - homeostatic_drives + emotion_system: open at Stage 6 start (most foundational,
    infant-stage). Currently absent in stage6 config — to be added when Stage 6
    is (re)launched (does not disturb the running Stage 5 process).
  - causal_discovery / number_sense / rule_induction / core_knowledge_loss: already
    enabled in Stage 6 (child-stage).
  - creativity_orchestrator: defer to after Step 3 (needs mature backbone).
  - llm_fusion: stay deferred to Step 6 end per ROADMAP:258.
- open-gaps A#1 row updated to reflect the locked timing.

### B-class modules activated in Stage 6 / B类模块复核并激活

- Re-audited open-gaps: B#5 (CrossDomainTransfer), B#6 (LLMFusionBridge), B#9
  (CausalDiscovery.intervene do-operator) were ALREADY implemented and wired into
  train.py — only not switched on for Stage 6. The original "unimplemented"
  verdict was a misread of C#8's placeholder interfaces.
- `configs/stage6_consolidation.yaml` now enables `causal_discovery`,
  `number_sense`, `rule_induction`, `llm_fusion` (each has a safe skip/throttle
  path). Only B#7 (ToM) is genuinely unimplemented — needs 3D social-teacher env
  (ROADMAP Step 6).
- `docs/open-gaps.md` corrected to reflect true status. No dead ends remain.

### Open gaps A+C implemented / 未闭合点 A类+C#8 全部实现

- Implemented all 6 A-class engineering items + C#8 evaluation yardstick,
  merged to main (does not affect the running Stage 5 process; changes take
  effect on next training start):
  - C#8: `src/eval/developmental_milestones.py` (estimated cognitive age).
  - A#3/P1: `core_knowledge_demos.py` + `BoundedReplayBuffer.prefill()`.
  - A#4/P2: `src/intrinsic/core_knowledge_loss.py` wired into PPO loss + config.
  - A#2: `BoundedExternalMemory` generic three-tier memory in skill_library.py.
  - A#10: `scripts/home/perpetual_supervise.sh` self-healing supervisor.
  - A#11: VRAM added to train.py PROF line.
  - A#1 core infra (cognitive aux-loss path) landed with A#4.
- Added `docs/open-gaps.md` tracking all 11 gaps + status. Verdict: all solvable,
  none a dead end. Plan order unchanged; C#8 strengthens (not reorders) the route.

### Core Knowledge P1+P2 adopted / 核心先验 P1+P2 解法落地

- Adopted concrete Core-Knowledge injection recipe (no academic breakthrough
  needed): **P1** procedurally-generated core-knowledge demo trajectories seeded
  into `bounded_replay.py`; **P2** differentiable auxiliary losses (object
  permanence / intuitive physics / number sense) wired into PPO total loss.
  P3 strengthens existing slot/number-sense biases. Folded into ROADMAP Step 3
  and `docs/path-to-northstar.md` §3/§4. Reordered priorities accordingly.

### Training-plan order optimized / 训练计划顺序优化

- Reordered ROADMAP into a dependency-driven global roadmap:
  1) Stage 5 → 2) Stage 6 main (永续 + B想象 + A分层记忆 + **B方案5模块接loss并产出谓词**)
  → 3) Core Knowledge 提前注入(Stage6后、支线前)→ 4) Y1 神经符号(复用Stage6谓词,不重复接loss)
  → 5) MiniGrid → 6) 3D + LLMFusion。
- Fixes prior issues: B-plan 5 modules now a first-class Stage 6 step (not buried
  in LLMFusion preconditions); Core Knowledge moved from "last" to "before
  cognitive sub-steps"; Y1 grafted directly onto Stage 6's predicate output to
  avoid duplicate loss-wiring.

### Neuro-symbolic Y1 folded into training plan / 神经符号 Y1 纳入训练计划

- Added ROADMAP "Neuro-symbolic path (Y1)" subsection under the Post-Stage-6
  cognitive branch: external `kanren`/`clplog` engine + neural predicate
  extraction + reinforce/clone learning back (sidesteps the unsolved gradient
  interface X). First validation domain: physics puzzles / simple algebra.
  Listed as the first cognitive engineering task after Stage 6, not blocked on
  open research.

### Neuro-symbolic interface has a workaround / 神经符号接口解法

- Clarified in `docs/path-to-northstar.md §1.4`: the "unsolved 30-year problem"
  is the neural↔symbol **gradient interface (X)**, NOT the lack of real symbolic
  reasoning in karbon (Y). Y is pure engineering gap with mature backends
  (Prolog/miniKanren/Z3). Adopted **Y1**: external symbol engine (recommend
  `kanren`) + neural predicate extraction + reinforce/clone learning back —
  sidesteps X entirely. First validation domain: physics puzzles or simple
  algebra. Industry 15y-level results (AlphaGeometry/Proof) all sidestep X.

### A+B scaling strategy adopted into training plan / A+B 规模解法纳入训练计划

- Chose **A+B combination** as the official answer to the scale-vs-Axiom-1
  conflict: B = Dreamer imagination training (already enabled in Stage 6 config,
  gradient back to main model); A = bounded hierarchical external memory
  (generalize Stage-4 LoRA skill library into GPU/CPU/SSD retrieval-injected
  memory). Documented in `docs/path-to-northstar.md §1.6`.
- Folded A+B into the **ROADMAP Stage 6 training plan** (not optional extras):
  B is live via `imagination.enabled`; A is a Stage-6 deepening task. Rejected
  fallbacks C (bounded MoE) / D (relax Axiom limit) kept on record only.

### Path-to-North-Star analysis / 通往北极星路径分析

- Added `docs/path-to-northstar.md`: maps the five common AI routes to karbon's
  actual code state and derives a combined plan (World-Model + Neuro-Symbolic +
  Core-Knowledge + frozen-LLM-anchor) that does **not** violate any PLAN Non-Goal.
- **Code-fact correction**: Dreamer-style imagination training
  (`src/training/imagination_trainer.py`, called at `train.py:2630` with gradient
  flowing back into the main model) is **already wired in** — it is merely gated
  by `imagination.enabled` in the yaml, currently off in Stage 5. So path-2
  completion is near-zero-effort (flip the switch), not a rewrite. Corrects the
  prior assumption that `imagine_step` was "unused".
- Also documents: Axiom-1 conflict with 500M–1B naive scaling; MuJoCo swap would
  break the developmental chain; neuro-symbolic bridge is cosine-match not real
  unification (accurate prior critique).

### LLMFusion timing decision / LLM 融合激活时机

- **Corrected a misstatement**: `src/models/llm_fusion.py` is a **local offline
  frozen Qwen-7B** (4-bit, ~5 GB, no external API; template-mode fallback), not
  a cloud dependency. Verified by reading the code (`_try_load_llm`).
- **Decision**: keep LLMFusion **deactivated during Stage 5–6 main line**; defer
  to the late B-plan "language emergence" phase. Raising `call_interval_steps`
  frequency / model size now would only slow training and destabilize the
  forming policy. Config stays at defaults. Rationale documented in ROADMAP
  Post-Stage-6 section.

### Stage 6 config + Roadmap branch clarification / Stage 6 配置与支线定位

- **`configs/stage6_consolidation.yaml` corrected to resume from Stage 5:**
  switched `hybrid_n_layers` 3 → 7, replaced the (incompatible) MiniGrid task
  list with the Stage-5 PhysicsSandbox difficulty ladder (same 64×64 obs / 8-action),
  bumped replay capacities to Stage-5 scale (hot 16384 / warm 131072 / cold 32×16384),
  and added `train.total_steps: 5000000` (was missing). Keeps Online EWC +
  generative replay + sleep loop.
- **ROADMAP: split cognitive branch from the developmental main line.** Stage 6
  now explicitly *continues on PhysicsSandbox* (the developmental backbone M1–M5);
  MiniGrid 3D + language is documented as a **post-Stage-6 cognitive branch**
  (instruction-following + sparse-reward multi-step planning — needed for the
  8–15y North Star but not on the critical path). Noted MiniGrid obs is
  incompatible with the Stage-5 vision encoder and needs its own adapter.

### Goal clarification / 目标澄清

- **Confirmed North Star: human 8–15-year-old intelligence, reached *from
  scratch* via autonomous developmental growth** — the goal is the altitude,
  but the *binding constraint is the path* (must grow like a human, not be
  pre-trained). Rewrote `PLAN.md §1.5` to state this North Star explicitly and
  frame **M1–M6 as a verifiable developmental ladder leading toward it** (not a
  replacement for it). Honest framing kept: a serious attempt, not a guarantee.
- **Verified Stage-4 exit criteria against `ckpt_stage4_002000000.pt`:**
  GPU-bound criterion **PASS** (GPU tier = 256 ≤ K under 2M steps);
  skill-reuse criterion **NOT MET** (`usage_count == 1` for all skills —
  library is add-only, no retrieval/re-apply path in `train.py`). Recorded as
  milestone **M2** gap; blocking Stage 5. ROADMAP Stage 4 annotated.

### M2 skill-reuse loop implemented / M2 技能复用闭环已实现

- **Closed the M2 gap with a genuine reuse path (not a counter-only hack):**
  - `src/memory/skill_library.py`: added `retrieve()` (cosine over flattened LoRA,
    reuse of `_flatten`) and `sample_for_injection()` (score-weighted pick from the
    GPU tier) so a stored skill can be selected and re-applied.
  - `src/train.py` `HybridActorCritic.forward`: added `skill_delta` LoRA-residual
    injection (`z = z + skill_delta.apply(z)`) — dims match skill_shape
    (d_in=d_out=d_model=128).
  - `src/train.py` rollout loop (Stage 4): at episode start inject a stored skill
    into the policy; at episode success, `record_use` the injected skill (real
    reuse → `usage_count > 1`) and distill a new candidate via `retrieve` +
    `_merge`/`add` to avoid duplicates.
  - By the goal-first rule this is the genuine reuse path (skill affects behavior),
    not the cheaper count-only shortcut.
- **Pre-existing crash fixed (blocked M2 verification):** `wm_last_loss` init dict
  at `train.py:1781` lacked the `'reward'` key, crashing the final summary log under
  smoke-only runs where the wm-update block is skipped. Added `'reward': 0.0`.
- **Tests:** added 5 M2 cases to `tests/test_skill_library.py`
  (`test_retrieve_*`, `test_sample_for_injection_*`, `test_m2_reuse_loop_records_usage`);
  all pass. Full `tests/test_skill_library.py` and `tests/test_stage4_skills.py` green.

### Known failing tests (pre-existing, TODO — not from planner/reward work)

- `tests/test_stage3_wm.py::test_stage3_config_validates`:
  `ConfigValidationError: unknown keys under model: ['slot_dim',
  'slot_num_iterations', 'slot_num_slots', 'use_slot_attention']` — config
  schema (`src/utils/config_schema.py`) doesn't declare SlotAttention keys.
- `tests/test_model_growth_v2.py::TestGrowerV2PlateauLP::test_spike_is_forgotten_so_growth_can_refire`.
- Both confirmed failing on `4c8d485` (before the planner-disable / reward work),
  so unrelated to this change. Fix or follow-up needed.

### Added (Stage-2 N-env vectorization — throughput)

- **N parallel 3D homes via `src/envs/vec_three_d_world.py`** (`VecEnv` generic
  serial wrapper + `VecThreeDWorld`, each sub-env seeded `base+ i`). One
  `env.step` returns a batched `VecStep` (obs `(N,H,W,C)`, reward/terminated/
  truncated `(N,)`); auto-resets each sub-env on its own `done`.
- **Batched actor-critic rollout.** The single `model(obs_t)` forward now takes
  `(N,3,H,W)` and returns `(N, A)` logits / `(N,)` value; `dist` is sampled
  for all envs at once. World-model curiosity + intention + count-based
  `expl_bonus` are batched too, so the GPU forward cost grows sub-linearly
  with `N` while env-steps/iter rise ~`N×`.
- **`RolloutBuffer` now stores `(T, N, *obs)`** and `as_batch()` flattens to
  `(T*N, *obs)`. `compute_gae_vec(rewards,values,dones,last_values,...)`
  computes GAE independently per env column in pure-tensor form (no `.item()`
  syncs) and is flattened for the PPO update.
- `phase1_infant_home.yaml` now sets `env.num_envs: 8`.
- `tests/test_vec_env.py` (CPU, no mujoco) covers VecEnv shapes/auto-reset,
  `(T,N)` buffer layout + flatten, and per-column parity of `compute_gae_vec`
  vs the scalar `compute_gae`.

### Changed (Stage-2 9M resume tuning / 续跑调参)

- **9M 续跑超参（config-only，本地已改、push 后远程 resume 生效）**：
  `exploration_bonus.coef 0.1→0.5`（加强探索奖励以突破 ~102 平台）、
  `train.entropy_coef 0.03→0.015`（收紧熵约束促使策略收敛）、
  `train.total_steps 9000000→11000000`（抬高上限使 9M resume 实际续跑而非立刻退出）。
  9M 终值 `mean_ret=101.4`、`entropy≈2.0` 印证 Plateau + 策略仍未收敛，需此干预再突破。

- **9.5M 第二轮调参（回退过强探索，促收敛）**：
  9M→11M 续跑在 100.8~101.1 震荡、封顶恰在 9M 旧峰值 101.4 之下；诊断确认
  `exploration_bonus.coef=0.5` 把 `entropy` 钉死在 ~2.0、阻止策略结晶。
  故回调：`exploration_bonus.coef 0.5→0.2`（退掉过度探索、留防死区）、
  `train.entropy_coef 0.015→0.008`（更强收敛压力，让熵干净跌破 1.5）。
  从 `ckpt_stage2_009520192.pt` resume，`total_steps` 维持 11M。

- **9.66M 回退第二轮调参（回归，撤销 358b432）**：新配置（bonus 0.2 / ent 0.008）
  实测 `mean_ret` 仅到 99.8 即回落至 99.5，严格差于旧配置（bonus 0.5 / ent 0.015）的
  100.8~101.1——探索奖励是**有用的奖励信号**而非单纯的"熵压制"。三种配置对比：
  原始(0.1/0.03)→101.4、9M-resume(0.5/0.015)→100.8~101.1、本轮(0.2/0.008)→99.5。
  **~101.4 是当前架构/预设的能力天花板**，超参仅在 99.5~101.4 内挪动、无法突破。
  故回退 bonus 0.5 / ent 0.015，`total_steps` 提到 13M 防自动停。

### Fixed (Stage-2 ModelGrowerV2 resume correctness / 续跑不随机化)

- **续跑时按检查点层数构建模型+生长器（防止 3 层权重误载入 2 层模型→随机初始化）。**
  原 `train.py` 用 config 的 `hybrid_n_layers`(=2) 构建模型与 `ModelGrowerV2`，而
  `ckpt_stage2_011000384.pt` 是 **3 层**模型；`model.load_state_dict` 因 `backbone.blocks.2.*`
  尺寸不匹配报错 → "starting model fresh" → 模型被**随机重新初始化**，生长器随后 2→3
  把随机模型当已训练模型继续训（表现为 `mean_ret` 续跑后暴跌到 ~70~82 且不再回升）。
  修复：续跑前用新增的 `_ckpt_layer_count(resume)` 窥探检查点的 `backbone.blocks.N` 层数，
  按该层数构建模型与生长器（`initial_layers` 同步），使 3 层权重正确载入。
  新增 `tests/test_resume_layer_count.py`（合成 2/3 层 state_dict + 缺失文件/无 model_state 用例）。
  实证：续跑日志出现 `Resume ckpt has 3 layers; building model+grower to match` 且
  `Model: HybridActorCubit (... layers=3 ...)`，首步 `mean_ret≈103`（真实权重，非随机）。

- **续跑生长冷静期用 `resumed_step` 而非 `state.step`（防续跑探索重置谷里误触 3→4）。**
  原冷却逻辑在 `state.step` 仍=0 时读取，导致 `last_growth_step` 被设为 0、1M 冷却形同虚设，
  续跑后仅数千步即在 `mean_ret` 探索谷（~73）触发 `grown to 4 layers`。
  改为在 `resumed_step` 解析后设置 `last_growth_step = max(原有, resumed_step)`，
  使下一次真实生长被 1M 冷却挡到架构真正平台期。当前 4 层任务已自发突破并自动备份
  （`watch_backup.sh` → `/root/autodl-tmp/karbon/backup/`）。

### Changed / Fixed (Stage 2 → Stage 3 饱和切换 / Saturation → World Model)

- **Stage 2 生长达到能力天花板（确认饱和，停止生长）。**
  6 层在 `step≈15.14M`（`ckpt_stage2_015160384.pt`）已封顶 `mean_ret≈102.47`；
  5→6 增益仅 +0.4（102.0→102.5）已趋平。6→7 在 `step=16142912` 触发后实测
  `mean_ret=100.44`、`lp=0.017`、`rmax=102.18` → **增益 = −0.29（负）且 `lp<0.05`**
  → 判定 SATURATED。6 层为最优架构，切换前已将最优检查点
  `ckpt_stage2_016140352.pt`（6 层最后状态，d_model=128）三重备份：
  live `/root/karbon/checkpoints/` + 系统盘镜像 `/root/karbon/backup/saturated_6layer_*`
  + 常规备份 `/root/autodl-tmp/karbon/backup/saturated_6layer_*`（sha256 一致）。

- **修复 `train.py` 世界模型反向传播变量名错误（`wm_loss` → `wm_out["loss"]`）。**
  `src/train.py` Stage-3 世界模型更新段误用未定义的 `wm_loss.backward()`，导致
  `NameError: name 'wm_loss' is not defined`。改为 `wm_out["loss"].backward()`
  （`wm_out` 由 `wm.compute_loss(...)` 返回）。否则 Stage 3 任何训练步必崩。

- **修复 `train.py` MiniGrid 分支提前引用未定义变量 `obs_shape`/`num_actions`。**
  `src/train.py` 在 `env.reset()` 之前于 `logger.info` 引用尚未赋值的 `obs_shape`/
  `num_actions`，走到 MiniGrid 分支即 `UnboundLocalError`。改为先 `reset()` 取
  `observation_shape`/`action_space_n` 后再统一打印；其他分支日志不受影响。

- **`stage3_world_model.yaml` 补齐与 Stage 2 一致的模型/编码器参数（确保 backbone 继承）。**
  Stage 3 config 原缺 `hidden_size` 与 SlotAttention 开关，导致用 preset 默认
  `hidden_size=256` 且 CNN encoder，与 Stage 2 的 d_model=128 + SlotAttention 权重
  尺寸全错 → `load_state_dict` mismatch → "starting model fresh"（6 层成果丢失）。
  补齐 `hidden_size: 128` + `use_slot_attention: true` + `slot_num_slots/slot_dim/
  slot_num_iterations` + `use_vision_encoder: false` + hybrid 子参数，与
  `phase2_infant_exploration.yaml` 的 `model:` 块完全一致；并加 `train.total_steps: 5000000`
  （原缺失 → 默认 200 步即停）。env 沿用 `PhysicsSandbox`（与 Stage 2 同分布，backbone
  直接迁移）。切换后日志确认 `Model: HybridActorCritic (d_model=128, layers=6 + SlotAttention)`
  且 `Cross-stage resume: loaded weights from stage 2 ckpt`，backbone 权重完整载入。

  **重要修正（原"RSSM win"措辞不实）**：v1 训练实测 `mean_ret` 从 102.47 天花板
  稳定爬升至 **105–106**（step 150k→490k 全程站上 105，未回落），但**此突破并非 RSSM
  驱动**——v1 的 `curiosity.mode` 默认 `"none"`，RSSM 只独立训练自身表征（recon 降到
  0.012）、**不进入 PPO 梯度**。突破真实来源是「resume 6 层 backbone 后**不被生长打断**的
  持续 PPO 续训 + RND 探索」让策略真正收敛到比 Stage-2 早停点更高的平台。RSSM 作为策略
  驱动信号尚未验证。

- **备选 `stage3_world_model_v2.yaml`（已备未启用）：让 RSSM 真正驱动策略。**
  v1 证明 backbone 继承 + 不间断续训可破 102.47 天花板；v2 进一步把 RSSM 变成真实辅助信号：
  新增 `curiosity: {mode: rssm_uncertainty, coef: 0.3}`，使 WM 预测误差作为 intrinsic reward
  注入 PPO（样本高效探索，Stage 3 设计本意）；并清理 `model` 块里 `hybrid_*` 的重复键
  （28–42 行曾重复定义 n_heads/swa_window/ttt_mini_batch/dropout，后写覆盖前写，实际生效
  swa_window=16 / ttt_mini_batch=8）。env/backbone 与 v1 完全一致，可直接 resume v1 最优
   检查点切换。待 v1 跑到 ~1M 步确认平台后，再决定是否用 v2 重启验证 RSSM 驱动增益。

- **Stage-3 三版全测 + 结论：6–7 层 hybrid(d_model=128)+SlotAttention 在 PhysicsSandbox
   的真实能力上限 ≈ 101–102，无任何组合稳定突破 Stage-2 的 102.47 天花板。**
   - **v1（RND，已停）**：开局 102.47 → 稳态 **99.5**（退化，RSSM 未进 PPO 梯度）。
   - **v2（`rssm_uncertainty` 好奇心，已停）**：开局 ~105 → 平台 **100–102** → 2M 后**滑落**
     （RSSM 仅作探索奖励，仍不驱动策略）。
   - **v3（v2 + `imagination.enabled=true`，RSSM 直训 actor/critic，已停）**：开局 **105.6**
     → 回落 **99**。Dreamer 式想象训练给出 transient 提升，但终值与 v1/v2 同落 ~99–100。
   - **v4（v3 + `model_growth` 重新开生长，已停）**：resume 2.3M 6层 ckpt 后 **6→7 立即触发**
     （cooldown 重置），7 层稳在 **100.7–101.1**，略高于 v3 的 99 但**未破 102.47**；
     `slope` 衰减到 ~0.95 进入平台，`rmax` 卡 104.77。
   - **四版对比**：v1=99.5 / v2=100–102(slide) / v3=105.6→99 / v4=101±1。
     WM、imagination、层数都只改变 **transient 动态**，无组合稳定突破 102.47。
     `max mean_ret` 各版均为开局 transient 尖峰（v4 max=109.99 @ step1），非真实平台。
   - **结论**：当前 6–7 层架构 + PhysicsSandbox 的**硬上限 ≈ 101–102**；继续训练 v4 信息增量≈0。
     真正破局需换**架构维度**（更大 `hidden_size` 128→256 / 更长 `imagination_horizon` 8→15）
     或**换 env 难度**，而非层数/WM/imagination 的现有组合。
    - v4 最优 7层 ckpt `ckpt_stage4_7layer_002460160.pt` 已三重备份（live + 两 backup 目录，
      sha256 `ff103b15…e802` 一致）；v3 6层最优 `ckpt_stage3_002300416.pt` 亦留档。

- **Stage-3 v5（宽度实验，否定宽度假设）：256 宽 7层 fresh 训练，稳态仍 ≈100.4，未破天花板。**
  为验证「~102 是否为容量/回报估计精度瓶颈」，v5 将 `hidden_size` 128→256（fresh 起 7层，
  关 growth，沿用 v4 的 RSSM+imagination+env）。结果：
  - step 32k 爬到 97（`slope=5`）、step 119k 摸到 `max=101.28`、step 272k **`slope=0` 进入平台**，
    稳态 **`mean_ret≈100.4`**——与 v4（128 宽 7层）的 100–101 **完全重合**。
  - **结论：宽度假设被否定。** 256 宽未带来任何突破，证明 ~102 不是 backbone 容量或回报估计
    精度问题（加宽无效）、也不是层数（6→7 无效）、也不是 WM/imagination（v1-v4 已证）。
    `cov=100%` 表明状态空间已完全覆盖，可达回报上限由 **PhysicsSandbox 的回报结构本身** 锁定。
  - **最终 Stage-3 结论**：PhysicsSandbox + 当前 hybrid(d_model 128/256)+SlotAttention+RSSM 栈的
    **真实能力上限 ≈ 100–102**，与环境回报结构绑定，非同维度容量可破。继续训练 v5 信息增量≈0。
  - **真正破局方向（换维度，非加容量）**：(a) 改 env 本身（更多可交互对象 / 更高回报上限）；
    (b) 改 reward 设计（当前回报信号限制了可达上限）；(c) 加 RSSM **反事实规划**（counterfactual
    planning，DEV_PLAN 提及但 v3/v4 的 imagination 仅是直训 actor，未做规划打分）。
   - v5 最优 256宽 7层 ckpt `ckpt_stage5_wide256_000260096.pt` 已三重备份（live + 两 backup 目录，
     sha256 `52c57cde…c586` 一致）。

- **Stage-3 v6（修复想象奖励 bug，仍无效）：证明 ~102 非想象奖励信号所致，RSSM 想象训练
   无法破局。** `ImaginationTrainer` 原用 reconstruction-error 当想象奖励
   （`imagined_r = -recon_err * 0.1`），与 PhysicsSandbox 真实回报无关。v6 改为
   `world_model.predict_reward(state)`（RSSM reward head，已在 `compute_loss` 用真实 replay
   reward 监督训练），使 Dreamer 式想象训练优化**真实回报**。结果：
   - 开局从 94.7 冲到 **`max=106.39`**（比 v4 错误奖励的开局 101 更高），但 step 2683k
     **`slope=0` 进入平台**，稳态 **`mean_ret≈101.9`**——与 v4（错奖励）的 101±1 **完全重合**。
   - **结论：修复想象奖励 bug 未破局。** 开局 transient 更高（106 vs 101），但稳态仍锁死在
     ~102 同一平台。这彻底证伪「v3/v4 无效是因想象奖励信号错」的假设；RSSM 想象训练无论用
     错/对奖励，都无法让 PPO 策略突破 PhysicsSandbox 的 ~102 回报上限。
   - **关键推论（削弱 counterfactual planning 先验）**：v6 已证明「在想象里用真实回报优化策略」
     对破局部最优无效，而 counterfactual planning 同样依赖 RSSM 在想象里评估回报来选动作——
     若想象回报评估对策略更新无益，planning 也未必更有用。剩余破局方向优先级重排：
     **(b) 改 reward 设计（定向拉高局部最优陷阱的梯度，最可能被低估）> (a) 换 env > (c) 规划**。
    - v6 最优 7层 ckpt `ckpt_stage6_imagfix_002680320.pt` 已三重备份（live + 两 backup 目录，
      sha256 `1d03de5d…7db3` 一致）。Stage-3 架构维度实验（层数/宽度/WM/imagination/奖励信号）
      **至此全部穷尽且均收敛 ~100–102**，确认该上限由环境回报结构 + 局部最优陷阱锁定。

- **Stage-3 v8（counterfactual planning / MCTS，仍仅 transient）：System 2 规划也未破局。**
    `CounterfactualPlanner` + `LongRangePlanner`（MCTS over RSSM）在 train.py 已接线
    （lines 1366-1375, 1507-1516, 2805-2826），用（v6 修好的）`predict_reward` 想象评估候选
    动作序列并覆盖 PPO 动作。v8 启用二者于 v6 的 7层 ckpt 上。结果：
    - 接管初期从 83 冲到 **`max=106.53`**（step 2695168），显著高于 v6 的 101；
      但 step 2730k 后回落到 **`mean_ret≈101.6`**（`cov=100%`, `slope` 缓降至 ~1.1），
      稳态与 v4/v6 的 101±1 **完全重合**。
    - **结论：MCTS 规划仅产生 transient 尖峰（106→101），未抬升稳态。** 印证 v6 的推论：
      planner 短期用真实回报选出更优动作，但 PPO 的梯度更新持续把策略拉回同一局部最优，
      两者拉锯后净稳态仍是 ~101。所有 v1-v8 版本**峰值均 ~106（transient）、稳态全锁 101±1**。
    - **最终 Stage-3 全维度穷尽结论**：层数(6/7) / 宽度(128/256) / WM / imagination(错+对奖励)
      / MCTS 规划 —— 全部只产生 transient 尖峰，无法抬升 PhysicsSandbox 的 ~101-102 稳态。
      上限由**回报景观的局部最优陷阱**锁定：当前 reward 对"被动蹭物体"（物体自身惯性/碰撞
      白送速度奖励）给分，agent 陷在"推几个物体"的局部最优，不主动最大化全局回报。
    - **唯一未排除的破局杠杆：(b) 改 PhysicsSandbox reward 设计** —— 直接重塑回报景观
      （移除被动速度白送、奖励 agent 主动加速物体），可能把 106 的 transient 变成稳态。
      这区别于所有"优化/架构"干预（只改 transient），是从景观根源破局。
    - v8 最优 7层 ckpt（规划版）已并入 v6 的 `ckpt_stage6_imagfix_002680320.pt` 保留；
      系统盘清理：删 258 个 stage2 + 625 个 stage0 + 137 个 stage3 密集 ckpt，释放 14G
      （83%→37%，不记文件）。

- **Stage-3 v9（关闭 MCTS planner / 破局成功）：稳态首次突破 ~102 天花板到 103.8。**
    ⚠️ **归因更正**：v9 config 声称做 reward redesign，但 `src/envs/physics_sandbox.py`
    的 reward 改动**从未上传到远程**（部署时只 sftp 了 config，漏传源码文件）。经核实远程
    环境全程使用**旧 reward**（`speed*0.05` 被动速度奖励，未改）。因此 103.8 **不是** reward
    redesign 的功劳。v8→v9 config 的**唯一真实差异是关闭了 MCTS planner**
    （`long_range_planner` / `counterfactual_planner` 从 v8 的 `enabled: true` 改为不启用）。
    - 同一 7层 ckpt `ckpt_stage6_imagfix_002680320.pt` resume、同一（旧）reward，
      **唯一变量 = planner 开/关**：
      - v8（planner **开**）：稳态 `mean_ret≈101.6`（`slope` 缓降未收敛）。
      - v9（planner **关**）：resume 后 slope 转正冲到 `max=104.75`（step 2706944），
        又训练近 20 万步（→ step 2902016）稳定在 **`mean_ret≈103.8`**，
        最后 15 步 `mean=103.75 std=0.06`，`slope` 收敛至 ~0（对称抖动）。
    - **真实结论：关闭 MCTS planner 使稳态 101.6→103.8（+2.2），且更稳、真收敛。**
      机理：planner 用不完美世界模型的 `predict_reward` 覆盖 PPO 动作 → ①想象回报有偏差、
      选的动作对真实环境非最优；②PPO 采到的是"planner 的动作"而非自身策略动作，
      **策略梯度与实际行为脱节**，拖累收敛。关掉后 PPO 端到端学，稳稳爬到 103.8。
      这进一步削弱"System 2 规划有用"的先验（与 v6 推论一致）。
    - **后续动作**：已把 `phase0_protozoan.yaml` / `phase2_infant_exploration.yaml` 的
      `long_range_planner` / `counterfactual_planner` 全部改为 `enabled: false`（附原因注释）；
      `stage3_world_model_v8_cf_plan.yaml` 保留 `true` 作为 planner-ON 失败对照（附注释）。
    - v9 里程碑 ckpt 已三重备份（sha256 `e7749e0e…c4cc` 一致），
      重命名为 `ckpt_stage7_no_planner_002900480.pt`（原 `_reward_redesign_` 名误导，已弃用）。

- **Stage-3 v10（真正的 reward redesign，已验证 / 结论：效果中性，无显著增益）。**
    v10 是首次把 acceleration reward **真正上传远程**（v9 漏传）并训练的实验：从 no-planner
    ckpt resume、无 planner、env 用新 reward（移除 `speed*0.05` 白送，改奖励接触时 `|Δv|`，
    contact 0.1→0.15，cap 2.0→3.0）。训练 ~15 万步收敛到 `mean_ret≈60`（新 reward 尺度，
    不可直接与旧 reward 的 104 比）。
    - 为跨 reward 尺度对比，写了 `eval_policy.py`（固定 seed + 100 episode，双 reward 尺子 +
      行为指标）。**在旧 reward 这把公平尺子上评估两个策略**：
      | 指标（100ep, seed 固定, stochastic） | no_planner（旧reward训练） | v10（新reward训练） |
      |---|---|---|
      | OLD reward（公平尺子） | **97.31** ±19.4 | **97.88** ±13.9 |
      | NEW reward | 59.96 | 63.81 |
      | 接触物体数/ep | 371.5 | 402.9（+8.5%） |
      | active \|Δv\|（主动推力） | 135.8 | 119.9 |
      | agent path len | 8.42 | 7.20 |
    - **结论：reward redesign 效果中性。** 在公平尺子（旧 reward）上 v10 vs no_planner =
      97.88 vs 97.31，差异 <1%、远在 std 内，**统计上持平**。v10 接触物体更多（+8.5%）但主动
      推力（active |Δv|）反而略低——行为风格不同，但**没有证据表明整体能力更强**。
    - **最终 Stage-3 破局归因**：真正抬升稳态的是**关闭 MCTS planner**（101.6→104）；
      reward redesign 只是把 agent 调到"同等强度、不同行为风格"。
    - **收尾**：远程 + 本地 env `physics_sandbox.py` 均已恢复旧 reward（与 no_planner 起点、
      后续 stage config 一致），reward redesign 代码归档在 git 历史（commit 044703a）+
      `stage3_world_model_v10_reward_on_noplanner.yaml` config 保留。`eval_policy.py` 保留为
      跨 reward 策略评估工具。
    - **附加发现**：greedy（argmax）下两个策略都塌缩成"永远动作 6"，轨迹退化；必须用
      stochastic 采样评估才能体现策略差异（已记录在 eval 脚本注释）。
    - no_planner 稳态复测：v9 继续训练到 step 3M，稳态实为 **~104.1**（比早期记的 103.8 略高，
      slope 完全收敛）。Stage-4 从 `ckpt_stage7_no_planner_002900480.pt` 起步。

- **Stage-4 启动 · Bounded Skill Library。**
    从 `ckpt_stage7_no_planner_002900480.pt`（旧 reward 稳态 ~104）跨 stage 3→4 resume，
    在验证过的 no-planner 7层 hybrid + SlotAttention + RSSM 骨干上叠加持久化技能库。
    - **重写 `configs/stage4_skills.yaml`**：旧版是过时的 3层脚手架（无 env/slot/imagination
      块、num_objects 用默认 3）。新版对齐**实际训练线**：7层 hybrid、SlotAttention(7 slots)、
      PhysicsSandbox(num_objects=10)、imagination、`model_growth.enabled=false`、无 planner、
      旧 reward。skills 块：3-tier 有界库（gpu=256 / cpu=2048 / ssd=64×128，总容量 10496）。
    - **schema 修复（`src/utils/config_schema.py`）**：补齐实际训练线一直在用、但 schema 缺失
      的键，修好长期存在的 config 校验失败：
      - `ModelSchema` 加 SlotAttention 字段（`use_slot_attention/slot_num_slots/slot_dim/
        slot_num_iterations`）+ 校验 → **顺带修复 pre-existing `test_stage3_config_validates`**。
      - `EnvSchema` 加 PhysicsSandbox 字段（`num_objects/render_size/gravity/action_force`），
        `num_envs` 改为可选（default 1）。
      - `TopLevelSchema` 加 `curiosity/imagination/model_growth` 三个可选块。
    - **env reward 回退**：v10 评估确认 reward redesign 无显著增益后，远程 + 本地
      `physics_sandbox.py` 均恢复旧 reward（`speed*0.05`，cap 2.0），与 no_planner 起点一致。
    - smoke（200 步）通过：7层+slot 加载 OK、BoundedSkillLibrary 初始化 OK（有界 10496）、
      旧 reward 得分正常。正式训练已启动（tmux `s4`+`s4rec`），前 4k 步 `mean_ret≈110-133`
      （与起点 ~104 平滑衔接，无 v10 那种 reward 尺度突变），`skills` 计数随训练增长（有界）。
    - 新增 `scripts/eval/eval_policy.py`：跨 reward 尺度的策略评估工具（固定 seed、双 reward
      尺子、行为指标），stochastic 采样（greedy 会塌缩成单动作）。
    - 已知遗留失败（与本次无关）：`test_model_growth_v2.py::...test_spike_is_forgotten...`
      （`GrowthConfigV2` 无 `rmax_decay` 参数）。




### Known limitation (first cut)

- Single-env-only cognitive blocks (homeostatic drives, emotion, number-sense /
  rule predicates, knowledge-gap, concept-graph, memory, creativity, LLM fusion,
  RND, skills/symbolic/reflection episode hooks, causal intervention, cross-modal
  bridge) are guarded with `n_envs == 1` and **skipped** when `N>1`. The
  batched hot path (actor-critic + WM curiosity + intention + expl-bonus + replay
  + PPO + imagination + growth) still runs under `N=8`. Set `env.num_envs: 1`
  to recover full single-env module coverage.

### Fixed (Marginal Gains integration — name collision & dead code)

- **Removed duplicate `KnowledgeGapDetector` from `src/models/marginal_gains.py`.**
  It collided by name with the existing, fully-wired
  `src.intrinsic.knowledge_gap.KnowledgeGapDetector`. The `marginal_gains`
  import (`src/train.py`) shadowed the intrinsic one, so `knowledge_gap` was
  constructed as the wrong class and its `get_gap_boost()` / `update()` calls
  silently failed (swallowed by `try/except`) — a regression of the working
  knowledge-gap curiosity boost. The marginal-gains copy was also dead code: an
  unconditional `knowledge_gap = None` reassignment clobbered it right after
  creation, so its `.detect()` was never called.
- `marginal_gains` now exposes only the genuinely-new, non-overlapping modules:
  **`CompositionalTester`** (compositional generalization test over ConceptGraph)
  and **`LearningProgressTracker`** (plateau detection → curiosity boost), both
  already correctly wired in `src/train.py`.
- Added `tests/test_marginal_gains.py` (7 tests, passing).

### Fixed (VisualAnalyzer — silent no-op & wrong trigger)

- **`describe_slot` NameError (`src/models/visual_analyzer.py`).** The f-string
  referenced `{texture}` but the variable was bound as `text`, so every call
  raised `NameError`. Because `train.py` wrapped `feed_to_graph` in
  `try/except Exception: pass`, the whole VisualAnalyzer → ConceptGraph path
  **silently did nothing** (zero nodes written). Renamed to `{texture}` and
  added `tests/test_visual_analyzer.py` (regression for the crash + motion).
- **Motion was always "still".** `feed_to_graph` / `describe_*` re-ran `forward`
  on the same frame, overwriting `_prev_slots` before motion was read, so the
  frame-to-frame diff was always 0. `forward` now caches `_last_out`; the
  describe/feed helpers read the cached result and never re-forward, so motion
  is estimated against the previous step. `train.py` now calls
  `visual_analyzer(slots)` **every step** (updating motion) and only persists to
  the graph every 500 steps.
- **Wrong trigger condition.** `state.step % 500 < rollout_capacity` is true for
  every step (since `step % 500 ∈ [0,499] < 512`), so it never meant "every 500
  steps". Changed to `state.step % 500 == 0`, matching the existing periodic
  hooks in `train.py`.
- `VisualAnalyzer` exported from `src/models/__init__.py`.
- **Not yet trained (known gap):** the classifier heads are randomly initialized
  and not part of the optimizer / checkpoint, so attribute predictions are
  unsupervised heuristics, not learned from SlotAttention. Wiring a supervised
  loss is a follow-up.

### Fixed (Env episode-return metric was an all-history mean)

- `mean_ret` / `summary()["mean_return"]` in `physics_sandbox`,
  `three_d_world`, `social_teacher`, and `minigrid_wrapper` were computed over an
  **unbounded** `_episode_returns` list — i.e. the mean over *every* episode
  since process start. Over multi-million-step runs the value becomes frozen and
  cannot reflect recent agent performance (the 2D `mean_ret=121→114` "drop" and
  the 3D `mean_ret=0.218` were both artifacts of this).
- Added the 1024-episode rolling-window cap (already present in
  `crafter_wrapper.py`) to all four envs. The list now keeps at most the last
  1024 returns/lengths, so `mean_ret` reflects **recent** performance. This also
  resolves an Axiom-1 unbounded-allocation finding in `scripts/ci/check_bounded.py`
  (lines annotated `BOUNDS-OK`).
- `train.py` log line still prints `mean_ret`; it now reads the bounded window,
  giving a truthful trend of recent episodes.

### Fixed (PPO scale-mismatch — 3D "flat loss" root cause)

- The `ReturnNormalizer` docstring promised: *"Denormalizes predicted values
  before GAE advantage computation"* — but the code did not do it. The value
  head is trained on `returns_norm` (normalized scale), so `batch.values` and
  `last_value_t` are also in normalized scale. Feeding those directly into
  `compute_gae` together with **raw** `batch.rewards` produced
  scale-mismatched advantages that could not carry a policy signal — the real
  driver of the 3D "flat loss / near-zero policy gradient" symptom.
- Fix: denormalize `batch.values` and `last_value_t` back to raw scale before
  `compute_gae` (`train.py:2117-2131`). The off-policy replay TD update
  (`train.py:2187-2201`) had the same bug — the TD target is now built in
  raw scale (`next_v` denormalized, raw reward added), then renormalized so
  both sides of the MSE match.
- Extracted advantage normalization into `_normalize_advantages()`
  (`train.py:481-501`): standardizes to ~N(0,1) with a zero-variance guard
  that falls back to raw centered advantages (handles constant-advantage 3D
  case) and also handles NaN std from single-element batches (torch semantics).
- Added `tests/test_ppo_normalization.py` (11 tests, passing) covering
  `ReturnNormalizer` round-trip + EMA behavior, the zero-variance guard, and
  an end-to-end GAE scale-consistency test that demonstrates the fix
  (denormalize-then-GAE recovers the raw-scale advantages).
- Added `tests/test_ppo_integration.py` (3 tests, passing) — proves the fix
  at the *whole-PPO-step* level: with values denormalized before GAE, one PPO
  gradient step actually moves the policy (`approx_kl != 0`, finite grads), and
  higher-reward steps get higher advantages; plus a regression guard asserting
  the OLD bug (normalized value-head output fed straight into GAE) distorts
   advantages so the denormalize step can never be silently removed.

### Fixed (Actor-Critic encoder — GB-scale Linear / 3D memory swing)

- `ActorCritic` (`train.py:123-135`) and both `HybridActorCritic` CNN
  encoder branches (`train.py:218-239`) flattened the **full** H×W
  feature map and fed it to `nn.Linear(32*h*w, …)` with no
  downsampling. At the 3D obs size (256×256) that is
  `Linear(2_097_152, …)` ≈ **0.5–1 GB of weights + an equally
  large gradient** allocated on every update — the source of the
  `~2.57 GB` per-step memory swing seen in the dead 3D run.
- Fix: insert `nn.AdaptiveAvgPool2d((8, 8))` before `Flatten()`
  and size the Linear at the fixed `32*8*8 = 2048` features
  (mirrors the already-fixed `RNDNet`, `src/intrinsic/rnd.py:98-102`).
  The 3D per-step allocation drops from ~1 GB to a few MB.
- Added `tests/test_actor_critic_encoder.py` (5 tests, passing) asserting
  the trunk's first Linear takes `32*8*8` (not `32*h*w`), the encoder
  has a downsample layer, and 256×256 / 64×64 inputs forward with
  correct `(B, n_actions)` / `(B,)` shapes.

### Fixed (Model growth was non-learning / wiped optimizer state)

- `ModelGrowerV2.grow()` claimed to preserve Adam momentum for
  matching parameters, but the copy guard `new_idx < len(new_state["state"])`
  is **always false for a fresh optimizer** (empty `state` dict),
  so the carry was a silent **no-op** — every growth wiped the
  policy optimizer's accumulated momentum, re-setting learning. Real fix
  in `_carry_over_adam_momentum()` (`src/models/model_growth_v2.py`):
  create the entry on the fresh optimizer and copy `exp_avg` / `exp_avg_sq`
  / `step` from the old one. (Caught by a unit test.)
- Distillation now also preserves the teacher's **policy distribution**
  via a KL term on the softmax logits (`model_growth_v2.py:_distill`),
  so a grown model keeps the agent's learned action preferences
  instead of resetting them; `distill_steps` raised 100 → 256.
- Growth frequency made rarer so it stops disrupting the policy:
  `min_steps_between_growths` 100_000 → 500_000 in
  `phase0_protozoan.yaml`, `phase1_infant_home.yaml`,
  `phase2_infant_exploration.yaml`, `phase9_llm_fusion.yaml`;
  `grow_trigger_coverage` 0.3 → 0.5 in phase0/phase1 (growth
  now requires broad exploration first).
- Added `tests/test_model_growth_v2.py` (6 tests, passing): KL distill
  moves the student's policy toward the teacher's, and the momentum
  carry-over helper actually transfers Adam state to a fresh optimizer.

### Added (Phase 0+: 工程补缺口 A-G)

- **Imagination Trainer** (`src/training/imagination_trainer.py`) — Dreamer-style
  world-model-driven imagination training. Uses RSSM to generate N-step imagined
  trajectories, then trains actor-critic on imagined data for ~10x sample
  efficiency (DEV_PLAN.md gap D).
- **Intention Achievement Curiosity** (`src/intrinsic/intention_curiosity.py`) —
  replaces blunt RND with action-conditioned curiosity: compares RSSM prior
  (predicted next state) vs posterior (actual state) to reward states where the
  agent's own model of cause-and-effect fails (DEV_PLAN.md gap C).
- **Knowledge Gap Detector** (`src/intrinsic/knowledge_gap.py`) — tracks per-slot
  (per-concept) prediction accuracy via EMA. Concepts with sustained high error
  are identified as knowledge gaps; curiosity is boosted for gap-related states
  (DEV_PLAN.md gap C).
- **Social Curiosity** (`src/intrinsic/social_curiosity.py`) — predicts caregiver
  next action; prediction error = social curiosity reward. For Phase 3+
  imitation/social learning (DEV_PLAN.md gap C).
- **Audio Encoder** (`src/sensory/audio_encoder.py`) — lightweight mel-spectrogram
  CNN encoder (~0.1 GB). Optional torchaudio dependency with NumPy fallback
  (DEV_PLAN.md gap B).
- **Episodic memory with surprise gating** preserved as `src/models/developmental_memory.py`
  (already implemented — gap G covered by existing MemoryManager).
- All new modules wired into `src/train.py` training loop with config-driven
  enable/disable, checkpointing, and logging.
- New config sections in `configs/phase0_protozoan.yaml`: `imagination`,
  `intention`, `knowledge_gap`, `social_curiosity`, `audio`.

### Changed (Counterfactual Planner — reward proxy fix)

- **RSSM reward head** (`src/models/world_model.py`) — added a bounded
  `reward_head` (predicts `r̂_t = Reward(h_t, z_t)`) and `predict_reward()`
  method. `compute_loss` now accepts an optional `reward_seq` and adds an MSE
  reward-prediction term, trained from replay `reward` in the world-model
  update step (`src/train.py`). This grounds counterfactual planning in
  **objective environment reward** instead of the policy's value estimate
  (Dreamer-style), so System 2 can surface plans the current policy would not
  choose.
- **CounterfactualPlanner** (`src/models/counterfactual_planner.py`) — now
  scores plans via `wm.predict_reward` (removed the broken `policy_model.
  value_head(decoded)` path that mismatched latent/obs dimensions and silently
  no-op'd). Also fixed two correctness bugs: prediction/actual history is now a
  bounded `deque` (Axiom 1), and `planning_accuracy` pairs predictions with
  actuals correctly instead of misaligning via `pop(0)`.
- **Reward-head train/serve gap narrowed** — `compute_loss` now also predicts
  reward from the **prior (imagined) state** (`imagine_step`) in addition to the
  posterior state, since at planning time rewards are scored from prior states.
  This reduces the posterior-vs-prior distribution gap (Dreamer-style).
- **`planning_accuracy` is now apples-to-apples** — `evaluate_plan` returns the
  predicted **first-step** reward (the action actually executed) alongside the
  total plan score; `select_best` records that for validation against the single
  observed reward, instead of comparing a multi-step summed total to a 1-step
  reward. Added `reward_loss_weight` config (`RSSMConfig` + `phase2` yaml) to
  balance the reward term.
- World-model training log now reports `rew=` (reward-loss) alongside recon/kl.

### Fixed (Model growth obs_shape + PPO loss log consistency)

- `ModelGrowerV2._create_larger_model()` hardcoded `obs_shape=(64,64,3)` when
  building the grown `HybridActorCritic`. For any env whose real obs is not
  64×64×3 (e.g. 4-channel / non-square), the grown encoder's first conv
  `in_channels` would be wrong and silently corrupt the expanded network. Fix:
  read the real shape from `model.obs_shape` (with `(64,64,3)` only as a
  fallback). `HybridActorCritic` now also stores `self.obs_shape` at
  construction (`train.py`), which additionally un-breaks
  `imagination_trainer`'s `actor_critic.obs_shape` access.
- PPO log line (`train.py:2551`) printed `loss=` from the **last** minibatch's
  combined loss while `p=/v=/ent=/kl=/cf=` are means over all minibatches —
  inconsistent. Now `ppo_losses["total"]` accumulates the combined loss every
  minibatch and the log reports its mean.
- Added `tests/test_model_growth_v2.py::TestGrowerV2ObsShapeCarryover` (1 test).

### Fixed (3D deadlock guard — state-dependent exploration bonus)

- **New `src/intrinsic/exploration_bonus.py` (`ExplorationBonus`)** — a
  bounded count-based exploration bonus that prevents the 3D training
  deadlock. When the env reward is sparse / near-constant, the value
  head eventually fits it → advantages collapse to ~0 → the policy
  gradient vanishes and the agent stops learning. RND-style intrinsic
  curiosity only patches this while its predictor error stays >0; once it
  converges on visited states the bonus decays to 0 and the deadlock
  returns.
- bonus(s) = `coef / sqrt(visit_count(s) + 1)`: highest for novel /
  rarely-visited states, decays toward 0 as a state is revisited but
  **never reaches 0**. Crucially it VARIES across states AND with
  visitation history, which the value head (sees only the current obs)
  cannot predict → it leaves a persistent residual in the advantages, so
  the policy always has an exploration signal. (A flat *constant* floor
  would be a no-op: advantages are invariant to adding a constant to
  every reward, because the value head fits the constant too.)
- Bounded (Axiom 1): visit counts live in a **fixed-capacity** tensor
  (`capacity` buckets, hashed from a downsampled obs); no unbounded
  growth. Exposes `capacity` + `__len__` for `HealthChecker`.
- Wired into `src/train.py`: `total_r += eb` each step (the bonus
  already carries its own `coef`, so it is added directly — **not**
  multiplied by `intrinsic_coef` again), plus `expl_bonus.update(obs_t)`.
  Decoupled as a **top-level** `exploration_bonus` config key (NOT under
  `intrinsic:`), so enabling it does **not** also switch on RND or
  change the `curiosity.mode` already in use. Enabled in
  `configs/stage1_curiosity.yaml` (the 3D/RND stage-1 run) and
  `configs/phase2_infant_exploration.yaml` (the 2D run):
  `coef=0.1`, `grid=8`, `capacity=65536`.
- Verified offline (CPU, tiny): constant env reward → advantage std
  = 0 (deadlock); constant reward + exploration bonus → advantage std
  > 0 (signal persists). Added `tests/test_exploration_bonus.py`
  (6 tests, passing), including the deadlock-vs-signal proof.

## [v1.1.0-stage7-cloud] - 2026-07-09

### Stage 7 COMPLETED - ALL 7 STAGES DONE

- **Wall time**: 29,274 s (8.1 h)
- **Steps**: 3,000,000
- **Final mean_return**: **0.955**
- **VRAM**: 2.82 GB, slope 0.0 ✅
- **Coverage**: **8.9% (363 buckets)**
- **Skills**: 10496/10496 (877,149 created)
- **EWC**: ✅ consolidated (Fisher L1=1265)
- **Sleep Loop**: replay_trim×292, skills_merge×146, ewc×29

### Key validation
1. **SelfModel** (metacognition) active every step.
2. **NeuralSymbolicLayer** + **LogicEngine** active (no rules extracted due to EWC interference).
3. **ReflectionLoop** fired every 10 episodes.
4. **InnerDialogue** generated natural-language reflections:
   - "I failed this episode with return 0.000." (during EWC interference)
   - "I succeeded this episode with return 0.955." (after recovery)
5. Agent recovered from EWC interference: 0.006 -> 0.955.

### Complete 7-stage journey
- Total wall time: ~58 hours
- Total steps: 21,000,000
- Final mean_ret: 0.955
- Peak VRAM: 2.82 GB / 32 GB (8.8%)
- Skills created total: ~2.5M across all stages
- Coverage growth: 0.1% -> 8.9% (89×)

See `docs/stage7_report.md` for full analysis.

## [v1.0.0-stage6-cloud] - 2026-07-08

### Stage 6 COMPLETED - ALL 6 STAGES DONE

- **Wall time**: 29,242 s (8.1 h)
- **Steps**: 3,000,000
- **Final mean_return**: **0.951** (recovered from EWC interference)
- **VRAM**: 2.81 GB, slope 0.0 ✅
- **Coverage**: **8.6% (353 buckets)** - 86× growth from Stage 0
- **Skills**: 10496/10496 (672,158 created)
- **EWC**: ✅ consolidated (7.26M params)
- **GR VAE**: gr=0.001 (converged)
- **Sleep Loop**: all 4 task types fired

### Complete training summary (Stage 0-6)
- Total wall time: ~50 hours across 6 stages
- Total steps: 18,000,000
- Final mean_ret: 0.951
- Peak VRAM: 2.81 GB / 32 GB (8.8%)
- Coverage growth: 0.1% -> 8.6% (86×)


Format loosely follows [Keep a Changelog](https://keepachangelog.com/).
遵循 Keep a Changelog 惯例。

---

## [v0.5.0-stage5-cloud] — 2026-07-07

### Stage 5 COMPLETED on cloud (AutoDL vGPU-32GB / RTX 4080)

- **Wall time**: 28,945 s (482 min, 8.0 h)
- **Total steps**: 3,000,000
- **Episodes**: 31,491
- **Final mean_return**: **0.937**
- **VRAM**: 2.71 GB, slope 0.0 ✅
- **Coverage**: **7.9% (324 buckets)** — 20× growth from 0.4%
- **Skills**: 10496/10496 (344,012 created, 333,516 merged/evicted)
- **Tasks mastered**: empty-5x5, empty-6x6, empty-8x8
- **Tasks explored**: doorkey-5x5, doorkey-6x6 (not solved)

### Key validation
1. **AutoCurriculum** with 5 tasks — multiple autonomous task switches.
2. **Explosive learning** on Empty-8x8 (new task): 0→0.906 in 27k steps.
3. **No permanent forgetting** — recovered from temporary regression.
4. **Coverage 20× growth** — each DoorKey exploration added new states.
5. **Skill library** stable at 10496/10496 with 344k total skill operations.

See `docs/stage5_report.md` for full analysis.

---

## [v0.4.0-stage4-cloud] — 2026-07-07

### Stage 4 COMPLETED on cloud (AutoDL vGPU-32GB / RTX 4080)

- **Wall time**: 50,939 s (849 min, 14.2 h)
- **Total steps**: 3,000,000
- **Episodes**: 331,211
- **Final mean_return**: **0.914** (V-shaped recovery from 0.600)
- **VRAM**: 2.71 GB, slope 0.0 ✅
- **Skills**: 10496/10496 (full, merge/eviction operational)
- **Skills created**: 317,150 (306,654 merged/evicted = 97% turnover)
- **Speed**: 32 step/s (3.5× slower due to per-episode skill extraction)

### Key validation
1. Skill library filled to capacity and stayed bounded (Axiom 1).
2. Merge (cosine > 0.9) + LRU eviction worked continuously.
3. V-shaped recovery: 0.955 → 0.600 → 0.914 (agent fully recovered).
4. All three tiers saturated: GPU(256) + CPU(2048) + SSD(64×128).

### Bug fixed during this run
- CNNEncoder wrapper changed state_dict keys → Stage 3 weights couldn't load →
  reverted to inline encoder (commit `a67d712`).

See `docs/stage4_report.md` for full analysis.

---

## [v0.3.0-stage3-cloud] — 2026-07-04

### Stage 3 COMPLETED on cloud (AutoDL vGPU-32GB / RTX 4080)

- **Wall time**: 26,536 s (442 min, 7.4 h)
- **Total steps**: 3,000,000
- **Episodes**: 542,262
- **Final mean_return**: **0.950** (stable, no Stage 2 regression)
- **VRAM**: 2.70 GB / 32 GB (8.4%)
- **VRAM slope**: 0.0 GB/h ✅
- **WM loss**: 1.0 (recon=4e-6, kl=1.0 at free_nats floor)
- **Model params**: ~7.4M (HybridActorCritic 7.26M + RSSM 186k)
- **Checkpoints**: 300

### Key validation
1. **RSSM world model** trains stably alongside PPO.
2. **No regression** — mean_ret 0.950 stable throughout (Stage 2 had 0.941→0.896).
3. VRAM unchanged from Stage 2 (RSSM adds only 186k params).
4. WM recon loss ≈ 4e-6 (MiniGrid dynamics are simple).

See `docs/stage3_report.md` for full analysis.

---

## [v0.2.0-stage2-cloud] — 2026-07-03

### Stage 2 COMPLETED on cloud (AutoDL vGPU-32GB / RTX 4080)

- **Wall time**: 25,037 s (417 min, 7.0 h)
- **Total steps**: 3,000,000
- **Episodes**: 272,998
- **Peak mean_return**: **0.941** (at step ~770k)
- **Final mean_return**: **0.896** (regression in latter half — see report)
- **VRAM**: stable 2.71 GB / 32 GB (8.4%)
- **VRAM slope**: 0.002 GB/h ✅
- **Model params**: ~1.5M (HybridActorCritic with TTT-Linear + SWA + FFN × 3)
- **Checkpoints saved**: 300

### Key validation
**TTT-Hybrid architecture works for RL.** This is the first-ever training of
a TTT-Linear + Sliding-Window Attention + FFN backbone in a reinforcement
learning setting. mean_ret peaked at 0.941 (vs Stage 0's 0.951), proving the
architecture is expressive enough for policy learning.

### Issues
- mean_ret regression 0.941 → 0.896 in the latter half (possible causes:
  entropy collapse, LR too high, RND interference — documented in report).
- alarm_fired=True (transient slope spike at one sample; VRAM was stable).

### Bugs fixed during this run
- NaN from batch-as-sequence: HybridActorCritic treated batch dim as temporal
  sequence, causing TTT-Linear W to explode. Fixed: each obs is independent
  seq_len=1 (commit `829cff2`).
- Triton warning spam: get_backend() not cached. Fixed with @lru_cache.

See `docs/stage2_report.md` for full analysis.

---

## [v0.1.0-stage1-cloud] — 2026-07-03

### Stage 1 COMPLETED on cloud (AutoDL vGPU-32GB / RTX 4080 sm_89)

- **Wall time**: 16,160 s (269 min, 4.5 h)
- **Total steps**: 3,000,000
- **Episodes**: 598,995
- **Final mean_return**: **0.955** (stable, inherited from Stage 0)
- **Peak VRAM**: 0.93 GB / 32 GB (2.9%)
- **VRAM slope at end**: **0.0 GB/h** ✅
- **alarm_fired**: **False** ✅ (warmup fix verified — no false positive)
- **Coverage**: 11 / 4096 buckets = 0.27% (MiniGrid-5x5 state space is tiny)
- **Replay final**: hot=4096, warm=32768, cold=8 shards (34496), total=71360/73728
- **Checkpoints saved**: 300
- **Unit tests**: 278 passed / 10 skipped
- **check_bounded**: OK across 37 source files

See `docs/stage1_report.md` for the full run card.

### Key validations
1. **RND curiosity**: stable throughout, no NaN.
2. **BoundedReplayBuffer 3-tier**: hot→warm→cold eviction cycle observed
   (periodic 73216→69632→73216 every ~512 steps). Axioms 1, 2, 3 empirically
   validated.
3. **MemoryWatcher warmup fix**: `alarm_fired=False` across entire 4.5h run.
4. **Cross-stage resume**: Stage 0 → Stage 1 step counter reset correctly.

### Bugs fixed during this run
- Cross-stage resume step counter reset (commit `3f3e5e2`).
- ColdShardTier capacity includes pending buffer (commit `490b34f`).

### Environment
- Ubuntu 22.04, Python 3.12.3, torch 2.5.1+cu124, triton 3.1.0
- Preset: `cloud_5090`
- Resumed from: `ckpt_stage0_003000000.pt`

---

## [Unreleased]

### Fixed — Cross-stage checkpoint resume (critical bug)

- `src/train.py::train`: when resuming from a checkpoint whose `stage` field
  differs from the current run's `stage`, **reset the step counter to 0**
  rather than inherit it. Previously, resuming a Stage 1 run from a Stage 0
  ckpt at step 3_000_000 would immediately exit because
  `state.step (3M) >= total_steps (3M)`.
- Same-stage resume behavior unchanged: continues the step counter (allows
  split-run training).
- `tests/test_resume_cross_stage.py`: 3 regression tests covering same-stage,
  cross-stage, and multi-stage-jump scenarios.

### Test coverage after this batch
- **276 tests passing** (was 273, +3 resume tests), 10 skipped.
- `check_bounded`: OK.

### Fixed — ModelGrowerV2 growth was dormant (real breakthrough path)

- Root cause: the autonomous layer-growth block in `src/train.py` was gated by
  `if model_grower_v2 is not None and coverage is not None`. The
  `phase2_infant_exploration.yaml` config had **no top-level `coverage:`**
  section, so `coverage` was always `None` and growth never ran — the agent
  stayed pinned at the ~101.4 hyperparam ceiling after 9M steps.
- Added `coverage:` block (`num_buckets: 4096`, `log_every_steps: 5000`) to the
  config; lowered `grow_trigger_coverage` 0.3 → 0.15 (raw-obs hash undercounts
  exploration after 9M steps) and spaced `min_steps_between_growths` → 1M.
- Added `ModelGrowerV2.plateau_lp(mean_return)`: returns headroom
  `max(0, 1 - mean_return / running_max)`, ≈0 on a genuine plateau so growth
  fires, >0 while returns still climb. Replaces the old
  `lp = 1.0 - mean_return` (≈-99, over-eager / nonsensical). Call site in
  `src/train.py` now uses it and logs `[growth-debug]` every 50k steps.
- `tests/test_model_growth_v2.py::TestGrowerV2PlateauLP`: 5 new tests plus a
   regression guard that the old formula was over-eager (not blocking).

- **`rmax` now decays (`GrowthConfigV2.rmax_decay = 0.98`)** so the growth
  trigger line (`0.95 × rmax`) can't be pinned forever by a one-off spike.
  Root cause: on checkpoint resume the first `mean_return` is inflated (e.g.
  105.7 → 113), and the raw running max latched it — a 3-layer model plateaued
  at ~101 could never reach `0.95 × 113 ≈ 107.5`, so the next 3→4 growth would
  never fire. `plateau_lp` now uses a decaying running max
  (`rmax = max(mr, rmax × rmax_decay)`) that forgets spikes within a few
  growth-check calls while still tracking genuinely sustained peaks. Added
  `tests/test_model_growth_v2.py::TestGrowerV2PlateauLP::test_spike_is_forgotten_so_growth_can_refire`.

- **Critical fix: ModelGrowerV2 state is now restored on resume** (`src/train.py`
  resume block loads `model_grower_v2_state`, previously only `model` +
  `optimizer` were loaded — the grower state was saved but never read back, per
  the TODO at the old line 1598). Without this, every resume recreated the
  grower as `initial_layers=2` while the model was already 3/4/… layers,
  causing (a) a spurious no-op "2→3" growth that wasted the 1M-step
  `min_steps_between_growths` budget, and (b) a catastrophic latent bug: a
  future resume onto a 4-layer checkpoint with a fresh 2-layer grower would
  call `_create_larger_model(model, 3)` on the 4-layer model and **silently
  DROP the 4th layer**. Added a post-resume layer-count safety sync
  (`grower._current_layers = model.backbone.n_layers`) and a
  `resume_warmup_calls` (default 5) plateau-check warmup so the inflated
  first-step `mean_return` on resume can't immediately force a growth while
  `rmax` decays back to the true plateau. Added
  `tests/test_model_growth_v2.py::TestGrowerV2ResumeLoad`.

- **Non-disruptive (Net2Net-style) layer growth.** Root cause of the first
  autonomous 3→4 growth landing *below* the 3-layer plateau (~95.6 vs ~101):
  `_create_larger_model` copied the old blocks but **randomly initialized the
  new block**, and `_distill` only distilled on **random-noise images** with
  **all** student params trainable — so the new block learned a *useful-but-
  different* transform from scratch on noise, permanently erasing ~5 points of
  the copied policy. Fix: `grow()` now accepts `distill_inputs` (real
  observations sampled from the replay buffer in `src/train.py`) and `_distill`
  **freezes every parameter except the freshly-added block**
  (`student.backbone.blocks[new_block_idx]`), training only that block to
  reproduce the teacher's outputs (KL + MSE on logits/values). The new block
  thus learns the **identity map on the agent's real data**, so the grown model
  is *equal* to the teacher at the growth step (no drop) and RL can then exploit
  the extra capacity. Verified offline: logit/value drift after growth ≈ 0.3%
  (was a full policy reset). Config gains `distill_steps: 400`, `distill_lr:
  1e-3`, `distill_batch: 1024` under `model_growth:`. Added
   `tests/test_model_growth_v2.py::TestGrowerV2ObsShapeCarryover::
   test_grow_is_non_disruptive_with_real_data` (drift < 10%).

### Verified — First live non-disruptive 3→4 growth (2026-07-17, remote)

- Triggered at **step=12,000,832** (`[growth] grown to 4 layers
  (step=12000832) (non-disruptive, real-data distill)`).
- Matches the design cooldown: `train.py` enforces
  `_last_growth_step = max(loaded, resumed_step=11.0M)` on resume, so the
  3→4 growth fires exactly at the 1M-step cooldown boundary (12.0M), not a
  spurious no-op on resume.
- **No drop at the growth step**: 4-layer mean_return started at ≈100.8
  (= the 3-layer plateau), then climbed past it to **≈101.3 by step
  12.04M** — the earlier catastrophic ~70 collapse is gone.
- **Value head self-healed**: the pre-growth jitter (`v` spiking to 0.8–1.2,
  `cf` up to 0.26) returned to a healthy **0.16–0.5** zone after growth.
- Checkpoint landed: `backup/growth_20260717_191118_ckpt_stage2_012020288.pt`
  (4 layers, step 12.02M). Note: the 3 earlier `growth_*` backups
  (11:55 / 12:12 / 12:28) are monitor growth-detection noise
  (no `[growth]` line), not real model surgery.

### Ops — Remote training backup to system disk

- `monitor.sh` / `watch_backup.sh` now also mirror the train logs +
  latest checkpoint to the **system disk** (`/root/karbon/...` alongside the
  existing `/root/autodl-tmp/karbon/backup/`), so a loss of the
  ephemeral `/root/autodl-tmp` volume can't take the run with it.

### Fixed — 4→5 growth was silently dormant (rmax-collapse bug, 2026-07-18)

- Root cause: `ModelGrowerV2.plateau_lp()` was called **every training step**
  from the `src/train.py` growth-check block, with `rmax` decaying
  `max(mr, rmax × rmax_decay)` **per call** (`rmax_decay=0.98`). At ~1k calls
  per second `rmax` collapsed to the instantaneous `mean_return` within ~35
  steps, so `lp = 1 - mr/rmax` sat at ≈0 **permanently** — yet growth still
  never fired because the post-resume `_warmup_remaining` + the
  `state.step % 50000 == 0` gating of the LP refresh meant the signal was
  effectively stale/never refreshed past the unlock point. Net effect: the
  4-layer agent plateaued at ~101–102 and 4→5 never triggered (observed
  step 14.02M→14.47M, `mean_ret` slowly drifting down to 101.4 with no growth).
- Fix (two-part, BOUNDS-OK):
  1. `plateau_lp` now uses a **fixed-capacity windowed max**
     (`deque(maxlen=rmax_window=40)`) instead of per-call exponential decay,
     so `rmax` tracks the genuine recent peak without being collapsed by the
     per-step call cadence. `rmax_window` added to `GrowthConfigV2` and
     persisted in `state_dict`/`load_state_dict`.
  2. `src/train.py` refreshes the LP signal **once per `growth_check_every`
     (50k) steps** using integer-division buckets (`state.step // N`), and the
     `[growth-debug]` log uses the same bucket approach — both now fire
     regardless of step-cadence alignment (steps advance by 512, so the old
     `state.step % 50000 == 0` condition almost never matched and left growth
     unobservable).
- Verified: after the fix the `[growth-debug]` line emits every 50k-step
  bucket (first at step=14,060,608, `lp=0.0000, layers=4`), and 4→5 is now
  expected to fire once `state.step` passes the 14.40M unlock with `lp≈0` and
  `cov≥0.3`. Relaunched from `ckpt_stage2_014060096.pt` (14.06M) with the
  corrected code; the growth-check block is again active.

---

## [Unreleased]

### Added — Stages 3 / 4 / 5 / 6 wiring (full pipeline integrated end-to-end)

**Stage 3 · World Model**
- `configs/stage3_world_model.yaml`: WM sub-block (z_dim, h_dim, embed, hidden,
  max_rollout_steps, kl_free_nats, lr, update_every_steps).
- `src/train.py`: builds `RSSM` when `world_model` config present; trains on
  transitions sampled from `BoundedReplayBuffer`; per-cycle recon/KL losses
  logged; WM state added to checkpoints.
- `tests/test_stage3_wm.py` (8 tests).

**Stage 4 · Skill Library**
- `configs/stage4_skills.yaml`: skills sub-block (LoRA rank, 3-tier capacities,
  merge threshold, score weights).
- `src/train.py`: builds `BoundedSkillLibrary`; registered in HealthChecker;
  count logged per step summary; skills state added to checkpoints.
- `tests/test_stage4_skills.py` (8 tests).

**Stage 5 · Auto Curriculum**
- `configs/stage5_curriculum.yaml`: curriculum sub-block with 5 declared
  MiniGrid tasks (Empty-5x5..8x8, DoorKey-5x5/6x6).
- `src/train.py`: builds `AutoCurriculum`; loads tasks from config; every
  `report_every_steps` reports (1 - mean_return) as LP error; every
  `switch_every_steps` re-samples an active task and rebuilds the env.
- `tests/test_stage5_curriculum.py` (9 tests).

**Stage 6 · Continual (Online EWC + Generative Replay + Sleep)**
- `configs/stage6_consolidation.yaml`: continual sub-block covering EWC
  (lambda, gamma, anchor mode, consolidate every-steps), Generative Replay VAE
  (latent_dim, hidden, lr, kl_weight, update-every, rehearsal batch,
  inject-every), and sleep periods (replay_trim / skills_merge / ttt_distill).
- `src/train.py`:
  - Builds `OnlineEWC`, `GenerativeReplayVAE`, `SleepConsolidationLoop`.
  - Adds EWC penalty to PPO loss once `ewc.has_consolidated()`.
  - Trains VAE from replay every N env-steps.
  - Sleep loop registers `replay_trim / skills_merge / ttt_distill /
    ewc_consolidate` callbacks. Sleep ticks in the training loop.
  - EWC / GR / sleep state added to checkpoints.
- `src/utils/config_schema.py`: `TopLevelSchema` now accepts `world_model`,
  `skills`, `curriculum`, `continual` optional sub-blocks (permissive).
- `tests/test_stage6_continual.py` (9 tests).

**Trainer defaults**
- `src/train.py::_DEFAULT_STAGE_CONFIGS` now maps stages 0–6 to their
  respective yaml files.

### Test coverage
- **273 tests passing** (was 239, +34 Stage 3–6), 10 skipped, 0 failing.
- `check_bounded`: OK across 37 source files.

### Full stack summary (Stage 6 active)
When `--stage 6 --preset cloud_5090`, the trainer builds:
- ActorCritic backed by HybridBackbone (TTT-Linear + SWA + FFN).
- RND intrinsic reward → augments extrinsic reward.
- BoundedReplayBuffer (3-tier GPU/CPU/SSD) fed every step.
- BoundedCoverage tracking state-visitation entropy.
- RSSM world model trained on replay.
- BoundedSkillLibrary registered in HealthChecker.
- AutoCurriculum switching envs by learning progress.
- OnlineEWC penalizing weight drift after each consolidation.
- Generative Replay VAE learning obs distribution.
- SleepConsolidationLoop firing periodic offline maintenance.

All components enforce their bounded-design axioms; each is state-serializable
and included in the checkpoint envelope.

---

## [Unreleased]

### Added — Stage 2 wiring (Hybrid backbone in the training loop)

- `configs/stage2_hybrid.yaml` — Stage 2 config: enables `use_hybrid_backbone`,
  keeps all Stage 1 blocks (RND / Replay / Coverage).
- `src/train.py`:
  - New `HybridActorCritic` class: CNN encoder → HybridBackbone
    (TTT-Linear + SWA + FFN) → policy/value heads. Treats the rollout batch as
    a length-B sequence so causal SWA + TTT-Linear see cross-step context.
  - `train()` now picks between vanilla `ActorCritic` and `HybridActorCritic`
    based on `config.model.use_hybrid_backbone`.
  - `d_model` auto-snapped up to a multiple of `n_heads` and to even (for PE).
- `src/utils/config_schema.py`: `ModelSchema` extended with 7 Hybrid knobs
  (`use_hybrid_backbone`, `hybrid_n_layers`, `hybrid_n_heads`, ...); validates
  hyperparameter ranges when hybrid is on.
- `src/train.py::main`: `--stage 2` auto-loads `stage2_hybrid.yaml`.
- `tests/test_stage2_hybrid.py`: 11 tests covering config load/validate,
  Hybrid output shape/grad flow/determinism, d_model snapping, param-count
  sanity, and shape parity with the baseline `ActorCritic`.

### Test coverage
- **239 tests passing** (was 228, +11 Stage 2), 10 skipped, 0 failing.
- `check_bounded`: OK across 37 source files.

### How to run Stage 2 on cloud
```bash
cd ~/karbon
git pull origin main
LATEST=$(ls -t /root/autodl-tmp/karbon/ckpts/ckpt_stage*_*.pt | head -1)
tmux new -d -s stage2 "source .venv/bin/activate && bash scripts/cloud/run_stage.sh 2 cloud_5090 --resume $LATEST"
tmux attach -t stage2
```

Note: resuming Stage 0 ckpts into Stage 2 will warn "Model state mismatch" and
start the Hybrid model fresh (expected — different architecture).

---

## [Unreleased]

### Added — Stage 1 wiring (RND + Bounded Replay + Coverage)

- `configs/stage1_curiosity.yaml` — Stage 1 config: intrinsic (RND), replay
  (3-tier bounded), coverage (fixed-bucket state-visitation).
- `src/train.py` — trainer now supports Stage 1:
  - Reads optional `intrinsic` / `replay` / `coverage` config blocks.
  - Adds RND intrinsic reward to environment reward (with `reward_coef`).
  - Pushes every transition to `BoundedReplayBuffer` (with PER priorities).
  - Every N env steps runs an off-policy TD update from replay
    (`--stage 1` triggers this path automatically).
  - Adds `BoundedCoverage` class (fixed hash-bucket state-visitation counter).
  - Stage 0 codepath fully unchanged (backward-compatible).
- `src/utils/config_schema.py` — top-level schema now accepts optional
  `intrinsic` / `replay` / `coverage` sub-blocks (permissive validation).
- `src/train.py::main` — auto-selects `stage{N}_baseline.yaml` or
  `stage{N}_curiosity.yaml` based on `--stage`.
- `src/monitoring/memory_watcher.py` — added `warmup_seconds` (default 300 s)
  to suppress startup slope alarms. Trainer passes this through via
  `monitor.warmup_seconds` config key.
- `tests/test_stage1_config.py` — 11 new tests covering:
  - Stage 1 config exists / loads / validates.
  - Intrinsic / replay / coverage hyperparameter sanity.
  - `BoundedCoverage` capacity enforcement, dedup on repeated states,
    state-dict roundtrip, `BoundedComponent` protocol conformance.

### Test coverage after this batch
- **228 tests passed** (was 217), 10 skipped, 0 failing.
- `check_bounded`: OK across 37 source files.
- Stage 0 backward-compat: verified via `test_config_presets` still green.

---

## [v0.0.0-stage0-cloud] — 2026-07-02

### Stage 0 COMPLETED on cloud (AutoDL vGPU-32GB / L40 sm_89)

- **Wall time**: 3906.8 s (65 min)
- **Total steps**: 3,000,000 (full run, not smoke)
- **Episodes**: 555,169
- **Final mean_return**: **0.951** (near-optimal for MiniGrid-Empty-5x5-v0)
- **Peak VRAM**: 0.82 GB / 32 GB (2.5%)
- **VRAM slope at end**: **0.0 GB/h** ✅ (Axiom 5 clean)
- **Checkpoints saved**: 300
- **Unit tests**: 217 passed / 10 skipped
- **check_bounded**: OK across 37 source files

See `docs/stage0_report.md` for the full run card.

### Fixed
- `MemoryWatcher`: added `warmup_seconds` (default 300 s) to suppress startup
  slope alarms. During the Stage-0 cloud run the alarm fired once during
  CUDA-context / Adam-state initialization (which looks like a fast slope in
  the 5-minute rolling window). Long-term slope was 0.0 GB/h. The alarm was
  a false positive; the warmup gate prevents it in future runs.
- Added `test_memory_watcher_warmup_suppresses_alarm` and
  `test_memory_watcher_alarm_fires_after_warmup` to guard the fix.

### Environment (locked for reproducibility)
- Ubuntu 22.04
- Python 3.12.3
- torch 2.5.1+cu124 (from AutoDL preset image)
- triton 3.1.0
- Preset: `cloud_5090` (32 GB budget matches vGPU-32GB exactly)
- Env vars: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` +
  `DEVAGI_{CKPT,DATA,LOGS}_DIR=/root/autodl-tmp/karbon/*`

---

## [Unreleased]

### Added — Nightly / interruptible training

- `scripts/cloud/nightly_run.sh` — one-command starter for off-peak-only
  cloud training. Auto-resumes from newest ckpt, launches trainer +
  autosync in parallel tmux sessions, optionally auto-stops after
  `--duration <sec>` or `--until <time>` and does a final Git sync.
- `configs/_presets/cloud_5090.yaml`: `ckpt_every_steps` reduced from
  25_000 → 10_000 so nightly interrupts lose at most ~15–30 min of work.
- `MIGRATION.md` §10: nightly rhythm, cost tradeoff, do/don't list.

### Interrupt tolerance summary
- Stage 0–5: FULL support (resume from any ckpt).
- Stage 0 24h longevity: NOT interruptible (hard exit criterion).
- Stage 6 30-day perpetual: NOT interruptible (final milestone).

### Cost scenarios (vGPU-32GB @ ¥1.58/h)
- 24/7:        ~30 days, ¥1140
- Night-only:  ~60 days, ¥570 (12h/day)
- Weekends:    ~90 days, ¥456 (~48h/wk)

---

## [Unreleased]

### Added — Autosync daemons (periodic GitHub / TOS / rsync push)

- `scripts/cloud/autosync_daemon.sh` — Linux daemon that every N seconds:
  1. Commits & pushes small text artefacts (docs/, configs/, CHANGELOG.md) to GitHub.
  2. `rsync`s checkpoints/figures to `${DEVAGI_REMOTE_TARGET}` if set.
  3. Optional `--export`: re-exports the newest ckpt into `exports/latest/` HF layout.
  Best-effort semantics — never crashes; trap SIGTERM/SIGINT for graceful shutdown.
- `scripts/local/autosync_daemon.ps1` — PowerShell equivalent for the Windows laptop.
- `MIGRATION.md` §9: bilingual guide on launching, stopping, and tuning the daemon.
- `tests/test_autosync_daemon.py`: 4 structural tests (LF endings, shebang, trap
  handler, no bare `set -e`, no leaked PATs).

### How to use during Stage 0 training on the cloud
```bash
# In one tmux session:
bash scripts/cloud/run_stage.sh 0 cloud_5090

# In another tmux session (parallel):
tmux new -d -s devagi_autosync \
    "bash scripts/cloud/autosync_daemon.sh --stage 0 --interval 3600"
```

The daemon will push whatever reports / config snapshots you drop into
`docs/` and `configs/` throughout the run, so the GitHub repo stays fresh
without manual intervention.

### Test coverage after this batch
- **215 tests passed** (was 211), 10 skipped, 0 failing.
- `check_bounded`: OK.

---

## [Unreleased]

### Added — Full-journey planning doc

- `FULL_JOURNEY.md` — Bilingual end-to-end Stage 0–6 timeline, cost matrix,
  hardware purchase guide, risk register, and copy-paste kickoff commands.
  Recommends Route C (hybrid: cloud vGPU/5090 for Stage 0–4, home 64G rig
  for Stage 5–6 perpetual).
- README: link to FULL_JOURNEY.md.

### Route summary (from FULL_JOURNEY.md)
- Route A · full cloud (vGPU):    ~85 days, ¥2900–3500
- Route B · full cloud (5090):    ~60 days, ¥3900–4550
- **Route C · hybrid (recommended): ~90 days, ¥1200–1600 + hardware (¥16k–60k)**

---

## [Unreleased]

### Added — Cloud-training operational scripts

- `scripts/preflight.py` — 10-step pre-training checklist (Python version, torch/CUDA, GPU inventory, Triton, disk, env vars, project imports, bounded check, preset load, 20-step smoke). Exits nonzero on any critical failure.
- `scripts/cloud/sync_to_git.sh` — pushes small stage artefacts (reports, figures, config snapshots, CHANGELOG) + optional git tag to remote. Explicitly documents the split: text → Git, binaries (`*.pt`, replay data) → TOS/rsync mirror.

### Guidance
- Reports (`docs/stage*_report.md`) and preset snapshots ARE synced to Git.
- Checkpoints, replay cold data, exports/ are NOT — they go to TOS or an rsync mirror.

---

## [Unreleased]

### Added — RTX 5090 / Blackwell support + platform-image (PyTorch 2.8 / CUDA 12.8) support

Target scenario: cloud VMs with pre-installed **PyTorch 2.8.0 / Python 3.12 /
Ubuntu 22.04 / CUDA 12.8** running on RTX 5090 (Blackwell / sm_120).

- `requirements/cuda128.txt` — new; `torch>=2.8,<2.9` + `triton>=3` on cu128 wheels.
  Ships sm_120 kernels required by RTX 5090.
- `configs/_presets/cloud_5090.yaml` — new preset:
  22 GB VRAM budget, batch=16, seq=96, 16 parallel envs, 6-layer × 384-dim model.
  Sits between `cloud_24g` and `home_64g`.
- `scripts/cloud/setup_env.sh`:
  * Auto-detects Python 3.10/3.11/3.12 (was 3.10-only).
  * Detects RTX 50-series → auto-uses `cuda128.txt`.
  * New `--skip-torch` flag to reuse pre-installed torch on platform images.
- `scripts/home/setup_env.sh`: same auto-detect + `--skip-torch` support.
- `pyproject.toml`: `requires-python = ">=3.10,<3.13"`; black targets py310/11/12.
- `HARDWARE_TOPOLOGY.md`: Phase-2 section now documents 5090 wheel selection.
- Preset test coverage extended to `cloud_5090` (211 tests total).

### Test coverage after this batch
- **211 passed** (was 208), 10 skipped, 0 failing.
- `check_bounded`: OK across 37 source files.

---

## [Unreleased]

### Added — HuggingFace-format export path (this session)

**Export tooling for TOS / HuggingFace Hub / ARK custom-model upload:**
- `scripts/export_hf.py` — converts `src.utils.ckpt.save_ckpt` payloads to HF
  layout: `config.json` + sharded `model.safetensors` (+ `model.safetensors.index.json`
  when >5 GB) + bilingual `README.md`.
- Supports architectures: `hybrid_backbone`, `rssm`, `rnd`, `ttt_linear`.
- Supports dtype cast: `float32` / `float16` / `bfloat16`.
- `scripts/build_demo_export.py` — one-shot generator for demo `exports/demo-hybrid-{fp32,fp16}/`
  directly uploadable to TOS.
- `requirements/base.txt`: adds `safetensors>=0.4`.
- `.gitignore`: ignores `exports/` (upload artefacts, not for Git).
- `.gitattributes`: marks `*.safetensors` as binary.
- `MIGRATION.md` §8: bilingual export + TOS upload guide.
- `tests/test_export_hf.py`: 8 unit tests covering flatten / shard / dtype cast
  / roundtrip / architecture whitelist.

### Test coverage after this batch
- **208 tests passing**, 10 skipped, 0 failing.
- `check_bounded`: OK across 37 source files.

---

## [Unreleased]

### Added — Full local pre-work batch (A–N)

**Models:**
- `src/models/ttt_mlp.py` — TTT-MLP with 2-layer inner MLP, analytic GELU derivative, mini-batch dual form.
- `src/models/world_model.py` — Dreamer-style RSSM: encoder / decoder / GRU / prior / posterior heads, bounded rollouts.

**Memory:**
- `src/memory/skill_library.py` — Bounded 3-tier LoRA-based skill library with LRU × usefulness × reward eviction and cosine-similarity merging.
- `src/memory/generative_replay.py` — Small MLP VAE for anti-forgetting rehearsal.

**Intrinsic / Curriculum / Continual:**
- `src/intrinsic/learning_progress.py` — Per-task ring buffer + LP metric, smoothing, priority normalization.
- `src/curriculum/auto_curriculum.py` — LP-driven task sampling with FIFO eviction and ε-exploration.
- `src/continual/online_ewc.py` — Single-Fisher exponentially-decayed EWC with penalty and gradient integration.
- `src/continual/consolidation.py` — Periodic sleep-consolidation loop with warmup gate and disabled-task support.

**Envs:**
- `src/envs/crafter_wrapper.py` — Stage-3 Crafter wrapper with lazy import, auto-reset, bounded episode-return history.

**Utils:**
- `src/utils/config_schema.py` — Dataclass-based config validation catching typos, wrong types, out-of-range values.

**Scripts:**
- `scripts/home/setup_env.sh` — Phase-3 home 64G rig setup with VRAM ≥40 GB sanity check.
- `scripts/home/run_perpetual.sh` — tmux-wrapped perpetual training launcher.
- `scripts/home/health_daemon.sh` — External CSV-logging health monitor with VRAM slope alarm.

**Docs & governance:**
- `AGENTS.md` — Operating protocol for automated coding assistants.
- `CONTRIBUTING.md` — Human contributor guide.
- `notebooks/memory_profiling.ipynb` — MemoryWatcher CSV visualization.
- `notebooks/skill_visualization.ipynb` — Skill library usage / weight-heatmap / similarity analysis.
- `notebooks/ttt_state_inspection.ipynb` — TTT-Linear inner-state per-segment norm plot.

**Tests:**
- `tests/test_ttt_mlp.py` (7)
- `tests/test_skill_library.py` (13)
- `tests/test_world_model.py` (10)
- `tests/test_learning_progress.py` (13)
- `tests/test_auto_curriculum.py` (10)
- `tests/test_online_ewc.py` (11)
- `tests/test_consolidation.py` (9)
- `tests/test_generative_replay.py` (9)
- `tests/test_config_schema.py` (16)
- `tests/test_integration_stage0.py` (2) — End-to-end wire-up test using a DummyEnv.
- `tests/test_crafter_wrapper.py` (7 + 1 skipped) — Fake-crafter-based mechanics tests.

### Test coverage after this batch
- **200 tests passing**, 10 skipped (Triton parity + Crafter install), 0 failing.
- `check_bounded`: OK across 37 source files.

### Stage readiness after this batch
- Stage 1 ready: `RND` + `BoundedReplayBuffer` complete.
- Stage 2 ready: `TTT-Linear` + `TTT-MLP` + `SlidingWindowAttention` + `HybridBackbone` complete.
- Stage 3 ready: `RSSM` world model + `CrafterWrapper` complete.
- Stage 4 ready: `BoundedSkillLibrary` complete.
- Stage 5 ready: `LearningProgressTracker` + `AutoCurriculum` complete.
- Stage 6 ready: `OnlineEWC` + `SleepConsolidationLoop` + `GenerativeReplayVAE` complete.
- Static enforcement of six axioms operational; `make check-bounds` integrated.

---



## [v0.0.0-stage0-local] — planned

### Added
- Full project skeleton at `D:\karbon\`.
- Documents: `PLAN.md`, `README.md`, `HARDWARE_TOPOLOGY.md`, `MIGRATION.md`, `DESIGN_PRINCIPLES.md`, `GLOSSARY.md`, `ROADMAP.md`.
- Requirements split: `base.txt`, `cpu.txt`, `cuda121.txt`, `dev.txt`.
- Platform abstraction: `src/platform/device.py`, `paths.py`, `memory_probe.py`.
- Monitoring: `src/monitoring/memory_watcher.py`, `longevity_test.py`, `health_check.py`.
- MiniGrid wrapper: `src/envs/minigrid_wrapper.py`.
- Minimal PPO baseline: `src/train.py`.
- Three-tier preset system: `configs/_presets/{local_smoke,cloud_24g,home_64g}.yaml`.
- Stage 0 config: `configs/stage0_baseline.yaml`.
- Unit tests for platform/presets/memory.
- Scripts: `scripts/local/setup_env.ps1`, `smoke_test.ps1`.

### Bootstrap decisions
- Project root: `D:\karbon\` (no sub-directory).
- Python: 3.10 + project-local venv.
- PyTorch: 2.5.1 (+cpu locally, +cu121 on cloud/home).
- Docs: bilingual, Chinese-friendly.
- Cloud platform: user-choice, platform-agnostic project code.
- Persistence: rsync after each training run.
- Longevity: perpetual target (Stage 6: 30 days uninterrupted).

---

## Template for future stages / 后续 Stage 模板

```
## [vX.Y.0-stageN] — YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...

### Deprecated / Removed
- ...

### Bounded-axiom review
- ...

### Longevity result
- 24h VRAM slope: ... GB/day
- ...
```
