# Bug Ledger / 缺陷台账

> 2026-10-09 建档。所有已发现的缺陷在此登记（代码/设计/运维三类）。
> 原则：**每个 bug 必须溯源到"哪一行违反了契约"**，而非补偿性 hack
> （AGENTS §13 标本兼治）。发现新同类问题时追加，并做同类扫描。

---

## A. 静默断链类（Silent broken wires — "声明完成但从未运行"）

| ID | 日期 | 症状 | 根因 | 修复 |
|---|---|---|---|---|
| A1 | 2026-09-17 | ReflectionLoop 830 次/段失败, 0 教训 | record_step 存 CPU 张量, SelfModel 在 CUDA; 失败跨 episode 泄漏 | 设备/类型对齐 + finally 清理 |
| A2 | 2026-09-17 | eval sym 指标恒 0 | 读错 key (total_success/total_usage) + 3D 探针确定性回放 | key 映射修正 + 每 eval 独立 seed 布局 |
| A3 | 2026-09-17 | LogicEngine 从未触发 | reason() 需 rule_projection 投影空间错配 + `Quantifier.ALWAYS` 不存在被裸 except 吞 | 投影修正 + 枚举修正 |
| A4 | 2026-09-17 | kanren 假指标 (自查询恒真 + 硬币 acc 0.499) | feedback 零消费; 自查询退化为同义反复 | 假指标退役 |
| A5 | 2026-09-17 | 叙事偏好恒 0.25 (无区分) | 读 `spec["difficulty"]`, TaskTemplate 排除该字段 | 改读 `t.difficulty` |
| A6 | 2026-09-17 | openness 事件永不触发 | 相对回报口径在稀疏奖励下不可达 | 行为探针 (≥0.06) 优先口径 + 单一写入者 |
| A7 | 2026-09-22 | probe_net 从未训练 | 更新在 rollout 的 `no_grad()` 下执行 | `torch.enable_grad()` 包裹 |
| A8 | **2026-09-25** | ckpt `facts_total=0` (全项目历史) | `train.py` 调 `causal_disc.get_edges()` — **该方法从不存在**, `hasattr` 恒 False → `add_causal_edges` 从未执行; 裸 except 掩盖 | 改读 `_graph.edges` + 带堆栈告警 + `[symbol] facts ingested` 变更日志 |
| A9 | **2026-09-25** | 规则 success_count 恒 0 | `RuleMemory.update()` **全仓库无调用者** | `_apply_rule_outcome_feedback` 接线 (usage 增量回传) |
| A10 | **2026-09-28** | aux 巩固 122/122 次全部失败 | 存 obs 带 batch 维 → 5 维张量 → 编码器 permute 崩; `torch.stack` 再叠一维 | 存储时 squeeze + `torch.cat` 重建批 |
| A11 | **2026-10-10** | soak3 整段 13046 proposals / **0 probes**; [meta] n=0 | M4 部署**只传了 train.py + deliberation.py, 漏传 meta_monitor.py** (缺新属性 `recent_pressure`) → 探针块 modulate 行每步 AttributeError; 外层 handler 吞掉 (全场仅 3 条 WARNING) → 探针路径整段死亡 | 补全部署 + modulate 防御性 fallback (基础 ε) + 限频响亮日志; **部署清单: 每个模块逐一验 marker** |
| A12 | **2026-10-11** | MetaMonitor / DeliberationGate **每次重启即清零** (skill 窗 2000 步重学); 同类扫描另发现 `hypothesis_tester_state` (probe_net 权重) 同样无存档侧 | `_restore_map` 声明了 `meta_monitor_state` 契约, 但 **save 侧从未写入该键** (`if key in _extra` 守卫使恢复端"看起来一切正常") — 声明-实现单向缺失 (3 键同病: meta_monitor / hypothesis_tester / deliberation) | save 侧补写 `meta_monitor_state` / `hypothesis_tester_state` / `deliberation_state` / `concept_bridge_state`; DeliberationGate 增 state_dict/load_state_dict; **新规则: 恢复映射与保存端双向核对** (脚本差分: restore keys 37, save keys 49, 交集核验 — `C:\...\Temp\opencode\crosscheck_restore.py` 可复用) |
| A13 | **2026-10-17** | teacher 恒以 ~5% 覆盖训练动作 (teach loss 2.16 常数贯穿整个物理时代); 学习动力学审计中发现 | **退火泄漏**: tgate 退火到 0 后外层 `if _tgate["teacher"] > 0.0` 整块停止运行, `env.occluder_teacher_force` 停留在最后一次赋的值 (0.05) → 教师永不真正关闭 (探针窗反正确保 0, 非探针窗残留 0.05) | 补 else 分支显式置 0.0 (train.py ~L2736); 部署 seg6 生效; 台账规则: **"开关"变量的关闭必须是显式赋值, 不能依赖门控块停止运行** |
| A14 | **2026-10-22** | ExtendedThreeDWorld **从未可运行**: 构造即 AttributeError (`self._scene_builder` 不存在) — 自父类 refactor 后 死代码; M2-p2 冒烟首次实例化时捕获 | 父类 `_build_scene` 的 builder 是局部变量, 子类覆写引用不存在的属性; 且子类先于父类构建就调用 builder (顺序也错) | 父类加 `_extend_scene(builder)` 钩子 (模型编译前); 子类改为钩子注入 siblings + 季节簿记; 冒烟 + 契约测试 1/1 |
| A15 | **2026-10-23** | B10 的 `occluder_reach_hold_max_steps` 自 2026-10-15 起**在训练中从未生效** (启动分支与 curriculum 重建分支都未传递该 kwarg; 仅测试/冒烟 env 显式传了) | 接线遗漏: 配置键 + env 参数 + 测试全齐, 但 train.py 的两处 env 构建没加这一行 (静默: env 用 ctor 默认 0=无界)。连带修正叙事: 露营冻结的解除实际来自 LR 脉冲/双稳态, 非 B10 窗口 (窗口今天才第一次真实生效) | 共享 helper `_threed_env_kwargs` (curriculum) + 启动 ThreeDWorld 分支补线; 双世界构建验证 hold_cap=15 ✓; 规则: **每个 config→env 参数必须有构建路径断言** (契约测试锁 27 维/12 动作/hold_cap) |
| A16 | **2026-10-25** | motor_babble 的衰减节律每段重置 (coef 每段回 0.3; 计数表重建) | 新模块加进训练回路但**未接存档/恢复** (与 A12 同类: 恢复映射与保存侧单边) | 补 `motor_babble_state` save/restore (train.py; 下段生效); 规则重申: 每个新模块必须同时出现在三处 — 构造/遥测/(save+restore) |
| A17 | **2026-10-27** | resume 后每 episode `[curriculum] ledger record failed: 'ema_ret_fast'` (ledger 静默停记) | LP 升级新增的 fast/slow EMA 在旧 ckpt 的 ledger state 中不存在; record 路径用 `st[...]` 直取 → KeyError, 外层 handler 吞掉 (静默: preference 用陈旧回报) | `.get(key, default)` 回退 + `test_lp_resume_from_legacy_state`; 规则: **新字段必须容忍旧存档** (resume 兼容测试纳入新特性清单) |

## B. 设计缺陷类（Design flaws — 逻辑本身不健全）

| ID | 日期 | 症状 | 根因 | 修复 |
|---|---|---|---|---|
| B1 | 2026-09-25 | reach=100 被截断为 10/次; 到达:滞留激励比从 100:4 塌到 10:4 | Stage-20w 修了 inner clamp, 外层 `min(10.0, reward)` 覆盖 (半修复) | clamp 端到端 (可配置 100) |
| B2 | 2026-09-25 | 可动物体后刷分: 推物→自诱导事件→到达奖励 | 近起点事件完整支付 (起点<1.2m 也付); op 0.75→0.12 | `occluder_search_min_start` 门控 (仅远起点支付) |
| B3 | 2026-09-26 | 规则置信度 runaway → 单规则捕获 → op 0.78→0.30 | `ep_ret>0` 在高回报制度下"全成功" | 相对均值判别 (高于平均才算成功) |
| B4 | 2026-09-27 | 谷期 value loss 爆至 1.3e3 → 评论家崩 → op 塌 | `ReturnNormalizer(α=0.01)` 滞后 ~4.5h 追不上 3× 奖励摆动; MSE 对越界二次放大 | α=0.1 + `smooth_l1(β=5)` |
| B5 | 2026-09-29 | 单规则捕获复发 (#336805/#332466, 72/80 匹配) | 两条转向通道未断: logit 偏置 0.5 + env 逻辑奖励 0.3 | 双通道冻结 (规则仅记录不转向) |
| B6 | **2026-10-09** | `[contradiction]` 40 次 cycle failed (M3 首测) | 规则嵌入混合设备 (cpu/cuda) → `torch.stack` 崩 | 归一化时统一移 CPU |
| B7 | **2026-10-09** | 125 对矛盾每周期重复降权 (0.7^k 衰减) | 解决动作无幂等性 (无已解决记忆) | `resolved_memory` (有界 256, 每对只解决一次) |
| B8 | **2026-10-09** | 事件日志文件跨重启无界增长 | 行数上限是进程内的, 不约束磁盘文件 | >32MB 时轮转 (打开时截断) |
| B9 | **2026-10-12** | chain11 首段 7 秒即退 ("no progress, stop"), 续跑链当场死亡 | **边界自退出**: 续跑脚本只在段末 +250k 扩 target, 而首段 resume 档步数 == total_steps (15,350,000) → 训练循环见 `step >= total_steps` 立即退出 (rc=0, "Training finished ... elapsed=0.0s") | v2: 每段**启动前**先扩 target (step+250k) — 覆盖"恰在边界续跑"场景; 段末扩 target 逻辑删除 (避免双重扩) |
| B10 | **2026-10-15** | **camp-freeze 退化**: 16.85M→17.35M 抓握率 10.8%→100%, op 0.47→0.28, 平台卡 ~0.30 逾 1.2M 步 | **衰减奖金无界**: `occluder_reach_hold: 5.0` 圈内每步支付 (最高 ~4/步, 无时长上限) + 抓握动作语义=原地冻结 (actions≥8 清零 dx,dy) → "到达+永久驻留"成为最优策略; 实测静止步均收 0.1417 > 移动 0.1308 (结构性实锤)。附带发现及早退守卫只查 target/shaping/orient, 漏掉 hold/reach/reveal | `occluder_reach_hold_max_steps: 15` 有界窗口 (到达一次性 100 + 揭示奖金 2.0 保留, 金矿砍除); 守卫补全七项; 测试 `test_occluder_hold_cap.py` 3/3 (含无界回归相); 验证: 下一训练段解剖抓握率回落 + op 趋势 |
| B11 | **2026-10-19** | Stage 23 M1 切换 (focus_op_only off) 后 op 0.80→0.20 (1.5 段), mean_ret 1400→58 | **奖励双路径分离**: 非 focus 路径只付 target(0.3)/shaping(0.3); 完整 op 阶梯 (reach 100/hold/orient/reveal) 只在 `_occluder_only_reward` (focus 路径调用); 且该路径硬截断 `min(5.0)` + 最终 `min(10.0)` 会二次吞掉 reach | 非 focus 路径改为 `reward += _occluder_only_reward()`, 两端截断改用 `_op_reward_clamp` (部署 2026-10-19, rich-path 实测 103.05 ✓); 规则: **双分支奖励必须同一阶梯, 截断必须同源可配** |
| B12 | **2026-10-22** | 若以 ExtendedThreeDWorld 训练: proprio 18 vs 模型 27 → resume 的 strict=False 会静默随机化 proprio 支路 | train.py 的 Extended 分支只传 5 个参数 (occluder/奖励/槽/教师全丢) | 分支改为传与 ThreeDWorld 相同的完整 kw 集 (+num_siblings/season_cycle_steps); 契约测试锁定 proprio=27/action=12 |
| B13 | **2026-10-23** | KnowledgeLedger 的 ZPD 选择信号在丰富奖励时代死掉 (preference 恒均匀) | mastery 用**绝对尺度** (ema_ret/100): 回报 1200-2400 时所有任务都算"已掌握" → 权重只剩地板 | mastery 改**相对尺度** (任务间平均回报为参照; 单任务回退绝对) — ZPD 恢复 (实测: 被练任务回报 +22.5%/15min, 权重回落) |
| B14 | **2026-10-25** | 第三例常数动作冻结: release 61.5% + rotate 24.4%, 移动仅 7.9%, op 0.37-0.44 | **机制族**: actions≥8 语义=原地 (清零 dx/dy) → 平坦梯度下"按零成本键"是稳定点; babble 未能根治 (采样熵使 n-gram 计数饱和 → 对 argmax 无差异化压力) | **动量语义**: actions 8-11 沿用上一次移动命令 (边走边操作); 冻结经济学溶解 (连按 release = 边走边松手); 取舍记录: 绝对静止不再可达 (合法静止价值低, 露营金矿已由 B10 关闭) |
| B15 | **2026-10-27** | op 带夜间缓慢下移 (0.65→0.50); 逐任务 diag 全体均匀下移 | **ZPD 水平口径的收益规模误排**: 回报本金随物体数增加 (稀疏 ~600-1200 vs 拥挤 ~2100-2500) → 稀疏任务永远"看起来没掌握" → 被过度练习 (p=0.44-0.64) → 高物体场景曝光不足 → 技能均匀漂移 | 选课改 **LP 口径** (快/慢 EMA 相对斜率: 练"正在变好"的任务; 停滞→地板; 未观测→探索权) — 测试 14/14 (含防预热假进度); 配置 `knowledge_lp_selection: true` |

## C. 运维类（Ops — 流程缺陷, 非代码）

| ID | 日期 | 症状 | 根因 | 修复/规则 |
|---|---|---|---|---|
| C1 | 2026-09-24 | 12M 段瞬间退出 (8 秒), GPU 空转 | 部署本地旧 config 覆盖服务器 `total_steps` (6.5M) | **铁律: config 部署必须 merge, 禁止整文件覆盖**; 本地同步 |
| C2 | 2026-09-25 | 自动关机失败, 空转 5.5h | `/usr/sbin/shutdown` 不存在 (实际 `/usr/bin/`) | 路径修正 (后验证有效) |
| C3 | 2026-09-27 | 全量评测可能被自动关机中途杀死 | busy() 名单不含 run_stage18/sysprobe | 补名单 |
| C4 | 2026-09-28 | watcher 按 step 去重 → 覆盖同名 ckpt 漏测 | 去重键 = step (新段覆盖后即被跳过) | 键 = step + 文件 mtime |
| C5 | 2026-09-29 | 12.45M resume 目标被 prune 误删 | 链 prune 按 mtime 保留 6 个, 最旧即目标 | cp-before-prune + 数据盘备份 |
| C6 | 2026-09-26..10-08 | 多次重复进程 (3× 训练/评测) | paramiko 重试环重放副作用 (nohup 挂 channel → 超时 → 重试) | **守护式 launcher** (pgrep 守卫) + 幂等步骤 |
| C7 | 2026-09-25 | `pkill -f "pattern"` 自杀 (self-match) | pkill 的 -f 匹配到执行它的 shell 自身 | 方括号模式 `src[.]train` |
| C8 | 2026-10-11 | G3 数感复验误读 (num 0.30, 非能力) | 选档错误: 用了 `12150000` (谷相, diag 0.172), 而非好相位 `012151904` (0.712) — 同名相近、仅差 1904 步 | **规则: 评测选档必须先查该档的 op diag 相位; 好相位 ckpt 及时备份并注明相位** |
| C9 | 2026-10-11 | 训练被"莫名关机"打断 (soak6 @14.94M) | **云平台侧重启** (容器 uptime 41min; auto_stop 无记录, NO_AUTO_STOP 在, dmesg 无 OOM, 用户确认无手动) | 恢复流程: 开机 → 查最新 ckpt → `launch_soak.sh` 续跑 (~2 分钟); 容器无 cron → 无法开机自恢复 (可在控制台查"自定义启动脚本"功能项); launcher 已加 ckpt 存在性检查 |
| C10 | **2026-10-27** | 链 16:24 到上限后停 2h (GPU 空转), 无人察觉 | **一次性接力在周期启动时即消耗** (它负责的是下一次 cap), 之后新周期的 cap 无保险 → 停链 | `relay_loop.sh` 自我续挂 (循环: 等链退出 → 检查 STOP/灾难 → 重启链 → 继续), 一次启动覆盖所有后续 cap; 规则: **任何'一次性保险'必须检查其覆盖面是否为后续事件** |

## 同类扫描记录（2026-10-09, 用户要求）

对 A/B 类做全模块扫描：
- **设备混用**: contradiction (已修 B6); meta_monitor/deliberation 全 CPU 一致性确认 ✓;
  `_event_log` 纯 JSON 无张量 ✓。
- **重复处罚/非幂等周期动作**: contradiction (已修 B7); `_apply_rule_outcome_feedback`
  usage-增量 ✓; EWC consolidate 属设计性重复 ✓。
- **无界增长**: 事件日志 (已修 B8); monitor/gate 的 deque 均 maxlen ✓;
  检测器解记忆有界 ✓。
- **混合精度**: contradiction 测试覆盖 double/float ✓。
- **跨进程写入**: 事件日志 append 交错 (重复实例时) — 由 C6 的守护式 launcher
  根控; 残留风险记录在案。
- **分布完整性 (A11 追加)**: 部分文件部署 → 新属性缺失 → 关键路径死亡。
  新增规则: **部署 = 全部相关文件 + 逐模块 marker 校验**; 遥测性功能
  (modulate 等) 必须有防御性 fallback, 不得拖死关键路径。
- **契约双向核对 (A12 追加, 2026-10-11)**: 恢复映射 37 键全部核验 ✓ (save
  侧 50 键)。反向残留: 4 个 **save-only** 键 (`llm_fusion_state` /
  `cross_domain_state` / `value_system_state` / `cf_regret_state`) — 模块
  有存档无读档, 重启即重置 (低危: 均为诊断/计数型瞬态; 待其成为承载状态时
  再补恢复端)。
- **一次性保险不覆盖后续事件 (C9, 2026-10-26)**: relay 脚本只安排了一次
  接力, 链结束后 2 小时空转 → **规则: 保险必须是自续体 (relay_loop 循环
  重挂), 不是一次性定时器**。
- **自续接力 (C10, 2026-10-27)**: `relay_loop.sh` 每 15 分钟检查, 无链则
  自动启动, 覆盖所有未来长任务 — 一次性 relay 的永久修复。
- **PowerShell 前置展开 (C11, 2026-10-28, 已复现 4 次)**: 在 `@""` here-string
  里, `$VAR` / `$(cmd)` 会被 PowerShell **先于** python 展开 (下载脚本变成
  `echo ... >> ""` 静默失败)。→ **规则: 任何带 `$` 的远程脚本必须写本地
  文件 → sftp 二进制上传 (+ sed 去 CRLF), 禁止内联 here-string 生成**。

---

## 统计

- A 类 (静默断链): **16** — 特征: `hasattr` 守卫/空 except/无调用者/单向契约/退火泄漏
  → **规则: 关键路径禁止静默吞异常 (§14); 新接线必须带"变更日志"证明生命;
  恢复映射与保存端必须双向核对 (A12); 开关变量的关闭必须显式赋值 (A13)**
- B 类 (设计): **18** — 特征: 半修复覆盖、口径错配、无幂等、**边界条件未覆盖**、
  **无界奖金形塑出退化吸引子 (B10)**、**双分支奖励阶梯分离 (B11)**、
  **评估仪器自欺 (B16: 真值表同值无区分度 + 松口径白送分)**、
  **词表子串匹配 (B17: "disappeared" 命中 "red") + 单测无反例 = 假绿**、
  **窗口边界状态未重置 (B18: goal 首次达成后 _awarded 不复位 → 473 窗口只记
  1 次; 阴影遥测在注入前抓住 — 先测后奖的纪律兑现)**
  → **规则: 修一处必查同类; 周期性动作必须幂等; 续跑链必须"边界安全"
  (resume 步数 ≥ target 时先扩 target); 任何"每步持续收益"必须设有界窗口;
  评测器上线前必须过"反例杀得死"自检; 词表/解析一律词边界匹配;
  窗口/周期状态必须在窗口开始时显式复位 (B18)**
- C 类 (运维): **11** — 特征: 覆盖式部署、路径假设、重试放大、一次性保险、
  PowerShell 前置展开
  → **规则: 部署必 merge; 启动用守护式 launcher; 停止用方括号模式;
  保险必自续; 带 `$` 的远程脚本一律本地写文件后 sftp**
