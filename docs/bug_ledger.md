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
- **分布完整性 (B9/A11 追加)**: 部分文件部署 → 新属性缺失 → 关键路径死亡。
  新增规则: **部署 = 全部相关文件 + 逐模块 marker 校验**; 遥测性功能
  (modulate 等) 必须有防御性 fallback, 不得拖死关键路径。

---

## 统计

- A 类 (静默断链): **10** — 特征: `hasattr` 守卫/空 except/无调用者
  → **规则: 关键路径禁止静默吞异常 (§14); 新接线必须带"变更日志"证明生命**
- B 类 (设计): **8** — 特征: 半修复覆盖、口径错配、无幂等
  → **规则: 修一处必查同类; 周期性动作必须幂等**
- C 类 (运维): **7** — 特征: 覆盖式部署、路径假设、重试放大
  → **规则: 部署必 merge; 启动用守护式 launcher; 停止用方括号模式**
