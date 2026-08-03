# Stage 15 · 3D 世界迁移规划

> 状态：**规划阶段** (2026-08-04)
> 路线：Path B — 从 MiniGrid 迁移到 MuJoCo 3D 环境

---

## 1. 迁移动机

### 1.1 信息天花板已确认 (Stage 14)

MiniGrid 的 2.5D 渲染信息量不足：
- 像素 WM 重建 loss ≈ 4e-6 (Stage 3 起), 即使放大 200 倍仍无法驱动学习
- 编码成本 (KL nats) ≈ 0.1–0.5 >> 重建收益 → posterior collapse
- 想象训练 (Stage 12) 完全无效: WM 无法预测未来帧
- 因果发现在 experience 模式下仅 58/512 边, 多数为 legacy

### 1.2 3D 世界解决什么

| 瓶颈 | MiniGrid | ThreeDWorld (MuJoCo) |
|-------|----------|---------------------|
| 像素信息量 | unique≈12, edge=0.29 | unique≈3431, edge=4.98 (close-up) |
| 物体永存 | 不存在/消失无物理过程 | 被遮挡后仍存在, 真实碰撞 |
| 因果发现 | 弱 (effect≈0) | 真实物理因果 (力→运动, 碰撞→位移) |
| 想象训练 | WM 退化 | 高信息量像素 → 重建收益 > 编码成本 |
| 符号"根" | 空洞化风险 | 感知经验有真实物理基础 |

### 1.3 已有能力可迁移

| 能力 | 迁移方式 | 风险 |
|------|---------|------|
| 分层决策 (Manager + Worker) | 直接复用 (action_space=8 一致) | 低 |
| 技能库 (LoRA + SSD) | 直接复用 (3D 技能自动注入) | 低 |
| EWC (λ=50, γ=0.99, task-switch hook) | 直接复用 | 低 |
| 因果图 (experience mode) | 作为初始化, 3D 因果边继续增长 | 低 |
| 课程系统 | 重新设计 3D 任务课程 | 中 |
| WM (latent-only) | 重新启用 pixel recon | 中 |

---

## 2. 验证结果 (本会话)

### 2.1 渲染修复

**问题**: MuJoCo 渲染输出全白 (mean=255, std=0)

**根因**:
1. 场景 XML 无 camera 定义 → free camera 在视野外
2. `Renderer.render()` 在 mujoco 3.x 返回 uint8, 代码误用 `clip(0,1)*255`

**修复**:
- 添加 `targetbody` camera 跟随 learner
- 修正像素缩放逻辑
- camera 参数化 (pos, fovy 可配置)

**效果对比** (128×128, 10 objects, 40 steps):

| Camera 配置 | edge | unique | luma_std | frame_diff |
|------------|------|--------|----------|------------|
| 高远 (0,-3,2.2) f50 | 0.008 | 9 | 0.4 | 0.145 |
| 当前 (0,-2.2,1.6) f60 | 0.292 | 98 | 23.3 | 1.367 |
| 低近 (0,-1.2,0.5) f70 | 1.838 | 1351 | 27.4 | 9.798 |
| **close-up (0,-1,0.8) f60** | **4.976** | **3431** | **51.2** | **6.192** |

### 2.2 物理交互验证

- ✅ agent 移动: 30 步移动 2.19 单位
- ✅ 接触检测: 基于几何半径和的动态阈值 (contact_reach)
- ✅ 物体推动: position change verified
- ⚠️ force_motion_pairs: 特定场景为 0 (物体被墙卡住/分布稀疏)
- ⚠️ occlusion_events: 需要特定运动模式 (远离→再接近)

### 2.3 环境接口兼容

- `env_id="ThreeDWorld"` 已在 train.py 支持 (line 719–735)
- `action_space_n=8` (与 MiniGrid 一致)
- `obs=(128,128,3) uint8` + `proprio=(12,)`
- `VecThreeDWorld` 向量化版本可用
- `ExtendedThreeDWorld` (4 房间 + 兄弟) 可用

---

## 3. 迁移策略

### 3.1 保留 (直接复用)

```
src/models/ttt_backbone.py          # TTT-Hybrid 7 层
src/models/slot_attention.py        # SlotAttention
src/models/working_memory.py        # WM (latent-only → 重建)
src/models/hierarchical_policy.py   # Manager + Worker
src/models/causal_discovery.py      # experience mode
src/models/online_ewc.py            # λ=50, γ=0.99
src/memory/skill_library.py         # LoRA + SSD
src/memory/episodic_replay.py       # PER replay
src/training/ppo_trainer.py         # PPO (不变)
```

### 3.2 迁移 (需要适配)

| 组件 | 变更 |
|------|------|
| `src/train.py` | env_id 切换, recon_loss_weight 从 0→1.0, curriculum task 定义 |
| `src/eval/independent_evaluator.py` | 添加3D评估任务 (物体永存/推动/遮挡) |
| `src/eval/developmental_milestones.py` | 添加3D里程碑 |
| configs/stage15_3d.yaml | 新配置文件 |

### 3.3 新建

| 组件 | 用途 |
|------|------|
| 3D curriculum tasks | 物体永存、推动实验、遮挡追踪 |
| 3D eval benchmarks | 物体永存 SR、推动成功率、遮挡理解 |
| WM 重建验证脚本 | 验证 3D 像素重建 loss 下降 |

---

## 4. 配置设计 (stage15_3d.yaml)

### 4.1 关键参数变更

```yaml
# 从 latent-only 切换到 pixel recon
world_model:
  recon_loss_weight: 1.0          # was 0.0 (latent-only)
  recon_pixel_weight: 200.0       # MSE weight
  kl_free_nats: 0.0               # 保持
  z_dim: 32
  h_dim: 64

# 3D 环境
env:
  id: "ThreeDWorld"
  num_objects: 8                  # 中等密度
  render_size: 128                # close-up camera 验证最优
  action_force: 50.0
  camera_pos: [0.0, -1.0, 0.8]
  camera_fovy: 60.0

# EWC (从 Stage 14 继承)
continual:
  ewc_lambda: 50.0
  ewc_gamma: 0.99
  ewc_consolidate_every_steps: 25000

# 课程: 3D 物理任务
curriculum:
  max_tasks: 5
  switch_every_steps: 50000
  tasks:
    - {id: 0, num_objects: 3, action_force: 70.0, difficulty: 0.1, tag: "3d-sparse"}
    - {id: 1, num_objects: 5, action_force: 60.0, difficulty: 0.3, tag: "3d-few"}
    - {id: 2, num_objects: 8, action_force: 50.0, difficulty: 0.5, tag: "3d-base"}
    - {id: 3, num_objects: 12, action_force: 45.0, difficulty: 0.7, tag: "3d-many"}
    - {id: 4, num_objects: 16, action_force: 40.0, difficulty: 0.9, tag: "3d-crowded"}
```

### 4.2 恢复 MiniGrid 知识保护

- EWC consolidation 在 curriculum switch 时触发 (已有 hook)
- 新增3D任务不会覆盖 MiniGrid Fisher 信息
- 评估同时跑 MiniGrid (保留) + 3D (新增)

---

## 5. 训练计划

### 5.1 阶段划分

| 子阶段 | 目标 | 步数 | 验证 |
|--------|------|------|------|
| 15a: Smoke test | 3D 环境跑通, WM 重建 loss 下降 | 50K | recon_loss 持续下降 |
| 15b: 基础训练 | 物体交互, 因果边增长 | 200K | effect>0, 物体位移 |
| 15c: 物体永存 | 遮挡→再现, 符号雏形 | 300K | occlusion_events>0 |
| 15d: 全认知 | 想象+因果+符号联合 | 500K | 想象重建 loss 下降 |

### 5.2 评估指标

| 指标 | 目标 | 基线 (Stage 14) |
|------|------|-----------------|
| WM recon loss | < 0.01 | N/A (latent-only) |
| 因果边数 | > 100 | 58/512 |
| 物体永存 SR | > 0.3 | N/A |
| 推动成功率 | > 0.5 | N/A |
| 遮挡理解 SR | > 0.2 | N/A |
| MiniGrid SR (保留) | > 0.2 | 0.20 (empty-5x5) |
| 内置 eval total | > 0.7 | 0.648 |

### 5.3 时间估算

- GPU: RTX 3080 Ti 12GB
- 预计 1M 步 ≈ 24–48 小时 (取决于 WM 重建计算量)
- 总训练预算: 1M 步

---

## 6. 风险分析

| 风险 | 影响 | 缓解 |
|------|------|------|
| 3D 渲染仍然信息不足 | WM 无法学习 | 调整 camera/分辨率/物体密度 |
| 物体太大/太小 | 接触检测失败 | 用 contact_reach 动态阈值 (已修复) |
| EWC 保护过度 | 3D 新任务学不动 | 降低 lambda 或用 task-specific fisher |
| GPU 内存不足 | 128px 渲染+WM 重建 | 降 render_size=64 或 batch_size |
| 因果发现噪声 | 错误边 | 提高 min_intervention_effect 阈值 |

---

## 7. 成功标准 (Stage 15 exit)

- [ ] WM recon loss 在 3D 上持续下降 (vs Stage 14 的 latent-only)
- [ ] 因果边 > 100 (vs 58)
- [ ] 物体永存 eval SR > 0.3
- [ ] MiniGrid SR 保持 > 0.2 (EWC 保护有效)
- [ ] `make test && make check-bounds` 通过
- [ ] CHANGELOG + stage15_report.md 更新

---

## 附录: 文件清单

### 新增文件
- `configs/stage15_3d.yaml`
- `docs/stage15_3d_migration_plan.md` (本文档)

### 修改文件
- `src/envs/three_d_world.py` — camera 修复 + 参数化 + contact_reach

### 服务器路径
- `/root/karbon/` — 项目根
- `/root/karbon/checkpoints/ckpt_stage14_001000000.pt` — Stage 14 最终 ckpt
- `/root/karbon/src/envs/three_d_world.py` — 已同步修复版
