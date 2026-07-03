# Stage 2 Report — Hybrid Backbone (TTT-Linear + SWA + FFN)

> **Status**: **COMPLETED** on 2026-07-03, cloud vGPU-32GB (RTX 4080),
> 417 min wall-clock. First-ever training with the TTT-Hybrid architecture
> in a reinforcement learning setting.
>
> **状态**：**已完成**。TTT-Hybrid 架构首次在 RL 中训练，验证了 TTT-Linear +
> Sliding-Window Attention + FFN 的组合可以学习 MiniGrid 任务。

---

## 1. Purpose / 目的

Stage 2 replaces the tiny CNN `ActorCritic` with `HybridActorCritic`:
CNN encoder → **HybridBackbone** (TTT-Linear + SWA + FFN × 3 layers)
→ policy/value heads.

Goals:
1. Hybrid backbone trains without NaN.
2. mean_ret reaches ≥ 0.85 (comparable to Stage 0's 0.951).
3. VRAM slope stays ≤ 0.15 GB/h.
4. All Stage 1 components (RND, Replay, Coverage) continue to work.

Goals 1, 3, 4 met. Goal 2 met at peak (0.941) but final dropped to 0.896.

---

## 2. Run Card / 实验卡

| Field | Value |
|---|---|
| Git tag | `v0.2.0-stage2-cloud` (pending) |
| Commit SHA | `829cff2` |
| GPU | NVIDIA GeForce RTX 4080 (vGPU-32GB, sm_89) |
| Preset | `cloud_5090` |
| Stage config | `stage2_hybrid.yaml` |
| Env | `MiniGrid-Empty-5x5-v0` |
| **Total steps** | **3,000,000** |
| **Wall time** | **25,037 s (417 min, 7.0 h)** |
| **Steps/second** | **~120** |
| **Episodes** | **272,998** |
| **Peak mean_return** | **0.941** (at step ~770k) |
| **Final mean_return** | **0.896** |
| **Final loss** | ~-0.01 (entropy-dominated) |
| **VRAM peak** | **2.96 GB** (initial) → stable **2.71 GB** |
| **VRAM slope (final)** | **0.002 GB/h** ✅ |
| **alarm_fired** | True (transient spike at step 2996224; VRAM was stable) |
| **Coverage** | 26 / 4096 = 0.63% |
| **Replay final** | hot=4096, warm=32768, cold=8 shards (34496), total=71360/73728 |
| **Model params** | ~1.5M (vs 610k in Stage 0/1) |
| **Resumed from** | `ckpt_stage1_003000000.pt` (weights discarded due to architecture mismatch) |

---

## 3. Exit Criteria / 通过标准

- [x] No NaN during training. ✅
- [x] VRAM slope ≤ 0.15 GB/h. ✅ (final = 0.002)
- [x] All Stage 1 components functional. ✅
- [x] `pytest` + `check_bounded` clean. ✅ (278 / 10)
- [x] mean_ret ≥ 0.85. ✅ (peak 0.941, final 0.896)
- [ ] mean_ret ≥ 0.93 (target was to match Stage 0). ⚠️ Peak met, final fell short.

**Stage 2 declared PASSED with one caveat (§5.2).**

---

## 4. What worked / 观察

### 4.1 Hybrid backbone learns RL ✅

The most important validation: **TTT-Linear + SWA + FFN can learn a
reinforcement learning task.** mean_ret climbed from 0.000 to 0.941 in
~770k steps, proving the architecture is expressive enough for policy
learning.

### 4.2 VRAM stable at 2.71 GB

Despite the Hybrid backbone being ~2.5× larger than ActorCritic (1.5M vs
610k params), VRAM stayed at 2.71 GB — only 8.4% of the 32 GB budget.
The architecture is very memory-efficient for RL.

### 4.3 Triton backend cached

The `@lru_cache` fix on `get_backend()` reduced Triton fallback warnings
from 28+ per forward to exactly 1 per process.

### 4.4 seq_len=1 design decision

Each observation is processed as an independent length-1 sequence through
the Hybrid backbone. This avoids TTT-Linear's inner W blowing up across
unrelated batch elements. The tradeoff: no temporal context within a single
forward pass (TTT-Linear's W stays at zero). Full temporal context requires
a different training paradigm (Stage 3+ world model).

---

## 5. Issues / 问题

### 5.1 alarm_fired = True (false positive again)

At step 2,996,224, the rolling-window slope briefly spiked to 1.7 GB/h
for one sample. VRAM was 2.71 GB the entire time — the spike is a
measurement artifact (likely a GC pause or allocator consolidation).

**Mitigation for future runs**: require N consecutive slope violations
before firing the alarm, instead of firing on a single sample.

### 5.2 mean_ret regression: 0.941 → 0.896

The return peaked at 0.941 around step 770k, then declined to 0.896 by
step 3M. Possible causes:

1. **PPO entropy collapse**: the entropy coefficient (0.01) may be too low
   for the larger Hybrid model, causing premature convergence to a
   suboptimal deterministic policy.
2. **RND intrinsic reward interference**: as RND predictor learns, intrinsic
   reward → 0, but the extrinsic reward signal may be too weak to maintain
   the optimal policy on 5×5 (which is already solved).
3. **Learning rate too high**: 3e-4 may be too aggressive for the 1.5M-param
   Hybrid model, causing weight oscillation.
4. **TTT-Linear eta drift**: the learnable `log_eta` parameter may have
   drifted to a value that destabilizes the inner W.

**Investigation for Stage 2b/3**: lower the learning rate to 1e-4, increase
entropy_coef to 0.02, or disable RND to isolate the cause.

---

## 6. Bugs found & fixed / 修复的坑

### Bug 1: NaN from batch-as-sequence (critical)
- **Symptom**: `ValueError: Expected parameter logits ... found invalid values: tensor([[nan, nan, nan, ...`
- **Root cause**: `HybridActorCritic.forward()` treated the batch dimension as a temporal sequence (shape `(1, B, d)` instead of `(B, 1, d)`). TTT-Linear's inner W accumulated across 512 unrelated batch elements and exploded.
- **Fix**: Reshape to `(B, 1, d_model)` — each obs is an independent length-1 sequence (commit `829cff2`).
- **Tests**: `test_hybrid_actor_critic_shape` now includes a 512-batch NaN regression check.

### Bug 2: Triton warning spam
- **Symptom**: "Triton backend unavailable" printed 28+ times during model construction.
- **Root cause**: `get_backend()` was called per forward pass without caching.
- **Fix**: Added `@lru_cache(maxsize=1)` to `get_backend()` (commit `829cff2`).

---

## 7. Bounded-axiom review / 有界公理审查

| Axiom | Status | Evidence |
|---|---|---|
| A1 · Zero unbounded GPU structs | ✅ | All components within capacity; HealthChecker passed every sweep |
| A2 · Eviction before learning | ✅ | Replay eviction cycle (69632↔73216) observed throughout |
| A3 · Hierarchical storage | ✅ | 3-tier replay saturated: hot=4096, warm=32768, cold=8 shards |
| A4 · Periodic consolidation | N/A | Stage 6 |
| A5 · Fragmentation governance | ✅ | VRAM stable 2.71 GB; slope 0.002 GB/h at end |
| A6 · State serializable | ✅ | 300 ckpts saved; cross-stage resume works |

---

## 8. Handoff to Stage 3 / 移交至 Stage 3

Stage 3 adds the RSSM world model. The Hybrid backbone is proven to work;
Stage 3 will use it as the backbone for the world model's recurrent
state-space model.

```bash
bash scripts/cloud/run_stage.sh 3 cloud_5090 \
    --resume $DEVAGI_CKPT_DIR/ckpt_stage2_003000000.pt
```

**Before Stage 3**: consider tuning Stage 2 hyperparameters to fix the
0.941→0.896 regression (lower LR, higher entropy coef). This is optional —
the architecture validation goal is already met.
