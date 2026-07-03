# Stage 1 Report — RND Curiosity + Bounded Replay (3-tier)

> **Status**: **COMPLETED** on 2026-07-03, cloud vGPU-32GB (RTX 4080 sm_89),
> 269 min wall-clock. Full 3M-step PPO+RND run with bounded 3-tier replay.
>
> **状态**：**已完成**（2026-07-03，AutoDL vGPU-32GB / RTX 4080，4.5 小时全量）。
> Stage 1 验证了 RND 好奇心 + 三层有界回放 + 覆盖率跟踪 + warmup 修复。

---

## 1. Purpose / 目的

Stage 1 adds **intrinsic motivation** (RND) and a **bounded three-tier
replay buffer** on top of the Stage 0 PPO baseline. Goals:

1. RND predictor trains stably alongside PPO.
2. BoundedReplayBuffer (GPU/CPU/SSD) fills and evicts correctly (Axioms 1, 2, 3).
3. BoundedCoverage tracks state-visitation without unbounded growth.
4. MemoryWatcher warmup fix prevents false slope alarms.
5. VRAM slope stays ≤ 0.15 GB/h over the full run.

All five goals met. See §3 for the exit-criteria checklist.

---

## 2. Run Card / 实验卡

| Field | Value |
|---|---|
| Git tag | `v0.1.0-stage1-cloud` (pending push) |
| Commit SHA | `490b34f` |
| Cloud vendor | AutoDL |
| Container | `c29141b9c1-00c92742` |
| GPU | NVIDIA GeForce RTX 4080 (vGPU-32GB slice, sm_89) |
| CPU | 16 cores Xeon Platinum 8352V |
| RAM | 62 GB |
| System disk | 30 GB (1.89% used) |
| Data disk | 50 GB @ `/root/autodl-tmp` (1.37% used) |
| Driver / CUDA | 580.105.08 / 13.0 |
| OS | Ubuntu 22.04 |
| Python | 3.12.3 |
| torch | 2.5.1+cu124 |
| Triton | 3.1.0 |
| Preset | `cloud_5090` (32 GB VRAM budget) |
| Stage config | `stage1_curiosity.yaml` |
| Env | `MiniGrid-Empty-5x5-v0` |
| **Total steps** | **3,000,000** |
| **Wall time** | **16,160 s (269 min, 4.5 h)** |
| **Steps/second** | **~186** |
| **Episodes** | **598,995** |
| **Final mean return** | **0.955** |
| **Final loss** | ~0.0000 (converged from Stage 0) |
| **Peak VRAM used** | **0.93 GB / 32 GB (2.9%)** |
| **VRAM slope (final)** | **0.0 GB/h** ✅ |
| **alarm_fired** | **False** ✅ (warmup fix verified) |
| **Coverage** | 11 unique buckets / 4096 = **0.27%** |
| **Replay final state** | hot=4096/4096, warm=32768/32768, cold=8 shards (34496 entries), total=71360/73728 |
| Checkpoints saved | 300 (every 10,000 steps) |
| Unit tests | 278 passed / 10 skipped |
| `check_bounded` | OK (37 source files) |

**Resumed from**: `ckpt_stage0_003000000.pt` (Stage 0 final). Cross-stage
resume reset step counter to 0 per the fix in commit `3f3e5e2`.

---

## 3. Exit Criteria / 通过标准

- [x] `make check-bounds` clean. ✅
- [x] `pytest -x tests/` all green. ✅ (278 / 10)
- [x] VRAM slope ≤ 0.15 GB/h. ✅ (final = 0.0 GB/h)
- [x] `alarm_fired == false`. ✅ (warmup fix worked)
- [x] BoundedReplayBuffer three-tier eviction functional. ✅ (hot/warm/cold all saturated; periodic eviction cycle observed)
- [x] RND predictor trained without NaN. ✅
- [x] Checkpoint round-trips with `--resume`. ✅ (300 ckpts saved + loaded across stages)
- [x] Cross-stage resume (Stage 0 → Stage 1) resets step counter. ✅

**Stage 1 declared PASSED.**

---

## 4. What worked / 观察

### 4.1 Three-tier bounded replay — eviction cycle observed

The replay buffer exhibited the expected **periodic eviction cycle**:

```
replay=73216/73728  ← near full (hot + warm + cold near capacity)
replay=69632/73728  ← eviction: ~4096 entries demoted/evicted
replay=70144/73728  ← climbing again
...
replay=73216/73728  ← full again → evict
```

This cycle (~512 steps) confirms:
- Hot ring (4096) fills → demotes to warm
- Warm ring (32768) fills → demotes to cold
- Cold shards (8 × 4096) fill → oldest shard deleted
- **Axioms 1, 2, 3 all empirically validated** ✅

### 4.2 Warmup fix — no false alarm

Stage 0 had `alarm_fired=True` due to startup CUDA-context allocation
looking like a fast slope. Stage 1 with the `warmup_seconds=300` fix:
- `alarm_fired=False` throughout the entire 4.5-hour run
- Slope started at 0.024 GB/h after warmup, monotonically decreased to 0.0

### 4.3 Cross-stage resume — step counter reset

Resuming Stage 1 from a Stage 0 ckpt at step 3,000,000 correctly reset the
step counter to 0 (commit `3f3e5e2`). Without this fix, the training loop
would have exited immediately (`state.step >= total_steps`).

### 4.4 ColdShardTier capacity fix

Mid-run (at step ~69632 in the first attempt), the HealthChecker crashed
because `ColdShardTier.__len__` exceeded its declared `capacity` by 512
entries (the pending buffer). Fixed by including one shard's worth of
pending allowance in `capacity` (commit `490b34f`). After the fix, the
run completed 3M steps without any capacity violations.

### 4.5 Stable performance

- VRAM usage: 0.87 → 0.93 GB (flat after initial warmup)
- Throughput: ~186 step/s (vs Stage 0's ~768 step/s — 4.1× slower due to
  RND forward, replay push, PER sampling, TD updates)
- mean_return: 0.955 throughout (inherited from Stage 0; MiniGrid-5x5 is
  already solved, so no further improvement expected)

---

## 5. What surprised us / 意外

1. **4.1× slowdown vs Stage 0** — heavier than expected. The bottleneck is
   CPU-side: `env.step()` + replay `add()` + RND `intrinsic_reward()` all
   run in the rollout loop. On GPU the cost is minimal (0.93 GB VRAM).

2. **Coverage only 0.27%** — MiniGrid-Empty-5x5-v0 has a tiny state space
   (~50 unique states). The agent converges to a narrow corridor of optimal
   states. Coverage will be meaningful starting Stage 5 (multiple env variants).

3. **Cold tier steady-state at 34496/36864** — not quite full because the
   eviction cycle keeps one shard's worth of headroom for the pending buffer.
   This is by design (the capacity fix).

4. **No return improvement over Stage 0** — expected, since MiniGrid-5x5 is
   already solved at 0.951. RND's value shows up in harder environments.

---

## 6. Bugs found & fixed / 修复的坑

### Bug 1: Cross-stage resume didn't reset step counter
- **Symptom**: Stage 1 exited immediately (`steps=3000000 elapsed=0.0s`)
- **Root cause**: `state.step` inherited from Stage 0 ckpt (3M) which equals Stage 1's `total_steps` (3M)
- **Fix**: When resumed stage ≠ target stage, reset `state.step = 0` (commit `3f3e5e2`)
- **Tests**: `test_resume_cross_stage.py` (3 regression tests)

### Bug 2: ColdShardTier capacity didn't account for pending buffer
- **Symptom**: `BoundedComponentError: Bounded component 'bounded_replay' exceeded capacity: size=70144 cap=69632`
- **Root cause**: `capacity` was `max_shards * shard_size` but `__len__` included pending entries (up to `shard_size - 1`)
- **Fix**: `capacity` now = `(max_shards + 1) * shard_size` + eviction-before-flush (commit `490b34f`)
- **Tests**: 2 regression tests in `test_bounded_replay.py`

---

## 7. Bounded-axiom review / 有界公理审查

| Axiom | Status | Evidence |
|---|---|---|
| A1 · Zero unbounded GPU structs | ✅ | All components declared capacities; HealthChecker passed every sweep |
| A2 · Eviction before learning | ✅ | Replay eviction cycle observed (73216 → 69632 → 73216 ...) |
| A3 · Hierarchical storage | ✅ | Hot (4096) → Warm (32768) → Cold (8 shards) all operational |
| A4 · Periodic consolidation | N/A | Introduced in Stage 6 |
| A5 · Fragmentation governance | ✅ | `alarm_fired=False`, slope 0.024→0.0 GB/h |
| A6 · State serializable | ✅ | 300 ckpts saved with RND + coverage state; `--resume` works across stages |

---

## 8. Metrics snapshot / 指标快照

- **Loss**: ~0.0000 throughout (converged from Stage 0)
- **Memory**: flat at 0.87–0.93 GB from step ~5k onward
- **Replay fill**: steady-state cycle 69632 ↔ 73216 (every ~512 steps)
- **Coverage**: 11 unique buckets out of 4096 (0.27%)

Raw data:
- `/root/autodl-tmp/karbon/logs/stage1/20260703_100951_490b34f/memory.csv`
- `/root/autodl-tmp/karbon/logs/stage1_run.log`

---

## 9. Reproducibility recipe / 复现

```bash
git clone https://github.com/wuhaodesq/karbon.git
cd karbon
git checkout v0.1.0-stage1-cloud
bash scripts/cloud/setup_env.sh --skip-torch
source .venv/bin/activate

export DEVAGI_CKPT_DIR=/root/autodl-tmp/karbon/ckpts
export DEVAGI_DATA_DIR=/root/autodl-tmp/karbon/data
export DEVAGI_LOGS_DIR=/root/autodl-tmp/karbon/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Requires a Stage 0 final checkpoint at $DEVAGI_CKPT_DIR/ckpt_stage0_003000000.pt
bash scripts/cloud/run_stage.sh 1 cloud_5090 \
    --resume $DEVAGI_CKPT_DIR/ckpt_stage0_003000000.pt
```

Expected: `Training finished. steps=3000000 ... mean_ret in [0.95, 0.96] ... elapsed ~ 16000s`

---

## 10. Handoff to Stage 2 / 移交至 Stage 2

Prerequisites verified for Stage 2:

- [x] Stage 1 ckpt at `ckpt_stage1_003000000.pt` saved
- [x] All Stage 1 modules (RND, Replay, Coverage) stable
- [x] `HybridActorCritic` class ready in `src/train.py` (Stage 2 pre-work)
- [x] `configs/stage2_hybrid.yaml` ready
- [x] Cross-stage resume proven (Stage 0 → Stage 1 worked)

**Next**: Replace `ActorCritic` with `HybridActorCritic` (TTT-Linear + SWA + FFN).
The model architecture changes, so `--resume` will warn "Model state mismatch"
and start the Hybrid model fresh — this is expected and correct.

```bash
bash scripts/cloud/run_stage.sh 2 cloud_5090 \
    --resume $DEVAGI_CKPT_DIR/ckpt_stage1_003000000.pt
```
