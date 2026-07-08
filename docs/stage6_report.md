# Stage 6 Report - Perpetual (EWC + GR + Sleep)

> **Status**: **COMPLETED** on 2026-07-08, cloud vGPU-32GB (RTX 4080),
> 487 min wall-clock. All 10 components ran simultaneously. EWC consolidated,
> VAE converged, Sleep loop fired. Agent recovered from EWC interference.
>
> **状态**：**已完成**。10 个组件同时运行，EWC 固化，VAE 收敛，睡眠循环触发。

---

## 1. Run Card

| Field | Value |
|---|---|
| Git tag | `v1.0.0-stage6-cloud` (pending) |
| **Total steps** | **3,000,000** |
| **Wall time** | **29,242 s (487 min, 8.1 h)** |
| **Episodes** | **96,516** |
| **Final mean_return** | **0.951** |
| **VRAM** | **2.81 GB**, slope 0.0 ✅ |
| **Coverage** | **8.6% (353 buckets)** |
| **Skills** | 10496/10496 (672,158 created) |
| **EWC** | ✅ consolidated (7.26M params tracked) |
| **GR VAE** | gr=0.001 (converged) |
| **Sleep Loop** | replay_trim×10, skills_merge×10, ewc×2 |

---

## 2. All 10 Components Active

```
Starting Stage 6: intrinsic=True replay=True coverage=True wm=True
                     skills=True curriculum=True ewc=True gr=True sleep=True
```

All ten components ran simultaneously for 3M steps without crash.

---

## 3. EWC Interference and Recovery

| Step | mean_ret | Event |
|---|---|---|
| 0 | 0.955 | Inherited from Stage 5 |
| 1.6M | EWC consolidated | During DoorKey (wrong task) |
| 1.94M | 0.001 | EWC penalty pulled weights toward DoorKey |
| 3.0M | **0.951** | **Recovered** - PPO overcame EWC penalty |

**Lesson**: EWC should consolidate AFTER mastering a task, not periodically.
The fix (step-delta trigger) would consolidate at step 100k (during Empty-8x8,
a mastered task), producing correct Fisher protection.

---

## 4. Sleep Loop Results

```
replay_trim_runs:     10
skills_merge_runs:    10
ttt_distill_runs:     10
ewc_consolidate_runs:  2
total_wall_seconds:    1.13s (negligible overhead)
```

All sleep tasks fired and completed without errors.

---

## 5. Coverage Growth Across All Stages

| Stage | Coverage | Buckets |
|---|---|---|
| Stage 0 | 0.1% | 5 |
| Stage 1 | 0.3% | 11 |
| Stage 2 | 0.6% | 26 |
| Stage 3 | 0.6% | 26 |
| Stage 4 | 0.6% | 26 |
| Stage 5 | 7.9% | 324 |
| **Stage 6** | **8.6%** | **353** |

Coverage grew 86× from Stage 0 to Stage 6.

---

## 6. Complete Training Summary (Stage 0-6)

| Stage | mean_ret | Time | VRAM | Key Achievement |
|---|---|---|---|---|
| 0 | 0.951 | 65 min | 0.82 GB | PPO baseline |
| 1 | 0.955 | 269 min | 0.93 GB | RND + Bounded Replay |
| 2 | 0.896 | 417 min | 2.71 GB | TTT-Hybrid backbone |
| 3 | 0.950 | 442 min | 2.70 GB | RSSM World Model |
| 4 | 0.914 | 849 min | 2.71 GB | Skill Library (10496) |
| 5 | 0.937 | 482 min | 2.71 GB | AutoCurriculum (3/5 tasks) |
| **6** | **0.951** | **487 min** | **2.81 GB** | **EWC + GR + Sleep (10 components)** |
| **Total** | | **~50 h** | | **6 stages, 18M steps** |
