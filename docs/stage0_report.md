# Stage 0 Report — Skeleton, Monitoring, PPO Baseline

> **Status template**: fill in the placeholders (`⟨...⟩`) after each experiment
> before committing. This file is the reproducibility artefact for
> `v0.0.0-stage0-local` and `v0.0.0-stage0-cloud`.
>
> **状态**：这是模板。跑完后把 `⟨...⟩` 占位替换成实测数据后提交，作为
> `v0.0.0-stage0-local` / `v0.0.0-stage0-cloud` tag 的可复现产物。

---

## 1. Purpose / 目的

Stage 0 exists to **validate the skeleton** — not to reach any RL performance
target. It must prove:

1. All axioms are enforced by construction (`make check-bounds` clean).
2. Preset system works across `local_smoke` / `cloud_24g` / `home_64g`.
3. `MemoryWatcher` samples continuously, writes CSV, detects slope alarms.
4. Longevity harness runs to completion on both local (short) and cloud (24 h).
5. Checkpoint schema round-trips.

阶段 0 只是骨架自证：所有公理、监控、preset、checkpoint 走通。**不追求 RL 分数**。

---

## 2. Run Cards / 实验卡

### 2.1 Local smoke — `v0.0.0-stage0-local`

| Field | Value |
|---|---|
| Git tag | `v0.0.0-stage0-local` |
| Commit SHA | `⟨fill-in⟩` |
| Machine | HP 240R G9, i5-1335U, 16 GB RAM, no dGPU |
| OS | Windows 11 |
| Python | 3.10.0 |
| torch | 2.5.1+cpu |
| Preset | `local_smoke` |
| Env | `MiniGrid-Empty-5x5-v0` |
| Total steps | ⟨200⟩ |
| Wall time | ⟨...s⟩ |
| Final mean episode return | ⟨...⟩ |
| Final loss | ⟨...⟩ |
| Peak process RSS | ⟨... GB⟩ |
| RSS slope (over run) | ⟨... GB/h⟩ |
| Unit tests | ⟨N passed / M skipped⟩ |
| `check_bounded` | ⟨OK / findings⟩ |

Attach: `logs/stage0/⟨run_id⟩/memory.csv` and any relevant chart under
`docs/figures/stage0_local_memory.png`.

### 2.2 Cloud first run — `v0.0.0-stage0-cloud`

| Field | Value |
|---|---|
| Git tag | `v0.0.0-stage0-cloud` |
| Commit SHA | `⟨fill-in⟩` |
| Machine | ⟨e.g., "AutoDL RTX 4090 24GB"⟩ |
| OS | Ubuntu ⟨22.04⟩ |
| Python | 3.10.⟨x⟩ |
| torch | 2.5.1+cu121 |
| triton | ⟨version⟩ |
| Preset | `cloud_24g` |
| Env | `MiniGrid-Empty-5x5-v0` |
| Total steps | ⟨...⟩ |
| Wall time | ⟨...h⟩ |
| Final mean return | ⟨...⟩ |
| Peak VRAM used | ⟨... GB⟩ |
| VRAM slope over 24 h | ⟨... GB/h⟩ |
| Alarm fired | ⟨no / yes⟩ |
| Unit tests | ⟨N passed / M skipped⟩ |
| `check_bounded` | ⟨OK⟩ |

Attach: `logs/stage0/⟨run_id⟩/memory.csv`, `docs/figures/stage0_cloud_memory.png`,
and `logs/stage0/⟨run_id⟩/longevity_report.json`.

---

## 3. Exit Criteria / 通过标准

Stage 0 is complete when **all** of the following hold:

- [ ] `make check-bounds` on the tagged commit prints "OK".
- [ ] `pytest -x tests/` all green on both local and cloud environments.
- [ ] Cloud 24 h longevity report shows `VRAM slope ≤ 0.2 GB/h`.
- [ ] `alarm_fired == false` in the longevity report.
- [ ] Cloud smoke round-trips a checkpoint: save → load → resume produces identical
      loss on the next batch (`docs/figures/stage0_ckpt_roundtrip.png`).
- [ ] All artefacts committed to Git; large logs referenced but stored via
      rsync-mirror (per `MIGRATION.md`).

---

## 4. What worked / 观察

- ⟨fill-in⟩ Skeleton wiring:
- ⟨fill-in⟩ Preset switch (local_smoke → cloud_24g):
- ⟨fill-in⟩ Memory-watcher slope alarm behaviour:
- ⟨fill-in⟩ Rollout buffer capacity enforcement:

---

## 5. What surprised us / 意外

- ⟨fill-in⟩

---

## 6. Bugs found & fixed / 修复的坑

- ⟨fill-in⟩

---

## 7. Bounded-axiom review / 有界公理审查

| Axiom | Compliance status | Notes |
|---|---|---|
| A1 · Zero unbounded GPU structs | ⟨OK⟩ | RolloutBuffer + all monitoring buffers pre-declared |
| A2 · Eviction before learning | ⟨OK⟩ | `RolloutBuffer.add` raises when full |
| A3 · Hierarchical storage | ⟨N/A this stage⟩ | Introduced in Stage 1 with `BoundedReplayBuffer` |
| A4 · Periodic consolidation | ⟨N/A this stage⟩ | Introduced in Stage 6 |
| A5 · Fragmentation governance | ⟨OK⟩ | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; empty_cache scheduled |
| A6 · State serializable | ⟨OK⟩ | Checkpoint round-trips; RolloutBuffer state_dict TBD in Stage 1 |

---

## 8. Metrics snapshot / 指标快照

Paste (or link to) key charts:

- **Loss over training steps** — `docs/figures/stage0_loss.png`
- **Memory over wall-clock** — `docs/figures/stage0_memory.png`
- **Episode length distribution** — `docs/figures/stage0_ep_lengths.png`

For long runs, prefer log-scale y-axis on memory to reveal creep.

---

## 9. Reproducibility recipe / 复现

```bash
# On the target machine:
git clone <repo>
cd karbon
git checkout v0.0.0-stage0-cloud     # or -local
bash scripts/cloud/setup_env.sh      # or PowerShell setup_env.ps1 on Windows
source .venv/bin/activate
pytest -x tests/
bash scripts/cloud/run_stage.sh 0 cloud_24g
```

Expected output: mean episode return within ⟨±X%⟩ of the number in §2.

---

## 10. Handoff to Stage 1 / 移交至 Stage 1

Prerequisites verified for Stage 1:

- [ ] `MemoryWatcher` has clean shutdown on SIGTERM (test in Stage 1's setup).
- [ ] `RolloutBuffer` cleanly implements `BoundedComponent` (already covered).
- [ ] `HealthChecker` registered in `train.py` — carry forward as Stage 1
      adds `BoundedReplayBuffer` and RND.
- [ ] Stage 0 config file frozen; Stage 1 will layer on top via YAML overlay.

Next: RND intrinsic reward + three-tier bounded replay.
See `configs/stage1_curiosity.yaml` (to be authored).
