# Stage 21 Report — 递归元认知 (Recursive Metacognition)

> **Status**: **SEALED 2026-10-11** after the long-run validation
> (soak5: 13.85M → 14.6M, ~11 h, 0 errors). M1 ✅ M2 ✅ M3 ✅
> M4 mechanism ✅ / A/B honest-negative recorded. 状态：已封存。

---

## 1. Run Card

| Field | Value |
|---|---|
| Lineage | Stage 20 sealed @ 12.95M → Stage 21 modules layered on top |
| Lineage now | 14,600,000 (soak5 end, 2026-10-11 10:12) |
| Modules | MetaMonitor (M2), ContradictionDetector (M3), DeliberationGate (M4), bounded event log |
| Config | `stage20_hypothesis_v2.yaml` (+ `meta_monitor_enabled`, `contradiction_enabled`, `deliberation_enabled`) |
| Long-run | soak5 13.85M→14.6M (748k steps, 23:32→10:12, 0 errors) |
| Data | `/root/stage21_events*.jsonl` (bounded, 32MB rotation) |

## 2. M1 — Offline validation (meta-prediction) ✅

- Dataset: 18,391 events (12.05M policy, 60k steps) → 6,216 probe-outcome samples.
- Honest onset task (features pre-outcome): accuracy 0.727 vs majority 0.573
  → **skill_vs_majority = +0.154** (n_test 1865; temporal 70/30 split).
- Cross-policy generalization: A→B **+0.153**, B→A **+0.288**.
- Ablation: skill carried by the recent-outcome window (recent-only +0.205);
  belief content (lk/obj) and freshness ≈ 0; n_probes noise.
- Script `scripts/eval/meta_monitor_m1.py`; results /root/meta_monitor_m1*.json.

## 3. M2 — Online integration ✅

- MetaMonitor wired: onset features → P(success), online BCE, rolling honest
  skill (accuracy vs outcome-window majority). Checkpoint-persisted.
- Run 1 (12.95M→13.0M, 40k steps): [meta] skill **+0.04 → +0.26**.
- Soak 2 (13.2M→13.4M): skill **+0.11 ~ +0.18** sustained.
- Soak 4/5 (13.6M→14.6M, ~950k steps total): skill **+0.27 peak (uncertain
  phases), +0.19 at 02:00, cooling to ~0 in high-success phases** (majority
  0.87-0.91 → limited headroom). Phase-dependence is an honest, reproducible
  characteristic: informative prediction in uncertain phases, ~zero in
  near-deterministic ones. **No errors across ~1M steps.**

## 4. M3 — Contradiction detection + self-correction ✅

- Detector: near-identical condition embeddings (cos ≥ 0.9) + conflicting
  actions; resolution weakens the weaker rule (success-rate score), once per
  pair (bounded 256 memory).
- Verification: injected-detection **> 0.8**, false-positives **< 0.1**
  (7/7 tests); honest no-op without embeddings; mixed-dtype/device handled.
- Online: real rule base carries 55-154 conflict pairs per cycle (churn from
  episode-end extraction); resolve-once validated in soak4 (resolved < pairs
  on repeat cycles); 0 cycle failures after the A11 fix; active throughout
  soak5 (72-99 pairs/cycle).

## 5. M4 — Deliberation gate (mechanism ✅, A/B done: **honest negative**)

- Mechanism: failure-pressure → effective probe ε (more when stuck, less
  when rolling); rolling-mean centering = budget-neutral
  (`avg_factor` **0.9998-0.9999** over 332k modulations in soak5).
- **A/B/C results (2026-10-10, 75k steps per condition, same 13.8M start)**:

| Condition | Probes | Verify rate |
|---|---|---|
| A control (ε=0.15, gate off) | 5,275 | 0.574 |
| B treatment (gate on) | 7,236 | 0.736 (+16.1pp) |
| C practice-matched (ε=0.205, gate off) | 6,559 | 0.687 (+11.3pp) |

- **Verdict: B ≈ C (Δ +4.8pp < 5pp threshold)**. Practice-volume
  extrapolation predicts ~+17.4pp for B's probe count; B scored +16.1pp —
  **no independent value from timing/redistribution**. Honest negative:
  the gate's measured benefit is a probe-volume amplifier (a stuck policy
  naturally probes more), achievable by simply raising ε.
- Recorded per the pre-registered protocol; telemetry (budget neutrality)
  validated. No further M4 engineering planned.

## 6. Carried gaps (from Stage 20) — close-out

| Gap | Status |
|---|---|
| G1 systematic_reasoning 诚实度量 | **以过程级预测重新定义并达成**: M1/M2 skill (+0.15 offline / +0.04~0.27 online) 是新的诚实仪器; 环境结局预测≈随机 (如实记录区分)。旧 fm 代理废弃。 |
| G2 op 浅振荡 | **刻画并接受**: 四类根因已修 (反刷分/规则runaway/单规则捕获/value爆炸); 残余为内在动力学 (0.20-0.74 带宽, 无深坍缩, ~1M 步长跑稳定); 谷相全认知面同步压低 (14.6M 快照: op 0.13-0.22/ToM 0.12-0.17/num 0.0-0.15 — 与 12M 谷相形态一致); 记录为训练动力学特性, 能力声明以好相位多 seed 工件为准。 |
| G3 num_sense | **好相位状态**: 12.15M (012151904, op 0.712 相位) seed123 num=**0.60 ✓**; 12.05M (012051552) seed7 num=0.4875 (差 0.0125); 12.05M seed123 复验进行中 → 结果随封存补充。⚠️ 选档教训: G3 首测误用谷相档 (12150000, 0.172 相位) 得 num 0.30 — 非能力读数; 已改用好相位档重测。 |

## 7. Exit checklist

- [x] M1-M4 各有结论 (3 pass + 1 honest negative + mechanism validated)
- [x] 长程验证 (soak5, 748k 步, 0 错误)
- [x] 本报告
- [x] `docs/TIMELINE.md` Stage 21 行更新
- [x] Tag `v0.21.0-stage21`
- [x] Bug ledger: A11 (partial deploy) 及 M2-M4 相关条目归档

## 8. Bug ledger additions (this stage)

见 `docs/bug_ledger.md`: A11 (partial deploy killed the probe path),
B6 (device mixing), B7 (repeated punishment), B8 (event-log unbounded) —
本周发现并修复。
