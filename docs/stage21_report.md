# Stage 21 Report — 递归元认知 (Recursive Metacognition) — DRAFT

> **Status**: **IN PROGRESS** (draft 2026-10-09). M1 ✅ M2 ✅ M3 ✅
> (online validation pending full-stack run), M4 mechanism ✅ /
> online A/B pending (protocol §5). 状态：进行中，草稿。

---

## 1. Run Card

| Field | Value |
|---|---|
| Lineage | Stage 20 sealed @ 12.95M → Stage 21 modules layered on top |
| Lineage now | 13,400,000 (soak2 end) |
| Modules | MetaMonitor (M2), ContradictionDetector (M3), DeliberationGate (M4), bounded event log |
| Config | `stage20_hypothesis_v2.yaml` (+ `meta_monitor_enabled`, `contradiction_enabled`, `deliberation_enabled`) |
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
  skill (accuracy vs outcome-window majority).
- Run 1 (12.95M→13.0M, 40k steps): [meta] skill **+0.04 → +0.26**.
- Soak 2 (13.2M→13.4M, 200k steps): skill **+0.11 ~ +0.18** sustained
  (acc 0.84-0.85, majority 0.67-0.73). No errors; op oscillated normally.

## 4. M3 — Contradiction detection + self-correction ✅

- Detector: near-identical condition embeddings (cos ≥ 0.9) + conflicting
  actions; resolution weakens the weaker rule (success-rate score), once per
  pair (bounded memory).
- Verification: injected-detection **> 0.8**, false-positives **< 0.1**
  (7/7 tests); honest no-op without embeddings; mixed-dtype handled.
- Online: real rule base contains ~110-150 conflict pairs per cycle
  (churn from episode-end extraction); soak2 ran 0 failures after the
  device fix; resolve-once activates next full-stack run.

## 5. M4 — Deliberation gate (mechanism ✅, A/B done: **honest negative**)

- Mechanism: failure-pressure → effective probe ε (more when stuck, less
  when rolling); rolling-mean centering = budget-neutral
  (`avg_factor` 1.006 online ✓).
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

## 6. Carried gaps (from Stage 20) — status

| Gap | Status at draft time |
|---|---|
| G1 systematic_reasoning 诚实度量 | **重新定义为过程级预测**: M1/M2 skill = 新诚实仪器 (+0.15 offline / +0.11~+0.26 online)。旧 fm 代理彻底废弃。是否算"阶段达标"取决于 M2 长期稳定性 + M4 收益 — 待 A/B。 |
| G2 op 浅振荡 | 持续 (0.25-0.74 带, 无深坍缩)。延迟到 M4 后: 专项诊断或显式接受。 |
| G3 num_sense 单 seed 差 0.0125 | 未动; A/B 窗口顺带复验。 |

## 7. Known issues / 待办

- v2 env-probe skill 在谷相位噪声极大 (-0.85..+0.08) — 已定性为环境预测
  噪声, 不作为阶段判据 (过程级 meta 才是)。
- resolve-once 与 M4 的在线行为都需下一次全栈运行确认。
- Stage 21 exit checklist: 报告定稿 + `v0.21.0-stage21` tag + A/B 结论。

## 8. Bug ledger additions (this stage)

见 `docs/bug_ledger.md`: B6 (device mixing), B7 (repeated punishment),
B8 (event-log unbounded growth) — 均本周发现并修复。
