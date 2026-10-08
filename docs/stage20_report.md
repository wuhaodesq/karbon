# Stage 20 Report - 假设-演绎引擎 (Hypothesis-Deduction Engine) - Seal

> **Status**: **SEALED** (2026-10-08) — mechanism complete & verified;
> capability gates op/ToM achieved in the physics world across seeds;
> **systematic_reasoning carried to Stage 21** (honest metric ≈ 0 after
> two consolidation attempts; the legacy milestone proxy is invalid).
>
> **状态**：**封存**（2026-10-08）——机制闭环完成并验证；能力门
> op/ToM 在物理世界跨 seed 达成；**systematic_reasoning 如实携带至
> Stage 21**（诚实度量两轮巩固后 ≈ 0；旧里程碑代理指标无效）。

---

## 1. Run Card

| Field | Value |
|---|---|
| Stage config | `stage20_hypothesis_v2.yaml` (frozen rules + anti-farm rewards) |
| Lineage | 1.8M → 12,950,000 (macro Stage 20; physics world from ~11.2M) |
| Env | ThreeDWorld: 8 objects **free joints (movable)**, 128px, 4 walls, crossing@100 steps |
| Reward (final) | focus_op_only + reach=100 (full, un-truncated) + anti-farm gate (start_d ≥ 1.2m) |
| Rules | **steering frozen** (`symbolic_bias_weight=0`, `logic_bonus_weight=0`); recorded for measurement only |
| Facts | kanren/symbol backend **512/512** (ingestion wired 2026-09-25; was 0 since inception) |
| Hypothesis loop | propose → probe (learned gate + ε warmup) → verify; 100k+ outcomes logged |
| Aux (systematic push) | event-outcome BCE → trunk, coef 0.05 (500k+ steps; **no skill gain**) |
| Value stability | ReturnNormalizer α=0.1 + Huber value loss (fixed 1.3e3-explosion collapse) |

---

## 2. Option-C instrument suite (honest gates)

| Instrument | Result |
|---|---|
| `formal_reasoning_probe.py` (G1-G4) | **10/10 scorable, 0 gaps** (multi-hop chaining implemented 9/25) |
| `systematic_probe_v2.py` | probe_prediction skill_vs_majority ≈ 0; rule_quality 0.20; fm_alignment 0.5-0.9 (substrate sanity) |
| `op_failure_diag.py --ckpts` | per-ckpt strict op curve (all re-verifications below) |
| majority-baseline correction | added 9/28 after the 0.812-on-5.8%-positives illusion |

## 3. Capability gates (physics world, multi-seed, 20 eps/task)

| Milestone | 12.05M / seed 7 | 12.15M / seed 123 | Gate |
|---|---|---|---|
| **object_permanence** | **0.692** ✅ | **0.673** ✅ | 0.6 |
| cross-probe op | 0.727 | 0.707 | — |
| **theory_of_mind** | **0.566** ✅ | **0.552** ✅ | 0.55 |
| means_ends | 1.000 ✅ | 1.000 ✅ | — |
| intuitive_physics | 0.898 ✅ | 0.771 ✅ | — |
| number_sense | 0.4875 ⚠️ | 0.600 ✅ | ~0.5 |
| systematic_reasoning (proxy) | 0.740 (invalid, see §5) | 0.744 (invalid) | 0.6 |

Supporting evidence: strict-op diag phases reached 0.71-0.81 at 10.95-11.15M,
12.45M (0.788), freeze-era 12.75-12.80M (0.61).

## 4. What was delivered (mechanism)

1. **Closed loops, previously dead wires all fixed** (the audit's 9 silent
   breaks): reflection device/dtype + finally-clear; evaluator sym keys;
   logic engine projection + quantifier; kanren fake metrics retired;
   narrative difficulty field; knowledge-ledger; openness event typing;
   probe_net durable loop; **fact ingestion (`get_edges` never existed)**;
   **rule outcome feedback (update() had no caller)**; multi-hop chaining.
2. **Physics world (Stage 20's core direction)**: free-joint movable
   objects + velocity clamp/NaN recovery; anti-farm reward gate; full reach
   payout (clamp bug fixed).
3. **Stability fixes over four collapses** (all root-caused):
   - contact-reward farming → `focus_op_only` + anti-farm gate
   - value-normalizer lag → α 0.1 + Huber (v-loss 1.3e3 → ~0.05)
   - rule confidence runaway → relative outcome criterion
   - single-rule capture (#336805/#332466) → **both steering channels frozen**
   Post-freeze: no deep collapse (band 0.36-0.62 shallow oscillation remains).

## 5. systematic_reasoning — NOT achieved (honest record)

- Legacy metric (`entropy*0.35 + fm_consistency*0.3 + rule*0.35`) is invalid:
  `fm_consistency` = push-direction concentration, not force→motion
  prediction; the 0.74 readings are artifacts.
- Honest metric (`systematic_probe_v2`): probe_prediction
  skill_vs_majority ≈ 0 (n≈800/ckpt); rule_quality 0.20.
- Consolidation attempts (both failed):
  (a) detached-feature probe_net (BCE, online) — chance;
  (b) trunk-backprop outcome BCE (aux coef 0.05, 500k+ steps) — no skill.
- Conclusion: needs a new mechanism (metric redesign + architecture), not
  more training. **Carried to Stage 21 as first task.**

## 6. Artifacts

| Ckpt | Note |
|---|---|
| `backup_stage20_10751904.pt` | pre-physics best (op 0.70-0.75 multi-seed) |
| `backup_stage20_012051552.pt` | **physics multi-seed verified** (op 0.692/0.727, ToM 0.566) |
| `backup_stage20_12150000.pt` | **physics multi-seed verified** (0.673/0.707, ToM 0.552) |
| `backup_stage20_12451200.pt` | best phase (op 0.788) |
| `freeze_ckpt_stage20_0126{00704,51904},012700000.pt` | freeze-era stable band |
| `ckpt_stage20_012950000.pt` | line end (12.95M) |

All under `/root/autodl-tmp/karbon_ckpts/` (data disk, persists).

## 7. Seal checklist (AGENTS §11)

- [x] Tests / bounds / CHANGELOG updated (continuous through the stage)
- [x] This report
- [x] `docs/TIMELINE.md` Stage 20 row → ✅ sealed (honest notes)
- [x] Tag `v0.20.0-stage20`
- [x] Key checkpoints backed up on the data disk
- [ ] Longevity test — replaced by the recurring-collapse analysis (§4.3); N/A for this exit

## 8. Hand-off to Stage 21 (递归元认知, 13-14y)

Priority list (details in `docs/stage21_design.md`):
1. **systematic_reasoning redesign** — honest metric + new mechanism
   (explicit event-level prediction supervision; prediction-gated rewards).
2. Oscillation mechanism (shallow residual band) — diagnosis vs. acceptance.
3. num_sense one-seed gap (0.4875 vs 0.5) — cheap re-verify.
4. Recursive metacognition architecture (the stage's own deliverable).
