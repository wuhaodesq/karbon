# Roadmap / 里程碑

Six sequential stages. Each stage has a hard exit criterion that must pass before proceeding.
六个顺序 Stage，每个 Stage 必须通过硬性 exit 标准才能进入下一个。

---

## Stage 0 · Skeleton + PPO Baseline + Longevity Harness

**Duration**: 1–2 weeks
**Hardware**: local laptop CPU (Phase 1)
**Deliverables**:
- Full project skeleton (all documents, all directories)
- Platform abstraction layer (`src/platform/`)
- Memory watcher + longevity harness (`src/monitoring/`)
- MiniGrid wrapper (`src/envs/minigrid_wrapper.py`)
- Minimal PPO baseline (`src/train.py`)
- `configs/_presets/*.yaml` (three-tier preset system)
- Unit tests for bounded primitives
- 5-minute local smoke test passes

**Exit criterion**:
- Smoke test finishes in <5 min on laptop CPU.
- `pytest` all green.
- After later cloud run: 24h VRAM drift ≤ 0.2 GB with `cloud_24g` preset.

**Git tags**: `v0.0.0-stage0-local` (local skeleton) → `v0.0.0-stage0-cloud` (cloud 24h passed).

---

## Stage 1 · RND Curiosity + Bounded Replay (3-tier)

**Duration**: 2 weeks
**Hardware**: cloud 24G (Phase 2 starts here)
**Deliverables**:
- `src/intrinsic/rnd.py` — target/predictor networks
- `src/memory/bounded_replay.py` — GPU ring / CPU ring / SSD archive, prioritized sampling
- Intrinsic reward wiring into PPO loop
- Coverage metric (state-visitation entropy)

**Exit criterion**:
- State coverage ≥ 2× the Stage-0 baseline on the same MiniGrid tasks.
- 24h longevity: VRAM slope <0.2 GB/day; replay respects its capacity strictly.

**Git tag**: `v0.1.0-stage1`.

---

## Stage 2 · TTT-Hybrid Backbone (Hot-Swap Triton)

**Duration**: 3–4 weeks
**Hardware**: cloud 24G
**Sub-stages**:
- **2a** (weeks 1–2): pure-PyTorch teaching implementation
  - `src/models/ttt_linear.py`
  - `src/models/ttt_mlp.py`
  - `src/models/sliding_attn.py`
  - `src/models/hybrid_backbone.py`
  - `src/models/ttt_backend.py` (backend protocol)
- **2b** (weeks 3–4): Triton fused kernels (dual-form)
  - `src/models/ttt_linear_triton.py`
  - `tests/test_ttt_backend_parity.py` — parity ≤1e-4
  - Config switch: `model.ttt_backend: "pytorch" | "triton"`

**Exit criterion**:
- On long-context task (seq_len ≥ 512), Hybrid outperforms same-parameter GRU baseline in loss.
- Triton parity vs PyTorch ≤1e-4 across randomized inputs.
- Fallback: if Triton import fails, auto-degrade to PyTorch with warning (no crash).

**Git tags**: `v0.2.0-stage2-pytorch` → `v0.2.1-stage2-triton`.

---

## Stage 3 · Dreamer-Style World Model

**Duration**: 3–4 weeks
**Hardware**: cloud 24G
**Deliverables**:
- `src/models/world_model.py` — RSSM (recurrent state-space model) using Hybrid backbone
- Truncated BPTT with configurable rollout length (default 15)
- Gradient checkpointing on the imagination unroll
- Switch from MiniGrid to Crafter env (`src/envs/crafter_wrapper.py`)

**Exit criterion**:
- On Crafter, sample efficiency (reward at N env-steps) ≥ 3× the model-free Stage-1 baseline.
- 24h longevity holds under `cloud_24g` preset (rollout cache is bounded).

**Git tag**: `v0.3.0-stage3`.

---

## Stage 4 · Bounded Skill Library

**Duration**: 4–6 weeks
**Hardware**: cloud 24G
**Deliverables**:
- `src/memory/skill_library.py`
- LoRA-style low-rank skill representation (rank ~8)
- Top-K GPU residency (K=32) with CPU cache (256) and SSD archive
- LRU × usefulness × avg-reward composite eviction
- Similarity-based merging (cosine >0.9 fuses)

**Exit criterion**:
- Observable **skill reuse** across tasks (a skill learned in task A is invoked ≥1× in task B).
- Skill count on GPU strictly ≤ K under long runs.

> **STATUS (project review):** GPU-bound criterion **PASS** (verified: GPU tier = 256 ≤ K under 2M-step run, `ckpt_stage4_002000000.pt`). Skill-reuse criterion **NOT MET** — training currently only *adds* skills to the library and never *retrieves/re-applies* them, so every `usage_count == 1`. Maps to milestone **M2** in `PLAN.md §1.5`; top priority before Stage 5. Do not tag `v0.4.0-stage4` until the reuse loop is closed.

**Git tag**: `v0.4.0-stage4`.

---

## Stage 5 · Auto Curriculum (Learning Progress)

**Duration**: 4–6 weeks
**Hardware**: **transition to home-64G rig (Phase 3)** if available; otherwise cloud
**Deliverables**:
- `src/intrinsic/learning_progress.py` — sliding-window LP metric
- `src/curriculum/auto_curriculum.py` — task pool + LP-driven sampling
- Fixed task-template capacity (~100 templates)

**Exit criterion**:
- Autonomous difficulty ramp: the agent picks harder tasks *without* human intervention as easy ones plateau.
- Trajectory of task-difficulty-vs-time shows monotonic-ish upward trend on average.

**Git tag**: `v0.5.0-stage5`.

---

## Stage 6 · Perpetual (Online EWC + Generative Replay + Sleep)

**Duration**: long-term (months)
**Hardware**: home-64G rig (Phase 3)
**Environment**: **continues on PhysicsSandbox** (same 64×64 obs / 8-action as
Stage 5) so the developmental chain (M1–M5) carries forward without an
architecture break. Resumes from the Stage 5 ckpt (7-layer hybrid + SlotAttention
+ RSSM + bounded skill library). Do **NOT** switch the base env to MiniGrid here
— see the post-Stage-6 cognitive branch below.
**Deliverables**:
- `src/continual/online_ewc.py` — single accumulating Fisher, weighted decay
- `src/memory/generative_replay.py` — small VAE substituting for stored history
- `src/continual/consolidation.py` — sleep loop:
  - every N steps: distill TTT slow-W into fixed weights
  - evict low-value replay entries
  - merge/prune skill library
- Health daemon (`scripts/home/health_daemon.sh`)

**North-Star scaling strategy (A+B, see `docs/path-to-northstar.md §1.6`)**:
The route to 8–15y physical intuition without violating Axiom 1 (bounded).
Both pieces are part of the Stage 6 training plan, not optional extras:
- **B · Dreamer-style imagination training** — `imagination.enabled: true` is
  already set in `configs/stage6_consolidation.yaml`; the `ImaginationTrainer`
  (train.py:2630) trains the actor-critic on RSSM-imagined rollouts, gradient
  flowing back to the main model. *Use "think more" to substitute for "more
  params"* — the core Dreamer insight.
- **A · bounded hierarchical external memory** — generalize the Stage-4 LoRA
  skill library into a general retrieval-injected memory (GPU-hot / CPU-warm /
  SSD-cold tiers, all capacity-declared). Amplifies *effective* intelligence
  volume without growing the single-model param count. Deepen this in Stage 6
  alongside B (sketch: promote `skill_library.py` → general cognitive memory).
- **Rejected fallbacks** (kept for record): C = bounded MoE (arch change),
  D = relax Axiom-1 scale limit (needs explicit user approval to amend the
  iron rules). Not used unless A+B is empirically insufficient.

**Exit criterion (the "AGI-esque" bar)**:
- **30 consecutive days** of uninterrupted training with no manual restarts.
- Agent learns **≥10 distinct tasks** in sequence with ≤10% drop on the earliest tasks.
- Memory footprint asymptotes (VRAM slope ≈ 0 over 7-day windows).

**Git tag**: `v1.0.0-stage6`.

---

## Post-Stage-6 Cognitive Branch (MiniGrid → 3D → language)

After the developmental backbone (M1–M5) is closed on PhysicsSandbox, a
**separate branch** validates the *upper* cognitive abilities the North Star
(8–15-year-old intelligence) requires but PhysicsSandbox alone cannot:

- **MiniGrid** (first step): instruction-following + sparse-reward multi-step
  planning (DoorKey etc.). Spins off from the mature Stage-6 ckpt as its own
  training line. NOTE: MiniGrid obs is a discrete grid encoding, **incompatible**
  with the 64×64 vision encoder — the branch needs its own grid/observation
  adapter (or `use_vision_encoder` reconfigured); it does **not** reuse the
  Stage-5 CNN encoder weights. The hybrid backbone + skill library still transfer.
- **3D world + real language** (Crafter / `three_d_world` + `social_teacher`):
  the next rung toward juvenile-level grounded language and longer-horizon
  planning. Already scaffolded in `src/envs/`; paused.

This branch is **complementary, not on the critical path**: the PhysicsSandbox
main line is the developmental foundation; the cognitive branch is the upper
room built on top of it. Starting the branch before Stage 6 closes would waste
the accumulated developmental state.

### LLMFusion timing / LLM 融合激活时机

`src/models/llm_fusion.py` is a **local, offline, frozen Qwen-7B** (4-bit,
~5 GB, zero network dependency; gracefully degrades to template mode if the
weights are absent). It is a *self-contained* capability, **not** an external
API — confirmed by reading the code.

Decision: **keep LLMFusion deactivated during Stage 5–6 main-line training,
and defer it to the late B-plan "language emergence" phase.** Rationale:
- Current bottleneck is *sensorimotor* development, not language; earlier/higher
  LLM involvement (smaller `call_interval_steps`, larger model) only slows
  training (Qwen forward is the speed bottleneck) and destabilizes the
  still-forming policy via FiLM modulation of noise.
- Its trainable projectors can only learn a useful modulation once the base
  policy + slot representations are mature.
- Config stays at defaults (`call_interval_steps: 50`, `max_new_tokens: 64`,
  Qwen-7B). Do **not** raise frequency/model size during the main line.
- Activate only after the 5 lightweight cognitive modules (HomeostaticDrives /
  Metacognition / LongRangePlanner / CausalDiscovery / CreativityOrchestrator)
  are wired into the loss loop and the backbone is stable.

---

## Cross-Cutting Deliverables (every stage)

- Git tag as listed.
- `docs/stage{N}_report.md` — bilingual technical report.
- `docs/figures/stage{N}_memory.png` — 24h memory drift chart.
- Config snapshot committed under `configs/`.
- Reproducibility: same tag + same seed + same preset → same headline number ±3%.

---

## Downgrade Ladder (requires explicit user approval)

If a stage stalls:

1. **Try harder** — allocate 2× the planned duration.
2. **Shrink scale** — halve model size, keep the phenomenon.
3. **Simplify env** — MiniGrid instead of Crafter, or Crafter-1 instead of Crafter-N.
4. **Relax perpetual target** — 7-day instead of 30-day for Stage 6.
5. **Bail on the stage** — document what was learned, freeze feature-flagged, proceed.
