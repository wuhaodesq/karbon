# Changelog

All notable changes to this project are documented here.
本项目所有值得记录的变更。

## [Unreleased]

### Added (Stage-2 N-env vectorization — throughput)

- **N parallel 3D homes via `src/envs/vec_three_d_world.py`** (`VecEnv` generic
  serial wrapper + `VecThreeDWorld`, each sub-env seeded `base+ i`). One
  `env.step` returns a batched `VecStep` (obs `(N,H,W,C)`, reward/terminated/
  truncated `(N,)`); auto-resets each sub-env on its own `done`.
- **Batched actor-critic rollout.** The single `model(obs_t)` forward now takes
  `(N,3,H,W)` and returns `(N, A)` logits / `(N,)` value; `dist` is sampled
  for all envs at once. World-model curiosity + intention + count-based
  `expl_bonus` are batched too, so the GPU forward cost grows sub-linearly
  with `N` while env-steps/iter rise ~`N×`.
- **`RolloutBuffer` now stores `(T, N, *obs)`** and `as_batch()` flattens to
  `(T*N, *obs)`. `compute_gae_vec(rewards,values,dones,last_values,...)`
  computes GAE independently per env column in pure-tensor form (no `.item()`
  syncs) and is flattened for the PPO update.
- `phase1_infant_home.yaml` now sets `env.num_envs: 8`.
- `tests/test_vec_env.py` (CPU, no mujoco) covers VecEnv shapes/auto-reset,
  `(T,N)` buffer layout + flatten, and per-column parity of `compute_gae_vec`
  vs the scalar `compute_gae`.

### Known limitation (first cut)

- Single-env-only cognitive blocks (homeostatic drives, emotion, number-sense /
  rule predicates, knowledge-gap, concept-graph, memory, creativity, LLM fusion,
  RND, skills/symbolic/reflection episode hooks, causal intervention, cross-modal
  bridge) are guarded with `n_envs == 1` and **skipped** when `N>1`. The
  batched hot path (actor-critic + WM curiosity + intention + expl-bonus + replay
  + PPO + imagination + growth) still runs under `N=8`. Set `env.num_envs: 1`
  to recover full single-env module coverage.

### Fixed (Marginal Gains integration — name collision & dead code)

- **Removed duplicate `KnowledgeGapDetector` from `src/models/marginal_gains.py`.**
  It collided by name with the existing, fully-wired
  `src.intrinsic.knowledge_gap.KnowledgeGapDetector`. The `marginal_gains`
  import (`src/train.py`) shadowed the intrinsic one, so `knowledge_gap` was
  constructed as the wrong class and its `get_gap_boost()` / `update()` calls
  silently failed (swallowed by `try/except`) — a regression of the working
  knowledge-gap curiosity boost. The marginal-gains copy was also dead code: an
  unconditional `knowledge_gap = None` reassignment clobbered it right after
  creation, so its `.detect()` was never called.
- `marginal_gains` now exposes only the genuinely-new, non-overlapping modules:
  **`CompositionalTester`** (compositional generalization test over ConceptGraph)
  and **`LearningProgressTracker`** (plateau detection → curiosity boost), both
  already correctly wired in `src/train.py`.
- Added `tests/test_marginal_gains.py` (7 tests, passing).

### Fixed (Env episode-return metric was an all-history mean)

- `mean_ret` / `summary()["mean_return"]` in `physics_sandbox`,
  `three_d_world`, `social_teacher`, and `minigrid_wrapper` were computed over an
  **unbounded** `_episode_returns` list — i.e. the mean over *every* episode
  since process start. Over multi-million-step runs the value becomes frozen and
  cannot reflect recent agent performance (the 2D `mean_ret=121→114` "drop" and
  the 3D `mean_ret=0.218` were both artifacts of this).
- Added the 1024-episode rolling-window cap (already present in
  `crafter_wrapper.py`) to all four envs. The list now keeps at most the last
  1024 returns/lengths, so `mean_ret` reflects **recent** performance. This also
  resolves an Axiom-1 unbounded-allocation finding in `scripts/ci/check_bounded.py`
  (lines annotated `BOUNDS-OK`).
- `train.py` log line still prints `mean_ret`; it now reads the bounded window,
  giving a truthful trend of recent episodes.

### Fixed (PPO scale-mismatch — 3D "flat loss" root cause)

- The `ReturnNormalizer` docstring promised: *"Denormalizes predicted values
  before GAE advantage computation"* — but the code did not do it. The value
  head is trained on `returns_norm` (normalized scale), so `batch.values` and
  `last_value_t` are also in normalized scale. Feeding those directly into
  `compute_gae` together with **raw** `batch.rewards` produced
  scale-mismatched advantages that could not carry a policy signal — the real
  driver of the 3D "flat loss / near-zero policy gradient" symptom.
- Fix: denormalize `batch.values` and `last_value_t` back to raw scale before
  `compute_gae` (`train.py:2117-2131`). The off-policy replay TD update
  (`train.py:2187-2201`) had the same bug — the TD target is now built in
  raw scale (`next_v` denormalized, raw reward added), then renormalized so
  both sides of the MSE match.
- Extracted advantage normalization into `_normalize_advantages()`
  (`train.py:481-501`): standardizes to ~N(0,1) with a zero-variance guard
  that falls back to raw centered advantages (handles constant-advantage 3D
  case) and also handles NaN std from single-element batches (torch semantics).
- Added `tests/test_ppo_normalization.py` (11 tests, passing) covering
  `ReturnNormalizer` round-trip + EMA behavior, the zero-variance guard, and
  an end-to-end GAE scale-consistency test that demonstrates the fix
  (denormalize-then-GAE recovers the raw-scale advantages).
- Added `tests/test_ppo_integration.py` (3 tests, passing) — proves the fix
  at the *whole-PPO-step* level: with values denormalized before GAE, one PPO
  gradient step actually moves the policy (`approx_kl != 0`, finite grads), and
  higher-reward steps get higher advantages; plus a regression guard asserting
  the OLD bug (normalized value-head output fed straight into GAE) distorts
   advantages so the denormalize step can never be silently removed.

### Fixed (Actor-Critic encoder — GB-scale Linear / 3D memory swing)

- `ActorCritic` (`train.py:123-135`) and both `HybridActorCritic` CNN
  encoder branches (`train.py:218-239`) flattened the **full** H×W
  feature map and fed it to `nn.Linear(32*h*w, …)` with no
  downsampling. At the 3D obs size (256×256) that is
  `Linear(2_097_152, …)` ≈ **0.5–1 GB of weights + an equally
  large gradient** allocated on every update — the source of the
  `~2.57 GB` per-step memory swing seen in the dead 3D run.
- Fix: insert `nn.AdaptiveAvgPool2d((8, 8))` before `Flatten()`
  and size the Linear at the fixed `32*8*8 = 2048` features
  (mirrors the already-fixed `RNDNet`, `src/intrinsic/rnd.py:98-102`).
  The 3D per-step allocation drops from ~1 GB to a few MB.
- Added `tests/test_actor_critic_encoder.py` (5 tests, passing) asserting
  the trunk's first Linear takes `32*8*8` (not `32*h*w`), the encoder
  has a downsample layer, and 256×256 / 64×64 inputs forward with
  correct `(B, n_actions)` / `(B,)` shapes.

### Fixed (Model growth was non-learning / wiped optimizer state)

- `ModelGrowerV2.grow()` claimed to preserve Adam momentum for
  matching parameters, but the copy guard `new_idx < len(new_state["state"])`
  is **always false for a fresh optimizer** (empty `state` dict),
  so the carry was a silent **no-op** — every growth wiped the
  policy optimizer's accumulated momentum, re-setting learning. Real fix
  in `_carry_over_adam_momentum()` (`src/models/model_growth_v2.py`):
  create the entry on the fresh optimizer and copy `exp_avg` / `exp_avg_sq`
  / `step` from the old one. (Caught by a unit test.)
- Distillation now also preserves the teacher's **policy distribution**
  via a KL term on the softmax logits (`model_growth_v2.py:_distill`),
  so a grown model keeps the agent's learned action preferences
  instead of resetting them; `distill_steps` raised 100 → 256.
- Growth frequency made rarer so it stops disrupting the policy:
  `min_steps_between_growths` 100_000 → 500_000 in
  `phase0_protozoan.yaml`, `phase1_infant_home.yaml`,
  `phase2_infant_exploration.yaml`, `phase9_llm_fusion.yaml`;
  `grow_trigger_coverage` 0.3 → 0.5 in phase0/phase1 (growth
  now requires broad exploration first).
- Added `tests/test_model_growth_v2.py` (6 tests, passing): KL distill
  moves the student's policy toward the teacher's, and the momentum
  carry-over helper actually transfers Adam state to a fresh optimizer.

### Added (Phase 0+: 工程补缺口 A-G)

- **Imagination Trainer** (`src/training/imagination_trainer.py`) — Dreamer-style
  world-model-driven imagination training. Uses RSSM to generate N-step imagined
  trajectories, then trains actor-critic on imagined data for ~10x sample
  efficiency (DEV_PLAN.md gap D).
- **Intention Achievement Curiosity** (`src/intrinsic/intention_curiosity.py`) —
  replaces blunt RND with action-conditioned curiosity: compares RSSM prior
  (predicted next state) vs posterior (actual state) to reward states where the
  agent's own model of cause-and-effect fails (DEV_PLAN.md gap C).
- **Knowledge Gap Detector** (`src/intrinsic/knowledge_gap.py`) — tracks per-slot
  (per-concept) prediction accuracy via EMA. Concepts with sustained high error
  are identified as knowledge gaps; curiosity is boosted for gap-related states
  (DEV_PLAN.md gap C).
- **Social Curiosity** (`src/intrinsic/social_curiosity.py`) — predicts caregiver
  next action; prediction error = social curiosity reward. For Phase 3+
  imitation/social learning (DEV_PLAN.md gap C).
- **Audio Encoder** (`src/sensory/audio_encoder.py`) — lightweight mel-spectrogram
  CNN encoder (~0.1 GB). Optional torchaudio dependency with NumPy fallback
  (DEV_PLAN.md gap B).
- **Episodic memory with surprise gating** preserved as `src/models/developmental_memory.py`
  (already implemented — gap G covered by existing MemoryManager).
- All new modules wired into `src/train.py` training loop with config-driven
  enable/disable, checkpointing, and logging.
- New config sections in `configs/phase0_protozoan.yaml`: `imagination`,
  `intention`, `knowledge_gap`, `social_curiosity`, `audio`.

### Changed (Counterfactual Planner — reward proxy fix)

- **RSSM reward head** (`src/models/world_model.py`) — added a bounded
  `reward_head` (predicts `r̂_t = Reward(h_t, z_t)`) and `predict_reward()`
  method. `compute_loss` now accepts an optional `reward_seq` and adds an MSE
  reward-prediction term, trained from replay `reward` in the world-model
  update step (`src/train.py`). This grounds counterfactual planning in
  **objective environment reward** instead of the policy's value estimate
  (Dreamer-style), so System 2 can surface plans the current policy would not
  choose.
- **CounterfactualPlanner** (`src/models/counterfactual_planner.py`) — now
  scores plans via `wm.predict_reward` (removed the broken `policy_model.
  value_head(decoded)` path that mismatched latent/obs dimensions and silently
  no-op'd). Also fixed two correctness bugs: prediction/actual history is now a
  bounded `deque` (Axiom 1), and `planning_accuracy` pairs predictions with
  actuals correctly instead of misaligning via `pop(0)`.
- **Reward-head train/serve gap narrowed** — `compute_loss` now also predicts
  reward from the **prior (imagined) state** (`imagine_step`) in addition to the
  posterior state, since at planning time rewards are scored from prior states.
  This reduces the posterior-vs-prior distribution gap (Dreamer-style).
- **`planning_accuracy` is now apples-to-apples** — `evaluate_plan` returns the
  predicted **first-step** reward (the action actually executed) alongside the
  total plan score; `select_best` records that for validation against the single
  observed reward, instead of comparing a multi-step summed total to a 1-step
  reward. Added `reward_loss_weight` config (`RSSMConfig` + `phase2` yaml) to
  balance the reward term.
- World-model training log now reports `rew=` (reward-loss) alongside recon/kl.

### Fixed (Model growth obs_shape + PPO loss log consistency)

- `ModelGrowerV2._create_larger_model()` hardcoded `obs_shape=(64,64,3)` when
  building the grown `HybridActorCritic`. For any env whose real obs is not
  64×64×3 (e.g. 4-channel / non-square), the grown encoder's first conv
  `in_channels` would be wrong and silently corrupt the expanded network. Fix:
  read the real shape from `model.obs_shape` (with `(64,64,3)` only as a
  fallback). `HybridActorCritic` now also stores `self.obs_shape` at
  construction (`train.py`), which additionally un-breaks
  `imagination_trainer`'s `actor_critic.obs_shape` access.
- PPO log line (`train.py:2551`) printed `loss=` from the **last** minibatch's
  combined loss while `p=/v=/ent=/kl=/cf=` are means over all minibatches —
  inconsistent. Now `ppo_losses["total"]` accumulates the combined loss every
  minibatch and the log reports its mean.
- Added `tests/test_model_growth_v2.py::TestGrowerV2ObsShapeCarryover` (1 test).

### Fixed (3D deadlock guard — state-dependent exploration bonus)

- **New `src/intrinsic/exploration_bonus.py` (`ExplorationBonus`)** — a
  bounded count-based exploration bonus that prevents the 3D training
  deadlock. When the env reward is sparse / near-constant, the value
  head eventually fits it → advantages collapse to ~0 → the policy
  gradient vanishes and the agent stops learning. RND-style intrinsic
  curiosity only patches this while its predictor error stays >0; once it
  converges on visited states the bonus decays to 0 and the deadlock
  returns.
- bonus(s) = `coef / sqrt(visit_count(s) + 1)`: highest for novel /
  rarely-visited states, decays toward 0 as a state is revisited but
  **never reaches 0**. Crucially it VARIES across states AND with
  visitation history, which the value head (sees only the current obs)
  cannot predict → it leaves a persistent residual in the advantages, so
  the policy always has an exploration signal. (A flat *constant* floor
  would be a no-op: advantages are invariant to adding a constant to
  every reward, because the value head fits the constant too.)
- Bounded (Axiom 1): visit counts live in a **fixed-capacity** tensor
  (`capacity` buckets, hashed from a downsampled obs); no unbounded
  growth. Exposes `capacity` + `__len__` for `HealthChecker`.
- Wired into `src/train.py`: `total_r += eb` each step (the bonus
  already carries its own `coef`, so it is added directly — **not**
  multiplied by `intrinsic_coef` again), plus `expl_bonus.update(obs_t)`.
  Decoupled as a **top-level** `exploration_bonus` config key (NOT under
  `intrinsic:`), so enabling it does **not** also switch on RND or
  change the `curiosity.mode` already in use. Enabled in
  `configs/stage1_curiosity.yaml` (the 3D/RND stage-1 run) and
  `configs/phase2_infant_exploration.yaml` (the 2D run):
  `coef=0.1`, `grid=8`, `capacity=65536`.
- Verified offline (CPU, tiny): constant env reward → advantage std
  = 0 (deadlock); constant reward + exploration bonus → advantage std
  > 0 (signal persists). Added `tests/test_exploration_bonus.py`
  (6 tests, passing), including the deadlock-vs-signal proof.

## [v1.1.0-stage7-cloud] - 2026-07-09

### Stage 7 COMPLETED - ALL 7 STAGES DONE

- **Wall time**: 29,274 s (8.1 h)
- **Steps**: 3,000,000
- **Final mean_return**: **0.955**
- **VRAM**: 2.82 GB, slope 0.0 ✅
- **Coverage**: **8.9% (363 buckets)**
- **Skills**: 10496/10496 (877,149 created)
- **EWC**: ✅ consolidated (Fisher L1=1265)
- **Sleep Loop**: replay_trim×292, skills_merge×146, ewc×29

### Key validation
1. **SelfModel** (metacognition) active every step.
2. **NeuralSymbolicLayer** + **LogicEngine** active (no rules extracted due to EWC interference).
3. **ReflectionLoop** fired every 10 episodes.
4. **InnerDialogue** generated natural-language reflections:
   - "I failed this episode with return 0.000." (during EWC interference)
   - "I succeeded this episode with return 0.955." (after recovery)
5. Agent recovered from EWC interference: 0.006 -> 0.955.

### Complete 7-stage journey
- Total wall time: ~58 hours
- Total steps: 21,000,000
- Final mean_ret: 0.955
- Peak VRAM: 2.82 GB / 32 GB (8.8%)
- Skills created total: ~2.5M across all stages
- Coverage growth: 0.1% -> 8.9% (89×)

See `docs/stage7_report.md` for full analysis.

## [v1.0.0-stage6-cloud] - 2026-07-08

### Stage 6 COMPLETED - ALL 6 STAGES DONE

- **Wall time**: 29,242 s (8.1 h)
- **Steps**: 3,000,000
- **Final mean_return**: **0.951** (recovered from EWC interference)
- **VRAM**: 2.81 GB, slope 0.0 ✅
- **Coverage**: **8.6% (353 buckets)** - 86× growth from Stage 0
- **Skills**: 10496/10496 (672,158 created)
- **EWC**: ✅ consolidated (7.26M params)
- **GR VAE**: gr=0.001 (converged)
- **Sleep Loop**: all 4 task types fired

### Complete training summary (Stage 0-6)
- Total wall time: ~50 hours across 6 stages
- Total steps: 18,000,000
- Final mean_ret: 0.951
- Peak VRAM: 2.81 GB / 32 GB (8.8%)
- Coverage growth: 0.1% -> 8.6% (86×)


Format loosely follows [Keep a Changelog](https://keepachangelog.com/).
遵循 Keep a Changelog 惯例。

---

## [v0.5.0-stage5-cloud] — 2026-07-07

### Stage 5 COMPLETED on cloud (AutoDL vGPU-32GB / RTX 4080)

- **Wall time**: 28,945 s (482 min, 8.0 h)
- **Total steps**: 3,000,000
- **Episodes**: 31,491
- **Final mean_return**: **0.937**
- **VRAM**: 2.71 GB, slope 0.0 ✅
- **Coverage**: **7.9% (324 buckets)** — 20× growth from 0.4%
- **Skills**: 10496/10496 (344,012 created, 333,516 merged/evicted)
- **Tasks mastered**: empty-5x5, empty-6x6, empty-8x8
- **Tasks explored**: doorkey-5x5, doorkey-6x6 (not solved)

### Key validation
1. **AutoCurriculum** with 5 tasks — multiple autonomous task switches.
2. **Explosive learning** on Empty-8x8 (new task): 0→0.906 in 27k steps.
3. **No permanent forgetting** — recovered from temporary regression.
4. **Coverage 20× growth** — each DoorKey exploration added new states.
5. **Skill library** stable at 10496/10496 with 344k total skill operations.

See `docs/stage5_report.md` for full analysis.

---

## [v0.4.0-stage4-cloud] — 2026-07-07

### Stage 4 COMPLETED on cloud (AutoDL vGPU-32GB / RTX 4080)

- **Wall time**: 50,939 s (849 min, 14.2 h)
- **Total steps**: 3,000,000
- **Episodes**: 331,211
- **Final mean_return**: **0.914** (V-shaped recovery from 0.600)
- **VRAM**: 2.71 GB, slope 0.0 ✅
- **Skills**: 10496/10496 (full, merge/eviction operational)
- **Skills created**: 317,150 (306,654 merged/evicted = 97% turnover)
- **Speed**: 32 step/s (3.5× slower due to per-episode skill extraction)

### Key validation
1. Skill library filled to capacity and stayed bounded (Axiom 1).
2. Merge (cosine > 0.9) + LRU eviction worked continuously.
3. V-shaped recovery: 0.955 → 0.600 → 0.914 (agent fully recovered).
4. All three tiers saturated: GPU(256) + CPU(2048) + SSD(64×128).

### Bug fixed during this run
- CNNEncoder wrapper changed state_dict keys → Stage 3 weights couldn't load →
  reverted to inline encoder (commit `a67d712`).

See `docs/stage4_report.md` for full analysis.

---

## [v0.3.0-stage3-cloud] — 2026-07-04

### Stage 3 COMPLETED on cloud (AutoDL vGPU-32GB / RTX 4080)

- **Wall time**: 26,536 s (442 min, 7.4 h)
- **Total steps**: 3,000,000
- **Episodes**: 542,262
- **Final mean_return**: **0.950** (stable, no Stage 2 regression)
- **VRAM**: 2.70 GB / 32 GB (8.4%)
- **VRAM slope**: 0.0 GB/h ✅
- **WM loss**: 1.0 (recon=4e-6, kl=1.0 at free_nats floor)
- **Model params**: ~7.4M (HybridActorCritic 7.26M + RSSM 186k)
- **Checkpoints**: 300

### Key validation
1. **RSSM world model** trains stably alongside PPO.
2. **No regression** — mean_ret 0.950 stable throughout (Stage 2 had 0.941→0.896).
3. VRAM unchanged from Stage 2 (RSSM adds only 186k params).
4. WM recon loss ≈ 4e-6 (MiniGrid dynamics are simple).

See `docs/stage3_report.md` for full analysis.

---

## [v0.2.0-stage2-cloud] — 2026-07-03

### Stage 2 COMPLETED on cloud (AutoDL vGPU-32GB / RTX 4080)

- **Wall time**: 25,037 s (417 min, 7.0 h)
- **Total steps**: 3,000,000
- **Episodes**: 272,998
- **Peak mean_return**: **0.941** (at step ~770k)
- **Final mean_return**: **0.896** (regression in latter half — see report)
- **VRAM**: stable 2.71 GB / 32 GB (8.4%)
- **VRAM slope**: 0.002 GB/h ✅
- **Model params**: ~1.5M (HybridActorCritic with TTT-Linear + SWA + FFN × 3)
- **Checkpoints saved**: 300

### Key validation
**TTT-Hybrid architecture works for RL.** This is the first-ever training of
a TTT-Linear + Sliding-Window Attention + FFN backbone in a reinforcement
learning setting. mean_ret peaked at 0.941 (vs Stage 0's 0.951), proving the
architecture is expressive enough for policy learning.

### Issues
- mean_ret regression 0.941 → 0.896 in the latter half (possible causes:
  entropy collapse, LR too high, RND interference — documented in report).
- alarm_fired=True (transient slope spike at one sample; VRAM was stable).

### Bugs fixed during this run
- NaN from batch-as-sequence: HybridActorCritic treated batch dim as temporal
  sequence, causing TTT-Linear W to explode. Fixed: each obs is independent
  seq_len=1 (commit `829cff2`).
- Triton warning spam: get_backend() not cached. Fixed with @lru_cache.

See `docs/stage2_report.md` for full analysis.

---

## [v0.1.0-stage1-cloud] — 2026-07-03

### Stage 1 COMPLETED on cloud (AutoDL vGPU-32GB / RTX 4080 sm_89)

- **Wall time**: 16,160 s (269 min, 4.5 h)
- **Total steps**: 3,000,000
- **Episodes**: 598,995
- **Final mean_return**: **0.955** (stable, inherited from Stage 0)
- **Peak VRAM**: 0.93 GB / 32 GB (2.9%)
- **VRAM slope at end**: **0.0 GB/h** ✅
- **alarm_fired**: **False** ✅ (warmup fix verified — no false positive)
- **Coverage**: 11 / 4096 buckets = 0.27% (MiniGrid-5x5 state space is tiny)
- **Replay final**: hot=4096, warm=32768, cold=8 shards (34496), total=71360/73728
- **Checkpoints saved**: 300
- **Unit tests**: 278 passed / 10 skipped
- **check_bounded**: OK across 37 source files

See `docs/stage1_report.md` for the full run card.

### Key validations
1. **RND curiosity**: stable throughout, no NaN.
2. **BoundedReplayBuffer 3-tier**: hot→warm→cold eviction cycle observed
   (periodic 73216→69632→73216 every ~512 steps). Axioms 1, 2, 3 empirically
   validated.
3. **MemoryWatcher warmup fix**: `alarm_fired=False` across entire 4.5h run.
4. **Cross-stage resume**: Stage 0 → Stage 1 step counter reset correctly.

### Bugs fixed during this run
- Cross-stage resume step counter reset (commit `3f3e5e2`).
- ColdShardTier capacity includes pending buffer (commit `490b34f`).

### Environment
- Ubuntu 22.04, Python 3.12.3, torch 2.5.1+cu124, triton 3.1.0
- Preset: `cloud_5090`
- Resumed from: `ckpt_stage0_003000000.pt`

---

## [Unreleased]

### Fixed — Cross-stage checkpoint resume (critical bug)

- `src/train.py::train`: when resuming from a checkpoint whose `stage` field
  differs from the current run's `stage`, **reset the step counter to 0**
  rather than inherit it. Previously, resuming a Stage 1 run from a Stage 0
  ckpt at step 3_000_000 would immediately exit because
  `state.step (3M) >= total_steps (3M)`.
- Same-stage resume behavior unchanged: continues the step counter (allows
  split-run training).
- `tests/test_resume_cross_stage.py`: 3 regression tests covering same-stage,
  cross-stage, and multi-stage-jump scenarios.

### Test coverage after this batch
- **276 tests passing** (was 273, +3 resume tests), 10 skipped.
- `check_bounded`: OK.

---

## [Unreleased]

### Added — Stages 3 / 4 / 5 / 6 wiring (full pipeline integrated end-to-end)

**Stage 3 · World Model**
- `configs/stage3_world_model.yaml`: WM sub-block (z_dim, h_dim, embed, hidden,
  max_rollout_steps, kl_free_nats, lr, update_every_steps).
- `src/train.py`: builds `RSSM` when `world_model` config present; trains on
  transitions sampled from `BoundedReplayBuffer`; per-cycle recon/KL losses
  logged; WM state added to checkpoints.
- `tests/test_stage3_wm.py` (8 tests).

**Stage 4 · Skill Library**
- `configs/stage4_skills.yaml`: skills sub-block (LoRA rank, 3-tier capacities,
  merge threshold, score weights).
- `src/train.py`: builds `BoundedSkillLibrary`; registered in HealthChecker;
  count logged per step summary; skills state added to checkpoints.
- `tests/test_stage4_skills.py` (8 tests).

**Stage 5 · Auto Curriculum**
- `configs/stage5_curriculum.yaml`: curriculum sub-block with 5 declared
  MiniGrid tasks (Empty-5x5..8x8, DoorKey-5x5/6x6).
- `src/train.py`: builds `AutoCurriculum`; loads tasks from config; every
  `report_every_steps` reports (1 - mean_return) as LP error; every
  `switch_every_steps` re-samples an active task and rebuilds the env.
- `tests/test_stage5_curriculum.py` (9 tests).

**Stage 6 · Continual (Online EWC + Generative Replay + Sleep)**
- `configs/stage6_consolidation.yaml`: continual sub-block covering EWC
  (lambda, gamma, anchor mode, consolidate every-steps), Generative Replay VAE
  (latent_dim, hidden, lr, kl_weight, update-every, rehearsal batch,
  inject-every), and sleep periods (replay_trim / skills_merge / ttt_distill).
- `src/train.py`:
  - Builds `OnlineEWC`, `GenerativeReplayVAE`, `SleepConsolidationLoop`.
  - Adds EWC penalty to PPO loss once `ewc.has_consolidated()`.
  - Trains VAE from replay every N env-steps.
  - Sleep loop registers `replay_trim / skills_merge / ttt_distill /
    ewc_consolidate` callbacks. Sleep ticks in the training loop.
  - EWC / GR / sleep state added to checkpoints.
- `src/utils/config_schema.py`: `TopLevelSchema` now accepts `world_model`,
  `skills`, `curriculum`, `continual` optional sub-blocks (permissive).
- `tests/test_stage6_continual.py` (9 tests).

**Trainer defaults**
- `src/train.py::_DEFAULT_STAGE_CONFIGS` now maps stages 0–6 to their
  respective yaml files.

### Test coverage
- **273 tests passing** (was 239, +34 Stage 3–6), 10 skipped, 0 failing.
- `check_bounded`: OK across 37 source files.

### Full stack summary (Stage 6 active)
When `--stage 6 --preset cloud_5090`, the trainer builds:
- ActorCritic backed by HybridBackbone (TTT-Linear + SWA + FFN).
- RND intrinsic reward → augments extrinsic reward.
- BoundedReplayBuffer (3-tier GPU/CPU/SSD) fed every step.
- BoundedCoverage tracking state-visitation entropy.
- RSSM world model trained on replay.
- BoundedSkillLibrary registered in HealthChecker.
- AutoCurriculum switching envs by learning progress.
- OnlineEWC penalizing weight drift after each consolidation.
- Generative Replay VAE learning obs distribution.
- SleepConsolidationLoop firing periodic offline maintenance.

All components enforce their bounded-design axioms; each is state-serializable
and included in the checkpoint envelope.

---

## [Unreleased]

### Added — Stage 2 wiring (Hybrid backbone in the training loop)

- `configs/stage2_hybrid.yaml` — Stage 2 config: enables `use_hybrid_backbone`,
  keeps all Stage 1 blocks (RND / Replay / Coverage).
- `src/train.py`:
  - New `HybridActorCritic` class: CNN encoder → HybridBackbone
    (TTT-Linear + SWA + FFN) → policy/value heads. Treats the rollout batch as
    a length-B sequence so causal SWA + TTT-Linear see cross-step context.
  - `train()` now picks between vanilla `ActorCritic` and `HybridActorCritic`
    based on `config.model.use_hybrid_backbone`.
  - `d_model` auto-snapped up to a multiple of `n_heads` and to even (for PE).
- `src/utils/config_schema.py`: `ModelSchema` extended with 7 Hybrid knobs
  (`use_hybrid_backbone`, `hybrid_n_layers`, `hybrid_n_heads`, ...); validates
  hyperparameter ranges when hybrid is on.
- `src/train.py::main`: `--stage 2` auto-loads `stage2_hybrid.yaml`.
- `tests/test_stage2_hybrid.py`: 11 tests covering config load/validate,
  Hybrid output shape/grad flow/determinism, d_model snapping, param-count
  sanity, and shape parity with the baseline `ActorCritic`.

### Test coverage
- **239 tests passing** (was 228, +11 Stage 2), 10 skipped, 0 failing.
- `check_bounded`: OK across 37 source files.

### How to run Stage 2 on cloud
```bash
cd ~/karbon
git pull origin main
LATEST=$(ls -t /root/autodl-tmp/karbon/ckpts/ckpt_stage*_*.pt | head -1)
tmux new -d -s stage2 "source .venv/bin/activate && bash scripts/cloud/run_stage.sh 2 cloud_5090 --resume $LATEST"
tmux attach -t stage2
```

Note: resuming Stage 0 ckpts into Stage 2 will warn "Model state mismatch" and
start the Hybrid model fresh (expected — different architecture).

---

## [Unreleased]

### Added — Stage 1 wiring (RND + Bounded Replay + Coverage)

- `configs/stage1_curiosity.yaml` — Stage 1 config: intrinsic (RND), replay
  (3-tier bounded), coverage (fixed-bucket state-visitation).
- `src/train.py` — trainer now supports Stage 1:
  - Reads optional `intrinsic` / `replay` / `coverage` config blocks.
  - Adds RND intrinsic reward to environment reward (with `reward_coef`).
  - Pushes every transition to `BoundedReplayBuffer` (with PER priorities).
  - Every N env steps runs an off-policy TD update from replay
    (`--stage 1` triggers this path automatically).
  - Adds `BoundedCoverage` class (fixed hash-bucket state-visitation counter).
  - Stage 0 codepath fully unchanged (backward-compatible).
- `src/utils/config_schema.py` — top-level schema now accepts optional
  `intrinsic` / `replay` / `coverage` sub-blocks (permissive validation).
- `src/train.py::main` — auto-selects `stage{N}_baseline.yaml` or
  `stage{N}_curiosity.yaml` based on `--stage`.
- `src/monitoring/memory_watcher.py` — added `warmup_seconds` (default 300 s)
  to suppress startup slope alarms. Trainer passes this through via
  `monitor.warmup_seconds` config key.
- `tests/test_stage1_config.py` — 11 new tests covering:
  - Stage 1 config exists / loads / validates.
  - Intrinsic / replay / coverage hyperparameter sanity.
  - `BoundedCoverage` capacity enforcement, dedup on repeated states,
    state-dict roundtrip, `BoundedComponent` protocol conformance.

### Test coverage after this batch
- **228 tests passed** (was 217), 10 skipped, 0 failing.
- `check_bounded`: OK across 37 source files.
- Stage 0 backward-compat: verified via `test_config_presets` still green.

---

## [v0.0.0-stage0-cloud] — 2026-07-02

### Stage 0 COMPLETED on cloud (AutoDL vGPU-32GB / L40 sm_89)

- **Wall time**: 3906.8 s (65 min)
- **Total steps**: 3,000,000 (full run, not smoke)
- **Episodes**: 555,169
- **Final mean_return**: **0.951** (near-optimal for MiniGrid-Empty-5x5-v0)
- **Peak VRAM**: 0.82 GB / 32 GB (2.5%)
- **VRAM slope at end**: **0.0 GB/h** ✅ (Axiom 5 clean)
- **Checkpoints saved**: 300
- **Unit tests**: 217 passed / 10 skipped
- **check_bounded**: OK across 37 source files

See `docs/stage0_report.md` for the full run card.

### Fixed
- `MemoryWatcher`: added `warmup_seconds` (default 300 s) to suppress startup
  slope alarms. During the Stage-0 cloud run the alarm fired once during
  CUDA-context / Adam-state initialization (which looks like a fast slope in
  the 5-minute rolling window). Long-term slope was 0.0 GB/h. The alarm was
  a false positive; the warmup gate prevents it in future runs.
- Added `test_memory_watcher_warmup_suppresses_alarm` and
  `test_memory_watcher_alarm_fires_after_warmup` to guard the fix.

### Environment (locked for reproducibility)
- Ubuntu 22.04
- Python 3.12.3
- torch 2.5.1+cu124 (from AutoDL preset image)
- triton 3.1.0
- Preset: `cloud_5090` (32 GB budget matches vGPU-32GB exactly)
- Env vars: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` +
  `DEVAGI_{CKPT,DATA,LOGS}_DIR=/root/autodl-tmp/karbon/*`

---

## [Unreleased]

### Added — Nightly / interruptible training

- `scripts/cloud/nightly_run.sh` — one-command starter for off-peak-only
  cloud training. Auto-resumes from newest ckpt, launches trainer +
  autosync in parallel tmux sessions, optionally auto-stops after
  `--duration <sec>` or `--until <time>` and does a final Git sync.
- `configs/_presets/cloud_5090.yaml`: `ckpt_every_steps` reduced from
  25_000 → 10_000 so nightly interrupts lose at most ~15–30 min of work.
- `MIGRATION.md` §10: nightly rhythm, cost tradeoff, do/don't list.

### Interrupt tolerance summary
- Stage 0–5: FULL support (resume from any ckpt).
- Stage 0 24h longevity: NOT interruptible (hard exit criterion).
- Stage 6 30-day perpetual: NOT interruptible (final milestone).

### Cost scenarios (vGPU-32GB @ ¥1.58/h)
- 24/7:        ~30 days, ¥1140
- Night-only:  ~60 days, ¥570 (12h/day)
- Weekends:    ~90 days, ¥456 (~48h/wk)

---

## [Unreleased]

### Added — Autosync daemons (periodic GitHub / TOS / rsync push)

- `scripts/cloud/autosync_daemon.sh` — Linux daemon that every N seconds:
  1. Commits & pushes small text artefacts (docs/, configs/, CHANGELOG.md) to GitHub.
  2. `rsync`s checkpoints/figures to `${DEVAGI_REMOTE_TARGET}` if set.
  3. Optional `--export`: re-exports the newest ckpt into `exports/latest/` HF layout.
  Best-effort semantics — never crashes; trap SIGTERM/SIGINT for graceful shutdown.
- `scripts/local/autosync_daemon.ps1` — PowerShell equivalent for the Windows laptop.
- `MIGRATION.md` §9: bilingual guide on launching, stopping, and tuning the daemon.
- `tests/test_autosync_daemon.py`: 4 structural tests (LF endings, shebang, trap
  handler, no bare `set -e`, no leaked PATs).

### How to use during Stage 0 training on the cloud
```bash
# In one tmux session:
bash scripts/cloud/run_stage.sh 0 cloud_5090

# In another tmux session (parallel):
tmux new -d -s devagi_autosync \
    "bash scripts/cloud/autosync_daemon.sh --stage 0 --interval 3600"
```

The daemon will push whatever reports / config snapshots you drop into
`docs/` and `configs/` throughout the run, so the GitHub repo stays fresh
without manual intervention.

### Test coverage after this batch
- **215 tests passed** (was 211), 10 skipped, 0 failing.
- `check_bounded`: OK.

---

## [Unreleased]

### Added — Full-journey planning doc

- `FULL_JOURNEY.md` — Bilingual end-to-end Stage 0–6 timeline, cost matrix,
  hardware purchase guide, risk register, and copy-paste kickoff commands.
  Recommends Route C (hybrid: cloud vGPU/5090 for Stage 0–4, home 64G rig
  for Stage 5–6 perpetual).
- README: link to FULL_JOURNEY.md.

### Route summary (from FULL_JOURNEY.md)
- Route A · full cloud (vGPU):    ~85 days, ¥2900–3500
- Route B · full cloud (5090):    ~60 days, ¥3900–4550
- **Route C · hybrid (recommended): ~90 days, ¥1200–1600 + hardware (¥16k–60k)**

---

## [Unreleased]

### Added — Cloud-training operational scripts

- `scripts/preflight.py` — 10-step pre-training checklist (Python version, torch/CUDA, GPU inventory, Triton, disk, env vars, project imports, bounded check, preset load, 20-step smoke). Exits nonzero on any critical failure.
- `scripts/cloud/sync_to_git.sh` — pushes small stage artefacts (reports, figures, config snapshots, CHANGELOG) + optional git tag to remote. Explicitly documents the split: text → Git, binaries (`*.pt`, replay data) → TOS/rsync mirror.

### Guidance
- Reports (`docs/stage*_report.md`) and preset snapshots ARE synced to Git.
- Checkpoints, replay cold data, exports/ are NOT — they go to TOS or an rsync mirror.

---

## [Unreleased]

### Added — RTX 5090 / Blackwell support + platform-image (PyTorch 2.8 / CUDA 12.8) support

Target scenario: cloud VMs with pre-installed **PyTorch 2.8.0 / Python 3.12 /
Ubuntu 22.04 / CUDA 12.8** running on RTX 5090 (Blackwell / sm_120).

- `requirements/cuda128.txt` — new; `torch>=2.8,<2.9` + `triton>=3` on cu128 wheels.
  Ships sm_120 kernels required by RTX 5090.
- `configs/_presets/cloud_5090.yaml` — new preset:
  22 GB VRAM budget, batch=16, seq=96, 16 parallel envs, 6-layer × 384-dim model.
  Sits between `cloud_24g` and `home_64g`.
- `scripts/cloud/setup_env.sh`:
  * Auto-detects Python 3.10/3.11/3.12 (was 3.10-only).
  * Detects RTX 50-series → auto-uses `cuda128.txt`.
  * New `--skip-torch` flag to reuse pre-installed torch on platform images.
- `scripts/home/setup_env.sh`: same auto-detect + `--skip-torch` support.
- `pyproject.toml`: `requires-python = ">=3.10,<3.13"`; black targets py310/11/12.
- `HARDWARE_TOPOLOGY.md`: Phase-2 section now documents 5090 wheel selection.
- Preset test coverage extended to `cloud_5090` (211 tests total).

### Test coverage after this batch
- **211 passed** (was 208), 10 skipped, 0 failing.
- `check_bounded`: OK across 37 source files.

---

## [Unreleased]

### Added — HuggingFace-format export path (this session)

**Export tooling for TOS / HuggingFace Hub / ARK custom-model upload:**
- `scripts/export_hf.py` — converts `src.utils.ckpt.save_ckpt` payloads to HF
  layout: `config.json` + sharded `model.safetensors` (+ `model.safetensors.index.json`
  when >5 GB) + bilingual `README.md`.
- Supports architectures: `hybrid_backbone`, `rssm`, `rnd`, `ttt_linear`.
- Supports dtype cast: `float32` / `float16` / `bfloat16`.
- `scripts/build_demo_export.py` — one-shot generator for demo `exports/demo-hybrid-{fp32,fp16}/`
  directly uploadable to TOS.
- `requirements/base.txt`: adds `safetensors>=0.4`.
- `.gitignore`: ignores `exports/` (upload artefacts, not for Git).
- `.gitattributes`: marks `*.safetensors` as binary.
- `MIGRATION.md` §8: bilingual export + TOS upload guide.
- `tests/test_export_hf.py`: 8 unit tests covering flatten / shard / dtype cast
  / roundtrip / architecture whitelist.

### Test coverage after this batch
- **208 tests passing**, 10 skipped, 0 failing.
- `check_bounded`: OK across 37 source files.

---

## [Unreleased]

### Added — Full local pre-work batch (A–N)

**Models:**
- `src/models/ttt_mlp.py` — TTT-MLP with 2-layer inner MLP, analytic GELU derivative, mini-batch dual form.
- `src/models/world_model.py` — Dreamer-style RSSM: encoder / decoder / GRU / prior / posterior heads, bounded rollouts.

**Memory:**
- `src/memory/skill_library.py` — Bounded 3-tier LoRA-based skill library with LRU × usefulness × reward eviction and cosine-similarity merging.
- `src/memory/generative_replay.py` — Small MLP VAE for anti-forgetting rehearsal.

**Intrinsic / Curriculum / Continual:**
- `src/intrinsic/learning_progress.py` — Per-task ring buffer + LP metric, smoothing, priority normalization.
- `src/curriculum/auto_curriculum.py` — LP-driven task sampling with FIFO eviction and ε-exploration.
- `src/continual/online_ewc.py` — Single-Fisher exponentially-decayed EWC with penalty and gradient integration.
- `src/continual/consolidation.py` — Periodic sleep-consolidation loop with warmup gate and disabled-task support.

**Envs:**
- `src/envs/crafter_wrapper.py` — Stage-3 Crafter wrapper with lazy import, auto-reset, bounded episode-return history.

**Utils:**
- `src/utils/config_schema.py` — Dataclass-based config validation catching typos, wrong types, out-of-range values.

**Scripts:**
- `scripts/home/setup_env.sh` — Phase-3 home 64G rig setup with VRAM ≥40 GB sanity check.
- `scripts/home/run_perpetual.sh` — tmux-wrapped perpetual training launcher.
- `scripts/home/health_daemon.sh` — External CSV-logging health monitor with VRAM slope alarm.

**Docs & governance:**
- `AGENTS.md` — Operating protocol for automated coding assistants.
- `CONTRIBUTING.md` — Human contributor guide.
- `notebooks/memory_profiling.ipynb` — MemoryWatcher CSV visualization.
- `notebooks/skill_visualization.ipynb` — Skill library usage / weight-heatmap / similarity analysis.
- `notebooks/ttt_state_inspection.ipynb` — TTT-Linear inner-state per-segment norm plot.

**Tests:**
- `tests/test_ttt_mlp.py` (7)
- `tests/test_skill_library.py` (13)
- `tests/test_world_model.py` (10)
- `tests/test_learning_progress.py` (13)
- `tests/test_auto_curriculum.py` (10)
- `tests/test_online_ewc.py` (11)
- `tests/test_consolidation.py` (9)
- `tests/test_generative_replay.py` (9)
- `tests/test_config_schema.py` (16)
- `tests/test_integration_stage0.py` (2) — End-to-end wire-up test using a DummyEnv.
- `tests/test_crafter_wrapper.py` (7 + 1 skipped) — Fake-crafter-based mechanics tests.

### Test coverage after this batch
- **200 tests passing**, 10 skipped (Triton parity + Crafter install), 0 failing.
- `check_bounded`: OK across 37 source files.

### Stage readiness after this batch
- Stage 1 ready: `RND` + `BoundedReplayBuffer` complete.
- Stage 2 ready: `TTT-Linear` + `TTT-MLP` + `SlidingWindowAttention` + `HybridBackbone` complete.
- Stage 3 ready: `RSSM` world model + `CrafterWrapper` complete.
- Stage 4 ready: `BoundedSkillLibrary` complete.
- Stage 5 ready: `LearningProgressTracker` + `AutoCurriculum` complete.
- Stage 6 ready: `OnlineEWC` + `SleepConsolidationLoop` + `GenerativeReplayVAE` complete.
- Static enforcement of six axioms operational; `make check-bounds` integrated.

---



## [v0.0.0-stage0-local] — planned

### Added
- Full project skeleton at `D:\karbon\`.
- Documents: `PLAN.md`, `README.md`, `HARDWARE_TOPOLOGY.md`, `MIGRATION.md`, `DESIGN_PRINCIPLES.md`, `GLOSSARY.md`, `ROADMAP.md`.
- Requirements split: `base.txt`, `cpu.txt`, `cuda121.txt`, `dev.txt`.
- Platform abstraction: `src/platform/device.py`, `paths.py`, `memory_probe.py`.
- Monitoring: `src/monitoring/memory_watcher.py`, `longevity_test.py`, `health_check.py`.
- MiniGrid wrapper: `src/envs/minigrid_wrapper.py`.
- Minimal PPO baseline: `src/train.py`.
- Three-tier preset system: `configs/_presets/{local_smoke,cloud_24g,home_64g}.yaml`.
- Stage 0 config: `configs/stage0_baseline.yaml`.
- Unit tests for platform/presets/memory.
- Scripts: `scripts/local/setup_env.ps1`, `smoke_test.ps1`.

### Bootstrap decisions
- Project root: `D:\karbon\` (no sub-directory).
- Python: 3.10 + project-local venv.
- PyTorch: 2.5.1 (+cpu locally, +cu121 on cloud/home).
- Docs: bilingual, Chinese-friendly.
- Cloud platform: user-choice, platform-agnostic project code.
- Persistence: rsync after each training run.
- Longevity: perpetual target (Stage 6: 30 days uninterrupted).

---

## Template for future stages / 后续 Stage 模板

```
## [vX.Y.0-stageN] — YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...

### Deprecated / Removed
- ...

### Bounded-axiom review
- ...

### Longevity result
- 24h VRAM slope: ... GB/day
- ...
```
