# Changelog

All notable changes to this project are documented here.
本项目所有值得记录的变更。

## [Unreleased]

### A+B scaling strategy adopted into training plan / A+B 规模解法纳入训练计划

- Chose **A+B combination** as the official answer to the scale-vs-Axiom-1
  conflict: B = Dreamer imagination training (already enabled in Stage 6 config,
  gradient back to main model); A = bounded hierarchical external memory
  (generalize Stage-4 LoRA skill library into GPU/CPU/SSD retrieval-injected
  memory). Documented in `docs/path-to-northstar.md §1.6`.
- Folded A+B into the **ROADMAP Stage 6 training plan** (not optional extras):
  B is live via `imagination.enabled`; A is a Stage-6 deepening task. Rejected
  fallbacks C (bounded MoE) / D (relax Axiom limit) kept on record only.

### Path-to-North-Star analysis / 通往北极星路径分析

- Added `docs/path-to-northstar.md`: maps the five common AI routes to karbon's
  actual code state and derives a combined plan (World-Model + Neuro-Symbolic +
  Core-Knowledge + frozen-LLM-anchor) that does **not** violate any PLAN Non-Goal.
- **Code-fact correction**: Dreamer-style imagination training
  (`src/training/imagination_trainer.py`, called at `train.py:2630` with gradient
  flowing back into the main model) is **already wired in** — it is merely gated
  by `imagination.enabled` in the yaml, currently off in Stage 5. So path-2
  completion is near-zero-effort (flip the switch), not a rewrite. Corrects the
  prior assumption that `imagine_step` was "unused".
- Also documents: Axiom-1 conflict with 500M–1B naive scaling; MuJoCo swap would
  break the developmental chain; neuro-symbolic bridge is cosine-match not real
  unification (accurate prior critique).

### LLMFusion timing decision / LLM 融合激活时机

- **Corrected a misstatement**: `src/models/llm_fusion.py` is a **local offline
  frozen Qwen-7B** (4-bit, ~5 GB, no external API; template-mode fallback), not
  a cloud dependency. Verified by reading the code (`_try_load_llm`).
- **Decision**: keep LLMFusion **deactivated during Stage 5–6 main line**; defer
  to the late B-plan "language emergence" phase. Raising `call_interval_steps`
  frequency / model size now would only slow training and destabilize the
  forming policy. Config stays at defaults. Rationale documented in ROADMAP
  Post-Stage-6 section.

### Stage 6 config + Roadmap branch clarification / Stage 6 配置与支线定位

- **`configs/stage6_consolidation.yaml` corrected to resume from Stage 5:**
  switched `hybrid_n_layers` 3 → 7, replaced the (incompatible) MiniGrid task
  list with the Stage-5 PhysicsSandbox difficulty ladder (same 64×64 obs / 8-action),
  bumped replay capacities to Stage-5 scale (hot 16384 / warm 131072 / cold 32×16384),
  and added `train.total_steps: 5000000` (was missing). Keeps Online EWC +
  generative replay + sleep loop.
- **ROADMAP: split cognitive branch from the developmental main line.** Stage 6
  now explicitly *continues on PhysicsSandbox* (the developmental backbone M1–M5);
  MiniGrid 3D + language is documented as a **post-Stage-6 cognitive branch**
  (instruction-following + sparse-reward multi-step planning — needed for the
  8–15y North Star but not on the critical path). Noted MiniGrid obs is
  incompatible with the Stage-5 vision encoder and needs its own adapter.

### Goal clarification / 目标澄清

- **Confirmed North Star: human 8–15-year-old intelligence, reached *from
  scratch* via autonomous developmental growth** — the goal is the altitude,
  but the *binding constraint is the path* (must grow like a human, not be
  pre-trained). Rewrote `PLAN.md §1.5` to state this North Star explicitly and
  frame **M1–M6 as a verifiable developmental ladder leading toward it** (not a
  replacement for it). Honest framing kept: a serious attempt, not a guarantee.
- **Verified Stage-4 exit criteria against `ckpt_stage4_002000000.pt`:**
  GPU-bound criterion **PASS** (GPU tier = 256 ≤ K under 2M steps);
  skill-reuse criterion **NOT MET** (`usage_count == 1` for all skills —
  library is add-only, no retrieval/re-apply path in `train.py`). Recorded as
  milestone **M2** gap; blocking Stage 5. ROADMAP Stage 4 annotated.

### M2 skill-reuse loop implemented / M2 技能复用闭环已实现

- **Closed the M2 gap with a genuine reuse path (not a counter-only hack):**
  - `src/memory/skill_library.py`: added `retrieve()` (cosine over flattened LoRA,
    reuse of `_flatten`) and `sample_for_injection()` (score-weighted pick from the
    GPU tier) so a stored skill can be selected and re-applied.
  - `src/train.py` `HybridActorCritic.forward`: added `skill_delta` LoRA-residual
    injection (`z = z + skill_delta.apply(z)`) — dims match skill_shape
    (d_in=d_out=d_model=128).
  - `src/train.py` rollout loop (Stage 4): at episode start inject a stored skill
    into the policy; at episode success, `record_use` the injected skill (real
    reuse → `usage_count > 1`) and distill a new candidate via `retrieve` +
    `_merge`/`add` to avoid duplicates.
  - By the goal-first rule this is the genuine reuse path (skill affects behavior),
    not the cheaper count-only shortcut.
- **Pre-existing crash fixed (blocked M2 verification):** `wm_last_loss` init dict
  at `train.py:1781` lacked the `'reward'` key, crashing the final summary log under
  smoke-only runs where the wm-update block is skipped. Added `'reward': 0.0`.
- **Tests:** added 5 M2 cases to `tests/test_skill_library.py`
  (`test_retrieve_*`, `test_sample_for_injection_*`, `test_m2_reuse_loop_records_usage`);
  all pass. Full `tests/test_skill_library.py` and `tests/test_stage4_skills.py` green.

### Known failing tests (pre-existing, TODO — not from planner/reward work)

- `tests/test_stage3_wm.py::test_stage3_config_validates`:
  `ConfigValidationError: unknown keys under model: ['slot_dim',
  'slot_num_iterations', 'slot_num_slots', 'use_slot_attention']` — config
  schema (`src/utils/config_schema.py`) doesn't declare SlotAttention keys.
- `tests/test_model_growth_v2.py::TestGrowerV2PlateauLP::test_spike_is_forgotten_so_growth_can_refire`.
- Both confirmed failing on `4c8d485` (before the planner-disable / reward work),
  so unrelated to this change. Fix or follow-up needed.

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

### Changed (Stage-2 9M resume tuning / 续跑调参)

- **9M 续跑超参（config-only，本地已改、push 后远程 resume 生效）**：
  `exploration_bonus.coef 0.1→0.5`（加强探索奖励以突破 ~102 平台）、
  `train.entropy_coef 0.03→0.015`（收紧熵约束促使策略收敛）、
  `train.total_steps 9000000→11000000`（抬高上限使 9M resume 实际续跑而非立刻退出）。
  9M 终值 `mean_ret=101.4`、`entropy≈2.0` 印证 Plateau + 策略仍未收敛，需此干预再突破。

- **9.5M 第二轮调参（回退过强探索，促收敛）**：
  9M→11M 续跑在 100.8~101.1 震荡、封顶恰在 9M 旧峰值 101.4 之下；诊断确认
  `exploration_bonus.coef=0.5` 把 `entropy` 钉死在 ~2.0、阻止策略结晶。
  故回调：`exploration_bonus.coef 0.5→0.2`（退掉过度探索、留防死区）、
  `train.entropy_coef 0.015→0.008`（更强收敛压力，让熵干净跌破 1.5）。
  从 `ckpt_stage2_009520192.pt` resume，`total_steps` 维持 11M。

- **9.66M 回退第二轮调参（回归，撤销 358b432）**：新配置（bonus 0.2 / ent 0.008）
  实测 `mean_ret` 仅到 99.8 即回落至 99.5，严格差于旧配置（bonus 0.5 / ent 0.015）的
  100.8~101.1——探索奖励是**有用的奖励信号**而非单纯的"熵压制"。三种配置对比：
  原始(0.1/0.03)→101.4、9M-resume(0.5/0.015)→100.8~101.1、本轮(0.2/0.008)→99.5。
  **~101.4 是当前架构/预设的能力天花板**，超参仅在 99.5~101.4 内挪动、无法突破。
  故回退 bonus 0.5 / ent 0.015，`total_steps` 提到 13M 防自动停。

### Fixed (Stage-2 ModelGrowerV2 resume correctness / 续跑不随机化)

- **续跑时按检查点层数构建模型+生长器（防止 3 层权重误载入 2 层模型→随机初始化）。**
  原 `train.py` 用 config 的 `hybrid_n_layers`(=2) 构建模型与 `ModelGrowerV2`，而
  `ckpt_stage2_011000384.pt` 是 **3 层**模型；`model.load_state_dict` 因 `backbone.blocks.2.*`
  尺寸不匹配报错 → "starting model fresh" → 模型被**随机重新初始化**，生长器随后 2→3
  把随机模型当已训练模型继续训（表现为 `mean_ret` 续跑后暴跌到 ~70~82 且不再回升）。
  修复：续跑前用新增的 `_ckpt_layer_count(resume)` 窥探检查点的 `backbone.blocks.N` 层数，
  按该层数构建模型与生长器（`initial_layers` 同步），使 3 层权重正确载入。
  新增 `tests/test_resume_layer_count.py`（合成 2/3 层 state_dict + 缺失文件/无 model_state 用例）。
  实证：续跑日志出现 `Resume ckpt has 3 layers; building model+grower to match` 且
  `Model: HybridActorCubit (... layers=3 ...)`，首步 `mean_ret≈103`（真实权重，非随机）。

- **续跑生长冷静期用 `resumed_step` 而非 `state.step`（防续跑探索重置谷里误触 3→4）。**
  原冷却逻辑在 `state.step` 仍=0 时读取，导致 `last_growth_step` 被设为 0、1M 冷却形同虚设，
  续跑后仅数千步即在 `mean_ret` 探索谷（~73）触发 `grown to 4 layers`。
  改为在 `resumed_step` 解析后设置 `last_growth_step = max(原有, resumed_step)`，
  使下一次真实生长被 1M 冷却挡到架构真正平台期。当前 4 层任务已自发突破并自动备份
  （`watch_backup.sh` → `/root/autodl-tmp/karbon/backup/`）。

### Changed / Fixed (Stage 2 → Stage 3 饱和切换 / Saturation → World Model)

- **Stage 2 生长达到能力天花板（确认饱和，停止生长）。**
  6 层在 `step≈15.14M`（`ckpt_stage2_015160384.pt`）已封顶 `mean_ret≈102.47`；
  5→6 增益仅 +0.4（102.0→102.5）已趋平。6→7 在 `step=16142912` 触发后实测
  `mean_ret=100.44`、`lp=0.017`、`rmax=102.18` → **增益 = −0.29（负）且 `lp<0.05`**
  → 判定 SATURATED。6 层为最优架构，切换前已将最优检查点
  `ckpt_stage2_016140352.pt`（6 层最后状态，d_model=128）三重备份：
  live `/root/karbon/checkpoints/` + 系统盘镜像 `/root/karbon/backup/saturated_6layer_*`
  + 常规备份 `/root/autodl-tmp/karbon/backup/saturated_6layer_*`（sha256 一致）。

- **修复 `train.py` 世界模型反向传播变量名错误（`wm_loss` → `wm_out["loss"]`）。**
  `src/train.py` Stage-3 世界模型更新段误用未定义的 `wm_loss.backward()`，导致
  `NameError: name 'wm_loss' is not defined`。改为 `wm_out["loss"].backward()`
  （`wm_out` 由 `wm.compute_loss(...)` 返回）。否则 Stage 3 任何训练步必崩。

- **修复 `train.py` MiniGrid 分支提前引用未定义变量 `obs_shape`/`num_actions`。**
  `src/train.py` 在 `env.reset()` 之前于 `logger.info` 引用尚未赋值的 `obs_shape`/
  `num_actions`，走到 MiniGrid 分支即 `UnboundLocalError`。改为先 `reset()` 取
  `observation_shape`/`action_space_n` 后再统一打印；其他分支日志不受影响。

- **`stage3_world_model.yaml` 补齐与 Stage 2 一致的模型/编码器参数（确保 backbone 继承）。**
  Stage 3 config 原缺 `hidden_size` 与 SlotAttention 开关，导致用 preset 默认
  `hidden_size=256` 且 CNN encoder，与 Stage 2 的 d_model=128 + SlotAttention 权重
  尺寸全错 → `load_state_dict` mismatch → "starting model fresh"（6 层成果丢失）。
  补齐 `hidden_size: 128` + `use_slot_attention: true` + `slot_num_slots/slot_dim/
  slot_num_iterations` + `use_vision_encoder: false` + hybrid 子参数，与
  `phase2_infant_exploration.yaml` 的 `model:` 块完全一致；并加 `train.total_steps: 5000000`
  （原缺失 → 默认 200 步即停）。env 沿用 `PhysicsSandbox`（与 Stage 2 同分布，backbone
  直接迁移）。切换后日志确认 `Model: HybridActorCritic (d_model=128, layers=6 + SlotAttention)`
  且 `Cross-stage resume: loaded weights from stage 2 ckpt`，backbone 权重完整载入。

  **重要修正（原"RSSM win"措辞不实）**：v1 训练实测 `mean_ret` 从 102.47 天花板
  稳定爬升至 **105–106**（step 150k→490k 全程站上 105，未回落），但**此突破并非 RSSM
  驱动**——v1 的 `curiosity.mode` 默认 `"none"`，RSSM 只独立训练自身表征（recon 降到
  0.012）、**不进入 PPO 梯度**。突破真实来源是「resume 6 层 backbone 后**不被生长打断**的
  持续 PPO 续训 + RND 探索」让策略真正收敛到比 Stage-2 早停点更高的平台。RSSM 作为策略
  驱动信号尚未验证。

- **备选 `stage3_world_model_v2.yaml`（已备未启用）：让 RSSM 真正驱动策略。**
  v1 证明 backbone 继承 + 不间断续训可破 102.47 天花板；v2 进一步把 RSSM 变成真实辅助信号：
  新增 `curiosity: {mode: rssm_uncertainty, coef: 0.3}`，使 WM 预测误差作为 intrinsic reward
  注入 PPO（样本高效探索，Stage 3 设计本意）；并清理 `model` 块里 `hybrid_*` 的重复键
  （28–42 行曾重复定义 n_heads/swa_window/ttt_mini_batch/dropout，后写覆盖前写，实际生效
  swa_window=16 / ttt_mini_batch=8）。env/backbone 与 v1 完全一致，可直接 resume v1 最优
   检查点切换。待 v1 跑到 ~1M 步确认平台后，再决定是否用 v2 重启验证 RSSM 驱动增益。

- **Stage-3 三版全测 + 结论：6–7 层 hybrid(d_model=128)+SlotAttention 在 PhysicsSandbox
   的真实能力上限 ≈ 101–102，无任何组合稳定突破 Stage-2 的 102.47 天花板。**
   - **v1（RND，已停）**：开局 102.47 → 稳态 **99.5**（退化，RSSM 未进 PPO 梯度）。
   - **v2（`rssm_uncertainty` 好奇心，已停）**：开局 ~105 → 平台 **100–102** → 2M 后**滑落**
     （RSSM 仅作探索奖励，仍不驱动策略）。
   - **v3（v2 + `imagination.enabled=true`，RSSM 直训 actor/critic，已停）**：开局 **105.6**
     → 回落 **99**。Dreamer 式想象训练给出 transient 提升，但终值与 v1/v2 同落 ~99–100。
   - **v4（v3 + `model_growth` 重新开生长，已停）**：resume 2.3M 6层 ckpt 后 **6→7 立即触发**
     （cooldown 重置），7 层稳在 **100.7–101.1**，略高于 v3 的 99 但**未破 102.47**；
     `slope` 衰减到 ~0.95 进入平台，`rmax` 卡 104.77。
   - **四版对比**：v1=99.5 / v2=100–102(slide) / v3=105.6→99 / v4=101±1。
     WM、imagination、层数都只改变 **transient 动态**，无组合稳定突破 102.47。
     `max mean_ret` 各版均为开局 transient 尖峰（v4 max=109.99 @ step1），非真实平台。
   - **结论**：当前 6–7 层架构 + PhysicsSandbox 的**硬上限 ≈ 101–102**；继续训练 v4 信息增量≈0。
     真正破局需换**架构维度**（更大 `hidden_size` 128→256 / 更长 `imagination_horizon` 8→15）
     或**换 env 难度**，而非层数/WM/imagination 的现有组合。
    - v4 最优 7层 ckpt `ckpt_stage4_7layer_002460160.pt` 已三重备份（live + 两 backup 目录，
      sha256 `ff103b15…e802` 一致）；v3 6层最优 `ckpt_stage3_002300416.pt` 亦留档。

- **Stage-3 v5（宽度实验，否定宽度假设）：256 宽 7层 fresh 训练，稳态仍 ≈100.4，未破天花板。**
  为验证「~102 是否为容量/回报估计精度瓶颈」，v5 将 `hidden_size` 128→256（fresh 起 7层，
  关 growth，沿用 v4 的 RSSM+imagination+env）。结果：
  - step 32k 爬到 97（`slope=5`）、step 119k 摸到 `max=101.28`、step 272k **`slope=0` 进入平台**，
    稳态 **`mean_ret≈100.4`**——与 v4（128 宽 7层）的 100–101 **完全重合**。
  - **结论：宽度假设被否定。** 256 宽未带来任何突破，证明 ~102 不是 backbone 容量或回报估计
    精度问题（加宽无效）、也不是层数（6→7 无效）、也不是 WM/imagination（v1-v4 已证）。
    `cov=100%` 表明状态空间已完全覆盖，可达回报上限由 **PhysicsSandbox 的回报结构本身** 锁定。
  - **最终 Stage-3 结论**：PhysicsSandbox + 当前 hybrid(d_model 128/256)+SlotAttention+RSSM 栈的
    **真实能力上限 ≈ 100–102**，与环境回报结构绑定，非同维度容量可破。继续训练 v5 信息增量≈0。
  - **真正破局方向（换维度，非加容量）**：(a) 改 env 本身（更多可交互对象 / 更高回报上限）；
    (b) 改 reward 设计（当前回报信号限制了可达上限）；(c) 加 RSSM **反事实规划**（counterfactual
    planning，DEV_PLAN 提及但 v3/v4 的 imagination 仅是直训 actor，未做规划打分）。
   - v5 最优 256宽 7层 ckpt `ckpt_stage5_wide256_000260096.pt` 已三重备份（live + 两 backup 目录，
     sha256 `52c57cde…c586` 一致）。

- **Stage-3 v6（修复想象奖励 bug，仍无效）：证明 ~102 非想象奖励信号所致，RSSM 想象训练
   无法破局。** `ImaginationTrainer` 原用 reconstruction-error 当想象奖励
   （`imagined_r = -recon_err * 0.1`），与 PhysicsSandbox 真实回报无关。v6 改为
   `world_model.predict_reward(state)`（RSSM reward head，已在 `compute_loss` 用真实 replay
   reward 监督训练），使 Dreamer 式想象训练优化**真实回报**。结果：
   - 开局从 94.7 冲到 **`max=106.39`**（比 v4 错误奖励的开局 101 更高），但 step 2683k
     **`slope=0` 进入平台**，稳态 **`mean_ret≈101.9`**——与 v4（错奖励）的 101±1 **完全重合**。
   - **结论：修复想象奖励 bug 未破局。** 开局 transient 更高（106 vs 101），但稳态仍锁死在
     ~102 同一平台。这彻底证伪「v3/v4 无效是因想象奖励信号错」的假设；RSSM 想象训练无论用
     错/对奖励，都无法让 PPO 策略突破 PhysicsSandbox 的 ~102 回报上限。
   - **关键推论（削弱 counterfactual planning 先验）**：v6 已证明「在想象里用真实回报优化策略」
     对破局部最优无效，而 counterfactual planning 同样依赖 RSSM 在想象里评估回报来选动作——
     若想象回报评估对策略更新无益，planning 也未必更有用。剩余破局方向优先级重排：
     **(b) 改 reward 设计（定向拉高局部最优陷阱的梯度，最可能被低估）> (a) 换 env > (c) 规划**。
    - v6 最优 7层 ckpt `ckpt_stage6_imagfix_002680320.pt` 已三重备份（live + 两 backup 目录，
      sha256 `1d03de5d…7db3` 一致）。Stage-3 架构维度实验（层数/宽度/WM/imagination/奖励信号）
      **至此全部穷尽且均收敛 ~100–102**，确认该上限由环境回报结构 + 局部最优陷阱锁定。

- **Stage-3 v8（counterfactual planning / MCTS，仍仅 transient）：System 2 规划也未破局。**
    `CounterfactualPlanner` + `LongRangePlanner`（MCTS over RSSM）在 train.py 已接线
    （lines 1366-1375, 1507-1516, 2805-2826），用（v6 修好的）`predict_reward` 想象评估候选
    动作序列并覆盖 PPO 动作。v8 启用二者于 v6 的 7层 ckpt 上。结果：
    - 接管初期从 83 冲到 **`max=106.53`**（step 2695168），显著高于 v6 的 101；
      但 step 2730k 后回落到 **`mean_ret≈101.6`**（`cov=100%`, `slope` 缓降至 ~1.1），
      稳态与 v4/v6 的 101±1 **完全重合**。
    - **结论：MCTS 规划仅产生 transient 尖峰（106→101），未抬升稳态。** 印证 v6 的推论：
      planner 短期用真实回报选出更优动作，但 PPO 的梯度更新持续把策略拉回同一局部最优，
      两者拉锯后净稳态仍是 ~101。所有 v1-v8 版本**峰值均 ~106（transient）、稳态全锁 101±1**。
    - **最终 Stage-3 全维度穷尽结论**：层数(6/7) / 宽度(128/256) / WM / imagination(错+对奖励)
      / MCTS 规划 —— 全部只产生 transient 尖峰，无法抬升 PhysicsSandbox 的 ~101-102 稳态。
      上限由**回报景观的局部最优陷阱**锁定：当前 reward 对"被动蹭物体"（物体自身惯性/碰撞
      白送速度奖励）给分，agent 陷在"推几个物体"的局部最优，不主动最大化全局回报。
    - **唯一未排除的破局杠杆：(b) 改 PhysicsSandbox reward 设计** —— 直接重塑回报景观
      （移除被动速度白送、奖励 agent 主动加速物体），可能把 106 的 transient 变成稳态。
      这区别于所有"优化/架构"干预（只改 transient），是从景观根源破局。
    - v8 最优 7层 ckpt（规划版）已并入 v6 的 `ckpt_stage6_imagfix_002680320.pt` 保留；
      系统盘清理：删 258 个 stage2 + 625 个 stage0 + 137 个 stage3 密集 ckpt，释放 14G
      （83%→37%，不记文件）。

- **Stage-3 v9（关闭 MCTS planner / 破局成功）：稳态首次突破 ~102 天花板到 103.8。**
    ⚠️ **归因更正**：v9 config 声称做 reward redesign，但 `src/envs/physics_sandbox.py`
    的 reward 改动**从未上传到远程**（部署时只 sftp 了 config，漏传源码文件）。经核实远程
    环境全程使用**旧 reward**（`speed*0.05` 被动速度奖励，未改）。因此 103.8 **不是** reward
    redesign 的功劳。v8→v9 config 的**唯一真实差异是关闭了 MCTS planner**
    （`long_range_planner` / `counterfactual_planner` 从 v8 的 `enabled: true` 改为不启用）。
    - 同一 7层 ckpt `ckpt_stage6_imagfix_002680320.pt` resume、同一（旧）reward，
      **唯一变量 = planner 开/关**：
      - v8（planner **开**）：稳态 `mean_ret≈101.6`（`slope` 缓降未收敛）。
      - v9（planner **关**）：resume 后 slope 转正冲到 `max=104.75`（step 2706944），
        又训练近 20 万步（→ step 2902016）稳定在 **`mean_ret≈103.8`**，
        最后 15 步 `mean=103.75 std=0.06`，`slope` 收敛至 ~0（对称抖动）。
    - **真实结论：关闭 MCTS planner 使稳态 101.6→103.8（+2.2），且更稳、真收敛。**
      机理：planner 用不完美世界模型的 `predict_reward` 覆盖 PPO 动作 → ①想象回报有偏差、
      选的动作对真实环境非最优；②PPO 采到的是"planner 的动作"而非自身策略动作，
      **策略梯度与实际行为脱节**，拖累收敛。关掉后 PPO 端到端学，稳稳爬到 103.8。
      这进一步削弱"System 2 规划有用"的先验（与 v6 推论一致）。
    - **后续动作**：已把 `phase0_protozoan.yaml` / `phase2_infant_exploration.yaml` 的
      `long_range_planner` / `counterfactual_planner` 全部改为 `enabled: false`（附原因注释）；
      `stage3_world_model_v8_cf_plan.yaml` 保留 `true` 作为 planner-ON 失败对照（附注释）。
    - v9 里程碑 ckpt 已三重备份（sha256 `e7749e0e…c4cc` 一致），
      重命名为 `ckpt_stage7_no_planner_002900480.pt`（原 `_reward_redesign_` 名误导，已弃用）。

- **Stage-3 v10（真正的 reward redesign，已验证 / 结论：效果中性，无显著增益）。**
    v10 是首次把 acceleration reward **真正上传远程**（v9 漏传）并训练的实验：从 no-planner
    ckpt resume、无 planner、env 用新 reward（移除 `speed*0.05` 白送，改奖励接触时 `|Δv|`，
    contact 0.1→0.15，cap 2.0→3.0）。训练 ~15 万步收敛到 `mean_ret≈60`（新 reward 尺度，
    不可直接与旧 reward 的 104 比）。
    - 为跨 reward 尺度对比，写了 `eval_policy.py`（固定 seed + 100 episode，双 reward 尺子 +
      行为指标）。**在旧 reward 这把公平尺子上评估两个策略**：
      | 指标（100ep, seed 固定, stochastic） | no_planner（旧reward训练） | v10（新reward训练） |
      |---|---|---|
      | OLD reward（公平尺子） | **97.31** ±19.4 | **97.88** ±13.9 |
      | NEW reward | 59.96 | 63.81 |
      | 接触物体数/ep | 371.5 | 402.9（+8.5%） |
      | active \|Δv\|（主动推力） | 135.8 | 119.9 |
      | agent path len | 8.42 | 7.20 |
    - **结论：reward redesign 效果中性。** 在公平尺子（旧 reward）上 v10 vs no_planner =
      97.88 vs 97.31，差异 <1%、远在 std 内，**统计上持平**。v10 接触物体更多（+8.5%）但主动
      推力（active |Δv|）反而略低——行为风格不同，但**没有证据表明整体能力更强**。
    - **最终 Stage-3 破局归因**：真正抬升稳态的是**关闭 MCTS planner**（101.6→104）；
      reward redesign 只是把 agent 调到"同等强度、不同行为风格"。
    - **收尾**：远程 + 本地 env `physics_sandbox.py` 均已恢复旧 reward（与 no_planner 起点、
      后续 stage config 一致），reward redesign 代码归档在 git 历史（commit 044703a）+
      `stage3_world_model_v10_reward_on_noplanner.yaml` config 保留。`eval_policy.py` 保留为
      跨 reward 策略评估工具。
    - **附加发现**：greedy（argmax）下两个策略都塌缩成"永远动作 6"，轨迹退化；必须用
      stochastic 采样评估才能体现策略差异（已记录在 eval 脚本注释）。
    - no_planner 稳态复测：v9 继续训练到 step 3M，稳态实为 **~104.1**（比早期记的 103.8 略高，
      slope 完全收敛）。Stage-4 从 `ckpt_stage7_no_planner_002900480.pt` 起步。

- **Stage-4 启动 · Bounded Skill Library。**
    从 `ckpt_stage7_no_planner_002900480.pt`（旧 reward 稳态 ~104）跨 stage 3→4 resume，
    在验证过的 no-planner 7层 hybrid + SlotAttention + RSSM 骨干上叠加持久化技能库。
    - **重写 `configs/stage4_skills.yaml`**：旧版是过时的 3层脚手架（无 env/slot/imagination
      块、num_objects 用默认 3）。新版对齐**实际训练线**：7层 hybrid、SlotAttention(7 slots)、
      PhysicsSandbox(num_objects=10)、imagination、`model_growth.enabled=false`、无 planner、
      旧 reward。skills 块：3-tier 有界库（gpu=256 / cpu=2048 / ssd=64×128，总容量 10496）。
    - **schema 修复（`src/utils/config_schema.py`）**：补齐实际训练线一直在用、但 schema 缺失
      的键，修好长期存在的 config 校验失败：
      - `ModelSchema` 加 SlotAttention 字段（`use_slot_attention/slot_num_slots/slot_dim/
        slot_num_iterations`）+ 校验 → **顺带修复 pre-existing `test_stage3_config_validates`**。
      - `EnvSchema` 加 PhysicsSandbox 字段（`num_objects/render_size/gravity/action_force`），
        `num_envs` 改为可选（default 1）。
      - `TopLevelSchema` 加 `curiosity/imagination/model_growth` 三个可选块。
    - **env reward 回退**：v10 评估确认 reward redesign 无显著增益后，远程 + 本地
      `physics_sandbox.py` 均恢复旧 reward（`speed*0.05`，cap 2.0），与 no_planner 起点一致。
    - smoke（200 步）通过：7层+slot 加载 OK、BoundedSkillLibrary 初始化 OK（有界 10496）、
      旧 reward 得分正常。正式训练已启动（tmux `s4`+`s4rec`），前 4k 步 `mean_ret≈110-133`
      （与起点 ~104 平滑衔接，无 v10 那种 reward 尺度突变），`skills` 计数随训练增长（有界）。
    - 新增 `scripts/eval/eval_policy.py`：跨 reward 尺度的策略评估工具（固定 seed、双 reward
      尺子、行为指标），stochastic 采样（greedy 会塌缩成单动作）。
    - 已知遗留失败（与本次无关）：`test_model_growth_v2.py::...test_spike_is_forgotten...`
      （`GrowthConfigV2` 无 `rmax_decay` 参数）。




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

### Fixed (VisualAnalyzer — silent no-op & wrong trigger)

- **`describe_slot` NameError (`src/models/visual_analyzer.py`).** The f-string
  referenced `{texture}` but the variable was bound as `text`, so every call
  raised `NameError`. Because `train.py` wrapped `feed_to_graph` in
  `try/except Exception: pass`, the whole VisualAnalyzer → ConceptGraph path
  **silently did nothing** (zero nodes written). Renamed to `{texture}` and
  added `tests/test_visual_analyzer.py` (regression for the crash + motion).
- **Motion was always "still".** `feed_to_graph` / `describe_*` re-ran `forward`
  on the same frame, overwriting `_prev_slots` before motion was read, so the
  frame-to-frame diff was always 0. `forward` now caches `_last_out`; the
  describe/feed helpers read the cached result and never re-forward, so motion
  is estimated against the previous step. `train.py` now calls
  `visual_analyzer(slots)` **every step** (updating motion) and only persists to
  the graph every 500 steps.
- **Wrong trigger condition.** `state.step % 500 < rollout_capacity` is true for
  every step (since `step % 500 ∈ [0,499] < 512`), so it never meant "every 500
  steps". Changed to `state.step % 500 == 0`, matching the existing periodic
  hooks in `train.py`.
- `VisualAnalyzer` exported from `src/models/__init__.py`.
- **Not yet trained (known gap):** the classifier heads are randomly initialized
  and not part of the optimizer / checkpoint, so attribute predictions are
  unsupervised heuristics, not learned from SlotAttention. Wiring a supervised
  loss is a follow-up.

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

### Fixed — ModelGrowerV2 growth was dormant (real breakthrough path)

- Root cause: the autonomous layer-growth block in `src/train.py` was gated by
  `if model_grower_v2 is not None and coverage is not None`. The
  `phase2_infant_exploration.yaml` config had **no top-level `coverage:`**
  section, so `coverage` was always `None` and growth never ran — the agent
  stayed pinned at the ~101.4 hyperparam ceiling after 9M steps.
- Added `coverage:` block (`num_buckets: 4096`, `log_every_steps: 5000`) to the
  config; lowered `grow_trigger_coverage` 0.3 → 0.15 (raw-obs hash undercounts
  exploration after 9M steps) and spaced `min_steps_between_growths` → 1M.
- Added `ModelGrowerV2.plateau_lp(mean_return)`: returns headroom
  `max(0, 1 - mean_return / running_max)`, ≈0 on a genuine plateau so growth
  fires, >0 while returns still climb. Replaces the old
  `lp = 1.0 - mean_return` (≈-99, over-eager / nonsensical). Call site in
  `src/train.py` now uses it and logs `[growth-debug]` every 50k steps.
- `tests/test_model_growth_v2.py::TestGrowerV2PlateauLP`: 5 new tests plus a
   regression guard that the old formula was over-eager (not blocking).

- **`rmax` now decays (`GrowthConfigV2.rmax_decay = 0.98`)** so the growth
  trigger line (`0.95 × rmax`) can't be pinned forever by a one-off spike.
  Root cause: on checkpoint resume the first `mean_return` is inflated (e.g.
  105.7 → 113), and the raw running max latched it — a 3-layer model plateaued
  at ~101 could never reach `0.95 × 113 ≈ 107.5`, so the next 3→4 growth would
  never fire. `plateau_lp` now uses a decaying running max
  (`rmax = max(mr, rmax × rmax_decay)`) that forgets spikes within a few
  growth-check calls while still tracking genuinely sustained peaks. Added
  `tests/test_model_growth_v2.py::TestGrowerV2PlateauLP::test_spike_is_forgotten_so_growth_can_refire`.

- **Critical fix: ModelGrowerV2 state is now restored on resume** (`src/train.py`
  resume block loads `model_grower_v2_state`, previously only `model` +
  `optimizer` were loaded — the grower state was saved but never read back, per
  the TODO at the old line 1598). Without this, every resume recreated the
  grower as `initial_layers=2` while the model was already 3/4/… layers,
  causing (a) a spurious no-op "2→3" growth that wasted the 1M-step
  `min_steps_between_growths` budget, and (b) a catastrophic latent bug: a
  future resume onto a 4-layer checkpoint with a fresh 2-layer grower would
  call `_create_larger_model(model, 3)` on the 4-layer model and **silently
  DROP the 4th layer**. Added a post-resume layer-count safety sync
  (`grower._current_layers = model.backbone.n_layers`) and a
  `resume_warmup_calls` (default 5) plateau-check warmup so the inflated
  first-step `mean_return` on resume can't immediately force a growth while
  `rmax` decays back to the true plateau. Added
  `tests/test_model_growth_v2.py::TestGrowerV2ResumeLoad`.

- **Non-disruptive (Net2Net-style) layer growth.** Root cause of the first
  autonomous 3→4 growth landing *below* the 3-layer plateau (~95.6 vs ~101):
  `_create_larger_model` copied the old blocks but **randomly initialized the
  new block**, and `_distill` only distilled on **random-noise images** with
  **all** student params trainable — so the new block learned a *useful-but-
  different* transform from scratch on noise, permanently erasing ~5 points of
  the copied policy. Fix: `grow()` now accepts `distill_inputs` (real
  observations sampled from the replay buffer in `src/train.py`) and `_distill`
  **freezes every parameter except the freshly-added block**
  (`student.backbone.blocks[new_block_idx]`), training only that block to
  reproduce the teacher's outputs (KL + MSE on logits/values). The new block
  thus learns the **identity map on the agent's real data**, so the grown model
  is *equal* to the teacher at the growth step (no drop) and RL can then exploit
  the extra capacity. Verified offline: logit/value drift after growth ≈ 0.3%
  (was a full policy reset). Config gains `distill_steps: 400`, `distill_lr:
  1e-3`, `distill_batch: 1024` under `model_growth:`. Added
   `tests/test_model_growth_v2.py::TestGrowerV2ObsShapeCarryover::
   test_grow_is_non_disruptive_with_real_data` (drift < 10%).

### Verified — First live non-disruptive 3→4 growth (2026-07-17, remote)

- Triggered at **step=12,000,832** (`[growth] grown to 4 layers
  (step=12000832) (non-disruptive, real-data distill)`).
- Matches the design cooldown: `train.py` enforces
  `_last_growth_step = max(loaded, resumed_step=11.0M)` on resume, so the
  3→4 growth fires exactly at the 1M-step cooldown boundary (12.0M), not a
  spurious no-op on resume.
- **No drop at the growth step**: 4-layer mean_return started at ≈100.8
  (= the 3-layer plateau), then climbed past it to **≈101.3 by step
  12.04M** — the earlier catastrophic ~70 collapse is gone.
- **Value head self-healed**: the pre-growth jitter (`v` spiking to 0.8–1.2,
  `cf` up to 0.26) returned to a healthy **0.16–0.5** zone after growth.
- Checkpoint landed: `backup/growth_20260717_191118_ckpt_stage2_012020288.pt`
  (4 layers, step 12.02M). Note: the 3 earlier `growth_*` backups
  (11:55 / 12:12 / 12:28) are monitor growth-detection noise
  (no `[growth]` line), not real model surgery.

### Ops — Remote training backup to system disk

- `monitor.sh` / `watch_backup.sh` now also mirror the train logs +
  latest checkpoint to the **system disk** (`/root/karbon/...` alongside the
  existing `/root/autodl-tmp/karbon/backup/`), so a loss of the
  ephemeral `/root/autodl-tmp` volume can't take the run with it.

### Fixed — 4→5 growth was silently dormant (rmax-collapse bug, 2026-07-18)

- Root cause: `ModelGrowerV2.plateau_lp()` was called **every training step**
  from the `src/train.py` growth-check block, with `rmax` decaying
  `max(mr, rmax × rmax_decay)` **per call** (`rmax_decay=0.98`). At ~1k calls
  per second `rmax` collapsed to the instantaneous `mean_return` within ~35
  steps, so `lp = 1 - mr/rmax` sat at ≈0 **permanently** — yet growth still
  never fired because the post-resume `_warmup_remaining` + the
  `state.step % 50000 == 0` gating of the LP refresh meant the signal was
  effectively stale/never refreshed past the unlock point. Net effect: the
  4-layer agent plateaued at ~101–102 and 4→5 never triggered (observed
  step 14.02M→14.47M, `mean_ret` slowly drifting down to 101.4 with no growth).
- Fix (two-part, BOUNDS-OK):
  1. `plateau_lp` now uses a **fixed-capacity windowed max**
     (`deque(maxlen=rmax_window=40)`) instead of per-call exponential decay,
     so `rmax` tracks the genuine recent peak without being collapsed by the
     per-step call cadence. `rmax_window` added to `GrowthConfigV2` and
     persisted in `state_dict`/`load_state_dict`.
  2. `src/train.py` refreshes the LP signal **once per `growth_check_every`
     (50k) steps** using integer-division buckets (`state.step // N`), and the
     `[growth-debug]` log uses the same bucket approach — both now fire
     regardless of step-cadence alignment (steps advance by 512, so the old
     `state.step % 50000 == 0` condition almost never matched and left growth
     unobservable).
- Verified: after the fix the `[growth-debug]` line emits every 50k-step
  bucket (first at step=14,060,608, `lp=0.0000, layers=4`), and 4→5 is now
  expected to fire once `state.step` passes the 14.40M unlock with `lp≈0` and
  `cov≥0.3`. Relaunched from `ckpt_stage2_014060096.pt` (14.06M) with the
  corrected code; the growth-check block is again active.

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
