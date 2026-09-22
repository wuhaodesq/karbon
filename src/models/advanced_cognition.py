"""Advanced cognitive modules: hypothesis testing, counterfactual reasoning,
behavior cloning, and meta-learning.

Four modules that push the agent from 12-year-old to 14-year-old cognition:

1. :class:`HypothesisTester` — actively probes the environment to verify
   or falsify rules from the NeuralSymbolicLayer. "If I drop the key, does
   the door become unopenable?" → designs an experiment to test it.

2. :class:`CounterfactualImagination` — uses the RSSM world model to imagine
   "what would have happened if I had done X instead of Y". Conditions the
   world model rollout on alternative action sequences.

3. :class:`BehaviorCloningHead` — an auxiliary loss that lets the agent
   learn from expert demonstrations. Given (obs, expert_action) pairs,
   adds a cross-entropy loss that pushes the policy toward expert behavior.

4. :class:`MetaLearner` (MAML-lite) — maintains a meta-gradient that
   captures "how to learn new tasks fast". After each task, stores the
   post-adaptation parameters; on the next task, initializes from the
   meta-average instead of the raw trained weights.

All four are **bounded** (fixed-size state, Axiom 1) and **optional**
(graceful degradation if components are missing).

四个认知扩展模块：假设检验、反事实推理、行为克隆、元学习。
把智能体从 12 岁推到 14 岁。全部有界、可选、可叠加。
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =====================================================================
# 1. HypothesisTester — active experiment design
# =====================================================================


@dataclass
class Hypothesis:
    """A testable hypothesis about the environment.

    - ``condition``: the state pattern that triggers the hypothesis.
    - ``predicted_action``: the action the rule predicts.
    - ``confidence``: current belief in this hypothesis.
    - ``tested``: whether it has been experimentally tested.
    - ``last_result``: the outcome of the last test (reward or None).
    """

    id: int
    condition_embedding: torch.Tensor  # (d_model,)
    predicted_action: int
    description: str = ""
    confidence: float = 0.5
    tested: bool = False
    last_result: float | None = None
    test_count: int = 0

    def update(self, result: float, decay: float = 0.9) -> None:
        self.tested = True
        self.last_result = result
        self.test_count += 1
        signal = 1.0 if result > 0 else 0.0
        self.confidence = decay * self.confidence + (1 - decay) * signal


class HypothesisTester(nn.Module):
    """Actively tests rules by probing the environment.

    Given a rule "IF see key THEN pick up (conf=0.5)", the tester can:
    1. Generate a "probe action" (pick up the key) to verify the rule.
    2. Observe the outcome (reward or not).
    3. Update the hypothesis confidence.

    This is the difference between "I think keys are useful" and
    "I tested it: picking up the key led to success 8/10 times".

    Bounded: max_hypotheses fixed. Each is a small struct. Axiom 1.

    假设检验：主动设计实验验证规则，从"我认为"升级到"我验证过"。
    """

    def __init__(
        self,
        d_model: int = 384,
        num_actions: int = 7,
        max_hypotheses: int = 32,
        probe_epsilon: float = 0.1,
        probe_lr: float = 1e-3,
        probe_buffer: int = 256,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self._num_actions = num_actions
        self._max = int(max_hypotheses)
        self._probe_epsilon = probe_epsilon
        self._hypotheses: deque[Hypothesis] = deque(maxlen=self._max)  # BOUNDS-OK: maxlen bounded
        self._next_id = 0
        self._active_hypothesis_id: int | None = None

        # Small network: maps hidden state → "should I test this rule?"
        self.probe_net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        # 2026-09-22: close the probe-learning loop. probe_net existed but was
        # never trained, and the training loop bypassed it with scripted
        # always-probe activation - the tracking behaviour was rehearsed by
        # scaffolding, never learned by the agent (op oscillated 0.13-0.65
        # across 50k-step checkpoints). Now every genuine tracking outcome
        # (arrival vs reveal-far / deadline) trains probe_net to predict WHEN
        # probing pays off. 探针闭环: 用真实追踪结局训练 probe_net。
        self._probe_optim = torch.optim.Adam(self.probe_net.parameters(), lr=float(probe_lr))
        self._probe_samples: deque = deque(maxlen=int(probe_buffer))  # BOUNDS-OK
        self._probe_updates = 0
        self._probe_loss_last = 0.0
        self._probe_last_error = ""
        # Durable probe bookkeeping: the old code used per-step local
        # variables, so arrival could only be checked on the occlusion step
        # and the timeout never fired; verification degenerated into
        # reveal-always-1.0 (theatre). These fields persist across steps.
        self._active_lk: tuple[float, float] | None = None
        self._active_step: int | None = None
        self._active_hidden: torch.Tensor | None = None

    @property
    def capacity(self) -> int:
        return self._max

    def __len__(self) -> int:
        return len(self._hypotheses)

    def propose_hypothesis(
        self,
        condition_embedding: torch.Tensor,
        predicted_action: int,
        description: str = "",
    ) -> Hypothesis:
        """Propose a new hypothesis from an unverified rule."""
        h = Hypothesis(
            id=self._next_id,
            condition_embedding=condition_embedding.detach().clone(),
            predicted_action=predicted_action,
            description=description,
            confidence=0.5,
        )
        self._next_id += 1
        self._hypotheses.append(h)
        return h

    def should_probe(self, hidden_state: torch.Tensor) -> bool:
        """Decide whether to probe (test a hypothesis) or act normally.

        Returns True if the agent should take a probe action.
        """
        with torch.no_grad():
            p = self.probe_net(hidden_state.unsqueeze(0) if hidden_state.dim() == 1 else hidden_state)
        return float(p.item()) > (1.0 - self._probe_epsilon)

    def get_probe_action(self) -> int | None:
        """Get the action to test (from the least-tested hypothesis).

        Allows re-testing the same hypothesis multiple times to build
        statistical confidence.
        """
        if not self._hypotheses:
            return None
        # Pick the hypothesis with the fewest tests
        h = min(self._hypotheses, key=lambda x: x.test_count)
        self._active_hypothesis_id = h.id
        return h.predicted_action

    def feedback(self, result: float, decay: float = 0.9) -> None:
        """Update the active hypothesis with the test result."""
        if self._active_hypothesis_id is None:
            return
        for h in self._hypotheses:
            if h.id == self._active_hypothesis_id:
                h.update(result, decay)
                break
        self._active_hypothesis_id = None

    # --- 2026-09-22: probe-outcome learning (durable scoring + probe_net) ---

    @property
    def probe_learning_summary(self) -> dict:
        """Probe-gate learning state (for logs / ckpt diagnostics)."""
        out = {
            "probe_updates": int(self._probe_updates),
            "probe_samples": len(self._probe_samples),
            "probe_loss": round(float(self._probe_loss_last), 4),
        }
        if self._probe_last_error:
            out["probe_error"] = self._probe_last_error
        return out

    def probe_epsilon(self) -> float:
        """Exploration rate for the learned gate.

        High early (keeps the rehearsal alive while probe_net is untrained),
        decaying with accumulated updates so the learned policy takes over.
        Warmup 0.9 ≈ the old scripted always-probe, so behaviour stays
        continuous at the switchover; floor 0.05 keeps minimal exploration.
        """
        return max(0.05, 0.9 * (1.0 - min(1.0, self._probe_updates / 2000.0)))

    def begin_probe_tracking(
        self,
        hidden_state: torch.Tensor,
        last_known: tuple[float, float],
        step: int,
    ) -> None:
        """Remember what the active probe predicts (durable across steps)."""
        self._active_hidden = hidden_state.detach().cpu()
        self._active_lk = (float(last_known[0]), float(last_known[1]))
        self._active_step = int(step)

    def check_probe_outcome(
        self,
        agent_xy: tuple[float, float],
        revealed: bool,
        step: int,
        arrival_radius: float = 1.2,
        deadline_steps: int = 30,
    ) -> float | None:
        """Score the active probe: 1.0 arrived, 0.0 failed, None pending.

        Arrival (reaching the last-known position within ``arrival_radius``)
        counts as success - before or at the reveal; a reveal while still far,
        or exceeding the deadline, counts as failure. 到达=成功; 未到即
        reveal/超时=失败。On a terminal outcome the hypothesis feedback is
        applied, probe_net takes one gradient step, and the probe is cleared.
        """
        if self._active_hypothesis_id is None or self._active_lk is None:
            return None
        d = math.hypot(agent_xy[0] - self._active_lk[0],
                       agent_xy[1] - self._active_lk[1])
        success: float | None = None
        if d < arrival_radius:
            success = 1.0
        elif revealed:
            success = 0.0
        elif self._active_step is not None and step - self._active_step > deadline_steps:
            success = 0.0
        if success is None:
            return None
        # try/finally: a failure in the probe_net update must never wedge the
        # probe state (the first experiment stalled with the active probe
        # stuck set - probing dropped from ~38k to 8 and nothing was learned).
        try:
            self.record_probe_outcome(success)
            self.feedback(success)
        finally:
            self._active_lk = None
            self._active_step = None
            self._active_hidden = None
        return success

    def record_probe_outcome(self, success: float) -> float:
        """Store (hidden, outcome) and take one probe_net gradient step."""
        if self._active_hidden is not None:
            self._probe_samples.append((self._active_hidden, float(success)))
            # take the pending hidden exactly once (repeated appends on a
            # failure path saturated the buffer to 256 in the first run)
            self._active_hidden = None
        if len(self._probe_samples) < 8:
            return 0.0
        try:
            dev = next(self.probe_net.parameters()).device
            hs = torch.stack([s[0] for s in self._probe_samples]).to(dev)
            ys = torch.tensor([s[1] for s in self._probe_samples],
                              dtype=torch.float32, device=dev)
            # enable_grad: this is called from the rollout loop where the
            # caller runs under torch.no_grad() (first live run failed with
            # "element 0 of tensors does not require grad and does not have
            # a grad_fn" - probe_net never trained).
            with torch.enable_grad():
                pred = self.probe_net(hs).squeeze(-1)
                loss = F.binary_cross_entropy(pred.clamp(1e-4, 1.0 - 1e-4), ys)
                self._probe_optim.zero_grad()
                loss.backward()
                self._probe_optim.step()
        except Exception as ex:
            # expose instead of swallowing (§14): the first live run silently
            # wedged here; the message shows up in [hypothesis] stats now.
            self._probe_last_error = f"{type(ex).__name__}: {ex}"[:160]
            return 0.0
        self._probe_last_error = ""
        self._probe_updates += 1
        self._probe_loss_last = float(loss.item())
        return self._probe_loss_last

    # ------------------------------------------------------------------ state

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        sd["_probe_updates"] = int(self._probe_updates)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        sd = dict(state_dict)
        self._probe_updates = int(sd.pop("_probe_updates", 0))
        return super().load_state_dict(sd, strict=strict)

    def get_verified_rules(self, min_confidence: float = 0.7, min_tests: int = 3) -> list[Hypothesis]:
        """Return hypotheses that have been tested enough and have high confidence."""
        return [
            h for h in self._hypotheses
            if h.tested and h.confidence >= min_confidence and h.test_count >= min_tests
        ]

    def summary(self) -> dict:
        verified = self.get_verified_rules()
        return {
            "total_hypotheses": len(self._hypotheses),
            "tested": sum(1 for h in self._hypotheses if h.tested),
            "verified": len(verified),
            "mean_confidence": (
                sum(h.confidence for h in self._hypotheses) / max(1, len(self._hypotheses))
            ),
        }


# =====================================================================
# 2. CounterfactualImagination — "what if I had done X?"
# =====================================================================


@dataclass
class CounterfactualResult:
    """Result of a counterfactual imagination.

    - ``actual_reward``: what actually happened.
    - ``imagined_reward``: what the world model predicts would have happened.
    - ``regret``: imagined - actual (positive = the alternative was better).
    - ``imagined_trajectory``: the latent states along the imagined path.
    """

    actual_reward: float
    imagined_reward: float

    @property
    def regret(self) -> float:
        return self.imagined_reward - self.actual_reward


class CounterfactualImagination(nn.Module):
    """Uses the world model to imagine alternative action sequences.

    Given:
    - The current latent state from RSSM.
    - The actual action taken.
    - An alternative action.

    The module rolls out the world model with the alternative action and
    compares the imagined reward to the actual reward.

    This gives the agent "regret" — the ability to realize "I should have
    gone left instead of right".

    Bounded: rollout length is fixed (max_imagination_steps). No growing
    state. Axiom 1.

    反事实推理：用世界模型想象"如果我当时做了别的会怎样"。
    让智能体有"后悔"能力——"我应该左转才对"。
    """

    def __init__(
        self,
        max_imagination_steps: int = 5,
        num_alternatives: int = 3,
    ) -> None:
        super().__init__()
        self._max_steps = int(max_imagination_steps)
        self._num_alt = int(num_alternatives)

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def imagine_alternative(
        self,
        world_model: Any,  # RSSM
        initial_state: Any,  # RSSMState
        alternative_action_onehot: torch.Tensor,
        rssm_action_dim: int,
    ) -> tuple[list[Any], torch.Tensor]:
        """Roll out the world model with an alternative action.

        Args:
            world_model: the RSSM instance.
            initial_state: the RSSMState at the point of divergence.
            alternative_action_onehot: (1, action_dim) the alternative action.
            rssm_action_dim: dimension of the RSSM action input.

        Returns:
            (trajectory of RSSMStates, imagined_reward)
        """
        trajectory: list[Any] = []
        state = initial_state
        total_reward = torch.zeros(1)

        # Repeat the alternative action for max_steps
        action = alternative_action_onehot.unsqueeze(1)  # (1, 1, action_dim)
        for step in range(self._max_steps):
            try:
                state, _ = world_model.imagine_step(
                    state, action[:, 0, :] if action.dim() == 3 else action,
                )
                trajectory.append(state)
                # Decode predicted obs and estimate reward from reconstruction
                recon = world_model.decode(state)
                # Simple reward proxy: norm of the reconstruction (higher = more "interesting")
                # Real implementation would have a reward head on the RSSM
                step_reward = recon.norm(dim=-1).mean() * 0.01
                total_reward = total_reward + step_reward
            except Exception as exc:
                logger.debug("Counterfactual imagination failed at step %d: %s", step, exc)
                break

        return trajectory, total_reward

    def compute_regret(
        self,
        world_model: Any,
        initial_state: Any,
        actual_action: int,
        actual_reward: float,
        num_actions: int,
    ) -> list[CounterfactualResult]:
        """Compute regret for alternative actions.

        For each action != actual_action, imagine the rollout and compute
        the imagined reward. Return a list of (action, regret) pairs.

        Bounded: num_actions is fixed. max_steps is fixed.
        """
        results: list[CounterfactualResult] = []

        for alt_action in range(num_actions):
            if alt_action == actual_action:
                continue

            # One-hot encode
            alt_onehot = torch.zeros(1, num_actions)
            alt_onehot[0, alt_action] = 1.0

            _, imagined_reward = self.imagine_alternative(
                world_model, initial_state, alt_onehot, num_actions,
            )

            results.append(CounterfactualResult(
                actual_reward=actual_reward,
                imagined_reward=float(imagined_reward.item()),
            ))

        return results


# =====================================================================
# 3. BehaviorCloningHead — learn from expert demonstrations
# =====================================================================


class BehaviorCloningHead(nn.Module):
    """Auxiliary loss head for learning from expert demonstrations.

    Given (obs, expert_action) pairs, adds a cross-entropy loss that
    pushes the policy toward expert behavior. This is NOT a separate
    policy — it's an auxiliary objective added to the PPO loss.

    Loss:
        L_BC = CrossEntropy(policy_logits(obs), expert_action)

    The BC loss is weighted by ``bc_coef`` (typically 0.1-0.5) and
    decayed over training as the agent transitions from imitation to
    self-improvement.

    Bounded: no extra state. Just a loss computation. Axiom 1.

    行为克隆：从专家轨迹学习，加速早期训练。
    """

    def __init__(self, bc_coef: float = 0.3, decay_per_step: float = 1e-6) -> None:
        super().__init__()
        self._bc_coef = float(bc_coef)
        self._decay = float(decay_per_step)
        self._step_count = 0

    @property
    def current_coef(self) -> float:
        """BC coefficient, decaying over time."""
        return self._bc_coef * max(0.0, 1.0 - self._step_count * self._decay)

    def step(self) -> None:
        self._step_count += 1

    def loss(
        self,
        policy_logits: torch.Tensor,
        expert_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BC loss.

        Args:
            policy_logits: (B, num_actions) from the policy network.
            expert_actions: (B,) long tensor of expert action indices.

        Returns:
            Scalar loss tensor (weighted by current_coef).
        """
        bc_loss = F.cross_entropy(policy_logits, expert_actions.to(torch.long))
        return self.current_coef * bc_loss

    def summary(self) -> dict:
        return {
            "bc_coef": self.current_coef,
            "step_count": self._step_count,
            "initial_coef": self._bc_coef,
        }


# =====================================================================
# 4. MetaLearner (MAML-lite) — learn how to learn
# =====================================================================


class MetaLearner(nn.Module):
    """MAML-lite: maintains a meta-averaged set of parameters.

    After each task, the agent stores its post-adaptation parameters.
    The meta-average of all post-adaptation parameter sets becomes the
    initialization for the next task. This gives "learning to learn":
    tasks get easier over time because the meta-init is closer to the
    solution manifold.

    Simplified MAML (no second-order gradients):
    1. Train on task A → get params_A
    2. Train on task B → get params_B
    3. meta_params = average(params_A, params_B, ...)
    4. For task C: initialize from meta_params (not from scratch)

    Bounded: stores a SINGLE meta-averaged state dict (same size as model).
    No per-task accumulation — just an exponential moving average.
    Axiom 1 satisfied (constant VRAM regardless of number of tasks).

    元学习：维护一份 meta-averaged 参数，学新任务越来越快。
    只存一份 EMA 参数，不随任务数增长。
    """

    def __init__(
        self,
        model: nn.Module,
        ema_decay: float = 0.9,
    ) -> None:
        super().__init__()
        self._ema_decay = float(ema_decay)
        self._has_meta = False

        # Meta parameters (EMA of post-task parameters)
        self._meta_params: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._meta_params[name] = p.detach().clone()
        self._has_meta = True

    @property
    def has_meta(self) -> bool:
        return self._has_meta

    def consolidate(self, model: nn.Module) -> None:
        """After finishing a task, update the meta-parameters with EMA.

        Args:
            model: the model after training on the current task.
        """
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self._meta_params and p.requires_grad:
                    self._meta_params[name].mul_(self._ema_decay).add_(
                        p.detach() * (1 - self._ema_decay)
                    )

    def get_meta_init(self) -> dict[str, torch.Tensor]:
        """Return the meta-averaged parameters for initializing the next task."""
        return {k: v.clone() for k, v in self._meta_params.items()}

    def initialize_model(self, model: nn.Module) -> None:
        """Initialize a model with the meta-parameters."""
        if not self._has_meta:
            return
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self._meta_params and p.requires_grad:
                    p.copy_(self._meta_params[name])

    def state_dict(self) -> dict:
        return {
            "ema_decay": self._ema_decay,
            "meta_params": {k: v.cpu() for k, v in self._meta_params.items()},
            "has_meta": self._has_meta,
        }

    def load_state_dict(self, state: dict) -> None:
        self._ema_decay = float(state["ema_decay"])
        self._meta_params = {k: v.clone() for k, v in state["meta_params"].items()}
        self._has_meta = bool(state["has_meta"])

    def summary(self) -> dict:
        total = sum(v.numel() for v in self._meta_params.values())
        return {
            "num_params": len(self._meta_params),
            "total_elements": total,
            "ema_decay": self._ema_decay,
            "has_meta": self._has_meta,
        }
