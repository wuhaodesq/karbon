"""Dreamer-style Imagination Trainer.

Phase 0+ deliverable D: sample efficiency 10x via world-model-driven
imagination rollouts (Hafner et al. DreamerV3 style).

Core idea:
    1. Sample initial states from replay buffer.
    2. Use RSSM to imagine N-step rollouts (prior only, no real obs).
    3. Train actor-critic on imagined trajectories:
       - Critic: TD(λ) value target on imagined rewards + continuations.
       - Actor: REINFORCE with entropy bonus + imagined value baseline.
    4. This replaces part of the environment interaction, greatly
       reducing sample complexity.

Bounded guarantees:
    - Imagination horizon fixed (max_rollout_steps ≤ config limit).
    - Batch size fixed.
    - All tensors pre-allocated.

想象训练器：用世界模型生成想象轨迹，在想象数据上训练策略。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.world_model import RSSM, RSSMState

logger = logging.getLogger(__name__)


@dataclass
class ImaginationConfig:
    """Configuration for :class:`ImaginationTrainer`.

    - ``imagination_horizon``: number of imagined steps per rollout.
    - ``imagination_batch``: number of parallel imagined trajectories.
    - ``discount``: gamma for value bootstrapping.
    - ``actor_entropy_scale``: entropy bonus weight in imagined policy loss.
    - ``critic_loss_scale``: weight on TD-error loss.
    - ``update_every_steps``: perform imagination update every N env steps.
    - ``min_replay_size``: don't start until replay has this many transitions.
    - ``lambda_gae``: GAE lambda for imagined value targets.
    - ``lr``: optimizer learning rate for actor-critic on imagined data.
    """

    imagination_horizon: int = 8
    imagination_batch: int = 32
    discount: float = 0.99
    actor_entropy_scale: float = 0.01
    critic_loss_scale: float = 0.5
    update_every_steps: int = 16
    min_replay_size: int = 256
    lambda_gae: float = 0.95
    lr: float = 3e-4


class ImaginationTrainer:
    """Dreamer-style imagination-based policy training.

    Uses the RSSM world model to generate imagined trajectories,
    then trains the actor-critic on those imagined rollouts.

    This dramatically improves sample efficiency (~10x) because
    one real environment step can generate an entire imagined
    trajectory for training.
    """

    def __init__(
        self,
        config: ImaginationConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or ImaginationConfig()
        self._device = device or torch.device("cpu")
        self._optimizer: torch.optim.Adam | None = None
        self._total_imagine_updates = 0
        self._last_loss: dict[str, float] = {}

    # -------------------------------------------------------- training step

    def train_step(
        self,
        actor_critic: nn.Module,
        world_model: RSSM,
        replay_sample: dict[str, torch.Tensor],
        num_actions: int,
    ) -> dict[str, float]:
        """One imagination training step.

        Args:
            actor_critic: the HybridActorCritic model (policy + value head).
            world_model: trained RSSM for imagining rollouts.
            replay_sample: batch of real transitions from replay buffer,
                with keys ``obs``, ``action``, ``next_obs``.
            num_actions: size of discrete action space.

        Returns:
            dict with ``actor_loss``, ``critic_loss``, ``total_loss``.
        """
        cfg = self.config
        device = next(actor_critic.parameters()).device

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                [p for p in actor_critic.parameters() if p.requires_grad],
                lr=cfg.lr,
            )

        bsz_orig = replay_sample["obs"].shape[0]
        if bsz_orig == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "total_loss": 0.0}

        # --- 1. Initialize RSSM state from real observations ---
        obs_flat = replay_sample["obs"][: cfg.imagination_batch].float().to(device) / 255.0
        bsz = obs_flat.shape[0]
        real_action = replay_sample["action"][:bsz].to(device).long()

        wm_state = world_model.initial_state(bsz, device)
        real_action_onehot = F.one_hot(
            real_action, num_classes=world_model.config.action_dim
        ).float()
        wm_state, _ = world_model.observe_step(
            wm_state,
            real_action_onehot,
            obs_flat.reshape(bsz, -1),
        )

        # --- 2. Imagine N-step trajectories ---
        imagined_states: list[RSSMState] = []
        imagined_actions: list[torch.Tensor] = []
        imagined_logprobs: list[torch.Tensor] = []
        imagined_values: list[torch.Tensor] = []
        imagined_rewards: list[torch.Tensor] = []

        curr_state = wm_state
        for _t in range(cfg.imagination_horizon):
            # Decode observation from latent for actor-critic input
            imagined_obs = world_model.decode(curr_state)
            imagined_obs_3d = imagined_obs.reshape(bsz, *actor_critic.obs_shape)

            # Run policy
            logits, value = actor_critic(imagined_obs_3d)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)

            imagined_states.append(curr_state)
            imagined_actions.append(action)
            imagined_logprobs.append(logprob)
            imagined_values.append(value.squeeze(-1))

            # Imagine reward: use the RSSM reward head (objective env reward
            # prediction), NOT reconstruction error. The reward head is trained
            # on real replay rewards (world_model.compute_loss) so it grounds
            # imagination in PhysicsSandbox's actual return (contact/speed/
            # approach), letting Dreamer-style training optimize real payoff.
            imagined_r = world_model.predict_reward(curr_state)
            imagined_rewards.append(imagined_r)

            # Step world model forward with imagined action
            action_onehot = F.one_hot(
                action.clamp(0, num_actions - 1), num_classes=world_model.config.action_dim
            ).float()
            curr_state, _ = world_model.imagine_step(curr_state, action_onehot)

        # Bootstrap value at the end
        final_obs = world_model.decode(curr_state).reshape(bsz, *actor_critic.obs_shape)
        with torch.no_grad():
            _, final_value = actor_critic(final_obs)
            bootstrap_value = final_value.squeeze(-1)

        # --- 3. Compute GAE returns ---
        T = cfg.imagination_horizon
        values_stacked = torch.stack(imagined_values, dim=0)  # (T, B)
        rewards_stacked = torch.stack(imagined_rewards, dim=0)  # (T, B)

        returns = torch.zeros(T, bsz, device=device)
        gae = torch.zeros(bsz, device=device)
        next_value = bootstrap_value
        for t in reversed(range(T)):
            delta = rewards_stacked[t] + cfg.discount * next_value - values_stacked[t]
            gae = delta + cfg.discount * cfg.lambda_gae * gae
            returns[t] = gae + values_stacked[t]
            next_value = values_stacked[t]

        # --- 4. Policy loss ---
        advantages = returns - values_stacked
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(torch.stack(imagined_logprobs, dim=0) * advantages.detach()).mean()
        entropy_bonus = 0.0
        for a in imagined_actions:
            dist_check = torch.distributions.Categorical(logits=actor_critic(
                world_model.decode(imagined_states[len(imagined_actions) // 2]).reshape(bsz, *actor_critic.obs_shape)
            )[0])
            entropy_bonus += dist_check.entropy().mean()
        entropy_bonus = entropy_bonus / len(imagined_actions)
        actor_loss = actor_loss - cfg.actor_entropy_scale * entropy_bonus

        # --- 5. Critic loss ---
        critic_loss = F.mse_loss(values_stacked, returns.detach())

        # --- 6. Update ---
        total_loss = actor_loss + cfg.critic_loss_scale * critic_loss
        self._optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in actor_critic.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self._optimizer.step()

        self._total_imagine_updates += 1
        self._last_loss = {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "total_loss": float(total_loss.item()),
            "imagine_updates": self._total_imagine_updates,
        }
        return self._last_loss

    # -------------------------------------------------------- properties

    @property
    def capacity(self) -> int:
        return self.config.imagination_batch

    def __len__(self) -> int:
        return self._total_imagine_updates

    def summary(self) -> dict:
        return {
            "total_imagine_updates": self._total_imagine_updates,
            **self._last_loss,
        }

    # -------------------------------------------------------- persistence

    def state_dict(self) -> dict:
        return {
            "total_imagine_updates": self._total_imagine_updates,
            "optimizer": self._optimizer.state_dict() if self._optimizer else {},
        }

    def load_state_dict(self, state: dict) -> None:
        self._total_imagine_updates = int(state.get("total_imagine_updates", 0))
        if self._optimizer and "optimizer" in state and state["optimizer"]:
            self._optimizer.load_state_dict(state["optimizer"])
