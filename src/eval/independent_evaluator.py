"""Independent Evaluator — three-dimensional agent scoring.

This is NOT a training reward module.  It is an external examiner that
periodically evaluates the agent on three axes—Curiosity, Homeostatic
Drive, and Task Competence—using pure environment signals (no intrinsic
bonuses).  Scores feed back into the training loop so the curriculum
can self-adjust.

默认权重 好奇心:内驱力:任务 = 2:1:2,可通过 config 的 ``eval_weights`` 覆盖。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..envs.physics_sandbox import PhysicsSandbox

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Evaluation schedule and weighting."""

    eval_every_steps: int = 100_000
    episodes_per_task: int = 20
    max_steps_per_ep: int = 200
    weights: list[float] = field(default_factory=lambda: [2.0, 1.0, 2.0])  # curiosity, drive, task
    # If task_score drops below this, advisory is "increase_task_pressure"
    task_floor: float = 0.15
    # If task_score is below floor for this many consecutive evals,
    # the training loop may reduce intrinsic reward coefficient
    task_pressure_threshold: int = 3
    # Report file (relative to ckpt dir or absolute)
    report_path: str = "eval_scores.jsonl"


@dataclass
class EvalReport:
    step: int
    curiosity: float
    drive: float
    task: float
    total: float
    task_vs_random: float  # ratio of agent mean reward / random mean reward
    advisory: str = ""
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class IndependentEvaluator:
    """Periodic, zero-intrinsic-bonus examiner for the developmental agent."""

    def __init__(self, config: dict[str, Any], device: torch.device):
        ecfg = config.get("independent_eval", {})
        self._cfg = EvalConfig(
            eval_every_steps=int(ecfg.get("eval_every_steps", 100_000)),
            episodes_per_task=int(ecfg.get("episodes_per_task", 20)),
            max_steps_per_ep=int(ecfg.get("max_steps_per_ep", 200)),
            weights=ecfg.get("eval_weights", [2.0, 1.0, 2.0]),
            task_floor=float(ecfg.get("task_floor", 0.15)),
            task_pressure_threshold=int(ecfg.get("task_pressure_threshold", 3)),
            report_path=str(ecfg.get("report_path", "eval_scores.jsonl")),
        )
        self._device = device
        self._history: list[EvalReport] = []
        self._consecutive_below_floor = 0

    # ------------------------------------------------------------------ public

    def should_evaluate(self, step: int, batch_size: int) -> bool:
        """Return True when *step* crosses the next eval boundary.

        Uses boundary-crossing logic because the training loop advances
        ``step`` in jumps of *batch_size* (e.g. 2048), not 1.
        """
        if step <= 0:
            return False
        prev = max(0, step - batch_size)
        every = self._cfg.eval_every_steps
        return (step // every) > (prev // every)

    @property
    def weights(self) -> list[float]:
        return list(self._cfg.weights)

    @property
    def last_report(self) -> EvalReport | None:
        return self._history[-1] if self._history else None

    @property
    def needs_task_pressure(self) -> bool:
        return self._consecutive_below_floor >= self._cfg.task_pressure_threshold

    def evaluate(
        self,
        model: nn.Module,
        drives_module: object | None,
        step: int,
    ) -> EvalReport:
        """Run full evaluation and return a structured report."""
        # ---- curiosity: state-visit diversity ----
        cur = self._measure_curiosity(model)

        # ---- drive: homeostatic satisfaction ----
        drv = self._measure_drive(model, drives_module)

        # ---- task: pure env reward vs random ----
        tsk, vs_random = self._measure_task(model)

        w = self._cfg.weights
        total = (cur * w[0] + drv * w[1] + tsk * w[2]) / sum(w)

        # --- advisory ---
        advisory = ""
        if tsk < self._cfg.task_floor and tsk < max(cur, drv) * 0.5:
            self._consecutive_below_floor += 1
            if self._consecutive_below_floor >= self._cfg.task_pressure_threshold:
                advisory = "increase_task_pressure"
        else:
            self._consecutive_below_floor = 0

        report = EvalReport(
            step=step,
            curiosity=cur,
            drive=drv,
            task=tsk,
            total=total,
            task_vs_random=vs_random,
            advisory=advisory,
        )
        self._history.append(report)
        return report

    def save_report(self, out_dir: str | Path) -> Path:
        """Append last report to the JSONL file in *out_dir*."""
        path = Path(out_dir) / self._cfg.report_path
        path.parent.mkdir(parents=True, exist_ok=True)
        rep = self._history[-1]
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "step": rep.step,
                "curiosity": round(rep.curiosity, 4),
                "drive": round(rep.drive, 4),
                "task": round(rep.task, 4),
                "total": round(rep.total, 4),
                "task_vs_random": round(rep.task_vs_random, 4),
                "advisory": rep.advisory,
                "timestamp": rep.timestamp,
            }, ensure_ascii=False) + "\n")
        return path

    # ----------------------------------------------------------------- private

    @staticmethod
    def _make_env(num_objects: int) -> PhysicsSandbox:
        return PhysicsSandbox(
            num_objects=num_objects,
            seed=0,
            max_episode_steps=200,
            render_size=64,
            gravity=-9.8,
            action_force=50.0,
        )

    @staticmethod
    def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
        t = torch.from_numpy(np.asarray(obs))
        if t.dim() == 3:
            t = t.unsqueeze(0)
        return t.to(device)

    def _rollout_episode(
        self, model: nn.Module, env: PhysicsSandbox, record_states: bool = False,
    ) -> tuple[float, list[np.ndarray] | None]:
        obs = env.reset()
        ep_ret = 0.0
        states: list[np.ndarray] = [] if record_states else None  # type: ignore[assignment]
        for _ in range(self._cfg.max_steps_per_ep):
            if record_states:
                states.append(obs.copy())  # type: ignore[union-attr]
            with torch.no_grad():
                out = model(self._obs_to_tensor(obs, self._device))
            logits = out[0] if isinstance(out, (tuple, list)) else out
            a = int(torch.argmax(logits, dim=-1).item())
            step_out = env.step(a)
            obs = step_out.obs
            ep_ret += float(step_out.reward)
            if step_out.terminated or step_out.truncated:
                break
        return ep_ret, states

    def _random_baseline(self, env: PhysicsSandbox) -> float:
        rets: list[float] = []
        rng = np.random.RandomState(42)
        for _ in range(self._cfg.episodes_per_task):
            obs = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
            ep_ret = 0.0
            for _ in range(self._cfg.max_steps_per_ep):
                a = int(rng.randint(0, env.action_space_n))
                step_out = env.step(a)
                obs = step_out.obs
                ep_ret += float(step_out.reward)
                if step_out.terminated or step_out.truncated:
                    break
            rets.append(ep_ret)
        return float(np.mean(rets))

    # --- dimension scorers ---

    def _measure_curiosity(self, model: nn.Module) -> float:
        """State-visitation diversity across tasks, normalized vs random."""
        env = self._make_env(num_objects=10)
        all_states: list[np.ndarray] = []
        for _ in range(self._cfg.episodes_per_task):
            _, states = self._rollout_episode(model, env, record_states=True)
            if states:
                all_states.extend(states)
        if not all_states:
            return 0.0
        # Flatten each obs and discretise into coarse buckets
        buckets: set[int] = set()
        for s in all_states:
            # Quantise each pixel to 4 levels → 64*64*3/16 ≈ 768-bit hash
            h = hash(tuple((s[::4, ::4, :].mean(axis=-1) // 16).astype(np.int32).ravel()[:64]))
            buckets.add(h)
        # Normalize by episode count (more episodes → more buckets expected)
        diversity = len(buckets) / (self._cfg.episodes_per_task * 10.0)
        return min(1.0, diversity)

    def _measure_drive(self, model: nn.Module, drives_module: object | None) -> float:
        """Fraction of steps where homeostatic drives are satisfied.

        Calls the real ``tick()`` interface with parameters estimated from
        the env snapshot (``read_states()``).  Counts ``is_homeostatic()``
        steps as "satisfied".
        """
        if drives_module is None:
            return 1.0
        env = self._make_env(num_objects=10)
        satisfied_count = 0
        total_steps = 0
        try:
            for _ in range(min(self._cfg.episodes_per_task, 5)):
                obs = env.reset()
                for _ in range(self._cfg.max_steps_per_ep):
                    with torch.no_grad():
                        out = model(self._obs_to_tensor(obs, self._device))
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                    a = int(torch.argmax(logits, dim=-1).item())
                    step_out = env.step(a)
                    obs = step_out.obs
                    total_steps += 1
                    # Build tick() args from env snapshot
                    st = env.read_states()
                    ax, ay = st["agent_pos"]
                    world_half = st.get("world_half", 1.0)
                    danger_level = max(
                        0.0,
                        1.0 - min(abs(ax), abs(ay)) / max(world_half, 0.01),
                    )
                    vx, vy = st["agent_vel"]
                    movement_level = min(1.0, (vx**2 + vy**2) ** 0.5 / 5.0)
                    drives_module.tick(  # type: ignore[union-attr]
                        novelty=0.5,
                        success=bool(step_out.reward > 0.0),
                        caregiver_proximity=0.0,
                        danger_level=danger_level,
                        movement_level=movement_level,
                    )
                    if drives_module.is_homeostatic():  # type: ignore[union-attr]
                        satisfied_count += 1
                    if step_out.terminated or step_out.truncated:
                        break
        except Exception:
            pass
        if total_steps == 0:
            return 0.5
        return satisfied_count / total_steps

    def _measure_task(self, model: nn.Module) -> tuple[float, float]:
        """Pure env reward on 3 task configs, normalised vs random."""
        task_configs = [
            (3, "few"),
            (8, "mid"),
            (18, "crowded"),
        ]
        agent_scores: list[float] = []
        random_scores: list[float] = []
        for n_obj, _name in task_configs:
            env = self._make_env(num_objects=n_obj)
            # Agent
            rets: list[float] = []
            for _ in range(self._cfg.episodes_per_task):
                r, _ = self._rollout_episode(model, env)
                rets.append(r)
            agent_scores.append(float(np.mean(rets)))
            # Random
            random_scores.append(self._random_baseline(env))
        # Normalise each task score as ratio vs random (clamped to 0-2)
        ratios = []
        for a, r in zip(agent_scores, random_scores):
            if r > 0:
                ratios.append(min(2.0, max(0.0, a / r)))
            else:
                ratios.append(1.0 if a >= 0 else 0.0)
        task_score = float(np.mean(ratios))
        vs_random = float(np.mean([a / max(r, 1e-6) for a, r in zip(agent_scores, random_scores)]))
        return task_score, vs_random
