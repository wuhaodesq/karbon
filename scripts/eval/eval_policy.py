#!/usr/bin/env python3
"""Scale-invariant policy evaluation for Stage-3 (runs on remote).

Loads a checkpoint, rolls out N episodes in PhysicsSandbox with GREEDY actions,
and scores each episode under BOTH reward functions:
  * NEW reward  = the env's live _compute_reward (agent-caused accel, cap 3.0)
  * OLD reward  = reproduced here (unconditional speed*0.05, cap 2.0)
plus behavior metrics that don't depend on reward scale:
  * mean distinct objects contacted per episode
  * mean total object |Δv| caused while in contact (active pushing)
  * mean agent path length (movement)

This lets us compare a policy trained on NEW reward (v10) against one trained on
OLD reward (no_planner) on the SAME rulers, answering: did the reward redesign
make the agent behave better, or just change the scoreboard?
"""
import argparse, sys
import numpy as np
import torch

sys.path.insert(0, "/root/karbon")
from src.envs.physics_sandbox import PhysicsSandbox
from src.utils.ckpt import load_ckpt
from src.train import HybridActorCritic


def old_reward(env) -> float:
    """Reproduce the PRE-v10 reward (unconditional speed*0.05, cap 2.0)."""
    reward = 0.0
    agent = env._agent
    contact_count = 0
    for i, obj in enumerate(env._objects):
        dx = agent.x - obj.x
        dy = agent.y - obj.y
        dist = float(np.sqrt(dx * dx + dy * dy))
        touch_dist = agent.radius + obj.radius
        if dist < touch_dist + 0.02:
            contact_count += 1
        speed = float(np.sqrt(obj.vx * obj.vx + obj.vy * obj.vy))
        reward += speed * 0.05
        prev = env._prev_distances.get(i, dist)
        if dist < prev:
            reward += (prev - dist) * 0.2
    reward += contact_count * 0.1
    wall_dist = env._hw - max(abs(agent.x), abs(agent.y), agent.radius + 0.05)
    if wall_dist < 0.1:
        reward -= (0.1 - wall_dist) * 0.5
    return float(max(-0.5, min(2.0, reward)))


def behavior_metrics(env, prev_obj_v):
    """Distinct-contact count this step + total agent-caused |dv| while touching."""
    agent = env._agent
    contacts = 0
    active_dv = 0.0
    for i, obj in enumerate(env._objects):
        dx = agent.x - obj.x
        dy = agent.y - obj.y
        dist = float(np.sqrt(dx * dx + dy * dy))
        touch = agent.radius + obj.radius
        if dist < touch + 0.02:
            contacts += 1
            pv = prev_obj_v.get(i, (obj.vx, obj.vy))
            active_dv += abs(obj.vx - pv[0]) + abs(obj.vy - pv[1])
    return contacts, active_dv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--num-objects", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--greedy", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    # Fixed RNG so stochastic sampling is reproducible across ckpts (variance ctrl).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = load_ckpt(args.ckpt)

    # Discover layer count from state dict to build matching model
    sd = payload["model_state"]
    layer_ids = set()
    for k in sd:
        if "backbone.blocks." in k:
            try:
                layer_ids.add(int(k.split("backbone.blocks.")[1].split(".")[0]))
            except (IndexError, ValueError):
                pass
    n_layers = (max(layer_ids) + 1) if layer_ids else 7

    obs_shape = (args.max_steps, 64, 64, 3)  # only H,W,C matter for encoder
    model = HybridActorCritic(
        obs_shape=(64, 64, 3),
        num_actions=8,
        d_model=128, n_layers=n_layers, n_heads=4,
        swa_window=16, ttt_mini_batch=8, ffn_hidden_mult=4, dropout=0.0,
        use_vision_encoder=False,
        use_slot_attention=True, slot_num_slots=7, slot_dim=128,
        slot_num_iterations=3,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f"[eval] loaded {args.ckpt} (n_layers={n_layers}, step={payload.get('step')})", flush=True)

    ep_new, ep_old, ep_contacts, ep_activedv, ep_path = [], [], [], [], []
    for ep in range(args.episodes):
        obs = env_reset(args, ep)
        env = _ENV  # set by env_reset
        tot_new = tot_old = tot_contacts = tot_activedv = path = 0.0
        prev_obj_v = {i: (o.vx, o.vy) for i, o in enumerate(env._objects)}
        prev_ax, prev_ay = env._agent.x, env._agent.y
        done = False
        while not done:
            obs_t = torch.from_numpy(np.asarray(obs)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(obs_t)
                action = int(logits.argmax(dim=-1).item()) if args.greedy \
                    else int(torch.distributions.Categorical(logits=logits).sample().item())
            r_old = old_reward(env)  # pre-step-integration snapshot uses same env state as env does
            # object velocities + contact set BEFORE the step
            v_before = [(o.vx, o.vy) for o in env._objects]
            contacts_before = []
            ax, ay, arad = env._agent.x, env._agent.y, env._agent.radius
            for o in env._objects:
                d = ((ax - o.x) ** 2 + (ay - o.y) ** 2) ** 0.5
                contacts_before.append(d < (arad + o.radius) + 0.02)
            c = sum(contacts_before)
            step = env.step(action)
            # true agent-caused |dv| = velocity change of objects that were in contact
            adv = 0.0
            for i, o in enumerate(env._objects):
                if contacts_before[i]:
                    adv += abs(o.vx - v_before[i][0]) + abs(o.vy - v_before[i][1])
            tot_new += step.reward
            tot_old += r_old
            tot_contacts += c
            tot_activedv += adv
            path += abs(env._agent.x - prev_ax) + abs(env._agent.y - prev_ay)
            prev_ax, prev_ay = env._agent.x, env._agent.y
            obs = step.obs
            done = step.terminated or step.truncated
        ep_new.append(tot_new); ep_old.append(tot_old)
        ep_contacts.append(tot_contacts); ep_activedv.append(tot_activedv); ep_path.append(path)

    def stat(a):
        a = np.array(a); return a.mean(), a.std()
    print("\n===== EVAL RESULTS (%d episodes, greedy) =====" % args.episodes)
    for name, arr in [("NEW reward (env, cap3.0)", ep_new),
                      ("OLD reward (repro, cap2.0)", ep_old),
                      ("contacts/ep (step-sum)", ep_contacts),
                      ("active |dv|/ep (pushing)", ep_activedv),
                      ("agent path len/ep", ep_path)]:
        m, s = stat(arr)
        print("  %-28s mean=%.3f  std=%.3f" % (name, m, s))


_ENV = None
def env_reset(args, ep):
    global _ENV
    _ENV = PhysicsSandbox(
        num_objects=args.num_objects, seed=args.seed + ep,
        max_episode_steps=args.max_steps, render_size=64,
        gravity=-9.8, action_force=50.0,
    )
    return _ENV.reset(seed=args.seed + ep)


if __name__ == "__main__":
    main()
