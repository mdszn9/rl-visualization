"""
PPO training for LunarLander-v2 with trajectory export.

Usage:
    python train_ppo.py                          # train + record
    python train_ppo.py --steps 300000           # shorter training
    python train_ppo.py --skip-train             # only record (weights must exist)
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym


# --------------------------------------------------------------------------- #
# Actor-Critic network
# --------------------------------------------------------------------------- #
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, act_dim)
        self.value = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def forward(self, obs):
        h = self.shared(obs)
        return self.policy(h), self.value(h).squeeze(-1)

    def act(self, obs):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(
    env_id: str = "LunarLander-v3",
    total_steps: int = 600_000,
    rollout_steps: int = 2048,
    num_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    device: str = "cpu",
    save_path: str = "ppo_lander.pt",
    seed: int = 0,
) -> "ActorCritic":
    env = gym.make(env_id)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    obs, _ = env.reset(seed=seed)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

    ep_rewards: list[float] = []
    ep_reward = 0.0
    step_count = 0
    iteration = 0
    best_mean = -1e9
    start = time.time()

    while step_count < total_steps:
        iteration += 1
        observations = torch.zeros((rollout_steps, obs_dim), device=device)
        actions = torch.zeros(rollout_steps, dtype=torch.long, device=device)
        log_probs = torch.zeros(rollout_steps, device=device)
        values = torch.zeros(rollout_steps, device=device)
        rewards = torch.zeros(rollout_steps, device=device)
        dones = torch.zeros(rollout_steps, device=device)

        with torch.no_grad():
            for t in range(rollout_steps):
                observations[t] = obs_t
                a, lp, v = model.act(obs_t.unsqueeze(0))
                actions[t] = a
                log_probs[t] = lp
                values[t] = v
                next_obs, r, term, trunc, _ = env.step(int(a.item()))
                rewards[t] = r
                done = term or trunc
                dones[t] = 1.0 if done else 0.0
                ep_reward += r
                if done:
                    ep_rewards.append(ep_reward)
                    ep_reward = 0.0
                    next_obs, _ = env.reset()
                obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            _, last_v = model(obs_t.unsqueeze(0))
            last_v = last_v.squeeze(0)
            step_count += rollout_steps

        # GAE
        advantages = torch.zeros(rollout_steps, device=device)
        returns = torch.zeros(rollout_steps, device=device)
        gae = 0.0
        next_v = last_v
        next_nonterm = 1.0
        for t in reversed(range(rollout_steps)):
            nonterm = 1.0 - dones[t].item()
            delta = rewards[t] + gamma * next_v * next_nonterm - values[t]
            gae = delta + gamma * lam * next_nonterm * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_v = values[t]
            next_nonterm = nonterm
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        inds = np.arange(rollout_steps)
        for _ in range(num_epochs):
            np.random.shuffle(inds)
            for s0 in range(0, rollout_steps, batch_size):
                mb = inds[s0 : s0 + batch_size]
                b_obs = observations[mb]
                b_a = actions[mb]
                b_lp = log_probs[mb]
                b_adv = advantages[mb]
                b_ret = returns[mb]

                logits, new_v = model(b_obs)
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(b_a)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - b_lp)
                s1 = ratio * b_adv
                s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
                pg_loss = -torch.min(s1, s2).mean()
                v_loss = F.mse_loss(new_v, b_ret)
                loss = pg_loss + vf_coef * v_loss - ent_coef * entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()

        if iteration % 2 == 0 or iteration == 1:
            recent = ep_rewards[-30:] if ep_rewards else [0.0]
            mean_r = float(np.mean(recent))
            elapsed = time.time() - start
            fps = step_count / max(elapsed, 1e-9)
            print(
                f"iter {iteration:4d} | steps {step_count:8d} | "
                f"mean_r(30) {mean_r:7.2f} | fps {fps:6.0f} | t {elapsed:5.1f}s"
            )
            if mean_r > best_mean and len(ep_rewards) >= 10:
                best_mean = mean_r
                torch.save(
                    {"model_state": model.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim},
                    save_path,
                )

    # Final save (in case it's the best)
    torch.save(
        {"model_state": model.state_dict(), "obs_dim": int(obs_dim), "act_dim": int(act_dim)},
        save_path + ".final",
    )
    env.close()
    return model


# --------------------------------------------------------------------------- #
# Trajectory recording
# --------------------------------------------------------------------------- #
def extract_terrain(env) -> list[list[float]]:
    """Pull the top surface of the moon out of the env's sky polygons."""
    e = env.unwrapped
    sky_polys = getattr(e, "sky_polys", None)
    if not sky_polys:
        return []
    pts: list[list[float]] = []
    for poly in sky_polys:
        # gymnasium layout: each sky polygon is
        # [chunk_x1, chunk_x2, (chunk_x2, H), (chunk_x1, H)]
        # so poly[0] and poly[1] are terrain points.
        if len(poly) >= 2:
            pts.append([float(poly[0][0]), float(poly[0][1])])
    last = sky_polys[-1]
    if len(last) >= 2:
        pts.append([float(last[1][0]), float(last[1][1])])
    return pts


def extract_helipad(env) -> dict:
    e = env.unwrapped
    return {
        "x1": float(e.helipad_x1),
        "x2": float(e.helipad_x2),
        "y": float(e.helipad_y),
    }


def record_trajectories(
    model,
    env_id: str = "LunarLander-v3",
    n_episodes: int = 6,
    untrained: bool = False,
    seed: int = 1000,
    device: str = "cpu",
    deterministic: bool = True,
) -> list[dict]:
    env = gym.make(env_id)
    episodes: list[dict] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        terrain = extract_terrain(env)
        helipad = extract_helipad(env)
        frames: list[dict] = []
        total_r = 0.0
        landed = False
        crashed = False
        done = False
        step = 0
        max_steps = 1000
        while not done and step < max_steps:
            lander = env.unwrapped.lander
            legs = env.unwrapped.legs
            frame = {
                "x": float(lander.position.x),
                "y": float(lander.position.y),
                "vx": float(lander.linearVelocity.x),
                "vy": float(lander.linearVelocity.y),
                "angle": float(lander.angle),
                "legs": [bool(legs[0].ground_contact), bool(legs[1].ground_contact)],
            }
            if untrained:
                action = int(env.action_space.sample())
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    logits, _ = model(obs_t)
                    if deterministic:
                        action = int(torch.argmax(logits, dim=-1).item())
                    else:
                        action = int(Categorical(logits=logits).sample().item())
            frame["action"] = action
            obs, r, term, trunc, info = env.step(action)
            frame["reward"] = float(r)
            frames.append(frame)
            total_r += r
            step += 1
            done = term or trunc
            if done:
                # Game flag: landed vs crashed is indicated by reward structure;
                # a successful landing gives +100, a crash gives -100.
                landed = bool(r > 50)
                crashed = not landed
                lander = env.unwrapped.lander
                legs = env.unwrapped.legs
                frames.append(
                    {
                        "x": float(lander.position.x),
                        "y": float(lander.position.y),
                        "vx": float(lander.linearVelocity.x),
                        "vy": float(lander.linearVelocity.y),
                        "angle": float(lander.angle),
                        "legs": [bool(legs[0].ground_contact), bool(legs[1].ground_contact)],
                        "action": -1,
                        "reward": 0.0,
                    }
                )
        episodes.append(
            {
                "untrained": untrained,
                "total_reward": total_r,
                "landed": landed,
                "crashed": crashed,
                "terrain": terrain,
                "helipad": helipad,
                "frames": frames,
            }
        )
        print(
            f"  ep {ep + 1:2d}/{n_episodes} | "
            f"{'UNTRAINED' if untrained else 'TRAINED  '} | "
            f"reward {total_r:7.1f} | "
            f"steps {len(frames):3d} | "
            f"{'LANDED' if landed else 'CRASHED'}"
        )
    env.close()
    return episodes


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600_000)
    parser.add_argument("--save", default="ppo_lander.pt")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "public", "trajectories.json"),
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--n-trained", type=int, default=8)
    parser.add_argument("--n-untrained", type=int, default=6)
    args = parser.parse_args()

    if not args.skip_train:
        print("Training PPO on LunarLander-v3 ...")
        train(total_steps=args.steps, save_path=args.save)

    print("\nLoading trained model for evaluation ...")
    ckpt = torch.load(args.save, weights_only=False)
    model = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("\nRecording trained episodes ...")
    trained = record_trajectories(model, n_episodes=args.n_trained, untrained=False, seed=2000)
    print("\nRecording untrained (random-action) episodes ...")
    untrained = record_trajectories(model, n_episodes=args.n_untrained, untrained=True, seed=3000)

    data = {
        "env": "LunarLander-v3",
        "viewport": {"w": 20.0, "h": 13.333},
        "trained": trained,
        "untrained": untrained,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
