# PPO training for LunarLander-v2

Trains a PPO agent on `LunarLander-v2` with PyTorch and records trajectories for
the JS demo.

## Setup

Dependencies (already present on this machine):
- Python 3.12
- `torch`
- `gymnasium[box2d]`

If starting fresh on another machine:

```bash
brew install swig              # Box2D needs swig
python3 -m venv .venv
source .venv/bin/activate
pip install torch "gymnasium[box2d]" numpy
```

## Train + record

```bash
python train_ppo.py --steps 600000
```

This:
1. Trains PPO for ~600K environment steps (takes ~15–25 min on a Mac CPU).
2. Saves the best checkpoint to `ppo_lander.pt`.
3. Records 8 trained episodes (deterministic argmax policy) and 6 untrained
   (uniform-random) episodes.
4. Writes them to `../public/trajectories.json` where the JS demo can load them.

### Options

- `--steps N` — total env steps (default 600k).
- `--skip-train` — only record episodes from an existing `ppo_lander.pt`.
- `--out PATH` — where to write the trajectories JSON.

## PPO settings

Standard: 2-layer MLP (64 hidden), Tanh, orthogonal init. GAE(γ=0.99, λ=0.95),
clip ε=0.2, 10 PPO epochs per rollout, minibatch 64, Adam lr=3e-4, entropy
coefficient 0.01, grad norm clipped at 0.5.

LunarLander-v2 is "solved" at ~200 mean reward over 100 episodes. 600K steps is
enough to get well past that on CPU.
