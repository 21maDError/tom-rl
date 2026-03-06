# Torl 🤖

> **Train reinforcement learning models from a single TOML config — zero code required.**

---

## Features

| Category | Options |
|---|---|
| **Algorithms** | DQN · REINFORCE · PPO (actor-critic, GAE) |
| **Environments** | CartPole · MountainCar · GridWorld |
| **Optimizers** | Adam · SGD |
| **Activations** | ReLU · Tanh · Sigmoid · Linear |
| **Extras** | Reward curve chart · checkpointing · early stopping · eval mode |

---

## Installation

### Steps:
1. Download the binary(torl.exe) from the [releases](https://github.com/21maDError/tom-rl/releases) page.
2. Add the binary to your PATH.
3. Cick on the torl.exe. This will open a blue window pop up saying "Windows protected your PC". Click on "More info" and then "Run anyway".
4. Now you can use the torl command in your terminal.

---

## Quickstart

### 1. Generate a config
```bash
torl init --algorithm dqn --env cartpole -o my_config.toml
```

### 2. Train
```bash
torl train -c my_config.toml
```

### 3. Evaluate the saved model
```bash
torl eval -c my_config.toml -m ./models/cartpole_dqn --episodes 10
```

---

## CLI Reference

```
USAGE:
    torl <COMMAND>

COMMANDS:
    train       Train an RL agent from a TOML config
    eval        Evaluate a saved model
    init        Generate a sample config file

OPTIONS (train):
    -c, --config <FILE>    Path to TOML config file
    -o, --output <PATH>    Override model output path
    -v, --verbose          Print per-episode details

OPTIONS (eval):
    -c, --config <FILE>    Path to TOML config file (for environment)
    -m, --model  <PATH>    Path to saved model (.json)
    -e, --episodes <N>     Number of eval episodes [default: 10]

OPTIONS (init):
    -a, --algorithm <ALG>  dqn | reinforce | ppo  [default: dqn]
    -e, --env <ENV>        cartpole | mountain_car | gridworld  [default: cartpole]
    -o, --output <FILE>    Output path  [default: config.toml]
```

---

## TOML Config Reference

```toml
# ── Environment ─────────────────────────────────────────────
[environment]
name              = "cartpole"   # cartpole | mountain_car | gridworld
max_steps         = 500          # max steps per episode
seed              = 42           # reproducibility seed
grid_size         = [5, 5]       # GridWorld only: rows × cols
obstacle_fraction = 0.1          # GridWorld only: fraction of cells as walls

# ── Network architecture ─────────────────────────────────────
[network]
hidden_layers = [128, 64]        # sizes of hidden layers
activation    = "relu"           # relu | tanh | sigmoid | linear

# ── Algorithm (shared) ───────────────────────────────────────
[algorithm]
name          = "dqn"            # dqn | reinforce | ppo
gamma         = 0.99             # discount factor γ
learning_rate = 1e-3             # step size for optimizer
optimizer     = "adam"           # adam | sgd

# ── DQN-specific ─────────────────────────────────────────────
[algorithm.dqn]
epsilon_start     = 1.0          # starting ε (exploration)
epsilon_end       = 0.01         # minimum ε
epsilon_decay     = 0.995        # ε decay per step
buffer_size       = 10000        # replay buffer capacity
batch_size        = 64           # mini-batch size
target_update_freq = 100         # steps between target network syncs
warmup_steps      = 500          # steps before training starts

# ── PPO-specific ─────────────────────────────────────────────
[algorithm.ppo]
clip_epsilon     = 0.2           # PPO surrogate clip range
epochs           = 4             # gradient epochs per update
batch_size       = 64            # mini-batch size per epoch
steps_per_update = 512           # env steps collected per update
gae_lambda       = 0.95          # GAE λ for advantage estimation
entropy_coef     = 0.01          # entropy bonus coefficient
value_coef       = 0.5           # value loss coefficient

# ── REINFORCE-specific ───────────────────────────────────────
[algorithm.reinforce]
normalize_returns = true          # normalise returns to zero mean, unit std
baseline          = "mean"        # none | mean — subtract baseline for variance reduction

# ── Training ─────────────────────────────────────────────────
[training]
episodes      = 600              # total training episodes
log_interval  = 10               # print log every N episodes
eval_interval = 100              # run evaluation every N episodes
eval_episodes = 5                # evaluation episodes per eval run
save_best     = true             # save checkpoint when new best avg is found
target_reward = 475.0            # optional: stop early when avg100 >= target

# ── Output ───────────────────────────────────────────────────
[output]
model_path  = "./models/run1"    # saved as <model_path>.json
format      = "json"             # currently: json
print_curve = true               # print ASCII reward curve at end
```

---

## Environments

### CartPole (`cartpole`)
Classic control problem. Balance a pole on a moving cart.
- **State**: `[cart_pos, cart_vel, pole_angle, pole_vel]`
- **Actions**: `0` = push left, `1` = push right
- **Reward**: +1 per step alive
- **Done**: pole falls > 12°, cart out of bounds, or max steps reached
- **Solving criterion**: avg reward ≥ 475 over 100 episodes

### MountainCar (`mountain_car`)
Drive an underpowered car up a mountain using momentum.
- **State**: `[position, velocity]`
- **Actions**: `0` = push left, `1` = neutral, `2` = push right
- **Reward**: −1 per step; +100 on reaching position ≥ 0.5
- **Done**: goal reached, or max steps

### GridWorld (`gridworld`)
Navigate a grid from start (0,0) to goal (rows-1, cols-1) avoiding walls.
- **State**: one-hot encoding of `(row, col)` → size `rows × cols`
- **Actions**: `0` up, `1` down, `2` left, `3` right
- **Reward**: +10 goal, −0.5 wall collision, −0.1 per step
- **Config**: `grid_size`, `obstacle_fraction`

---

## Algorithms

### DQN (Deep Q-Network)
- Experience replay with a configurable replay buffer
- Target network for stable Bellman targets
- ε-greedy exploration with exponential decay
- Mean-squared Bellman error loss
- Gradient accumulation over mini-batches

### REINFORCE (Monte-Carlo Policy Gradient)
- Full episode rollouts, discounted returns
- Optional mean baseline for variance reduction
- Optional return normalization
- Adam gradient ascent on log π(a|s) · G_t

### PPO (Proximal Policy Optimization)
- Separate actor and critic networks
- GAE (Generalized Advantage Estimation) for advantages
- Clipped surrogate objective (ratio clipping)
- Multiple epochs of mini-batch updates per rollout
- Entropy bonus to encourage exploration
- Value function loss with configurable coefficient

---

## Model Format

Models are saved as JSON files containing:
```json
{
  "algorithm": "dqn",
  "environment": "cartpole",
  "state_size": 4,
  "action_size": 2,
  "training_episodes": 600,
  "best_avg_reward": 487.3,
  "policy_network": {
    "layers": [
      { "weights": [[...]], "biases": [...], "activation": "Relu" },
      ...
    ]
  },
  "value_network": null,
  "metadata": { "epsilon_final": 0.01, "total_steps": 150000 }
}
```

---

## Example Configs

| File | Algorithm | Env | Notes |
|---|---|---|---|
| `examples/cartpole_dqn.toml` | DQN | CartPole | Classic baseline |
| `examples/cartpole_ppo.toml` | PPO | CartPole | Actor-critic, faster convergence |
| `examples/cartpole_reinforce.toml` | REINFORCE | CartPole | Policy gradient |
| `examples/mountaincar_dqn.toml` | DQN | MountainCar | Reward shaping included |
| `examples/gridworld_ppo.toml` | PPO | GridWorld | 6×6 grid with 10% obstacles |

---

## License
MIT