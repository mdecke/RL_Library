# RL Library

A reinforcement learning library for continuous control tasks using PyTorch and Gymnasium.

**Supported Algorithms:** TD3, DDPG, SAC, TDn

## Installation

```bash
# Clone and install
git clone <repository-url>
cd RL_Library
pip install -e .

# With video recording support
pip install -e .[video]
```

## Quick Start

### Train

```bash
python scripts/train.py \
    --task Pendulum-v1 \
    --num_envs 10 \
    --max_iterations 7500 \
    --path_to_saved_policy ./saved \
    --algorithm TD3
```

### Validate

```bash
python scripts/validate.py \
    --task Pendulum-v1 \
    --path_to_saved_policy ./saved \
    --algorithm TD3 \
    --video \
    --video_steps 500
```

### Monitor Training

```bash
tensorboard --logdir=logs/tensorboard
```

Open http://localhost:6006 in your browser.

## Key Metrics

- **Episode/Return**: Main performance metric (higher is better)
- **Loss/Policy**: Actor loss
- **Loss/Critic**: Q-function loss

## Output

- **Models**: `saved/{task}/{algorithm}/`
- **Logs**: `logs/{task}/{algorithm}/`
- **Videos**: `logs/{task}/{algorithm}/`

## Wiki

For detailed documentation, configuration options, and advanced usage, see the [Wiki](../../wiki).
