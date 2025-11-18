# RL Library

A reinforcement learning library for continuous control tasks using PyTorch and Gymnasium.

**Supported Algorithms:** DDPG, TDn

**Expert Guidance:** MLE, GMM, CNF-based expert models for guided policy learning

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
    --algorithm tdn
```

### Train with Expert Guidance

First, fit an expert model on demonstration data:

```bash
python scripts/expert_fitting.py \
    --task Ant-v5 \
    --expert_type mle
```

Then train with expert guidance:

```bash
python scripts/train.py \
    --task Ant-v5 \
    --algorithm tdn \
    --expert_guidance mle
```

**Expert Types:** `mle` (Maximum Likelihood), `gmm` (Gaussian Mixture Model), `cnf` (Conditional Normalizing Flow)

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
- **Guidance/Eta**: Expert guidance weight (0=full expert, 1=full RL policy)

## Output

- **RL Models**: `saved/{task}/{algorithm}/RL_models/`
- **Expert-Guided Models**: `saved/{task}/{algorithm}/EGOP_models/`
- **Expert Models**: `saved/{task}/expert/`
- **Logs**: `logs/{task}/{algorithm}/training_stats/` or `/guided_training_stats/`
- **Videos**: `logs/{task}/{algorithm}/validation_stats/`

## Wiki

For detailed documentation, configuration options, and advanced usage, see the [Wiki](../../wiki).
