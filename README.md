# RL Library

A PyTorch-based reinforcement learning library for continuous control tasks using Gymnasium environments. This library implements state-of-the-art off-policy RL algorithms with optional expert-guided policy optimization (EGOP) for accelerated learning.

## Supported Algorithms

### Off-Policy Actor-Critic Methods

- **TDn (TD3 with n-step returns)**: Twin Delayed Deep Deterministic Policy Gradient with configurable temporal difference horizons. Reduces overestimation bias through double Q-learning and delayed policy updates.

- **DDPG**: Deep Deterministic Policy Gradient for continuous control. Uses deterministic policy gradient with experience replay and target networks.

- **SAC** *(in development)*: Soft Actor-Critic with entropy regularization for robust exploration.

**Key Features:**
- Parallel environment support for efficient data collection
- Observation preprocessing with running standardization
- Configurable network architectures and hyperparameters
- TensorBoard integration for real-time monitoring
- GPU/CPU support

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd RL_Library

# Install in editable mode
pip install -e .

# Optional: Install with video recording support
pip install -e .[video]
```

## Complete Workflow

This library supports two training paradigms:

### 1. Standard RL Training (Baseline)

Train an RL agent from scratch to collect expert demonstration data.

### 2. Expert-Guided RL (EGOP)

Accelerate learning by using behavioral cloning models to guide exploration during RL training.

---

## Step-by-Step Usage

### Step 1: Train Baseline Agent (Data Collection)

First, train a standard RL agent without expert guidance. This simulates collecting expert demonstrations.

```bash
python scripts/train.py \
    --task Ant-v5 \
    --algorithm tdn \
    --num_envs 10 \
    --max_iterations 100000 \
    --device cpu \
    --seed 42
```

**Arguments:**
- `--task`: Gymnasium environment name (e.g., `Ant-v5`, `Hopper-v5`, `Pendulum-v1`)
- `--algorithm`: RL algorithm (`tdn`, `ddpg`)
- `--num_envs`: Number of parallel environments for faster data collection
- `--max_iterations`: Total training steps
- `--device`: `cpu` or `cuda`
- `--seed`: Random seed for reproducibility

**Output:**
- Trained policy: `saved/{task}/{algorithm}/RL_models/best_policy.pth`
- Training logs: `logs/{task}/{algorithm}/training_stats/seed_{seed}.csv`
- TensorBoard logs: `logs/tensorboard/{task}/{algorithm}/`

### Step 2: Validate and Generate Expert Data

Validate the trained policy and record trajectories to use as expert demonstrations.

```bash
python scripts/validate.py \
    --task Ant-v5 \
    --algorithm tdn \
    --num_episodes 100 \
    --video \
    --video_steps 500 \
    --device cpu
```

**Arguments:**
- `--num_episodes`: Number of validation episodes to run
- `--video`: Enable video recording of policy rollouts
- `--video_steps`: Maximum steps per video episode

**Output:**
- Validation metrics: `logs/{task}/{algorithm}/validation_stats/seed_{seed}.csv`
- Rendered videos: `logs/{task}/{algorithm}/validation_stats/trained_policy_rendering/`
- Expert trajectories stored internally for fitting

### Step 3: Fit Expert Model

Train a behavioral cloning model on the collected expert data. Three model types are supported:

```bash
# Maximum Likelihood Estimation (fastest, simplest)
python scripts/expert_fitting.py \
    --task Ant-v5 \
    --expert_type mle \
    --device cpu

# Gaussian Mixture Model (captures multimodal behavior)
python scripts/expert_fitting.py \
    --task Ant-v5 \
    --expert_type gmm \
    --device cpu

# Conditional Normalizing Flow (most expressive, slowest)
python scripts/expert_fitting.py \
    --task Ant-v5 \
    --expert_type cnf \
    --device cpu
```

**Expert Model Types:**

| Type | Description | Pros | Cons |
|------|-------------|------|------|
| **MLE** | Single Gaussian per state | Fast training, simple | Cannot model multimodal policies |
| **GMM** | Mixture of Gaussians | Handles multimodality | More parameters to tune |
| **CNF** | Normalizing flow | Most expressive, flexible | Slowest training, complex |

**Output:**
- Expert model: `saved/{task}/expert/{expert_type}_expert.pth`
- Preprocessor: `saved/{task}/expert/obs_preprocessor.pth`
- Training plots: `plots/{task}/expert/{expert_type}/`

### Step 4: Train with Expert Guidance (EGOP)

Train a new RL agent with expert-guided exploration for faster convergence.

```bash
python scripts/train.py \
    --task Ant-v5 \
    --algorithm tdn \
    --expert_guidance mle \
    --num_envs 10 \
    --max_iterations 100000 \
    --device cpu \
    --seed 42
```

**How Expert Guidance Works:**

The expert model guides action selection during training through adaptive mixing:

```
action = (1 - η) × π_RL(s) + η × π_expert(s)
```

Where:
- `η` (eta) is dynamically adjusted based on episode performance
- High `η` → more expert influence (used when performance is poor)
- Low `η` → more RL policy influence (used when performance is good)

**Eta Smoothing:**
- Exponential moving average (EMA) prevents rapid oscillations
- Rate limiting prevents sudden changes in guidance weight
- Automatically logged to TensorBoard as `Guidance/Eta`

**Output:**
- Expert-guided policy: `saved/{task}/{algorithm}/EGOP_models/best_policy.pth`
- Training logs: `logs/{task}/{algorithm}/guided_training_stats/seed_{seed}.csv`

### Step 5: Compare Results

Generate comparison plots to evaluate expert guidance effectiveness.

```bash
python scripts/plot.py \
    --task Ant-v5 \
    --algorithm tdn \
    --smoothing_window 100 \
    --show
```

**Arguments:**
- `--smoothing_window`: Moving average window for curve smoothing
- `--show`: Display plots interactively (omit to save only)

**Output:**
- Comparison plot: `plots/{task}/{algorithm}/training_comparison.svg`
- Shows side-by-side returns, policy loss, and Q-loss for both methods

---

## Monitoring Training

### TensorBoard

Launch TensorBoard to monitor training in real-time:

```bash
tensorboard --logdir=logs/tensorboard
```

Open http://localhost:6006 in your browser.

**Key Metrics:**

| Metric | Description | Goal |
|--------|-------------|------|
| `Episode/Return` | Cumulative episode reward | Higher is better |
| `Episode/Length` | Steps per episode | Task-dependent |
| `Loss/Policy` | Actor network loss | Should decrease |
| `Loss/Critic` | Q-function loss | Should stabilize |
| `Training/Mean_Q_Value` | Average Q-value estimates | Tracks value learning |
| `Guidance/Eta` | Expert influence weight (EGOP only) | Should decrease as policy improves |

---

## Configuration

Algorithm and training parameters are defined in YAML config files in `configs/`:

- `TDnConfig.yaml`: TD3/TDn algorithm settings
- `DdpgConfig.yaml`: DDPG algorithm settings
- `ExpertConfig.yaml`: Expert model architectures and training (if exists)

**Example Configuration (TDnConfig.yaml):**

```yaml
models:
  policy:
    lr: 0.0003
    hidden_layers: [512, 512, 256, 128]
  
agent:
  discount_factor: 0.99
  policy_delay: 2
  gradient_clip: 0.5
  
training:
  max_iterations: 100000
  warm_up: 1000
  random_steps: 1000
```

---

## Directory Structure

```
RL_Library/
├── agents/                    # RL algorithm implementations
├── base_classes/              # Shared components (models, memory, utils)
├── configs/                   # YAML configuration files
├── scripts/                   # Training and evaluation scripts
├── saved/                     # Trained models
│   └── {task}/
│       ├── {algorithm}/
│       │   ├── RL_models/     # Baseline trained policies
│       │   └── EGOP_models/   # Expert-guided policies
│       └── expert/            # Expert behavioral models
├── logs/                      # Training logs and metrics
│   ├── {task}/{algorithm}/
│   │   ├── training_stats/
│   │   ├── guided_training_stats/
│   │   └── validation_stats/
│   └── tensorboard/           # TensorBoard event files
└── plots/                     # Generated comparison plots
```

---

## Tips for Best Results

### For Baseline Training:
- Use 10+ parallel environments for faster data collection
- Train for at least 50k-100k steps for complex tasks
- Monitor TensorBoard to detect convergence
- Save multiple seeds for robustness analysis

### For Expert Fitting:
- Use 50-100 validation episodes for sufficient data
- Start with MLE for quick prototyping
- Use GMM if expert policy is multimodal (e.g., multiple strategies)
- Use CNF for maximum expressiveness on complex tasks

### For Expert-Guided Training:
- Ensure expert model is well-fitted (low validation loss)
- Compare with baseline using same seeds
- Monitor `Guidance/Eta` to verify adaptive mixing
- Expect 1.5-3x faster convergence with good expert models

---

## Citation

If you use this library in your research, please cite:

```bibtex
@software{rl_library,
  title={RL Library: Expert-Guided Reinforcement Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/RL_Library}
}
```

---

## License

See [LICENSE](LICENSE) for details.
