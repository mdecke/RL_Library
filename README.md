# RL Library

A reinforcement learning library for continuous control tasks using PyTorch and Gymnasium.

**Supported Algorithms:** DDPG, TDn

**Expert Guidance:** MLE, GMM, CNF-based expert models for guided policy learning

## Installation

### Prerequisites

- Python 3.8 or higher (tested with Python 3.8-3.13)
- pip (Python package manager)
- git

### Install from GitHub

```bash
# Basic installation
pip install git+https://github.com/mdecke/RL_Library.git

# With video recording support
pip install "git+https://github.com/mdecke/RL_Library.git#egg=RL_Library[video]"

# With development tools
pip install "git+https://github.com/mdecke/RL_Library.git#egg=RL_Library[dev]"

# With all optional dependencies
pip install "git+https://github.com/mdecke/RL_Library.git#egg=RL_Library[all]"
```

### Install for Development

```bash
# Clone the repository
git clone https://github.com/mdecke/RL_Library.git
cd RL_Library

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .

# Or with optional dependencies
pip install -e .[video]  # Video recording
pip install -e .[dev]    # Development tools
pip install -e .[all]    # Everything
```

### Verify Installation

```bash
python -c "import agents; import base_classes; print('Installation successful!')"
```

### Dependencies

**Core (automatically installed):**
- gymnasium>=0.29.0
- torch>=2.0.0
- numpy>=1.24.0
- pandas>=2.0.0
- tqdm>=4.65.0
- pyyaml>=6.0
- torchinfo>=1.8.0
- skrl>=1.0.0
- matplotlib>=3.7.0
- tensorboard>=2.13.0

**Optional:**
- `[video]`: moviepy, imageio, imageio-ffmpeg (requires system ffmpeg)
- `[dev]`: pytest, black, flake8, isort

**Note:** For video recording, install ffmpeg:
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`
- Windows: Download from https://ffmpeg.org/download.html

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
