#!/usr/bin/env bash
set -euo pipefail

TASK="Ant-v5" # Options: Pendulum-v1, HalfCheetah-v4, Ant-v5, Hopper-v4
ALGO="tdn" # Options: ddpg, tdn, sac



echo "=== Expert Modeling ==="
python scripts/expert_fitting.py \
    --expert_type gmm \
    --expert_domain time \
    --expert_data_path ./logs/$TASK/$ALGO/validation_stats \
    --path_to_saved_expert ./saved/$TASK/expert \
    --path_to_figures ./plots/$TASK/expert \