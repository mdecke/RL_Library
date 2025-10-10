#!/usr/bin/env bash
set -euo pipefail

python scripts/train.py \
    --task Pendulum-v1 \
    --num_envs 10 \
    --max_iterations 7500 \
    --path_to_saved_policy ./saved \
    --algorithm TDn

python scripts/validate.py \
    --task Pendulum-v1 \
    --num_envs 10 \
    --max_iterations 1000 \
    --path_to_saved_policy ./saved \
    --algorithm TDn \
    --video \
    --video_steps 300

