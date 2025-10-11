#!/usr/bin/env bash
set -euo pipefail

TASK="Pendulum-v1"

python scripts/train.py \
    --task $TASK \
    --num_envs 10 \
    --max_iterations 7500 \
    --path_to_saved_policy ./saved \
    --algorithm ddpg \

python scripts/validate.py \
    --task $TASK \
    --num_envs 10 \
    --max_iterations 1000 \
    --path_to_saved_policy ./saved \
    --algorithm ddpg \
    --video \
    --video_steps 300

