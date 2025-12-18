#!/usr/bin/env bash
set -euo pipefail

TASK="Ant-v5" # Options: Pendulum-v1, HalfCheetah-v4, Ant-v5, Hopper-v4
ALGO="tdn" # Options: ddpg, tdn, sac

# SEEDS=(42 7 2024 123 0 999) # 2023 31415 2718 1618 1)
SEEDS=(42)

for SEED in "${SEEDS[@]}"; do
    echo
    echo "=== Training $TASK with $ALGO (seed=$SEED) ==="
    echo
    python scripts/train.py \
        --task $TASK \
        --num_envs 5 \
        --max_iterations 150000 \
        --path_to_saved_policy ./saved \
        --algorithm $ALGO \
        --seed $SEED \
        --save_method last \
        --expert_guidance cnf \
    
    echo
    echo "=== Validating $TASK with $ALGO (seed=$SEED) ==="
    echo
    python scripts/validate.py \
        --task $TASK \
        --num_envs 3 \
        --max_iterations 1000 \
        --path_to_saved_policy ./saved \
        --algorithm $ALGO \
        --video \
        --video_steps 1000 \
        --seed $SEED \
        --plot \

done

echo
echo "=== Plotting results ==="
python scripts/plot.py \
    --task $TASK \
    --algorithm $ALGO \
    --log_dir ./logs \
    --save_dir ./plots \
    --smoothing_window 20 \
    --show