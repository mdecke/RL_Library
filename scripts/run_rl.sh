#!/usr/bin/env bash
set -euo pipefail

TASK="Ant-v5" # Options: Pendulum-v1, HalfCheetah-v4, Ant-v5, Hopper-v4
ALGO="tdn" # Options: ddpg, tdn, sac
EXPERT_GUIDANCE="" # Options: mle, gmm, cnf, or leave empty for no expert guidance

# SEEDS=(42 7 2024 123 0 999) # 2023 31415 2718 1618 1)
SEEDS=(42)

for SEED in "${SEEDS[@]}"; do
    echo
    echo "=== Training $TASK with $ALGO (seed=$SEED) ==="
    echo
    
    # Build the training command
    TRAIN_CMD="python scripts/train.py \
        --task $TASK \
        --num_envs 10 \
        --max_iterations 100000 \
        --path_to_saved_policy ./saved \
        --algorithm $ALGO \
        --seed $SEED \
        --save_method last"
    
    # Add expert guidance if specified
    if [ -n "$EXPERT_GUIDANCE" ]; then
        TRAIN_CMD="$TRAIN_CMD --expert_guidance $EXPERT_GUIDANCE"
    fi
    
    # Execute training
    # eval $TRAIN_CMD
    
    echo
    echo "=== Validating $TASK with $ALGO (seed=$SEED) ==="
    echo
    
    # Build the validation command
    VAL_CMD="python scripts/validate.py \
        --task $TASK \
        --num_envs 3 \
        --max_iterations 1000 \
        --path_to_saved_policy ./saved \
        --algorithm $ALGO \
        --video \
        --video_steps 1000 \
        --seed $SEED \
        --plot"
    
    # Add expert guidance flag if specified (to load correct model)
    if [ -n "$EXPERT_GUIDANCE" ]; then
        VAL_CMD="$VAL_CMD --expert_guidance $EXPERT_GUIDANCE"
    fi
    
    # Execute validation
    eval $VAL_CMD

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