import os

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym

import torch
from torch.utils.data import DataLoader, TensorDataset

from agents import create_expert
from base_classes.utils import load_config, make_data_frame, evaluate_expert_rollout
from plot import plot_prediction_accuracy, plot_action_predictions, plot_rollout_rewards


def augment_boundaries_combined(obss, acts, boundary_threshold=0.8):
    is_boundary = (torch.abs(acts) > boundary_threshold).any(dim=-1)
    
    boundary_obss = obss[is_boundary]
    boundary_acts = acts[is_boundary]
    
    n_boundary = len(boundary_obss)
    print(f"\n[INFO] Boundary Augmentation:")
    print(f"       Original dataset: {len(acts)} samples")
    print(f"       Boundary samples (|action| > {boundary_threshold}): {n_boundary} ({100*n_boundary/len(acts):.1f}%)")
    
    augmented_obss_list = [obss]
    augmented_acts_list = [acts]
    
    augmented_obss_list.append(boundary_obss)
    augmented_acts_list.append(boundary_acts)
    print(f"       + {n_boundary} exact duplicates")
    
    for i in range(3):
        obs_noise = torch.randn_like(boundary_obss) * 0.02  
        act_noise = torch.randn_like(boundary_acts) * 0.01  
        
        noisy_obss = boundary_obss + obs_noise
        noisy_acts = torch.clamp(boundary_acts + act_noise, -1.0, 1.0) #TODO: generalize to action ranges 
        
        augmented_obss_list.append(noisy_obss)
        augmented_acts_list.append(noisy_acts)
    
    print(f"       + {n_boundary * 3} noisy variations")
    
    augmented_obss = torch.cat(augmented_obss_list, dim=0)
    augmented_acts = torch.cat(augmented_acts_list, dim=0)
    
    perm = torch.randperm(len(augmented_obss))
    augmented_obss = augmented_obss[perm]
    augmented_acts = augmented_acts[perm]
    
    print(f"       Augmented dataset: {len(augmented_obss)} samples")
    print(f"       Boundary representation: {100*n_boundary/len(obss):.1f}% → {100*(n_boundary*5)/len(augmented_obss):.1f}%\n")
    
    return augmented_obss, augmented_acts


parser = argparse.ArgumentParser(description="Fit an expert model to data.")
parser.add_argument("--expert_type", type=str, choices=["mle","gmm","cnf"], default="mle", help="Type of expert model to fit")
parser.add_argument("--expert_domain", type=str, choices=["time","frequency"], default="time", help="Domain in which to fit the expert model")
parser.add_argument("--expert_data_path", type=str, required=True, help="Path to the training data (CSV format)")
parser.add_argument("--path_to_saved_expert", type=str, help="Directory to save the trained expert model")
parser.add_argument("--path_to_figures", type=str, help="Directory to save the prediction accuracy figures")
parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (cpu or cuda)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--env_name", type=str, default="Pendulum-v1", help="Gymnasium environment name for rollout evaluation")
parser.add_argument("--n_eval_episodes", type=int, default=5, help="Number of episodes for rollout evaluation")

args = parser.parse_args()


def main():

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config_file = os.path.join("configs", "ExpertConfig.yaml")
    full_cfg = load_config(config_file, args)
    
    cfg = {k: v for k, v in full_cfg.items() if k not in ['mle', 'gmm', 'cnf']}
    cfg['expert_domain'] = args.expert_domain
    cfg[args.expert_type] = full_cfg[args.expert_type]

    if not os.path.exists(args.path_to_saved_expert):
        os.makedirs(args.path_to_saved_expert, exist_ok=True)

    expert_data = make_data_frame(args.expert_data_path)

    # Extract states and actions
    state_cols = [col for col in expert_data.columns if col.startswith('obs_')]
    action_cols = [col for col in expert_data.columns if col.startswith('act_')]
    
    states = expert_data[state_cols].values.astype(np.float32)
    actions = expert_data[action_cols].values.astype(np.float32)

    cfg["obs_dim"] = states.shape[1]
    cfg["action_dim"] = actions.shape[1]
    
    print(f"Data shape: {states.shape[0]} samples")
    print(f"State dim: {states.shape[1]}")
    print(f"Action dim: {actions.shape[1]}")
    print(f"Action stats: min={actions.min():.3f}, max={actions.max():.3f}, mean={actions.mean():.3f}")
    
    # index = np.arange(len(obss_tensor))
    index = np.arange(len(states))
    np.random.shuffle(index)
    
    test_idx = index[:int(cfg["test_split"] * len(index))]
    val_idx = index[int(cfg["test_split"] * len(index)):int(cfg["validation_split"] * len(index))+int(cfg["test_split"] * len(index))]
    train_idx = index[int(cfg["validation_split"] * len(index))+int(cfg["test_split"] * len(index)):]

    train_obss = torch.tensor(states[train_idx], dtype=torch.float32).to(args.device)
    train_acts = torch.tensor(actions[train_idx], dtype=torch.float32).to(args.device)
    test_obss = torch.tensor(states[test_idx], dtype=torch.float32).to(args.device)
    test_acts = torch.tensor(actions[test_idx], dtype=torch.float32).to(args.device)
    val_obss = torch.tensor(states[val_idx], dtype=torch.float32).to(args.device)
    val_acts = torch.tensor(actions[val_idx], dtype=torch.float32).to(args.device)

    # Apply boundary augmentation if enabled in config
    if cfg.get("augment_boundaries", False):
        boundary_threshold = cfg.get("boundary_threshold", 0.8)
        train_obss, train_acts = augment_boundaries_combined(
            train_obss, train_acts, 
            boundary_threshold=boundary_threshold
        )


    train_data = TensorDataset(train_obss, train_acts)
    train_loader = DataLoader(train_data, batch_size=cfg[f"{args.expert_type}"]["batch_size"], shuffle=True)
    val_data = TensorDataset(val_obss, val_acts)
    val_loader = DataLoader(val_data, shuffle=False)

    test_data = TensorDataset(test_obss, test_acts)
    test_loader = DataLoader(test_data, shuffle=False)

    
    expert = create_expert(args.expert_type, cfg)
    
    if expert.preprocess_inputs:
        with torch.no_grad():
            expert.fit_obs_preprocessor(train_obss)
    
    train_losses, val_losses = expert.train(train_loader, val_loader)
    test_loss = expert.validate(test_loader)
    expert.save(args.path_to_saved_expert)

    expert.eval()
    with torch.no_grad():
        predicted_actions = expert.most_likely_component(test_obss)
    
    plots_dir = os.path.join(args.path_to_figures, args.expert_type)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir, exist_ok=True)
    
    plot_action_predictions(test_acts.to("cpu").numpy(), predicted_actions.to("cpu").numpy(), test_acts.shape[1],plots_dir)
    figures = plot_prediction_accuracy(test_acts.to("cpu").numpy(), predicted_actions.to("cpu").numpy(), test_acts.shape[1])

    for fig_idx, fig in enumerate(figures):
        filename = f"action_predictions_fig{fig_idx+1}.png" if len(figures) > 1 else "action_predictions.png"
        fig.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved {filename}")

    # Evaluate expert with rollout in environment
    print("\n" + "="*50)
    print("Evaluating Expert Policy with Rollouts")
    print("="*50)
    episode_rewards, inst_rewards, cum_rewards = evaluate_expert_rollout(
        expert, args.env_name, n_episodes=args.n_eval_episodes, seed=args.seed
    )
    plot_rollout_rewards(inst_rewards, cum_rewards, plots_dir)

    plt.show()  # Show all figures


if __name__ == "__main__":
    main()
