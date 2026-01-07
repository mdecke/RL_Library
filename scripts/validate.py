import os

import argparse
from tqdm import tqdm as tqdm

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from base_classes.models import Actor
from base_classes.utils import load_config, load_scaler
from scripts.plot import plot_trajectories


parser = argparse.ArgumentParser(description='Train or test TD3 agent on a given environment')
parser.add_argument('--task', type=str, default='Pendulum-v1', help='Gym environment name')
parser.add_argument('--num_envs', type=int, default=1, help='Number of parallel environments')
parser.add_argument('--max_iterations', type=int, default=1000, help='Maximum number of iterations')
parser.add_argument('--path_to_saved_policy', type=str, help='Path to the saved policy')
parser.add_argument('--algorithm', type=str, default='tdn', help='RL algorithm to use')
parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu or cuda)')
parser.add_argument('--video', action='store_true', help='Record video of the trained policy')
parser.add_argument('--video_steps', type=int, default=500, help='Number of steps to record in video')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--plot', action='store_true', help='Plot the trajectories of the agent')
parser.add_argument('--expert_guidance', type=str, choices=["mle","gmm","cnf"], default=None, help='Type of expert used during training (if any)')

args = parser.parse_args()


def main():

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    config_file = os.path.join("configs", f"{args.algorithm}Config.yaml")
    general_cfg = load_config(config_file, args)
    
    log_dir = os.path.join("logs", args.task, args.algorithm, "validation_stats")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    trajs = {"obss":[], "acts":[], "rews":[], "terms":[]}

    env = gym.make_vec(args.task, num_envs=args.num_envs)
    action_dim = env.single_action_space.shape[0] * general_cfg["agent"]["action_history"]
    scaling = general_cfg["agent"].get("preprocess_inputs", None)
    
    # Determine which policy to load based on whether expert guidance was used
    if args.expert_guidance is not None:
        policy_filepath = os.path.join(args.path_to_saved_policy, args.task, args.algorithm, "EGOP_models", "scripted_policy.pt")
        scaler_filepath = os.path.join(args.path_to_saved_policy, args.task, args.algorithm, "EGOP_models", "scripted_obs_preprocessor.pt")
        print("[INFO]: Loading policy trained with expert guidance from EGOP_models")
    else:
        policy_filepath = os.path.join(args.path_to_saved_policy, args.task, args.algorithm, "RL_models", "scripted_policy.pt")
        scaler_filepath = os.path.join(args.path_to_saved_policy, args.task, args.algorithm, "RL_models", "scripted_obs_preprocessor.pt")
        print("[INFO]: Loading policy trained without expert guidance from RL_models")
    
    policy = torch.jit.load(policy_filepath, map_location=args.device)
    policy.eval()
    if scaling is not None:
        print("[INFO]: Loading observation preprocessor scaler")
        scaler = torch.jit.load(scaler_filepath, map_location=args.device)
        scaler.eval()

    
    cumulative_reward = np.zeros((args.num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.num_envs,), dtype=np.int32)

    if args.task == "Pendulum-v1":
        obs, _ = env.reset(seed=args.seed, options={'x_init': np.pi, 'y_init': 8.0})
    else:
        obs, _ = env.reset(seed=args.seed)
    
    for t in range(args.max_iterations):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device)
            if scaling is not None:
                normalized_obs = scaler(obs_tensor)
            else:
                normalized_obs = obs_tensor
            action = policy(normalized_obs)
            action_np = action.cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action_np)
            cumulative_reward += reward
            episode_lengths += 1

            trajs["obss"].append(obs)
            trajs["acts"].append(action_np)
            trajs["rews"].append(reward)
            trajs["terms"].append(terminated)

            print(f"timesteps/num_validation_steps: {t}/{args.max_iterations}")


            if terminated.any() or truncated.any():
                env_reset_idx = np.where(terminated | truncated)[0]
                if env_reset_idx.ndim == 0:
                    env_reset_idx = np.array([env_reset_idx])
                for idx in env_reset_idx:
                    print(f"Episode length (env {idx}): {episode_lengths[idx]}")
                    print(f"Cumulative reward (env {idx}): {cumulative_reward[idx]}")
                    cumulative_reward[idx] = 0.0
                    episode_lengths[idx] = 0
                

    env.close()

    obss_stacked = np.array(trajs["obss"])  # Shape: (num_timesteps, num_envs, obs_dim)
    acts_stacked = np.array(trajs["acts"])  # Shape: (num_timesteps, num_envs, action_dim)
    rews_stacked = np.array(trajs["rews"])  # Shape: (num_timesteps, num_envs)
    terms_stacked = np.array(trajs["terms"]) # Shape: (num_timesteps, num_envs)
    
    # Transpose to (num_envs, num_timesteps, dim) then reshape to (num_envs * num_timesteps, dim)
    obss_array = np.transpose(obss_stacked, (1, 0, 2)).reshape(-1, obss_stacked.shape[2])
    acts_array = np.transpose(acts_stacked, (1, 0, 2)).reshape(-1, acts_stacked.shape[2])
    rews_array = np.transpose(rews_stacked, (1, 0)).reshape(-1)
    terms_array = np.transpose(terms_stacked, (1, 0)).reshape(-1)
    
    data_dict = {}
    
    for i in range(obss_array.shape[1]):
        data_dict[f'obs_{i}'] = obss_array[:, i]
    
    for i in range(acts_array.shape[1]):
        data_dict[f'act_{i}'] = acts_array[:, i]
    
    data_dict['reward'] = rews_array
    data_dict['terminated'] = terms_array
    
    df = pd.DataFrame(data_dict)
    df.to_csv(os.path.join(log_dir, f"seed_{args.seed}.csv"), index=False)

    if args.video:
        print("\n[INFO]: Recording validation video...")
        video_folder = os.path.join(log_dir, "trained_policy_rendering")
    
        env_render = gym.make(args.task, render_mode="rgb_array")
        env_render = RecordVideo(
            env_render, 
            video_folder=video_folder,
            episode_trigger=lambda ep: ep == 0,
            name_prefix=f"seed_{args.seed}"
        )
        
        if args.task == "Pendulum-v1":
            obs, _ = env_render.reset(seed=args.seed, options={'x_init': np.pi, 'y_init': 8.0})
        else:
            obs, _ = env_render.reset(seed=args.seed)
        
        print(f"[INFO]: Recording 1 episode for {args.video_steps} steps to {log_dir}")
        
        for _ in range(args.video_steps):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device)
                if scaling is not None:
                    normalized_obs = scaler(obs_tensor)
                else:
                    normalized_obs = obs_tensor
                action = policy(normalized_obs)
                action_np = action.cpu().numpy()
                obs, reward, terminated, _, _ = env_render.step(action_np)
                if terminated:
                    if args.task == "Pendulum-v1":
                        obs, _ = env_render.reset(options={'x_init': np.pi, 'y_init': 8.0})
                    else:
                        obs, _ = env_render.reset()
        env_render.close()
        print(f"[INFO]: Video saved to {log_dir}")

    if args.plot:
        print("\n[INFO]: Generating trajectory plots...")
        data_array = acts_array.reshape(args.num_envs, -1, action_dim)
        plots_dir = os.path.join("plots", args.task, args.algorithm)
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir, exist_ok=True)
        for env_idx in range(args.num_envs):
            figures = plot_trajectories(data_array[env_idx, :, :], "Action")

        for fig_idx, fig in enumerate(figures):
            filename = f"action_predictions_fig{fig_idx+1}.png" if len(figures) > 1 else "action_predictions.png"
            fig.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved {filename}")

        plt.show()  # Show all figures

        

if __name__ == "__main__":
    main()
