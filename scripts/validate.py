import os

import argparse
from tqdm import tqdm as tqdm

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import pandas as pd

import torch

from base_classes.models import Actor
from base_classes.utils import load_config, load_scaler


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
    obs_dim = env.single_observation_space.shape[0] * general_cfg["agent"]["state_history"]
    action_dim = env.single_action_space.shape[0] * general_cfg["agent"]["action_history"]
    policy = Actor(input_dim=obs_dim,
                   output_dim=action_dim,
                   action_limit=float(env.single_action_space.high[0]),
                   hidden_dims=general_cfg['models']["policy"]['hidden_layers'],
                   lr=general_cfg['models']["policy"]['lr'],
                   activation_fct=general_cfg['models']["policy"]['activation_fct'],
                   stochastic=general_cfg['models']["policy"]['stochastic'],
                   seed=args.seed).to(args.device)

    print("[INFO]: Loading trained model")
    policy.load(filepath=os.path.join(args.path_to_saved_policy, args.task, args.algorithm, "RL_models", "best_policy.pth"))
    policy.eval()
    scaling = getattr(general_cfg["agent"], "preprocess_inputs", None)
    if scaling is not None:
        scaler = load_scaler(size=obs_dim,
                            scaler_filepath=os.path.join(args.path_to_saved_policy, args.task, args.algorithm, "RL_models", "obs_preprocessor.pth")).to(args.device)

    cumulative_reward = np.zeros((args.num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.num_envs,), dtype=np.int32)

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
                print(f"Episode lengths: {episode_lengths}")
                print(f"Cumulative rewards: {cumulative_reward}")
                obs, _ = env.reset()
                cumulative_reward = np.zeros((args.num_envs,), dtype=np.float32)
                episode_lengths = np.zeros((args.num_envs,), dtype=np.int32)

    env.close()

    obss_array = np.vstack(trajs["obss"]) 
    acts_array = np.vstack(trajs["acts"]) 
    rews_array = np.concatenate(trajs["rews"])
    terms_array = np.concatenate(trajs["terms"])
    
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
        
        obs, _ = env_render.reset(seed=args.seed)
        
        print(f"[INFO]: Recording 1 episode for {args.video_steps} steps to {log_dir}")
        
        for _ in range(args.video_steps):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device)
                normalized_obs = scaler(obs_tensor)
                action = policy(normalized_obs)
                action_np = action.cpu().numpy()
                obs, reward, terminated, _, _ = env_render.step(action_np)
                if terminated:
                    obs, _ = env_render.reset()
        env_render.close()
        print(f"[INFO]: Video saved to {log_dir}")

if __name__ == "__main__":
    main()
