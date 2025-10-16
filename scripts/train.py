import os

import argparse
from tqdm import tqdm as tqdm

import gymnasium as gym
import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

import agents
from base_classes.utils import load_config, get_noise_model


parser = argparse.ArgumentParser(description='Train or test TD3 agent on a given environment')
parser.add_argument('--task', type=str, default='Pendulum-v1', help='Gym environment name')
parser.add_argument('--num_envs', type=int, default=1, help='Number of parallel environments')
parser.add_argument('--max_iterations', type=int, default=1000, help='Maximum number of iterations')
parser.add_argument('--path_to_saved_policy', type=str, help='Path to the saved policy')
parser.add_argument('--algorithm', type=str, default='tdn', help='RL algorithm to use')
parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu or cuda)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

args = parser.parse_args()


def main():
    
    # Set random seeds for reproducibility (exploration noise sampling --> line 90)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    config_file = os.path.join("configs", f"{args.algorithm}Config.yaml")
    general_cfg = load_config(config_file, args)

    warm_up = general_cfg['training']['warm_up']
    random_steps = general_cfg['training']['random_steps']

    log_dir = os.path.join("logs", args.task, args.algorithm, "training_stats")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    save_dir = os.path.join(args.path_to_saved_policy, args.task, args.algorithm)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_dir = os.path.join("logs", "tensorboard", args.task, args.algorithm)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"[INFO]: TensorBoard logging to {tensorboard_dir}")
    print(f"[INFO]: Run 'tensorboard --logdir=logs/tensorboard' to view metrics")

    data_list = []

    env = gym.make_vec(args.task, num_envs=args.num_envs)

    agent = agents.create_agent(env, args.algorithm, general_cfg)

    noise = get_noise_model(general_cfg)

    cumulative_reward = torch.zeros((args.num_envs,1), dtype=torch.float32, device=agent.device)
    episode_lengths = torch.zeros((args.num_envs,), dtype=torch.int32, device=agent.device)
    avg_return = []
    progress_bar = tqdm(range(args.max_iterations), unit="step")
    BEST_SO_FAR = general_cfg.get('BEST_SO_FAR', -float('inf'))
    
    # Tracking for tensorboard
    total_episodes = 0

    if args.task == "Pendulum-v1":
        obs, _ = env.reset(seed=args.seed, options={'x_init': np.pi, 'y_init': 8.0})
    else:
        obs, _ = env.reset(seed=args.seed)  
    for t in progress_bar:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            if t < warm_up or t < random_steps:
                action = env.action_space.sample()
                clipped_action = torch.as_tensor(action, dtype=torch.float32, device=agent.device)
            else:
                normalized_obs = agent.obs_preprocessor(obs_tensor)
                action = agent.policy.forward(normalized_obs)
                expl_noise = noise.sample(action.shape).to(agent.device)
                noisy_action = action + expl_noise
                clipped_action = noisy_action.clamp(min=agent.action_low, max=agent.action_high)
        
            obs_, reward, terminated, truncated, info = env.step(clipped_action.cpu().numpy())
            
            next_obs_tensor = torch.as_tensor(obs_, dtype=torch.float32, device=agent.device).view(args.num_envs, -1)
            reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=agent.device).view(args.num_envs, -1)
            terminated_tensor = torch.as_tensor(terminated, dtype=torch.bool, device=agent.device).view(args.num_envs, -1)
            truncated_tensor = torch.as_tensor(truncated, dtype=torch.bool, device=agent.device).view(args.num_envs, -1)
            cumulative_reward += reward_tensor
            episode_lengths += 1
            
            agent.memory.add_sample(obs=obs_tensor, actions=clipped_action, next_obs=next_obs_tensor, rewards=reward_tensor, done=terminated_tensor)

        
        if (t >= warm_up) and (agent.memory.filled_lines >= general_cfg['memory']['batch_size']):
            agent.update()
            
            # Log training metrics to TensorBoard
            if len(agent.policy_loss) > 0:
                writer.add_scalar('Loss/Policy', agent.policy_loss[-1], t)
            if len(agent.critic_loss) > 0:
                writer.add_scalar('Loss/Critic', agent.critic_loss[-1], t)
            if hasattr(agent, 'mean_q_value') and len(agent.mean_q_value) > 0:
                writer.add_scalar('Training/Mean_Q_Value', agent.mean_q_value[-1], t)

        if (any(terminated) or any(truncated)):
            env_idx = np.array(np.where(terminated | truncated)).squeeze()
            if env_idx.ndim == 0:  # Handle single env case
                env_idx = np.array([env_idx])
            
            # Log episode metrics to TensorBoard
            for idx in env_idx:
                episode_reward = cumulative_reward[idx].item()
                episode_len = episode_lengths[idx].item()
                
                writer.add_scalar('Episode/Return', episode_reward, total_episodes)
                writer.add_scalar('Episode/Length', episode_len, total_episodes)
                total_episodes += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'return': f'{episode_reward:.1f}',
                    'episodes': total_episodes,
                    'best': f'{BEST_SO_FAR:.1f}'
                })
            
            avg_return.append(torch.mean(cumulative_reward[env_idx,:].cpu()))
            best_ending = torch.max(cumulative_reward[env_idx,:].cpu())
            if best_ending >= BEST_SO_FAR:
                BEST_SO_FAR = best_ending
                agent.save_checkpoint(save_dir)
                torch.save(agent.obs_preprocessor.state_dict(), os.path.join(save_dir, "obs_preprocessor.pth"))
                writer.add_scalar('Episode/Best_Return', BEST_SO_FAR, total_episodes)
                # general_cfg['BEST_SO_FAR'] = BEST_SO_FAR
                # with open(config_file, 'w') as f:
                #     yaml.dump(general_cfg, f) # Save updated best return to config file this allows to keep best return across multiple training sessions


            cumulative_reward[env_idx,:] = 0.0
            episode_lengths[env_idx] = 0
            if args.task == "Pendulum-v1":
                obs, _ = env.reset(seed=args.seed, options={'x_init': np.pi, 'y_init': 8.0})
            else:
                obs, _ = env.reset(seed=args.seed)
        else:
            obs = obs_.copy()
    
    writer.close()
    env.close()

    # Collect stats
    for i in range(len(agent.policy_loss)):
        data_list.append({
            'step': i,
            'policy_loss': agent.policy_loss[i],
            'q_loss': agent.critic_loss[i],
            'return': avg_return[i].item() if i < len(avg_return) else np.nan,
        })

    df = pd.DataFrame(data_list)
    df.to_csv(os.path.join(log_dir, f"seed_{args.seed}.csv"), index=False)

if __name__ == "__main__":
    main()