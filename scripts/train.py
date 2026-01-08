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
from base_classes.utils import load_config, get_noise_model, ReturnNormalizer, save_model


parser = argparse.ArgumentParser(description='Train or test TD3 agent on a given environment')
parser.add_argument('--task', type=str, default='Pendulum-v1', help='Gym environment name')
parser.add_argument('--num_envs', type=int, default=1, help='Number of parallel environments')
parser.add_argument('--max_iterations', type=int, default=1000, help='Maximum number of iterations')
parser.add_argument('--path_to_saved_policy', type=str, help='Path to the saved policy')
parser.add_argument('--algorithm', type=str, default='tdn', help='RL algorithm to use')
parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu or cuda)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--save_method', type=str, choices=['best', 'last'], default='last', help='Method to save the model')
parser.add_argument('--expert_guidance', type=str, choices=["mle","gmm","cnf"], default=None, help='Type of expert to use for guidance (if any)')
args = parser.parse_args()


def main():
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    config_file = os.path.join("configs", f"{args.algorithm}Config.yaml")
    general_cfg = load_config(config_file, args)

    warm_up = general_cfg['training']['warm_up']
    random_steps = general_cfg['training']['random_steps']

    if args.expert_guidance is not None:
        log_dir = os.path.join("logs", args.task, args.algorithm, "guided_training_stats")
        save_dir = os.path.join("saved", args.task, args.algorithm, f"EGOP_models")
    else:
        log_dir = os.path.join("logs", args.task, args.algorithm, "training_stats")
        save_dir = os.path.join("saved", args.task, args.algorithm, "RL_models")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
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

    if args.expert_guidance is not None:
        config_file = os.path.join("configs", "ExpertConfig.yaml")
        full_cfg = load_config(config_file, args)
        expert_cfg = {k: v for k, v in full_cfg.items() if k not in ['mle', 'gmm', 'cnf']}
        expert_cfg[args.expert_guidance] = full_cfg[args.expert_guidance]
        expert_cfg['device'] = args.device
        expert_cfg['expert_domain'] = 'time'
        expert_cfg["obs_dim"] = env.single_observation_space.shape[0]
        expert_cfg["action_dim"] = env.single_action_space.shape[0]
        expert_model_path = os.path.join('saved', args.task,'expert', 'scripted_expert.pt')
        expert = torch.jit.load(expert_model_path, map_location=args.device)
        expert.eval()
        print(f"[INFO]: Using {args.expert_guidance} expert for guidance during training.")
    else:
        expert = None
        print(f"[INFO]: No expert guidance used during training.")

    cumulative_reward = torch.zeros((args.num_envs,1), dtype=torch.float32, device=agent.device)
    episode_lengths = torch.zeros((args.num_envs,), dtype=torch.int32, device=agent.device)
    avg_return = []
    progress_bar = tqdm(range(args.max_iterations), unit="step")
    BEST_SO_FAR = general_cfg.get('BEST_SO_FAR', -float('inf'))
    
    # Tracking for tensorboard
    total_episodes = 0
    update_starts = warm_up // args.num_envs

    return_normalizer = ReturnNormalizer(clip=3.0)
    eta = 0.0
    eta_smoothing = 0.95  # EMA smoothing factor for eta (higher = more smoothing)
    max_eta_change = 0.05  # Maximum change in eta per episode update

    warm_up_done = 0
    learn_has_started = 0
    
    if args.task == "Pendulum-v1":
        obs, _ = env.reset(seed=args.seed, options={'x_init': np.pi, 'y_init': 8.0})
    else:
        obs, _ = env.reset(seed=args.seed)  
    
    for t in progress_bar:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if agent.preprocess_inputs:
            normalized_obs = agent.obs_preprocessor(obs_tensor, train=True)
        else:
            normalized_obs = obs_tensor
        
        with torch.no_grad():
            if t < warm_up or t < random_steps:
                if args.expert_guidance is not None:
                    # Use expert action during warm-up if expert guidance is enabled
                    action = expert.most_likely_component(obs_tensor)  # normalization handled by expert
                    clipped_action = action.clamp(min=agent.action_low, max=agent.action_high)
                else:
                    action = env.action_space.sample()
                    clipped_action = torch.as_tensor(action, dtype=torch.float32, device=agent.device)
            else:
                if warm_up_done == 0:
                    print("[INFO]: Warm-up phase completed. Starting policy-based actions.")
                    warm_up_done = 1
                action = agent.policy.forward(normalized_obs)
                if expert is not None:
                    # expl_noise = expert.sample(obs_tensor)
                    expert_sample = expert.most_likely_component(obs_tensor) # normalization handled by expert
                    noisy_action = eta*action + (1-eta)*expert_sample
                else:
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

        
        if (t >= update_starts) and (agent.memory.filled_lines >= general_cfg['memory']['batch_size']):
            agent.update()
            if learn_has_started == 0:
                print("[INFO]: Learning updates have started.")
                learn_has_started = 1
            
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

            # Update progress bar with aggregated stats (outside the loop)
            mean_return = cumulative_reward[env_idx].mean().item()
            progress_bar.set_postfix({
                'return': f'{mean_return:.1f}',
                'episodes': total_episodes,
                'best': f'{BEST_SO_FAR:.1f}'
            })
            
            avg_return.append(torch.mean(cumulative_reward[env_idx,:].cpu()))
            best_ending = avg_return[-1] 
            
            # Save best model if using 'best' save method
            if best_ending >= BEST_SO_FAR:
                BEST_SO_FAR = best_ending
                writer.add_scalar('Episode/Best_Avg_Return', BEST_SO_FAR, total_episodes)
                if args.save_method == 'best':
                    save_model(agent.policy, "policy", save_dir, obs_dim=env.single_observation_space.shape[0])
                    if agent.preprocess_inputs:
                        save_model(agent.obs_preprocessor, "obs_preprocessor", save_dir, obs_dim=env.single_observation_space.shape[0])
                    

            current_returns = cumulative_reward[env_idx].cpu().numpy().flatten()
            return_normalizer.update(current_returns)
            normalized_return = return_normalizer.normalize(current_returns.mean())
            
            new_eta = max(0.0, 1.0 - normalized_return)
            new_eta = min(1.0, new_eta)
        
            new_eta = np.clip(new_eta, eta - max_eta_change, eta + max_eta_change)
            eta = eta_smoothing * eta + (1 - eta_smoothing) * new_eta
            
            # Log eta to TensorBoard
            if args.expert_guidance is not None:
                writer.add_scalar('Guidance/Eta', eta, total_episodes)
            
            cumulative_reward[env_idx,:] = 0.0
            episode_lengths[env_idx] = 0
        
        obs = obs_.copy()

    # Save final model if using 'last' save method
    if args.save_method == 'last':
        save_model(agent.policy, "policy", save_dir, obs_dim=env.single_observation_space.shape[0])
        if agent.preprocess_inputs:
            save_model(agent.obs_preprocessor, "obs_preprocessor", save_dir, obs_dim=env.single_observation_space.shape[0])
        print(f"[INFO]: Saved final model to {save_dir}")
    
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