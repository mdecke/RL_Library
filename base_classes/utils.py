import os

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torchinfo import summary
import yaml

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from skrl.resources.preprocessors.torch import RunningStandardScaler

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    "softplus": nn.Softplus,
}

NOISE_MODELS = {
    "normal": dist.Normal,
    "uniform": dist.Uniform,
}

def get_activation(name: str) -> nn.Module:
    name_lower = name.strip().lower()
    if name_lower not in ACTIVATIONS:
        raise ValueError(f"{name} is not an acceptable activation function, provide a different one.")
    cls = ACTIVATIONS[name_lower]
    if cls is nn.Softmax:
        return cls(dim=-1)
    return cls()


def init_model_weights(model:nn.Module, mean:float=0.0, std:float=0.1, seed:int=42):
    if seed is not None:
        torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "weight" in name:
                nn.init.normal_(param, mean=mean, std=std)
            elif "bias" in name:
                nn.init.normal_(param, mean=mean, std=std)


def soft_update(target:nn.Module, behavior:nn.Module, tau:float):
    for target_param, source_param in zip(target.parameters(), behavior.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * source_param.data)


def load_config(config_path:str, args) -> Dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config.update(vars(args))
    config['seed'] = args.seed
    config['device'] = args.device
    return config


def get_noise_model(cfg:Dict, source:str="action") -> dist:
    noise_type = cfg['agent'][f'{source}_noise_type']
    params = cfg['agent'][f'{source}_noise_params']
    return NOISE_MODELS[noise_type](**params)


def print_model_summary(model:nn.Module, input_size:Tuple[int])->None:
    summary(model, input_size=input_size)
    


def load_scaler(size:int, scaler_filepath: str, device='cpu'):
    scaler = RunningStandardScaler(size=size, device=device)
    scaler.load_state_dict(torch.load(scaler_filepath, map_location=device))
    return scaler


def load_file(filepath: str) -> pd.DataFrame:
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        return pd.read_excel(filepath)
    elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
        return pd.read_pickle(filepath)
    elif filepath.endswith('.json'):
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format for {filepath}. Supported formats are: .csv, .xlsx, .xls, .pkl, .pickle, .json")


def gaussian_nll_loss(mu:torch.Tensor, log_sigma:torch.Tensor, target:torch.Tensor)->torch.Tensor:
    sigma = torch.exp(log_sigma) + 1e-5  # Ensure sigma is not zero for numerical stability
    nll = 0.5 * torch.log(2 * torch.pi * (sigma ** 2)) + 0.5 * ((target - mu) ** 2) / (sigma ** 2) # also 0.5 * torch.log(2 * torch.pi) + 0.5 * log_sigma + 0.5 * ((target - mu) ** 2) / (sigma ** 2)  -- valid formulation
    # loss = nn.GaussianNLLLoss() might be pb mean over batch dimension
    mse = F.mse_loss(mu, target) * 0.001
    sigma_penalty = torch.relu(sigma - 1.0).mean() * 0.1
    return nll.mean() + sigma_penalty + mse


def gmm_nll_loss(mus:torch.Tensor, log_sigmas:torch.Tensor, pis:torch.Tensor, target:torch.Tensor)->torch.Tensor:
    """Compute the negative log-likelihood loss for a Gaussian Mixture Model."""
    batch_size, n_components, action_dim = mus.size()
    target_expanded = target.unsqueeze(1).expand(-1, n_components, -1)  

    sigmas = torch.exp(log_sigmas) + 1e-5  
    normal_dist = dist.Normal(mus, sigmas)
    log_probs = normal_dist.log_prob(target_expanded)  
    log_probs_sum = log_probs.sum(dim=2)  

    weighted_log_probs = log_probs_sum + torch.log(pis + 1e-8)  
    log_sum_exp = torch.logsumexp(weighted_log_probs, dim=1)  

    nll = -log_sum_exp.mean()  

    mse = F.mse_loss((pis.unsqueeze(2) * mus).sum(dim=1), target) * 0.001  
    sigma_penalty = torch.relu(sigmas - 1.0).mean() * 0.1  

    return nll + sigma_penalty + mse


def make_data_frame(data_dir:str) -> pd.DataFrame:
    csv_paths = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".csv"):
            file = os.path.join(data_dir, file_name)
            csv_paths.append(file)
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    training_data = pd.DataFrame()
    for csv_path in csv_paths:
        df = load_file(csv_path)
        df["seed"] = int(os.path.basename(csv_path).split("_")[-1].split(".")[0])
        training_data = pd.concat([training_data, df], ignore_index=True)
    return training_data

class EarlyStopping:
    def __init__(self, patience:int=20, min_delta:float=1e-4, verbose:bool=True):

        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss:float, model:nn.Module)->bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f"Validation loss improved: {self.best_loss:.6f} → {val_loss:.6f}")
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        
        return self.early_stop
    
    def load_best_model(self, model:nn.Module)->None:
        """Load the best model state"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def evaluate_expert_rollout(expert, env_name, n_episodes=5, seed=42, video=False):
    """Evaluate expert policy in the environment with rollouts."""
    env = gym.make(env_name)
    
    episode_rewards = []
    all_instantaneous_rewards = []
    all_cumulative_rewards = []
    
    env_render = None  # Track video recording environment
    
    for episode in range(n_episodes):
        if video and episode == 0:
            print("\n[INFO]: Recording validation video...")
            video_folder = os.path.join(f"logs/{env_name}/tdn", "expert_rollout_rendering")
            if not os.path.exists(video_folder):
                os.makedirs(video_folder, exist_ok=True)
        
            env_render = gym.make(env_name, render_mode="rgb_array")
            env_render = RecordVideo(
                env_render, 
                video_folder=video_folder,
                episode_trigger=lambda ep: ep == 0,
                name_prefix=f"seed_{seed}"
            )
            env_to_use = env_render
        else:
            env_to_use = env
        
        obs, info = env_to_use.reset(seed=seed + episode)
        done = False
        truncated = False
        episode_reward = 0
        cumulative_reward = 0
        instantaneous_rewards = []
        cumulative_rewards = []
        
        while not (done or truncated):
            # Get action from expert
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = expert.most_likely_component(obs_tensor).squeeze(0).cpu().numpy()
            
            # Step environment
            obs, reward, done, truncated, info = env_to_use.step(action)
            
            episode_reward += reward
            cumulative_reward += reward
            instantaneous_rewards.append(reward)
            cumulative_rewards.append(cumulative_reward)
        
        episode_rewards.append(episode_reward)
        all_instantaneous_rewards.append(instantaneous_rewards)
        all_cumulative_rewards.append(cumulative_rewards)
        
        print(f"Episode {episode + 1}/{n_episodes}: Reward = {episode_reward:.2f}")
        
        # Close video environment after first episode
        if video and episode == 0 and env_render is not None:
            env_render.close()
            print(f"[INFO]: Video saved to {video_folder}")
    
    # Close regular environment
    env.close()
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\nRollout Evaluation:")
    print(f"Average Episode Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    
    return episode_rewards, all_instantaneous_rewards, all_cumulative_rewards

class ReturnNormalizer:
    """Normalize episodic returns to [-1, 1] range for exploration control"""
    def __init__(self, clip=3.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.clip = clip  # clip to ±clip std devs
    
    def update(self, returns):
        """Update with batch of returns (can be single value or list)"""
        if isinstance(returns, (int, float)):
            returns = [returns]
        returns = np.array(returns)
        
        batch_mean = np.mean(returns)
        batch_var = np.var(returns)
        batch_count = len(returns)
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean += delta * batch_count / total_count
        self.var = (self.count * self.var + batch_count * batch_var + 
                    delta**2 * self.count * batch_count / total_count) / total_count
        self.count = total_count
    
    def normalize(self, episodic_return):
        """Normalize to [-1, 1] range"""
        if self.count < 5:  # Not enough data yet
            return 0.0
        
        std = np.sqrt(self.var) + 1e-8
        normalized = (episodic_return - self.mean) / std
        clipped = np.clip(normalized, -self.clip, self.clip)
        return clipped / self.clip  # scale to [-1, 1]
