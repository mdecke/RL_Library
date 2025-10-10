from typing import Tuple, Optional, Dict
import torch


class OffPolicyMemory:
    def __init__(self,
                 episode_horizon:int,
                 n_envs:int,
                 observation_dim:Tuple[int,...] | int,
                 action_dim:Tuple[int,...] | int,
                 cfg:Optional[Dict],
                 asymmetric_flag:bool=False,
                 device:str="cpu",
                 action_dtype:torch.dtype=torch.float32):
        
        self.N = episode_horizon
        self.envs = n_envs
        self.obs_dim = observation_dim
        self.act_dim = action_dim
        if isinstance(self.obs_dim, int):
            obs_shape = (self.obs_dim,)
        else:
            obs_shape = tuple(self.obs_dim)
        if isinstance(self.act_dim, int):
            act_shape = (self.act_dim,)
        else:
            act_shape = tuple(self.act_dim)
        self.device = device
        self.asym_flag = asymmetric_flag


        # Sequence of Arrays (SoA) --> each variable is stored in a (N by #_envs) tensor
        self.obs = torch.empty((self.N, self.envs, *obs_shape), dtype=torch.float32, device=device)
        self.next_obs = torch.empty((self.N, self.envs, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.empty((self.N, self.envs, *act_shape), dtype=action_dtype, device=device)
        self.rewards = torch.empty((self.N, self.envs, 1), dtype=torch.float32, device=device)
        self.dones = torch.empty((self.N, self.envs, 1), dtype=torch.bool, device=device)
        self.infos = torch.empty((self.N, self.envs), dtype=torch.bool, device=device)

        if self.asym_flag:
            self.priv_obs_dim = cfg["privileged_observation_dim"]
            if isinstance(self.priv_obs_dim, int):
                priv_shape = (self.priv_obs_dim,)
            else:
                priv_shape = tuple(self.priv_obs_dim)
            self.privileged_obs = torch.empty((self.N, self.envs, *priv_shape), dtype=torch.float32, device=device)
            self.next_privileged_obs = torch.empty((self.N, self.envs, *priv_shape), dtype=torch.float32, device=device)
        
        
        self.env_steps = 0
        self.filled_lines = 0

    def add_sample(self,
                   obs:torch.Tensor,
                   actions:torch.Tensor,
                   next_obs:torch.Tensor,
                   rewards:torch.Tensor,
                   done:torch.Tensor,
                   priv_obs:Optional[torch.Tensor] = None,
                   next_priv_obs:Optional[torch.Tensor] = None)->None:

        index = self.env_steps % self.N 

        self.obs[index].copy_(obs)
        self.actions[index].copy_(actions)
        self.rewards[index].copy_(rewards.view(self.envs, 1))
        self.next_obs[index].copy_(next_obs)
        self.dones[index].copy_(done.view(self.envs, 1))

        if self.asym_flag:
            self.privileged_obs[index].copy_(priv_obs)
            self.next_privileged_obs[index].copy_(next_priv_obs)
        
        self.env_steps+=1
        self.filled_lines = min(self.env_steps,self.N)

    def sample(self, batch_size:int, n_step_horizon:int=3):
        # Safety check: ensure we have enough samples for n-step returns
        max_start = max(1, self.filled_lines - n_step_horizon)
        start_time = torch.randint(0, max_start, (batch_size,), device=self.device)
        sampled_envs  = torch.randint(0, self.envs,  (batch_size,), device=self.device)

        offset= torch.arange(n_step_horizon, device=self.device) #-> [0,1,2,...,n_step_horizon]       
        time_window = (start_time[:, None] + offset[None, :]) % self.N  #-> tensor[[start_time[0], start_time[0]+1, start_time[0]+2,...],start_time[1]] wrapped around N
        env_window = sampled_envs[:, None].expand(batch_size, n_step_horizon) #-> convert sampled envs array to 2D array with lines same env id

        obs_hist = self.obs[time_window,env_window,:]
        act_hist = self.actions[time_window,env_window,:]
        next_obs_hist = self.next_obs[time_window,env_window,:]
        r_hist = self.rewards[time_window,env_window,:]
        d_hist = self.dones[time_window,env_window,:]

        batch = {"obs": obs_hist,
                 "acts": act_hist,
                 "next_obs": next_obs_hist,
                 "r": r_hist,
                 "d": d_hist}

        if self.asym_flag:
            priv_obs_hist = self.privileged_obs[time_window,env_window,:]
            next_priv_obs_hist = self.next_privileged_obs[time_window,env_window,:]
            batch["priv_obs"] = priv_obs_hist
            batch["next_priv_obs"] = next_priv_obs_hist

        if self.asym_flag:
            print("batch priv_obs: ", batch["priv_obs"])
            print("batch next_priv_obs: ", batch["next_priv_obs"])

        return batch
    