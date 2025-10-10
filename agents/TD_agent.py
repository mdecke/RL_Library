from typing import Dict
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.resources.preprocessors.torch import RunningStandardScaler

from base_classes.models import Actor, Critic
from base_classes.memory_buffers import OffPolicyMemory
from base_classes.utils import init_model_weights, soft_update, get_noise_model, print_model_summary


class TdAgent:
    def __init__(self,env:gym.Env, cfg:Dict):
        self.cfg = cfg
        
        self.seed = cfg["seed"]
        self.device = cfg["device"]
        
        self.env = env
        single_obs_space = getattr(self.env, 'single_observation_space', self.env.observation_space)
        single_act_space = getattr(self.env, 'single_action_space', self.env.action_space)
        self.obs_dim = single_obs_space.shape[0] * cfg["agent"]["state_history"]
        self.act_dim = single_act_space.shape[0] * cfg["agent"]["action_history"]

        self.action_low = torch.as_tensor(single_act_space.low, dtype=torch.float32, device=self.device).view(-1)
        self.action_high = torch.as_tensor(single_act_space.high, dtype=torch.float32, device=self.device).view(-1)

        print(f"[INFO]: Observation dimension: {self.obs_dim}, Action dimension: {self.act_dim}")
        print(f"[INFO]: Action bounds (per action dim): low {self.action_low.tolist()}, high {self.action_high.tolist()}")
    
        self.asymmetry = cfg["agent"]["asymmetricAC"]
        self.gradient_steps = cfg["agent"]["gradient_steps"]
        self.preprocess_inputs = self.cfg["agent"]["preprocess_inputs"]
        self.gamma = cfg["agent"]["discount_factor"]
        if self.preprocess_inputs:
            self.obs_preprocessor = RunningStandardScaler(size=self.obs_dim, device=self.device)
        self.policy_delay = self.cfg["agent"]["policy_delay"]
        self.gradient_clip = self.cfg["agent"]["gradient_clip"]
        self.polyak = self.cfg["agent"]["polyak"]
        self.target_noise_range = self.cfg["agent"]["noise_range"]
        self.target_policy_noise = get_noise_model(self.cfg,source="target")

        self.temporal_diff_horizon = self.cfg["memory"]["td_horizon"]

        self._init_memory()
        self._init_models()

        self.critic_loss = []
        self.policy_loss = []
        self.mean_q_value = []  # Track mean Q-values for TensorBoard


    def _init_memory(self):
        num_envs = self.env.num_envs
        N = self.cfg["memory"]["buffer_size"]

        self.memory = OffPolicyMemory(episode_horizon=N,
                                      n_envs=num_envs,
                                      observation_dim=self.obs_dim,
                                      action_dim=self.act_dim,
                                      cfg=self.cfg,
                                      asymmetric_flag=self.asymmetry,
                                      device=self.device)
        print("[INFO]: Memory class initialized")

    
    def _init_models(self) -> None:
        policy_lr = self.cfg["models"]["policy"]["lr"]
        policy_activation_fct = self.cfg["models"]["policy"]["activation_fct"]
        stochastic = self.cfg["models"]["policy"]["stochastic"]
        policy_hidden_layers = self.cfg["models"]["policy"]["hidden_layers"]
        self.policy = Actor(input_dim=self.obs_dim,
                            output_dim=self.act_dim,
                            action_limit=self.action_high,  # Assuming symmetric action bounds
                            hidden_dims=policy_hidden_layers,
                            lr=policy_lr,
                            activation_fct=policy_activation_fct,
                            stochastic=stochastic)

        critic1_lr = self.cfg["models"]["critic1"]["lr"]
        critic1_activation_fct = self.cfg["models"]["critic1"]["activation_fct"]
        critic1_hidden_layers = self.cfg["models"]["critic1"]["hidden_layers"]
        self.critic_1 = Critic(input_dim=self.obs_dim+self.act_dim,
                              hidden_dims=critic1_hidden_layers,
                              lr=critic1_lr,
                              activation_fct=critic1_activation_fct)
        critic2_lr = self.cfg["models"]["critic2"]["lr"]
        critic2_activation_fct = self.cfg["models"]["critic2"]["activation_fct"]
        critic2_hidden_layers = self.cfg["models"]["critic2"]["hidden_layers"]
        self.critic_2 = Critic(input_dim=self.obs_dim+self.act_dim,
                              hidden_dims=critic2_hidden_layers,
                              lr=critic2_lr,
                              activation_fct=critic2_activation_fct)

        init_model_weights(self.policy, seed=self.seed)
        init_model_weights(self.critic_1, seed=self.seed)
        init_model_weights(self.critic_2, seed=self.seed)

        self.target_policy = Actor(input_dim=self.obs_dim,
                                   output_dim=self.act_dim,
                                   action_limit=self.action_high,  # Assuming symmetric action bounds
                                   hidden_dims=policy_hidden_layers,
                                   lr=policy_lr,
                                   activation_fct=policy_activation_fct,
                                   stochastic=stochastic)

        self.target_critic_1 = Critic(input_dim=self.obs_dim+self.act_dim,
                                      hidden_dims=critic1_hidden_layers,
                                      lr=critic1_lr,
                                      activation_fct=critic1_activation_fct)

        self.target_critic_2 = Critic(input_dim=self.obs_dim+self.act_dim,
                                      hidden_dims=critic2_hidden_layers,
                                      lr=critic2_lr,
                                      activation_fct=critic2_activation_fct)

        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        print("[INFO]: Models initialized")
        print("[INFO]: Policy summary:")
        print_model_summary(self.policy, input_size=(self.obs_dim,))
        print("\n[INFO]: Critic 1 summary:")
        print_model_summary(self.critic_1, input_size=(self.obs_dim + self.act_dim,))
        print("\n[INFO]: Critic 2 summary:")
        print_model_summary(self.critic_2, input_size=(self.obs_dim + self.act_dim,))


    def update(self):
        for step in range(self.gradient_steps):
            # Sample a batch from memory
            batch = self.memory.sample(batch_size=self.cfg["memory"]["batch_size"], n_step_horizon=self.temporal_diff_horizon)
            
            if self.preprocess_inputs:
                batch["obs"] = self.obs_preprocessor(batch["obs"])
                batch["next_obs"] = self.obs_preprocessor(batch["next_obs"])
            
            with torch.no_grad():
                epsilon_2 = torch.clamp(self.target_policy_noise.sample(batch["acts"][:,0,:].shape).to(self.device), -self.target_noise_range, self.target_noise_range)
                last_action = self.target_policy.forward(input=batch["next_obs"][:,-1,:])
                last_action_clipped = (last_action + epsilon_2).clamp(self.action_low, self.action_high)
                last_q_input = torch.cat((batch["next_obs"][:,-1,:], last_action_clipped), dim=-1)
                q1_target = self.target_critic_1.forward(last_q_input)
                q2_target = self.target_critic_2.forward(last_q_input)
                q_target = torch.minimum(q1_target, q2_target)

                #n-step temporal difference target:
                y = q_target
                for k in range(self.temporal_diff_horizon-1, -1, -1):
                    d = batch["d"][:,k,:]
                    r = batch["r"][:,k,:]
                    y = r + (~d)*self.gamma*y

            first_q_input = torch.cat((batch["obs"][:,0,:], batch["acts"][:,0,:]), dim=-1)
            q1_behavior = self.critic_1.forward(first_q_input)
            q2_behavior = self.critic_2.forward(first_q_input)

            # Track mean Q-value for monitoring
            self.mean_q_value.append(q1_behavior.mean().item())

            critic_loss = (F.mse_loss(q1_behavior, y) + F.mse_loss(q2_behavior, y)) / 2
            self.critic_loss.append(critic_loss.item())
            
            #critic optimization step
            self.critic_1.optimizer.zero_grad()
            self.critic_2.optimizer.zero_grad()
            
            critic_loss.backward()
            
            if self.gradient_clip is not None:
                nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.gradient_clip)
                nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.gradient_clip)
            
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()

            if step % self.policy_delay == 0:
                action = self.policy.forward(input=batch["obs"][:,0,:])
                critic_input = torch.cat((batch["obs"][:,0,:], action), dim=-1)
                policy_loss = -self.critic_1.forward(critic_input).mean()
                self.policy_loss.append(policy_loss.item())
                
                self.policy.optimizer.zero_grad()
                policy_loss.backward()
                if self.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
                self.policy.optimizer.step()

            soft_update(self.target_policy, self.policy, tau=self.polyak)
            soft_update(self.target_critic_1, self.critic_1, tau=self.polyak)
            soft_update(self.target_critic_2, self.critic_2, tau=self.polyak)

    