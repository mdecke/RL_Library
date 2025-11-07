from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from base_classes.utils import get_activation


class Actor(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, action_limit:float, hidden_dims:List[int], lr:float, activation_fct:str, stochastic:bool, seed:int=42):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.action_limit = action_limit

        layers = []
        prev_dim = self.input_dim

        self.stochastic = stochastic

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation_fct))
            prev_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        
        if self.stochastic:
            self.mu_head = nn.Linear(prev_dim, self.output_dim)
            self.log_sigma_head = nn.Linear(prev_dim, self.output_dim)
        else:
            self.out = nn.Linear(prev_dim, self.output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input:torch.Tensor)->torch.Tensor:
        logits = self.net(input)
        if self.stochastic:
            mu = self.mu_head(logits)
            log_std = self.log_sigma_head(logits)
            if self.training:
                sigma = torch.exp(log_std) + 1e-5 #ensure not sigma not zero for numerical stibility
                actions = torch.normal(mu,sigma, generator=self.generator)
            else:
                actions =  mu
        else:
            actions = self.out(logits)
        
        bounded_actions = torch.tanh(actions) * self.action_limit
        # bounded_actions = torch.clamp(actions, -self.action_limit, self.action_limit)
        return bounded_actions

    def save(self, filepath:str)->None:
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath:str, map_location=None)->None:
        if map_location is None:
            self.load_state_dict(torch.load(filepath))
        else:
            self.load_state_dict(torch.load(filepath, map_location=map_location))
        self.eval()



class Critic(nn.Module):
    def __init__(self, input_dim:int, hidden_dims:List[int], lr:float, activation_fct:str, output_dim:int=1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr

        layers = []
        prev_dim = self.input_dim


        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation_fct))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input:torch.Tensor)->torch.Tensor:
        return self.net(input)

    def save(self, filepath:str)->None:
        torch.save(self.state_dict(), filepath)

    def load(self, filepath:str, map_location=None)->None:
        if map_location is None:
            self.load_state_dict(torch.load(filepath))
        else:
            self.load_state_dict(torch.load(filepath, map_location=map_location))
        self.eval()



class MLE(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, action_lim:float, hidden_dims:List[int], 
                 lr:float, activation_fct:str, dropout:float=0.0, weight_decay:float=0.0, tanh_flag:bool=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.action_lim = action_lim
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.use_tanh = tanh_flag

        layers = []
        prev_dim = self.input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation_fct))

            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev_dim, self.output_dim)
        self.log_sigma_head = nn.Linear(prev_dim, self.output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                           factor=0.5, patience=3)  # Reduced from 5 to 3

    def forward(self, input:torch.Tensor)->torch.Tensor:
        logits = self.net(input)
        if self.use_tanh:
            mu = torch.tanh(self.mu_head(logits)) * self.action_lim
        else:
            mu = self.mu_head(logits)
        log_sigma = torch.clamp(self.log_sigma_head(logits), min=-5.0, max=0.5)  # Clamp for numerical stability
        return mu, log_sigma
    
    def sample(self, inputs:torch.Tensor, generator:Optional[torch.Generator]=None)->torch.Tensor:
        mu, log_sigma = self.forward(inputs)
        sigma = torch.exp(log_sigma) + 1e-5  # Ensure sigma is not zero for numerical stability
        return torch.normal(mu, sigma, generator=generator)

    def save(self, filepath:str)->None:
        torch.save(self.state_dict(), filepath)

    def load(self, filepath:str)->None:
        self.load_state_dict(torch.load(filepath))
        self.eval()

    def most_likely_component(self, inputs:torch.Tensor)->torch.Tensor:
        return self.forward(inputs)[0]


class GMM(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, action_lim:float, num_components:int, hidden_dims:List[int], 
                 lr:float, activation_fct:str, dropout:float=0.0, weight_decay:float=0.0, tanh_flag:bool=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_components = num_components
        self.lr = lr

        self.action_lim = action_lim
        self.dropout = dropout
        self.weight_decay = weight_decay

        self.use_tanh = tanh_flag

        layers = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation_fct))
            prev_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.mixture_weights_head = nn.Linear(prev_dim, self.num_components)
        self.mu_head = nn.Linear(prev_dim, self.num_components * self.output_dim)
        self.log_sigma_head = nn.Linear(prev_dim, self.num_components * self.output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                           factor=0.5, patience=3)  # Reduced from 5 to 3

    def forward(self, input:torch.Tensor):
        logits = self.net(input)
        pi_logits = self.mixture_weights_head(logits)
        pi = nn.functional.softmax(pi_logits, dim=-1)

        if self.use_tanh:  
            mu = torch.tanh(self.mu_head(logits).view(-1, self.num_components, self.output_dim)) * self.action_lim
        else:
            mu = self.mu_head(logits).view(-1, self.num_components, self.output_dim)
        log_sigma = self.log_sigma_head(logits).view(-1, self.num_components, self.output_dim)

        return mu, log_sigma, pi
    
    def sample(self, inputs:torch.Tensor, generator:Optional[torch.Generator]=None)->torch.Tensor:
        mu, log_sigma, pi = self.forward(inputs)
        batch_size = inputs.size(0)
        categorical = torch.distributions.Categorical(pi)
        component_indices = categorical.sample(generator=generator)

        means = mu[torch.arange(batch_size), component_indices]
        sigma = torch.exp(log_sigma) + 1e-5  # Ensure sigma is not zero for numerical stability
        stds = sigma[torch.arange(batch_size), component_indices]
        sampled_actions = torch.normal(means, stds, generator=generator)

        return sampled_actions
    
    def most_likely_component(self, inputs:torch.Tensor)->torch.Tensor:
        mu, _, pi = self.forward(inputs)
        batch_size = inputs.size(0)
        _, component_indices = torch.max(pi, dim=-1)

        most_likely_means = mu[torch.arange(batch_size), component_indices]
        return most_likely_means
    

    def save(self, filepath:str)->None:
        torch.save(self.state_dict(), filepath)

    def load(self, filepath:str)->None:
        self.load_state_dict(torch.load(filepath))
        self.eval()


class AffineCouplingConditioner(nn.Module):
    def __init__(self, dim_split:int, output_dim:int, hidden_dims:List[int], lr:float, activation_fct:str, dropout:float=0.02):
        super().__init__()

        self.input_dim = dim_split # split_idx 0 to split_idx
        self.output_dim = output_dim  # split_idx to end: D-split_idx usually floor(D/2)
        self.lr = lr

        layers = []
        prev_dim = self.input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation_fct))
        
            if i < len(hidden_dims) - 1:
                    layers.append(nn.Dropout(dropout))
                
            prev_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.translation_head = nn.Linear(prev_dim, self.output_dim)
        self.log_scale_head = nn.Linear(prev_dim, self.output_dim)

    def forward(self, z_split_lower: torch.Tensor) -> torch.Tensor:
        logits = self.net(z_split_lower)
        t = self.translation_head(logits)
        s = self.log_scale_head(logits)
        return s, t
    
    
class AffineTrasformer(nn.Module):
    def __init__(self, dim_split:int):
        super().__init__()

        self.split_dim = dim_split
    
    def forward(self, z:torch.Tensor, s:torch.Tensor, t:torch.Tensor)->torch.Tensor:
        z_split_lower = z[:, :self.split_dim]
        z_split_upper = z[:, self.split_dim:]

        z_prime_upper = z_split_upper * torch.exp(s) + t
        z_prime = torch.cat([z_split_lower, z_prime_upper], dim=1)

        log_det_jacobian = torch.sum(s, dim=1)
        
        return z_prime, log_det_jacobian

    def inverse(self, z_prime:torch.Tensor, s:torch.Tensor, t:torch.Tensor)->torch.Tensor:
        z_prime_split_lower = z_prime[:, :self.split_dim]
        z_prime_split_upper = z_prime[:, self.split_dim:]

        z_upper = (z_prime_split_upper - t) * torch.exp(-s)
        z = torch.cat([z_prime_split_lower, z_upper], dim=1)

        log_det_jacobian = -torch.sum(s, dim=1)

        return z, log_det_jacobian
